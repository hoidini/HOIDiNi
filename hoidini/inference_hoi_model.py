import json
import torch
from functools import partial
import os
import numpy as np
from dataclasses import dataclass, fields
from typing import Dict, Generic, List, Optional, TypeVar, Union, Any
from omegaconf import DictConfig, OmegaConf, open_dict
import datetime
import pickle
import hydra
from hydra.utils import instantiate
from tqdm import tqdm


from hoidini.datasets.grab.grab_utils import (
    get_df_prompts,
    get_table_params,
    grab_seq_data_to_object_name,
    grab_seq_path_to_seq_id,
    grab_seq_path_to_unique_name,
    load_mesh,
    parse_npz,
    reduce_seq_data,
)
from hoidini.general_utils import get_least_busy_device, get_model_identifier
from hoidini.closd.diffusion_planner.utils import dist_util
from hoidini.inference_utils import auto_regressive_dno, load_model
from hoidini.model_utils import model_kwargs_to_device
from hoidini.normalizer import Normalizer
from hoidini.closd.diffusion_planner.data_loaders.get_data import get_dataset_loader
from hoidini.object_contact_prediction.cpdm_dataset import ContactPairsSequence
from hoidini.objects_fk import ObjectModel
from hoidini.optimize_latent.dno import DNOOptions
from hoidini.datasets.smpldata import SmplData, SmplModelsFK
from hoidini.datasets.grab.grab_object_records import ContactRecord
from hoidini.optimize_latent.dno_loss_functions import (
    DnoConditionOld,
    Scheduler,
    DnoLossSource,
)
from hoidini.general_utils import TMP_DIR
from hoidini.blender_utils.visualize_hoi_animation import (
    AnimationSetup,
    visualize_hoi_animation,
)
from hoidini.datasets.dataset_smplrifke import (
    Hml3DGrabDataset,
    HoiGrabDataset,
    collate_smplrifke_mdm,
    get_grab_amass_datapoint_name,
    samples_to_smpldata_lst,
)
from hoidini import plot_dno_loss


T = TypeVar("T")


@dataclass
class InitConfig(Generic[T]):
    _target_: str
    _args_: Dict[str, Any] = None


@dataclass
class LoadContactRecordsCfg:
    grab_seq_paths: Optional[List[str]] = None
    reduce_contacts_to_anchors: bool = False
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None
    reduce_contact_to_anchors: bool = True


@dataclass
class LoadContactPairsCfg:
    load_from: str
    dir: Optional[str] = None
    files: Optional[List[str]] = None
    texts: Optional[List[str]] = None
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None
    added_6dof_velocity: Optional[List[float]] = None

    n_chunks: Optional[int] = None
    chunk_ind: Optional[int] = None


@dataclass
class PredictContactPairsCfg:
    nearest_neighbor_per_frame: bool
    n_samples: int
    n_frames: int


@dataclass
class SamplingConfig:
    out_dir: str | None
    model_path: str
    dno_options: DNOOptions
    dno_loss_coefficients: Dict[str, Union[float, InitConfig[Scheduler]]]
    batch_size_gen: int
    batch_size_dataset: int
    use_obj_encoding_cache: bool
    use_real_data_as_prefix: bool
    use_dno: bool
    save_result_pickle: bool
    save_result_only_first_in_batch: bool  # We generate xB samples per contact pair
    save_result_only_first_in_batch: bool  # We generate xB samples per contact pair
    dno_loss_source: InitConfig[DnoLossSource]
    load_contact_pairs_cfg: InitConfig[LoadContactPairsCfg]
    predict_contact_pairs_cfg: InitConfig[PredictContactPairsCfg]
    load_contact_records_cfg: InitConfig[LoadContactRecordsCfg]

    n_simplify_hands: int
    n_simplify_object: int
    guidance_param: float
    autoregressive_include_prefix: bool
    th_contact_force: float
    anim_save: bool
    anim_setup: InitConfig[AnimationSetup]
    anim_save_every_n_gens: Optional[int]
    use_dataset_cache: bool
    data_load_lim: Optional[float]
    translation_obj: Optional[List[float]]
    train: DictConfig = None

    def __post_init__(self):
        if self.anim_save_every_n_gens is not None:
            assert self.anim_save

    def __post_init__(self):
        if self.anim_save_every_n_gens is not None:
            assert self.anim_save


@dataclass
class HoiResult:
    raw_sample: torch.Tensor  # (batch_size, n_feats, 1, n_frames) for debug
    text: str
    object_name: str
    grab_seq_path: str
    start_frame: int
    translation_obj: torch.Tensor
    contact_record: ContactRecord
    optim_info_lst: Optional[list[list[dict[str, list[float]]]]] = (
        None  # [ar_step][batch][loss_name][step]
    )
    contact_pairs_seq: Optional[ContactPairsSequence] = None
    smpldata: Optional[SmplData] = None
    mat_lst: Optional[list[torch.Tensor]] = None  # (batch_size, n_frames, 4, 4)
    seed: Optional[int] = None

    def save(self, path: str):
        d = {field.name: getattr(self, field.name) for field in fields(self)}
        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu()
            elif isinstance(v, (ContactRecord, ContactPairsSequence, SmplData)):
                v = v.detach().to("cpu").to_dict()
            d[k] = v
        with open(path, "wb") as f:
            pickle.dump(d, f)

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            d = pickle.load(f)
        for k, v in d.items():
            if k == "contact_pairs_seq" and v is not None:
                v = ContactPairsSequence(**v)
            elif k == "smpldata" and v is not None:
                v = SmplData(**v)
            elif k == "contact_record" and v is not None:
                v = ContactRecord(**v)
            d[k] = v
        return cls(**d)


def add_training_args(cfg: SamplingConfig):
    experiment_dir = os.path.dirname(cfg.model_path)
    cfg_train = OmegaConf.load(os.path.join(experiment_dir, "args.json"))
    cfg_train.experiment_dir = experiment_dir
    cfg_train.cfg_type = "text"
    cfg_train.dataset_data = cfg_train.get("dataset_data", "hml3d")
    cfg_train.model_path = cfg.model_path
    with open_dict(cfg):
        cfg.experiment_dir = experiment_dir
        cfg.train = cfg_train


# def get_sampler_inputs_for_contact_pairs(cfg: SamplingConfig) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
#     cfg_load_cp: LoadContactPairsCfg = cfg.load_contact_pairs_cfg
#     if cfg_load_cp.load_from == 'files':
#         cps_paths = cfg_load_cp.files
#     elif cfg_load_cp.load_from == 'dir':
#         cps_paths = [os.path.join(cfg_load_cp.dir, f) for f in os.listdir(cfg_load_cp.dir) if f.endswith('.npz')]
#     else:
#         cps_paths = None
#     if cps_paths is not None:
#         return None
#     grab_seq_paths = []
#     for cps_path in cps_paths:
#         cps = ContactPairsSequence.load(cps_path)
#         grab_seq_paths.append(cps.metadata.grab_seq_path)
#     return grab_seq_paths


def get_dataloader(cfg: SamplingConfig, grab_seq_paths: Optional[List[str]] = None):
    data = get_dataset_loader(
        name=cfg.train.dataset,
        batch_size=cfg.batch_size_dataset,
        fixed_len=cfg.train.pred_len + cfg.train.context_len,
        pred_len=cfg.train.pred_len,
        hml_type=cfg.train.hml_type,
        split=None,
        device=dist_util.dev(),
        data_load_lim=cfg.data_load_lim,
        experiment_dir=cfg.experiment_dir,
        dataset_data=cfg.train.dataset_data,
        use_cache=cfg.use_dataset_cache,
        features_string=cfg.train.get("features_string"),
        grab_seq_paths=grab_seq_paths,
    )
    return data


def get_input_params(
    cfg: SamplingConfig, dataset, model, n_frames, grab_seq_path, text, object_name
):
    batch_size = cfg.batch_size_gen
    if isinstance(dataset, HoiGrabDataset):
        dp_ind = [data["grab_seq_path"] for data in dataset.data].index(grab_seq_path)
        dp = dataset.__getitem__(dp_ind, 0)
        dp["obj_points"] = dataset.get_object_pcd(object_name)
    elif isinstance(dataset, Hml3DGrabDataset):
        grab_amass_dp_name = get_grab_amass_datapoint_name(
            grab_seq_path_to_seq_id(grab_seq_path), dataset.df_grab_prompts
        )
        dp_ind = dataset.dp_names.index(grab_amass_dp_name)
        dp = dataset.__getitem__(dp_ind, start_frame=0)
    else:
        dp = dataset[np.random.randint(len(dataset))]  # some random data point

    collate = partial(
        collate_smplrifke_mdm,
        pred_len=cfg.train.pred_len,
        context_len=cfg.train.context_len,
    )
    _, model_kwargs = collate([dp] * batch_size)
    model_kwargs["y"]["text"] = [text for _ in range(batch_size)]
    if isinstance(dataset, Hml3DGrabDataset):
        model_kwargs["y"]["text"] = [
            f"{text} [SEP] {text}" if "[SEP]" not in text else text
            for _ in range(batch_size)
        ]
    if not cfg.use_real_data_as_prefix:
        model_kwargs["y"]["prefix"] = torch.zeros_like(model_kwargs["y"]["prefix"])

    model_kwargs_to_device(model_kwargs, dist_util.dev())
    if cfg.use_obj_encoding_cache and isinstance(dataset, HoiGrabDataset):
        with torch.no_grad():
            model_kwargs["y"]["obj_emb"] = model.encode_obj(
                model_kwargs["y"]["obj_points"]
            )

    if cfg.guidance_param != 1:
        model_kwargs["y"]["scale"] = (
            torch.ones(cfg.batch_size_gen, device=dist_util.dev()) * cfg.guidance_param
        )
    motion_shape = (cfg.batch_size_gen, model.njoints, model.nfeats, n_frames)
    return model_kwargs, motion_shape


def get_loaded_contact_pairs_grab_seq_paths(cfg: SamplingConfig):
    cfg_cps: LoadContactPairsCfg = instantiate(cfg.load_contact_pairs_cfg)
    if cfg_cps.load_from == "dir":
        cps_paths = [
            os.path.join(cfg_cps.dir, f)
            for f in os.listdir(cfg_cps.dir)
            if f.endswith(".npz")
        ]
    else:
        cps_paths = cfg_cps.files
    grab_seq_paths = []
    for cps_path in cps_paths:
        contact_pairs = ContactPairsSequence.load(cps_path)
        grab_seq_paths.append(contact_pairs.metadata.grab_seq_path)
    return grab_seq_paths


def iter_contact_record_kwargs_lst(cfg: SamplingConfig):
    df_prompts = get_df_prompts()
    cfg_cpr: LoadContactRecordsCfg = instantiate(cfg.load_contact_records_cfg)
    grab_seq_paths = cfg_cpr.grab_seq_paths

    for grab_seq_path in grab_seq_paths:
        contact_record = ContactRecord.from_grab_data(
            grab_seq_path, n_simplify_object=cfg.n_simplify_object
        )
        if cfg_cpr.reduce_contacts_to_anchors:
            contact_record.reduce_contact_to_anchors()
        if cfg_cpr.start_frame is not None or cfg_cpr.end_frame is not None:
            contact_record = contact_record.cut(cfg_cpr.start_frame, cfg_cpr.end_frame)
        if cfg.translation_obj is not None:
            translation_obj = torch.tensor(cfg.translation_obj)
            contact_record = contact_record.translate(translation_obj)
        else:
            translation_obj = None

        object_name = grab_seq_path_to_object_name(grab_seq_path)
        obj_verts, obj_faces = load_mesh(object_name, cfg.n_simplify_object)
        obj_fk = ObjectModel(v_template=obj_verts, batch_size=cfg.train.pred_len).to(
            dist_util.dev()
        )
        obj_verts = torch.from_numpy(obj_verts).to(dist_util.dev())
        obj_faces = torch.from_numpy(obj_faces).to(dist_util.dev())
        yield {
            "contact_record": contact_record.to(dist_util.dev()),
            "grab_seq_path": grab_seq_path,
            "text": df_prompts.loc[grab_seq_path_to_seq_id(grab_seq_path)]["Prompt"],
            "object_name": grab_seq_path_to_object_name(grab_seq_path),
            "n_frames": len(contact_record),
            "obj_fk": obj_fk,
            "obj_faces": obj_faces,
            "obj_verts": obj_verts,
            "start_frame": cfg_cpr.start_frame,
            "translation_obj": translation_obj,
        }


def iter_contact_pairs_kwargs_lst(cfg: SamplingConfig):
    df_prompts = get_df_prompts()
    cfg_cps: LoadContactPairsCfg = instantiate(cfg.load_contact_pairs_cfg)

    if cfg_cps.load_from == "dir":
        cps_paths = sorted(
            [
                os.path.join(cfg_cps.dir, f)
                for f in os.listdir(cfg_cps.dir)
                if f.endswith(".npz")
            ]
        )
        cps_paths = sorted(
            [
                os.path.join(cfg_cps.dir, f)
                for f in os.listdir(cfg_cps.dir)
                if f.endswith(".npz")
            ]
        )
    else:
        cps_paths = cfg_cps.files

    if cfg_cps.n_chunks is not None:
        assert cfg_cps.chunk_ind is not None
        assert cfg_cps.chunk_ind < cfg_cps.n_chunks

        n_total = len(cps_paths)
        chunk_size = (
            n_total + cfg_cps.n_chunks - 1
        ) // cfg_cps.n_chunks  # ceiling division
        start = cfg_cps.chunk_ind * chunk_size
        end = min(start + chunk_size, n_total)
        print(
            f"Processing chunk {cfg_cps.chunk_ind} of {cfg_cps.n_chunks}, {start} to {end} out of {n_total} contact pairs"
        )
        cps_paths = cps_paths[start:end]

    if cfg_cps.n_chunks is not None:
        assert cfg_cps.chunk_ind is not None
        assert cfg_cps.chunk_ind < cfg_cps.n_chunks

        n_total = len(cps_paths)
        chunk_size = (
            n_total + cfg_cps.n_chunks - 1
        ) // cfg_cps.n_chunks  # ceiling division
        start = cfg_cps.chunk_ind * chunk_size
        end = min(start + chunk_size, n_total)
        print(
            f"Processing chunk {cfg_cps.chunk_ind} of {cfg_cps.n_chunks}, {start} to {end} out of {n_total} contact pairs"
        )
        cps_paths = cps_paths[start:end]

    for i, cps_path in enumerate(cps_paths):
        contact_pairs = ContactPairsSequence.load(cps_path)
        grab_seq_path = contact_pairs.metadata.grab_seq_path
        if cfg.translation_obj is not None:
            translation_obj = torch.tensor(cfg.translation_obj)
            contact_pairs = contact_pairs.translate(translation_obj)
        else:
            translation_obj = None
        if cfg_cps.start_frame is not None or cfg_cps.end_frame is not None:
            contact_pairs = contact_pairs.cut(cfg_cps.start_frame, cfg_cps.end_frame)
        object_name = contact_pairs.metadata.object_name
        obj_verts, obj_faces = load_mesh(object_name, cfg.n_simplify_object)
        obj_fk = ObjectModel(v_template=obj_verts, batch_size=cfg.train.pred_len).to(
            dist_util.dev()
        )
        obj_verts = torch.from_numpy(obj_verts).to(dist_util.dev())
        obj_faces = torch.from_numpy(obj_faces).to(dist_util.dev())

        if cfg_cps.texts is not None:
            text = cfg_cps.texts[i]
        else:
            text = df_prompts.loc[grab_seq_path_to_seq_id(grab_seq_path)]["Prompt"]

        if cfg_cps.added_6dof_velocity is not None:
            contact_pairs.object_trans = add_velocity_to_6dof(
                cfg, cfg_cps, contact_pairs.contacts, contact_pairs.object_trans
            )

        yield {
            "contact_pairs": contact_pairs.to(dist_util.dev()),
            "obj_faces": obj_faces,
            "obj_verts": obj_verts,
            "grab_seq_path": grab_seq_path,
            "text": text,
            "obj_fk": obj_fk,
            "object_name": object_name,
            "n_frames": len(contact_pairs),
            "start_frame": cfg_cps.start_frame,
            "translation_obj": translation_obj,
        }


def add_velocity_to_6dof(cfg, cfg_cps, contacts, object_trans):

    mask_contact_frames = contacts.sum(dim=1) > 0
    # Find the first frame with contact
    first_contact_frame = torch.argmax(mask_contact_frames.float())

    # Apply velocity to object translation over time
    velocity = torch.tensor(cfg_cps.added_6dof_velocity, device=object_trans.device)
    # Scale velocity by time delta (1/fps) for proper physical simulation
    time_delta = 1.0 / cfg.train.get("fps", 20)
    scaled_velocity = velocity * time_delta

    # Create time steps for the entire sequence
    time_steps = torch.arange(object_trans.shape[0], device=object_trans.device)

    # Apply velocity to all frames after first contact (not just during contact)
    # This ensures the object continues moving with the applied velocity after contact ends
    active_steps = time_steps >= first_contact_frame
    effective_steps = torch.zeros_like(time_steps)
    effective_steps[active_steps] = time_steps[active_steps] - first_contact_frame
    effective_steps = effective_steps.unsqueeze(1)

    # Calculate velocity offset for all frames after first contact
    velocity_offset = effective_steps * scaled_velocity

    # Apply the offset to the object translation
    object_trans = object_trans + velocity_offset
    return object_trans


def add_velocity_to_6dof(cfg, cfg_cps, contacts, object_trans):

    mask_contact_frames = contacts.sum(dim=1) > 0
    # Find the first frame with contact
    first_contact_frame = torch.argmax(mask_contact_frames.float())

    # Apply velocity to object translation over time
    velocity = torch.tensor(cfg_cps.added_6dof_velocity, device=object_trans.device)
    # Scale velocity by time delta (1/fps) for proper physical simulation
    time_delta = 1.0 / cfg.train.get("fps", 20)
    scaled_velocity = velocity * time_delta

    # Create time steps for the entire sequence
    time_steps = torch.arange(object_trans.shape[0], device=object_trans.device)

    # Apply velocity to all frames after first contact (not just during contact)
    # This ensures the object continues moving with the applied velocity after contact ends
    active_steps = time_steps >= first_contact_frame
    effective_steps = torch.zeros_like(time_steps)
    effective_steps[active_steps] = time_steps[active_steps] - first_contact_frame
    effective_steps = effective_steps.unsqueeze(1)

    # Calculate velocity offset for all frames after first contact
    velocity_offset = effective_steps * scaled_velocity

    # Apply the offset to the object translation
    object_trans = object_trans + velocity_offset
    return object_trans


def iter_predict_contacts_mode_kwargs(cfg: SamplingConfig):
    raise NotImplementedError()
    cfg_predict: PredictContactPairsCfg = instantiate(cfg.predict_contact_pairs_cfg)
    df_prompts = get_df_prompts()
    grab_seq_paths = np.random.choice(
        get_all_grab_seq_paths(), cfg_predict.n_samples, replace=False
    )
    kw_args_lst = []
    for grab_seq_path in grab_seq_paths:
        kw_args_lst.append(
            {
                "text": df_prompts.loc[grab_seq_path_to_seq_id(grab_seq_path)][
                    "Prompt"
                ],
                "object_name": grab_seq_path_to_object_name(grab_seq_path),
                "grab_seq_path": grab_seq_path,
                "n_frames": cfg_predict.n_frames,
            }
        )
    return kw_args_lst


def get_no_dno_mode_kwargs(cfg: SamplingConfig):
    cfg_no_dno = cfg.no_dno_cfg
    df_prompts = get_df_prompts()
    grab_seq_paths = cfg_no_dno.grab_seq_paths
    kwargs_lst = []
    for grab_seq_path in grab_seq_paths:
        grab_seq = parse_npz(grab_seq_path)
        grab_seq = reduce_seq_data(grab_seq, 20)
        n_frames = grab_seq["n_frames"]
        kwargs_lst.append(
            {
                "text": df_prompts.loc[grab_seq_path_to_seq_id(grab_seq_path)][
                    "Prompt"
                ],
                "object_name": grab_seq_data_to_object_name(grab_seq),
                "grab_seq_path": grab_seq_path,
                "n_frames": n_frames,
                "start_frame": 0,
                "translation_obj": None,
            }
        )
    return kwargs_lst


def get_save_paths(grab_seq_path, text, out_dir: str, total_ind: int):
    txt_fname = text.replace(".", "").replace(" ", "_").replace("[SEP]", "-SEP-")
    grab_name = grab_seq_path_to_unique_name(grab_seq_path)
    result_name = f"hoi_animation_{total_ind}__{grab_name}__{txt_fname}.pickle"
    result_path = os.path.join(out_dir, result_name)
    save_paths = {
        "pickle": result_path,
        "blender": result_path.replace(".pickle", ".blend"),
        "dno_loss_fig": result_path.replace(".pickle", "_loss.png"),
        "dno_loss_fig_reduced": result_path.replace(
            ".pickle", "_loss_penetration_contact.png"
        ),
    }
    return save_paths


def save_artifacts(cfg, result, save_paths, total_ind: int):
    if cfg.save_result_pickle:
        print(f"Saving result to {save_paths['pickle']}")
        result.save(save_paths["pickle"])

    if cfg.use_dno:
        print(f"Plotting DNO loss to {save_paths['dno_loss_fig']}")
        plot_dno_loss.plot_dno_loss(
            [result.optim_info_lst], save_path=save_paths["dno_loss_fig"], show=False
        )
        print(f"Plotting DNO loss to {save_paths['dno_loss_fig_reduced']}")
        plot_dno_loss.plot_dno_loss(
            [result.optim_info_lst],
            save_path=save_paths["dno_loss_fig_reduced"],
            show=False,
            keys=list(cfg.dno_loss_coefficients.keys()),
            log_scale=False,
        )

    if cfg.anim_save and (
        cfg.anim_save_every_n_gens is None
        or total_ind % cfg.anim_save_every_n_gens == 0
    ):
        print(f"Visualizing result to {save_paths['blender']}")
        visualize_hoi_animation(
            [result.smpldata],
            object_path_or_name=result.object_name,
            grab_seq_path=result.grab_seq_path,
            text=result.text,
            start_frame=result.start_frame,
            translation_obj=result.translation_obj,
            save_path=save_paths["blender"],
            contact_record=result.contact_record,
            anim_setup=AnimationSetup[cfg.anim_setup],
            contact_pairs_seq=result.contact_pairs_seq,
            n_simplify_object=cfg.n_simplify_object,
            text_config=json.dumps(OmegaConf.to_container(cfg), indent=4),
            mat_lst=result.mat_lst,
        )


@hydra.main(version_base="1.2", config_path="configs", config_name="sampling_mdm.yaml")
def main(cfg: SamplingConfig):
    add_training_args(cfg)
    print(OmegaConf.to_yaml(cfg))
    device = get_least_busy_device()
    dist_util.setup_dist(device.index)
    # torch.autograd.set_detect_anomaly(True)  # debug
    # dist_util.setup_dist(-1)  # debug

    dno_loss_source = DnoLossSource(cfg.dno_loss_source)
    ######################
    # Prepare output dir
    ######################
    if cfg.out_dir is None:
        model_id = get_model_identifier(cfg.model_path)
        dir_name = "__".join(
            [
                "results_hoi",
                datetime.datetime.now().strftime("%m_%d_%H_%M"),
                "wo_dno" if not cfg.use_dno else f"dno_{dno_loss_source.value}",
                model_id,
            ]
        )
        out_dir = os.path.join(TMP_DIR, dir_name)
    else:
        out_dir = cfg.out_dir
    print(f"Saving results to {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    # save config for reproducibility
    OmegaConf.save(config=cfg, f=os.path.join(out_dir, "sampling_config.yaml"))

    #########################
    # Prepare input kwargs for different input modes
    #########################
    grab_seq_paths = None
    if cfg.use_dno:
        if dno_loss_source == DnoLossSource.FROM_CONTACT_RECORD:  # mainly for debugging
            input_kwargs_iterator = iter(iter_contact_record_kwargs_lst(cfg))
        elif dno_loss_source == DnoLossSource.FROM_CONTACT_PAIRS:  # main method
            input_kwargs_iterator = iter(iter_contact_pairs_kwargs_lst(cfg))
            grab_seq_paths = get_loaded_contact_pairs_grab_seq_paths(cfg)
        elif dno_loss_source == DnoLossSource.FROM_NN:  # baseline method
            raise NotImplementedError()
            input_kwargs_iterator = iter(iter_predict_contacts_mode_kwargs(cfg))
        else:
            raise ValueError()
    else:
        dno_cond = None
        input_kwargs_iterator = get_no_dno_mode_kwargs(cfg)
        grab_seq_paths = [e["grab_seq_path"] for e in input_kwargs_iterator]

    normalizer = Normalizer.from_dir(cfg.experiment_dir, device=dist_util.dev())
    dataloader = get_dataloader(cfg, grab_seq_paths)
    dataset = dataloader.dataset
    feature_processor = dataset.feature_processor
    model, diffusion = load_model(cfg.train, normalizer.n_feats)
    smpl_fk = SmplModelsFK.create("smplx", cfg.train.pred_len, device=dist_util.dev())

    #########################
    # Iterate over input kwargs
    #########################
    total_ind = 0
    for input_kwargs in tqdm(input_kwargs_iterator, desc="Hoi Sampling Sampling"):
        # must have kw_args
        text = input_kwargs["text"]
        object_name = input_kwargs["object_name"]
        n_frames = input_kwargs["n_frames"]
        grab_seq_path = input_kwargs["grab_seq_path"]
        start_frame = input_kwargs["start_frame"]

        # optional kw_args
        translation_obj = input_kwargs["translation_obj"]
        contact_record = input_kwargs.get("contact_record")
        contact_pairs = input_kwargs.get("contact_pairs")
        obj_fk = input_kwargs.get("obj_fk")
        obj_faces = input_kwargs.get("obj_faces")
        obj_verts = input_kwargs.get("obj_verts")
        nearest_neighbor_per_frame = input_kwargs.get("nearest_neighbor_per_frame")

        # if "s10" in grab_seq_path or "lift" in grab_seq_path:
        #     print(f"Grab seq path: {grab_seq_path}")
        #     print("Use only training set for inference")
        #     continue

        if "above_table" in cfg.dno_loss_coefficients:
            table_corner_locs = get_table_params(grab_seq_path)[2]
            table_corner_locs = table_corner_locs.expand(cfg.batch_size_gen, -1, -1)
        else:
            table_corner_locs = None

        #########################
        # Prepare DNO condition
        #########################
        if cfg.use_dno:
            dno_cond = DnoConditionOld(
                smpl_fk=smpl_fk,
                normalizer=normalizer,
                feature_processor=feature_processor,
                dno_loss_source=dno_loss_source,
                th_contact_force=cfg.th_contact_force,
                n_simplify_faces_hand=cfg.n_simplify_hands,
                loss_coefficients=instantiate(cfg.dno_loss_coefficients),
                obj_fk=obj_fk,
                obj_faces=obj_faces,
                obj_verts=obj_verts,
                nearest_neighbor_per_frame=nearest_neighbor_per_frame,
                table_corner_locs=table_corner_locs,
            )
        else:
            dno_cond = None

        #########################
        # Set model inputs
        #########################
        model_kwargs, motion_shape = get_input_params(
            cfg, dataset, model, n_frames, grab_seq_path, text, object_name
        )

        #########################
        # Sample
        #########################
        sample_raw, optim_info_lst = auto_regressive_dno(
            cfg,
            model,
            n_frames,
            diffusion.ddim_sample_loop,
            instantiate(cfg.dno_options),
            shape=motion_shape,
            model_kwargs=model_kwargs,
            use_dno=cfg.use_dno,
            dno_cond=dno_cond,
            sequence_conditions=contact_record or contact_pairs,
        )

        #########################
        # Denormalize, decode features and fill in missing features using Forward Kinematics
        #########################
        smpldata_lst = samples_to_smpldata_lst(
            sample_raw, normalizer, feature_processor
        )
        for smpldata in smpldata_lst:
            if feature_processor.is_fk_required():
                smpldata.fill_in_using_fk()

        #########################
        # Iterate over batch elements and save results and artifacts
        #########################
        curr_batch_size = sample_raw.shape[0]
        for b in range(curr_batch_size):
            if cfg.save_result_only_first_in_batch and b != 0:
                continue
            if cfg.save_result_only_first_in_batch and b != 0:
                continue
            #########################
            # Create result container
            #########################
            result = HoiResult(
                raw_sample=sample_raw[b].detach().clone(),
                text=text,
                object_name=object_name,
                grab_seq_path=grab_seq_path,
                start_frame=start_frame,
                translation_obj=translation_obj,
                contact_record=contact_record,
                optim_info_lst=(
                    optim_info_lst[b] if optim_info_lst is not None else None
                ),
                contact_pairs_seq=contact_pairs,
                smpldata=smpldata_lst[b].detach().to("cpu"),
            )

            #########################
            # Define save paths
            #########################
            total_ind += 1
            result_save_paths = get_save_paths(grab_seq_path, text, out_dir, total_ind)

            #########################
            # Save artifacts
            #########################
            save_artifacts(cfg, result, result_save_paths, total_ind)
            save_artifacts(cfg, result, result_save_paths, total_ind)


if __name__ == "__main__":
    main()


"""
conda activate mahoi
export PYTHONPATH=$(pwd)
python inference_hoi_model.py

"""
