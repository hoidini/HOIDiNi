import datetime
import hydra
from dataclasses import dataclass
import os
from typing import Optional
import numpy as np
from omegaconf import OmegaConf, open_dict
import torch
from hydra.utils import instantiate
from hoidini.blender_utils.general_blender_utils import (
    CollectionManager,
    blend_scp_and_run,
    reset_blender,
    save_blender_file,
    set_fps,
    set_frame_end,
)
from hoidini.blender_utils.visualize_mesh_figure_blender import animate_mesh
from hoidini.closd.diffusion_planner.utils import dist_util
from hoidini.closd.diffusion_planner.utils.model_util import (
    create_gaussian_diffusion,
    load_saved_model,
)
from hoidini.closd.diffusion_planner.utils.sampler_util import ClassifierFreeSampleModel
from hoidini.datasets.grab.grab_utils import (
    get_all_grab_seq_paths,
    get_df_grab_index,
    grab_seq_id_to_seq_path,
    get_grab_split_seq_paths,
    grab_seq_path_to_object_name,
    grab_seq_path_to_seq_id,
    grab_seq_path_to_unique_name,
    get_df_prompts,
)
from hoidini.general_utils import (
    TMP_DIR,
    create_new_dir,
    get_least_busy_device,
    get_model_identifier,
    read_json,
)
from hoidini.inference_hoi_model import InitConfig
from hoidini.inference_utils import auto_regressive_dno
from hoidini.model_utils import model_kwargs_to_device
from hoidini.normalizer import Normalizer
from hoidini.object_contact_prediction.animate_contact_pair_sequence import (
    animate_contact_pair_sequence,
)
from hoidini.object_contact_prediction.cpdm_dataset import (
    ContactPairsDataset,
    ContactPairsSequence,
    CpsMetadata,
    get_contact_pairs_dataloader,
)
from hoidini.object_contact_prediction.cpdm import CPDM
from hoidini.object_contact_prediction.cpdm_dno_conds import (
    AboveTableLoss,
    CpdmDNOCond,
    Similar6DoFLoss,
    get_geoms_batch,
)
from hoidini.optimize_latent.dno import DNOOptions
from hoidini.plot_dno_loss import plot_dno_loss


VIS_OFFSETS = {
    "wo_dno": torch.tensor([0, 0.8, 0.0]),
    "w_dno": torch.tensor([0, 1.6, 0]),
    "real": torch.tensor([0, 0, 0]),
}
VIS_COLORS = {
    "wo_dno": (0, 0, 1, 0.8),
    "w_dno": (1, 0, 0, 0.8),
    "real": (0, 1, 0, 0.8),
}


DNO_OPTIONS = DNOOptions(
    num_opt_steps=60,
    lr=0.5,
    lr_warm_up_steps=30,
    perturb_scale=0.00001,
    diff_penalty_scale=0.01,
    decorrelate_scale=10,
)


@dataclass
class CPDMSamplingConfig:
    model_path: str
    grab_dataset_path: str
    max_batch_size: int
    autoregressive_include_prefix: bool
    guidance_param: float
    cond_source_to_use: str  # "grab_seq_paths", "specified", "random", "seqs_json"
    n_frames: Optional[int] = None
    cond_source_grab_seq_paths: Optional[list[str]] = None
    cond_source_specified: Optional[list[dict]] = None
    cond_source_random: Optional[dict] = None
    cond_source_seqs_json: Optional[str] = None
    out_dir: Optional[str] = None
    visualize: bool = True
    grab_seq_paths: Optional[list[str]] = None
    use_dno: bool = True
    dno_options: InitConfig[DNOOptions] = None
    train: dict = None


def add_training_args(cfg: CPDMSamplingConfig):
    experiment_dir = os.path.dirname(cfg.model_path)
    cfg_train = OmegaConf.load(os.path.join(experiment_dir, "args.json"))
    cfg_train.experiment_dir = experiment_dir
    cfg_train.model_path = cfg.model_path
    with open_dict(cfg):
        cfg.experiment_dir = experiment_dir
        cfg.train = cfg_train


def get_grab_seq_path_gen_args(grab_seq_path: str, fps: int = 20):
    df_prompts = get_df_prompts()
    df_index = get_df_grab_index(tgt_fps=fps)
    n_frames = df_index.loc[grab_seq_path_to_seq_id(grab_seq_path)]["n_frames"]
    return {
        "grab_seq_path": grab_seq_path,
        "text": df_prompts.loc[grab_seq_path_to_seq_id(grab_seq_path)]["Prompt"],
        "object_name": grab_seq_path_to_object_name(grab_seq_path),
        "n_frames": n_frames,
    }


def get_save_paths(
    out_dir,
    model_identifier,
    batch_size,
    batch_idx,
    i,
    grab_seq_path,
    object_name,
    text,
):
    total_ind = batch_idx * batch_size + i
    text_str = text.replace(" ", "_").replace(".", "_").replace(",", "_")
    grab_id = grab_seq_path_to_unique_name(grab_seq_path)
    save_paths = {}
    base_fname = f"contact_pairs__{model_identifier}__{total_ind:04d}__{grab_id}__{object_name}__{text_str}"
    fpath_template = os.path.join(out_dir, f"{base_fname}_SUFFIX")
    save_paths["cps_w_dno"] = fpath_template.replace("SUFFIX", "w_dno.npz")
    save_paths["cps_wo_dno"] = fpath_template.replace("SUFFIX", "wo_dno.npz")
    save_paths["blend_vis"] = os.path.join(out_dir, f"{base_fname}_vis.blend")
    # save_paths["dno_loss_graph"] = fpath_template.replace("SUFFIX", "dno_loss_graph.png")
    # save_paths["dno_loss_graph_reduced"] = fpath_template.replace("SUFFIX", "dno_loss_graph_reduced.png")
    return save_paths


def get_specified_gen_args(cfg):
    raise NotImplementedError("Only support grab_seq_paths for now")
    all_grab_seq_paths = get_all_grab_seq_paths(cfg.grab_dataset_path)
    df_prompts = get_df_prompts()
    gen_arg_lst = []
    for d in cfg.cond_source_specified:
        obj_name = d["object_name"]
        if d["real_prefix"] is not None:
            grab_seq_path = d["real_prefix"]
            if not os.path.exists(grab_seq_path):
                grab_seq_path = os.path.join(cfg.grab_dataset_path, grab_seq_path)
        else:
            grab_seq_path = np.random.choice(all_grab_seq_paths)
        text = df_prompts.loc[grab_seq_path_to_seq_id(grab_seq_path)]["Prompt"]

        gen_arg_lst.append(
            {
                "grab_seq_path": grab_seq_path,
                "text": text,
                "object_name": obj_name,
            }
        )
    return gen_arg_lst


def cpdm_inference(cfg: CPDMSamplingConfig):
    add_training_args(cfg)
    print(OmegaConf.to_yaml(cfg))
    cfg_train = cfg.train
    device = get_least_busy_device()
    dist_util.setup_dist(device.index)

    model_identifier = get_model_identifier(cfg.model_path)
    if cfg.out_dir is None:
        cfg.out_dir = os.path.join(
            TMP_DIR,
            f"cpdm_{datetime.datetime.now().strftime('%m_%d_%H_%M')}_{model_identifier}",
        )
    print(f"Output directory: {cfg.out_dir}")
    create_new_dir(cfg.out_dir)
    OmegaConf.save(config=cfg, f=os.path.join(cfg.out_dir, "sampling_cpdm.yaml"))

    if cfg.cond_source_to_use == "grab_seq_paths":
        grab_seq_paths = cfg.cond_source_grab_seq_paths
    elif cfg.cond_source_to_use == "specified":
        raise NotImplementedError("Not implemented")
    elif cfg.cond_source_to_use == "random":
        grab_train_seq_paths = get_grab_split_seq_paths("train", cfg.grab_dataset_path)
        n_samples = cfg.cond_source_random["n_samples"]
        replace = n_samples > len(grab_train_seq_paths)
        grab_seq_paths = np.random.choice(
            grab_train_seq_paths, size=n_samples, replace=replace
        )
    elif cfg.cond_source_to_use == "seqs_json":
        seq_ids = read_json(cfg.cond_source_seqs_json)
        grab_seq_paths = [grab_seq_id_to_seq_path(seq_id) for seq_id in seq_ids]
    else:
        raise ValueError(f"Invalid condition source: {cfg.cond_source_to_use}")

    batch_size = min(cfg.max_batch_size, len(grab_seq_paths))

    data = get_contact_pairs_dataloader(
        grab_dataset_path=cfg.grab_dataset_path,
        batch_size=batch_size,
        experiment_dir=cfg.train.save_dir,
        is_training=False,
        context_len=cfg.train.context_len,
        pred_len=cfg.train.pred_len,
        pred_len_dataset=2,  # we only need the prefix
        n_points=cfg.train.pcd_n_points,
        feature_names=cfg.train.get("feature_names"),
        grab_seq_paths=grab_seq_paths,
        use_sampler=True,
        is_enforce_motion_length=True,
    )
    dataset: ContactPairsDataset = data.dataset
    diffusion = create_gaussian_diffusion(cfg_train)
    model = CPDM(
        pred_len=cfg_train.pred_len,
        context_len=cfg_train.context_len,
        n_feats=data.dataset.n_feats,
        num_layers=cfg_train.layers,
        cond_mask_prob=cfg_train.cond_mask_prob,
    )
    model.eval()
    normalizer = Normalizer.from_dir(cfg_train.save_dir, device=dist_util.dev())

    load_saved_model(model, cfg.model_path, use_avg=cfg_train.use_ema)
    if cfg.guidance_param != 1.0:
        model = ClassifierFreeSampleModel(model)
    model.freeze_object_encoder()
    model.to(dist_util.dev())
    model.eval()
    for batch_idx, (_, model_kwargs) in enumerate(data):
        model_kwargs_to_device(model_kwargs, dist_util.dev())
        with torch.no_grad():
            model_kwargs["y"]["obj_emb"] = model.object_encoder(
                model_kwargs["y"]["obj_points"]
            )
        model_kwargs["y"]["scale"] = (
            torch.ones(batch_size, device=dist_util.dev()) * cfg.guidance_param
        )

        cur_batch_size = model_kwargs["y"]["prefix"].shape[0]
        if cfg.n_frames is None:
            grab_seq_paths = [m["grab_seq_path"] for m in model_kwargs["y"]["metadata"]]
            n_frames_lst = [
                get_grab_seq_path_gen_args(p, cfg.train.fps)["n_frames"]
                for p in grab_seq_paths
            ]
        else:
            n_frames_lst = [cfg.n_frames] * cur_batch_size

        n_frames_max = max(n_frames_lst)
        motion_shape = (batch_size, dataset.n_feats, 1, n_frames_max)

        features_per_mode = {}
        for mode_name, use_dno in [("w_dno", True), ("wo_dno", False)]:
            if not cfg.use_dno and mode_name == "w_dno":
                continue
            if use_dno:
                (
                    table_faces_batch,
                    table_verts_batch,
                    table_corner_locs_batch,
                    obj_v_template_batch,
                ) = get_geoms_batch(model_kwargs)

                dno_losses = {
                    # "reach_table": (0.2, ReachTablesLoss(table_corner_locs, torch.arange(70, 115))),
                    # "random_offset_table": (0.2, RandomOffsetTableLoss(table_corner_locs, torch.arange(70, 100), offset_std=0.0)),
                    "above_table": (0.5, AboveTableLoss(table_corner_locs_batch)),
                    "specific_6d": (
                        0.5,
                        Similar6DoFLoss(
                            src_frame=1,
                            target_frames=[
                                torch.arange(nf - 10, nf) for nf in n_frames_lst
                            ],
                        ),
                    ),
                }
                assert not (
                    use_dno
                    and "reach_table" in dno_losses
                    and "random_offset_table" in dno_losses
                ), "reach_table and random_offset_table cannot be used together"
                dno_cond = CpdmDNOCond(
                    normalizer,
                    dataset.feature_processor,
                    dno_losses,
                    obj_v_template_batch,
                )
            else:
                dno_cond = None
            features, dno_loss_dict = auto_regressive_dno(
                cfg,
                model,
                n_frames_max,
                diffusion.ddim_sample_loop,
                instantiate(cfg.dno_options),
                motion_shape,
                model_kwargs,
                use_dno=use_dno,
                dno_cond=dno_cond,
                seed=batch_idx,
            )
            if use_dno:
                plot_dno_loss(
                    dno_loss_dict,
                    save_path=os.path.join(".", "dno_loss.png"),
                    log_scale=False,
                )
                plot_dno_loss(
                    dno_loss_dict,
                    save_path=os.path.join(".", "dno_loss_reduced.png"),
                    log_scale=False,
                    keys=list(dno_losses.keys()),
                )

            features = features.squeeze(2).permute(0, 2, 1)  # (bs, seq_len, n_features)
            features_per_mode[mode_name] = features

        # Save and (optionally) visualize the contact pairs
        for b in range(len(features)):
            grab_seq_path = model_kwargs["y"]["metadata"][b]["grab_seq_path"]
            object_name = model_kwargs["y"]["metadata"][b]["object_name"]
            text = model_kwargs["y"]["text"][b]

            contact_pairs_per_mode = {}
            save_paths = get_save_paths(
                cfg.out_dir,
                model_identifier,
                batch_size,
                batch_idx,
                b,
                grab_seq_path,
                object_name,
                text,
            )
            for mode_name, features in features_per_mode.items():
                n_frames = n_frames_lst[b]
                feature = normalizer.denormalize(features[b][:n_frames])
                contact_pairs: ContactPairsSequence = (
                    dataset.feature_processor.decode_features(
                        feature.detach().cpu(), trans_mode=cfg.decode_mode
                    )
                )
                contact_pairs.metadata = CpsMetadata(grab_seq_path, object_name, text)

                contact_pairs_per_mode[mode_name] = contact_pairs
                if mode_name != "w_dno":  # only save dno results
                    continue
                print(f"Saving contact pairs to {save_paths['cps_w_dno']}")
                contact_pairs.save(save_paths["cps_w_dno"])

            # Visualize generated and real contact pairs
            if cfg.visualize:
                reset_blender()
                # visualize wo_dno result
                animate_contact_pair_sequence(
                    contact_pairs_per_mode["wo_dno"],
                    object_name,
                    unq_id="Generated_wo_dno",
                    offset_all=VIS_OFFSETS["wo_dno"],
                    reset_blender=False,
                    meshes_color=(0, 0, 1, 0.8),
                    grab_seq_path_for_table=grab_seq_path,
                )
                # visualize dno result
                if cfg.use_dno:
                    table_verts_anim = (
                        table_verts_batch[b] + VIS_OFFSETS["w_dno"].cpu().numpy()
                    )
                    animate_mesh(
                        "TableDnoCond",
                        table_faces_batch[b],
                        table_verts_anim[None],
                        color=VIS_COLORS["w_dno"],
                    )
                    animate_contact_pair_sequence(
                        contact_pairs_per_mode["w_dno"],
                        object_name,
                        unq_id="Generated_w_dno",
                        offset_all=VIS_OFFSETS["w_dno"],
                        reset_blender=False,
                        meshes_color=VIS_COLORS["w_dno"],
                        grab_seq_path_for_table=grab_seq_path,
                    )
                # visualize real contact pairs
                dp_ind = [data["grab_seq_path"] for data in dataset.data].index(
                    grab_seq_path
                )
                real_contact_pairs = dataset.data[dp_ind]["contact_pairs"]
                animate_contact_pair_sequence(
                    real_contact_pairs,
                    object_name,
                    unq_id="Real",
                    reset_blender=False,
                    offset_all=VIS_OFFSETS["real"],
                    meshes_color=VIS_COLORS["real"],
                    grab_seq_path_for_table=grab_seq_path,
                )
                collections_map = {
                    "Generated_wo_dno": "Generated_wo_dno",
                    "Real": "Real",
                    "Generated_w_dno": "Generated_w_dno" if cfg.use_dno else None,
                }
                CollectionManager.organize_collections(collections_map)
                set_frame_end(end_frame=len(contact_pairs_per_mode["wo_dno"]))
                set_fps(cfg.train.get("fps", 20))
                blender_vis_path = save_paths["blend_vis"]
                save_blender_file(blender_vis_path)
                blend_scp_and_run(blender_vis_path)


@hydra.main(
    version_base="1.2", config_path="../configs", config_name="sampling_cpdm.yaml"
)
def main(cfg: CPDMSamplingConfig):
    cpdm_inference(cfg)


if __name__ == "__main__":
    main()
