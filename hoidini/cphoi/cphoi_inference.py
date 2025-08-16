from dataclasses import dataclass, field
from functools import partial
import json
import os
import hydra
from typing import Dict, List, Optional, Union
import numpy as np
from omegaconf import OmegaConf
import torch
from hydra.utils import instantiate
import bpy
from hydra.core.config_store import ConfigStore

from hoidini.blender_utils.general_blender_utils import (
    CollectionManager,
    blend_scp_and_run,
    reset_blender,
)
from hoidini.blender_utils.visualize_hoi_animation import (
    AnimationSetup,
    visualize_hoi_animation,
)
from hoidini.blender_utils.visualize_stick_figure_blender import save_blender_file
from hoidini.closd.diffusion_planner.utils import dist_util
from hoidini.closd.diffusion_planner.utils.model_util import (
    create_gaussian_diffusion,
    load_saved_model,
)
from hoidini.closd.diffusion_planner.utils.sampler_util import ClassifierFreeSampleModel
from hoidini.cphoi.cphoi_dataset import LazyInferenceDataset, TfmsManager
from hoidini.cphoi.cphoi_model import CPHOI
from hoidini.cphoi.cphoi_optim_losses import get_cphoi_geometries_batch
from hoidini.cphoi.samplers.samplers import (
    InferenceJob,
    KitchenSampler,
    KitchenSamplerPan,
)
from hoidini.cphoi.cphoi_utils import (
    FeaturesDecoderWrapper,
    extract_local_obj_points_w_nn,
    smpldata_to_contact_pairs,
)
from hoidini.cphoi.samplers.samplers import BaseSampler
from hoidini.datasets.dataset_smplrifke import collate_smplrifke_mdm
from hoidini.datasets.smpldata import SmplData, SmplModelsFK
from hoidini.general_utils import get_least_busy_device, get_model_identifier
from hoidini.geometry3d.hands_intersection_loss import HandIntersectionLoss
from hoidini.inference_hoi_model import HoiResult
from hoidini.inference_utils import auto_regressive_dno
from hoidini.model_utils import model_kwargs_to_device
from hoidini.normalizer import Normalizer
from hoidini.object_contact_prediction.cpdm_inference import add_training_args
from hoidini.datasets.grab.grab_utils import grab_seq_path_to_unique_name
from hoidini.objects_fk import ObjectModel
from hoidini.object_contact_prediction.cpdm_dno_conds import (
    AboveTableLoss,
    BatchedObjectModel,
    KeepObjectStaticLoss,
    Similar6DoFLoss,
    get_geoms_batch,
)
from hoidini.optimize_latent.dno_loss_functions import (
    FootSkateLoss,
    HandsContactLoss,
    HoiAboveTableLoss,
    HoiSideTableLoss,
    JointsJitterLoss,
)
from hoidini.optimize_latent.dno_losses import DnoLossComponent
from hoidini.plot_dno_loss import plot_dno_loss
from hoidini.optimize_latent.dno import DNOOptions
from hoidini.optimize_latent.dno_losses import DnoCondition, DnoLossSource


@dataclass
class CphoiInferenceConfig:
    model_path: str = field()
    out_dir: str = field()
    grab_dataset_path: str = field()
    batch_size_gen: int = field()
    device: Optional[int] = field(default=None)

    sampler_config: Optional[dict] = field(default=None)  # Define the sampling tasks
    autoregressive_include_prefix: bool = field(default=True)
    guidance_param: float = field(default=3.0)
    n_simplify_object: int = field(default=3000)
    th_contact_force: float = field(default=0.3)
    n_simplify_hands: int = field(default=1000)
    seed: Optional[int] = field(default=None)

    # DNO
    one_phase_experiment: bool = field(default=False)
    dno_options_phase1: Optional[dict] = field(default=None)
    dno_options_phase2: Optional[dict] = field(default=None)
    dno_loss_coefficients_phase1: Optional[dict] = field(default=None)
    dno_loss_coefficients_phase2: Optional[dict] = field(default=None)
    use_dno_object: bool = field(default=True)
    use_dno_body: bool = field(default=True)

    # Artifacts
    result_phases_to_save: List[str] = field(default_factory=lambda: ["final"])
    anim_setup: str = field(default="NO_MESH")
    result_phases_to_animate: List[str] = field(
        default_factory=lambda: ["phase0", "phase1", "phase2", "final"]
    )
    anim_save_every_n_gens: Optional[int] = field(default=None)
    store_loss_data: bool = field(default=False)

    # classifier guidance (paper experiments)
    use_classifier_guidance: bool = field(default=False)
    classifier_guidance_lr_factor: Optional[float] = field(default=None)
    classifier_guidance_n_steps: Optional[int] = field(default=None)

    # nearest neighbors (paper experiments)
    replace_cps_with_nearest_neighbors: bool = field(default=False)

    # Training cfg
    train: Optional[dict] = field(default=None)


cs = ConfigStore.instance()
cs.store(name="cphoi_inference_config", node=CphoiInferenceConfig)


def get_save_paths(
    out_dir, model_identifier, total_ind, grab_seq_path, object_name, text
):
    text_str = (
        text.replace(" ", "_")
        .replace(".", "_")
        .replace(",", "_")
        .replace("[SEP]", "_SEP_")
    )
    grab_id = grab_seq_path_to_unique_name(grab_seq_path)
    save_paths = {}
    object_name = object_name.split("/")[-1].split(".")[0]
    base_fname = f"cphoi__{model_identifier}__{total_ind:04d}__{grab_id}__{object_name}__{text_str}"
    fpath_template = os.path.join(out_dir, f"{base_fname}_SUFFIX")
    save_paths["phase0_pickle"] = fpath_template.replace("SUFFIX", "phase0.pickle")
    save_paths["phase1_pickle"] = fpath_template.replace("SUFFIX", "phase1.pickle")
    save_paths["phase2_pickle"] = fpath_template.replace("SUFFIX", "phase2.pickle")
    save_paths["final_pickle"] = fpath_template.replace("SUFFIX", "final.pickle")
    save_paths["blend_vis"] = os.path.join(out_dir, f"{base_fname}_vis.blend")
    save_paths["phase1_dno_loss_fig"] = fpath_template.replace(
        "SUFFIX", "phase1_dno_loss.png"
    )
    save_paths["phase2_dno_loss_fig"] = fpath_template.replace(
        "SUFFIX", "phase2_dno_loss.png"
    )
    save_paths["phase1_dno_loss_fig_reduced"] = fpath_template.replace(
        "SUFFIX", "phase1_dno_loss_reduced.png"
    )
    save_paths["phase2_dno_loss_fig_reduced"] = fpath_template.replace(
        "SUFFIX", "phase2_dno_loss_reduced.png"
    )

    # for one phase experiment
    save_paths["final_dno_loss_fig"] = fpath_template.replace(
        "SUFFIX", "final_dno_loss.png"
    )
    save_paths["final_dno_loss_fig_reduced"] = fpath_template.replace(
        "SUFFIX", "final_dno_loss_reduced.png"
    )
    return save_paths


KEYS = ["stick_figure", "mesh", "object_6dof"]
COLORS_PER_PHASE = {
    "phase0": {"stick_figure": (127 / 255, 88 / 255, 175 / 255, 1)},
    "phase1": {"stick_figure": (100 / 255, 197 / 255, 235 / 255, 1)},
    "phase2": {"stick_figure": (232 / 255, 77 / 255, 138 / 255, 1)},
    "final": {"stick_figure": (254 / 255, 179 / 255, 38 / 255, 1)},
}


def get_phase1_default_dno_losses(
    cfg: CphoiInferenceConfig,
    geoms_batch: Dict[str, torch.Tensor],
    n_frames_lst: List[int],
) -> List[DnoLossComponent]:
    loss_coeffs = instantiate(cfg.dno_loss_coefficients_phase1)
    dno_losses = [
        DnoLossComponent(
            name="above_table",
            weight=loss_coeffs.above_table,
            loss_fn=AboveTableLoss(
                torch.stack(geoms_batch["table_corner_locs"]).to(dist_util.dev())
            ),
        ),
        DnoLossComponent(
            name="similar_6dof_1",
            weight=loss_coeffs.similar_6dof_1,
            loss_fn=Similar6DoFLoss(
                src_frame=1, target_frames=[torch.arange(0, 28) for _ in n_frames_lst]
            ).to(dist_util.dev()),
        ),
        DnoLossComponent(
            name="similar_6dof_2",
            weight=loss_coeffs.similar_6dof_2,
            loss_fn=Similar6DoFLoss(
                src_frame=1,
                target_frames=[torch.arange(nf - 22, nf) for nf in n_frames_lst],
            ).to(dist_util.dev()),
        ),
        DnoLossComponent(
            name="keep_object_static",
            weight=loss_coeffs.keep_object_static,
            loss_fn=KeepObjectStaticLoss().to(dist_util.dev()),
        ),
    ]
    return dno_losses


def get_phase2_default_dno_losses(
    cfg: CphoiInferenceConfig, table_corner_locs: torch.Tensor
) -> List[DnoLossComponent]:
    loss_coeffs = instantiate(cfg.dno_loss_coefficients_phase2)
    dno_losses = [
        DnoLossComponent(
            name="contact",
            weight=loss_coeffs.contact,
            loss_fn=HandsContactLoss(cfg.th_contact_force, use_anchors=True),
        ),
        DnoLossComponent(
            name="penetration",
            weight=loss_coeffs.penetration,
            loss_fn=HandIntersectionLoss(
                device=dist_util.dev(), n_simplify_faces_hands=cfg.n_simplify_hands
            ),
        ),
        DnoLossComponent(
            name="above_table",
            weight=loss_coeffs.above_table,
            loss_fn=HoiAboveTableLoss(table_corner_locs),
        ),
        DnoLossComponent(
            name="side_table",
            weight=loss_coeffs.side_table,
            loss_fn=HoiSideTableLoss(table_corner_locs),
        ),
        DnoLossComponent(
            name="jitter_joints",
            weight=loss_coeffs.jitter_joints,
            loss_fn=JointsJitterLoss(fps=20, p=2.0, threshold=0.01),
        ),
        DnoLossComponent(
            name="foot_skate",
            weight=loss_coeffs.foot_skate,
            loss_fn=FootSkateLoss(th_contact=0.02),
        ),
    ]
    return dno_losses


def get_phase1_loss_comps(
    cfg, sampler, model_kwargs, n_frames_lst
) -> List[DnoLossComponent]:
    if hasattr(sampler, "get_phase1_dno_losses"):
        phase1_dno_loss_comps = sampler.get_phase1_dno_losses()
    else:
        phase1_dno_loss_comps = get_phase1_default_dno_losses(
            cfg, get_geoms_batch(model_kwargs), n_frames_lst
        )
    return phase1_dno_loss_comps


def get_phase2_loss_comps(cfg, sampler, phase2_geometries) -> List[DnoLossComponent]:
    if not isinstance(sampler, KitchenSampler):
        table_corner_locs = torch.stack(phase2_geometries["table_corner_locs"])
    else:
        table_corner_locs = sampler.get_table_corner_locs().to(dist_util.dev())
        assert table_corner_locs.dim() == 3
    phase2_dno_loss_comps = get_phase2_default_dno_losses(cfg, table_corner_locs)
    return phase2_dno_loss_comps


class SamplingFlow:
    def __init__(
        self,
        cfg: CphoiInferenceConfig,
        diffusion,
        decoder_wrapper: FeaturesDecoderWrapper,
        model,
        smpl_fk,
        seed,
        model_kwargs,
        n_frames_lst,
        motion_shape,
        tfms_manager,
        dno_opts1: DNOOptions,
        dno_opts2: DNOOptions,
        phase1_dno_loss_comps: List[DnoLossComponent],
        phase2_dno_loss_comps: List[DnoLossComponent],
        cphoi_phase2_geometries: Dict[str, torch.Tensor],
        sampler=None,
    ):
        self.cfg = cfg
        self.diffusion = diffusion
        self.decoder_wrapper = decoder_wrapper
        self.model = model
        self.smpl_fk = smpl_fk
        self.seed = seed
        self.model_kwargs = model_kwargs
        self.n_frames_lst = n_frames_lst
        self.motion_shape = motion_shape
        self.tfms_manager = tfms_manager
        self.dno_options_phase1 = dno_opts1
        self.dno_options_phase2 = dno_opts2
        self.phase1_dno_loss_comps = phase1_dno_loss_comps
        self.phase2_dno_loss_comps = phase2_dno_loss_comps
        self.sampler = sampler
        self.cphoi_phase2_geometries = cphoi_phase2_geometries
        self.debug_data = {}

    def run(self):
        smpldata_lst_phase0 = self.phase0()  # no DNO   # (15 + 85)

        samples_dict = {"phase0": smpldata_lst_phase0}
        if self.cfg.one_phase_experiment:
            smpldata_lst_final = self.phase_1_and_2_together()
        else:
            smpldata_lst_phase1 = self.phase1()  # DNO on the object
            if self.cfg.replace_cps_with_nearest_neighbors:
                smpldata_lst_phase1 = extract_local_obj_points_w_nn(
                    smpldata_lst_phase1, self.cphoi_phase2_geometries["obj_v_template"]
                )
            smpldata_lst_phase2 = self.phase2(
                smpldata_lst_phase1
            )  # DNO on the body using the contact pairs from phase1
            samples_dict.update(
                {
                    "phase1": smpldata_lst_phase1,
                    "phase2": smpldata_lst_phase2,
                }
            )
            smpldata_lst_final = self.merge_samples(
                smpldata_lst_phase1, smpldata_lst_phase2
            )
        # cut to required length
        smpldata_lst_final = [
            sd.cut(0, n_frames)
            for sd, n_frames in zip(smpldata_lst_final, self.n_frames_lst)
        ]
        samples_dict["final"] = smpldata_lst_final

        if self.cfg.store_loss_data:
            for phase, debug_data in self.debug_data.items():
                if "optim_info_lst" not in debug_data:
                    continue
                optim_info_lst = debug_data["optim_info_lst"]
                if optim_info_lst is None:
                    continue
                for ar_step in range(len(optim_info_lst)):
                    optim_info = optim_info_lst[ar_step]
                    for oi in optim_info:
                        x_lst = oi["x"]
                        oi["smpldata"] = [
                            self.decoder_wrapper.decode(
                                x.to(dist_util.dev()).unsqueeze(0)
                            )
                            for x in x_lst
                        ]

        return {"samples": samples_dict, "debug_data": self.debug_data}

    def merge_samples(self, smpldata_lst_phase1, smpldata_lst_phase2):
        assert len(smpldata_lst_phase1) == len(smpldata_lst_phase2)
        final_smpldata_lst = []
        for i in range(len(smpldata_lst_phase1)):
            smpldata_phase1 = smpldata_lst_phase1[i]
            smpldata_phase2 = smpldata_lst_phase2[i]
            assert len(smpldata_phase1) == len(smpldata_phase2)
            smpldata = SmplData(
                # body
                poses=smpldata_phase2.poses,
                trans=smpldata_phase2.trans,
                joints=smpldata_phase2.joints,  # not really necessary
                # object 6DoF
                poses_obj=smpldata_phase1.poses_obj,
                trans_obj=smpldata_phase1.trans_obj,
                # contact
                local_object_points=smpldata_phase1.local_object_points,
                contact=smpldata_phase1.contact,
            )
            final_smpldata_lst.append(smpldata)
        return final_smpldata_lst

    def get_dno_cond(self, phase: str) -> DnoCondition:
        assert phase in ["phase1", "phase2", "one_phase"]
        smpl_fk = None
        obj_model = None
        obj_fk_lst = None
        obj_faces_lst = None
        obj_verts_lst = None

        if phase in ["phase1", "one_phase"]:
            loss_components = self.phase1_dno_loss_comps
            geoms_batch = get_geoms_batch(self.model_kwargs)
            obj_model = BatchedObjectModel(torch.stack(geoms_batch["obj_v_template"]))
        if phase in ["phase2", "one_phase"]:
            smpl_fk = self.smpl_fk
            loss_components = self.phase2_dno_loss_comps
            obj_fk_lst = [
                ObjectModel(obj_v_template, batch_size=self.cfg.train.pred_len).to(
                    dist_util.dev()
                )
                for obj_v_template in self.cphoi_phase2_geometries["obj_v_template"]
            ]
            obj_faces_lst = [
                torch.tensor(obj_faces).to(dist_util.dev())
                for obj_faces in self.cphoi_phase2_geometries["obj_faces"]
            ]
            obj_verts_lst = [
                torch.tensor(obj_verts).to(dist_util.dev())
                for obj_verts in self.cphoi_phase2_geometries["obj_v_template"]
            ]
        if phase == "one_phase":
            loss_components = self.phase1_dno_loss_comps + self.phase2_dno_loss_comps

        if phase == "one_phase":
            dno_loss_source = DnoLossSource.FROM_CONTACT_PAIRS_ONE_PHASE
        elif phase == "phase1":
            dno_loss_source = None
        elif phase == "phase2":
            dno_loss_source = DnoLossSource.FROM_CONTACT_PAIRS

        dno_cond = DnoCondition(
            decoder_wrapper=self.decoder_wrapper,
            loss_components=loss_components,
            obj_model=obj_model,
            smpl_fk=smpl_fk,
            pred_len=self.cfg.train.pred_len,
            obj_verts_lst=obj_verts_lst,
            obj_faces_lst=obj_faces_lst,
            obj_fk_lst=obj_fk_lst,
            dno_loss_source=dno_loss_source,
        )

        return dno_cond

    def phase0(self):
        print(100 * "=", "phase0")
        features, debug_data = auto_regressive_dno(
            self.cfg,
            self.model,
            max(self.n_frames_lst),
            self.diffusion.ddim_sample_loop,
            noise_opt_conf=None,
            shape=self.motion_shape,
            model_kwargs=self.model_kwargs,
            use_dno=False,
            dno_cond=None,
            seed=self.seed,
            tfms_manager=self.tfms_manager,
            sequence_conditions=None,
        )
        self.debug_data["phase0"] = debug_data
        return self.decoder_wrapper.decode(features)

    def phase_1_and_2_together(self):
        print(100 * "=", "phase_1_and_2_together")
        dno_cond = self.get_dno_cond("one_phase")
        features, debug_data = auto_regressive_dno(
            self.cfg,
            self.model,
            max(self.n_frames_lst),
            self.diffusion.ddim_sample_loop,
            self.dno_options_phase1,
            self.motion_shape,
            self.model_kwargs,
            use_dno=True,
            dno_cond=dno_cond,
            seed=self.seed,
            tfms_manager=self.tfms_manager,
            use_classifier_guidance=self.cfg.use_classifier_guidance,
            diffusion=self.diffusion,
        )
        self.debug_data["final"] = debug_data
        return self.decoder_wrapper.decode(features)

    def phase1(self):
        print(100 * "=", "phase1")
        dno_cond = self.get_dno_cond("phase1")
        features, debug_data = auto_regressive_dno(
            self.cfg,
            self.model,
            max(self.n_frames_lst),
            self.diffusion.ddim_sample_loop,
            noise_opt_conf=self.dno_options_phase1,
            shape=self.motion_shape,
            model_kwargs=self.model_kwargs,
            use_dno=self.cfg.use_dno_object,
            dno_cond=dno_cond,
            seed=self.seed,
            tfms_manager=self.tfms_manager,
            use_classifier_guidance=self.cfg.use_classifier_guidance,
            diffusion=self.diffusion,
        )
        self.debug_data["phase1"] = debug_data
        return self.decoder_wrapper.decode(features)

    def phase2(self, smpldata_for_contact_pairs_lst):
        print(100 * "=", "phase2")
        contact_pairs_lst = smpldata_to_contact_pairs(smpldata_for_contact_pairs_lst)
        dno_cond = self.get_dno_cond("phase2")
        features, debug_data = auto_regressive_dno(
            self.cfg,
            self.model,
            max(self.n_frames_lst),
            self.diffusion.ddim_sample_loop,
            self.dno_options_phase2,
            self.motion_shape,
            self.model_kwargs,
            use_dno=self.cfg.use_dno_body,
            dno_cond=dno_cond,
            seed=self.seed,
            tfms_manager=self.tfms_manager,
            sequence_conditions=contact_pairs_lst,
            use_classifier_guidance=self.cfg.use_classifier_guidance,
            diffusion=self.diffusion,
        )
        self.debug_data["phase2"] = debug_data
        return self.decoder_wrapper.decode(features)


def validate_cfg(cfg: CphoiInferenceConfig):
    if cfg.use_classifier_guidance:
        assert cfg.use_dno_object or cfg.use_dno_body
    if cfg.replace_cps_with_nearest_neighbors:
        assert cfg.use_dno_body


def cphoi_inference(cfg: CphoiInferenceConfig):
    print(cfg)
    validate_cfg(cfg)
    device_ind = cfg.device if cfg.device is not None else get_least_busy_device().index
    # device_ind = -1
    print(f"Using device {device_ind}")
    dist_util.setup_dist(device_ind)
    add_training_args(cfg)
    print(OmegaConf.to_yaml(cfg))
    cfg_train = cfg.train
    normalizer = Normalizer.from_dir(cfg_train.save_dir, device=dist_util.dev())

    ##########
    # For classifier guidance experiments
    ##########
    if cfg.use_classifier_guidance:
        assert cfg.classifier_guidance_lr_factor is not None
        assert cfg.classifier_guidance_n_steps is not None
        classifier_guidance_n_steps = cfg.classifier_guidance_n_steps
        print(
            "⚠️ Warning: Classifier guidance is enabled - adjusting loss coefficients by factor",
            cfg.classifier_guidance_lr_factor,
        )
        for loss_name, loss_coeff in cfg.dno_loss_coefficients_phase1.items():
            cfg.dno_loss_coefficients_phase1[loss_name] = (
                loss_coeff * cfg.classifier_guidance_lr_factor
            )
            print(
                f"Adjusted loss coefficient for {loss_name} from {loss_coeff} to {cfg.dno_loss_coefficients_phase1[loss_name]}"
            )
        for loss_name, loss_coeff in cfg.dno_loss_coefficients_phase2.items():
            cfg.dno_loss_coefficients_phase2[loss_name] = (
                loss_coeff * cfg.classifier_guidance_lr_factor
            )
            print(
                f"Adjusted loss coefficient for {loss_name} from {loss_coeff} to {cfg.dno_loss_coefficients_phase2[loss_name]}"
            )
    else:
        classifier_guidance_n_steps = None
    diffusion = create_gaussian_diffusion(cfg_train, classifier_guidance_n_steps)
    model_identifier = get_model_identifier(cfg.model_path)
    out_dir = cfg.out_dir
    os.makedirs(out_dir, exist_ok=True)
    OmegaConf.save(config=cfg, f=os.path.join(cfg.out_dir, "sampling_cphoi.yaml"))
    if cfg.seed is None:
        global_seed = np.random.randint(1000000)
    else:
        global_seed = cfg.seed
    model = CPHOI(
        pred_len=cfg_train.pred_len,
        context_len=cfg_train.context_len,
        n_feats=normalizer.n_feats,
        num_layers=cfg_train.layers,
        cond_mask_prob=cfg_train.cond_mask_prob,
    )
    load_saved_model(model, cfg.model_path, use_avg=cfg_train.use_ema)
    if cfg.guidance_param != 1.0:
        model = ClassifierFreeSampleModel(model)
    model.freeze_object_encoder()
    model.to(dist_util.dev())
    model.eval()

    dataset_lazy = LazyInferenceDataset(
        cfg.grab_dataset_path,
        cfg_train.pcd_n_points,
        Normalizer.from_dir(cfg_train.save_dir),
        cfg_train.feature_names,
        mixed_dataset=cfg.train.get("mixed_dataset", False),
    )

    sampler: Union[BaseSampler, KitchenSampler] = instantiate(cfg.sampler_config)
    inference_jobs: List[InferenceJob] = sampler.get_inference_jobs()

    feature_processor = dataset_lazy.feature_processor
    smpl_fk = SmplModelsFK.create(
        "smplx", cfg.train.pred_len + 1, device=dist_util.dev()
    )  # +1 for continuity loss

    inference_job_batches = [
        inference_jobs[i : i + cfg.batch_size_gen]
        for i in range(0, len(inference_jobs), cfg.batch_size_gen)
    ]
    total_sample_ind = -1
    collate_fn = partial(
        collate_smplrifke_mdm,
        pred_len=cfg_train.pred_len,
        context_len=cfg_train.context_len,
        enforce_motion_length=cfg_train.context_len + cfg_train.pred_len,
    )
    for batch_idx, job_batch in enumerate(inference_job_batches):

        cur_bs = len(job_batch)
        dps = [dataset_lazy.get_datapoint(job, sampler) for job in job_batch]
        _, model_kwargs = collate_fn(dps)
        model_kwargs_to_device(model_kwargs, dist_util.dev())
        with torch.no_grad():  # calculate object embedding once
            model_kwargs["y"]["obj_emb"] = model.object_encoder(
                model_kwargs["y"]["obj_points"]
            )
        if cfg.guidance_param != 1:
            model_kwargs["y"]["scale"] = (
                torch.ones(cur_bs, device=dist_util.dev()) * cfg.guidance_param
            )

        n_frames_lst = [
            job.n_frames for job in job_batch
        ]  # TODO: use this to cut batch elements with smaller n_frames
        n_frames_max = max(n_frames_lst)

        motion_shape = (cur_bs, dataset_lazy.normalizer.n_feats, 1, n_frames_max)

        tfms_manager = TfmsManager(
            feature_processor, model_kwargs["y"]["tfm_processor"], normalizer
        )

        #########################
        # Sampling
        #########################

        # phase 1 DNO losses
        phase1_dno_loss_comps = get_phase1_loss_comps(
            cfg, sampler, model_kwargs, n_frames_lst
        )

        # phase 2 DNO losses
        phase2_geometries = get_cphoi_geometries_batch(
            model_kwargs, n_simplify_object=cfg.n_simplify_object
        )
        phase2_dno_loss_comps = get_phase2_loss_comps(cfg, sampler, phase2_geometries)

        decoder_wrapper = FeaturesDecoderWrapper(
            feature_processor, normalizer, model_kwargs["y"]["tfm_processor"]
        )
        seed = global_seed + batch_idx
        print(f"Using seed {seed}")
        sampling_result = SamplingFlow(
            cfg,
            diffusion,
            decoder_wrapper,
            model,
            smpl_fk,
            seed,
            model_kwargs,
            n_frames_lst,
            motion_shape,
            tfms_manager,
            dno_opts1=instantiate(cfg.dno_options_phase1),
            dno_opts2=instantiate(cfg.dno_options_phase2),
            phase1_dno_loss_comps=phase1_dno_loss_comps,
            phase2_dno_loss_comps=phase2_dno_loss_comps,
            cphoi_phase2_geometries=phase2_geometries,
            sampler=sampler,
        ).run()

        smpldata_lst_per_phase = sampling_result["samples"]
        # debug_data = sampling_result["debug_data"]

        #########################
        # Iterate over batch elements and save results and artifacts
        #########################
        curr_batch_size = len(job_batch)
        for b in range(curr_batch_size):
            total_sample_ind += 1
            grab_seq_path = model_kwargs["y"]["metadata"][b]["grab_seq_path"]
            text = model_kwargs["y"]["text"][b]
            object_name = model_kwargs["y"]["metadata"][b]["object_name"]
            start_frame = model_kwargs["y"]["metadata"][b]["range"][0]
            save_paths = get_save_paths(
                cfg.out_dir,
                model_identifier,
                total_sample_ind,
                grab_seq_path,
                object_name,
                text,
            )
            result_per_mode = {}
            for phase in smpldata_lst_per_phase.keys():
                smpldata = smpldata_lst_per_phase[phase][b].detach().to("cpu")

                if hasattr(sampler, "transform_smpldata"):
                    sampler: KitchenSamplerPan = sampler
                    smpldata = sampler.transform_smpldata(smpldata)
                #########################
                # Create result container
                #########################
                optim_info_lst_b = (
                    sampling_result["debug_data"].get(phase, {}).get("optim_info_lst")
                )
                optim_info_lst_b = (
                    optim_info_lst_b[b] if optim_info_lst_b is not None else None
                )
                mat_lst_b = sampling_result["debug_data"].get(phase, {}).get("mat_lst")
                mat_lst_b = mat_lst_b[b] if mat_lst_b is not None else None
                result = HoiResult(
                    raw_sample=None,
                    text=text,
                    object_name=object_name,
                    grab_seq_path=grab_seq_path,
                    start_frame=start_frame,
                    translation_obj=None,
                    contact_record=None,
                    optim_info_lst=(
                        optim_info_lst_b if cfg.store_loss_data else None
                    ),  # takes a lot of storage
                    contact_pairs_seq=smpldata_to_contact_pairs([smpldata])[0],
                    smpldata=smpldata,
                    mat_lst=mat_lst_b,
                    seed=seed,
                )
                if phase in cfg.result_phases_to_save:
                    result.save(save_paths[f"{phase}_pickle"])
                result_per_mode[phase] = result

                if optim_info_lst_b is not None and (
                    (cfg.use_dno_object and phase == "phase1")
                    or (cfg.use_dno_body and phase == "phase2")
                    or (cfg.one_phase_experiment and phase == "final")
                ):
                    plot_dno_loss(
                        optim_info_lst_b, save_path=save_paths[f"{phase}_dno_loss_fig"]
                    )
                    print(
                        f"Saved dno loss figures to {save_paths[f'{phase}_dno_loss_fig']}"
                    )

            #########################
            # Visualize all phases
            #########################

            if (
                cfg.anim_save_every_n_gens is not None
                and total_sample_ind % cfg.anim_save_every_n_gens != 0
            ):
                continue

            blend_save_path = save_paths["blend_vis"]
            reset_blender()
            print(f"Visualizing result to {blend_save_path}")
            if "phase0" in cfg.result_phases_to_animate:
                # phase 0
                visualize_hoi_animation(
                    [result_per_mode["phase0"].smpldata],
                    object_path_or_name=object_name,
                    grab_seq_path=grab_seq_path,
                    start_frame=start_frame,
                    translation_obj=None,
                    anim_setup=AnimationSetup.NO_MESH,
                    contact_pairs_seq=result_per_mode["phase0"].contact_pairs_seq,
                    n_simplify_object=cfg.n_simplify_object,
                    colors_dict=COLORS_PER_PHASE["phase0"],
                    mat_lst=result_per_mode["phase0"].mat_lst,
                    unq_id="phase0",
                    reset_blender=False,
                    save=False,
                )

            if "phase1" in cfg.result_phases_to_animate:
                # phase 1
                visualize_hoi_animation(
                    [result_per_mode["phase1"].smpldata],
                    object_path_or_name=object_name,
                    start_frame=start_frame,
                    translation_obj=None,
                    anim_setup=AnimationSetup.NO_MESH,
                    # contact_pairs_seq=result_per_mode["phase1"].contact_pairs_seq,
                    n_simplify_object=cfg.n_simplify_object,
                    colors_dict=COLORS_PER_PHASE["phase1"],
                    mat_lst=result_per_mode["phase1"].mat_lst,
                    unq_id="phase1",
                    reset_blender=False,
                    save=False,
                )

            if "phase2" in cfg.result_phases_to_animate:
                # phase 2
                visualize_hoi_animation(
                    [result_per_mode["phase2"].smpldata],
                    object_path_or_name=object_name,
                    start_frame=start_frame,
                    translation_obj=None,
                    anim_setup=AnimationSetup.NO_MESH,
                    # contact_pairs_seq=result_per_mode["phase2"].contact_pairs_seq,
                    n_simplify_object=cfg.n_simplify_object,
                    colors_dict=COLORS_PER_PHASE["phase2"],
                    mat_lst=result_per_mode["phase2"].mat_lst,
                    unq_id="phase2",
                    reset_blender=False,
                    save=False,
                )

            if "final" in cfg.result_phases_to_animate:
                # final (merged)
                visualize_hoi_animation(
                    [result_per_mode["final"].smpldata],
                    object_path_or_name=object_name,
                    text=text,
                    start_frame=start_frame,
                    translation_obj=None,
                    anim_setup=AnimationSetup[cfg.anim_setup],
                    # contact_pairs_seq=result_per_mode["final"].contact_pairs_seq,
                    n_simplify_object=cfg.n_simplify_object,
                    colors_dict=COLORS_PER_PHASE["final"],
                    table_grab_seq_path=grab_seq_path,
                    mat_lst=result_per_mode["final"].mat_lst,
                    unq_id="final",
                    reset_blender=False,
                    save=False,
                )

            CollectionManager.organize_collections(
                ["phase0", "phase1", "phase2", "final"]
            )
            # hide_collection(["phase0", "phase1", "phase2"])
            bpy.data.texts.new("Metadata").write(
                json.dumps(OmegaConf.to_container(cfg), indent=4)
            )
            save_blender_file(blend_save_path)
            blend_scp_and_run(blend_save_path)


@hydra.main(
    version_base="1.2", config_path="../configs", config_name="sampling_cphoi.yaml"
)
def main(cfg: CphoiInferenceConfig):
    cphoi_inference(cfg)


if __name__ == "__main__":
    main()
