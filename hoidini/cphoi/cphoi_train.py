# This code is based on https://github.com/openai/guided-diffusion
from dataclasses import dataclass
from typing import Optional
import hydra
from hydra.core.config_store import ConfigStore
import os
import json
import shutil
import torch
from omegaconf import OmegaConf
from hoidini.closd.diffusion_planner.train.training_loop import TrainLoop
from hoidini.closd.diffusion_planner.utils import dist_util
from hoidini.closd.diffusion_planner.utils.fixseed import fixseed
from hoidini.closd.diffusion_planner.utils.model_util import create_gaussian_diffusion
from hoidini.cphoi.cphoi_dataset import get_cphoi_dataloader
from hoidini.cphoi.cphoi_model import CPHOI
from hoidini.general_utils import get_least_busy_device
from hoidini.resource_paths import GRAB_DATA_PATH
from hoidini.closd.diffusion_planner.train.train_platforms import (
    ClearmlPlatform,
    TensorboardPlatform,
    NoPlatform,
    WandBPlatform,
)

os.environ["WANDB__SERVICE_WAIT"] = "300"


PLATFORM_REGISTRY = {
    "WandBPlatform": WandBPlatform,
    "TensorboardPlatform": TensorboardPlatform,
    "ClearmlPlatform": ClearmlPlatform,
    "NoPlatform": NoPlatform,
}


@dataclass
class CphoiTrainConfig:
    # training parameters
    debug_mode: bool = True
    save_dir: str = "/home/dcor/roeyron/trumans_utils/src/Experiments/cphoi_train"
    grab_dataset_path: str = GRAB_DATA_PATH
    save_interval: int = 20000
    num_steps: int = 200000
    lr: float = 0.0001
    adam_beta2: float = 0.999
    lr_anneal_steps: int = 0
    log_interval: int = 1000
    overwrite: bool = False
    seed: int = 42
    device: int = 0
    train_platform_type: str = "WandBPlatform"
    wandb_project: str = "cphoi"
    resume_checkpoint: str = ""
    weight_decay: float = 0.0
    use_ema: bool = True
    avg_model_beta: float = 0.9999
    unconstrained: bool = False

    # data parameters
    dataset: str = ""  # required by training loop
    batch_size: int = 64
    data_load_lim: Optional[int] = None
    pcd_n_points: int = 512
    pcd_augment_rot_z: bool = True
    pcd_augment_jitter: bool = True
    fps: int = 20
    feature_names: Optional[str] = "cphoi"

    # diffusion parameters
    diffusion_steps: int = 40
    noise_schedule: str = "cosine"
    sigma_small: bool = True

    # model parameters
    context_len: int = 15
    pred_len: int = 100
    layers: int = 8

    # Generation parameters
    guidance_param: float = 7.5
    autoregressive_include_prefix: bool = False

    zero_prefix_prob: float = 0.0
    mask_frames: bool = True
    cond_mask_prob: Optional[float] = 0.1

    lambda_fc: float = 0.0
    lambda_rcxyz: float = 0.0
    lambda_target_loc: float = 0.0
    lambda_vel: float = 0.0

    # ############################################################
    # Non relevant parameters required by the training loop --->
    # ############################################################
    eval_during_training: bool = False
    keyframe_cond_type: str = ""
    spatial_condition: Optional[str] = None
    multi_target_cond: bool = False
    gen_during_training: bool = False

    mixed_dataset: bool = False
    augment_xy_plane_prob: float = 0.0
    mask_obj_related_features: bool = False


cs = ConfigStore.instance()
cs.store(name="cphoi_train_cfg", node=CphoiTrainConfig)


def train(args: CphoiTrainConfig):
    if args.seed is not None:
        fixseed(args.seed)

    TrainPlatform = PLATFORM_REGISTRY[args.train_platform_type]
    train_platform = TrainPlatform(args.save_dir, project=args.wandb_project)
    train_platform.report_args(dict(args), name="Args")

    if args.save_dir is None:
        raise FileNotFoundError("save_dir was not specified.")
    elif os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError("save_dir [{}] already exists.".format(args.save_dir))
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args_path = os.path.join(args.save_dir, "args.json")
    with open(args_path, "w") as fw:
        json.dump(
            OmegaConf.to_container(args, resolve=True), fw, indent=4, sort_keys=True
        )

    dist_util.setup_dist(args.device)

    print("creating data loader...")
    data = get_cphoi_dataloader(
        grab_dataset_path=args.grab_dataset_path,
        batch_size=args.batch_size,
        experiment_dir=args.save_dir,
        is_training=True,
        context_len=args.context_len,
        pred_len=args.pred_len,
        lim=args.data_load_lim,
        n_points=args.pcd_n_points,
        fps=args.fps,
        pcd_augment_rot_z=args.pcd_augment_rot_z,
        pcd_augment_jitter=args.pcd_augment_jitter,
        feature_names=args.feature_names,
        grab_split="train",
        mixed_dataset=args.mixed_dataset,
        augment_xy_plane_prob=args.augment_xy_plane_prob,
        mask_obj_related_features=args.mask_obj_related_features,
    )

    print("creating model and diffusion...")
    diffusion = create_gaussian_diffusion(args)
    model = CPHOI(
        pred_len=args.pred_len,
        context_len=args.context_len,
        n_feats=data.dataset.n_feats,
        num_layers=args.layers,
        cond_mask_prob=args.cond_mask_prob,
    )
    model.to(dist_util.dev())

    print(
        "Total params: %.2fM"
        % (sum(p.numel() for p in model.parameters_wo_clip()) / 1e6)
    )
    print("Training...")
    TrainLoop(args, train_platform, model, diffusion, data).run_loop()
    train_platform.close()


@hydra.main(version_base=None, config_name="cphoi_train_cfg")
def main(cfg: CphoiTrainConfig):
    dist_util.setup_dist(cfg.device)
    assert torch.cuda.is_available()
    if cfg.debug_mode:
        device = get_least_busy_device()
        cfg.device = device.index
        dist_util.setup_dist(device.index)
        print("Debug mode ⚠️")
        cfg.train_platform_type = "NoPlatform"
        cfg.save_dir = "/home/dcor/roeyron/trumans_utils/src/Experiments/cphoi_debug"
        cfg.mixed_dataset = True
        cfg.mask_obj_related_features = True
        cfg.augment_xy_plane_prob = 0.1
        cfg.data_load_lim = 16
        cfg.batch_size = 16
        cfg.save_interval = 1000
        cfg.context_len = 5
        cfg.pred_len = 10
        # cfg.augment_xy_plane = True
        # cfg.motion_suffix_prob = 0.2
        # cfg.motion_suffix_max_len = 4

        if os.path.exists(cfg.save_dir):
            shutil.rmtree(cfg.save_dir)
    train(cfg)


if __name__ == "__main__":
    main()


"""
python src/object_contact_prediction/train_cpdm.py \
    batch_size=64 \
    save_dir=/home/dcor/roeyron/trumans_utils/src/Experiments/cpdm_v0 \
"""
