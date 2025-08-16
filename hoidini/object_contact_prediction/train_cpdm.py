# This code is based on https://github.com/openai/guided-diffusion
from dataclasses import dataclass
from typing import Optional
import hydra
from hydra.core.config_store import ConfigStore
import os
import json
import shutil

from omegaconf import OmegaConf
from hoidini.closd.diffusion_planner.train.training_loop import TrainLoop
from hoidini.closd.diffusion_planner.utils import dist_util
from hoidini.closd.diffusion_planner.utils.fixseed import fixseed
from hoidini.closd.diffusion_planner.utils.model_util import create_gaussian_diffusion
from general_utils import get_least_busy_device
from object_contact_prediction.cpdm_dataset import get_contact_pairs_dataloader
from object_contact_prediction.cpdm import CPDM
from resource_paths import GRAB_DATA_PATH

from hoidini.closd.diffusion_planner.train.train_platforms import (
    ClearmlPlatform,
    TensorboardPlatform,
    NoPlatform,
    WandBPlatform,
)

PLATFORM_REGISTRY = {
    "WandBPlatform": WandBPlatform,
    "TensorboardPlatform": TensorboardPlatform,
    "ClearmlPlatform": ClearmlPlatform,
    "NoPlatform": NoPlatform,
}


@dataclass
class Config:
    # training parameters
    debug_mode: bool = True
    save_dir: str = "/home/dcor/roeyron/trumans_utils/src/Experiments/cpdm_debug"
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
    wandb_project: str = "cpdm"
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
    feature_names: Optional[str] = None

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
    # motion_suffix_prob: float = 0.3
    # motion_suffix_max_len_range: tuple[int, int] = (15, 16)
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

    # ## Consider removing --->
    # arch: str = "trans_dec"
    # autoregressive: bool = True
    # autoregressive_init: str = "data"
    # cuda: bool = True
    # data_dir: str = ""
    # dataset_data: str = "grabhoi"
    # emb_before_mask: bool = False
    # emb_policy: str = "concat"
    # emb_trans_dec: bool = False
    # eval_batch_size: int = 32
    # eval_num_samples: int = 1000
    # eval_rep_times: int = 3
    # eval_split: str = "test"
    # external_mode: bool = False
    # features_string: str = "hoi_body_hands"

    # gen_guidance_param: float = 7.5
    # gen_num_repetitions: int = 2
    # gen_num_samples: int = 3
    # hml_type: str = None
    # keyframe_cond_prob: float = 0.5
    # latent_dim: int = 512
    # multi_encoder_type: str = "single"
    # pos_embed_max_len: int = 5000
    # sampling_mode: str = "none"
    # use_inpainting: bool = False
    # use_obj_encoder: bool = True
    # use_recon_guidance: bool = False

    # ############################################################
    # <--- Non relevant parameters required by training loop
    # ############################################################


cs = ConfigStore.instance()
cs.store(name="cpdm_cfg", node=Config)


def train(args: Config):
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
    data = get_contact_pairs_dataloader(
        grab_dataset_path=args.grab_dataset_path,
        batch_size=args.batch_size,
        experiment_dir=args.save_dir,
        is_training=True,
        context_len=args.context_len,
        pred_len=args.pred_len,
        lim=args.data_load_lim,
        n_points=args.pcd_n_points,
        fps=args.get("fps", 20),
        pcd_augment_rot_z=args.get("pcd_augment_rot_z", False),
        pcd_augment_jitter=args.get("pcd_augment_jitter", False),
        feature_names=args.get("feature_names", None),
        grab_split="train",
    )

    print("creating model and diffusion...")
    diffusion = create_gaussian_diffusion(args)
    model = CPDM(
        pred_len=args.pred_len,
        context_len=args.context_len,
        n_feats=data.dataset.n_feats,
        num_layers=args.layers,
        cond_mask_prob=args.cond_mask_prob,
    )
    model.to(args.device)

    print(
        "Total params: %.2fM"
        % (sum(p.numel() for p in model.parameters_wo_clip()) / 1e6)
    )
    print("Training...")
    TrainLoop(args, train_platform, model, diffusion, data).run_loop()
    train_platform.close()


@hydra.main(version_base=None, config_name="cpdm_cfg")
def main(cfg: Config):
    device = get_least_busy_device()
    dist_util.setup_dist(device.index)
    if cfg.debug_mode:
        print("Debug mode ⚠️")
        cfg.train_platform_type = "NoPlatform"
        cfg.save_dir = "/home/dcor/roeyron/trumans_utils/src/Experiments/cpdm_debug"
        cfg.data_load_lim = 16
        cfg.batch_size = 16
        cfg.save_interval = 1000
        cfg.context_len = 5
        cfg.pred_len = 10
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
