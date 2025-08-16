# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on images.
"""

import os
import json
from hoidini.closd.diffusion_planner.utils.fixseed import fixseed
from hoidini.closd.diffusion_planner.utils.parser_util import train_args
from hoidini.closd.diffusion_planner.utils import dist_util
from hoidini.closd.diffusion_planner.train.training_loop import TrainLoop
from hoidini.closd.diffusion_planner.data_loaders.get_data import get_dataset_loader
from hoidini.closd.diffusion_planner.utils.model_util import create_model_and_diffusion
from hoidini.closd.diffusion_planner.train.train_platforms import (
    ClearmlPlatform,
    TensorboardPlatform,
    NoPlatform,
    WandBPlatform,
)  # required for the eval operation

PLATFORM_REGISTRY = {
    "WandBPlatform": WandBPlatform,
    "TensorboardPlatform": TensorboardPlatform,
    "ClearmlPlatform": ClearmlPlatform,
    "NoPlatform": NoPlatform,
}


def main():
    args = train_args()
    fixseed(args.seed)
    TrainPlatform = PLATFORM_REGISTRY[args.train_platform_type]
    train_platform = TrainPlatform(args.save_dir)
    train_platform.report_args(args, name="Args")

    if args.save_dir is None:
        raise FileNotFoundError("save_dir was not specified.")
    elif os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError("save_dir [{}] already exists.".format(args.save_dir))
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args_path = os.path.join(args.save_dir, "args.json")
    with open(args_path, "w") as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    dist_util.setup_dist(args.device)

    print("creating data loader...")

    data = get_dataset_loader(
        name=args.dataset,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        fixed_len=args.context_len + args.pred_len,
        pred_len=args.pred_len,
        hml_type=args.hml_type,
        device=dist_util.dev(),
        experiment_dir=args.save_dir,
        is_training=True,
        dataset_data=args.dataset_data,
        seed=args.seed,
        features_string=args.features_string,
        data_load_lim=args.data_load_lim,
        grab_split="train",
        condition_input_feature=args.condition_input_feature,
    )

    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data.dataset.n_feats)
    model.to(dist_util.dev())

    print(
        "Total params: %.2fM"
        % (sum(p.numel() for p in model.parameters_wo_clip()) / 1e6)
    )
    print("Training...")
    TrainLoop(args, train_platform, model, diffusion, data).run_loop()
    train_platform.close()


if __name__ == "__main__":
    main()
