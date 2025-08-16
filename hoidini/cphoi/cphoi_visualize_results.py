from dataclasses import dataclass
import glob
import json
import os
from typing import Optional

import hydra
from hydra.core.config_store import ConfigStore
from hoidini.blender_utils.visualize_hoi_animation import (
    AnimationSetup,
    visualize_hoi_animation,
)
from hoidini.closd.diffusion_planner.utils import dist_util
from inference_hoi_model import HoiResult


@dataclass
class VisualizationConfig:
    hoi_result_dir: Optional[str] = None
    hoi_result_path: Optional[str] = None
    hoi_results_json_path: Optional[str] = None
    hoi_results_json_key: Optional[str] = None
    out_dir: Optional[str] = None
    max_results: int = 10
    device: int = -1
    animation_setup: str = "NO_MESH"
    n_chunks: int = 1
    chunk_id: int = 0
    overwrite: bool = False

    def __post_init__(self):
        assert (
            sum(
                [
                    e is not None
                    for e in [
                        self.hoi_result_dir,
                        self.hoi_result_path,
                        self.hoi_results_json_path,
                    ]
                ]
            )
            == 1
        ), "Either hoi_result_dir or hoi_result_path or hoi_results_json_path must be provided"
        assert self.out_dir is not None, "out_dir must be provided"


cs = ConfigStore.instance()
cs.store(name="cphoi_visualization_config", node=VisualizationConfig)


def visualize_results(cfg: VisualizationConfig) -> None:
    dist_util.setup_dist(cfg.device)

    # Get all HOI result files
    if cfg.hoi_result_dir is not None:
        hoi_result_paths = glob.glob(os.path.join(cfg.hoi_result_dir, "*.pickle"))
    elif cfg.hoi_result_dir is None:
        hoi_result_paths = [cfg.hoi_result_path]
    elif cfg.hoi_results_json_path is not None:
        with open(cfg.hoi_results_json_path, "r") as f:
            hoi_result_paths = json.load(f)
        if cfg.hoi_results_json_key is not None:
            hoi_result_paths = hoi_result_paths[cfg.hoi_results_json_key]
    else:
        raise ValueError("No hoi result paths provided")

    hoi_result_paths = hoi_result_paths[cfg.chunk_id :: cfg.n_chunks]
    if cfg.out_dir is None and cfg.hoi_result_path is not None:
        save_dir = os.path.dirname(cfg.hoi_result_path)
    else:
        orig_dir_name = os.path.basename(os.path.dirname(hoi_result_paths[0]))
        save_dir = os.path.join(cfg.out_dir, orig_dir_name)
    os.makedirs(save_dir, exist_ok=True)

    # Sort and limit results
    hoi_result_paths = sorted(hoi_result_paths)
    hoi_result_paths = hoi_result_paths[: cfg.max_results]
    print(f"No. of hoi result paths: {len(hoi_result_paths)}")

    anim_setup = getattr(AnimationSetup, cfg.animation_setup)

    print("Saving to: ", save_dir)
    print(
        f"\nrsync -avz --ignore-existing \
    roeyron@c-006.cs.tau.ac.il:{save_dir}/ \
    ~/Downloads/{save_dir}/\n"
    )

    for hoi_result_path in hoi_result_paths:
        save_path = os.path.join(
            save_dir,
            os.path.basename(hoi_result_path).replace(
                ".pickle", f"{anim_setup.name}.blend"
            ),
        )
        if os.path.exists(save_path) and not cfg.overwrite:
            continue

        result = HoiResult.load(hoi_result_path)

        visualize_hoi_animation(
            [result.smpldata],
            object_path_or_name=result.object_name,
            grab_seq_path=result.grab_seq_path,
            text=result.text,
            start_frame=result.start_frame,
            translation_obj=result.translation_obj,
            save_path=save_path,
            contact_record=result.contact_record,
            anim_setup=anim_setup,
            contact_pairs_seq=result.contact_pairs_seq,
        )


@hydra.main(version_base="1.2", config_name="cphoi_visualization_config")
def main(cfg: VisualizationConfig):
    debug = False
    if debug:
        cfg.hoi_result_dir = (
            "/home/dcor/roeyron/trumans_utils/results/cphoi/668838_ours_2phases"
        )
        cfg.out_dir = "/home/dcor/roeyron/tmp/668838_ours_2phases"
    visualize_results(cfg)


if __name__ == "__main__":
    main()


"""
cd /home/dcor/roeyron/trumans_utils/src/
PYTHONPATH=$(pwd)
conda activate mahoi
python blender_utils/src/cphoi/cphoi_visualize_results.py \
    hoi_result_dir=/home/dcor/roeyron/trumans_utils/results/dip/04290104_test_chunks_table_losses \
    out_dir=/home/dcor/roeyron/tmp/04290104_test_chunks_table_losses \
    max_results=10 \
    device=-1 \
    animation_setup=NO_MESH \
    n_chunks=1 \
    chunk_id=0


Download commands:
1. From generated results dir:
rsync -avz --ignore-existing \
    "roeyron@c-006.cs.tau.ac.il:/home/dcor/roeyron/trumans_utils/results/dip/04290104_test_chunks_table_losses/*.blend" \
    "~/Downloads/04290104_test_chunks_table_losses/"

2. From separate generation:
rsync -avz --ignore-existing \
    roeyron@c-006.cs.tau.ac.il:/home/dcor/roeyron/tmp/04290104_test_chunks_table_losses/ \
    ~/Downloads/04290104_test_chunks_table_losses/
"""
