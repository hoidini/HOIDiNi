import os
import time
from datasets.grab.grab_utils import get_all_grab_seq_paths, grab_seq_path_to_seq_id

SLURM_TEMPLATE = """#!/bin/sh
#SBATCH --job-name=JOBNAME
#SBATCH --output=LOG_PATH
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --signal=USR1@120 # how to end job when time's up
#SBATCH --time=1400
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --exclude="n-301,rack-gww-dgx1,n-801,n-804,n-306,n-350,n-805,n-307,n-602"
#SBATCH --constraint="geforce_rtx_3090|tesla_v100|l40s|a6000"
##SBATCH --constraint="l40s"

echo "hostname"
hostname

echo "date"
date

echo "nvidia-smi"
nvidia-smi

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/dcor/roeyron/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/dcor/roeyron/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/dcor/roeyron/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/dcor/roeyron/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<



#################
# CPHOI Inference
#################
cd /home/dcor/roeyron/trumans_utils/src
conda activate mahoi
export PYTHONPATH=$(pwd)
CMD
"""


_TASKS = [
    {
        "blender_path": "/home/dcor/roeyron/trumans_utils/rendering/general/combined_0.blend",
        "collections": [
            "COL_general_results__s10_binoculars_pass_1_0_V0_4",
            "COL_general_results__s10_bowl_pass_1_1_V2_5",
            "COL_general_results__s10_doorknob_lift_1_V2_6",
            "COL_general_results__s10_flute_pass_1_0_V2_7",
            "COL_general_results__s10_gamecontroller_lift_1_V0_0",
            "COL_general_results__s10_spherelarge_inspect_1_0_V1_1",
            "COL_general_results__s10_spherelarge_lift_1_V1_3",
            "COL_general_results__s10_spherelarge_pass_1_0_V1_2",
        ],
    },
    {
        "blender_path": "/home/dcor/roeyron/trumans_utils/rendering/general/combined_1.blend",
        "collections": [
            "COL_1_ours__s10_camera_takepicture_1_v_05_20_13_57_4_1",
            "COL_1_ours__s10_camera_takepicture_1_v_05_20_16_10_0_3",
            "COL_1_ours__s10_camera_takepicture_1_v_05_20_16_10_1_4",
            "COL_1_ours__s10_camera_takepicture_1_v_05_20_16_10_2_5",
            "COL_1_ours__s10_gamecontroller_play_1_V5_0_6",
            "COL_1_ours__s10_wineglass_drink_2_v_05_20_13_57_0_2",
            "COL_general_results__s10_waterbottle_drink_1_0_V2_0",
            "COL_ours__s10_camera_takepicture_1_v_05_20_16_10_1_7",
        ],
    },
    {
        "blender_path": "/home/dcor/roeyron/trumans_utils/rendering/general/combined_2.blend",
        "collections": [
            "COL_1_ours__s10_gamecontroller_play_1_V5_1_0",
            "COL_general_results__s10_airplane_lift_1_V0_5",
            "COL_general_results__s10_spherelarge_inspect_1_1_V1_6",
            "COL_general_results__s10_spheresmall_inspect_1_0_V0_4",
            "COL_general_results__s10_wineglass_drink_1_0_V0_7",
            "COL_general_results__s10_wineglass_drink_2_0_V0_2",
            "COL_general_results__s10_wineglass_drink_2_1_V0_3",
        ],
    },
    {
        "blender_path": "/home/dcor/roeyron/trumans_utils/rendering/general/combined_3.blend",
        "collections": [
            "COL_general_results__s10_mug_drink_1_0_V2_0",
            "COL_general_results__s10_mug_drink_1_1_V2_1",
            "COL_general_results__s10_mug_drink_2_1_V2_5",
            "COL_general_results__s10_piggybank_use_1_0_V2_2",
            "COL_general_results__s10_pyramidlarge_pass_1_0_V2_6",
            "COL_general_results__s10_spherelarge_lift_0_V2_3",
            "COL_general_results__s10_spherelarge_lift_1_V2_4",
            "COL_general_results__s10_spherelarge_pass_1_0_V2_7",
        ],
    },
]


TASKS = []
for fb_dict in _TASKS:
    for collection in fb_dict["collections"]:
        TASKS.append((fb_dict["blender_path"], collection))

import os

dir_path = "/home/dcor/roeyron/trumans_utils/rendering/general/render_output_zoom"
finished = []
for dirname in os.listdir(dir_path):
    n = len(os.listdir(os.path.join(dir_path, dirname)))
    print(dirname.ljust(60), n)
    if n >= 115:
        finished.append(dirname)

TASKS = [e for e in TASKS if e[1] not in finished]

# to_render = ['COL_general_results__s10_spherelarge_pass_1_0_V2_7',
#  'COL_general_results__s10_wineglass_drink_1_0_V0_7',
#  'COL_general_results__s10_wineglass_drink_2_1_V0_3',
#  'COL_general_results__s10_wineglass_drink_2_0_V0_2',
#  'COL_general_results__s10_mug_drink_2_1_V2_5',
#  'COL_general_results__s10_mug_drink_1_1_V2_1',
#  'COL_general_results__s10_mug_drink_1_0_V2_0']

# TASKS = [e for e in TASKS if e[1] in to_render]


def chunked_list(lst, number_of_chunks):
    """Yield successive chunks splitting lst into the given number of chunks."""
    k, m = divmod(len(lst), number_of_chunks)
    start = 0
    for i in range(number_of_chunks):
        end = start + k + (1 if i < m else 0)
        yield lst[start:end]
        start = end


def main():
    n_tasks = min(24, len(TASKS))

    blender_paths_chunks = list(chunked_list(TASKS, number_of_chunks=n_tasks))

    for chunk_id, blender_paths_chunk in enumerate(blender_paths_chunks):
        # start fresh each chunk
        slurm_content = SLURM_TEMPLATE.replace("JOBNAME", f"br_{chunk_id}").replace(
            "LOG_PATH",
            os.path.join(
                "/home/dcor/roeyron/trumans_utils/slurm_logs/render_zoom",
                "%j_slurm.log",
            ),
        )
        cmd_lines = []
        for blender_path, collection_name in blender_paths_chunk:
            cmd_lines.append(
                f"python /home/dcor/roeyron/trumans_utils/rendering/general/render_general_zoom.py --blender_path {blender_path} --collection_name {collection_name}"
            )

        cmd_block = "\n\n\n".join(cmd_lines)
        slurm_content = slurm_content.replace("CMD", cmd_block)

        slurm_file_path = os.path.join(
            "/home/dcor/roeyron/tmp", f"chunk_render{chunk_id}.slurm"
        )
        with open(slurm_file_path, "w") as f:
            f.write(slurm_content)

        print("\n\n\n", "=" * 40)
        print(chunk_id)
        print("=" * 40)
        print(slurm_file_path)
        print(slurm_content)

        print(os.system(f"sbatch {slurm_file_path}"))


if __name__ == "__main__":
    main()
