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
#SBATCH --exclude="n-301,rack-gww-dgx1,n-801,n-804,n-306,n-350"
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

blender_paths = [
    # '/home/dcor/roeyron/trumans_utils/user_study/ours/s10_binoculars_pass_1.blend',
    # '/home/dcor/roeyron/trumans_utils/user_study/ours/s10_pyramidlarge_pass_1.blend',
    # '/home/dcor/roeyron/trumans_utils/user_study/ours/s10_mug_drink_1.blend',
    # '/home/dcor/roeyron/trumans_utils/user_study/ours/s10_hammer_use_1.blend',
    # '/home/dcor/roeyron/trumans_utils/user_study/ours/s10_banana_pass_1.blend',
    # '/home/dcor/roeyron/trumans_utils/user_study/ours/s10_scissors_pass_1.blend',
    # '/home/dcor/roeyron/trumans_utils/user_study/ours/s10_stapler_staple_1.blend',
    # '/home/dcor/roeyron/trumans_utils/user_study/ours/s10_stanfordbunny_pass_1.blend',
    # '/home/dcor/roeyron/trumans_utils/user_study/ours/s10_train_play_1.blend',
    # '/home/dcor/roeyron/trumans_utils/user_study/ours/s10_lightbulb_pass_1.blend',
    # '/home/dcor/roeyron/trumans_utils/user_study/classifier_guidance/s10_pyramidlarge_pass_1.blend',
    # '/home/dcor/roeyron/trumans_utils/user_study/classifier_guidance/s10_mug_drink_1.blend',
    # '/home/dcor/roeyron/trumans_utils/user_study/classifier_guidance/s10_hammer_use_1.blend',
    # '/home/dcor/roeyron/trumans_utils/user_study/classifier_guidance/s10_banana_pass_1.blend',
    # '/home/dcor/roeyron/trumans_utils/user_study/classifier_guidance/s10_scissors_pass_1.blend',
    # '/home/dcor/roeyron/trumans_utils/user_study/classifier_guidance/s10_stapler_staple_1.blend',
    # '/home/dcor/roeyron/trumans_utils/user_study/classifier_guidance/s10_stanfordbunny_pass_1.blend',
    # '/home/dcor/roeyron/trumans_utils/user_study/classifier_guidance/s10_train_play_1.blend',
    # '/home/dcor/roeyron/trumans_utils/user_study/classifier_guidance/s10_lightbulb_pass_1.blend',
    # '/home/dcor/roeyron/trumans_utils/user_study/classifier_guidance/s10_binoculars_pass_1.blend',
    # '/home/dcor/roeyron/trumans_utils/user_study/inference_only/s10_pyramidlarge_pass_1.blend',
    # '/home/dcor/roeyron/trumans_utils/user_study/inference_only/s10_mug_drink_1.blend',
    # '/home/dcor/roeyron/trumans_utils/user_study/inference_only/s10_hammer_use_1.blend',
    # '/home/dcor/roeyron/trumans_utils/user_study/inference_only/s10_banana_pass_1.blend',
    # '/home/dcor/roeyron/trumans_utils/user_study/inference_only/s10_scissors_pass_1.blend',
    # '/home/dcor/roeyron/trumans_utils/user_study/inference_only/s10_stapler_staple_1.blend',
    # '/home/dcor/roeyron/trumans_utils/user_study/inference_only/s10_stanfordbunny_pass_1.blend',
    # '/home/dcor/roeyron/trumans_utils/user_study/inference_only/s10_train_play_1.blend',
    # '/home/dcor/roeyron/trumans_utils/user_study/inference_only/s10_lightbulb_pass_1.blend',
    # '/home/dcor/roeyron/trumans_utils/user_study/inference_only/s10_binoculars_pass_1.blend',
    # '/home/dcor/roeyron/trumans_utils/user_study/imos/s10_binoculars_pass_1.blend',
    # '/home/dcor/roeyron/trumans_utils/user_study/imos/s10_pyramidlarge_pass_1.blend',
    # '/home/dcor/roeyron/trumans_utils/user_study/imos/s10_mug_drink_1.blend',
    # '/home/dcor/roeyron/trumans_utils/user_study/imos/s10_hammer_use_1.blend',
    # '/home/dcor/roeyron/trumans_utils/user_study/imos/s10_banana_pass_1.blend',
    # '/home/dcor/roeyron/trumans_utils/user_study/imos/s10_scissors_pass_1.blend',
    # '/home/dcor/roeyron/trumans_utils/user_study/imos/s10_stapler_staple_1.blend',
    # '/home/dcor/roeyron/trumans_utils/user_study/imos/s10_stanfordbunny_pass_1.blend',
    # '/home/dcor/roeyron/trumans_utils/user_study/imos/s10_train_play_1.blend',
    # '/home/dcor/roeyron/trumans_utils/user_study/imos/s10_lightbulb_pass_1.blend',
    "/home/dcor/roeyron/trumans_utils/user_study/nn/s10_binoculars_pass_1.blend",
    "/home/dcor/roeyron/trumans_utils/user_study/nn/s10_pyramidlarge_pass_1.blend",
    "/home/dcor/roeyron/trumans_utils/user_study/nn/s10_mug_drink_1.blend",
    "/home/dcor/roeyron/trumans_utils/user_study/nn/s10_hammer_use_1.blend",
    "/home/dcor/roeyron/trumans_utils/user_study/nn/s10_banana_pass_1.blend",
    "/home/dcor/roeyron/trumans_utils/user_study/nn/s10_scissors_pass_1.blend",
    "/home/dcor/roeyron/trumans_utils/user_study/nn/s10_stapler_staple_1.blend",
    "/home/dcor/roeyron/trumans_utils/user_study/nn/s10_stanfordbunny_pass_1.blend",
    # '/home/dcor/roeyron/trumans_utils/user_study/nn/s10_train_play_1.blend',
    # '/home/dcor/roeyron/trumans_utils/user_study/nn/s10_lightbulb_pass_1.blend'
]


def chunked_list(lst, number_of_chunks):
    """Yield successive chunks splitting lst into the given number of chunks."""
    k, m = divmod(len(lst), number_of_chunks)
    start = 0
    for i in range(number_of_chunks):
        end = start + k + (1 if i < m else 0)
        yield lst[start:end]
        start = end


def main():
    n_tasks = min(20, len(blender_paths))

    blender_paths_chunks = list(chunked_list(blender_paths, number_of_chunks=n_tasks))

    for chunk_id, blender_paths_chunk in enumerate(blender_paths_chunks):
        # start fresh each chunk
        slurm_content = SLURM_TEMPLATE.replace("JOBNAME", f"br_{chunk_id}").replace(
            "LOG_PATH",
            os.path.join(
                "/home/dcor/roeyron/trumans_utils/slurm_logs/user_study", "%j_slurm.log"
            ),
        )
        cmd_lines = []
        for blender_path in blender_paths_chunk:
            cmd_lines.append(
                f"python /home/dcor/roeyron/trumans_utils/user_study/user_study_render.py --blender_path {blender_path}"
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
        time.sleep(1)


if __name__ == "__main__":
    main()
