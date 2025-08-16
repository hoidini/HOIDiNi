import os

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

SEQ_IDS = [
    # "s10/waterbottle_drink_1",
    # "s10/binoculars_see_1",
    # "s10/flashlight_on_1",
    # "s10/binoculars_lift",
    # "s10/airplane_lift",
    # "s10/gamecontroller_pass_1",
    # "s10/wineglass_lift",
    # "s10/spheresmall_inspect_1",
    # "s10/gamecontroller_play_1",
    # "s10/spheresmall_pass_1",
    # "s10/wineglass_toast_1",
    # "s10/airplane_fly_1",
    # "s10/camera_browse_1",
    # "s10/bowl_drink_1_Retake",
    # "s10/toruslarge_pass_1",
    # "s10/flute_play_1",
    # "s10/camera_takepicture_1",
    # "s10/wineglass_drink_2",
    # "s10/alarmclock_pass_1",
    # "s10/gamecontroller_lift",
    # "s10/mug_drink_2",
    # "s10/wineglass_drink_1",
    # "s10/cubesmall_inspect_1",
    # "s10/waterbottle_pass_1",
    # "s10/pyramidmedium_inspect_1",
    # "s10/torussmall_inspect_1",
    # "s10/binoculars_pass_1",
    "s10/spherelarge_lift",
    "s10/spherelarge_pass_1",
    # "s10/waterbottle_pour_1",
    # "s10/bowl_pass_1",
    "s10/spherelarge_inspect_1",
    # "s10/torusmedium_inspect_1",
]
SEQ_IDS = [
    grab_seq_path_to_seq_id(e)
    for e in get_all_grab_seq_paths()
    if "s10" in e
    if "s10/" in e
]

N_REPS = 2


def chunked_list(lst, number_of_chunks):
    """Yield successive chunks splitting lst into the given number of chunks."""
    k, m = divmod(len(lst), number_of_chunks)
    start = 0
    for i in range(number_of_chunks):
        end = start + k + (1 if i < m else 0)
        yield lst[start:end]
        start = end


def main():

    out_dir = "/home/dcor/roeyron/trumans_utils/results/general_results"
    VERSION = "V2"
    MESH_ALL = False
    SEED = 11
    n_tasks = 16
    os.makedirs(out_dir, exist_ok=True)
    config = "1_ours_compare.yaml"

    seed_offset = 0

    args = {
        "--config-name": config,
        "anim_setup": "MESH_ALL" if MESH_ALL else "NO_MESH",
    }

    # swap rep and seq loops so you do all SEQ_IDS first, then reps
    seq_ids_w_reps = [(seq_id, rep) for seq_id in SEQ_IDS for rep in range(N_REPS)]
    seq_ids_w_reps_chunks = list(chunked_list(seq_ids_w_reps, number_of_chunks=n_tasks))

    for chunk_id, chunk in enumerate(seq_ids_w_reps_chunks):
        # start fresh each chunk
        slurm_content = SLURM_TEMPLATE.replace("JOBNAME", f"c_{chunk_id}").replace(
            "LOG_PATH", os.path.join(out_dir, "%j_slurm.log")
        )

        cmd_lines = []
        for seq_id, rep in chunk:
            out_dir_seq = os.path.join(
                out_dir, seq_id.replace("/", "_") + f"_{rep}_{VERSION}"
            )
            seed_offset += 1
            args_loop = dict(args)
            args_loop["sampler_config.grab_seq_ids"] = [seq_id]
            args_loop["out_dir"] = out_dir_seq
            args_loop["seed"] = SEED + seed_offset
            # if "additional_args" in d:
            #     for k, v in d["additional_args"].items():
            #         args_loop[k] = v

            cmd_args = " ".join([f'"{k}={v}"' for k, v in args_loop.items()])
            cmd_lines.append(f"python cphoi/cphoi_inference.py {cmd_args}")

        cmd_block = "\n\n\n".join(cmd_lines)
        slurm_content = slurm_content.replace("CMD", cmd_block)

        slurm_file_path = os.path.join(
            "/home/dcor/roeyron/tmp", f"chunk{chunk_id}.slurm"
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
