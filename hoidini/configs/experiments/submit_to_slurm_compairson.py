import os
import shutil
from datetime import datetime


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
    # "s10/wineglass_lift",
    # "s10/bowl_drink_2",
    # "s10/gamecontroller_play_1",
    # "s10/wineglass_toast_1",
    # "s10/phone_call_1",
    # "s10/bowl_lift",
    # "s10/flute_pass_1",
    # "s10/camera_browse_1",
    "s10/camera_takepicture_1",
    # "s10/wineglass_drink_2",
    "s10/waterbottle_drink_1",
    # "s10/eyeglasses_wear_1",
]

N_REPS = 3

EXPERIMENTS = {
    "1_ours": {
        "config": "1_ours_compare.yaml",
    },
    "2_inference_only": {
        "config": "1b_inference_compare.yaml",
    },
    "3_nearest_neighbors_instead_of_cps": {
        "config": "5_nearest_neighbors_instead_of_cps_compare.yaml",
    },
    "4_phase1_inference_phase2_dno": {
        "config": "2_phase1_inference_phase2_dno_compare.yaml",
    },
    "5_classifier_guidance": {
        "config": "3_classifier_guidance_compare.yaml",
        "additional_args": {"classifier_guidance_lr_factor": "100.0"},
    },
    # "6_IMoS": {
    #     "config": "IMoS.yaml",
    # },
}


def chunked_list(lst, number_of_chunks):
    """Yield successive chunks splitting lst into the given number of chunks."""
    k, m = divmod(len(lst), number_of_chunks)
    start = 0
    for i in range(number_of_chunks):
        end = start + k + (1 if i < m else 0)
        yield lst[start:end]
        start = end


def main():
    debug = False
    if debug:
        out_dir = (
            "/home/dcor/roeyron/trumans_utils/results/cphoi/comparison_figure_debug"
        )
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = "/home/dcor/roeyron/trumans_utils/results/comparison_figure"

    # VERSION = f"v_{datetime.now().strftime('%m_%d_%H_%M')}"
    VERSION = "V6"
    MESH_ALL = True
    SEED = 2316
    n_tasks_per_experiment = 2
    os.makedirs(out_dir, exist_ok=True)

    for exp_name, d in EXPERIMENTS.items():

        seed_offset = 0
        # config_name = d["config"]
        out_dir_exp = os.path.join(out_dir, exp_name)

        args = {
            "--config-name": d["config"],
            "anim_setup": "MESH_ALL" if MESH_ALL else "NO_MESH",
        }

        # swap rep and seq loops so you do all SEQ_IDS first, then reps
        seq_ids_w_reps = [(seq_id, rep) for seq_id in SEQ_IDS for rep in range(N_REPS)]
        seq_ids_w_reps_chunks = list(
            chunked_list(seq_ids_w_reps, number_of_chunks=n_tasks_per_experiment)
        )

        for chunk_id, chunk in enumerate(seq_ids_w_reps_chunks):
            # start fresh each chunk
            slurm_content = SLURM_TEMPLATE.replace("JOBNAME", exp_name).replace(
                "LOG_PATH", os.path.join(out_dir_exp, "%j_slurm.log")
            )

            cmd_lines = []
            for seq_id, rep in chunk:
                out_dir_seq = os.path.join(
                    out_dir, exp_name, seq_id.replace("/", "_") + f"_{VERSION}_{rep}"
                )
                seed_offset += 1
                args_loop = dict(args)
                args_loop["sampler_config.grab_seq_ids"] = [seq_id]
                args_loop["out_dir"] = out_dir_seq
                args_loop["seed"] = SEED + seed_offset
                if "additional_args" in d:
                    for k, v in d["additional_args"].items():
                        args_loop[k] = v

                cmd_args = " ".join([f'"{k}={v}"' for k, v in args_loop.items()])
                cmd_lines.append(f"python cphoi/cphoi_inference.py {cmd_args}")

            cmd_block = "\n\n\n".join(cmd_lines)
            slurm_content = slurm_content.replace("CMD", cmd_block)

            slurm_file_path = os.path.join(
                "/home/dcor/roeyron/tmp", f"{exp_name}_chunk{chunk_id}.slurm"
            )
            with open(slurm_file_path, "w") as f:
                f.write(slurm_content)

            print("\n\n\n", "=" * 40)
            print(exp_name)
            print("=" * 40)
            print(slurm_file_path)
            print(slurm_content)

            print(os.system(f"sbatch {slurm_file_path}"))


if __name__ == "__main__":
    main()
