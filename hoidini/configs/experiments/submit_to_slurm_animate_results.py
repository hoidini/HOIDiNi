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

    VERSION = f"v_{datetime.now().strftime('%m_%d_%H_%M')}"
    NO_MESH = True
    n_tasks_per_experiment = 4
    os.makedirs(out_dir, exist_ok=True)
    for exp_name, d in EXPERIMENTS.items():
        # config_name = d["config"]
        out_dir_exp = os.path.join(out_dir, exp_name)
        # os.makedirs(out_dir_exp, exist_ok=True)

        args = {
            "--config-name": "0_base_config_comparison.yaml",
        }

        # swap rep and seq loops so you do all SEQ_IDS first, then reps
        seq_ids_w_reps = [(seq_id, rep) for seq_id in SEQ_IDS for rep in range(N_REPS)]
        seq_ids_w_reps_chunks = list(
            chunked_list(seq_ids_w_reps, number_of_chunks=n_tasks_per_experiment)
        )

        for i, chunk in enumerate(seq_ids_w_reps_chunks):
            # start fresh each chunk
            slurm_content = SLURM_TEMPLATE.replace("JOBNAME", exp_name).replace(
                "LOG_PATH", os.path.join(out_dir_exp, "%j_slurm.log")
            )

            cmd_lines = []
            for seq_id, rep in chunk:
                out_dir_seq = os.path.join(
                    out_dir, exp_name, seq_id.replace("/", "_") + f"_{VERSION}_{rep}"
                )
                os.makedirs(out_dir_seq, exist_ok=True)

                args_loop = dict(args)
                args_loop["sampler_config.grab_seq_ids"] = [seq_id]
                args_loop["out_dir"] = out_dir_seq
                args_loop["anim_setup"] = "NO_MESH" if NO_MESH else "MESH_AL"

                cmd_args = " ".join([f'"{k}={v}"' for k, v in args_loop.items()])
                cmd_lines.append(f"python cphoi/cphoi_inference.py {cmd_args}")

            cmd_block = "\n\n\n".join(cmd_lines)
            slurm_content = slurm_content.replace("CMD", cmd_block)

            slurm_file_path = os.path.join(
                "/home/dcor/roeyron/tmp", f"{exp_name}_chunk{i}.slurm"
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
