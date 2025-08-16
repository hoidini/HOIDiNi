import os
import shutil

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
#SBATCH --exclude="n-301,rack-gww-dgx1,n-801,n-804,n-306,n-350,n-307"
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
python cphoi/cphoi_inference.py \\
    "--config-name=CONFIGNAME" \\
    "out_dir=OUTDIR" \\
    EXTRA_ARGS
"""


EXPERIMENTS = {
    "1_ours": {
        "config": "1_ours.yaml",
    },
    "1b_inference_only": {
        "config": "1b_inference_only.yaml",
    },
    "2_phase1_inference_phase2_dno": {
        "config": "2_phase1_inference_phase2_dno.yaml",
    },
    "2b_phase1_inference_phase2_dno_no_table_losses": {
        "config": "2b_phase1_inference_phase2_dno_no_table_losses.yaml",
    },
    "3_classifier_guidance": {
        "config": "3_classifier_guidance.yaml",
        "variants": [
            "classifier_guidance_lr_factor=1.0",
            "classifier_guidance_lr_factor=10.0",
            "classifier_guidance_lr_factor=20.0",
            "classifier_guidance_lr_factor=50.0",
            "classifier_guidance_lr_factor=100.0",
        ],
    },
    "4_contact_vs_penetration": {
        "config": "4_contact_vs_penetration.yaml",
    },
    "4b_contact_vs_penetration": {
        "config": "4b_contact_vs_penetration.yaml",
    },
    "5_nearest_neighbors_instead_of_cps": {
        "config": "5_nearest_neighbors_instead_of_cps.yaml",
    },
    "7_single_phase": {
        "config": "7_single_phase.yaml",
    },
}


def main():
    debug = False
    if debug:
        out_dir = "/home/dcor/roeyron/trumans_utils/results/cphoi/debug"
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = "/home/dcor/roeyron/trumans_utils/results/Results_June_10"

    for exp_name, exp_config in EXPERIMENTS.items():
        variants = exp_config.get("variants", [None])
        for i, variant in enumerate(variants):
            exp_name_i = f"{exp_name}_{i}"
            out_dir_exp = os.path.join(out_dir, exp_name_i)

            slurm_content = SLURM_TEMPLATE
            slurm_content = slurm_content.replace("JOBNAME", f"{exp_name_i}")
            slurm_content = slurm_content.replace(
                "LOG_PATH", os.path.join(out_dir_exp, "%j_slurm.log")
            )
            slurm_content = slurm_content.replace("CONFIGNAME", exp_config["config"])
            slurm_content = slurm_content.replace("OUTDIR", out_dir_exp)
            slurm_file_path = os.path.join(
                "/home/dcor/roeyron/tmp/", f"{exp_name}.slurm"
            )
            extra_args = [variant]
            if debug:
                extra_args.append("sampler_config.n_samples=3")
                extra_args.append("dno_options_phase1.num_opt_steps=3")
                extra_args.append("dno_options_phase2.num_opt_steps=3")
            extra_args = [e for e in extra_args if e is not None]
            extra_args = [f'"{e}"' for e in extra_args]
            extra_args = " ".join(extra_args)
            slurm_content = slurm_content.replace("EXTRA_ARGS", extra_args)
            with open(slurm_file_path, "w") as f:
                f.write(slurm_content)

            print(3 * "\n", 40 * "=")
            print(exp_name, variant)
            print(40 * "=")
            print(slurm_file_path)
            print(slurm_content)

            print(os.system(f"sbatch {slurm_file_path}"))


if __name__ == "__main__":
    main()
