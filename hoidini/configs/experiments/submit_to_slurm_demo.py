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

EXPERIMENTS = {
    "stove_pan": [
        # "s8/fryingpan_cook_2",
        "s8/fryingpan_cook_2",
        "s1/fryingpan_cook_2",
        "s8/fryingpan_cook_2",
        # "s10/fryingpan_cook_2",
        "s4/fryingpan_cook_2",
        "s4/fryingpan_cook_2",
        "s4/fryingpan_cook_2",
        "s4/fryingpan_cook_2",
        "s7/fryingpan_cook_1",
        "s8/fryingpan_cook_3",
        # "s10/fryingpan_cook_2",
        "s4/fryingpan_cook_1",
        "s8/fryingpan_offhand_1",
        "s1/fryingpan_cook_1",
        # "s8/fryingpan_cook_1"
    ],
    "island_mug": [
        "s8/mug_drink_2",
        "s7/mug_drink_2",
        "s5/mug_toast_1",
        "s5/mug_drink_1",
        "s8/mug_offhand_1",
        "s10/mug_toast_1",
        "s9/mug_drink_2",
        "s3/mug_drink_2",
        "s10/mug_drink_2",
        "s10/mug_drink_1",
    ],
    # "island_bowl": [
    #     "s1/bowl_drink_2",
    #     "s3/bowl_drink_2",
    #     "s3/bowl_drink_1",
    #     "s10/bowl_drink_2",
    #     "s4/bowl_drink_1",
    #     "s1/bowl_drink_1",
    #     "s8/bowl_drink_1",
    #     "s8/bowl_drink_2",
    #     "s7/bowl_drink_1",
    #     "s10/bowl_drink_1_Retake"
    # ],
    "island_cup": [
        "s3/cup_drink_1",
        "s5/cup_drink_2",
        "s7/cup_drink_1",
        "s4/cup_drink_1",
        "s6/cup_pour_1",
        # "s5/cup_pour_1",
        "s1/cup_pour_1",
        "s3/cup_drink_2",
        "s6/cup_drink_2",
        "s6/cup_drink_1",
    ],
    # "island_knife": [
    #     "s10/knife_chop_1",
    #     "s7/knife_chop_1",
    #     "s6/knife_peel_1",
    #     "s8/knife_peel_1",
    #     "s8/knife_chop_1",
    #     "s10/knife_peel_1"
    # ],
    # "island_wineglass": [
    #     "s3/wineglass_drink_1",
    #     "s6/wineglass_drink_1",
    #     "s9/wineglass_drink_1",
    #     "s8/wineglass_drink_1",
    #     "s9/wineglass_drink_2",
    #     "s10/wineglass_drink_1",
    #     "s8/wineglass_toast_1",
    #     "s4/wineglass_drink_1",
    #     "s5/wineglass_drink_1",
    #     "s7/wineglass_toast_1"
    # ],
    # "island_phone": [
    #     "s8/phone_call_1",
    #     "s1/phone_call_1",
    #     "s7/phone_call_1",
    #     "s6/phone_call_1",
    #     "s1/phone_offhand_1",
    #     "s2/phone_call_1",
    #     "s4/phone_call_1",
    #     "s10/phone_call_1",
    #     "s3/phone_call_1",
    #     "s8/phone_offhand_1"
    # ],
    "lader_bulb": [
        "s3/lightbulb_screw_1",
        "s10/lightbulb_screw_1",
        "s9/lightbulb_screw_1",
        "s9/lightbulb_screw_1",
        "s9/lightbulb_screw_1",
        "s9/lightbulb_screw_1",
        "s9/lightbulb_screw_1",
        "s9/lightbulb_screw_1",
        "s9/lightbulb_screw_1",
    ],
}


def main():
    debug = False
    if debug:
        out_dir = "/home/dcor/roeyron/trumans_utils/results/cphoi/kitchen_debug"
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = "/home/dcor/roeyron/trumans_utils/results/kitchen"
    config_name = "sampling_cphoi_kitchen.yaml"
    version = "v10"
    for exp_name, seq_ids in EXPERIMENTS.items():
        out_dir_exp = os.path.join(out_dir, exp_name)
        os.makedirs(out_dir_exp, exist_ok=True)
        slurm_content = SLURM_TEMPLATE
        slurm_content = slurm_content.replace("JOBNAME", exp_name)
        slurm_content = slurm_content.replace(
            "LOG_PATH", os.path.join(out_dir_exp, "%j_slurm.log")
        )
        slurm_file_path = os.path.join("/home/dcor/roeyron/tmp/", f"{exp_name}.slurm")

        args = {
            "--config-name": config_name,
            "sampler_config.surface_name": exp_name.split("_")[0],
            "dno_options_phase1.num_opt_steps": 350,
            "dno_options_phase2.num_opt_steps": 350,
            "dno_options_phase1.diff_penalty_scale": 0.0001,
        }

        cmd_lines = []
        for i, seq_id in enumerate(seq_ids):
            out_dir_seq = os.path.join(
                out_dir, exp_name, seq_id.replace("/", "_") + f"_{version}_{i}"
            )
            os.makedirs(out_dir_seq, exist_ok=True)
            args_loop = dict(args)  # copy the base args
            args_loop["sampler_config.grab_seq_id"] = seq_id
            args_loop["out_dir"] = out_dir_seq
            cmd_args = " ".join([f'"{k}={v}"' for k, v in args_loop.items()])
            cmd_lines.append(f"python cphoi/cphoi_inference.py {cmd_args}")
        cmd_block = "\n".join(cmd_lines)
        slurm_content = slurm_content.replace("CMD", cmd_block)

        with open(slurm_file_path, "w") as f:
            f.write(slurm_content)

        print(3 * "\n", 40 * "=")
        print(exp_name)
        print(40 * "=")
        print(slurm_file_path)
        print(slurm_content)

        print(os.system(f"sbatch {slurm_file_path}"))


if __name__ == "__main__":
    main()
