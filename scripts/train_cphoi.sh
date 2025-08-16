#!/bin/bash
conda activate hoidini
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
SAVE_DIR=hoidini_training/cphoi_v0


# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

python hoidini/cphoi/cphoi_train.py \
    save_dir=$SAVE_DIR \
    debug_mode=False \
    device=0 \
    batch_size=64 \
    pcd_n_points=512 \
    pcd_augment_rot_z=True \
    pcd_augment_jitter=True \
    pred_len=100 \
    context_len=15 \
    diffusion_steps=8 \
    augment_xy_plane_prob=0.5 \
    mixed_dataset=False \
    overwrite=True