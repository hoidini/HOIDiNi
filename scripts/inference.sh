#!/bin/bash
conda activate hoidini
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
OUT_DIR=hoidini_results


python hoidini/cphoi/cphoi_inference.py \
    --config-name="0_base_config.yaml" \
    model_path=hoidini_data/models/cphoi_05011024_c15p100_v0/model000120000.pt \
    out_dir=$OUT_DIR \
    dno_options_phase1.num_opt_steps=200 \
    dno_options_phase2.num_opt_steps=200 \
    sampler_config.n_samples=10 \
    sampler_config.n_frames=115 
    # n_simplify_hands=700 \
    # n_simplify_object=700 \

# Uncomment n_simplify_hands and n_simplify_object to run with low GPU memory (will harm the quality)