#!/bin/bash

# Source conda and create environment
# source /home/dcor/roeyron/miniconda3/etc/profile.d/conda.sh
conda create -n "hoidini" python=3.11 -y
conda activate hoidini
echo $(which python)
# Update pip to the latest version
pip install --upgrade pip

# pytorch
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y

# pytorch3d
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub

# Select the correct pytorch3d version for the current pytorch version:
# https://github.com/facebookresearch/pytorch3d/discussions/1752
pip install --extra-index-url https://miropsota.github.io/torch_packages_builder pytorch3d==0.7.8+pt2.5.1cu121

pip install torch-cluster -f https://data.pyg.org/whl/torch-2.5+124.html


# pytorch-geometric
conda install pytorch-geometric -c pyg -y

# pytorch-lightning
pip install git+https://github.com/otaheri/chamfer_distance
pip install git+https://github.com/otaheri/bps_torch


pip install -r requirements.txt
