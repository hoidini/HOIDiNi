# Installation Guide

This guide provides detailed installation instructions for HOIDiNi: Human–Object Interaction through Diffusion Noise Optimization.

## Prerequisites

- **Operating System**: Linux (tested on Ubuntu)
- **Python**: 3.11 (recommended)
- **CUDA**: 12.1 or compatible
- **Conda**: Miniconda or Anaconda installed

## 1. Clone the Repository

First, clone the HOIDiNi repository:

```bash
git clone https://github.com/roey1rg/hoidini.git
cd hoidini
```

## 2. Quick Installation (Recommended)

The easiest way to install HOIDiNi is using the provided setup script:

```bash
bash setup.sh
```

This script will automatically handle all dependencies and create the proper environment.

## 2. Manual Installation (Alternative)

If you prefer to install manually or need to customize the installation, follow these steps:

### 1. Create Conda Environment

```bash
conda create -n hoidini python=3.11 -y
conda activate hoidini
```

### 2. Update pip

```bash
pip install --upgrade pip
```

### 3. Install PyTorch with CUDA Support

```bash
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

### 4. Install PyTorch3D Dependencies

```bash
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
```

### 5. Install PyTorch3D

Install the specific PyTorch3D version compatible with PyTorch 2.5.1:

```bash
pip install --extra-index-url https://miropsota.github.io/torch_packages_builder pytorch3d==0.7.8+pt2.5.1cu121
```

### 6. Install PyTorch Cluster

```bash
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.5+124.html
```

### 7. Install PyTorch Geometric

```bash
conda install pytorch-geometric -c pyg -y
```

### 8. Install Git-based Dependencies

```bash
pip install git+https://github.com/otaheri/chamfer_distance
pip install git+https://github.com/otaheri/bps_torch
pip install git+https://github.com/openai/CLIP.git
```

### 9. Install Remaining Dependencies

```bash
pip install -r requirements.txt
```