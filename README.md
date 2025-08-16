# HOIDiNi: Human–Object Interaction through Diffusion Noise Optimization



This is the official implementation of the HOIDiNi paper. For more information, please see the [project website](https://hoidini.github.io/) and the [arXiv paper](https://arxiv.org/abs/2506.15625).

HOIDiNi generates realistic 3D human–object interactions conditioned on text prompts, object geometry and scene constraints. It combines diffusion-based motion synthesis with contact-aware diffusion noise optimization to produce visually plausible contacts and smooth, temporally coherent motions.

[![Project Page](https://img.shields.io/badge/Project-Website-1e90ff.svg)](https://hoidini.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-2506.15625-b31b1b.svg)](https://arxiv.org/abs/2506.15625)
[![Video](https://img.shields.io/badge/Video-YouTube-ff0000.svg?logo=youtube&logoColor=white)](https://youtu.be/lN1260WEe7U)
[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![Black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](LICENSE)

 <img src="assets/teaser.png" alt="HOIDiNi teaser" width="800"/>

## 📜 TODO List

- [x] Release the main code
- [x] Release the pretrained model
- [ ] Release evaluation code

<!-- ## 💻 Demo
<a href="https://hoidini.github.io/static/figures/results1.mp4">
  <img src="assets/teaser.png" alt="Demo" width="600"/>
</a> -->

## 📥 Quick Setup

> 📋 **For detailed installation instructions, troubleshooting, and manual setup options, see [INSTALL.md](INSTALL.md)**

### 1. Clone the repository
```bash
git clone https://github.com/roeyron/hoidini.git
cd hoidini
```

### 2. Run setup script (recommended)
```bash
# Run the setup script which creates conda environment and installs all dependencies
bash setup.sh
```

### 4. Download data from Hugging Face
```bash
hf download Roey/hoidini --repo-type dataset --local-dir hoidini_data
```

### 5. Ready to use!
The code will automatically use the downloaded data from `hoidini_data/`. No path configuration needed!

**Directory structure after setup:**
```
hoidini/                          # Main code repository
├── hoidini/                      # Core library
├── scripts/                      # Training and inference scripts  
└── hoidini_data/                 # Downloaded from Hugging Face
    ├── datasets/
    │   ├── GRAB_RETARGETED_compressed/  # Main dataset (2.7GB)
    │   └── MANO_SMPLX_vertex_ids.pkl    # Hand-object mapping
    ├── smpl_models/                     # SMPL/SMPL-X model files
    │   ├── smpl/                        # SMPL body models
    │   ├── smplh/                       # SMPL+H models (with hands)
    │   ├── smplx/                       # SMPL-X models (full body)
    │   └── mano/                        # MANO hand models
    └── models/
        └── cphoi_05011024_c15p100_v0/   # Trained model weights
```




## 🚀 Quick Start

### Run Inference
```bash
# Using the script (recommended)
./scripts/inference.sh

# Or run directly
python hoidini/cphoi/cphoi_inference.py \
    out_dir=outputs/demo \
    --config-name="0_base_config.yaml" \
    model_path=hoidini_data/models/cphoi_05011024_c15p100_v0/model000120000.pt \
    dno_options_phase1.num_opt_steps=200 \
    dno_options_phase2.num_opt_steps=200 \
    sampler_config.n_samples=2 \
    sampler_config.n_frames=100
```

### Train Model
```bash
# Using the script (recommended)
./scripts/train_cphoi.sh

# Or run directly
python hoidini/cphoi/cphoi_train.py \
    save_dir=outputs/train_run \
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
```

The scripts handle all path configuration and conda environment detection automatically.


## 📊 Evaluation (TODO)
- HOIDiNi evaluation utilities (statistical metrics, action recognition) are under `hoidini/eval/`.
- Additional training/evaluation docs will be released.

## 🖼 Visualization

HOIDiNi uses Blender for high-quality 3D visualization of human-object interactions. Visualization is controlled by the `anim_setup` parameter in your configuration.

### Visualization Options

**`NO_MESH`** (Default, Fast)
- Shows skeleton/stick figure animation only
- All frames rendered
- Fast rendering, good for quick previews
- Best for development and debugging

**`MESH_PARTIAL`** (Balanced)
- Shows full 3D mesh visualization
- Renders every 5th frame (for performance)
- Good balance between quality and speed
- Suitable for previewing final results

**`MESH_ALL`** (High Quality, Slow)
- Shows full 3D mesh visualization
- Renders all frames
- Highest quality output
- Best for final results and publications
- ⚠️ Can be very slow for long sequences

### Usage Examples

```bash
# Fast preview (skeleton only)
python hoidini/cphoi/cphoi_inference.py \
    --config-name="0_base_config.yaml" \
    anim_setup=NO_MESH

# Balanced quality
python hoidini/cphoi/cphoi_inference.py \
    --config-name="0_base_config.yaml" \
    anim_setup=MESH_PARTIAL

# High quality (slow)
python hoidini/cphoi/cphoi_inference.py \
    --config-name="0_base_config.yaml" \
    anim_setup=MESH_ALL
```

Visualization outputs are saved as `.blend` files alongside `.pickle` results when `anim_save=true`.

## 🤝 Citation

If you find this repository useful for your work, please consider citing:

```bibtex
@article{ron2025hoidini,
  title={HOIDiNi: Human-Object Interaction through Diffusion Noise Optimization},
  author={Ron, Roey and Tevet, Guy and Sawdayee, Haim and Bermano, Amit H},
  journal={arXiv preprint arXiv:2506.15625},
  year={2025}
}
```

## 🙏 Acknowledgements
This codebase adapts components from [CLoSD](https://github.com/GuyTevet/CLoSD) and [STMC](https://github.com/mathis-petrovich/stmc), and relies on SMPL/SMPL-X ecosystems, PyTorch3D, PyG and related projects. We thank the authors and maintainers of these works.


### Key References

```bibtex
@inproceedings{tevet2025closd,
  title={{CL}o{SD}: Closing the Loop between Simulation and Diffusion for multi-task character control},
  author={Guy Tevet and Sigal Raab and Setareh Cohan and Daniele Reda and Zhengyi Luo and Xue Bin Peng and Amit Haim Bermano and Michiel van de Panne},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=pZISppZSTv}
}

@inproceedings{petrovich2024stmc,
  title={Multi-Track Timeline Control for Text-Driven 3D Human Motion Generation},
  author={Petrovich, Mathis and Litany, Or and Iqbal, Umar and Black, Michael J. and Varol, Gül and Peng, Xue Bin and Rempe, Davis},
  booktitle={CVPR Workshop on Human Motion Generation},
  year={2024}
}

@inproceedings{GRAB:2020,
  title = {{GRAB}: A Dataset of Whole-Body Human Grasping of Objects},
  author = {Taheri, Omid and Ghorbani, Nima and Black, Michael J. and Tzionas, Dimitrios},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year = {2020},
  url = {https://grab.is.tue.mpg.de}
}
```

 ### 3D Assets
- "Kitchen Blender Scene" by Heinzelnisse, available at [BlendSwap](https://www.blendswap.com/blend/5156), licensed under CC-BY-SA.
