# P4 — Deep Visual-Inertial Odometry (DeepVIO)

**Group 4** — Shakthibala Sivagami Balamurugan, Aditya Patwardhan, Bradley Kohler  
**Course:** RBE 549 — Computer Vision, Spring 2026, WPI

## Project Overview

DeepVIO: Deep Visual-Inertial Odometry with FiLM Fusion on Synthetic Data.  
We implement three model variants — DeepVO (vision-only), DeepIO (IMU-only), and DeepVIO (fused) — trained on synthetic Blender-generated data with 3D Gaussian Splatting rendering.

## Directory Structure

```
Group4_p4/
├── README.md                          # This file
├── Code/
│   └── Phase 2/
│       ├── models.py                  # V1 model architectures (CNN + LSTM)
│       ├── models_v2.py               # V2 models (ResNet-18 + FiLM fusion)
│       ├── dataset.py                 # PyTorch dataset loader
│       ├── train.py                   # Training script (V1 models)
│       ├── train_twostage.py          # Two-stage FiLM training script
│       ├── evaluate.py                # Evaluation + trajectory plotting
│       ├── eval_all_models.py         # Batch evaluation of all 5 models
│       ├── vio_model.py               # Model wrapper
│       ├── oystersim_imuutlils.py     # OysterSim IMU noise simulation
│       ├── blender_script.py          # Blender data generation pipeline
│       ├── gen_v2_trajectories.py     # V2 trajectory visualization
│       ├── gen_report_plots.py        # Generate all report figures
│       ├── visualize_trajectory.py    # 3D trajectory plotting
│       ├── cleanup_bad_sequences.py   # Dataset quality filtering
│       ├── hard_test.py               # Cross-trajectory generalization test
│       ├── quick_viz.py               # Quick visualization helper
│       ├── train_v2.sh                # V2 training pipeline script
│       ├── auto_train_after_regen.sh  # Auto-train after data regeneration
│       ├── overnight_experiments.sh   # Full experiment suite
│       ├── datasetgen.blend           # Blender scene file for data generation
│       ├── report.tex                 # LaTeX report source
│       ├── checkpoints/               # Trained model weights
│       │   ├── v2_resnet_visual_best.pt   # Best DeepVO (ResNet-18)
│       │   ├── v2_twostage_film_best.pt   # Best DeepVIO (FiLM two-stage)
│       │   └── v2_film_combined_best.pt   # Best DeepVIO (FiLM one-shot)
│       ├── visualizations/            # All experiment result plots
│       └── figures/                   # Report figures
├── Output.mp4                         # Output demonstration video
├── VideoPresentation.mp4              # Video presentation
└── Report.pdf                         # Final report (IEEE format)
```

## Phase 2: Deep VIO

### Requirements

```bash
pip install torch torchvision numpy matplotlib scipy
```

- Python 3.8+
- PyTorch 2.0+
- Blender 4.5 (for data generation only)

### How to Run

#### 1. Generate Synthetic Data (requires Blender)

```bash
# Run Blender headless to generate training sequences
blender --background datasetgen.blend --python blender_script.py
```

**Note:** Data is NOT included in this submission. Generate it first, or contact the authors.

#### 2. Train Models

```bash
# Train all V1 models (DeepVO CNN, DeepIO LSTM, DeepVIO concat)
python train.py --mode visual --epochs 50
python train.py --mode imu --epochs 20
python train.py --mode combined --epochs 30

# Train V2 models (ResNet-18 + FiLM)
bash train_v2.sh

# Train two-stage FiLM (best model)
python train_twostage.py --visual_ckpt checkpoints/v2_resnet_visual_best.pt \
                         --epochs_frozen 10 --epochs_finetune 50
```

Default values are provided for all command-line arguments.

#### 3. Evaluate

```bash
# Evaluate a single model
python evaluate.py --model_type visual_v2 \
                   --checkpoint checkpoints/v2_resnet_visual_best.pt \
                   --data_dir ./output

# Evaluate all models and generate comparison plots
python eval_all_models.py --data_dir ./output

# Generate all report figures
python gen_report_plots.py
```

#### 4. Quick Inference with Pretrained Weights

```bash
# Run inference using provided checkpoints
python evaluate.py --model_type film_twostage \
                   --checkpoint checkpoints/v2_twostage_film_best.pt \
                   --data_dir ./output
```

### Key Results

| Model | Test Loss | ATE (m) |
|-------|-----------|---------|
| DeepVO (CNN) | 0.3017 | 6.6 |
| DeepIO (LSTM) | 0.5430 | 10.6 |
| DeepVIO (Concat) | 0.5129 | 17.1 |
| DeepVO (ResNet-18) | 0.0295 | **1.5** |
| **DeepVIO FiLM (two-stage)** | **0.0237** | 3.5 |

### Architecture

- **Visual Encoder:** ResNet-18 (ImageNet pretrained), modified for 6-channel input
- **IMU Encoder:** 2-layer Bidirectional LSTM with LayerNorm
- **Fusion:** Feature-wise Linear Modulation (FiLM) with soft gating
- **Training:** Two-stage strategy (visual pretrain → frozen fusion → joint finetune)
- **Rendering:** 3D Gaussian Splatting via gsplat for photorealistic synthetic data


# Splats video is not included on the slides due to space constraint

- The link for the splat are in this link: ``` bash  https://drive.google.com/drive/folders/16WqCDxDA3hWMpzP5yoHXiH_Ceoxm3lwd ```
