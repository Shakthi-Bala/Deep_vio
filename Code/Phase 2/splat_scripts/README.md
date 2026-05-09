# Deep_vio Code

This folder contains a synthetic homography-based dataset generator, three pose regression models, and training / evaluation scripts for deep VIO.

## Setup

```bash
cd /home/alien/Deep_vio/Code
python -m pip install -r requirements.txt
```

## Generate synthetic sequences

```bash
python dataset/generate_synthetic_data.py --output-dir data/synthetic --num-sequences 20
```

## Train a model

Use any supported data root, for example `data/synthetic`, `../seq_001`, `../output`, or `../output_grass`.

Vision-only:

```bash
python train_vision.py --data-dir ../seq_001 --epochs 40 --batch-size 32 --out-dir checkpoints/vision
```

IMU-only:

```bash
python train_imu.py --data-dir ../seq_001 --epochs 40 --batch-size 32 --out-dir checkpoints/imu
```

Visual-inertial:

```bash
python train_vio.py --data-dir ../seq_001 --epochs 40 --batch-size 32 --out-dir checkpoints/vio
```

## Evaluate

```bash
python evaluate.py --data-dir data/synthetic --checkpoint checkpoints/vio/best.pth --mode vio
```
