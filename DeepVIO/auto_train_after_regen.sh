#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cv_p3
cd ~/Documents/Spring_26/CV/p4/DeepVIO

echo "Waiting for Blender generation to finish..."
while ps aux | grep -v grep | grep "blender.*blender_script" > /dev/null; do
    sleep 30
done
echo "Generation done at $(date)"

# Cleanup bad sequences
python3 << 'PYEOF'
import cv2, os, shutil
import numpy as np
output_dir = 'output'
removed = 0; kept = 0
for split in ['train', 'val', 'test']:
    split_dir = os.path.join(output_dir, split)
    if not os.path.exists(split_dir): continue
    seqs = sorted(os.listdir(split_dir))
    for seq in seqs:
        img_dir = os.path.join(split_dir, seq, 'images')
        if not os.path.exists(img_dir): continue
        blur_scores = []
        for frame in ['00005.png', '00050.png', '00100.png']:
            fpath = os.path.join(img_dir, frame)
            if not os.path.exists(fpath): continue
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                blur_scores.append(cv2.Laplacian(img, cv2.CV_64F).var())
        if not blur_scores: continue
        if np.mean(blur_scores) < 30:
            shutil.rmtree(os.path.join(split_dir, seq))
            removed += 1
        else:
            kept += 1
for split in ['train', 'val', 'test']:
    split_dir = os.path.join(output_dir, split)
    if not os.path.exists(split_dir): continue
    seqs = sorted(os.listdir(split_dir))
    for i, seq in enumerate(seqs, 1):
        new_name = f"seq_{i:03d}"
        if seq != new_name:
            os.rename(os.path.join(split_dir, seq), os.path.join(split_dir, new_name))
print(f"Cleanup: kept={kept}, removed={removed}")
PYEOF

echo ""
echo "=== TRAINING ON NEW DATASET — $(date) ===" | tee newdata_training.log

echo "" | tee -a newdata_training.log
echo "▸ VISUAL MODEL" | tee -a newdata_training.log
python train.py --data output/ --model visual --epochs 50 --attention \
    --batch-size 32 --lr 1e-3 --patience 15 \
    --checkpoint-dir checkpoints/newdata_visual 2>&1 | tee -a newdata_training.log

echo "" | tee -a newdata_training.log
echo "▸ IMU MODEL" | tee -a newdata_training.log
python train.py --data output/ --model imu --epochs 50 --attention \
    --batch-size 64 --lr 1e-3 --patience 15 \
    --checkpoint-dir checkpoints/newdata_imu 2>&1 | tee -a newdata_training.log

echo "" | tee -a newdata_training.log
echo "▸ COMBINED MODEL" | tee -a newdata_training.log
python train.py --data output/ --model combined --epochs 50 --attention \
    --batch-size 32 --lr 1e-3 --patience 15 \
    --checkpoint-dir checkpoints/newdata_combined 2>&1 | tee -a newdata_training.log

echo "" | tee -a newdata_training.log
echo "=== ALL COMPLETE — $(date) ===" | tee -a newdata_training.log
