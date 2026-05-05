#!/bin/bash
#SBATCH --job-name=evaluate
#SBATCH --output=evaluate_%j.out
#SBATCH --error=evaluate_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=long
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1


source ~/.bashrc
conda activate cv

cd /home/sbalamurugan/cv_p4/blender_splats/Code

python evaluate.py \
  --mode vision \
  --checkpoint checkpoints/vision/best.pth \
  --data-dir /home/sbalamurugan/cv_p4/blender_splats/new_data_out \
  --split val \
  --output-plot checkpoints/vision/trajectory.png

python evaluate.py \
  --mode imu \
  --checkpoint checkpoints/imu/best.pth \
  --data-dir /home/sbalamurugan/cv_p4/blender_splats/new_data_out \
  --split val \
  --output-plot checkpoints/imu/trajectory.png

python evaluate.py \
  --mode vio \
  --checkpoint checkpoints/vio/best.pth \
  --data-dir /home/sbalamurugan/cv_p4/blender_splats/new_data_out \
  --split val \
  --output-plot checkpoints/vio/trajectory.png
