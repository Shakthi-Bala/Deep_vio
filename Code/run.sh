#!/bin/bash
#SBATCH --job-name=vision_train
#SBATCH --output=vision_train_%j.out
#SBATCH --error=vision_train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=long
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1


source /home/sbalamurugan/miniconda3/etc/profile.d/conda.sh
conda activate cv

cd /home/sbalamurugan/cv_p4/blender_splats/Code

python train_vision.py \
  --data-dir /home/sbalamurugan/cv_p4/blender_splats/new_data_out \
  --epochs 60 \
  --batch-size 32 \
  --out-dir checkpoints/vision