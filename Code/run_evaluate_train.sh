#!/bin/bash
#SBATCH --job-name=evaluate_train
#SBATCH --output=evaluate_train_%j.out
#SBATCH --error=evaluate_train_%j.err
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

python evaluate_train.py \
  --data-dir /home/sbalamurugan/cv_p4/blender_splats/new_data_out
