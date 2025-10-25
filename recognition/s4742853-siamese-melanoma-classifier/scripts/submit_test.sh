#!/bin/bash
#SBATCH --job-name=melanoma_test
#SBATCH --partition=a100-test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err

cd ~/melanoma
source venv/bin/activate

python train.py \
    --train-csv data/cleaned/train_pairs.csv \
    --train-img-dir data/cleaned/train_images_224 \
    --val-csv data/cleaned/validation_pairs.csv \
    --val-img-dir data/cleaned/validation_images_224 \
    --batch-size 128 \
    --epochs 1 \
    --lr 1e-4 \
    --save-dir models \
    --model custom