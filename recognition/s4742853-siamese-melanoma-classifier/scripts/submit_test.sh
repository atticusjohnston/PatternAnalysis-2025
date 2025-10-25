#!/bin/bash
#SBATCH --job-name=melanoma_test
#SBATCH --partition=a100-test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=atticus.johnston@student.uq.edu.au

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU devices: $CUDA_VISIBLE_DEVICES"

cd ~/melanoma
source venv/bin/activate

python train.py \
    --train-csv data/cleaned/train_pairs.csv \
    --train-img-dir data/cleaned/train_images_224 \
    --val-csv data/cleaned/validation_pairs.csv \
    --val-img-dir data/cleaned/validation_images_224 \
    --batch-size 256 \
    --epochs 1 \
    --lr 1e-3 \
    --save-dir models \
    --model custom \
    --log-file logs/train_$SLURM_JOB_ID.log

echo "Job finished at: $(date)"