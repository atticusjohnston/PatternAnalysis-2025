#!/bin/bash
#SBATCH --job-name=melanoma_train
#SBATCH --partition=a100
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=atticus.johnston@student.uq.edu.au

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

cd ~/PatternAnalysis-2025/recognition/s4742853-siamese-melanoma-classifier
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
    --model pretrained \
    --log-file logs/train_$SLURM_JOB_ID.log

echo "Job finished at: $(date)"