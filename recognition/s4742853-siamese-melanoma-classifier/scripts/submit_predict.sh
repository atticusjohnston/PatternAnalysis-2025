#!/bin/bash
#SBATCH --job-name=melanoma-predict
#SBATCH --partition=a100-test
#SBATCH --output=logs/predict_%j.out
#SBATCH --error=logs/predict_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

MODEL_PATH=${1:-}
IMAGE_PATH=${2:-}

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Model: $MODEL_PATH"
echo "Image: $IMAGE_PATH"

cd ~/PatternAnalysis-2025/recognition/s4742853-siamese-melanoma-classifier
source venv/bin/activate

python predict.py \
    --model-path $MODEL_PATH \
    --image-path $IMAGE_PATH \
    --ref-csv data/cleaned/validation.csv \
    --ref-img-dir data/cleaned/validation_images_224 \
    --k 10 \
    --model-type pretrained

echo "Job finished at: $(date)"