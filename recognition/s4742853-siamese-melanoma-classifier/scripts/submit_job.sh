#!/bin/bash
#SBATCH --job-name=melanoma
#SBATCH --partition=a100
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=atticus.johnston@student.uq.edu.au

MODE=${1:-train}
BATCH_SIZE=${2:-512}
EPOCHS=${3:-10}
LR=${4:-1e-5}
MODEL=${5:-pretrained}
MODEL_TIMESTAMP=${6:-}
K=${7:-10}

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Mode: $MODE"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LR"
echo "Model: $MODEL"
echo "Model timestamp: $MODEL_TIMESTAMP"
echo "K: $K"

cd ~/PatternAnalysis-2025/recognition/s4742853-siamese-melanoma-classifier
source venv/bin/activate

ARGS="--mode $MODE \
    --train-csv data/cleaned/train_pairs.csv \
    --train-img-dir data/cleaned/train_images_224 \
    --val-csv data/cleaned/validation.csv \
    --val-img-dir data/cleaned/validation_images_224 \
    --test-csv data/cleaned/test.csv \
    --test-img-dir data/cleaned/test_images_224 \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --k $K \
    --save-dir models \
    --results-dir results \
    --model $MODEL \
    --log-file logs/${MODE}_$SLURM_JOB_ID.log"

if [ -n "$MODEL_TIMESTAMP" ]; then
    ARGS="$ARGS --model-timestamp $MODEL_TIMESTAMP"
fi

python train.py $ARGS

echo "Job finished at: $(date)"