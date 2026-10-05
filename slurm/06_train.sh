#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=128G
#SBATCH --time=20:00:00
set -euo pipefail
root="$(cd "$(dirname "$0")/.." 2>/dev/null && pwd || true)"
if [ -f "$root/config.sh" ]; then cd "$root"; else cd "${SLURM_SUBMIT_DIR:-.}"; fi
source ./config.sh

source "model_backbone_experiments/configs/$MODEL.env"

export DATA_PATH="$DATA_ROOT/final/$VARIANT/training_data"
export OUTPUT_DIR="$RUNS_ROOT/${MODEL}_${VARIANT}/checkpoints"
export LOG_DIR="$RUNS_ROOT/${MODEL}_${VARIANT}/logs"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

test -d "$DATA_PATH"

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=6
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800

echo "train MODEL=$MODEL VARIANT=$VARIANT start=$(date)"
echo "MODEL_PATH=$MODEL_PATH"
echo "DATA_PATH=$DATA_PATH"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "PER_DEVICE_BATCH_SIZE=$PER_DEVICE_BATCH_SIZE GRADIENT_CHECKPOINTING=$GRADIENT_CHECKPOINTING NUM_GPUS=$NUM_GPUS"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

torchrun --standalone --nproc_per_node="$NUM_GPUS" replications/tartunlp_qwen35/06_train.py

latest=$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -type d -name 'step_*' -print | sort | tail -n 1)
test -n "$latest"
test -s "$latest/model/config.json"
echo "final checkpoint: $latest/model"
echo "train MODEL=$MODEL VARIANT=$VARIANT end=$(date)"
