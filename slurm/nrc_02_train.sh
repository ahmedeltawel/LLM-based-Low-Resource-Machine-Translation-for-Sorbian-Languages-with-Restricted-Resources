#!/bin/bash
#SBATCH --job-name=nrc-train
#SBATCH --output=logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
set -euo pipefail
root="$(cd "$(dirname "$0")/.." 2>/dev/null && pwd || true)"
if [ -f "$root/config.sh" ]; then cd "$root"; else cd "${SLURM_SUBMIT_DIR:-.}"; fi
source ./config.sh

NRC_DATASET_DIR="${NRC_DATASET_DIR:-$DATA_ROOT/nrc/sft}"
NRC_OUTPUT_DIR="${NRC_OUTPUT_DIR:-$RUNS_ROOT/nrc_qwen25_15b}"

if (( 32 % NUM_GPUS != 0 )); then
    echo "NUM_GPUS must divide 32 to keep the effective batch at 8 x accumulation x GPUs = 256 (got $NUM_GPUS)" >&2
    exit 1
fi
GRAD_ACCUM=$(( 32 / NUM_GPUS ))

test -s "$NRC_DATASET_DIR/train.jsonl"
test -s "$NRC_DATASET_DIR/dataset_info.json"

export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=6
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1

echo "dataset_dir: $NRC_DATASET_DIR"
echo "output_dir:  $NRC_OUTPUT_DIR"
echo "gpus:        $NUM_GPUS"
echo "grad_accum:  $GRAD_ACCUM"

FORCE_TORCHRUN=1 NNODES=1 NODE_RANK=0 NPROC_PER_NODE="$NUM_GPUS" \
    llamafactory-cli train replications/nrc/configs/train.yaml \
    model_name_or_path="$MODELS_ROOT/Qwen2.5-1.5B-Instruct" \
    dataset_dir="$NRC_DATASET_DIR" \
    output_dir="$NRC_OUTPUT_DIR" \
    gradient_accumulation_steps="$GRAD_ACCUM"

test -s "$NRC_OUTPUT_DIR/config.json"
echo "final model: $NRC_OUTPUT_DIR"
