#!/bin/bash
#SBATCH --job-name=average
#SBATCH --output=logs/%x-%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
set -euo pipefail
root="$(cd "$(dirname "$0")/.." 2>/dev/null && pwd || true)"
if [ -f "$root/config.sh" ]; then cd "$root"; else cd "${SLURM_SUBMIT_DIR:-.}"; fi
source ./config.sh

MODEL_A="${MODEL_A:-$RUNS_ROOT/qwen35_4b_0p25x/checkpoints/step_002391/model}"
MODEL_B="${MODEL_B:-$RUNS_ROOT/qwen35_4b_base/checkpoints/step_002391/model}"
AVERAGE_DIR="${AVERAGE_DIR:-$RUNS_ROOT/qwen35_4b_avg/model}"

export PYTHONUNBUFFERED=1

echo "average start=$(date)"
python -u decoding/soup.py \
  --model-a "$MODEL_A" \
  --model-b "$MODEL_B" \
  --out "$AVERAGE_DIR"
echo "average end=$(date)"
