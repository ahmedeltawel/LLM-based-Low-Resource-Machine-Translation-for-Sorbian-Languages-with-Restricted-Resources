#!/bin/bash
#SBATCH --job-name=download
#SBATCH --output=logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
set -euo pipefail
root="$(cd "$(dirname "$0")/.." 2>/dev/null && pwd || true)"
if [ -f "$root/config.sh" ]; then cd "$root"; else cd "${SLURM_SUBMIT_DIR:-.}"; fi
source ./config.sh

RAW="$DATA_ROOT/raw"
echo "download raw=$RAW start=$(date)"

bash replications/tartunlp/download_wmt_data.sh "$RAW"

python -u replications/tartunlp/download_hf_data.py --out "$RAW"

python -u replications/tartunlp/prepare_dev_sets.py \
  --repo-dir "$RAW/llms-limited-resources2025" \
  --out-dir "$DATA_ROOT/eval"

bash model_backbone_experiments/stage_model.sh

echo "download end=$(date)"
