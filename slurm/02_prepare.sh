#!/bin/bash
#SBATCH --job-name=prepare
#SBATCH --output=logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
set -euo pipefail
root="$(cd "$(dirname "$0")/.." 2>/dev/null && pwd || true)"
if [ -f "$root/config.sh" ]; then cd "$root"; else cd "${SLURM_SUBMIT_DIR:-.}"; fi
source ./config.sh

RAW="$DATA_ROOT/raw"
PROCESSED="$DATA_ROOT/processed"
echo "prepare raw=$RAW processed=$PROCESSED start=$(date)"

python -u replications/tartunlp/01_filter_instructions.py \
  --input-dir "$RAW/instructions" \
  --output-dir "$PROCESSED/instructions"

python -u replications/tartunlp/01b_oasst2_conversations.py \
  --input-dir "$RAW/instructions" \
  --output-dir "$PROCESSED/instructions"

python -u replications/tartunlp/02_format_mt_chat.py \
  --parallel-dir "$RAW/parallel" \
  --out-dir "$PROCESSED/mt"

python -u replications/tartunlp/03_deduplicate.py \
  --raw-dir "$RAW" \
  --out-dir "$PROCESSED"

echo "prepare end=$(date)"
