#!/bin/bash
#SBATCH --job-name=clean
#SBATCH --output=logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
set -euo pipefail
root="$(cd "$(dirname "$0")/.." 2>/dev/null && pwd || true)"
if [ -f "$root/config.sh" ]; then cd "$root"; else cd "${SLURM_SUBMIT_DIR:-.}"; fi
source ./config.sh

case "$VARIANT" in
  0p25x) MT_IN="$DATA_ROOT/augmentation/mt_aug_0p25x" ;;
  base) MT_IN="$DATA_ROOT/processed/mt/mt" ;;
  *) echo "VARIANT must be 0p25x or base (got $VARIANT)"; exit 1 ;;
esac

OUT="$DATA_ROOT/clean/$VARIANT"
echo "clean VARIANT=$VARIANT mt=$MT_IN out=$OUT start=$(date)"

python -u data_cleaning/clean_mt.py \
  --mt "$MT_IN" \
  --mtrev "$DATA_ROOT/processed/mt/mtrev" \
  --dev-dir "$DATA_ROOT/eval" \
  --out "$OUT"

echo "clean VARIANT=$VARIANT end=$(date)"
