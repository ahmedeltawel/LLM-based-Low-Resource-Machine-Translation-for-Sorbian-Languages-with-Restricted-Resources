#!/bin/bash
#SBATCH --job-name=significance
#SBATCH --output=logs/%x-%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
set -euo pipefail
root="$(cd "$(dirname "$0")/.." 2>/dev/null && pwd || true)"
if [ -f "$root/config.sh" ]; then cd "$root"; else cd "${SLURM_SUBMIT_DIR:-.}"; fi
source ./config.sh

: "${A:?set A to the result name of system A (a directory under RESULTS_ROOT)}"
: "${B:?set B to the result name of system B (a directory under RESULTS_ROOT)}"
REPS="${REPS:-10000}"
METRIC="${METRIC:-chrf}"
SIG_DIR="$RESULTS_ROOT/significance"

export PYTHONUNBUFFERED=1

mkdir -p "$SIG_DIR"
for PAIR in hsb dsb; do
  echo "deu-$PAIR: $A vs $B"
  python -u sig_eval/paired_bootstrap.py \
    --a "$RESULTS_ROOT/$A/samples_deu-$PAIR.jsonl" \
    --b "$RESULTS_ROOT/$B/samples_deu-$PAIR.jsonl" \
    --reps "$REPS" \
    --seed "$BOOTSTRAP_SEED" \
    --metric "$METRIC" \
    --json "$SIG_DIR/${A}_vs_${B}_${METRIC}_deu-$PAIR.json"
done
