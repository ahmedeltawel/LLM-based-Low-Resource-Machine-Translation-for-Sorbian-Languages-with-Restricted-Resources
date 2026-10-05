#!/bin/bash
#SBATCH --job-name=evaluate
#SBATCH --output=logs/%x-%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
set -euo pipefail
root="$(cd "$(dirname "$0")/.." 2>/dev/null && pwd || true)"
if [ -f "$root/config.sh" ]; then cd "$root"; else cd "${SLURM_SUBMIT_DIR:-.}"; fi
source ./config.sh

PROMPT_FORMAT="${PROMPT_FORMAT:-qwen}"
case "$PROMPT_FORMAT" in
  qwen)
    MODEL_DIR="${MODEL_DIR:-$RUNS_ROOT/${MODEL}_${VARIANT}/checkpoints/step_002391/model}"
    NAME="${NAME:-${MODEL}_${VARIANT}}"
    ;;
  nrc)
    MODEL_DIR="${MODEL_DIR:-$RUNS_ROOT/nrc_qwen25_15b}"
    NAME="${NAME:-nrc_qwen25_15b}"
    ;;
  *) echo "PROMPT_FORMAT must be qwen or nrc (got $PROMPT_FORMAT)"; exit 1 ;;
esac
BATCH_SIZE="${BATCH_SIZE:-8}"
OUT_DIR="$RESULTS_ROOT/$NAME"

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

mkdir -p "$OUT_DIR"
echo "evaluate model=$MODEL_DIR format=$PROMPT_FORMAT out=$OUT_DIR start=$(date)"

for PAIR in hsb dsb; do
  python -u decoding/evaluate.py \
    --model "$MODEL_DIR" \
    --pair "$PAIR" \
    --dev-csv "$DATA_ROOT/eval/dev.de-$PAIR.csv" \
    --out-dir "$OUT_DIR" \
    --batch-size "$BATCH_SIZE" \
    --prompt-format "$PROMPT_FORMAT"
done

test -s "$OUT_DIR/metrics_deu-hsb.json"
test -s "$OUT_DIR/metrics_deu-dsb.json"
echo "evaluate end=$(date)"
