#!/bin/bash
#SBATCH --job-name=tokenize-pack
#SBATCH --output=logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
set -euo pipefail
root="$(cd "$(dirname "$0")/.." 2>/dev/null && pwd || true)"
if [ -f "$root/config.sh" ]; then cd "$root"; else cd "${SLURM_SUBMIT_DIR:-.}"; fi
source ./config.sh

CLEAN="$DATA_ROOT/clean/$VARIANT"
COMMON="$DATA_ROOT/tokenized/common"
MT_TOK="$DATA_ROOT/tokenized/$VARIANT"
FINAL="$DATA_ROOT/final/$VARIANT"

test -d "$CLEAN/mt"
test -d "$CLEAN/mtrev"

echo "tokenize-pack VARIANT=$VARIANT tokenizer=$TOKENIZER_MODEL start=$(date)"

common_ready=1
for g in mono_hsb_sent mono_hsb_doc mono_dsb_sent mono_dsb_doc inst_aya inst_magpie inst_oasst2 inst_flan_v2; do
  [ -f "$COMMON/$g/dataset_info.json" ] || common_ready=0
done
if [ "$common_ready" = 1 ]; then
  echo "common groups already tokenized in $COMMON"
else
  python -u replications/tartunlp_qwen35/04_tokenize_all.py \
    --only common \
    --processed-dir "$DATA_ROOT/processed" \
    --out-common "$COMMON" \
    --tokenizer "$TOKENIZER_MODEL"
fi

rm -rf "$MT_TOK"
python -u replications/tartunlp_qwen35/04_tokenize_all.py \
  --only mt \
  --mt-dir "$CLEAN" \
  --out-mt "$MT_TOK" \
  --tokenizer "$TOKENIZER_MODEL"

rm -rf "$FINAL"
python -u replications/tartunlp_qwen35/05_pack_and_mix.py \
  --common-dir "$COMMON" \
  --mt-dir "$MT_TOK" \
  --out "$FINAL" \
  --seed "$PACK_SEED"

echo "tokenize-pack VARIANT=$VARIANT end=$(date)"
