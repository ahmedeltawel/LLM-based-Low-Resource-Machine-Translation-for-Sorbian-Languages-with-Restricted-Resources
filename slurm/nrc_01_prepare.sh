#!/bin/bash
#SBATCH --job-name=nrc-prepare
#SBATCH --output=logs/%x-%j.out
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
set -euo pipefail
root="$(cd "$(dirname "$0")/.." 2>/dev/null && pwd || true)"
if [ -f "$root/config.sh" ]; then cd "$root"; else cd "${SLURM_SUBMIT_DIR:-.}"; fi
source ./config.sh

NRC_DIR="${NRC_DIR:-$DATA_ROOT/nrc}"

echo "raw data: $DATA_ROOT/raw"
echo "output:   $NRC_DIR"

python -u replications/nrc/01_locate_data.py \
    --raw-dir "$DATA_ROOT/raw" \
    --out "$NRC_DIR/manifest.json"

python -u replications/nrc/02_clean.py \
    --manifest "$NRC_DIR/manifest.json" \
    --out-dir "$NRC_DIR/cleaned" \
    --manifest-out "$NRC_DIR/cleaned_manifest.json"

python -u replications/nrc/03_dedupe.py \
    --manifest-raw "$NRC_DIR/manifest.json" \
    --manifest-clean "$NRC_DIR/cleaned_manifest.json" \
    --out-dir "$NRC_DIR/deduped" \
    --manifest-out "$NRC_DIR/deduped_manifest.json"

python -u replications/nrc/04_format_sft.py \
    --manifest "$NRC_DIR/deduped_manifest.json" \
    --out-dir "$NRC_DIR/sft"

echo "done: $NRC_DIR/sft/train.jsonl"
