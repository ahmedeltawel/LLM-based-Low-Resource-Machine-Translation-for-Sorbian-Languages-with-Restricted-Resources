#!/bin/bash
#SBATCH --job-name=stage-model
#SBATCH --output=logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
set -euo pipefail
root="$(cd "$(dirname "$0")/.." 2>/dev/null && pwd || true)"
if [ -f "$root/config.sh" ]; then cd "$root"; else cd "${SLURM_SUBMIT_DIR:-.}"; fi
source ./config.sh

export HF_HUB_DISABLE_TELEMETRY=1

if [ "$#" -eq 0 ]; then
  set -- qwen35_08b qwen35_2b qwen35_4b teacher nrc_base
fi

for name in "$@"; do
  (
    unset MODEL_ID MODEL_REVISION
    source "model_backbone_experiments/configs/$name.env"
    : "${MODEL_ID:?MODEL_ID is not set in configs/$name.env}"
    : "${MODEL_REVISION:?MODEL_REVISION is not set in configs/$name.env}"

    echo "staging $MODEL_ID@$MODEL_REVISION -> $MODEL_PATH"
    mkdir -p "$MODEL_PATH"

    python -c 'from huggingface_hub import snapshot_download; import os; snapshot_download(repo_id=os.environ["MODEL_ID"], revision=os.environ["MODEL_REVISION"], local_dir=os.environ["MODEL_PATH"])'

    test -s "$MODEL_PATH/config.json"
    test -s "$MODEL_PATH/tokenizer.json"
    weights=$(find "$MODEL_PATH" -maxdepth 1 -name '*.safetensors' -type f -size +1M -print -quit)
    test -n "$weights"

    echo "staged $MODEL_ID@$MODEL_REVISION -> $MODEL_PATH"
    du -sh "$MODEL_PATH"
  )
done
