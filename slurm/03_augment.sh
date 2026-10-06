#!/bin/bash
#SBATCH --job-name=augment
#SBATCH --output=logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
set -euo pipefail
root="$(cd "$(dirname "$0")/.." 2>/dev/null && pwd || true)"
if [ -f "$root/config.sh" ]; then cd "$root"; else cd "${SLURM_SUBMIT_DIR:-.}"; fi
source ./config.sh

NUM_SHARDS="${NUM_SHARDS:-4}"
AUG_DIR="$DATA_ROOT/augmentation"

python -u data_augmentation/01_download_filter.py \
    --download-dir "$AUG_DIR" \
    --out "$AUG_DIR/filtered_de.txt"

python -u data_augmentation/02_dedup_sample.py \
    --filtered "$AUG_DIR/filtered_de.txt" \
    --processed-dir "$DATA_ROOT/processed" \
    --dev-test-dir "$DATA_ROOT/raw/llms-limited-resources2025" \
    --out "$AUG_DIR/sample_750k_de.txt" \
    --seed "$AUG_SEED"

mkdir -p "$AUG_DIR/translated"
rm -f "$AUG_DIR"/translated/shard*.csv
IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((NUM_GPUS - 1)))}"
PIDS=()
trap 'kill $(jobs -p) 2>/dev/null || true' EXIT
for ((i = 0; i < NUM_SHARDS; i++)); do
    CUDA_VISIBLE_DEVICES="${GPU_IDS[$((i % ${#GPU_IDS[@]}))]}" \
    python -u data_augmentation/03_translate.py \
        --model "$TEACHER_MODEL" \
        --src "$AUG_DIR/sample_750k_de.txt" \
        --out-dir "$AUG_DIR/translated" \
        --shard "$i" \
        --num-shards "$NUM_SHARDS" &
    PIDS+=("$!")
done
for pid in "${PIDS[@]}"; do
    wait "$pid"
done
trap - EXIT

python -u data_augmentation/04_postfilter.py \
    --in-dir "$AUG_DIR/translated" \
    --out "$AUG_DIR/clean.csv"

python -u data_augmentation/05_build_aug_dataset.py \
    --clean "$AUG_DIR/clean.csv" \
    --mt "$DATA_ROOT/processed/mt/mt" \
    --out "$AUG_DIR/mt_aug_0p25x" \
    --seed "$AUG_SEED"
