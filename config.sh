export ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
export MODELS_ROOT="${MODELS_ROOT:-$ROOT/models}"
export RUNS_ROOT="${RUNS_ROOT:-$ROOT/runs}"
export RESULTS_ROOT="${RESULTS_ROOT:-$ROOT/results}"
export LOGS_ROOT="${LOGS_ROOT:-$ROOT/logs}"

export VARIANT="${VARIANT:-0p25x}"
export MODEL="${MODEL:-qwen35_08b}"
export NUM_GPUS="${NUM_GPUS:-4}"

export TOKENIZER_MODEL="${TOKENIZER_MODEL:-$MODELS_ROOT/Qwen3.5-0.8B-Base}"
export TEACHER_MODEL="${TEACHER_MODEL:-$MODELS_ROOT/Qwen2.5-3B-Instruct-hsb-dsb}"

export AUG_PAIRS="${AUG_PAIRS:-159000}"
export AUG_SEED="${AUG_SEED:-17}"
export PACK_SEED="${PACK_SEED:-42}"
export BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-12345}"

mkdir -p "$LOGS_ROOT"
