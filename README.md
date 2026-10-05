# LLM-based Low-Resource Machine Translation for Sorbian Languages with Restricted Resources

Code for the Master's thesis by Ahmed Eltawel (Technical University of Munich, 2026).
It trains decoder-only language models to translate German into Upper Sorbian (`hsb`)
and Lower Sorbian (`dsb`) in the setting of the WMT25 shared task on LLMs with limited
resources for Slavic languages.

## Models

| Model | Description |
|-------|-------------|
| [ahmedeltawel/Qwen3.5-4B-hsb-dsb](https://huggingface.co/ahmedeltawel/Qwen3.5-4B-hsb-dsb) | Weight average of the two Qwen3.5-4B runs (step 7) |
| [ahmedeltawel/Qwen3.5-0.8B-hsb-dsb](https://huggingface.co/ahmedeltawel/Qwen3.5-0.8B-hsb-dsb) | Qwen3.5-0.8B-Base trained with the forward-translated pairs (`qwen35_08b_0p25x`) |
| [ahmedeltawel/Qwen2.5-1.5B-Instruct-hsb-dsb](https://huggingface.co/ahmedeltawel/Qwen2.5-1.5B-Instruct-hsb-dsb) | NRC recipe |

The models are licensed under CC BY-NC-SA 4.0, following the licence of the Sorbian training data.

## Setup

Python 3.11, `git`, `wget` and CUDA GPUs. Steps 3 and 6 and the NRC training use four
GPUs (the 0.8B model was trained on 4 x A100 80 GB, the 2B and 4B models on 4 x H100
94 GB). Evaluation uses one GPU; `decoding/translate.py` runs on one GPU or, more
slowly, on the CPU.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps fla-core==0.5.0 flash-linear-attention==0.5.0
```

All paths and the augmentation, packing and bootstrap seeds are set in `config.sh` and
can be overridden with environment variables (`DATA_ROOT`, `MODELS_ROOT`, `RUNS_ROOT`,
`RESULTS_ROOT`, `NUM_GPUS`, ...). Training shuffles its data with seed 42.

Every step in `slurm/` runs from the repository root, either as a Slurm job
(`sbatch slurm/<step>.sh`, add your cluster's partition options) or directly
(`bash slurm/<step>.sh`). Start each step only after the one before it has finished.
If you change `NUM_GPUS`, request the same number of GPUs from Slurm
(`sbatch --gres=gpu:N ...`). Steps 1 to 3 need internet access.

## Pipeline

| Step | Command | Output |
|------|---------|--------|
| 1. Download data and models | `sbatch slurm/01_download.sh` | `data/raw`, `data/eval` (WMT25 development sets), `models/` |
| 2. Prepare | `sbatch slurm/02_prepare.sh` | `data/processed` (instruction data, MT chat data, deduplicated corpora) |
| 3. Forward translation | `sbatch slurm/03_augment.sh` | `data/augmentation/mt_aug_0p25x` (MT chat data plus 159,000 synthetic German-Upper Sorbian pairs) |
| 4. Clean | `VARIANT=0p25x sbatch slurm/04_clean.sh` | `data/clean/0p25x` |
| 5. Tokenize and pack | `VARIANT=0p25x sbatch slurm/05_tokenize_pack.sh` | `data/final/0p25x/training_data` |
| 6. Train | `MODEL=qwen35_08b VARIANT=0p25x sbatch slurm/06_train.sh` | `runs/qwen35_08b_0p25x/checkpoints/step_002391/model` |
| 7. Weight averaging | `sbatch slurm/07_average.sh` | `runs/qwen35_4b_avg/model` |
| 8. Evaluate | `MODEL=qwen35_08b VARIANT=0p25x sbatch slurm/08_evaluate.sh` | `results/qwen35_08b_0p25x` |
| 9. Significance | `A=<result> B=<result> sbatch slurm/09_significance.sh` | `results/significance` |

Step 4 removes duplicate German-Sorbian pairs and every pair that matches a pair of the
WMT25 development sets, in both translation directions. Training takes its translation
pairs only from this cleaned data; the monolingual and instruction data come from step 2.

`VARIANT` selects the translation data: `0p25x` (with the forward-translated pairs) or
`base` (without them). `MODEL` selects the model: `qwen35_08b` (Qwen3.5-0.8B-Base),
`qwen35_2b` (Qwen3.5-2B) or `qwen35_4b` (Qwen3.5-4B). Model revisions are pinned in
`model_backbone_experiments/configs/`. Training runs 2,391 steps with 128 sequences of
4,096 tokens per step and cannot be resumed, so give step 6 enough time on slower GPUs
(`sbatch --time=...`).

Weight averaging (step 7) needs two 4B runs that start from Qwen3.5-4B and differ only
in the training data: run steps 4 and 5 also with `VARIANT=base` (one variant at a time,
since step 5 also tokenizes the shared `data/tokenized/common`), then step 6 with
`MODEL=qwen35_4b VARIANT=0p25x` and with `MODEL=qwen35_4b VARIANT=base`. Evaluate the
averaged model with `MODEL_DIR=runs/qwen35_4b_avg/model NAME=qwen35_4b_avg sbatch slurm/08_evaluate.sh`.

Evaluation translates the 4,000 sentences of each WMT25 development set and writes chrF++
and BLEU (sacreBLEU) to `results/<name>/metrics_deu-{hsb,dsb}.json` and the per-sentence
outputs to `results/<name>/samples_deu-{hsb,dsb}.jsonl`. Step 9 compares two results with
a paired bootstrap test (10,000 resamples).

Optional variables: `NUM_SHARDS` (step 3, default 4), `MODEL_A`, `MODEL_B` and
`AVERAGE_DIR` (step 7), `BATCH_SIZE` (step 8, default 8), `REPS` (step 9, default 10000)
and `METRIC` (step 9, `chrf` or `bleu`), `NRC_DIR`, `NRC_DATASET_DIR` and
`NRC_OUTPUT_DIR` (NRC steps).

## Translating German text

```bash
python decoding/translate.py --model ahmedeltawel/Qwen3.5-4B-hsb-dsb --target hsb \
    --input input.de.txt --output output.hsb.txt
```

The input has one German sentence per line. Use `--target dsb` for Lower Sorbian.
`--model` also accepts a local model directory such as `runs/qwen35_4b_avg/model`.

## NRC recipe

Supervised fine-tuning of Qwen2.5-1.5B-Instruct with LLaMA-Factory on the parallel data
from step 1:

```bash
pip install -r replications/nrc/requirements.txt
sbatch slurm/nrc_01_prepare.sh
sbatch slurm/nrc_02_train.sh
PROMPT_FORMAT=nrc sbatch slurm/08_evaluate.sh
```

Training runs 66,820 steps. If `nrc_02_train.sh` reaches its time limit, submit it again;
LLaMA-Factory continues from the last checkpoint in `runs/nrc_qwen25_15b`.

## Repository layout

```
config.sh                      paths, models and seeds
slurm/                         pipeline steps in order
replications/tartunlp/         data download and preparation
data_augmentation/             forward translation with tartuNLP/Qwen2.5-3B-Instruct-hsb-dsb
data_cleaning/                 deduplication and development-set overlap removal
replications/tartunlp_qwen35/  tokenization, packing and training
model_backbone_experiments/    model staging and per-model training settings
decoding/                      evaluation, weight averaging and translation
sig_eval/                      paired bootstrap test
replications/nrc/              NRC recipe
```

## Data

The corpora are not included. They are downloaded from their original sources: the WMT
shared tasks on Sorbian (2020-2022, 2025), FineWeb-2, Wikipedia, Magpie, FLAN v2,
OpenAssistant 2, Aya and German News Crawl 2023. Each keeps its own licence.
