#!/usr/bin/env python3
import argparse
import csv
import gc
import os
import random

from datasets import Dataset, concatenate_datasets, load_from_disk

SYSTEM = (
    "You are a professional translator. Translate the following text "
    "from German to Upper Sorbian. Answer with the translated text."
)


def to_example(de, hsb):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": de.strip()},
            {"role": "assistant", "content": hsb.strip()},
        ],
        "src_lang": "de",
        "tgt_lang": "hsb",
    }


def main():
    data_root = os.environ.get("DATA_ROOT", "data")
    parser = argparse.ArgumentParser(
        description="Add a seeded sample of synthetic de->hsb pairs to the MT chat dataset."
    )
    parser.add_argument(
        "--clean", default=f"{data_root}/augmentation/clean.csv", help="output of 04_postfilter.py"
    )
    parser.add_argument(
        "--mt",
        default=f"{data_root}/processed/mt/mt",
        help="MT chat dataset (save_to_disk) to extend",
    )
    parser.add_argument("--out", default=f"{data_root}/augmentation/mt_aug_0p25x")
    parser.add_argument(
        "--pairs", type=int, default=159_000, help="number of synthetic pairs to add"
    )
    parser.add_argument("--seed", type=int, default=int(os.environ.get("AUG_SEED", 17)))
    args = parser.parse_args()
    n = args.pairs

    print(f"Building augmented MT set ({n:,} synthetic pairs)")
    print(f"Reading {args.clean}...")
    random.seed(args.seed)
    pairs = []
    with open(args.clean) as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if len(row) == 2:
                pairs.append((row[0], row[1]))
    print(f"  total clean pairs: {len(pairs):,}")
    random.shuffle(pairs)
    subset = pairs[:n]
    del pairs
    gc.collect()
    print(f"  using subset: {len(subset):,}")

    aug = Dataset.from_list([to_example(d, h) for d, h in subset])
    del subset
    gc.collect()
    print(f"  synthetic HF Dataset: {len(aug):,}")

    print(f"Loading MT dataset {args.mt}...")
    existing = load_from_disk(args.mt)
    print(f"  base MT: {len(existing):,}")

    merged = concatenate_datasets([existing, aug]).shuffle(seed=args.seed)
    del existing, aug
    gc.collect()
    print(f"  merged: {len(merged):,}")

    print(f"Saving to {args.out} ...")
    merged.save_to_disk(args.out)
    print("done")


if __name__ == "__main__":
    main()
