#!/usr/bin/env python3
import argparse
import csv
import os
import re
import unicodedata

from datasets import load_from_disk

csv.field_size_limit(10**9)

DATA_ROOT = os.environ.get("DATA_ROOT", "data")
LANGUAGES = ("hsb", "dsb")


def normalize(text):
    text = unicodedata.normalize("NFKC", str(text)).strip().lower()
    return re.sub(r"\s+", " ", text)


def message_text(messages, role):
    for message in messages:
        if message.get("role") == role:
            return (message.get("content") or "").strip()
    return ""


def pair_key(example):
    source = message_text(example["messages"], "user")
    target = message_text(example["messages"], "assistant")
    if not source or not target:
        return None
    return normalize(source), normalize(target)


def load_dev_pairs(dev_dir):
    dev_pairs = {}
    for lang in LANGUAGES:
        path = os.path.join(dev_dir, f"dev.de-{lang}.csv")
        pairs = set()
        with open(path, encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if "de" in row and lang in row:
                    pairs.add((normalize(row["de"]), normalize(row[lang])))
        dev_pairs[lang] = pairs
        print(f"dev pairs {lang}: {len(pairs):,} ({path})")
    return dev_pairs


def clean_forward(dataset, dev_pairs):
    seen = set()
    dropped_dev = set()
    kept = []
    n_dev = 0
    n_dup = 0
    for index, example in enumerate(dataset):
        tgt_lang = example.get("tgt_lang")
        src_lang = example.get("src_lang")
        if tgt_lang not in LANGUAGES or src_lang != "de":
            kept.append(index)
            continue
        key = pair_key(example)
        if key is None:
            kept.append(index)
            continue
        if key in dev_pairs[tgt_lang]:
            n_dev += 1
            dropped_dev.add(key)
            continue
        if key in seen:
            n_dup += 1
            continue
        seen.add(key)
        kept.append(index)
    return kept, dropped_dev, n_dev, n_dup


def clean_reverse(dataset, dev_reverse):
    seen = set()
    kept = []
    n_dev = 0
    n_dup = 0
    for index, example in enumerate(dataset):
        src_lang = example.get("src_lang")
        tgt_lang = example.get("tgt_lang")
        if tgt_lang != "de" or src_lang not in LANGUAGES:
            kept.append(index)
            continue
        key = pair_key(example)
        if key is None:
            kept.append(index)
            continue
        if key in dev_reverse:
            n_dev += 1
            continue
        if key in seen:
            n_dup += 1
            continue
        seen.add(key)
        kept.append(index)
    return kept, n_dev, n_dup


def main():
    parser = argparse.ArgumentParser(
        description="Remove development-set overlap and duplicate pairs from the mt and mtrev chat datasets."
    )
    parser.add_argument(
        "--mt",
        required=True,
        help="forward MT dataset (load_from_disk dir), e.g. data/augmentation/mt_aug_0p25x or data/processed/mt/mt",
    )
    parser.add_argument(
        "--mtrev",
        default=f"{DATA_ROOT}/processed/mt/mtrev",
        help="reverse MT dataset (load_from_disk dir)",
    )
    parser.add_argument(
        "--dev-dir",
        default=f"{DATA_ROOT}/eval",
        help="dir with dev.de-hsb.csv and dev.de-dsb.csv",
    )
    parser.add_argument("--out", required=True, help="output dir; writes <out>/mt and <out>/mtrev")
    args = parser.parse_args()

    dev_pairs = load_dev_pairs(args.dev_dir)

    mt = load_from_disk(args.mt)
    print(f"\nmt: {len(mt):,} ({args.mt})")
    kept, dropped_dev, n_dev, n_dup = clean_forward(mt, dev_pairs)
    print(f"  removed dev overlap {n_dev:,} | duplicates {n_dup:,} | kept {len(kept):,}")
    out_mt = os.path.join(args.out, "mt")
    mt.select(kept).save_to_disk(out_mt)
    print(f"  saved {out_mt}")

    dev_reverse = {(target, source) for source, target in dropped_dev}
    mtrev = load_from_disk(args.mtrev)
    print(f"\nmtrev: {len(mtrev):,} ({args.mtrev})")
    kept_rev, n_dev_rev, n_dup_rev = clean_reverse(mtrev, dev_reverse)
    print(
        f"  removed dev overlap {n_dev_rev:,} | duplicates {n_dup_rev:,} | kept {len(kept_rev):,}"
    )
    out_mtrev = os.path.join(args.out, "mtrev")
    mtrev.select(kept_rev).save_to_disk(out_mtrev)
    print(f"  saved {out_mtrev}")


if __name__ == "__main__":
    main()
