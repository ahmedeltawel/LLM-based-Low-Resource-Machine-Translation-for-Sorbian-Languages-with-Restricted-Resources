#!/usr/bin/env python3
import argparse
import hashlib
import os
import random
import sys


def main():
    data_root = os.environ.get("DATA_ROOT", "data")
    parser = argparse.ArgumentParser(
        description="Remove German sentences already in the training or WMT25 dev/test data and sample the sentences to translate."
    )
    parser.add_argument(
        "--filtered",
        default=f"{data_root}/augmentation/filtered_de.txt",
        help="output of 01_download_filter.py",
    )
    parser.add_argument(
        "--processed-dir",
        default=f"{data_root}/processed",
        help="directory holding parallel/de-hsb/de-hsb.de and parallel/de-dsb/de-dsb.de",
    )
    parser.add_argument(
        "--dev-test-dir",
        default=f"{data_root}/raw/llms-limited-resources2025",
        help="clone of llms-limited-resources2025 (Sorbian/{hsb,dsb}/MT/{dev,test}.de-*.de)",
    )
    parser.add_argument("--out", default=f"{data_root}/augmentation/sample_750k_de.txt")
    parser.add_argument("--target", type=int, default=750_000, help="number of sentences to keep")
    parser.add_argument("--seed", type=int, default=int(os.environ.get("AUG_SEED", 17)))
    args = parser.parse_args()

    existing_de = [
        f"{args.processed_dir}/parallel/de-hsb/de-hsb.de",
        f"{args.processed_dir}/parallel/de-dsb/de-dsb.de",
        f"{args.dev_test_dir}/Sorbian/hsb/MT/dev.de-hsb.de",
        f"{args.dev_test_dir}/Sorbian/dsb/MT/dev.de-dsb.de",
        f"{args.dev_test_dir}/Sorbian/hsb/MT/test.de-hsb.de",
        f"{args.dev_test_dir}/Sorbian/dsb/MT/test.de-dsb.de",
    ]
    missing = [p for p in existing_de if not os.path.isfile(p)]
    if missing:
        for p in missing:
            print(f"  {p}: NOT FOUND", file=sys.stderr)
        sys.exit(f"{len(missing)} required exclusion file(s) missing")

    print("Building dedup set from existing de sources...")
    seen = set()
    for p in existing_de:
        with open(p) as f:
            for line in f:
                s = line.strip()
                if s:
                    seen.add(hashlib.md5(s.encode()).digest())
        print(f"  {p}: cumulative {len(seen):,}")

    print(f"Reading {args.filtered}...")
    candidates = []
    with open(args.filtered) as f:
        for line in f:
            s = line.rstrip("\n")
            if hashlib.md5(s.encode()).digest() in seen:
                continue
            candidates.append(s)
    print(f"  unique vs existing: {len(candidates):,}")

    random.seed(args.seed)
    random.shuffle(candidates)
    keep = candidates[: args.target]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        for s in keep:
            f.write(s + "\n")
    print(f"Wrote {len(keep):,} sentences to {args.out}")


if __name__ == "__main__":
    main()
