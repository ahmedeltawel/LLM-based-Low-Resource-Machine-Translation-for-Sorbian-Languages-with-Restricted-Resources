#!/usr/bin/env python3
import argparse
import csv
import os
import re
from collections import Counter

REPEAT_RE = re.compile(r"\b(\S+)(?:\s+\1\b){3,}", re.UNICODE)


def is_bad(de: str, hsb: str) -> str:
    hsb = hsb.strip()
    de = de.strip()
    if not hsb or len(hsb) < 3:
        return "empty"
    ld, lh = len(de), len(hsb)
    if lh < 0.5 * ld or lh > 2.0 * ld:
        return "len_ratio"
    if REPEAT_RE.search(hsb):
        return "repeat"
    if de.lower() == hsb.lower():
        return "echo"
    return ""


def main():
    data_root = os.environ.get("DATA_ROOT", "data")
    parser = argparse.ArgumentParser(
        description="Merge the translated shards and drop empty, length-mismatched, repetitive and copied outputs."
    )
    parser.add_argument(
        "--in-dir",
        default=f"{data_root}/augmentation/translated",
        help="directory with shard*.csv from 03_translate.py",
    )
    parser.add_argument("--out", default=f"{data_root}/augmentation/clean.csv")
    args = parser.parse_args()

    shards = sorted(
        f for f in os.listdir(args.in_dir) if f.startswith("shard") and f.endswith(".csv")
    )
    print("Shards:", shards)
    stats = Counter()
    written = 0
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["de", "hsb"])
        for sh in shards:
            with open(os.path.join(args.in_dir, sh)) as fi:
                r = csv.reader(fi)
                next(r, None)
                for row in r:
                    if len(row) != 2:
                        stats["malformed"] += 1
                        continue
                    de, hsb = row
                    why = is_bad(de, hsb)
                    if why:
                        stats[why] += 1
                        continue
                    w.writerow([de, hsb])
                    written += 1
                    stats["kept"] += 1
    total = sum(stats.values())
    print(f"\nWrote {written:,} clean pairs to {args.out}")
    for k, n in stats.most_common():
        print(f"  {k:<12} {n:>8,}  ({100*n/total:.2f}%)")


if __name__ == "__main__":
    main()
