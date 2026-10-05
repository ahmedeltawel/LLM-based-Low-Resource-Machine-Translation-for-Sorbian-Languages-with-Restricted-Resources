#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import pandas as pd


def make_csv(file1, file2, header1, header2, out_path):
    with open(file1, "r", encoding="utf-8") as f:
        a_lines = [line.strip() for line in f]

    with open(file2, "r", encoding="utf-8") as f:
        b_lines = [line.strip() for line in f]

    pd.DataFrame({header1: a_lines, header2: b_lines}).to_csv(out_path, index=False)
    print(f"  {out_path}: {len(a_lines):,} rows ({header1},{header2})")


def main():
    data_root = os.environ.get("DATA_ROOT", "data")
    parser = argparse.ArgumentParser(
        description="Write the WMT25 Sorbian development sets as dev.de-{hsb,dsb}.csv"
    )
    parser.add_argument(
        "--repo-dir",
        default=os.path.join(data_root, "raw", "llms-limited-resources2025"),
        help="clone of TUM-NLP/llms-limited-resources2025 (default: $DATA_ROOT/raw/llms-limited-resources2025)",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(data_root, "eval"),
        help="output directory for the CSV files (default: $DATA_ROOT/eval)",
    )
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for lang in ["dsb", "hsb"]:
        mt_dir = repo_dir / "Sorbian" / lang / "MT"
        make_csv(
            str(mt_dir / f"dev.de-{lang}.de"),
            str(mt_dir / f"dev.de-{lang}.{lang}"),
            "de",
            lang,
            str(out_dir / f"dev.de-{lang}.csv"),
        )


if __name__ == "__main__":
    main()
