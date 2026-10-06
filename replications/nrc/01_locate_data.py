#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "data"))
TOLERANCE = 2


def train_corpora(raw: Path):
    return [
        ("2020.devel.hsb-de.de-hsb",       raw/"devtest/devel.hsb-de.de",              raw/"devtest/devel.hsb-de.hsb",             "de", "hsb", 2000),
        ("2020.devel_test.hsb-de.de-hsb",  raw/"devtest/devel_test.hsb-de.de",         raw/"devtest/devel_test.hsb-de.hsb",        "de", "hsb", 2000),
        ("2020.train.hsb-de.de-hsb",       raw/"parallel/de-hsb/wmt20_train.de",       raw/"parallel/de-hsb/wmt20_train.hsb",      "de", "hsb", 60000),
        ("2021.devel.dsb-de.de-dsb",       raw/"parallel/de-dsb/wmt21_devel.de",       raw/"parallel/de-dsb/wmt21_devel.dsb",      "de", "dsb", 601),
        ("2021.devel_test.dsb-de.de-dsb",  raw/"devtest/devel_test.dsb-de.de",         raw/"devtest/devel_test.dsb-de.dsb",        "de", "dsb", 602),
        ("2021.train.hsb-de.de-hsb",       raw/"parallel/de-hsb/wmt21_train.de",       raw/"parallel/de-hsb/wmt21_train.hsb",      "de", "hsb", 87520),
        ("2022.40194_train_dsb_de.de-dsb", raw/"parallel/de-dsb/wmt22_train.de",       raw/"parallel/de-dsb/wmt22_train.dsb",      "de", "dsb", 40194),
        ("2022.dev_dsb_hsb_new.dsb-hsb",   raw/"devtest/dev_dsb_hsb_new.dsb",          raw/"devtest/dev_dsb_hsb_new.hsb",          "dsb", "hsb", 700),
        ("2022.HSB-DE.dev.tsv.de-hsb",     raw/"devtest/wmt22_dev_hsb-de.tsv",         None,                                       "de", "hsb", 2000),
        ("2022.HSB-DE_train.tsv.de-hsb",   raw/"parallel/de-hsb/wmt22_train.tsv",      None,                                       "de", "hsb", 301536),
        ("2022.valid.de-dsb",              raw/"devtest/wmt22_valid_dsb-de.de",        raw/"devtest/wmt22_valid_dsb-de.dsb",       "de", "dsb", 1353),
        ("2022.valid_dsb_hsb.dsb-hsb",     raw/"devtest/valid_dsb_hsb.dsb",            raw/"devtest/valid_dsb_hsb.hsb",            "dsb", "hsb", 709),
        ("2025.train.de-dsb",              raw/"parallel/de-dsb/train.de-dsb.de",      raw/"parallel/de-dsb/train.de-dsb.dsb",     "de", "dsb", 171964),
        ("2025.train.de-hsb",              raw/"parallel/de-hsb/train.de-hsb.de",      raw/"parallel/de-hsb/train.de-hsb.hsb",     "de", "hsb", 187270),
    ]


def dev_sets(raw: Path):
    return [
        ("2025.dev.de-dsb", raw/"devtest/dev.de-dsb.de", raw/"devtest/dev.de-dsb.dsb", "de", "dsb", 4000),
        ("2025.dev.de-hsb", raw/"devtest/dev.de-hsb.de", raw/"devtest/dev.de-hsb.hsb", "de", "hsb", 4000),
    ]


def count_lines(p: Path) -> int:
    if not p.exists():
        return -1
    with p.open("rb") as f:
        return sum(1 for _ in f)


def verify(rows, label):
    print(f"\n{label}")
    print(f"{'corpus':<36} {'expected':>10} {'actual':>10} {'status':>12}")
    print("-" * 78)
    manifest = []
    n_ok = 0
    total_exp = 0
    total_act = 0
    for name, src, tgt, sl, tl, exp in rows:
        src_n = count_lines(src)
        if tgt is None:
            actual = src_n
            status = "MISSING" if src_n < 0 else ("OK" if abs(src_n - exp) <= TOLERANCE else "DIFF")
        else:
            tgt_n = count_lines(tgt)
            if src_n < 0 or tgt_n < 0:
                actual, status = -1, "MISSING"
            elif src_n != tgt_n:
                actual, status = -1, f"MISMATCH({src_n}/{tgt_n})"
            else:
                actual = src_n
                status = "OK" if abs(src_n - exp) <= TOLERANCE else "DIFF"
        if status == "OK":
            n_ok += 1
        print(f"{name:<36} {exp:>10} {actual:>10} {status:>12}")
        total_exp += exp
        if actual > 0:
            total_act += actual
        manifest.append({
            "name": name,
            "src": str(src),
            "tgt": str(tgt) if tgt else None,
            "src_lang": sl,
            "tgt_lang": tl,
            "expected": exp,
            "actual": actual,
            "status": status,
        })
    print("-" * 78)
    print(f"{'TOTAL':<36} {total_exp:>10} {total_act:>10}    {n_ok}/{len(rows)} OK")
    return manifest, n_ok == len(rows)


def main():
    parser = argparse.ArgumentParser(description="Map the NRC training and development corpora to files under the raw data directory, check their line counts and write manifest.json.")
    parser.add_argument("--raw-dir", type=Path, default=DATA_ROOT / "raw")
    parser.add_argument("--out", type=Path, default=DATA_ROOT / "nrc" / "manifest.json")
    args = parser.parse_args()

    raw = args.raw_dir
    out = args.out
    if not raw.exists():
        print(f"ERROR: {raw} not found. Download the raw data first.", file=sys.stderr)
        sys.exit(2)
    train_man, train_ok = verify(train_corpora(raw), "Training corpora")
    dev_man, dev_ok = verify(dev_sets(raw), "Development sets (held out, removed from training in step 3)")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump({"train": train_man, "dev": dev_man, "tolerance": TOLERANCE}, f, indent=2)
    print(f"\nManifest written to {out}")
    if not (train_ok and dev_ok):
        print("\nERROR: some corpora are missing or do not have the expected number of lines. Check the raw data layout before step 2.", file=sys.stderr)
        sys.exit(1)
    print(f"\nAll corpora found with the expected number of lines (tolerance {TOLERANCE}).")


if __name__ == "__main__":
    main()
