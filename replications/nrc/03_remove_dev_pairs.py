#!/usr/bin/env python3
import argparse
import json
import os
import re
from pathlib import Path

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "data"))

CONTROL_CHARS = re.compile(r"[\x01-\x09\x0B\x0C\x0E-\x1D\x7F]")
CR            = re.compile(r"\x0D")
TAB           = re.compile(r"\t")
DOUBLE_ESC_NL = re.compile(r"\\\\ ?[rn]")
SINGLE_ESC_NL = re.compile(r"\\ ?[rn]")
MULTI_SPACE   = re.compile(r"  +")
TRAILING_WS   = re.compile(r" +$")


def clean(s: str) -> str:
    s = CONTROL_CHARS.sub("", s)
    s = CR.sub("", s)
    s = TAB.sub(" ", s)
    s = DOUBLE_ESC_NL.sub(" ", s)
    s = SINGLE_ESC_NL.sub(" ", s)
    s = MULTI_SPACE.sub(" ", s)
    s = TRAILING_WS.sub("", s)
    if s.startswith("\ufeff"):
        s = s[1:]
    return s


def read_pair_files(src: Path, tgt: Path):
    with src.open(encoding="utf-8") as f:
        s_lines = [ln.rstrip("\r\n") for ln in f]
    with tgt.open(encoding="utf-8") as f:
        t_lines = [ln.rstrip("\r\n") for ln in f]
    if len(s_lines) != len(t_lines):
        raise ValueError(f"length mismatch {src} vs {tgt}: {len(s_lines)} vs {len(t_lines)}")
    return list(zip(s_lines, t_lines))


def load_2025_dev_keys(raw_manifest):
    banned = set()
    counts = {}
    for entry in raw_manifest["dev"]:
        pairs = read_pair_files(Path(entry["src"]), Path(entry["tgt"]))
        n = 0
        for a, b in pairs:
            ac, bc = clean(a), clean(b)
            if ac and bc:
                banned.add((ac, bc))
                n += 1
        counts[entry["name"]] = n
    return banned, counts


def main():
    parser = argparse.ArgumentParser(description="Remove the WMT25 development pairs from the cleaned NRC training corpora.")
    parser.add_argument("--manifest-raw", type=Path, default=DATA_ROOT / "nrc" / "manifest.json")
    parser.add_argument("--manifest-clean", type=Path, default=DATA_ROOT / "nrc" / "cleaned_manifest.json")
    parser.add_argument("--out-dir", type=Path, default=DATA_ROOT / "nrc" / "filtered")
    parser.add_argument("--manifest-out", type=Path, default=DATA_ROOT / "nrc" / "filtered_manifest.json")
    args = parser.parse_args()

    raw_manifest = json.loads(args.manifest_raw.read_text())
    clean_manifest = json.loads(args.manifest_clean.read_text())

    banned, banned_counts = load_2025_dev_keys(raw_manifest)
    print("\nWMT25 development pairs after cleaning:")
    for name, n in banned_counts.items():
        print(f"  {name:<20} {n:>6} pairs")
    print(f"  unique pairs: {len(banned)}")

    print("\nRemoving the WMT25 development pairs from the training corpora")
    print(f"{'corpus':<36} {'after_regex':>11} {'drop_dev':>9} {'kept':>8}")
    print("-" * 67)

    results = []
    total_kept = 0

    for entry in clean_manifest["train"]:
        name = entry["name"]
        src_lang = entry["src_lang"]
        tgt_lang = entry["tgt_lang"]
        src_path = Path(entry["src_cleaned"])
        tgt_path = Path(entry["tgt_cleaned"])

        pairs = read_pair_files(src_path, tgt_path)

        kept = []
        drop_dev = 0
        for a, b in pairs:
            if (a, b) in banned:
                drop_dev += 1
                continue
            kept.append((a, b))

        out_dir = args.out_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{src_lang}.txt").write_text(
            "".join(s + "\n" for s, _ in kept), encoding="utf-8")
        (out_dir / f"{tgt_lang}.txt").write_text(
            "".join(t + "\n" for _, t in kept), encoding="utf-8")

        total_kept += len(kept)

        print(f"{name:<36} {len(pairs):>11} {drop_dev:>9} {len(kept):>8}")

        results.append({
            "name": name,
            "src_filtered": str(out_dir / f"{src_lang}.txt"),
            "tgt_filtered": str(out_dir / f"{tgt_lang}.txt"),
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "after_regex": len(pairs),
            "dropped_dev_overlap": drop_dev,
            "kept": len(kept),
        })

    print("-" * 67)
    print(f"{'TOTAL':<36} {sum(r['after_regex'] for r in results):>11} "
          f"{sum(r['dropped_dev_overlap'] for r in results):>9} {total_kept:>8}")

    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(json.dumps({
        "train": results,
        "total_kept": total_kept,
    }, indent=2))
    print(f"\nWritten: {args.manifest_out}")
    print(f"\nKept {total_kept} training pairs after removing the WMT25 development pairs.")


if __name__ == "__main__":
    main()
