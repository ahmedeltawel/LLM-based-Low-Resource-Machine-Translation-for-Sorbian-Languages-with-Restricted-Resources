#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "data"))

PAPER_CLEANED = {
    "2020.devel.hsb-de.de-hsb":         1986,
    "2020.devel_test.hsb-de.de-hsb":    1981,
    "2020.train.hsb-de.de-hsb":        59703,
    "2021.devel.dsb-de.de-dsb":          601,
    "2021.devel_test.dsb-de.de-dsb":     602,
    "2021.train.hsb-de.de-hsb":        86719,
    "2022.40194_train_dsb_de.de-dsb":  40194,
    "2022.dev_dsb_hsb_new.dsb-hsb":      700,
    "2022.HSB-DE.dev.tsv.de-hsb":         34,
    "2022.HSB-DE_train.tsv.de-hsb":   301536,
    "2022.valid.de-dsb":                   1,
    "2022.valid_dsb_hsb.dsb-hsb":        709,
    "2025.train.de-dsb":              171964,
    "2025.train.de-hsb":              187270,
}
PAPER_TOTAL = sum(PAPER_CLEANED.values())

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
    parser = argparse.ArgumentParser(description="Remove WMT25 dev pairs from the cleaned NRC training corpora.")
    parser.add_argument("--manifest-raw", type=Path, default=DATA_ROOT / "nrc" / "manifest.json")
    parser.add_argument("--manifest-clean", type=Path, default=DATA_ROOT / "nrc" / "cleaned_manifest.json")
    parser.add_argument("--out-dir", type=Path, default=DATA_ROOT / "nrc" / "deduped")
    parser.add_argument("--manifest-out", type=Path, default=DATA_ROOT / "nrc" / "deduped_manifest.json")
    args = parser.parse_args()

    raw_manifest = json.loads(args.manifest_raw.read_text())
    clean_manifest = json.loads(args.manifest_clean.read_text())

    banned, banned_counts = load_2025_dev_keys(raw_manifest)
    print("\nLoaded 2025 dev pairs as banned set:")
    for name, n in banned_counts.items():
        print(f"  {name:<20} {n:>6} pairs")
    print(f"  total banned keys: {len(banned)}")

    print("\nPair dedup (drop 2025-dev matches only; no cross-corpus dedup)")
    print(f"{'corpus':<36} {'after_regex':>11} {'drop_dev':>9} {'kept':>8} {'paper':>8} {'delta':>6}  {'status':<8}")
    print('-'*92)

    results = []
    total_kept = 0
    total_paper = 0
    n_ok = 0
    n_diff = 0

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

        paper = PAPER_CLEANED.get(name)
        delta = len(kept) - paper if paper is not None else 0
        tol = max(2, int(abs(paper) * 0.01)) if paper else 2
        status = "OK" if paper is not None and abs(delta) <= tol else "DIFF"
        if status == "OK":
            n_ok += 1
        else:
            n_diff += 1
        total_kept += len(kept)
        if paper is not None:
            total_paper += paper

        print(f"{name:<36} {len(pairs):>11} {drop_dev:>9} "
              f"{len(kept):>8} {paper:>8} {delta:>+6}  {status:<8}")

        results.append({
            "name": name,
            "src_deduped": str(out_dir / f"{src_lang}.txt"),
            "tgt_deduped": str(out_dir / f"{tgt_lang}.txt"),
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "after_regex": len(pairs),
            "dropped_dev_overlap": drop_dev,
            "kept": len(kept),
            "paper_cleaned": paper,
            "delta_vs_paper": delta,
            "status": status,
        })

    print('-'*92)
    overall_delta = total_kept - total_paper
    print(f"{'TOTAL':<36} {sum(r['after_regex'] for r in results):>11} "
          f"{sum(r['dropped_dev_overlap'] for r in results):>9} "
          f"{total_kept:>8} {total_paper:>8} {overall_delta:>+6}  OK:{n_ok} DIFF:{n_diff}")

    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(json.dumps({
        "train": results,
        "total_kept": total_kept,
        "paper_total": PAPER_TOTAL,
        "delta_vs_paper": overall_delta,
    }, indent=2))
    print(f"\nWritten: {args.manifest_out}")

    if abs(overall_delta) > max(100, int(PAPER_TOTAL * 0.01)):
        print(f"\nTotal off by {overall_delta:+d} (more than 1% of paper target {PAPER_TOTAL}).", file=sys.stderr)
        sys.exit(1)
    print(f"\nPASS: total {total_kept} vs paper {PAPER_TOTAL} ({overall_delta:+d}, "
          f"{100*overall_delta/PAPER_TOTAL:+.2f}%, within 1%). "
          f"{n_ok}/{len(results)} per-corpus within tolerance.")


if __name__ == "__main__":
    main()
