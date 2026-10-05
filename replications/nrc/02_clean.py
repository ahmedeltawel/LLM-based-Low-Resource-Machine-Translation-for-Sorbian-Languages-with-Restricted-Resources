#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
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


def read_pair(src_path: Path, tgt_path: Path):
    with src_path.open(encoding="utf-8", errors="replace") as f:
        src = [line.rstrip("\r\n") for line in f]
    with tgt_path.open(encoding="utf-8", errors="replace") as f:
        tgt = [line.rstrip("\r\n") for line in f]
    if len(src) != len(tgt):
        raise ValueError(f"length mismatch: {src_path}={len(src)} vs {tgt_path}={len(tgt)}")
    return list(zip(src, tgt))


def read_tsv(path: Path, expected_src_lang: str, expected_tgt_lang: str):
    with path.open(encoding="utf-8", errors="replace") as f:
        lines = [line.rstrip("\r\n") for line in f]
    if not lines:
        return [], False
    header_parts = lines[0].lstrip("\ufeff").split("\t")
    is_header = (
        len(header_parts) == 2
        and header_parts[0].lower() in {"de", "dsb", "hsb"}
        and header_parts[1].lower() in {"de", "dsb", "hsb"}
    )
    if is_header:
        col0 = header_parts[0].lower()
        data = lines[1:]
    else:
        col0 = expected_src_lang
        data = lines

    swapped = (col0 != expected_src_lang)
    pairs = []
    for line in data:
        parts = line.split("\t", 1)
        if len(parts) < 2:
            pairs.append((line, ""))
            continue
        a, b = parts
        if swapped:
            pairs.append((b, a))
        else:
            pairs.append((a, b))
    return pairs, swapped


def process(entry, out_root: Path):
    name = entry["name"]
    src_path = Path(entry["src"])
    tgt_path_str = entry["tgt"]
    src_lang = entry["src_lang"]
    tgt_lang = entry["tgt_lang"]

    swapped_info = None
    if tgt_path_str is None:
        pairs, swapped = read_tsv(src_path, src_lang, tgt_lang)
        swapped_info = "swapped" if swapped else "native"
    else:
        pairs = read_pair(src_path, Path(tgt_path_str))
    raw_count = len(pairs)

    cleaned_pairs = []
    for a, b in pairs:
        ac = clean(a)
        bc = clean(b)
        if ac and bc:
            cleaned_pairs.append((ac, bc))
    cleaned_count = len(cleaned_pairs)

    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    src_out = out_dir / f"{src_lang}.txt"
    tgt_out = out_dir / f"{tgt_lang}.txt"
    with src_out.open("w", encoding="utf-8") as f:
        for s, _ in cleaned_pairs:
            f.write(s + "\n")
    with tgt_out.open("w", encoding="utf-8") as f:
        for _, t in cleaned_pairs:
            f.write(t + "\n")

    src_words = sum(len(s.split()) for s, _ in cleaned_pairs)
    tgt_words = sum(len(t.split()) for _, t in cleaned_pairs)

    paper_final = PAPER_CLEANED.get(name)
    regex_drops = raw_count - cleaned_count

    return {
        "name": name,
        "src_cleaned": str(src_out),
        "tgt_cleaned": str(tgt_out),
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "raw_count": raw_count,
        "cleaned_count": cleaned_count,
        "regex_drops": regex_drops,
        "paper_final_cleaned": paper_final,
        "src_words": src_words,
        "tgt_words": tgt_words,
        "tsv_columns": swapped_info,
    }


def main():
    parser = argparse.ArgumentParser(description="Apply the NRC Figure 3 cleaning regex to every training corpus in the manifest.")
    parser.add_argument("--manifest", type=Path, default=DATA_ROOT / "nrc" / "manifest.json")
    parser.add_argument("--out-dir", type=Path, default=DATA_ROOT / "nrc" / "cleaned")
    parser.add_argument("--manifest-out", type=Path, default=DATA_ROOT / "nrc" / "cleaned_manifest.json")
    args = parser.parse_args()

    with args.manifest.open() as f:
        manifest = json.load(f)

    print("\nCleaning (NRC paper Fig 3 regex, pairs empty after cleaning dropped). Dev-pair removal is step 3.")
    print(f"{'corpus':<36} {'raw':>8} {'after_regex':>12} {'regex_drops':>12} {'paper_final':>12} {'gap_to_paper':>13} {'tsv':<8}")
    print('-'*100)

    results = []
    total_raw = 0
    total_clean = 0
    total_paper_final = 0
    for entry in manifest["train"]:
        r = process(entry, args.out_dir)
        results.append(r)
        total_raw += r["raw_count"]
        total_clean += r["cleaned_count"]
        if r["paper_final_cleaned"] is not None:
            total_paper_final += r["paper_final_cleaned"]
        paper_final = r["paper_final_cleaned"]
        gap = (r["cleaned_count"] - paper_final) if paper_final is not None else 0
        tsv = r["tsv_columns"] or "-"
        print(f"{r['name']:<36} {r['raw_count']:>8} {r['cleaned_count']:>12} "
              f"{r['regex_drops']:>12} {paper_final:>12} {gap:>+13}  {tsv:<8}")
    print('-'*100)
    print(f"{'TOTAL':<36} {total_raw:>8} {total_clean:>12} {total_raw-total_clean:>12} {total_paper_final:>12} "
          f"{total_clean - total_paper_final:>+13}")

    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest_out.open("w") as f:
        json.dump({
            "train": results,
            "paper_total_cleaned": total_paper_final,
            "total_after_regex": total_clean,
            "total_raw": total_raw,
        }, f, indent=2)
    print(f"\nWritten: {args.manifest_out}")
    print(f"\nPASS: regex applied to {len(results)} corpora. "
          f"The gap of {total_clean - total_paper_final:+d} to the paper's cleaned total "
          f"is closed by step 3 (dev-pair removal).")


if __name__ == "__main__":
    main()
