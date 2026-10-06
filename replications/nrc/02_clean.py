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


def read_pair(src_path: Path, tgt_path: Path):
    with src_path.open(encoding="utf-8", errors="replace") as f:
        src = [line.rstrip("\r\n") for line in f]
    with tgt_path.open(encoding="utf-8", errors="replace") as f:
        tgt = [line.rstrip("\r\n") for line in f]
    if len(src) != len(tgt):
        raise ValueError(f"length mismatch: {src_path}={len(src)} vs {tgt_path}={len(tgt)}")
    return list(zip(src, tgt))


def read_tsv(path: Path, expected_src_lang: str):
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
        pairs, swapped = read_tsv(src_path, src_lang)
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
        "src_words": src_words,
        "tgt_words": tgt_words,
        "tsv_columns": swapped_info,
    }


def main():
    parser = argparse.ArgumentParser(description="Apply the NRC cleaning regex to every training corpus in the manifest.")
    parser.add_argument("--manifest", type=Path, default=DATA_ROOT / "nrc" / "manifest.json")
    parser.add_argument("--out-dir", type=Path, default=DATA_ROOT / "nrc" / "cleaned")
    parser.add_argument("--manifest-out", type=Path, default=DATA_ROOT / "nrc" / "cleaned_manifest.json")
    args = parser.parse_args()

    with args.manifest.open() as f:
        manifest = json.load(f)

    print("\nApplying the NRC cleaning regex. Pairs with an empty side after cleaning are dropped.")
    print(f"{'corpus':<36} {'raw':>8} {'after_regex':>12} {'regex_drops':>12}  {'tsv':<8}")
    print("-" * 80)

    results = []
    total_raw = 0
    total_clean = 0
    for entry in manifest["train"]:
        r = process(entry, args.out_dir)
        results.append(r)
        total_raw += r["raw_count"]
        total_clean += r["cleaned_count"]
        tsv = r["tsv_columns"] or "-"
        print(f"{r['name']:<36} {r['raw_count']:>8} {r['cleaned_count']:>12} {r['regex_drops']:>12}  {tsv:<8}")
    print("-" * 80)
    print(f"{'TOTAL':<36} {total_raw:>8} {total_clean:>12} {total_raw - total_clean:>12}")

    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest_out.open("w") as f:
        json.dump({
            "train": results,
            "total_after_regex": total_clean,
            "total_raw": total_raw,
        }, f, indent=2)
    print(f"\nWritten: {args.manifest_out}")
    print(f"\nCleaned {len(results)} corpora: kept {total_clean} of {total_raw} pairs.")


if __name__ == "__main__":
    main()
