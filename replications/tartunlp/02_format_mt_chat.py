#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

from datasets import Dataset

LANG_NAMES = {
    "de": "German",
    "hsb": "Upper Sorbian",
    "dsb": "Lower Sorbian",
}

SYSTEM_PROMPT = (
    "You are a professional translator. "
    "Translate the following text from {src} to {tgt}. "
    "Answer with the translated text."
)


def make_chat(src_text, tgt_text, src_lang, tgt_lang):
    return {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(
                    src=LANG_NAMES[src_lang], tgt=LANG_NAMES[tgt_lang]
                ),
            },
            {"role": "user", "content": src_text.strip()},
            {"role": "assistant", "content": tgt_text.strip()},
        ],
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
    }


def read_parallel_files(src_path, tgt_path):
    with open(src_path, "r", encoding="utf-8") as f:
        src_lines = f.readlines()
    with open(tgt_path, "r", encoding="utf-8") as f:
        tgt_lines = f.readlines()

    assert len(src_lines) == len(tgt_lines), (
        f"Line count mismatch: {src_path} ({len(src_lines)}) vs {tgt_path} ({len(tgt_lines)})"
    )

    pairs = []
    for s, t in zip(src_lines, tgt_lines):
        s, t = s.strip(), t.strip()
        if s and t:
            pairs.append((s, t))
    return pairs


def read_tsv_file(tsv_path):
    pairs = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2 and parts[0] and parts[1]:
                pairs.append((parts[0], parts[1]))
    return pairs


def load_de_hsb_pairs(parallel_dir):
    d = parallel_dir / "de-hsb"
    all_pairs = []

    de_file = d / "wmt20_train.de"
    hsb_file = d / "wmt20_train.hsb"
    if de_file.exists() and hsb_file.exists():
        pairs = read_parallel_files(de_file, hsb_file)
        print(f"    WMT20: {len(pairs)} pairs")
        all_pairs.extend(pairs)

    de_file = d / "wmt21_train.de"
    hsb_file = d / "wmt21_train.hsb"
    if de_file.exists() and hsb_file.exists():
        pairs = read_parallel_files(de_file, hsb_file)
        print(f"    WMT21: {len(pairs)} pairs")
        all_pairs.extend(pairs)

    tsv_file = d / "wmt22_train.tsv"
    if tsv_file.exists():
        pairs = read_tsv_file(tsv_file)
        print(f"    WMT22: {len(pairs)} pairs (TSV)")
        all_pairs.extend(pairs)

    de_file = d / "train.de-hsb.de"
    hsb_file = d / "train.de-hsb.hsb"
    if de_file.exists() and hsb_file.exists():
        pairs = read_parallel_files(de_file, hsb_file)
        print(f"    WMT25: {len(pairs)} pairs")
        all_pairs.extend(pairs)

    return all_pairs


def load_de_dsb_pairs(parallel_dir):
    d = parallel_dir / "de-dsb"
    all_pairs = []

    de_file = d / "wmt21_devel.de"
    dsb_file = d / "wmt21_devel.dsb"
    if de_file.exists() and dsb_file.exists():
        pairs = read_parallel_files(de_file, dsb_file)
        print(f"    WMT21 dev: {len(pairs)} pairs")
        all_pairs.extend(pairs)

    de_file = d / "wmt22_train.de"
    dsb_file = d / "wmt22_train.dsb"
    if de_file.exists() and dsb_file.exists():
        pairs = read_parallel_files(de_file, dsb_file)
        print(f"    WMT22: {len(pairs)} pairs")
        all_pairs.extend(pairs)

    de_file = d / "train.de-dsb.de"
    dsb_file = d / "train.de-dsb.dsb"
    if de_file.exists() and dsb_file.exists():
        pairs = read_parallel_files(de_file, dsb_file)
        print(f"    WMT25: {len(pairs)} pairs")
        all_pairs.extend(pairs)

    return all_pairs


def load_dsb_hsb_pairs(parallel_dir):
    d = parallel_dir / "dsb-hsb"
    all_pairs = []

    dsb_file = d / "wmt22_train.dsb"
    hsb_file = d / "wmt22_train.hsb"
    if dsb_file.exists() and hsb_file.exists():
        pairs = read_parallel_files(dsb_file, hsb_file)
        print(f"    WMT22: {len(pairs)} pairs")
        all_pairs.extend(pairs)

    return all_pairs


def main():
    data_root = os.environ.get("DATA_ROOT", "data")
    parser = argparse.ArgumentParser(
        description="Format the parallel data as chat examples (mt and mtrev datasets)"
    )
    parser.add_argument(
        "--parallel-dir",
        default=os.path.join(data_root, "raw", "parallel"),
        help="raw parallel data with de-hsb, de-dsb and dsb-hsb subdirectories (default: $DATA_ROOT/raw/parallel)",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(data_root, "processed", "mt"),
        help="output directory for the mt and mtrev datasets (default: $DATA_ROOT/processed/mt)",
    )
    args = parser.parse_args()

    parallel_dir = Path(args.parallel_dir)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Formatting MT parallel data as chat instructions")
    print(f"Parallel dir: {parallel_dir}")
    print(f"Output dir:   {output_dir}")

    print("\nLoading de-hsb:")
    de_hsb = load_de_hsb_pairs(parallel_dir)
    print(f"  Total de-hsb: {len(de_hsb)}")

    print("\nLoading de-dsb:")
    de_dsb = load_de_dsb_pairs(parallel_dir)
    print(f"  Total de-dsb: {len(de_dsb)}")

    print("\nLoading dsb-hsb:")
    dsb_hsb = load_dsb_hsb_pairs(parallel_dir)
    print(f"  Total dsb-hsb: {len(dsb_hsb)}")

    print("\nCreating MT dataset (4 translation directions into Sorbian)")

    mt_examples = []

    for de, hsb in de_hsb:
        mt_examples.append(make_chat(de, hsb, "de", "hsb"))
    print(f"  de->hsb: {len(de_hsb)} examples")

    for de, dsb in de_dsb:
        mt_examples.append(make_chat(de, dsb, "de", "dsb"))
    print(f"  de->dsb: {len(de_dsb)} examples")

    for dsb, hsb in dsb_hsb:
        mt_examples.append(make_chat(hsb, dsb, "hsb", "dsb"))
    print(f"  hsb->dsb: {len(dsb_hsb)} examples")

    for dsb, hsb in dsb_hsb:
        mt_examples.append(make_chat(dsb, hsb, "dsb", "hsb"))
    print(f"  dsb->hsb: {len(dsb_hsb)} examples")

    print(f"\n  Total MT examples: {len(mt_examples)}")

    mt_ds = Dataset.from_list(mt_examples)
    mt_path = output_dir / "mt"
    mt_ds.save_to_disk(str(mt_path))
    print(f"  Saved to: {mt_path}")

    print("\nCreating MTrev dataset (reverse directions into German)")

    mtrev_examples = []

    for de, hsb in de_hsb:
        mtrev_examples.append(make_chat(hsb, de, "hsb", "de"))
    print(f"  hsb->de: {len(de_hsb)} examples")

    for de, dsb in de_dsb:
        mtrev_examples.append(make_chat(dsb, de, "dsb", "de"))
    print(f"  dsb->de: {len(de_dsb)} examples")

    print(f"\n  Total MTrev examples: {len(mtrev_examples)}")

    mtrev_ds = Dataset.from_list(mtrev_examples)
    mtrev_path = output_dir / "mtrev"
    mtrev_ds.save_to_disk(str(mtrev_path))
    print(f"  Saved to: {mtrev_path}")

    print("\nSUMMARY")
    print("  Parallel pairs loaded:")
    print(f"    de-hsb:  {len(de_hsb):>10,}")
    print(f"    de-dsb:  {len(de_dsb):>10,}")
    print(f"    dsb-hsb: {len(dsb_hsb):>10,}")
    print()
    print(f"  MT dataset:    {len(mt_examples):>10,} examples (de->hsb + de->dsb + hsb->dsb + dsb->hsb)")
    print(f"  MTrev dataset: {len(mtrev_examples):>10,} examples (hsb->de + dsb->de)")


if __name__ == "__main__":
    main()
