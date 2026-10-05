#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from datasets import load_dataset

INSTRUCTION_DATASETS = [
    ("magpie", "Magpie-Align/Magpie-Llama-3.1-Pro-MT-300K-Filtered", "Magpie"),
    ("flan_v2", "ai2-adapt-dev/flan_v2_converted", "FLAN"),
    ("oasst2", "OpenAssistant/oasst2", "OpenAssistant2"),
    ("aya", "CohereLabs/aya_dataset", "Aya"),
]


def download_fineweb2(raw_dir):
    for lang in ["hsb", "dsb"]:
        print(f"Downloading FineWeb-2 {lang.upper()}...")
        ds = load_dataset("HuggingFaceFW/fineweb-2", name=f"{lang}_Latn", split="train")
        out_path = raw_dir / "monolingual" / lang / "fineweb2"
        ds.save_to_disk(str(out_path))
        print(f"  {lang.upper()} FineWeb-2: {len(ds)} documents -> {out_path}")


def download_wikipedia(raw_dir):
    for lang in ["hsb", "dsb"]:
        print(f"Downloading Wikipedia {lang.upper()}...")
        ds = load_dataset("wikimedia/wikipedia", f"20231101.{lang}", split="train")
        out_path = raw_dir / "monolingual" / lang / "wikipedia"
        ds.save_to_disk(str(out_path))
        print(f"  {lang.upper()} Wikipedia: {len(ds)} articles -> {out_path}")


def download_instructions(raw_dir):
    for name, hf_id, label in INSTRUCTION_DATASETS:
        print(f"Downloading {hf_id}...")
        ds = load_dataset(hf_id, split="train")
        out_path = raw_dir / "instructions" / name
        ds.save_to_disk(str(out_path))
        print(f"  {label}: {len(ds)} examples -> {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download FineWeb-2, Wikipedia and the instruction datasets from the Hugging Face Hub"
    )
    parser.add_argument(
        "--out",
        default=os.path.join(os.environ.get("DATA_ROOT", "data"), "raw"),
        help="raw data directory (default: $DATA_ROOT/raw)",
    )
    args = parser.parse_args()

    raw_dir = Path(args.out)
    print(f"Output dir: {raw_dir}")

    download_fineweb2(raw_dir)
    download_wikipedia(raw_dir)
    download_instructions(raw_dir)

    print("Hugging Face downloads complete")


if __name__ == "__main__":
    main()
