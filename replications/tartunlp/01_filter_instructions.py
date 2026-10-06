#!/usr/bin/env python3
import argparse
import os
from collections import defaultdict
from pathlib import Path

from datasets import Dataset, load_from_disk

TARGET_LANGS = {"en", "de", "pl", "cs", "sk", "sl"}

LANG_MAP = {
    "eng": "en", "deu": "de", "pol": "pl", "ces": "cs", "slk": "sk", "slv": "sl",
    "english": "en", "german": "de", "polish": "pl", "czech": "cs", "slovak": "sk", "slovenian": "sl",
}


def normalize_lang(lang):
    if lang is None:
        return None
    lang = lang.lower().strip()
    return LANG_MAP.get(lang, lang)


def is_target_lang(lang):
    return normalize_lang(lang) in TARGET_LANGS


def detect_language_column(ds):
    for col in ["language", "lang", "locale", "language_code"]:
        if col in ds.column_names:
            return col
    return None


def filter_aya(ds):
    lang_col = detect_language_column(ds)
    if lang_col is None:
        print("  No language column found. Keeping all.")
        return ds

    filtered = ds.filter(lambda x: is_target_lang(x[lang_col]), num_proc=4)
    return filtered


def filter_magpie(ds):
    lang_col = detect_language_column(ds)
    if lang_col:
        filtered = ds.filter(lambda x: is_target_lang(x[lang_col]), num_proc=4)
        return filtered

    print("  No language column found. Keeping all (mostly English).")
    return ds


def build_oasst2_conversations(ds):
    messages = {}
    children = defaultdict(list)

    for row in ds:
        msg_id = row["message_id"]
        parent_id = row["parent_id"]
        messages[msg_id] = row
        if parent_id:
            children[parent_id].append(msg_id)

    leaves = [mid for mid in messages if mid not in children]
    print(f"  Found {len(leaves)} leaf nodes")

    conversations = []

    for leaf_id in leaves:
        path = []
        current = leaf_id
        while current:
            path.append(messages[current])
            current = messages[current]["parent_id"]
        path.reverse()

        if len(path) < 2:
            continue

        if path[-1]["role"] != "assistant":
            continue

        if any(m.get("deleted", False) for m in path):
            continue

        lang = path[0].get("lang", "")
        if lang not in TARGET_LANGS:
            continue

        conv = []
        for msg in path:
            role = msg["role"]
            if role == "prompter":
                role = "user"
            conv.append({"role": role, "content": msg["text"]})

        if conv and conv[0]["role"] == "user":
            conversations.append({"conversations": conv, "lang": lang})

    return Dataset.from_list(conversations)


def filter_flan_v2(ds):
    print("  FLAN is mostly English. Keeping all.")
    return ds


def process_dataset(name, path, filter_fn, output_dir):
    print(f"\nProcessing: {name}")
    print(f"  Input: {path}")

    ds = load_from_disk(str(path))
    print(f"  Loaded: {len(ds)} examples")

    filtered = filter_fn(ds)
    print(f"  After filtering: {len(filtered)} examples")

    out_path = output_dir / name
    filtered.save_to_disk(str(out_path))
    print(f"  Saved to: {out_path}")

    return len(ds), len(filtered)


def main():
    data_root = os.environ.get("DATA_ROOT", "data")
    parser = argparse.ArgumentParser(
        description="Filter the instruction datasets to en, de, pl, cs, sk and sl and build the OpenAssistant 2 conversations"
    )
    parser.add_argument(
        "--input-dir",
        default=os.path.join(data_root, "raw", "instructions"),
        help="downloaded instruction datasets (default: $DATA_ROOT/raw/instructions)",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(data_root, "processed", "instructions"),
        help="filtered output (default: $DATA_ROOT/processed/instructions)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Instruction dataset filtering")
    print("Target languages: en, de, pl, cs, sk, sl")
    print(f"Input dir:  {input_dir}")
    print(f"Output dir: {output_dir}")

    results = {}

    datasets_config = [
        ("aya", "aya", filter_aya),
        ("magpie", "magpie", filter_magpie),
        ("oasst2", "oasst2", build_oasst2_conversations),
        ("flan_v2", "flan_v2", filter_flan_v2),
    ]

    total_before = 0
    total_after = 0

    for name, subdir, filter_fn in datasets_config:
        path = input_dir / subdir
        if not path.exists():
            print(f"\n  SKIPPING {name}: {path} not found")
            continue

        before, after = process_dataset(name, path, filter_fn, output_dir)
        results[name] = (before, after)
        total_before += before
        total_after += after

    print("\nSUMMARY")
    print(f"{'Dataset':<30} {'Before':>10} {'After':>10} {'Kept%':>8}")
    print("-" * 60)
    for name, (before, after) in results.items():
        pct = 100 * after / before if before > 0 else 0
        print(f"{name:<30} {before:>10,} {after:>10,} {pct:>7.1f}%")
    print("-" * 60)
    print(f"{'TOTAL':<30} {total_before:>10,} {total_after:>10,}")


if __name__ == "__main__":
    main()
