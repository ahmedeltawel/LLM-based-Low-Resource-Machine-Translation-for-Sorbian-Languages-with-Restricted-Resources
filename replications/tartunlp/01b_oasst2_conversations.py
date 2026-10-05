#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from datasets import load_from_disk, load_dataset, concatenate_datasets, Dataset
from collections import defaultdict

TARGET_LANGS = {"en", "de", "pl", "cs", "sk", "sl"}


def build_oasst2_conversations(input_dir, output_dir):
    print("Reconstructing oasst2 conversation trees")

    ds_train = load_from_disk(str(input_dir / "oasst2"))
    ds_val = load_dataset("OpenAssistant/oasst2", split="validation")
    ds = concatenate_datasets([ds_train, ds_val])
    print(f"  Train: {len(ds_train)}, Val: {len(ds_val)}, Combined: {len(ds)}")

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

    print(f"  Reconstructed conversations: {len(conversations)}")

    out_ds = Dataset.from_list(conversations)
    out_path = output_dir / "oasst2"
    out_ds.save_to_disk(str(out_path))
    print(f"  Saved to: {out_path}")
    print(f"  Paper target: 24,256")
    print(f"  Difference: {len(conversations) - 24256:+,}")

    return len(conversations)


def main():
    data_root = os.environ.get("DATA_ROOT", "data")
    parser = argparse.ArgumentParser(
        description="Build oasst2 leaf-to-root conversations (train + validation)"
    )
    parser.add_argument(
        "--input-dir",
        default=os.path.join(data_root, "raw", "instructions"),
        help="downloaded instruction datasets (default: $DATA_ROOT/raw/instructions)",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(data_root, "processed", "instructions"),
        help="filtered output written by 01_filter_instructions.py (default: $DATA_ROOT/processed/instructions)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    oasst2_count = build_oasst2_conversations(input_dir, output_dir)

    aya_count = len(load_from_disk(str(output_dir / "aya")))
    magpie_count = len(load_from_disk(str(output_dir / "magpie")))
    flan_count = len(load_from_disk(str(output_dir / "flan_v2")))

    total = aya_count + magpie_count + oasst2_count + flan_count

    print("\nSUMMARY")
    print(f"  aya:         {aya_count:>10,}  (target: 5,668)")
    print(f"  magpie:      {magpie_count:>10,}  (target: 296,121)")
    print(f"  oasst2:      {oasst2_count:>10,}  (target: 24,256)")
    print(f"  flan_v2:     {flan_count:>10,}  (target: 89,982)")
    print(f"  {'-'*40}")
    print(f"  TOTAL:       {total:>10,}")


if __name__ == "__main__":
    main()
