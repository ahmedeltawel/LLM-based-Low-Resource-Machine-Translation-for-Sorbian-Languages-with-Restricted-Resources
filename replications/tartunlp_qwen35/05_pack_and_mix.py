#!/usr/bin/env python3
import argparse
import os
import random
import shutil
from pathlib import Path
from datasets import load_from_disk, Dataset, concatenate_datasets

DATA_ROOT = os.environ.get("DATA_ROOT", "data")
VARIANT = os.environ.get("VARIANT", "0p25x")

SEQ_LEN = 4096
CHUNK_SIZE = 50000
MT_GROUPS = ("mt", "mtrev")


def dataset_path(dataset_name, common_dir, mt_dir):
    if dataset_name in MT_GROUPS:
        return mt_dir / dataset_name
    return common_dir / dataset_name


def stream_tokens(path, epochs=1):
    if not path.exists():
        print(f"  WARNING: {path} not found, skipping")
        return

    ds = load_from_disk(str(path))
    n = len(ds)

    for epoch in range(epochs):
        indices = list(range(n))
        random.shuffle(indices)

        for idx in indices:
            row = ds[idx]
            yield row["input_ids"], row["loss_mask"]

    del ds


def count_tokens(path):
    if not path.exists():
        return 0
    ds = load_from_disk(str(path))
    n = len(ds)
    sample_n = min(100, n)
    total = sum(len(ds[i]["input_ids"]) for i in range(sample_n))
    avg = total / sample_n
    del ds
    return int(avg * n)


def main():
    ap = argparse.ArgumentParser(description="Pack the tokenized groups into 4096-token sequences and build the shuffled training mix.")
    ap.add_argument("--common-dir", default=f"{DATA_ROOT}/tokenized/common", help="dir with the tokenized monolingual and instruction groups")
    ap.add_argument("--mt-dir", default=f"{DATA_ROOT}/tokenized/{VARIANT}", help="dir with the tokenized mt and mtrev groups")
    ap.add_argument("--out", default=f"{DATA_ROOT}/final/{VARIANT}", help="output dir; writes <out>/training_data")
    ap.add_argument("--seed", type=int, default=int(os.environ.get("PACK_SEED", "42")), help="shuffle seed")
    args = ap.parse_args()

    common_dir = Path(args.common_dir)
    mt_dir = Path(args.mt_dir)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = args.seed

    random.seed(seed)

    print("Building final training mix")
    print(f"  common groups: {common_dir}")
    print(f"  mt groups:     {mt_dir}")
    print(f"  seed:          {seed}")

    groups = {
        "hsb": {
            "datasets": ["mono_hsb_sent", "mono_hsb_doc"],
            "epochs": 4,
        },
        "dsb": {
            "datasets": ["mono_dsb_sent", "mono_dsb_doc"],
            "epochs": 4,
        },
        "mt": {
            "datasets": ["mt"],
            "epochs": 4,
        },
        "mtrev": {
            "datasets": ["mtrev"],
            "epochs": 1,
        },
        "inst": {
            "datasets": ["inst_aya", "inst_magpie", "inst_oasst2", "inst_flan_v2"],
            "epochs": 1,
        },
    }

    print("\nEstimating token counts per group:")
    group_tokens = {}
    for group_name, config in groups.items():
        total = 0
        for ds_name in config["datasets"]:
            t = count_tokens(dataset_path(ds_name, common_dir, mt_dir))
            total += t
        total_with_epochs = total * config["epochs"]
        group_tokens[group_name] = total_with_epochs
        print(f"  {group_name}: ~{total_with_epochs:,} tokens ({config['epochs']}x)")

    total_all = sum(group_tokens.values())
    print(f"\n  Total estimated: ~{total_all:,} tokens")

    print("\n  Mix percentages:")
    for name, tokens in group_tokens.items():
        pct = 100 * tokens / total_all
        print(f"    {name:<10} {pct:>5.1f}%")

    print(f"\nPacking into {SEQ_LEN}-token sequences")

    all_chunk_paths = []
    total_sequences = 0
    chunk_dir = output_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    for group_name, config in groups.items():
        print(f"\nProcessing group: {group_name} ({config['epochs']}x)")

        current_ids = []
        current_mask = []
        packed_ids = []
        packed_masks = []
        group_seqs = 0

        for ds_name in config["datasets"]:
            path = dataset_path(ds_name, common_dir, mt_dir)
            print(f"  Streaming: {ds_name} ({path})")
            for ids, mask in stream_tokens(path, epochs=config["epochs"]):
                current_ids.extend(ids)
                current_mask.extend(mask)

                while len(current_ids) >= SEQ_LEN:
                    packed_ids.append(current_ids[:SEQ_LEN])
                    packed_masks.append(current_mask[:SEQ_LEN])
                    current_ids = current_ids[SEQ_LEN:]
                    current_mask = current_mask[SEQ_LEN:]
                    group_seqs += 1

                    if len(packed_ids) >= CHUNK_SIZE:
                        chunk_path = chunk_dir / f"{group_name}_{len(all_chunk_paths):04d}"
                        ds = Dataset.from_dict({"input_ids": packed_ids, "loss_mask": packed_masks})
                        ds.save_to_disk(str(chunk_path))
                        all_chunk_paths.append(chunk_path)
                        print(f"    Saved chunk: {len(packed_ids):,} seqs (total: {group_seqs:,})")
                        packed_ids = []
                        packed_masks = []
                        del ds

        if len(current_ids) > SEQ_LEN // 10:
            pad_len = SEQ_LEN - len(current_ids)
            current_ids.extend([0] * pad_len)
            current_mask.extend([0] * pad_len)
            packed_ids.append(current_ids[:SEQ_LEN])
            packed_masks.append(current_mask[:SEQ_LEN])
            group_seqs += 1

        if packed_ids:
            chunk_path = chunk_dir / f"{group_name}_{len(all_chunk_paths):04d}"
            ds = Dataset.from_dict({"input_ids": packed_ids, "loss_mask": packed_masks})
            ds.save_to_disk(str(chunk_path))
            all_chunk_paths.append(chunk_path)
            print(f"    Saved final chunk: {len(packed_ids):,} seqs")
            packed_ids = []
            packed_masks = []
            del ds

        total_sequences += group_seqs
        print(f"  Group {group_name}: {group_seqs:,} packed sequences")

    print(f"\nMerging {len(all_chunk_paths)} chunks ({total_sequences:,} total sequences)")

    all_datasets = []
    for cp in all_chunk_paths:
        all_datasets.append(load_from_disk(str(cp)))

    merged = concatenate_datasets(all_datasets)
    del all_datasets

    print(f"Shuffling {len(merged):,} sequences...")
    merged = merged.shuffle(seed=seed)

    out_path = output_dir / "training_data"
    print(f"Saving to {out_path}...")
    merged.save_to_disk(str(out_path))

    total_tokens = len(merged) * SEQ_LEN
    print("\nFinal training data")
    print(f"  Sequences:     {len(merged):,}")
    print(f"  Seq length:    {SEQ_LEN}")
    print(f"  Total tokens:  {total_tokens:,}")
    print(f"  Training steps (batch=128): {len(merged) // 128}")

    shutil.rmtree(str(chunk_dir))
    print(f"\n  Removed chunk directory {chunk_dir}")
    print("  Done")


if __name__ == "__main__":
    main()
