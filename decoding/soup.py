#!/usr/bin/env python3
import argparse
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file


RUNS_ROOT = os.environ.get("RUNS_ROOT", "runs")
INDEX_NAME = "model.safetensors.index.json"
SINGLE_NAME = "model.safetensors"


def weight_layout(model_dir):
    index_path = os.path.join(model_dir, INDEX_NAME)
    if os.path.exists(index_path):
        with open(index_path, encoding="utf-8") as handle:
            weight_map = json.load(handle)["weight_map"]
        layout = {}
        for key, name in weight_map.items():
            layout.setdefault(name, []).append(key)
        return layout
    return {SINGLE_NAME: None}


def load_weights(model_dir, layout):
    tensors = {}
    for name in layout:
        tensors.update(load_file(os.path.join(model_dir, name)))
    return tensors


def main():
    parser = argparse.ArgumentParser(
        description="Uniform 0.5/0.5 weight average of two checkpoints with identical architecture."
    )
    parser.add_argument(
        "--model-a",
        default=f"{RUNS_ROOT}/qwen35_4b_0p25x/checkpoints/step_002391/model",
        help="first checkpoint; config, generation config and tokenizer files are copied from it",
    )
    parser.add_argument(
        "--model-b",
        default=f"{RUNS_ROOT}/qwen35_4b_base/checkpoints/step_002391/model",
    )
    parser.add_argument("--out", default=f"{RUNS_ROOT}/qwen35_4b_avg/model")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    layout_a = weight_layout(args.model_a)
    layout_b = weight_layout(args.model_b)
    a = load_weights(args.model_a, layout_a)
    b = load_weights(args.model_b, layout_b)
    print(f"model a: {args.model_a} ({len(layout_a)} weight file(s))")
    print(f"model b: {args.model_b} ({len(layout_b)} weight file(s))")
    assert a.keys() == b.keys(), f"key mismatch: {set(a) ^ set(b)}"
    out, nfloat, ndiff, maxd = {}, 0, 0, 0.0
    for k in a:
        assert a[k].shape == b[k].shape and a[k].dtype == b[k].dtype, k
        if a[k].is_floating_point():
            nfloat += 1
            d = (a[k].float() - b[k].float()).abs().max().item()
            maxd = max(maxd, d)
            ndiff += d > 0
            out[k] = ((a[k].float() + b[k].float()) / 2).to(a[k].dtype)
        else:
            assert torch.equal(a[k], b[k]), f"non-float tensor differs: {k}"
            out[k] = a[k]
    print(f"tensors {len(a)}, float {nfloat}, differing {ndiff}, max |a-b| {maxd:.4f}")
    assert ndiff > 0, "the two checkpoints are identical - averaging would be a no-op"

    for name, keys in layout_a.items():
        if keys is None:
            save_file(out, os.path.join(args.out, name), metadata={"format": "pt"})
        else:
            save_file(
                {k: out[k] for k in keys},
                os.path.join(args.out, name),
                metadata={"format": "pt"},
            )
        print(f"wrote {os.path.join(args.out, name)}")
    for f in os.listdir(args.model_a):
        source = os.path.join(args.model_a, f)
        if f not in layout_a and os.path.isfile(source):
            shutil.copy2(source, os.path.join(args.out, f))
            print(f"copied {f}")
    print(f"average written to {args.out}")


if __name__ == "__main__":
    main()
