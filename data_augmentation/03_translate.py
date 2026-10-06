#!/usr/bin/env python3
import argparse
import csv
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM = (
    "You are a professional translator. Translate the following text "
    "from German to Upper Sorbian. Answer with the translated text."
)


def main():
    data_root = os.environ.get("DATA_ROOT", "data")
    parser = argparse.ArgumentParser(
        description="Translate one shard of the German sample into Upper Sorbian with the teacher model."
    )
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument(
        "--model", default=os.environ.get("TEACHER_MODEL", "tartuNLP/Qwen2.5-3B-Instruct-hsb-dsb")
    )
    parser.add_argument("--src", default=f"{data_root}/augmentation/sample_750k_de.txt")
    parser.add_argument("--out-dir", default=f"{data_root}/augmentation/translated")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = f"{args.out_dir}/shard{args.shard}.csv"

    with open(args.src) as f:
        all_sents = [s.rstrip("\n") for s in f]
    sents = all_sents[args.shard :: args.num_shards]
    print(f"[shard {args.shard}/{args.num_shards}] total={len(sents):,}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model.eval()
    print(f"[shard {args.shard}] model loaded", flush=True)

    eos = [tok.eos_token_id]
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    if im_end and im_end != tok.unk_token_id:
        eos.append(im_end)

    with open(out_path, "w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["de", "hsb"])
        t0 = time.time()
        for i in range(0, len(sents), args.batch):
            batch_de = sents[i : i + args.batch]
            chats = [
                tok.apply_chat_template(
                    [{"role": "system", "content": SYSTEM}, {"role": "user", "content": s}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for s in batch_de
            ]
            inputs = tok(
                chats, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to("cuda:0")
            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    eos_token_id=eos,
                    pad_token_id=tok.pad_token_id,
                )
            gen = out[:, inputs["input_ids"].shape[1] :]
            preds = tok.batch_decode(gen, skip_special_tokens=True)
            for de, hsb in zip(batch_de, preds):
                w.writerow([de, hsb.strip()])
            fo.flush()
            if (i // args.batch) % 5 == 0:
                el = time.time() - t0
                rate = (i + len(batch_de)) / max(el, 1)
                eta = (len(sents) - i - len(batch_de)) / max(rate, 1e-6)
                print(
                    f"[shard {args.shard}] {i+len(batch_de):,}/{len(sents):,}  "
                    f"{rate:.1f} sent/s  eta {eta/60:.1f}min",
                    flush=True,
                )
    print(f"[shard {args.shard}] DONE in {(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
