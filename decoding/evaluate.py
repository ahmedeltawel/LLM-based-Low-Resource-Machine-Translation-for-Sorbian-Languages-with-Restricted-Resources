#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import time
from pathlib import Path

import sacrebleu
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


LANGUAGES = {"hsb": "Upper Sorbian", "dsb": "Lower Sorbian"}
NRC_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
NRC_FALLBACK = "[invalid]"


def qwen_prompt(source, language):
    system = (
        "You are are a professional translator. Translate the following text "
        f"from German to {language}. Answer with the translated text."
    )
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{source}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def nrc_prompt(source, language, pair):
    user = (
        f"Translate the following German text to {language}. "
        f"Put it in this format <{pair}> {language} translation </{pair}>.\n"
        f"<deu> {source} </deu>"
    )
    return (
        f"<|im_start|>system\n{NRC_SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def extract_tagged(text, pair):
    for stop in (f"</{pair}>", "<|im_end|>"):
        text = text.split(stop)[0]
    matches = re.findall(rf"<{pair}>\s*(.+)", text)
    if matches:
        return matches[0].strip()
    return NRC_FALLBACK


def first_line(text):
    return text.split("\n")[0].strip()


def main():
    parser = argparse.ArgumentParser(
        description="Translate the German side of a WMT25 Sorbian dev CSV and score it with chrF++ and BLEU."
    )
    parser.add_argument("--model", required=True, help="model directory or Hugging Face id")
    parser.add_argument("--pair", required=True, choices=["hsb", "dsb"])
    parser.add_argument(
        "--dev-csv",
        default=None,
        help="CSV with columns de,<pair> (default: $DATA_ROOT/eval/dev.de-<pair>.csv)",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--prompt-format", default="qwen", choices=["qwen", "nrc"])
    args = parser.parse_args()

    pair = args.pair
    language = LANGUAGES[pair]
    csv_path = Path(
        args.dev_csv
        or os.path.join(os.environ.get("DATA_ROOT", "data"), "eval", f"dev.de-{pair}.csv")
    )
    output_dir = Path(args.out_dir)

    with csv_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    sources = [row["de"] for row in rows]
    references = [row[pair] for row in rows]
    print(f"{csv_path}: {len(rows)} sentences", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = im_end_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=getattr(torch, args.dtype),
        trust_remote_code=True,
    ).to("cuda").eval()

    if args.prompt_format == "qwen":
        prompts = [qwen_prompt(source, language) for source in sources]
        order = list(range(len(prompts)))
    else:
        prompts = [nrc_prompt(source, language, pair) for source in sources]
        order = sorted(
            range(len(prompts)),
            key=lambda i: (-len(tokenizer.encode(prompts[i])), prompts[i]),
        )

    if args.prompt_format == "qwen":
        stop_ids, pad_id = im_end_id, im_end_id
    else:
        configured = model.generation_config.eos_token_id
        configured = configured if isinstance(configured, list) else [configured]
        stop_ids = [im_end_id] + [i for i in configured if i is not None and i != im_end_id]
        pad_id = tokenizer.pad_token_id

    hypotheses = [None] * len(prompts)
    batch_size = args.batch_size
    started = time.time()
    for start in range(0, len(order), batch_size):
        batch = order[start : start + batch_size]
        encoded = tokenizer(
            [prompts[i] for i in batch],
            return_tensors="pt",
            padding=True,
        ).to("cuda")
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                num_beams=args.num_beams,
                early_stopping=(args.num_beams > 1),
                eos_token_id=stop_ids,
                pad_token_id=pad_id,
            )
        prefix_length = encoded.input_ids.shape[1]
        for i, row in zip(batch, generated):
            hypotheses[i] = tokenizer.decode(row[prefix_length:], skip_special_tokens=True)

    if args.prompt_format == "qwen":
        clean = [first_line(hypothesis) for hypothesis in hypotheses]
    else:
        clean = [extract_tagged(hypothesis, pair) for hypothesis in hypotheses]
    clean_chrf = sacrebleu.corpus_chrf(
        clean, [references], char_order=6, word_order=2, beta=2
    ).score
    clean_bleu = sacrebleu.corpus_bleu(clean, [references]).score

    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / f"samples_deu-{pair}.jsonl"
    with samples_path.open("w", encoding="utf-8") as handle:
        for index, (source, reference, hypothesis, cleaned) in enumerate(
            zip(sources, references, hypotheses, clean)
        ):
            handle.write(
                json.dumps(
                    {
                        "index": index,
                        "source": source,
                        "reference": reference,
                        "hypothesis": hypothesis,
                        "clean_hypothesis": cleaned,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    result = {
        "pair": pair,
        "n": len(hypotheses),
        "clean_chrfpp": clean_chrf,
        "clean_bleu": clean_bleu,
        "model": args.model,
        "dev_csv": str(csv_path),
        "prompt_format": args.prompt_format,
        "batch_size": batch_size,
        "dtype": args.dtype,
        "num_beams": args.num_beams,
        "max_new_tokens": args.max_new_tokens,
        "elapsed_seconds": time.time() - started,
    }
    metrics_path = output_dir / f"metrics_deu-{pair}.json"
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {samples_path}", flush=True)
    print(f"wrote {metrics_path}", flush=True)
    print(
        f"deu-{pair}: chrF++ {clean_chrf:.2f}  BLEU {clean_bleu:.2f}  n={len(hypotheses)}  "
        f"({result['elapsed_seconds']:.0f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
