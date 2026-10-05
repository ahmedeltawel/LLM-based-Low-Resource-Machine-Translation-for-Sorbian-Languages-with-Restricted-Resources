#!/usr/bin/env python3
import argparse
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


LANGUAGES = {"hsb": "Upper Sorbian", "dsb": "Lower Sorbian"}
MAX_NEW_TOKENS = 256


def prompt(source, language):
    system = (
        "You are a professional translator. "
        f"Translate the following text from German to {language}. "
        "Answer with the translated text."
    )
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{source}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def first_line(text):
    return text.split("\n")[0].strip()


def read_lines(path):
    if path == "-":
        sys.stdin.reconfigure(encoding="utf-8")
        return [line.strip() for line in sys.stdin]
    with open(path, encoding="utf-8") as handle:
        return [line.strip() for line in handle]


def write_lines(path, lines):
    if path == "-":
        sys.stdout.reconfigure(encoding="utf-8")
        for line in lines:
            sys.stdout.write(line + "\n")
        sys.stdout.flush()
        return
    with open(path, "w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Translate German text (one sentence per line) into Upper or Lower Sorbian."
    )
    parser.add_argument("--model", required=True, help="model directory or Hugging Face id")
    parser.add_argument("--target", required=True, choices=["hsb", "dsb"])
    parser.add_argument("--input", default="-", help="input file, or - for stdin")
    parser.add_argument("--output", default="-", help="output file, or - for stdout")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = parser.parse_args()

    language = LANGUAGES[args.target]
    sources = read_lines(args.input)
    print(f"{len(sources)} lines to translate into {language}", file=sys.stderr, flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = im_end_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=getattr(torch, args.dtype),
        trust_remote_code=True,
    ).to(device).eval()

    translations = []
    for start in range(0, len(sources), args.batch_size):
        batch_sources = sources[start : start + args.batch_size]
        encoded = tokenizer(
            [prompt(source, language) for source in batch_sources],
            return_tensors="pt",
            padding=True,
        ).to(device)
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                eos_token_id=im_end_id,
                pad_token_id=im_end_id,
            )
        prefix_length = encoded.input_ids.shape[1]
        for row in generated:
            translations.append(
                first_line(tokenizer.decode(row[prefix_length:], skip_special_tokens=True))
            )
        print(f"translated {len(translations)}/{len(sources)}", file=sys.stderr, flush=True)

    write_lines(args.output, translations)
    if args.output != "-":
        print(f"wrote {args.output}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
