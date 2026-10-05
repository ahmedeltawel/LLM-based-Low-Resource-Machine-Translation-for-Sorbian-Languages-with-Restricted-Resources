#!/usr/bin/env python3
import argparse
import os
import sys
import traceback
from pathlib import Path
from transformers import AutoTokenizer
from datasets import Dataset, load_from_disk

DATA_ROOT = os.environ.get("DATA_ROOT", "data")
VARIANT = os.environ.get("VARIANT", "0p25x")

SIMPLE_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{%- if message['role'] == 'system' %}"
    "{{ '<|im_start|>system\n' + message['content'] + '<|im_end|>\n' }}"
    "{%- elif message['role'] == 'user' %}"
    "{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}"
    "{%- elif message['role'] == 'assistant' %}"
    "{{ '<|im_start|>assistant\n' + message['content'] + '<|im_end|>\n' }}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{%- endif %}"
)

INSTRUCTION_SETS = ["aya", "magpie", "oasst2", "flan_v2"]


def load_tokenizer(model_id):
    print(f"Loading tokenizer {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  Special tokens: {tokenizer.all_special_tokens}")
    tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    print("  Chat template set to the plain ChatML format without think tags")
    check = tokenizer.apply_chat_template(
        [{"role": "user", "content": "ping"}, {"role": "assistant", "content": "pong"}],
        tokenize=False, add_generation_prompt=False
    )
    assert "<think>" not in check, f"Template still has <think>! Got: {check!r}"
    print("  Template check OK")
    return tokenizer


def tokenize_monolingual(tokenizer, out_dir, name, filepath):
    print(f"\nTokenizing monolingual: {name}")

    with open(filepath, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    print(f"  Lines: {len(lines):,}")

    examples = []
    total_tokens = 0

    for i, line in enumerate(lines):
        ids = tokenizer.encode(line, add_special_tokens=False)
        if not ids:
            continue
        examples.append({
            "input_ids": ids,
            "loss_mask": [1] * len(ids),
        })
        total_tokens += len(ids)

        if (i + 1) % 500000 == 0:
            print(f"    Processed {i+1:,} lines, {total_tokens:,} tokens")

    print(f"  Total: {len(examples):,} examples, {total_tokens:,} tokens")

    ds = Dataset.from_list(examples)
    out_path = out_dir / name
    ds.save_to_disk(str(out_path))
    print(f"  Saved to: {out_path}")
    return total_tokens


def tokenize_documents(tokenizer, out_dir, name, filepath):
    print(f"\nTokenizing documents: {name}")

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    docs = [d.strip() for d in content.split("\n\n") if d.strip()]
    print(f"  Documents: {len(docs):,}")

    examples = []
    total_tokens = 0

    for i, doc in enumerate(docs):
        ids = tokenizer.encode(doc, add_special_tokens=False)
        if not ids:
            continue
        examples.append({
            "input_ids": ids,
            "loss_mask": [1] * len(ids),
        })
        total_tokens += len(ids)

        if (i + 1) % 20000 == 0:
            print(f"    Processed {i+1:,} docs, {total_tokens:,} tokens")

    print(f"  Total: {len(examples):,} examples, {total_tokens:,} tokens")

    ds = Dataset.from_list(examples)
    out_path = out_dir / name
    ds.save_to_disk(str(out_path))
    print(f"  Saved to: {out_path}")
    return total_tokens


def tokenize_chat_dataset(tokenizer, out_dir, name, dataset_path, msg_key="messages"):
    print(f"\nTokenizing chat: {name}")

    ds = load_from_disk(str(dataset_path))
    print(f"  Examples: {len(ds):,}")
    print(f"  Columns: {ds.column_names}")

    if msg_key not in ds.column_names:
        for col in ["messages", "conversations"]:
            if col in ds.column_names:
                msg_key = col
                break
    if msg_key not in ds.column_names and "inputs" in ds.column_names and "targets" in ds.column_names:
        print("  Dataset has inputs/targets format, converting to messages")
        def _to_msgs(ex):
            return {"messages": [
                {"role": "user", "content": ex["inputs"]},
                {"role": "assistant", "content": ex["targets"]},
            ]}
        ds = ds.map(_to_msgs, num_proc=4)
        msg_key = "messages"
    print(f"  Using message column: {msg_key}")

    examples = []
    total_tokens = 0
    total_loss_tokens = 0

    for i, row in enumerate(ds):
        messages = row[msg_key]

        if not messages or not isinstance(messages[0], dict):
            continue

        if "from" in messages[0] and "value" in messages[0]:
            ROLE_MAP = {"human": "user", "gpt": "assistant", "system": "system",
                        "user": "user", "assistant": "assistant"}
            messages = [
                {"role": ROLE_MAP.get(m["from"], m["from"]), "content": m["value"]}
                for m in messages
            ]

        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)

        if not full_ids:
            continue

        loss_mask = [0] * len(full_ids)

        prefix_messages = []
        for msg in messages:
            prefix_messages.append(msg)
            if msg["role"] == "assistant":
                pre_messages = prefix_messages[:-1]

                pre_text_no_asst = tokenizer.apply_chat_template(
                    pre_messages, tokenize=False, add_generation_prompt=True
                )
                pre_ids = tokenizer.encode(pre_text_no_asst, add_special_tokens=False)

                full_with_asst = tokenizer.apply_chat_template(
                    prefix_messages, tokenize=False, add_generation_prompt=False
                )
                full_with_asst_ids = tokenizer.encode(full_with_asst, add_special_tokens=False)

                start = len(pre_ids)
                end = len(full_with_asst_ids)

                start = min(start, len(full_ids))
                end = min(end, len(full_ids))

                for j in range(start, end):
                    loss_mask[j] = 1

        loss_tokens = sum(loss_mask)
        if loss_tokens == 0:
            continue

        examples.append({
            "input_ids": full_ids,
            "loss_mask": loss_mask,
        })
        total_tokens += len(full_ids)
        total_loss_tokens += loss_tokens

        if (i + 1) % 50000 == 0:
            print(f"    Processed {i+1:,}, tokens: {total_tokens:,}, loss_tokens: {total_loss_tokens:,}")

    print(f"  Total: {len(examples):,} examples")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Loss tokens: {total_loss_tokens:,} ({100*total_loss_tokens/max(total_tokens,1):.1f}%)")

    if not examples:
        raise RuntimeError(f"zero valid examples for {name}")

    out_ds = Dataset.from_list(examples)
    out_path = out_dir / name
    out_ds.save_to_disk(str(out_path))
    print(f"  Saved to: {out_path}")
    return total_tokens, total_loss_tokens


def is_already_tokenized(out_dir, name):
    out_path = out_dir / name
    if (out_path / "dataset_info.json").exists():
        ds = load_from_disk(str(out_path))
        n = len(ds)
        total_tokens = sum(len(row["input_ids"]) for row in ds.select(range(min(100, n))))
        avg_tokens = total_tokens / min(100, n)
        est_total = int(avg_tokens * n)
        print(f"  Skipping {name}: already tokenized ({n:,} examples, ~{est_total:,} tokens)")
        return True, est_total
    return False, 0


def instruction_column(ds_path):
    ds_temp = load_from_disk(str(ds_path))
    msg_col = "messages"
    for col in ["messages", "conversations"]:
        if col in ds_temp.column_names:
            msg_col = col
            break
    del ds_temp
    return msg_col


def tokenize_group(tokenizer, out_dir, name, kind, src):
    if kind == "sent":
        return tokenize_monolingual(tokenizer, out_dir, name, src)
    if kind == "doc":
        return tokenize_documents(tokenizer, out_dir, name, src)
    if kind == "inst":
        tokens, loss = tokenize_chat_dataset(tokenizer, out_dir, name, src, msg_key=instruction_column(src))
        return tokens
    tokens, loss = tokenize_chat_dataset(tokenizer, out_dir, name, src)
    return tokens


def main():
    ap = argparse.ArgumentParser(description="Tokenize the monolingual, instruction and MT groups with loss masks.")
    ap.add_argument("--processed-dir", default=f"{DATA_ROOT}/processed", help="dir with monolingual/ and instructions/")
    ap.add_argument("--mt-dir", default=f"{DATA_ROOT}/clean/{VARIANT}", help="dir with the mt and mtrev chat datasets")
    ap.add_argument("--out-common", default=f"{DATA_ROOT}/tokenized/common", help="output dir for the monolingual and instruction groups")
    ap.add_argument("--out-mt", default=f"{DATA_ROOT}/tokenized/{VARIANT}", help="output dir for the mt and mtrev groups")
    ap.add_argument("--only", choices=["all", "common", "mt"], default="all", help="which groups to tokenize")
    ap.add_argument("--tokenizer", default=os.environ.get("TOKENIZER_MODEL", "Qwen/Qwen3.5-0.8B-Base"), help="tokenizer id or path")
    args = ap.parse_args()

    processed_dir = Path(args.processed_dir)
    mt_dir = Path(args.mt_dir)
    out_common = Path(args.out_common)
    out_mt = Path(args.out_mt)

    groups = []
    if args.only in ("all", "common"):
        for lang in ["hsb", "dsb"]:
            groups.append((f"mono_{lang}_sent", "sent", processed_dir / "monolingual" / lang / f"{lang}_sentences_deduped.txt", out_common))
        for lang in ["hsb", "dsb"]:
            groups.append((f"mono_{lang}_doc", "doc", processed_dir / "monolingual" / lang / f"{lang}_documents_deduped.txt", out_common))
    if args.only in ("all", "mt"):
        for name in ["mt", "mtrev"]:
            groups.append((name, "chat", mt_dir / name, out_mt))
    if args.only in ("all", "common"):
        for ds_name in INSTRUCTION_SETS:
            groups.append((f"inst_{ds_name}", "inst", processed_dir / "instructions" / ds_name, out_common))

    tokenizer = load_tokenizer(args.tokenizer)

    token_counts = {}
    failures = []
    for name, kind, src, out_dir in groups:
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            done, est = is_already_tokenized(out_dir, name)
            if done:
                token_counts[name] = est
                continue
            if not src.exists():
                raise FileNotFoundError(f"input not found: {src}")
            token_counts[name] = tokenize_group(tokenizer, out_dir, name, kind, src)
        except Exception:
            traceback.print_exc()
            print(f"  FAILED: {name}")
            failures.append(name)

    print("\nTokenization summary")
    for name, tokens in token_counts.items():
        print(f"  {name:<25} {tokens:>15,} tokens")
    print(f"  {'TOTAL':<25} {sum(token_counts.values()):>15,} tokens")
    print("  (per-epoch counts; the packing step applies the epoch multipliers)")

    if failures:
        print(f"Failed groups: {', '.join(failures)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
