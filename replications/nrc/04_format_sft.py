#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "data"))

SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

LANG_NAME = {"de": "German", "hsb": "Upper Sorbian", "dsb": "Lower Sorbian"}
LANG_TAG  = {"de": "deu",    "hsb": "hsb",           "dsb": "dsb"}


def user_prompt(src_lang: str, tgt_lang: str, src_text: str) -> str:
    src_name, tgt_name = LANG_NAME[src_lang], LANG_NAME[tgt_lang]
    src_tag, tgt_tag = LANG_TAG[src_lang], LANG_TAG[tgt_lang]
    return (
        f"Translate the following {src_name} text to {tgt_name}. "
        f"Put it in this format <{tgt_tag}> {tgt_name} translation </{tgt_tag}>.\n"
        f"<{src_tag}> {src_text} </{src_tag}>"
    )


def assistant_reply(tgt_lang: str, tgt_text: str) -> str:
    tgt_tag = LANG_TAG[tgt_lang]
    return f"<{tgt_tag}> {tgt_text} </{tgt_tag}>"


def emit(src_lang: str, tgt_lang: str, src_text: str, tgt_text: str) -> dict:
    return {
        "conversations": [
            {"from": "human", "value": user_prompt(src_lang, tgt_lang, src_text)},
            {"from": "gpt",   "value": assistant_reply(tgt_lang, tgt_text)},
        ],
        "system": SYSTEM_PROMPT,
    }


def main():
    parser = argparse.ArgumentParser(description="Write the NRC SFT examples (LLaMA-Factory sharegpt) and dataset_info.json.")
    parser.add_argument("--manifest", type=Path, default=DATA_ROOT / "nrc" / "filtered_manifest.json")
    parser.add_argument("--out-dir", type=Path, default=DATA_ROOT / "nrc" / "sft")
    args = parser.parse_args()

    out_dir = args.out_dir
    out_jsonl = out_dir / "train.jsonl"
    out_dsinfo = out_dir / "dataset_info.json"

    manifest = json.loads(args.manifest.read_text())

    out_dir.mkdir(parents=True, exist_ok=True)

    per_corpus = []
    total = 0
    per_direction = {"de-hsb": 0, "de-dsb": 0, "dsb-hsb": 0, "hsb-dsb": 0}

    with out_jsonl.open("w", encoding="utf-8") as fout:
        for entry in manifest["train"]:
            name = entry["name"]
            sl, tl = entry["src_lang"], entry["tgt_lang"]
            src_lines = Path(entry["src_filtered"]).read_text(encoding="utf-8").splitlines()
            tgt_lines = Path(entry["tgt_filtered"]).read_text(encoding="utf-8").splitlines()
            assert len(src_lines) == len(tgt_lines)

            if "de" in (sl, tl):
                if sl == "de":
                    directions = [(sl, tl)]
                else:
                    directions = [(tl, sl)]
            else:
                directions = [(sl, tl), (tl, sl)]

            n_emitted = 0
            for (dsl, dtl) in directions:
                if dsl == sl:
                    a_col, b_col = src_lines, tgt_lines
                else:
                    a_col, b_col = tgt_lines, src_lines
                for a, b in zip(a_col, b_col):
                    fout.write(json.dumps(emit(dsl, dtl, a, b), ensure_ascii=False) + "\n")
                    n_emitted += 1
                    key = f"{dsl}-{dtl}"
                    per_direction[key] = per_direction.get(key, 0) + 1

            per_corpus.append({
                "name": name, "pairs": len(src_lines),
                "directions": ["-".join(d) for d in directions],
                "examples": n_emitted,
            })
            total += n_emitted

    print("\nSFT JSONL format (LLaMA-Factory sharegpt + qwen chatml template)")
    print(f"{'corpus':<36} {'pairs':>8} {'examples':>10}  directions")
    print("-" * 80)
    for r in per_corpus:
        print(f"{r['name']:<36} {r['pairs']:>8} {r['examples']:>10}  {r['directions']}")
    print("-" * 80)
    print(f"{'TOTAL examples':<36} {'':<8} {total:>10}")

    print("\nBy direction:")
    for k, v in sorted(per_direction.items()):
        print(f"  {k:<8} {v:>8,}")

    dataset_info = {
        "nrc_sorbian_mt": {
            "file_name": "train.jsonl",
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "system":   "system",
            },
            "tags": {
                "role_tag":    "from",
                "content_tag": "value",
                "user_tag":    "human",
                "assistant_tag": "gpt",
            },
        }
    }
    out_dsinfo.write_text(json.dumps(dataset_info, indent=2, ensure_ascii=False))
    print("\nWritten:")
    print(f"  {out_jsonl}  ({out_jsonl.stat().st_size / 1e6:.1f} MB)")
    print(f"  {out_dsinfo}")


if __name__ == "__main__":
    main()
