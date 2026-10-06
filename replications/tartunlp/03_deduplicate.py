#!/usr/bin/env python3
import argparse
import os
import unicodedata
from pathlib import Path

from datasets import load_from_disk
from stopes.pipelines.monolingual.utils.text_normalizer import (
    remove_non_printing_char,
    replace_unicode_punct,
)


def normalize(text):
    text = remove_non_printing_char(text)
    text = replace_unicode_punct(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.strip()
    text = " ".join(text.split())
    text = text.lower()
    return text


def load_devtest_parallel_pairs(data_dir):
    devtest_dir = data_dir / "devtest"
    pairs = set()
    single_lines = set()

    if not devtest_dir.exists():
        print("  WARNING: No devtest directory found")
        return pairs, single_lines

    for f in devtest_dir.rglob("*"):
        if f.is_dir() or f.suffix in (".tar", ".tgz", ".gz"):
            continue
        try:
            with open(f, "r", encoding="utf-8") as fh:
                lines = [line.strip() for line in fh if line.strip()]

            if ".tsv" in f.name:
                for line in lines:
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        pairs.add((normalize(parts[0]), normalize(parts[1])))
                        single_lines.add(normalize(parts[0]))
                        single_lines.add(normalize(parts[1]))
            else:
                for line in lines:
                    single_lines.add(normalize(line))
        except (UnicodeDecodeError, IsADirectoryError):
            continue

    print(f"  Loaded {len(pairs)} dev/test parallel pairs")
    print(f"  Loaded {len(single_lines)} unique dev/test lines (for monolingual dedup)")
    return pairs, single_lines


def dedup_monolingual_sentences(lang, excluded_lines, data_dir, output_dir):
    print(f"\nDeduplicating {lang.upper()} sentence-level monolingual data")

    wmt_dir = data_dir / "monolingual" / lang / "wmt"
    out_dir = output_dir / "monolingual" / lang
    out_dir.mkdir(parents=True, exist_ok=True)

    all_lines = []
    sources = {}

    if wmt_dir.exists():
        for f in sorted(wmt_dir.iterdir()):
            if f.suffix in (".gz", ".tar", ".tgz"):
                continue
            if f.is_dir():
                continue
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    lines = [line.strip() for line in fh if line.strip()]
                sources[f.name] = len(lines)
                all_lines.extend(lines)
            except UnicodeDecodeError:
                print(f"  Skipping {f.name} (encoding error)")

    print("  Sources loaded:")
    for name, count in sources.items():
        print(f"    {name}: {count:,} lines")
    print(f"  Total before dedup: {len(all_lines):,}")

    seen = set()
    deduped = []
    removed_devtest = 0
    removed_dup = 0

    for line in all_lines:
        norm = normalize(line)
        if not norm:
            continue
        if norm in excluded_lines:
            removed_devtest += 1
            continue
        if norm in seen:
            removed_dup += 1
            continue
        seen.add(norm)
        deduped.append(line)

    print(f"  Removed (dev/test overlap): {removed_devtest:,}")
    print(f"  Removed (duplicates): {removed_dup:,}")
    print(f"  After dedup: {len(deduped):,}")

    out_file = out_dir / f"{lang}_sentences_deduped.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        for line in deduped:
            f.write(line + "\n")
    print(f"  Saved to: {out_file}")

    word_count = sum(len(line.split()) for line in deduped)
    char_count = sum(len(line) for line in deduped)
    print(f"  Words: {word_count:,}, Chars: {char_count:,}")
    return len(deduped), word_count, char_count


def dedup_monolingual_documents(lang, data_dir, output_dir):
    print(f"\nDeduplicating {lang.upper()} document-level monolingual data")

    out_dir = output_dir / "monolingual" / lang
    out_dir.mkdir(parents=True, exist_ok=True)

    all_docs = []

    for src_name in ["fineweb2", "wikipedia"]:
        src_path = data_dir / "monolingual" / lang / src_name
        if not src_path.exists():
            continue
        try:
            ds = load_from_disk(str(src_path))
            texts = ds["text"]
            texts = [t for t in texts if t and len(t.strip()) > 10]
            print(f"  {src_name}: {len(texts):,} documents")
            all_docs.extend(texts)
        except Exception as e:
            print(f"  {src_name}: ERROR - {e}")

    print(f"  Total before dedup: {len(all_docs):,}")

    seen = set()
    deduped = []
    removed_dup = 0

    for doc in all_docs:
        norm = normalize(doc[:500])
        if not norm:
            continue
        if norm in seen:
            removed_dup += 1
            continue
        seen.add(norm)
        deduped.append(doc)

    print(f"  Removed (duplicates): {removed_dup:,}")
    print(f"  After dedup: {len(deduped):,}")

    out_file = out_dir / f"{lang}_documents_deduped.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        for doc in deduped:
            f.write(doc.strip() + "\n\n")
    print(f"  Saved to: {out_file}")

    word_count = sum(len(doc.split()) for doc in deduped)
    char_count = sum(len(doc) for doc in deduped)
    print(f"  Words: {word_count:,}, Chars: {char_count:,}")
    return len(deduped), word_count, char_count


def dedup_parallel(pair_name, src_lang, tgt_lang, excluded_pairs, excluded_lines, data_dir, output_dir):
    print(f"\nDeduplicating {pair_name} parallel data")

    pair_dir = data_dir / "parallel" / pair_name
    out_dir = output_dir / "parallel" / pair_name
    out_dir.mkdir(parents=True, exist_ok=True)

    all_pairs = []

    for f in sorted(pair_dir.iterdir()):
        if f.suffix in (".gz", ".tar", ".tgz"):
            continue
        if f.is_dir():
            continue

        name = f.name

        if ".tsv" in name:
            with open(f, "r", encoding="utf-8") as fh:
                pairs = []
                for line in fh:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2 and parts[0].strip() and parts[1].strip():
                        pairs.append((parts[0].strip(), parts[1].strip()))
            print(f"  {name}: {len(pairs):,} pairs (TSV)")
            all_pairs.extend(pairs)

        elif name.endswith(f".{src_lang}"):
            expected_tgt_name = name[: -len(src_lang)] + tgt_lang
            tgt_file = pair_dir / expected_tgt_name
            if not tgt_file.exists() or tgt_file == f:
                tgt_file = None

            if tgt_file:
                with open(f, "r", encoding="utf-8") as fh:
                    src_lines = [line.strip() for line in fh]
                with open(tgt_file, "r", encoding="utf-8") as fh:
                    tgt_lines = [line.strip() for line in fh]

                if len(src_lines) == len(tgt_lines):
                    pairs = [(s, t) for s, t in zip(src_lines, tgt_lines) if s and t]
                    print(f"  {name} + {tgt_file.name}: {len(pairs):,} pairs")
                    all_pairs.extend(pairs)
                else:
                    print(f"  WARNING: line mismatch {name}({len(src_lines)}) vs {tgt_file.name}({len(tgt_lines)})")

    print(f"  Total before dedup: {len(all_pairs):,}")

    seen = set()
    deduped = []
    removed_devtest = 0
    removed_dup = 0

    for src, tgt in all_pairs:
        norm_src = normalize(src)
        norm_tgt = normalize(tgt)
        if not norm_src or not norm_tgt:
            continue

        if (norm_src, norm_tgt) in excluded_pairs or (norm_tgt, norm_src) in excluded_pairs:
            removed_devtest += 1
            continue

        if norm_src in excluded_lines and norm_tgt in excluded_lines:
            removed_devtest += 1
            continue

        pair_key = (norm_src, norm_tgt)
        if pair_key in seen:
            removed_dup += 1
            continue
        seen.add(pair_key)
        deduped.append((src, tgt))

    print(f"  Removed (dev/test overlap): {removed_devtest:,}")
    print(f"  Removed (duplicates): {removed_dup:,}")
    print(f"  After dedup: {len(deduped):,}")

    src_file = out_dir / f"{pair_name}.{src_lang}"
    tgt_file = out_dir / f"{pair_name}.{tgt_lang}"
    with open(src_file, "w", encoding="utf-8") as fs, \
         open(tgt_file, "w", encoding="utf-8") as ft:
        for src, tgt in deduped:
            fs.write(src + "\n")
            ft.write(tgt + "\n")
    print(f"  Saved to: {src_file}, {tgt_file}")

    return len(deduped)


def main():
    data_root = os.environ.get("DATA_ROOT", "data")
    parser = argparse.ArgumentParser(
        description="Deduplicate monolingual and parallel data with Stopes normalization and remove dev/test lines"
    )
    parser.add_argument(
        "--raw-dir",
        default=os.path.join(data_root, "raw"),
        help="raw data with devtest, monolingual and parallel subdirectories (default: $DATA_ROOT/raw)",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(data_root, "processed"),
        help="output directory (default: $DATA_ROOT/processed)",
    )
    args = parser.parse_args()

    data_dir = Path(args.raw_dir)
    output_dir = Path(args.out_dir)

    print("Deduplication with Stopes normalization")
    print(f"Raw dir:    {data_dir}")
    print(f"Output dir: {output_dir}")

    print("\nLoading dev/test data for exclusion...")
    excluded_pairs, excluded_lines = load_devtest_parallel_pairs(data_dir)

    hsb_sent, hsb_sent_w, hsb_sent_c = dedup_monolingual_sentences("hsb", excluded_lines, data_dir, output_dir)
    dsb_sent, dsb_sent_w, dsb_sent_c = dedup_monolingual_sentences("dsb", excluded_lines, data_dir, output_dir)

    hsb_doc, hsb_doc_w, hsb_doc_c = dedup_monolingual_documents("hsb", data_dir, output_dir)
    dsb_doc, dsb_doc_w, dsb_doc_c = dedup_monolingual_documents("dsb", data_dir, output_dir)

    de_hsb = dedup_parallel("de-hsb", "de", "hsb", excluded_pairs, excluded_lines, data_dir, output_dir)
    de_dsb = dedup_parallel("de-dsb", "de", "dsb", excluded_pairs, excluded_lines, data_dir, output_dir)
    dsb_hsb = dedup_parallel("dsb-hsb", "dsb", "hsb", excluded_pairs, excluded_lines, data_dir, output_dir)

    print("\nDEDUPLICATION SUMMARY")

    print("\nMonolingual HSB:")
    print(f"  Sentence-level: {hsb_sent:>10,} texts, {hsb_sent_w:>12,} words, {hsb_sent_c:>14,} chars")
    print(f"  Document-level: {hsb_doc:>10,} docs,  {hsb_doc_w:>12,} words, {hsb_doc_c:>14,} chars")
    hsb_total_w = hsb_sent_w + hsb_doc_w
    hsb_total_c = hsb_sent_c + hsb_doc_c
    print(f"  Combined:                    {hsb_total_w:>12,} words, {hsb_total_c:>14,} chars")

    print("\nMonolingual DSB:")
    print(f"  Sentence-level: {dsb_sent:>10,} texts, {dsb_sent_w:>12,} words, {dsb_sent_c:>14,} chars")
    print(f"  Document-level: {dsb_doc:>10,} docs,  {dsb_doc_w:>12,} words, {dsb_doc_c:>14,} chars")
    dsb_total_w = dsb_sent_w + dsb_doc_w
    dsb_total_c = dsb_sent_c + dsb_doc_c
    print(f"  Combined:                    {dsb_total_w:>12,} words, {dsb_total_c:>14,} chars")

    print("\nParallel (after dedup):")
    print(f"  de-hsb:  {de_hsb:>10,}")
    print(f"  de-dsb:  {de_dsb:>10,}")
    print(f"  dsb-hsb: {dsb_hsb:>10,}")


if __name__ == "__main__":
    main()
