#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import sacrebleu


def load(path):
    with open(path, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    return [r["clean_hypothesis"] for r in rows], [r["reference"] for r in rows]


def build_metric(name):
    if name == "chrf":
        return sacrebleu.CHRF(char_order=6, word_order=2, beta=2)
    return sacrebleu.BLEU()


def main():
    parser = argparse.ArgumentParser(
        description="Paired bootstrap resampling over two samples_deu-<pair>.jsonl files of the same test set."
    )
    parser.add_argument("--a", required=True, help="samples file of system A")
    parser.add_argument("--b", required=True, help="samples file of system B")
    parser.add_argument("--reps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--metric", default="chrf", choices=["chrf", "bleu"])
    parser.add_argument("--json", default=None, help="optional path for a JSON summary")
    args = parser.parse_args()

    metric = build_metric(args.metric)

    def score(stats):
        return metric._compute_score_from_stats(list(stats.sum(0))).score

    hyp_a, ref_a = load(args.a)
    hyp_b, ref_b = load(args.b)
    assert ref_a == ref_b, f"reference mismatch {args.a} {args.b}"
    A = np.array(metric._extract_corpus_statistics(hyp_a, [ref_a]), dtype=np.float64)
    B = np.array(metric._extract_corpus_statistics(hyp_b, [ref_b]), dtype=np.float64)

    n = len(A)
    score_a = score(A)
    score_b = score(B)
    d0 = score_a - score_b
    rng = np.random.default_rng(args.seed)
    ds = np.empty(args.reps)
    for i in range(args.reps):
        idx = rng.integers(0, n, n)
        ds[i] = score(A[idx]) - score(B[idx])
    pval = float(np.mean(ds <= 0)) if d0 > 0 else float(np.mean(ds >= 0))
    low = float(np.percentile(ds, 2.5))
    high = float(np.percentile(ds, 97.5))

    print(f"A {args.a}: {score_a:.2f}")
    print(f"B {args.b}: {score_b:.2f}")
    print(
        f"{args.metric} A - B: {d0:+.2f}  CI[{low:+.2f},{high:+.2f}]  p={pval:.4f}  "
        f"(n={n}, reps={args.reps}, seed={args.seed})",
        flush=True,
    )

    if args.json:
        summary = {
            "a": args.a,
            "b": args.b,
            "metric": args.metric,
            "n": n,
            "score_a": round(score_a, 2),
            "score_b": round(score_b, 2),
            "delta": round(d0, 2),
            "ci95": [round(low, 2), round(high, 2)],
            "p": round(pval, 4),
            "reps": args.reps,
            "seed": args.seed,
        }
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=1)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
