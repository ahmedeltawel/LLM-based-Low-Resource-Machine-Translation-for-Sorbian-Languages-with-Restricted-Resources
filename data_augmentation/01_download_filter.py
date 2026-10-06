#!/usr/bin/env python3
import argparse
import gzip
import hashlib
import os
import re
import urllib.request

URLS = [
    "https://data.statmt.org/news-crawl/de/news.2023.de.shuffled.deduped.gz",
]

MAX_KEEP = 1_500_000

URL_RE = re.compile(r"https?://|www\.")


def is_clean(s: str) -> bool:
    s = s.strip()
    n = len(s.split())
    if n < 6 or n > 80:
        return False
    if URL_RE.search(s):
        return False
    if sum(c.isupper() for c in s if c.isalpha()) > 0.5 * sum(1 for c in s if c.isalpha()):
        return False
    alpha = sum(1 for t in s.split() if any(c.isalpha() for c in t))
    if alpha < 0.6 * n:
        return False
    return True


def main():
    data_root = os.environ.get("DATA_ROOT", "data")
    parser = argparse.ArgumentParser(
        description="Download German News Crawl 2023 and keep filtered, unique sentences."
    )
    parser.add_argument(
        "--download-dir",
        default=f"{data_root}/augmentation",
        help="directory for the downloaded .gz file",
    )
    parser.add_argument(
        "--out",
        default=f"{data_root}/augmentation/filtered_de.txt",
        help="output file, one sentence per line",
    )
    args = parser.parse_args()

    os.makedirs(args.download_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    seen = set()
    kept = 0
    with open(args.out, "w") as fo:
        for url in URLS:
            fn = os.path.join(args.download_dir, os.path.basename(url))
            if not os.path.exists(fn):
                print(f"Downloading {url} -> {fn}", flush=True)
                urllib.request.urlretrieve(url, fn)
            print(f"Filtering {fn}", flush=True)
            with gzip.open(fn, "rt", encoding="utf-8", errors="replace") as f:
                for i, line in enumerate(f):
                    if i % 1_000_000 == 0 and i:
                        print(f"  read={i:,} kept={kept:,}", flush=True)
                    line = line.rstrip("\n").strip()
                    if not is_clean(line):
                        continue
                    h = hashlib.md5(line.encode()).digest()
                    if h in seen:
                        continue
                    seen.add(h)
                    fo.write(line + "\n")
                    kept += 1
                    if kept >= MAX_KEEP:
                        print(f"Kept {MAX_KEEP:,} sentences; stopping early.")
                        return
    print(f"Done. kept={kept:,}")


if __name__ == "__main__":
    main()
