#!/bin/bash
set -euo pipefail

RAW="${1:-${DATA_ROOT:-data}/raw}"
mkdir -p "$RAW"
cd "$RAW"

GH_WMT22="https://raw.githubusercontent.com/mariondimarco/WMT22_UnsupVeryLowResMT_Data/main"
STATMT20="https://www.statmt.org/wmt20/unsup_and_very_low_res"
STATMT21="https://www.statmt.org/wmt21/unsup_and_very_low_res"
WMT25_REPO="https://github.com/TUM-NLP/llms-limited-resources2025.git"
WMT25_DIR="llms-limited-resources2025"

dl() {
  local url="$1" out="$2"
  echo "  Downloading: $(basename "$out")"
  if wget -q --show-progress -O "$out" "$url"; then
    if [ ! -s "$out" ]; then
      echo "  WARNING: 0-byte file: $out"
    fi
  else
    echo "  FAILED: $url"
  fi
}

echo "Target directory: $(pwd)"

mkdir -p parallel/{de-hsb,de-dsb,dsb-hsb}

echo ""
echo "de-hsb parallel"
dl "$STATMT20/train.hsb-de.hsb.gz" parallel/de-hsb/wmt20_train.hsb.gz
dl "$STATMT20/train.hsb-de.de.gz"  parallel/de-hsb/wmt20_train.de.gz
dl "$GH_WMT22/train2021.hsb-de.hsb.gz" parallel/de-hsb/wmt21_train.hsb.gz
dl "$GH_WMT22/train2021.hsb-de.de.gz"  parallel/de-hsb/wmt21_train.de.gz
dl "$GH_WMT22/HSB-DE_train.tsv.gz" parallel/de-hsb/wmt22_train.tsv.gz

echo ""
echo "de-dsb parallel"
dl "$STATMT21/devel.dsb-de.de"  parallel/de-dsb/wmt21_devel.de
dl "$STATMT21/devel.dsb-de.dsb" parallel/de-dsb/wmt21_devel.dsb
dl "$GH_WMT22/40194_train_dsb_de.de.gz"  parallel/de-dsb/wmt22_train.de.gz
dl "$GH_WMT22/40194_train_dsb_de.dsb.gz" parallel/de-dsb/wmt22_train.dsb.gz

echo ""
echo "dsb-hsb parallel"
dl "$GH_WMT22/train_dsb_hsb_62564.dsb.gz" parallel/dsb-hsb/wmt22_train.dsb.gz
dl "$GH_WMT22/train_dsb_hsb_62564.hsb.gz" parallel/dsb-hsb/wmt22_train.hsb.gz

echo ""
echo "WMT25 shared task data"
if [ ! -d "$WMT25_DIR" ]; then
  git clone --depth 1 "$WMT25_REPO" "$WMT25_DIR"
else
  echo "  WMT25 repo already cloned"
fi

cp -v "$WMT25_DIR/Sorbian/hsb/MT/train.de-hsb.de" "$WMT25_DIR/Sorbian/hsb/MT/train.de-hsb.hsb" parallel/de-hsb/
cp -v "$WMT25_DIR/Sorbian/dsb/MT/train.de-dsb.de" "$WMT25_DIR/Sorbian/dsb/MT/train.de-dsb.dsb" parallel/de-dsb/

mkdir -p devtest

echo ""
echo "Dev/test sets"
dl "$STATMT20/devtest.tar.gz" devtest/wmt20_devtest_hsb-de.tar.gz
dl "$STATMT21/devtest.dsb-de.tgz" devtest/wmt21_devtest_dsb-de.tgz
dl "$GH_WMT22/HSB-DE_dev.tsv.gz" devtest/wmt22_dev_hsb-de.tsv.gz
dl "$GH_WMT22/valid.de.gz"  devtest/wmt22_valid_dsb-de.de.gz
dl "$GH_WMT22/valid.dsb.gz" devtest/wmt22_valid_dsb-de.dsb.gz
dl "$GH_WMT22/devtest_dsb_hsb_2022.tar.gz" devtest/wmt22_devtest_dsb-hsb.tar.gz

cp -v "$WMT25_DIR/Sorbian/hsb/MT/dev.de-hsb.de" "$WMT25_DIR/Sorbian/hsb/MT/dev.de-hsb.hsb" devtest/
cp -v "$WMT25_DIR/Sorbian/dsb/MT/dev.de-dsb.de" "$WMT25_DIR/Sorbian/dsb/MT/dev.de-dsb.dsb" devtest/

mkdir -p monolingual/{hsb,dsb}/wmt

echo ""
echo "hsb monolingual"
dl "$STATMT20/sorbian_institute_monolingual.hsb.gz" monolingual/hsb/wmt/wmt20_sorbian_institute.hsb.gz
dl "$STATMT20/witaj_monolingual.hsb.gz"             monolingual/hsb/wmt/wmt20_witaj.hsb.gz
dl "$STATMT20/web_monolingual.hsb.gz"               monolingual/hsb/wmt/wmt20_web.hsb.gz
dl "$GH_WMT22/HSB_monolingual.txt.gz" monolingual/hsb/wmt/wmt22_monolingual.hsb.gz

echo ""
echo "dsb monolingual"
dl "$GH_WMT22/mono.dsb.gz" monolingual/dsb/wmt/wmt21_mono.dsb.gz
dl "$GH_WMT22/66408_DSB_monolingual.txt.gz"      monolingual/dsb/wmt/wmt22_monolingual.dsb.gz
dl "$GH_WMT22/8815_DSB_wikipedia_2021.txt.gz"    monolingual/dsb/wmt/wmt22_wikipedia.dsb.gz

echo ""
echo "Decompressing .gz files"
find parallel monolingual devtest -name '*.gz' -exec gunzip -kf {} \; 2>/dev/null || true

echo ""
echo "Extracting tar archives"
cd devtest
for f in *.tar.gz *.tgz; do
  [ -f "$f" ] && tar xzf "$f" && echo "  Extracted: $f"
done
cd ..

echo ""
echo "Done: $(pwd)"
