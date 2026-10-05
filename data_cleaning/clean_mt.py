import argparse, csv, os, re, unicodedata
from datasets import load_from_disk

csv.field_size_limit(10**9)

DATA_ROOT=os.environ.get("DATA_ROOT","data")


def norm(s): return re.sub(r"\s+"," ",unicodedata.normalize("NFKC",str(s)).strip().lower())


def get(m,role):
    for t in m:
        if t.get("role")==role: return (t.get("content") or "").strip()


def load_dev_pairs(dev_dir):
    EV={}
    for pair in ("hsb","dsb"):
        p=os.path.join(dev_dir,f"dev.de-{pair}.csv")
        s=set()
        with open(p,encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if "de" in r and pair in r: s.add((norm(r["de"]),norm(r[pair])))
        EV[pair]=s
        print(f"dev pairs {pair}: {len(s):,} ({p})")
    return EV


def clean_forward(ds,EV):
    seen=set(); drop_dev=set(); n_dev=0; n_dup=0; keep_flags=[]
    for ex in ds:
        tl=ex.get("tgt_lang"); sl=ex.get("src_lang")
        if tl not in ("hsb","dsb") or sl!="de":
            keep_flags.append(True); continue
        s,t=get(ex["messages"],"user"),get(ex["messages"],"assistant")
        if not s or not t: keep_flags.append(True); continue
        k=(norm(s),norm(t))
        if k in EV[tl]:
            keep_flags.append(False); n_dev+=1; drop_dev.add(k); continue
        if k in seen:
            keep_flags.append(False); n_dup+=1; continue
        seen.add(k); keep_flags.append(True)
    kept=[i for i,v in enumerate(keep_flags) if v]
    return kept,drop_dev,n_dev,n_dup


def clean_reverse(r,rev):
    seen=set(); n_dev=0; n_dup=0; keep_flags=[]
    for ex in r:
        sl=ex.get("src_lang"); tl=ex.get("tgt_lang")
        if tl!="de" or sl not in ("hsb","dsb"): keep_flags.append(True); continue
        s,t=get(ex["messages"],"user"),get(ex["messages"],"assistant")
        if not s or not t: keep_flags.append(True); continue
        k=(norm(s),norm(t))
        if k in rev: keep_flags.append(False); n_dev+=1; continue
        if k in seen: keep_flags.append(False); n_dup+=1; continue
        seen.add(k); keep_flags.append(True)
    kept=[i for i,v in enumerate(keep_flags) if v]
    return kept,n_dev,n_dup


def main():
    ap=argparse.ArgumentParser(description="Remove development-set overlap and duplicate pairs from the mt and mtrev chat datasets.")
    ap.add_argument("--mt",required=True,help="forward MT dataset (load_from_disk dir), e.g. data/augmentation/mt_aug_0p25x or data/processed/mt/mt")
    ap.add_argument("--mtrev",default=f"{DATA_ROOT}/processed/mt/mtrev",help="reverse MT dataset (load_from_disk dir)")
    ap.add_argument("--dev-dir",default=f"{DATA_ROOT}/eval",help="dir with dev.de-hsb.csv and dev.de-dsb.csv")
    ap.add_argument("--out",required=True,help="output dir; writes <out>/mt and <out>/mtrev")
    args=ap.parse_args()

    EV=load_dev_pairs(args.dev_dir)

    ds=load_from_disk(args.mt)
    print(f"\nmt: {len(ds):,} ({args.mt})")
    kept,drop_dev,n_dev,n_dup=clean_forward(ds,EV)
    print(f"  removed dev overlap {n_dev:,} | duplicates {n_dup:,} | kept {len(kept):,}")
    out_mt=os.path.join(args.out,"mt")
    ds.select(kept).save_to_disk(out_mt)
    print(f"  saved {out_mt}")

    rev={(t,s) for (s,t) in drop_dev}
    r=load_from_disk(args.mtrev)
    print(f"\nmtrev: {len(r):,} ({args.mtrev})")
    kept_rev,n_dev_rev,n_dup_rev=clean_reverse(r,rev)
    print(f"  removed dev overlap {n_dev_rev:,} | duplicates {n_dup_rev:,} | kept {len(kept_rev):,}")
    out_mtrev=os.path.join(args.out,"mtrev")
    r.select(kept_rev).save_to_disk(out_mtrev)
    print(f"  saved {out_mtrev}")


if __name__=="__main__":
    main()
