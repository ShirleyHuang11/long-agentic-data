"""Candidate axis #4 — near-duplication (the hygiene axis). For each cached corpus,
mean pairwise Jaccard of word-5-gram shingle sets across the cached episodes:
high = episodes are near-copies (redundant corpus), low = episodes are distinct.
Distinct from scaffold_frac (exact shared LINES / boilerplate) — shingles catch
broader near-copy content. Reports its correlation with scaffold (are they the
same thing?) and the content axes. Pure read of samples_cache.
"""
import os, re, glob
from itertools import combinations
import pandas as pd

BREAK = re.compile(r"={3,}\s*(?:EPISODE|EP)\s+BREAK\s*={3,}")
WORD = re.compile(r"\w+")
K = 5

def shingles(text):
    toks = WORD.findall(text.lower())
    return {tuple(toks[i:i+K]) for i in range(len(toks) - K + 1)} if len(toks) >= K else set()

def neardup(path):
    txt = open(path, encoding="utf-8", errors="ignore").read()
    eps = [e for e in BREAK.split(txt) if e.strip()]
    if len(eps) < 2:
        return None
    sh = [shingles(e) for e in eps]
    sims = []
    for a, b in combinations(sh, 2):
        u = len(a | b)
        sims.append(len(a & b) / u if u else 0.0)
    return sum(sims) / len(sims) if sims else None

rows = []
for p in sorted(glob.glob("samples_cache/*.txt")):
    nd = neardup(p)
    if nd is not None:
        rows.append({"slug": os.path.basename(p)[:-4], "neardup": round(nd, 4)})
nd_df = pd.DataFrame(rows)
nd_df.to_csv("data/neardup.csv", index=False)
print(f"measured neardup for {len(nd_df)} corpora -> data/neardup.csv")
print(nd_df["neardup"].describe().round(3).to_string())

m = pd.read_csv("data/merged_analysis.csv")
m = m[m["role"].isin(["TRAIN", "EVAL_TASK", "EVAL_TRAJ"])]
df = m.merge(nd_df, on="slug", how="inner")
sf = pd.read_csv("data/scaffold_frac.csv")
df = df.merge(sf, on="slug", how="left")
print(f"\njoined n={len(df)}")
print("\n=== Spearman(neardup, X) ===")
for c in ["scaffold_frac", "h_inf", "bpc_32768", "alpha", "mean_turns"]:
    print(f"  {c:14s}: {df['neardup'].corr(df[c], method='spearman'):+.2f}")
