"""Candidate axis #3 — credit-assignment horizon (the long-horizon-specific axis),
robust proxy: normalized long-range REUSE DISTANCE. For each content word (len>=5)
that recurs within an episode, span = (last_pos - first_pos) / n_tokens; mean over
such words, averaged across cached episodes. High = information is referenced
across the whole trajectory (long horizon); low = references are local. Key test
(brainstorm flag): is this captured by Hurst / alpha (existing LRD stats) or is it
an independent axis? Pure read of samples_cache.
"""
import os, re, glob
import pandas as pd

BREAK = re.compile(r"={3,}\s*(?:EPISODE|EP)\s+BREAK\s*={3,}")
WORD = re.compile(r"[a-zA-Z][a-zA-Z_]{4,}")  # content words, len>=5, skip numbers

def reuse_distance(text):
    toks = [w.lower() for w in WORD.findall(text)]
    n = len(toks)
    if n < 20:
        return None
    pos = {}
    for i, t in enumerate(toks):
        pos.setdefault(t, []).append(i)
    spans = [(p[-1] - p[0]) / n for p in pos.values() if len(p) >= 2]
    return sum(spans) / len(spans) if spans else None

def corpus_horizon(path):
    txt = open(path, encoding="utf-8", errors="ignore").read()
    eps = [e for e in BREAK.split(txt) if e.strip()]
    vals = [reuse_distance(e) for e in eps]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None

rows = []
for p in sorted(glob.glob("samples_cache/*.txt")):
    h = corpus_horizon(p)
    if h is not None:
        rows.append({"slug": os.path.basename(p)[:-4], "reuse_dist": round(h, 4)})
ch = pd.DataFrame(rows)
ch.to_csv("data/credit_horizon.csv", index=False)
print(f"measured reuse_dist for {len(ch)} corpora -> data/credit_horizon.csv")
print(ch["reuse_dist"].describe().round(3).to_string())

m = pd.read_csv("data/merged_analysis.csv")
m = m[m["role"].isin(["TRAIN", "EVAL_TASK", "EVAL_TRAJ"])]
df = m.merge(ch, on="slug", how="inner")
print(f"\njoined n={len(df)}")
print("\n=== Spearman(reuse_dist, X) — independent of content/length? ===")
for c in ["h_inf", "alpha", "bpc_32768", "mean_turns"]:
    print(f"  {c:12s}: {df['reuse_dist'].corr(df[c], method='spearman'):+.2f}")

# KEY TEST: captured by Hurst?
hu = df.dropna(subset=["hurst"])
print(f"\n=== KEY: vs Hurst (n={len(hu)}) ===")
print(f"  Spearman(reuse_dist, Hurst) = {hu['reuse_dist'].corr(hu['hurst'], method='spearman'):+.2f}")
print("  (high |ρ| => Hurst already captures it; ~0 => independent long-horizon axis)")
