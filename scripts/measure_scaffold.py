"""Measure SCAFFOLD FRACTION for every cached corpus (samples_cache/*.txt): the
byte share of lines that recur across >=2 of the cached episodes. This turns the
'shared-scaffold pooling' cause of H_inf=0 (paper Sec 5.3) from a latent mechanism
into a directly-measured coordinate -- candidate axis #2 from the metric-space
brainstorm. Then test whether it adds an independent dimension to the unified
content x length x density frame, and whether it predicts the H_inf=0 pooled
cluster. No network / GPU -- pure read of the cached first-3-episode samples.
"""
import os, re, glob
import pandas as pd, numpy as np

BREAK = re.compile(r"={3,}\s*(?:EPISODE|EP)\s+BREAK\s*={3,}")


def scaffold_fraction(path):
    txt = open(path, encoding="utf-8", errors="ignore").read()
    eps = [e for e in BREAK.split(txt) if e.strip()]
    if len(eps) < 2:
        return None  # need >=2 episodes to define 'shared across episodes'
    # line -> set of episode indices it appears in (ignore trivial short lines)
    seen = {}
    per_ep_lines = []
    for i, e in enumerate(eps):
        lines = [ln.strip() for ln in e.splitlines() if len(ln.strip()) > 3]
        per_ep_lines.append(lines)
        for ln in set(lines):
            seen.setdefault(ln, set()).add(i)
    shared = {ln for ln, ids in seen.items() if len(ids) >= 2}
    tot = sum(len(ln) for lines in per_ep_lines for ln in lines)
    sca = sum(len(ln) for lines in per_ep_lines for ln in lines if ln in shared)
    return sca / tot if tot else None


rows = []
for p in sorted(glob.glob("samples_cache/*.txt")):
    slug = os.path.basename(p)[:-4]
    sf = scaffold_fraction(p)
    if sf is not None:
        rows.append({"slug": slug, "scaffold_frac": round(sf, 4)})
sf_df = pd.DataFrame(rows)
sf_df.to_csv("data/scaffold_frac.csv", index=False)
print(f"measured scaffold_frac for {len(sf_df)} corpora -> data/scaffold_frac.csv")
print(sf_df["scaffold_frac"].describe().round(3).to_string())

# ---- join + test independence / predictive value ----
m = pd.read_csv("data/merged_analysis.csv")
m = m[m["role"].isin(["TRAIN", "EVAL_TASK", "EVAL_TRAJ"])]
df = m.merge(sf_df, on="slug", how="inner")
print(f"\njoined n={len(df)} active corpora with scaffold_frac")

print("\n=== Spearman(scaffold_frac, X) ===")
for c in ["h_inf", "bpc_32768", "alpha", "mean_turns"]:
    print(f"  {c:12s}: {df['scaffold_frac'].corr(df[c], method='spearman'):+.2f}")

print("\n=== does scaffold_frac separate the H_inf=0 pooled cluster? ===")
pooled = df[df["h_inf"] == 0]
healthy = df[df["h_inf"] >= 0.6]
print(f"  H_inf=0   (n={len(pooled):3d}): scaffold_frac median {pooled['scaffold_frac'].median():.3f}")
print(f"  H_inf>=0.6(n={len(healthy):3d}): scaffold_frac median {healthy['scaffold_frac'].median():.3f}")

# partial: among content-bearing corpora only, is it still informative?
cb = df[df["bpc_32768"] > 1.0]
print(f"\n=== among content-dense (BPC@32K>1) n={len(cb)}: Spearman(scaffold,H_inf) "
      f"= {cb['scaffold_frac'].corr(cb['h_inf'], 'spearman'):+.2f} ===")
