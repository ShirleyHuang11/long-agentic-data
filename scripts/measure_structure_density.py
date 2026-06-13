"""Candidate axis #1 (reasoning-vs-action), measured ROBUSTLY. Explicit
Thought:/Action: markers are too sparse across the registry (12%/7% of corpora)
for a marker-based ratio, so instead measure a format-free proxy for the
prose-reasoning vs structured-action balance:
  structure_density = non-alphabetic, non-whitespace chars / non-whitespace chars
High in JSON / code / tool serializations (action-heavy), low in natural-language
reasoning prose. Then test: (a) is it independent of the content axes? (b) does it
explain beta (the 'repetition/format' axis -- the original 'reasoning sets beta'
hypothesis)? (c) or is it just a domain proxy? Pure read of samples_cache.
"""
import os, glob
import pandas as pd, numpy as np

def structure_density(path):
    t = open(path, encoding="utf-8", errors="ignore").read()
    ns = [c for c in t if not c.isspace()]
    if not ns:
        return None
    nonalpha = sum(1 for c in ns if not c.isalpha())
    return nonalpha / len(ns)

rows = []
for p in sorted(glob.glob("samples_cache/*.txt")):
    sd = structure_density(p)
    if sd is not None:
        rows.append({"slug": os.path.basename(p)[:-4], "structure_density": round(sd, 4)})
sd_df = pd.DataFrame(rows)
sd_df.to_csv("data/structure_density.csv", index=False)
print(f"measured structure_density for {len(sd_df)} corpora -> data/structure_density.csv")
print(sd_df["structure_density"].describe().round(3).to_string())

m = pd.read_csv("data/merged_analysis.csv")
m = m[m["role"].isin(["TRAIN", "EVAL_TASK", "EVAL_TRAJ"])]
df = m.merge(sd_df, on="slug", how="inner")
print(f"\njoined n={len(df)} active corpora")

print("\n=== (a) Spearman(structure_density, X) — independence ===")
for c in ["h_inf", "bpc_32768", "alpha", "mean_turns"]:
    print(f"  {c:12s}: {df['structure_density'].corr(df[c], method='spearman'):+.2f}")

# (b) does it explain beta?
beta = pd.read_csv("data/gamma_beta.csv")
bcol = "beta" if "beta" in beta.columns else beta.columns[-1]
beta = beta.rename(columns={beta.columns[0]: "slug"})[["slug", bcol]].dropna()
db = df.merge(beta, on="slug", how="inner")
print(f"\n=== (b) vs beta (n={len(db)}): Spearman(structure_density, beta) = "
      f"{db['structure_density'].corr(db[bcol], method='spearman'):+.2f} ===")

# (c) domain proxy?
def eta2(v, lab):
    g = v.mean(); tot = ((v-g)**2).sum()
    bet = sum(len(x)*(x.mean()-g)**2 for _, x in v.groupby(lab))
    return bet/tot
print("\n=== (c) eta^2(structure_density | label) — is it just domain? ===")
for c in ["role", "domain", "source"]:
    print(f"  {c:8s}: {eta2(df['structure_density'], df[c]):.3f}")
print("\nstructure_density by domain (median):")
print(df.groupby('domain')['structure_density'].median().sort_values(ascending=False).round(3).to_string())
