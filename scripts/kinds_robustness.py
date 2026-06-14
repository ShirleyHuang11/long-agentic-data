"""Robustness of the 4 'kinds' (unified_interpretation Sec 4): do they survive
when we cluster in the FULL 8-axis space instead of just the 4 core metrics?
Compares the 4-metric k-means (the published kinds) to an 8-axis k-means via
Adjusted Rand Index, checks both still cut across role, and profiles each 8-axis
kind by all axes. Also refreshes the centerpiece figure's clustering basis.
"""
import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

m = pd.read_csv("data/merged_analysis.csv")
m = m[m["role"].isin(["TRAIN", "EVAL_TASK", "EVAL_TRAJ"])].copy()
m["log_turns"] = np.log10(m["mean_turns"].clip(lower=1))
for f in ["scaffold_frac.csv", "structure_density.csv", "neardup.csv", "credit_horizon.csv"]:
    m = m.merge(pd.read_csv(f"data/{f}"), on="slug", how="left")

CORE = ["alpha", "h_inf", "bpc_32768", "log_turns"]
FULL = CORE + ["scaffold_frac", "structure_density", "neardup", "reuse_dist"]
df = m.dropna(subset=FULL).reset_index(drop=True)
print(f"n={len(df)} corpora with all 8 axes\n")

km_core = KMeans(4, n_init=10, random_state=0).fit_predict(StandardScaler().fit_transform(df[CORE]))
km_full = KMeans(4, n_init=10, random_state=0).fit_predict(StandardScaler().fit_transform(df[FULL]))
df["core_kind"] = km_core
df["full_kind"] = km_full

ari = adjusted_rand_score(km_core, km_full)
print(f"=== Adjusted Rand Index (4-metric kinds vs 8-axis kinds) = {ari:.2f} ===")
print("   (1.0 = identical partition; ~0 = unrelated; >0.5 = substantially stable)\n")

print("=== cross-tab: 4-metric kind x 8-axis kind ===")
print(pd.crosstab(df["core_kind"], df["full_kind"]))

print("\n=== do 8-axis kinds still cut across role? ===")
print(pd.crosstab(df["full_kind"], df["role"]))

print("\n=== 8-axis kind profiles (medians, all axes) ===")
prof = df.groupby("full_kind")[["h_inf", "bpc_32768", "mean_turns",
       "scaffold_frac", "structure_density", "neardup", "reuse_dist"]].median().round(2)
print(prof.to_string())
