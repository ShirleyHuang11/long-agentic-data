"""Consolidated independence summary of ALL measured axes — the unified
interpretation in compact form. Merges the core byte metrics (alpha, H_inf,
BPC@32K, length) with the session's new measured axes (scaffold_frac,
structure_density, neardup), prints the Spearman matrix, runs PCA to count the
genuinely-independent dimensions, and writes a correlation heatmap.
"""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

m = pd.read_csv("data/merged_analysis.csv")
m = m[m["role"].isin(["TRAIN", "EVAL_TASK", "EVAL_TRAJ"])].copy()
m["log_turns"] = np.log10(m["mean_turns"].clip(lower=1))
for f in ["scaffold_frac.csv", "structure_density.csv", "neardup.csv",
          "credit_horizon.csv"]:
    m = m.merge(pd.read_csv(f"data/{f}"), on="slug", how="left")
m = m.drop(columns=[c for c in ["hurst"] if c in m.columns])  # stale 25-row col
m = m.merge(pd.read_csv("data/hurst.csv").drop_duplicates("slug")[["slug", "hurst"]],
            on="slug", how="left")

AX = ["h_inf", "alpha", "bpc_32768", "log_turns",
      "scaffold_frac", "structure_density", "neardup", "reuse_dist", "hurst"]
LBL = {"h_inf": "H∞", "alpha": "α", "bpc_32768": "BPC@32K", "log_turns": "length",
       "scaffold_frac": "scaffold", "structure_density": "structure",
       "neardup": "neardup", "reuse_dist": "horizon", "hurst": "Hurst"}

df = m.dropna(subset=AX)
print(f"n={len(df)} corpora with all {len(AX)} axes measured\n")
C = df[AX].corr(method="spearman")
print("=== Spearman correlation matrix (all measured axes) ===")
print(C.rename(index=LBL, columns=LBL).round(2).to_string())

# PCA on standardized axes
X = StandardScaler().fit_transform(df[AX])
pca = PCA().fit(X)
ev = pca.explained_variance_ratio_
print("\n=== PCA: how many independent dimensions? ===")
print("explained var ratio:", np.round(ev, 3))
print("cumulative:", np.round(np.cumsum(ev), 3))
print(f"dims for 90% var: {int(np.argmax(np.cumsum(ev) >= 0.90) + 1)}")

# heatmap
fig, ax = plt.subplots(figsize=(7.5, 6.5))
Cl = C.rename(index=LBL, columns=LBL)
im = ax.imshow(Cl.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(AX))); ax.set_xticklabels(Cl.columns, rotation=45, ha="right")
ax.set_yticks(range(len(AX))); ax.set_yticklabels(Cl.index)
for i in range(len(AX)):
    for j in range(len(AX)):
        v = Cl.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                color="white" if abs(v) > 0.55 else "black", fontsize=9)
fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman ρ")
ax.set_title(f"Unified frame — all measured axes (n={len(df)})\n"
             "content bundle (H∞/α/BPC) · length ⊥ · structure ⊥ · "
             "redundancy (scaffold≈neardup)", fontsize=10.5)
fig.tight_layout()
fig.savefig("figures/fig_extended_axes_corr.png", dpi=160)
print("\nwrote figures/fig_extended_axes_corr.png")
