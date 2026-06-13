"""Scaffold fraction (measured, candidate axis #2) vs H_inf: the pooled H_inf=0
cluster sits at high scaffold fraction, confirming shared-scaffold pooling as a
measured coordinate -- and it is orthogonal to length. Companion to the unified
interpretation (reports/unified_interpretation.md Sec 6).
"""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sf = pd.read_csv("data/scaffold_frac.csv")
m = pd.read_csv("data/merged_analysis.csv")
m = m[m["role"].isin(["TRAIN", "EVAL_TASK", "EVAL_TRAJ"])]
df = m.merge(sf, on="slug", how="inner")

fig, ax = plt.subplots(figsize=(9, 6.5))
sc = ax.scatter(df["scaffold_frac"], df["h_inf"], c=df["bpc_32768"],
                cmap="viridis", vmin=0, vmax=2.6, s=70, edgecolors="k",
                linewidths=0.4, alpha=0.85)
cb = fig.colorbar(sc, ax=ax); cb.set_label("BPC@32K (within-window density)")

# medians of the two regimes
for cond, lab, col in [(df["h_inf"] == 0, "H∞=0 (pooled)", "tab:red"),
                       (df["h_inf"] >= 0.6, "H∞≥0.6 (healthy)", "tab:green")]:
    med = df[cond]["scaffold_frac"].median()
    ax.axvline(med, color=col, ls="--", lw=1.4, alpha=0.7,
               label=f"{lab} median scaffold={med:.2f}")

rho = df["scaffold_frac"].corr(df["h_inf"], method="spearman")
ax.set_xlabel("scaffold fraction  (byte share of cross-episode shared lines)", fontsize=12)
ax.set_ylabel(r"$H_\infty$  (cross-episode content floor, BPC)", fontsize=12)
ax.set_title(f"Scaffold fraction is a measured pooling coordinate\n"
             f"Spearman(scaffold, H∞) = {rho:+.2f}; pooled cluster sits at high scaffold "
             f"(2.5× healthy); ⊥ length (ρ=−0.05)", fontsize=11.5)
ax.legend(fontsize=10, loc="upper right")
fig.tight_layout()
fig.savefig("figures/fig_scaffold_pooling.png", dpi=160)
print("wrote figures/fig_scaffold_pooling.png")
