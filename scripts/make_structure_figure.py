"""structure_density (symbol/serialization heaviness) vs H_inf (content): a near-
flat cloud (Spearman +0.08) -> structure_density is a near-independent axis,
orthogonal to content. Companion to reports/unified_interpretation.md Sec 6.
"""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sd = pd.read_csv("data/structure_density.csv")
m = pd.read_csv("data/merged_analysis.csv")
m = m[m["role"].isin(["TRAIN", "EVAL_TASK", "EVAL_TRAJ"])]
df = m.merge(sd, on="slug", how="inner")

doms = df["domain"].value_counts().index.tolist()
cmap = plt.cm.tab10(np.linspace(0, 1, len(doms)))
cidx = {d: cmap[i] for i, d in enumerate(doms)}

fig, ax = plt.subplots(figsize=(9, 6.5))
for d in doms:
    s = df[df["domain"] == d]
    ax.scatter(s["h_inf"], s["structure_density"], color=cidx[d], s=60,
               edgecolors="k", linewidths=0.3, alpha=0.85, label=f"{d} ({len(s)})")
rho = df["h_inf"].corr(df["structure_density"], method="spearman")
ax.set_xlabel(r"$H_\infty$  (cross-episode content floor, BPC)", fontsize=12)
ax.set_ylabel("structure density  (non-alphabetic char fraction)", fontsize=12)
ax.set_title(f"structure density is a near-independent axis\n"
             f"Spearman(structure, H∞) = {rho:+.2f} — content and serialization "
             f"symbol-heaviness vary independently", fontsize=11.5)
ax.legend(fontsize=8, ncol=2, title="domain")
fig.tight_layout()
fig.savefig("figures/fig_structure_density.png", dpi=160)
print("wrote figures/fig_structure_density.png")
