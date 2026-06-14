"""8-axis kind signatures: z-scored median of each axis per kind. Shows the 4
kinds' full profiles and, in particular, that the two H_inf=0 kinds separate by
scaffold/redundancy (pooled vs non-pooled) -- the new axes resolving the H_inf=0
degeneracy. Companion to unified_interpretation Sec 4 robustness note.
"""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

m = pd.read_csv("data/merged_analysis.csv")
m = m[m["role"].isin(["TRAIN", "EVAL_TASK", "EVAL_TRAJ"])].copy()
m["log_turns"] = np.log10(m["mean_turns"].clip(lower=1))
for f in ["scaffold_frac.csv", "structure_density.csv", "neardup.csv", "credit_horizon.csv"]:
    m = m.merge(pd.read_csv(f"data/{f}"), on="slug", how="left")

AX = ["h_inf", "bpc_32768", "log_turns", "scaffold_frac",
      "structure_density", "neardup", "reuse_dist"]
LBL = ["H∞", "BPC@32K", "length", "scaffold", "structure", "neardup", "horizon"]
df = m.dropna(subset=AX + ["alpha"]).reset_index(drop=True)

Xfull = StandardScaler().fit_transform(df[["alpha"] + AX])
df["kind"] = KMeans(4, n_init=10, random_state=0).fit_predict(Xfull)

# z-score axes for display, then median per kind
Z = pd.DataFrame(StandardScaler().fit_transform(df[AX]), columns=LBL)
Z["kind"] = df["kind"].values
prof = Z.groupby("kind").median()

# name kinds by H_inf + scaffold
raw = df.groupby("kind")[["h_inf", "scaffold_frac", "mean_turns"]].median()
names = {}
for k, r in raw.iterrows():
    if r["h_inf"] < 0.2:
        names[k] = f"H∞=0 pooled\n(scaffold {r['scaffold_frac']:.2f})" if r["scaffold_frac"] > 0.3 \
                   else f"H∞=0 non-pooled\n(scaffold {r['scaffold_frac']:.2f})"
    else:
        names[k] = "long + rich" if r["mean_turns"] > 40 else "short + dense"

fig, ax = plt.subplots(figsize=(9, 5.5))
im = ax.imshow(prof.values, cmap="RdBu_r", vmin=-1.3, vmax=1.3, aspect="auto")
ax.set_xticks(range(len(LBL))); ax.set_xticklabels(LBL, rotation=30, ha="right")
ax.set_yticks(range(4)); ax.set_yticklabels([names[k] for k in prof.index], fontsize=9)
for i in range(4):
    for j in range(len(LBL)):
        ax.text(j, i, f"{prof.values[i,j]:+.1f}", ha="center", va="center",
                color="white" if abs(prof.values[i,j]) > 0.8 else "black", fontsize=8)
fig.colorbar(im, ax=ax, shrink=0.8, label="median (z-scored)")
ax.set_title(f"The 4 kinds in the full 8-axis space (n={len(df)})\n"
             "new axes split H∞=0 into pooled (high scaffold) vs non-pooled — "
             "all kinds cut across train/eval", fontsize=10.5)
fig.tight_layout()
fig.savefig("figures/fig_kinds_profile.png", dpi=160)
print("wrote figures/fig_kinds_profile.png")
print("\nkind names:", {k: v.replace(chr(10), ' ') for k, v in names.items()})
