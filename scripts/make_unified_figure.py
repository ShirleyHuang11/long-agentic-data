"""Centerpiece for the unified (role-independent) interpretation.
Left: the corpus continuum in (content richness H_inf) x (length, turns) space,
marker = role (train/eval) -> roles are interleaved, not separated; color =
unsupervised cluster -> the real 'kinds' cut across role.
Right: eta^2 of each metric by role/domain/source -> role is the weakest
organizer of every metric except turn-count.
"""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv("data/merged_analysis.csv")
df = df[df["role"].isin(["TRAIN", "EVAL_TASK", "EVAL_TRAJ"])].reset_index(drop=True)
df["log_turns"] = np.log10(df["mean_turns"].clip(lower=1))
FEATS = ["alpha", "h_inf", "bpc_32768", "log_turns"]
X = StandardScaler().fit_transform(df[FEATS])
df["cluster"] = KMeans(n_clusters=4, n_init=10, random_state=0).fit_predict(X)

# name clusters by their median signature
cmed = df.groupby("cluster")[["h_inf", "mean_turns", "bpc_32768"]].median()
cname = {}
for k, r in cmed.iterrows():
    if r["h_inf"] < 0.2:
        cname[k] = "pooled / degenerate" if r["bpc_32768"] > 1.2 else "low-density"
    else:
        cname[k] = "long + content-rich" if r["mean_turns"] > 20 else "short + dense"

def eta2(v, lab):
    g = v.mean(); tot = ((v-g)**2).sum()
    bet = sum(len(x)*(x.mean()-g)**2 for _, x in v.groupby(lab))
    return bet/tot

fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6.2),
                               gridspec_kw={"width_ratios": [1.45, 1]})

# ---- Left: continuum ----
markers = {"TRAIN": "o", "EVAL_TASK": "s", "EVAL_TRAJ": "^"}
colors = plt.cm.viridis(np.linspace(0.15, 0.9, 4))
for cl in sorted(df["cluster"].unique()):
    for role, mk in markers.items():
        s = df[(df["cluster"] == cl) & (df["role"] == role)]
        axL.scatter(s["h_inf"], s["mean_turns"], marker=mk, s=70,
                    color=colors[cl], edgecolors="k", linewidths=0.4, alpha=0.85)
axL.set_yscale("log")
axL.set_xlabel(r"$H_\infty$  (cross-episode content floor, BPC)", fontsize=12)
axL.set_ylabel("mean turns / episode (length)", fontsize=12)
axL.set_title("The corpus continuum — independent of train/eval\n"
              "marker = role (interleaved); color = unsupervised kind (cuts across role)",
              fontsize=11.5)
# legends
from matplotlib.lines import Line2D
role_h = [Line2D([], [], marker=m, color="gray", ls="", mec="k", ms=9,
                 label=r) for r, m in markers.items()]
clus_h = [Line2D([], [], marker="o", color=colors[c], ls="", mec="k", ms=9,
                 label=cname[c]) for c in sorted(cname)]
leg1 = axL.legend(handles=role_h, title="role", loc="upper right", fontsize=9)
axL.add_artist(leg1)
axL.legend(handles=clus_h, title="unsupervised kind", loc="lower right", fontsize=9)

# ---- Right: eta^2 ----
metrics = [("alpha", r"$\alpha$"), ("h_inf", r"$H_\infty$"),
           ("bpc_32768", "BPC@32K"), ("log_turns", "length")]
labels = ["role", "domain", "source"]
bar_c = {"role": "tab:red", "domain": "tab:blue", "source": "tab:green"}
x = np.arange(len(metrics)); w = 0.26
for i, lab in enumerate(labels):
    vals = [eta2(df[m], df[lab]) for m, _ in metrics]
    axR.bar(x + (i-1)*w, vals, w, label=lab, color=bar_c[lab], edgecolor="k", lw=0.4)
axR.set_xticks(x); axR.set_xticklabels([n for _, n in metrics], fontsize=11)
axR.set_ylabel(r"$\eta^2$  (variance explained)", fontsize=12)
axR.set_title("Which label organizes each metric?\n"
              "role (train/eval) is weakest everywhere except length", fontsize=11.5)
axR.legend(title="label", fontsize=10)
axR.axhline(0, color="k", lw=0.5)

fig.tight_layout()
fig.savefig("figures/fig_unified_interpretation.png", dpi=160)
print("wrote figures/fig_unified_interpretation.png")
print("cluster names:", cname)
