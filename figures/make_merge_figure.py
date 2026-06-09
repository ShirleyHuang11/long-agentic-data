"""Centerpiece figure for the merged train+eval analysis paper.

Two panels:
 (a) reference-exact H_inf (content) by generator source, split by role
     (TRAIN vs EVAL_TASK vs EVAL_TRAJ) -- shows content tracks source, not role.
 (b) alpha (pattern/structure) vs H_inf (content), colored by source, marker by
     role -- the pattern/content plane; structure is ~constant while content
     spreads with source.
Reads data/merged_analysis.csv. Canonical content axis = ref-exact H_inf.
"""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
rows = [r for r in csv.DictReader(open(ROOT / "data" / "merged_analysis.csv"))
        if r["role"] != "EXCLUDED"]

SRC_ORDER = ["human_task", "human_demo", "frontier", "synth_task", "mid", "distill"]
SRC_LABEL = {"human_task": "human\ntask", "human_demo": "human\ndemo",
             "frontier": "frontier\nrollout", "synth_task": "synth\ntask",
             "mid": "mid-size\nrollout", "distill": "distilled\nSFT"}
SRC_COLOR = {"human_task": "#1b7837", "human_demo": "#5aae61", "frontier": "#2166ac",
             "synth_task": "#9970ab", "mid": "#d6604d", "distill": "#b2182b"}
ROLE_MARK = {"TRAIN": "o", "EVAL_TASK": "s", "EVAL_TRAJ": "^"}
ROLE_LABEL = {"TRAIN": "train", "EVAL_TASK": "eval/benchmark task", "EVAL_TRAJ": "eval rollout"}

fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 6.2))

# --- panel (a): H_inf by source, jittered strip, split markers by role -------
rng = np.random.default_rng(0)
for i, src in enumerate(SRC_ORDER):
    grp = [r for r in rows if r["source"] == src]
    for role in ROLE_MARK:
        sub = [float(r["h_inf"]) for r in grp if r["role"] == role]
        if not sub:
            continue
        xs = i + rng.uniform(-0.18, 0.18, len(sub))
        axA.scatter(xs, sub, marker=ROLE_MARK[role], s=70, alpha=0.8,
                    facecolor=SRC_COLOR[src], edgecolor="k", linewidth=0.5, zorder=3)
    vals = [float(r["h_inf"]) for r in grp]
    med = np.median(vals)
    axA.plot([i - 0.3, i + 0.3], [med, med], color="k", lw=2.5, zorder=4)
    axA.text(i, -0.13, f"med\n{med:.2f}", ha="center", va="top", fontsize=9)
axA.axhline(0.3, color="gray", ls=":", lw=1)
axA.text(5.45, 0.33, "content\nthreshold 0.3", fontsize=8, color="gray", ha="right")
axA.set_xticks(range(len(SRC_ORDER)))
axA.set_xticklabels([SRC_LABEL[s] for s in SRC_ORDER], fontsize=10)
axA.set_ylabel("reference-exact $H_\\infty$  (content density, bits/char)", fontsize=11)
axA.set_title("(a) Content is set by generator source, not by train/eval role",
              fontsize=12, weight="bold")
axA.set_ylim(-0.35, 2.1)
axA.margins(x=0.05)
mk_handles = [plt.Line2D([], [], marker=m, ls="", mfc="lightgray", mec="k",
              ms=9, label=ROLE_LABEL[r]) for r, m in ROLE_MARK.items()]
axA.legend(handles=mk_handles, loc="upper right", fontsize=9, framealpha=0.9)

# --- panel (b): alpha vs H_inf, color=source, marker=role -------------------
for r in rows:
    axB.scatter(float(r["alpha"]), float(r["h_inf"]), marker=ROLE_MARK[r["role"]],
                s=70, alpha=0.8, facecolor=SRC_COLOR[r["source"]],
                edgecolor="k", linewidth=0.5, zorder=3)
axB.set_xlabel("$\\alpha$  (context-scaling exponent / long-range structure)", fontsize=11)
axB.set_ylabel("reference-exact $H_\\infty$  (content density)", fontsize=11)
axB.set_title("(b) Pattern $\\times$ content plane\n($\\alpha$ ~ constant; $H_\\infty$ spreads with source)",
              fontsize=12, weight="bold")
axB.axhline(0.3, color="gray", ls=":", lw=1)
src_handles = [plt.Line2D([], [], marker="o", ls="", mfc=SRC_COLOR[s], mec="k",
               ms=9, label=SRC_LABEL[s].replace("\n", " ")) for s in SRC_ORDER]
axB.legend(handles=src_handles, loc="upper left", fontsize=8.5, framealpha=0.9,
           title="generator source", title_fontsize=9)
axB.set_ylim(-0.1, 2.1)

fig.suptitle("Merged long-horizon agentic data (n=93): training + evaluation/benchmark corpora",
             fontsize=13.5, weight="bold", y=1.00)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = ROOT / "figures" / "fig_merge_content_source.png"
fig.savefig(out, dpi=170, bbox_inches="tight")
print("wrote", out)
