"""Fig 8: (Hurst, H_inf) two-axis content/form map.

Hurst (R/S on byte n-gram surprisal, data/hurst.csv) measures long-range
*organization* — but repetition is also organization, so form-LRD and
content-LRD are confounded in Hurst alone. H_inf measures the content floor.
The pair separates them.
"""

import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

reg = {r["slug"]: r for r in csv.DictReader(open("data/agentic_alpha_hinf.csv"))}
hu = {r["slug"]: float(r["hurst"]) for r in csv.DictReader(open("data/hurst.csv"))}

POINTS = {
    "toucan-15m-kimi-k2": ("Toucan Kimi-K2", "#2ca02c"),
    "jetbrains-swe-test-minus-verified": ("JetBrains GPT-5.2", "#1f77b4"),
    "swe-zero-12m-traj": ("SWE-ZERO-12M", "#1f77b4"),
    "glaive-fc-v2": ("glaive-FC", "#2ca02c"),
    "weblinx-actions": ("WebLINX actions (human)", "#9467bd"),
    "apigen-mt-5k": ("APIGen (template)", "#7f7f7f"),
    "ko-agent-traj-train": ("Ko-Agent (template)", "#7f7f7f"),
    "aider-polyglot-r2egym32b": ("aider R2EGym-32B (flail)", "#d62728"),
    "agentnet-text": ("AgentNet annotations", "#8c564b"),
}

fig, ax = plt.subplots(figsize=(9, 7))
for slug, (label, col) in POINTS.items():
    x, y = hu[slug], float(reg[slug]["h_inf"])
    ax.scatter([x], [y], c=col, s=140, edgecolors="k", zorder=3)
    dy = 8 if slug != "agentnet-text" else -14
    ax.annotate(label, (x, y), fontsize=8.5, xytext=(6, dy),
                textcoords="offset points")

ax.axhline(0.3, color="k", lw=0.7, ls="--", alpha=0.5)
ax.axvline(0.78, color="k", lw=0.7, ls=":", alpha=0.4)

ax.text(0.985, 2.02, "organized + dense\n(ideal training data)", fontsize=9,
        ha="right", color="#2ca02c")
ax.text(0.62, 2.02, "dense, weak form-dependence\n(human action streams)",
        fontsize=9, ha="left", color="#9467bd")
ax.text(0.985, 0.08, "organized emptiness\n(templates & failure loops:\n"
        "repetition IS long-range dependence)", fontsize=9, ha="right",
        color="#7f7f7f")
ax.text(0.62, 0.08, "churn\n(fast-decorrelating annotations)", fontsize=9,
        ha="left", color="#8c564b")

ax.set_xlabel("Hurst exponent H  (R/S on byte n-gram surprisal — long-range "
              "ORGANIZATION, form+content confounded)")
ax.set_ylabel("H$_\\infty$  (incompressible entropy — CONTENT floor)")
ax.set_xlim(0.60, 1.0)
ax.set_ylim(-0.12, 2.2)
ax.set_title("Finding 18: Hurst alone cannot rate agentic data —\n"
             "templates/failure-loops are as 'long-range dependent' as healthy data; "
             "the (H, H$_\\infty$) pair separates form from content")
fig.tight_layout()
fig.savefig("figures/fig8_hurst_vs_hinf.png", dpi=160)
print("wrote figures/fig8_hurst_vs_hinf.png")
