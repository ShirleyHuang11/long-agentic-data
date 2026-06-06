"""Fig 8: (H_inf, Hurst) two-axis content/form map.

Axis convention (user, 2026-06-06): H_inf (content floor) on x,
Hurst (long-range organization, form+content confounded) on y.
Hurst from R/S on byte n-gram surprisal (data/hurst.csv); repetition is also
organization, so Hurst alone confounds form-LRD with content-LRD — the pair
separates them.
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
    x, y = float(reg[slug]["h_inf"]), hu[slug]
    ax.scatter([x], [y], c=col, s=140, edgecolors="k", zorder=3)
    dx = 6 if slug != "weblinx-actions" else -6
    ha = "left" if slug != "weblinx-actions" else "right"
    ax.annotate(label, (x, y), fontsize=8.5, xytext=(dx, 7), ha=ha,
                textcoords="offset points")

ax.axvline(0.3, color="k", lw=0.7, ls="--", alpha=0.5)
ax.axhline(0.78, color="k", lw=0.7, ls=":", alpha=0.4)

ax.text(2.15, 0.975, "organized + dense\n(ideal training data)", fontsize=9,
        ha="right", color="#2ca02c")
ax.text(2.15, 0.63, "dense, weak form-dependence\n(human action streams)",
        fontsize=9, ha="right", color="#9467bd")
ax.text(-0.07, 0.975, "organized emptiness\n(templates & failure loops:\n"
        "repetition IS long-range dependence)", fontsize=9, ha="left",
        color="#7f7f7f")
ax.text(-0.07, 0.63, "churn\n(fast-decorrelating annotations)", fontsize=9,
        ha="left", color="#8c564b")

ax.set_xlabel("H$_\\infty$  (incompressible entropy — CONTENT floor)")
ax.set_ylabel("Hurst exponent H  (R/S on byte n-gram surprisal — long-range "
              "ORGANIZATION,\nform+content confounded)")
ax.set_xlim(-0.12, 2.2)
ax.set_ylim(0.60, 1.0)
ax.set_title("Finding 18: Hurst alone cannot rate agentic data —\n"
             "templates/failure-loops are as 'long-range dependent' as healthy data; "
             "the (H$_\\infty$, H) pair separates form from content")
fig.tight_layout()
fig.savefig("figures/fig8_hurst_vs_hinf.png", dpi=160)
print("wrote figures/fig8_hurst_vs_hinf.png")
