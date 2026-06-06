"""Gamma-beta phase diagram (Cagnetta et al. 2026, alpha_D = gamma/(2*beta)).

gamma = registry alpha (zstd proxy for the conditional-entropy decay exponent);
beta  = byte-level two-point correlation decay from data/gamma_beta.csv.
Reference points TinyStories/WikiText are the paper's token-level values —
cross-tokenization comparison is qualitative only.
"""

import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

reg = {r["slug"]: r for r in csv.DictReader(open("data/agentic_alpha_hinf.csv"))}
gb = {r["slug"]: float(r["beta"]) for r in csv.DictReader(open("data/gamma_beta.csv"))}

LABEL = {
    "toucan-15m-kimi-k2": "Toucan Kimi-K2",
    "jetbrains-swe-test-minus-verified": "JetBrains GPT-5.2",
    "apigen-mt-5k": "APIGen-MT (template)",
    "swe-zero-12m-traj": "SWE-ZERO-12M",
    "weblinx-actions": "WebLINX actions (human)",
    "agentnet-text": "AgentNet annotated",
    "glaive-fc-v2": "glaive-FC",
}

fig, ax = plt.subplots(figsize=(9, 7))

# alpha_D = gamma / (2 beta) contours
bb = np.linspace(0.05, 1.45, 300)
gg = np.linspace(0.0, 0.55, 300)
B, G = np.meshgrid(bb, gg)
AD = G / (2 * B)
cs = ax.contour(B, G, AD, levels=[0.05, 0.1, 0.14, 0.19, 0.3, 0.5, 0.8],
                colors="gray", linewidths=0.8, alpha=0.7)
ax.clabel(cs, fmt=lambda v: f"$\\alpha_D$={v:g}", fontsize=7)

# our datasets, colored by H_inf
hs = [float(reg[s]["h_inf"]) for s in LABEL]
sc = ax.scatter([gb[s] for s in LABEL], [float(reg[s]["alpha"]) for s in LABEL],
                c=hs, cmap="viridis", s=130, edgecolors="k", zorder=3,
                vmin=0, vmax=2)
for s, lab in LABEL.items():
    ax.annotate(lab, (gb[s], float(reg[s]["alpha"])), fontsize=8,
                xytext=(7, 5), textcoords="offset points")
cb = fig.colorbar(sc, ax=ax, shrink=0.8)
cb.set_label("H$_\\infty$ (registry)")

# paper reference points (token-level!)
for name, b, g in [("TinyStories (paper)", 0.88, 0.34), ("WikiText (paper)", 0.94, 0.27)]:
    ax.scatter([b], [g], marker="*", s=260, c="crimson", edgecolors="k", zorder=4)
    ax.annotate(name, (b, g), fontsize=8, xytext=(7, -11),
                textcoords="offset points", color="crimson")

ax.axvspan(0.05, 0.55, alpha=0.06, color="tab:blue")
ax.text(0.30, 0.53, "agentic trajectories:\nslow correlation decay\n(long structural memory)",
        fontsize=8, ha="center", color="tab:blue")
ax.text(1.30, 0.50, "fast decorrelation\n(annotation churn)", fontsize=8,
        ha="center", color="tab:green")

ax.set_xlabel(r"$\beta$  (token-token correlation decay,  $\|C(n)\|_{op}\propto n^{-\beta}$)")
ax.set_ylabel(r"$\gamma$  (conditional-entropy decay; registry $\alpha$ as zstd proxy)")
ax.set_title("Gamma-beta phase diagram (Cagnetta et al. '26: $\\alpha_D=\\gamma/2\\beta$)\n"
             "contours = predicted data-limited scaling exponent")
fig.tight_layout()
fig.savefig("figures/fig7_gamma_beta_phase.png", dpi=160)
print("wrote figures/fig7_gamma_beta_phase.png")
for s in LABEL:
    g, b = float(reg[s]["alpha"]), gb[s]
    print(f"{s:38s} gamma={g:.3f} beta={b:.3f} -> alpha_D={g/(2*b):.3f}")
