"""Fig 9: gamma-beta phase plane — FineFineWeb natural-language domains vs
agentic trajectories, same byte-level protocol for both populations.

FFW: gamma = alpha column of reference/all-lz_Hinf_ffw.csv (same LZ oracle),
beta = data/gamma_beta_ffw.csv (measure_beta_ffw.py). Agentic: registry alpha
+ data/gamma_beta.csv. Contours: alpha_D = gamma/(2 beta) (Cagnetta et al.).
Paper's token-level TinyStories/WikiText stars shown for reference only.
"""

import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ffw_ref = {r["subset_or_config"]: r for r in
           csv.DictReader(open("reference/all-lz_Hinf_ffw.csv"))}
ffw_beta = {r["subset"]: float(r["beta"]) for r in
            csv.DictReader(open("data/gamma_beta_ffw.csv"))}
reg = {r["slug"]: r for r in csv.DictReader(open("data/agentic_alpha_hinf.csv"))}
ag_beta = {r["slug"]: float(r["beta"]) for r in
           csv.DictReader(open("data/gamma_beta.csv"))}

AG_LABEL = {
    "toucan-15m-kimi-k2": "Toucan Kimi-K2",
    "jetbrains-swe-test-minus-verified": "JetBrains GPT-5.2",
    "apigen-mt-5k": "APIGen (template)",
    "swe-zero-12m-traj": "SWE-ZERO-12M",
    "weblinx-actions": "WebLINX actions",
    "agentnet-text": "AgentNet annotated",
    "glaive-fc-v2": "glaive-FC",
}

fig, ax = plt.subplots(figsize=(10, 7.5))

bb = np.linspace(0.05, 1.6, 300)
gg = np.linspace(0.0, 0.6, 300)
B, G = np.meshgrid(bb, gg)
cs = ax.contour(B, G, G / (2 * B), levels=[0.05, 0.1, 0.14, 0.19, 0.3, 0.5, 0.8],
                colors="gray", linewidths=0.8, alpha=0.6)
ax.clabel(cs, fmt=lambda v: f"$\\alpha_D$={v:g}", fontsize=7)

# FFW cloud
fx = [ffw_beta[s] for s in ffw_beta if s in ffw_ref]
fy = [float(ffw_ref[s]["alpha"]) for s in ffw_beta if s in ffw_ref]
fh = [float(ffw_ref[s]["h_inf"]) for s in ffw_beta if s in ffw_ref]
sc = ax.scatter(fx, fy, c=fh, cmap="viridis", s=36, alpha=0.85, vmin=0, vmax=3,
                marker="o", label="FineFineWeb domains (n=67, web text)")
for s in ("news", "physics", "topicality", "optical_engineering"):
    if s in ffw_beta and s in ffw_ref:
        ax.annotate(s, (ffw_beta[s], float(ffw_ref[s]["alpha"])), fontsize=7,
                    xytext=(4, 4), textcoords="offset points", alpha=0.8)

# agentic points
axx = [ag_beta[s] for s in AG_LABEL]
ayy = [float(reg[s]["alpha"]) for s in AG_LABEL]
ahh = [float(reg[s]["h_inf"]) for s in AG_LABEL]
ax.scatter(axx, ayy, c=ahh, cmap="viridis", s=170, vmin=0, vmax=3, marker="D",
           edgecolors="k", linewidths=1.2, label="agentic trajectories (n=7)")
for s, lab in AG_LABEL.items():
    ax.annotate(lab, (ag_beta[s], float(reg[s]["alpha"])), fontsize=8,
                xytext=(6, 6), textcoords="offset points")

cb = fig.colorbar(sc, ax=ax, shrink=0.8)
cb.set_label("H$_\\infty$ (content floor, BPC)")

for name, b, g in [("TinyStories (paper, token-level)", 0.88, 0.34),
                   ("WikiText (paper, token-level)", 0.94, 0.27)]:
    ax.scatter([b], [g], marker="*", s=240, c="crimson", edgecolors="k", zorder=4)
    ax.annotate(name, (b, g), fontsize=7.5, xytext=(6, -12),
                textcoords="offset points", color="crimson")

ax.axvspan(0.05, 0.55, alpha=0.05, color="tab:blue")
ax.text(0.30, 0.575, "agentic phase:\nslow decorrelation", fontsize=9,
        ha="center", color="tab:blue")
ax.text(1.25, 0.575, "natural-language phase:\nfast decorrelation", fontsize=9,
        ha="center", color="tab:green")

ax.set_xlabel(r"$\beta$  (byte-level correlation decay, $\|C(n)\|_{op}\propto n^{-\beta}$)")
ax.set_ylabel(r"$\gamma$  (entropy decay; LZ-oracle $\alpha$)")
ax.set_title("Fig 9: gamma-beta phase plane — agentic trajectories vs 67 FineFineWeb "
             "domains\n(same byte-level protocol; contours = predicted "
             "data-limited exponent $\\alpha_D=\\gamma/2\\beta$)")
ax.legend(fontsize=8.5, loc="lower right")
fig.tight_layout()
fig.savefig("figures/fig9_gamma_beta_ffw.png", dpi=160)
print("wrote figures/fig9_gamma_beta_ffw.png")
