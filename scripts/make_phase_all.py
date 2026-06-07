"""Fig 9 (canonical): gamma-beta phase plane — agentic trajectories vs the
diverse pretraining corpora of reference/_lz_Hinf_all.csv (code / math /
prose-web / mixed), all under one byte-level protocol.

gamma = alpha column of the reference CSV (same LZ oracle); beta from
data/gamma_beta_all.csv (measure_beta_all.py). Fit artifacts (negative or
near-zero beta from non-monotone correlation curves: FineMath,
Pile Uncopyrighted) are excluded and listed in the console output.
"""

import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ref = {(r["dataset"], r["subset_or_config"] or ""): r
       for r in csv.DictReader(open("reference/_lz_Hinf_all.csv"))}
beta = {(r["dataset"], r["subset_or_config"] or ""): float(r["beta"])
        for r in csv.DictReader(open("data/gamma_beta_all.csv"))}
reg = {r["slug"]: r for r in csv.DictReader(open("data/agentic_alpha_hinf.csv"))}
ag_beta = {r["slug"]: float(r["beta"]) for r in
           csv.DictReader(open("data/gamma_beta.csv"))}

CODE_MATH = {"Stack Edu Python", "Python Code (structured code)", "Proof Pile 2"}
MIXED = {"The Pile Deduplicated"}
ARTIFACT = {("FineMath", "finemath-3plus"), ("Pile Uncopyrighted", "")}

AG_LABEL = {
    "toucan-15m-kimi-k2": "Toucan Kimi-K2",
    "jetbrains-swe-test-minus-verified": "JetBrains GPT-5.2",
    "apigen-mt-5k": "APIGen (template)",
    "swe-zero-12m-traj": "SWE-ZERO-12M",
    "weblinx-actions": "WebLINX actions",
    "agentnet-text": "AgentNet annotated",
    "glaive-fc-v2": "glaive-FC",
}

fig, ax = plt.subplots(figsize=(10.5, 7.5))
bb = np.linspace(0.05, 1.6, 300)
gg = np.linspace(0.0, 0.65, 300)
B, G = np.meshgrid(bb, gg)
cs = ax.contour(B, G, G / (2 * B), levels=[0.05, 0.1, 0.14, 0.19, 0.3, 0.5, 0.8],
                colors="gray", linewidths=0.8, alpha=0.6)
ax.clabel(cs, fmt=lambda v: f"$\\alpha_D$={v:g}", fontsize=7)

groups = {
    "prose / web": dict(marker="o", size=55),
    "code / math": dict(marker="s", size=80),
    "mixed (Pile)": dict(marker="^", size=80),
}
sc = None
for key, b in beta.items():
    if key in ARTIFACT or key not in ref:
        print("excluded (artifact/missing):", key)
        continue
    name = key[0]
    grp = ("code / math" if name in CODE_MATH
           else "mixed (Pile)" if name in MIXED else "prose / web")
    g = float(ref[key]["alpha"])
    h = float(ref[key]["h_inf"])
    st = groups[grp]
    sc = ax.scatter([b], [g], c=[h], cmap="viridis", vmin=0, vmax=3,
                    marker=st["marker"], s=st["size"], alpha=0.9,
                    edgecolors="k", linewidths=0.5)
    short = {"Stack Edu Python": "Stack-Edu-Py", "Python Code (structured code)": "PythonCode",
             "Proof Pile 2": {"algebraic-stack": "PP2-algebra", "arxiv": "PP2-arxiv",
                              "open-web-math": "PP2-owmath"}.get(key[1], "PP2"),
             "The Pile Deduplicated": "Pile-dedup", "TinyStories (minimal logic)": "TinyStories",
             "WikiText-103 (high-quality wiki)": "WikiText",
             "BookCorpus (long-form literature)": "BookCorpus",
             "OpenWebText (noisy web)": "OpenWebText"}.get(name)
    if short:
        ax.annotate(short, (b, g), fontsize=7, xytext=(4, 4),
                    textcoords="offset points", alpha=0.85)

# agentic diamonds
for s, lab in AG_LABEL.items():
    b, g, h = ag_beta[s], float(reg[s]["alpha"]), float(reg[s]["h_inf"])
    ax.scatter([b], [g], c=[h], cmap="viridis", vmin=0, vmax=3, marker="D",
               s=170, edgecolors="k", linewidths=1.2)
    ax.annotate(lab, (b, g), fontsize=8, xytext=(6, 6), textcoords="offset points")

cb = fig.colorbar(sc, ax=ax, shrink=0.8)
cb.set_label("H$_\\infty$ (content floor, BPC)")

for name, b, g in [("TinyStories (paper, token-level)", 0.88, 0.34),
                   ("WikiText (paper, token-level)", 0.94, 0.27)]:
    ax.scatter([b], [g], marker="*", s=240, c="crimson", edgecolors="k", zorder=4)
    ax.annotate(name, (b, g), fontsize=7, xytext=(6, -12),
                textcoords="offset points", color="crimson")

ax.axvspan(0.05, 0.52, alpha=0.05, color="tab:blue")
ax.axvspan(0.52, 0.85, alpha=0.05, color="tab:orange")
ax.text(0.28, 0.62, "agentic phase", fontsize=9, ha="center", color="tab:blue")
ax.text(0.68, 0.62, "code / math\n(bridge)", fontsize=9, ha="center", color="tab:orange")
ax.text(1.25, 0.62, "prose / web", fontsize=9, ha="center", color="tab:green")

# legend for marker shapes
for grp, st in groups.items():
    ax.scatter([], [], marker=st["marker"], s=st["size"], c="gray",
               edgecolors="k", label=grp)
ax.scatter([], [], marker="D", s=170, c="gray", edgecolors="k",
           label="agentic trajectories")
ax.legend(fontsize=8.5, loc="lower right")

ax.set_xlabel(r"$\beta$  (byte-level correlation decay, $\|C(n)\|_{op}\propto n^{-\beta}$)")
ax.set_ylabel(r"$\gamma$  (entropy decay; LZ-oracle $\alpha$)")
ax.set_title("Fig 9: gamma-beta phase plane — agentic vs code/math vs prose "
             "(one byte-level protocol)\ncode sits between prose and agentic: "
             "the LRD bridge domain; contours = $\\alpha_D=\\gamma/2\\beta$")
fig.tight_layout()
fig.savefig("figures/fig9_gamma_beta_all.png", dpi=160)
print("wrote figures/fig9_gamma_beta_all.png")
