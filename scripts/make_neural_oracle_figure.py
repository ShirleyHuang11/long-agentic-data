"""Figure for the agentic LZ-vs-neural validation (§8, results/neural_oracle_validation.md).

(a) neural bits/token vs the pooled 3-point H∞ (weak, ρ=0.17) overlaid with
    neural vs BPC@2048 (the agreeing finite-context rate, ρ=0.59) — the divergence
    is at the ∞-extrapolation, not the model.
(b) Spearman(neural, LZ measure) across the context-length series: the neural
    oracle matches every finite-context LZ rate (peaking at its own 2048 window)
    and collapses only against the ∞-context H∞.
"""
import csv, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
    "savefig.dpi": 200, "axes.grid": True, "grid.alpha": 0.25, "font.family": "DejaVu Sans"})

nrows = list(csv.DictReader(open("data/neural_oracle_bpc.csv")))
reg = {r["slug"]: r for r in csv.DictReader(open("data/agentic_alpha_hinf.csv"))}
nb = [float(r["neural_bpc"]) for r in nrows]
hinf = [float(r["lz_h_inf"]) for r in nrows]
bpc2k = [float(reg[r["slug"]]["bpc_2048"]) for r in nrows]

def spear(xs, ys):
    n = len(xs)
    def rk(v):
        o = sorted(range(n), key=lambda i: v[i]); r = [0]*n; i = 0
        while i < n:
            j = i
            while j+1 < n and v[o[j+1]] == v[o[i]]: j += 1
            for k in range(i, j+1): r[o[k]] = (i+j)/2+1
            i = j+1
        return r
    rx, ry = rk(xs), rk(ys); mx, my = st.fmean(rx), st.fmean(ry)
    d = (sum((a-mx)**2 for a in rx)*sum((b-my)**2 for b in ry))**.5
    return sum((a-mx)*(b-my) for a, b in zip(rx, ry))/d if d else 0

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.2))

# (a) scatter: neural vs H∞ (diverge) and neural vs BPC@2048 (agree)
axA.scatter(hinf, nb, s=42, c="#d62728", alpha=.7, edgecolors="white", lw=.6,
            label=f"vs pooled H∞ (∞-ctx)  ρ={spear(nb,hinf):.2f}")
axA.scatter(bpc2k, nb, s=42, c="#1f77b4", alpha=.55, marker="^", edgecolors="white", lw=.6,
            label=f"vs BPC@2048 (finite-ctx)  ρ={spear(nb,bpc2k):.2f}")
# annotate the poster case agent-flan (H∞ 0, neural max)
for r, x, y in zip(nrows, hinf, nb):
    if r["slug"] == "agent-flan-all":
        axA.annotate("agent-flan\n(H∞ 0, neural max)", (x, y), fontsize=10, fontweight="bold",
                     color="#b22", xytext=(12, -4), textcoords="offset points",
                     arrowprops=dict(arrowstyle="->", color="#b22", lw=1))
axA.set_xlabel("LZ measure  (H∞ or BPC@2048)"); axA.set_ylabel("neural bits/token (Qwen2.5-0.5B)")
axA.set_title("(a) Neural oracle agrees with finite-context LZ,\nnot the ∞-extrapolated H∞", loc="left")
axA.legend(loc="lower right", framealpha=.95, fontsize=10.5)

# (b) correlation across the context series
labels = ["BPC@128", "BPC@2048", "BPC@32K", "H∞\n(∞-ctx)"]
cols = ["bpc_128", "bpc_2048", "bpc_32768", "h_inf"]
rhos = [spear(nb, [float(reg[r["slug"]][c]) for r in nrows]) for c in cols]
colors = ["#1f77b4", "#1f77b4", "#1f77b4", "#d62728"]
bars = axB.bar(range(4), rhos, color=colors, edgecolor="black", lw=.7, width=.62)
for i, v in enumerate(rhos):
    axB.text(i, v+.015, f"{v:.2f}", ha="center", fontweight="bold", fontsize=12)
axB.set_xticks(range(4)); axB.set_xticklabels(labels)
axB.set_ylabel("Spearman(neural bits/token, LZ measure)"); axB.set_ylim(0, .72)
axB.set_title("(b) Divergence is localized to the ∞-extrapolation\n(neural peaks at its own 2048-tok window)", loc="left")
axB.grid(axis="x", visible=False)

fig.suptitle(f"Agentic LZ-vs-neural validation (n={len(nrows)}): H∞ measures cross-episode incompressibility",
             fontsize=15, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("figures/fig_neural_validation.png")
print("wrote figures/fig_neural_validation.png  rhos:", [round(x, 2) for x in rhos])
