"""Fig 10: OpenThoughts-Agent reproduction, visualized (3 panels).

A: where the winning recipe sits on the terminal/SWE H_inf landscape.
B: the teacher panel — compression-identical, training-different (2x).
C: finding 20 schematic — data value is student-relative (form vs choices).
"""

import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

reg = {r["slug"]: r for r in csv.DictReader(open("data/agentic_alpha_hinf.csv"))}

fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))

# ---------------- Panel A: landscape ----------------
ax = axes[0]
ENTRIES = [  # (label, slug, group)
    ("Opus 4.8 CLI sessions (n=4)", "opus48-pi-traces", "healthy"),
    ("JetBrains GPT-5.2 (real issues)", "jetbrains-swe-test-minus-verified", "healthy"),
    ("CLI sessions sampler (n=10)", "cli-agent-sessions-sampler", "healthy"),
    ("GLM-4.7 terminus (SWE-Gym)", "dcagent-glm47-terminus2", "healthy"),
    ("SWE-ZERO-12M", "swe-zero-12m-traj", "healthy"),
    ("OT-Agent v1-SFT  <<< WINNING RECIPE", "openthoughts-agent-v1-sft", "ot"),
    ("NL2Bash x GLM-4.6 teacher", "nl2bash-teacher-glm46", "ot"),
    ("finance-terminal a3-RL", "dcagent3-financeagent-a3rl", "template"),
    ("aider-polyglot 7B (flail)", "aider-polyglot-sweagentlm7b", "template"),
    ("Qwen3-32B +100k SFT", "aider-polyglot-qwen32b-ntc100k", "template"),
]
COL = {"healthy": "#2ca02c", "ot": "#d62728", "template": "#7f7f7f"}
ys = np.arange(len(ENTRIES))[::-1]
for y, (label, slug, grp) in zip(ys, ENTRIES):
    h = float(reg[slug]["h_inf"])
    ax.barh(y, max(h, 0.015), color=COL[grp], height=0.65,
            alpha=0.95 if grp == "ot" else 0.8)
    ax.text(max(h, 0.015) + 0.02, y, f"{h:.2f}", va="center", fontsize=8)
ax.set_yticks(ys)
ax.set_yticklabels([e[0] for e in ENTRIES], fontsize=8)
ax.axvline(0.3, color="k", lw=0.7, ls="--", alpha=0.5)
ax.set_xlabel("H$_\\infty$ (content floor)")
ax.set_title("A. The best small-model recipe lives in the\n"
             "template band of the terminal landscape", fontsize=10)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COL.values()]
ax.legend(handles, ["healthy band", "OpenThoughts recipe", "template band"],
          fontsize=7.5, loc="lower right")

# ---------------- Panel B: teacher panel ----------------
ax = axes[1]
TEACH = [("GLM-4.6\n(their winner)", "nl2bash-teacher-glm46"),
         ("Qwen3-Coder\n480B", "nl2bash-teacher-qwen3coder480b"),
         ("MiniMax\nM2.7", "nl2bash-teacher-minimax-m27")]
x = np.arange(len(TEACH))
alphas = [float(reg[s]["alpha"]) for _, s in TEACH]
hs = [float(reg[s]["h_inf"]) for _, s in TEACH]
ax.bar(x - 0.18, alphas, width=0.36, label=r"$\alpha$ (structure)", color="#1f77b4")
ax.bar(x + 0.18, [max(h, 0.004) for h in hs], width=0.36,
       label=r"H$_\infty$ (content)", color="#9467bd")
for xi, a in zip(x, alphas):
    ax.text(xi - 0.18, a + 0.004, f"{a:.3f}", ha="center", fontsize=8)
for xi in x:
    ax.text(xi + 0.18, 0.012, "0.00", ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels([t[0] for t in TEACH], fontsize=8.5)
ax.set_ylim(0, 0.24)
ax.annotate("their training ablation:\nGLM-4.6 teacher $\\approx$ 2$\\times$ downstream\n"
            "— statistically INVISIBLE here",
            xy=(0, 0.14), xytext=(0.85, 0.19), fontsize=9, color="#d62728",
            arrowprops=dict(arrowstyle="->", color="#d62728"))
ax.set_title("B. Same tasks, four teachers: compression ties,\n"
             "training differs 2x — the probe's resolution boundary", fontsize=10)
ax.legend(fontsize=8)

# ---------------- Panel C: finding 20 schematic ----------------
ax = axes[2]
s = np.linspace(0, 1, 200)
form_value = np.exp(-4 * s)               # value of template-band (form) data
choice_value = 1 / (1 + np.exp(-8 * (s - 0.45)))  # value of healthy-band data
ax.plot(s, form_value, lw=2.5, color="#7f7f7f",
        label="template-band data (teaches FORM / echo)")
ax.plot(s, choice_value, lw=2.5, color="#2ca02c",
        label="healthy-band data (teaches CHOICES / H$_\\infty$)")
ax.axvspan(0.0, 0.35, color="#7f7f7f", alpha=0.08)
ax.axvspan(0.6, 1.0, color="#2ca02c", alpha=0.08)
ax.text(0.16, 1.02, "OT-Agent's SFT stage\n(8B model, form deficit)",
        ha="center", fontsize=8)
ax.text(0.8, 1.02, "frontier regime\n(form saturated,\nchoice deficit)",
        ha="center", fontsize=8)
ax.annotate("their RL stage (+1-2%):\nbuying choices with a\n~700-task novelty budget",
            xy=(0.55, 0.72), xytext=(0.06, 0.55), fontsize=8,
            arrowprops=dict(arrowstyle="->", alpha=0.7))
ax.set_xlabel("student capability  (small base model  $\\rightarrow$  frontier)")
ax.set_ylabel("marginal value of the data type  (schematic)")
ax.set_ylim(0, 1.18)
ax.set_title("C. Finding 20 (schematic): data value is student-relative —\n"
             "the same H$_\\infty$=0 corpus is a curriculum or inert", fontsize=10)
ax.legend(fontsize=8, loc="center right")

fig.suptitle("OpenThoughts-Agent reproduction: why a template-band recipe wins at 8B "
             "and why compression can't pick teachers", fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig("figures/fig10_openthoughts.png", dpi=160, bbox_inches="tight")
print("wrote figures/fig10_openthoughts.png")
