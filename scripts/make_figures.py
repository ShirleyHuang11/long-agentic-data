"""Figures interpreting the (alpha, H_inf) registry — reads data/agentic_alpha_hinf.csv
and data/seed_sigma.csv, writes PNGs to figures/.

Categories are hand-tagged per slug (interpretive layer, mirrors SAMPLES.md §V).
"""

import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- data
rows = {r["slug"]: r for r in csv.DictReader(open("data/agentic_alpha_hinf.csv"))}
EXCLUDED = {"nemotron-agentic-v1", "rlenv-appworld-train",
            "taubench-sonnet-proxy", "aguvis-s2-androidctl-text"}

CAT = {  # slug -> category (mirrors SAMPLES.md cluster table)
    # healthy SWE trajectories (frontier generators)
    **{s: "SWE traj (frontier)" for s in [
        "swe-rebench-oh-traj", "openhands-feedback", "swe-hero-oh-traj",
        "swe-smith-traj-xml", "swe-zero-oh-traj", "swe-gym-oh-sft-traj",
        "swe-gym-oh-sampled-traj", "swe-gym-oh-verifier-traj",
        "jetbrains-swe-test-minus-verified", "jetbrains-swesmith-subset",
        "dcagent-glm47-terminus2", "swe-zero-12m-traj", "nemotron-rl-swe-pivot"]},
    # mid-size generator runs
    **{s: "mid-size generator" for s in [
        "aider-polyglot-qwen3coder30b", "aider-polyglot-sweagentlm7b",
        "aider-polyglot-r2egym32b", "aider-polyglot-qwen32b-ntc1k",
        "aider-polyglot-qwen32b-ntc100k", "r2e-gym-swe-agent-lm-traj",
        "kwai-klear-mini-swe-66k"]},
    # healthy tool / search trajectories
    **{s: "tool/search traj" for s in [
        "toucan-15m-kimi-k2", "toucan-15m-oss", "toucan-15m-qwen3",
        "toucan-15m-sft", "miroverse-agentic-sft-new", "smolagents-gaia-traces",
        "ii-agent-gaia-traj", "nemotron-sft-v2-search",
        "nemotron-sft-v2-interactive", "glaive-fc-v2", "hermes-fc-multiturn",
        "fireact-multitask", "dtap-claude-opus-46", "dtap-claude-sonnet-45",
        "dtap-gemini-3-pro", "agenttrek", "nnetnav-wa", "nnetnav-live"]},
    # template-degenerate SFT
    **{s: "template SFT" for s in [
        "agentinstruct-all", "agentgym-agenttraj-l", "agent-flan-all",
        "code-act-codeact", "toolbench-toolllama-dfs", "apigen-mt-5k",
        "lumos-ground-iter", "scienceworld-expert-traj",
        "taubench-traces-jkazdan", "nebius-swe-agent-traj",
        "ko-agent-traj-train", "app1-agentic-safety-sft", "rebel-alfworld-sft",
        "factory-agent-task-rollouts", "deep-research-sft-0406",
        "fractal-deepresearch-sft", "nemotron-sft-v2-tool",
        "nemotron-rl-conv-tool-pivot", "agentbank-all"]},
    # derived views
    **{s: "compact action view" for s in [
        "weblinx-actions", "mind2web-actions", "agentnet-actions",
        "rebel-alfworld-actions"]},
    **{s: "full-obs / annotated view" for s in [
        "weblinx-fullobs", "mind2web-fullobs", "agentnet-text"]},
    **{s: "agent-text-only view" for s in [
        "jetbrains-swe-assistant-only", "swe-rebench-oh-assistant-only"]},
    # task corpora (section IV)
    **{s: "task corpus" for s in [
        "swe-bench-verified", "swe-gym-tasks", "swe-rebench-test",
        "r2e-gym-lite-tasks", "gdpval-tasks", "nemotron-rl-injection-v1"]},
}

COLORS = {
    "SWE traj (frontier)": "#1f77b4",
    "tool/search traj": "#2ca02c",
    "mid-size generator": "#d62728",
    "template SFT": "#7f7f7f",
    "compact action view": "#9467bd",
    "full-obs / annotated view": "#8c564b",
    "agent-text-only view": "#17becf",
    "task corpus": "#ff7f0e",
}

data = []
for slug, r in rows.items():
    if slug in EXCLUDED:
        continue
    cat = CAT.get(slug)
    if cat is None:
        print(f"WARN: untagged slug {slug}")
        continue
    data.append(dict(slug=slug, cat=cat, alpha=float(r["alpha"]),
                     h=float(r["h_inf"]), turns=float(r["mean_turns"]),
                     bytes_ep=float(r["mean_doc_bytes"]),
                     b1=float(r["bpc_128"]), b2=float(r["bpc_2048"]),
                     b3=float(r["bpc_32768"])))

# ---------------------------------------------------------- fig 1: signature map
# Axis convention (user, 2026-06-06): H_inf on x, structure exponent on y.
fig, ax = plt.subplots(figsize=(9.5, 7))
for cat, col in COLORS.items():
    pts = [d for d in data if d["cat"] == cat]
    ax.scatter([d["h"] for d in pts], [d["alpha"] for d in pts],
               s=[18 + 14 * (d["bytes_ep"] ** 0.33) for d in pts],
               c=col, alpha=0.75, edgecolors="white", linewidths=0.5, label=cat)
ann = {"weblinx-actions": "WebLINX act", "fireact-multitask": "FireAct",
       "gdpval-tasks": "GDPval", "jetbrains-swe-test-minus-verified": "JetBrains GPT-5.2",
       "smolagents-gaia-traces": "smolagents GAIA", "toucan-15m-kimi-k2": "Toucan Kimi-K2",
       "openhands-feedback": "OH feedback", "swe-zero-12m-traj": "SWE-ZERO-12M",
       "agentnet-actions": "AgentNet act", "agentnet-text": "AgentNet annot.",
       "aider-polyglot-sweagentlm7b": "aider 7B", "rebel-alfworld-actions": "ReBel act",
       "nemotron-rl-conv-tool-pivot": "Nemotron conv-pivot", "swe-bench-verified": "SWE-bench-V"}
for d in data:
    if d["slug"] in ann:
        ax.annotate(ann[d["slug"]], (d["h"], d["alpha"]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
ax.axvline(0.3, color="k", lw=0.6, ls="--", alpha=0.5)
ax.text(0.32, 0.02, "healthy threshold H$_\\infty$=0.3", fontsize=7, alpha=0.7,
        rotation=90)
ax.set_xlabel("H$_\\infty$ (incompressible entropy, BPC) — content floor")
ax.set_ylabel("alpha (context-scaling exponent) — structure")
ax.set_title("Signature map: 72 agentic datasets/views (size ~ bytes/episode)")
ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9)
fig.tight_layout()
fig.savefig(f"{OUT}/fig1_signature_map.png", dpi=160)

# ------------------------------------------------- fig 2: horizon vs density
fig, ax = plt.subplots(figsize=(9, 6.5))
traj = [d for d in data if d["cat"] not in ("task corpus",) and d["turns"] > 1]
for cat, col in COLORS.items():
    pts = [d for d in traj if d["cat"] == cat]
    if not pts:
        continue
    ax.scatter([d["bytes_ep"] for d in pts], [d["h"] for d in pts],
               c=col, s=40, alpha=0.8, edgecolors="white", lw=0.5, label=cat)
for d in traj:
    if d["bytes_ep"] > 1.2e5 or d["slug"] in (
            "openhands-feedback", "jetbrains-swe-test-minus-verified"):
        ax.annotate(ann.get(d["slug"], d["slug"]).replace("aider-polyglot-", "aider "),
                    (d["bytes_ep"], d["h"]), fontsize=7, xytext=(4, 4),
                    textcoords="offset points")
ax.set_xscale("log")
ax.axhline(0.3, color="k", lw=0.6, ls="--", alpha=0.5)
ax.set_xlabel("bytes per episode (log)")
ax.set_ylabel("H$_\\infty$")
ax.set_title("Finding 12: the longest episodes are the emptiest\n"
             "(failure-retry loops inflate horizon at H$_\\infty\\approx$0)")
ax.legend(fontsize=7.5, framealpha=0.9)
fig.tight_layout()
fig.savefig(f"{OUT}/fig2_horizon_vs_density.png", dpi=160)

# ------------------------------------------ fig 3: view decomposition (paired)
PAIRS = [  # (stripped/compact view, merged/full view, label, domain)
    ("mind2web-actions", "mind2web-fullobs", "Mind2Web", "web"),
    ("weblinx-actions", "weblinx-fullobs", "WebLINX", "web"),
    ("agentnet-actions", "agentnet-text", "AgentNet", "desktop GUI"),
    ("rebel-alfworld-actions", "rebel-alfworld-sft", "ReBel-ALFWorld", "embodied"),
    ("jetbrains-swe-assistant-only", "jetbrains-swe-test-minus-verified",
     "JetBrains", "SWE"),
    ("swe-rebench-oh-assistant-only", "swe-rebench-oh-traj",
     "SWE-rebench-OH", "SWE"),
]
fig, ax = plt.subplots(figsize=(8.5, 5.5))
for i, (a, b, label, dom) in enumerate(PAIRS):
    ha, hb = rows[a], rows[b]
    h_strip, h_full = float(ha["h_inf"]), float(hb["h_inf"])
    up = h_strip > h_full
    col = "#2ca02c" if up else "#d62728"
    ax.annotate("", xy=(i, h_strip), xytext=(i, h_full),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=2))
    ax.scatter([i], [h_full], c="#555555", zorder=3, s=45,
               label="full / annotated view" if i == 0 else None)
    ax.scatter([i], [h_strip], c=col, zorder=3, s=45,
               label=None)
    ax.text(i, max(h_strip, h_full) + 0.07, f"{label}\n({dom})",
            ha="center", fontsize=8)
ax.set_xticks([])
ax.set_ylim(-0.12, 2.25)
ax.set_ylabel("H$_\\infty$")
ax.set_title("Findings 15/16: stripping observations/annotations\n"
             "recovers density in web/GUI/embodied — but REMOVES it in SWE")
ax.text(0.02, 0.03, "arrow: full view -> stripped view  "
        "(green = density recovered, red = density lost)",
        transform=ax.transAxes, fontsize=8, alpha=0.8)
fig.tight_layout()
fig.savefig(f"{OUT}/fig3_view_decomposition.png", dpi=160)

# ------------------------------------------------ fig 4: generator spectrum
GEN = [  # (label, slug, tier)
    ("GPT-5.2 mix (real issues)", "jetbrains-swe-test-minus-verified", "frontier"),
    ("GPT-5.2 mix (synth issues)", "jetbrains-swesmith-subset", "frontier"),
    ("gpt-4o (GAIA)", "smolagents-gaia-traces", "frontier"),
    ("Kimi-K2 (Toucan)", "toucan-15m-kimi-k2", "frontier"),
    ("Qwen3 teacher (Toucan)", "toucan-15m-qwen3", "frontier"),
    ("gpt-oss (Toucan)", "toucan-15m-oss", "frontier"),
    ("GLM-4.7 (terminus-2)", "dcagent-glm47-terminus2", "frontier"),
    ("Claude Opus 4.6 (DTap)", "dtap-claude-opus-46", "frontier"),
    ("Claude Sonnet 4.5 (DTap)", "dtap-claude-sonnet-45", "frontier"),
    ("Gemini 3 Pro (DTap)", "dtap-gemini-3-pro", "frontier"),
    ("Qwen3-Coder-30B", "aider-polyglot-qwen3coder30b", "mid-size"),
    ("SWE-agent-LM-7B", "aider-polyglot-sweagentlm7b", "mid-size"),
    ("R2EGym-32B", "aider-polyglot-r2egym32b", "mid-size"),
    ("Qwen3-32B +1k SFT", "aider-polyglot-qwen32b-ntc1k", "mid-size"),
    ("Qwen3-32B +100k SFT", "aider-polyglot-qwen32b-ntc100k", "mid-size"),
    ("SWE-agent-LM-32B (R2E)", "r2e-gym-swe-agent-lm-traj", "mid-size"),
    ("human demos (AgentNet act)", "agentnet-actions", "human"),
    ("human demos (WebLINX act)", "weblinx-actions", "human"),
    ("rule planner (ALFWorld act)", "rebel-alfworld-actions", "planner"),
]
TIER_COL = {"frontier": "#1f77b4", "mid-size": "#d62728",
            "human": "#2ca02c", "planner": "#9467bd"}
fig, ax = plt.subplots(figsize=(8, 7))
ys = range(len(GEN))[::-1]
for y, (label, slug, tier) in zip(ys, GEN):
    h = float(rows[slug]["h_inf"])
    ax.barh(y, h, color=TIER_COL[tier], height=0.62, alpha=0.85)
    ax.text(h + 0.02, y, f"{h:.2f}", va="center", fontsize=8)
ax.set_yticks(list(ys))
ax.set_yticklabels([g[0] for g in GEN], fontsize=8)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in TIER_COL.values()]
ax.legend(handles, TIER_COL.keys(), fontsize=8, loc="lower right")
ax.set_xlabel("H$_\\infty$")
ax.set_title("Generator spectrum (findings 6/11/12/15):\n"
             "frontier 0.6-1.6  >>  mid-size ~0;  human action streams top")
fig.tight_layout()
fig.savefig(f"{OUT}/fig4_generator_spectrum.png", dpi=160)

# ------------------------------------------------------- fig 5: seed sigma
sig = defaultdict(list)
for r in csv.DictReader(open("data/seed_sigma.csv")):
    sig[r["slug"]].append((int(r["seed_offset"]), float(r["h_inf"])))
order = ["toucan-15m-kimi-k2", "glaive-fc-v2", "swe-zero-12m-traj",
         "app1-agentic-safety-sft", "dtap-claude-opus-46",
         "dtap-claude-sonnet-45", "dtap-gemini-3-pro"]
labels = ["Toucan Kimi-K2\n(homogeneous)", "glaive-FC\n(homogeneous)",
          "SWE-ZERO-12M\n(heterogeneous)", "APP1\n(template)",
          "DTap Opus 4.6", "DTap Sonnet 4.5", "DTap Gemini 3 Pro"]
fig, ax = plt.subplots(figsize=(9, 5.5))
for i, slug in enumerate(order):
    pts = sorted(sig.get(slug, []))
    # registry (seed-1) value comes from the main CSV for the DTap slices
    if slug.startswith("dtap"):
        pts = [(0, float(rows[slug]["h_inf"]))] + pts
    hs = [h for _, h in pts]
    ax.plot([i] * len(hs), hs, "o", ms=6, alpha=0.7,
            color="#d62728" if slug.startswith("dtap") else "#1f77b4")
    if slug.startswith("dtap"):
        ax.annotate("", xy=(i, hs[-1]), xytext=(i, hs[0]),
                    arrowprops=dict(arrowstyle="-|>", color="#d62728",
                                    lw=1.5, alpha=0.7))
ax.set_xticks(range(len(order)))
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("H$_\\infty$ per disjoint sample")
ax.set_title("Findings 14/17: sampling stability — homogeneous pipelines tight,\n"
             "heterogeneous swing, DTap deep-tail (repeated scenarios) collapses "
             "all frontier models")
fig.tight_layout()
fig.savefig(f"{OUT}/fig5_seed_sigma.png", dpi=160)

# ----------------------------------------- fig 6: BPC scaling curves (method)
PICK = [("weblinx-actions", "WebLINX actions (H=1.95)"),
        ("gdpval-tasks", "GDPval tasks (H=1.67)"),
        ("toucan-15m-kimi-k2", "Toucan Kimi-K2 (H=1.34)"),
        ("swe-zero-12m-traj", "SWE-ZERO-12M (H=0.80)"),
        ("kwai-klear-mini-swe-66k", "Kwai-Klear 66k (H=0.26)"),
        ("apigen-mt-5k", "APIGen-MT (H=0.00)"),
        ("nemotron-rl-conv-tool-pivot", "Nemotron conv-pivot (H=0.00)")]
NS = [128, 2048, 32768]
fig, ax = plt.subplots(figsize=(8, 6))
for slug, label in PICK:
    r = rows[slug]
    bpcs = [float(r["bpc_128"]), float(r["bpc_2048"]), float(r["bpc_32768"])]
    h = float(r["h_inf"])
    ax.plot(NS, bpcs, "o-", label=label)
    ax.plot([NS[-1], 3e5], [bpcs[-1], h], ":", color=ax.lines[-1].get_color(),
            alpha=0.6)
    ax.plot([3e5], [h], "x", color=ax.lines[-1].get_color())
ax.set_xscale("log")
ax.set_xlabel("compression context n (bytes)")
ax.set_ylabel("bits per character")
ax.set_title("Method: BPC(n) = H$_\\infty$ + c$\\cdot$n$^{-\\alpha}$ — dotted: "
             "extrapolation, x: H$_\\infty$")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{OUT}/fig6_bpc_scaling.png", dpi=160)

print("wrote 6 figures ->", OUT)
