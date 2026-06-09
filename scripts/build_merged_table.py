"""Build the merged train+eval/benchmark analysis table for the paper.

Classifies every registry row by:
  role   : TRAIN | EVAL_TASK | EVAL_TRAJ | EXCLUDED  (fit-artifact rows)
  domain : swe | web | gui | tool | search | terminal | safety | embodied | mixed
  source : human_task | human_demo | synth_task | frontier | mid | distill

Joins reference-exact (alpha, H_inf) + BPC@32K from agentic_alpha_hinf.csv,
beta from gamma_beta_all.csv (corpus-level, agentic subset only), and Hurst
from hurst.csv where measured. Emits data/merged_analysis.csv and prints
aggregate stats by role and source for the paper.

Canonical content metric = reference-exact clamped H_inf (data_format.md,
validated LZ<->neural Spearman 0.97). BPC@32K is the supplementary directly
measured companion. No recomputation of any score here -- pure join + classify.
"""
import csv
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "agentic_alpha_hinf.csv"
OUT = ROOT / "data" / "merged_analysis.csv"

# slug -> (role, domain, source). EXCLUDED rows are 3-point fit artifacts
# (alpha<0 or H_inf>>1) plus the single-episode tau-bench proxy; they are kept
# in the registry CSV for transparency but dropped from all analysis.
CLASS = {
    # --- benchmark task/problem corpora (what an agent is scored against) ---
    "swe-bench-verified":      ("EVAL_TASK", "swe",    "human_task"),
    "swe-rebench-test":        ("EVAL_TASK", "swe",    "human_task"),
    "swe-gym-tasks":           ("EVAL_TASK", "swe",    "human_task"),
    "r2e-gym-lite-tasks":      ("EVAL_TASK", "swe",    "synth_task"),
    "gdpval-tasks":            ("EVAL_TASK", "mixed",  "human_task"),
    "nemotron-rl-injection-v1":("EVAL_TASK", "safety", "synth_task"),
    "inferredbugs-tasks-clean":("EVAL_TASK", "swe",    "human_task"),
    "swe-factory-gym-tasks":   ("EVAL_TASK", "swe",    "human_task"),  # multilingual SWE
    # --- agent rollouts collected ON a benchmark (for evaluation/analysis) ---
    "ii-agent-gaia-traj":      ("EVAL_TRAJ", "search", "frontier"),
    "smolagents-gaia-traces":  ("EVAL_TRAJ", "search", "frontier"),
    "dtap-claude-opus-46":     ("EVAL_TRAJ", "safety", "frontier"),
    "dtap-claude-sonnet-45":   ("EVAL_TRAJ", "safety", "frontier"),
    "dtap-gemini-3-pro":       ("EVAL_TRAJ", "safety", "frontier"),
    "aider-polyglot-qwen3coder30b": ("EVAL_TRAJ", "swe", "mid"),
    "aider-polyglot-sweagentlm7b":  ("EVAL_TRAJ", "swe", "mid"),
    "aider-polyglot-r2egym32b":     ("EVAL_TRAJ", "swe", "mid"),
    "aider-polyglot-qwen32b-ntc1k": ("EVAL_TRAJ", "swe", "mid"),
    "aider-polyglot-qwen32b-ntc100k":("EVAL_TRAJ","swe", "mid"),
    "taubench-traces-jkazdan": ("EVAL_TRAJ", "tool",   "distill"),
    # iter 69: benchmark eval rollouts (mid-size generators)
    "coderforge-32b-swebench-verified-eval": ("EVAL_TRAJ", "swe",      "mid"),
    "terminal-bench2-qwen3-32b":             ("EVAL_TRAJ", "terminal", "mid"),
    "gaia127-r2egym-32b-eval":               ("EVAL_TRAJ", "search",   "mid"),
    # iter 70: frontier Terminal-Bench-2 rollouts (vs iter-69 mid on same bench)
    "terminal-bench2-gpt5":                  ("EVAL_TRAJ", "terminal", "frontier"),
    "terminal-bench2-claude-sonnet45":       ("EVAL_TRAJ", "terminal", "frontier"),
    # iter 71: frontier SWE-bench-Verified rollout (mini-swe-agent harness)
    "swebench-verified-claude-sonnet45-eval":("EVAL_TRAJ", "swe",      "frontier"),
    # iter 72: extend terminus-2 capability ladder (small models)
    "terminal-bench2-gpt5nano":              ("EVAL_TRAJ", "terminal", "mid"),
    "terminal-bench2-claude-haiku45":        ("EVAL_TRAJ", "terminal", "mid"),
    # iter 73: frontier tool-use eval rollout (tau-bench, shared policy prompt)
    "taubench-deepseek-r1-eval":             ("EVAL_TRAJ", "tool",     "frontier"),
    # iter 113: tau2-bench (successor to tau-bench) frontier vs mid
    "tau2-airline-gpt51codex-eval":          ("EVAL_TRAJ", "tool",     "frontier"),
    "tau2-retail-qwen35-9b-eval":            ("EVAL_TRAJ", "tool",     "mid"),
    # iter 75: within-harness frontier-vs-small SWE-bench-Verified (light scaffold)
    "swebench-verified-gpt52-eval":          ("EVAL_TRAJ", "swe",      "frontier"),
    "swebench-verified-gpt5mini-eval":       ("EVAL_TRAJ", "swe",      "mid"),
    # iter 103: multilingual SWE-bench eval rollouts (frontier, non-Python)
    "swebench-multiling-glm5-eval":          ("EVAL_TRAJ", "swe",      "frontier"),
    "swebench-multiling-minimax-m25-eval":   ("EVAL_TRAJ", "swe",      "frontier"),
    # iter 81: single- vs multi-agent serialization of the same SWE-smith tasks
    "swesmith-singleagent-traj":             ("TRAIN", "swe", "frontier"),
    "swesmith-multiagent-traj":              ("TRAIN", "swe", "frontier"),
    # iter 100: two new SWE training sources (mid llama-70b vs frontier Kimi-K2)
    "swe-agent-llama70b-traj":               ("TRAIN", "swe", "mid"),
    "deepswe-kimi-k2-traj":                  ("TRAIN", "swe", "frontier"),
    "deepswe-kimi-k2-rejsample-traj":        ("TRAIN", "swe", "frontier"),
    # --- human-demonstration datasets (benchmarks w/ test splits + demos) ---
    "mind2web-actions":        ("EVAL_TASK", "web",    "human_demo"),
    "mind2web-fullobs":        ("EVAL_TASK", "web",    "human_demo"),
    "weblinx-actions":         ("EVAL_TASK", "web",    "human_demo"),
    "weblinx-fullobs":         ("EVAL_TASK", "web",    "human_demo"),
    "gui-odyssey-actions":     ("EVAL_TASK", "gui",    "human_demo"),
    "android-control-text":    ("EVAL_TASK", "gui",    "human_demo"),
    # --- training trajectories: frontier-model rollouts (healthy) ---
    "swe-rebench-oh-traj":     ("TRAIN", "swe", "frontier"),
    "swe-rebench-oh-assistant-only": ("TRAIN", "swe", "frontier"),
    "swe-gym-oh-sft-traj":     ("TRAIN", "swe", "frontier"),
    "swe-gym-oh-sampled-traj": ("TRAIN", "swe", "frontier"),
    "swe-gym-oh-verifier-traj":("TRAIN", "swe", "frontier"),
    "swe-zero-oh-traj":        ("TRAIN", "swe", "frontier"),
    "swe-hero-oh-traj":        ("TRAIN", "swe", "frontier"),
    "swe-smith-traj-xml":      ("TRAIN", "swe", "frontier"),
    "swe-zero-12m-traj":       ("TRAIN", "swe", "frontier"),
    "kwai-klear-mini-swe-66k": ("TRAIN", "swe", "mid"),
    "jetbrains-swe-test-minus-verified": ("TRAIN", "swe", "frontier"),
    "jetbrains-swe-assistant-only":      ("TRAIN", "swe", "frontier"),
    "jetbrains-swesmith-subset":         ("TRAIN", "swe", "frontier"),
    "dcagent-glm47-terminus2": ("TRAIN", "terminal", "frontier"),
    "openhands-feedback":      ("TRAIN", "swe", "frontier"),
    "miroverse-agentic-sft-new":("TRAIN", "search", "frontier"),
    "nemotron-sft-v2-search":  ("TRAIN", "search", "frontier"),
    "nemotron-sft-v2-interactive": ("TRAIN", "tool", "frontier"),
    "toucan-15m-kimi-k2":      ("TRAIN", "tool", "frontier"),
    "toucan-15m-oss":          ("TRAIN", "tool", "frontier"),
    "toucan-15m-qwen3":        ("TRAIN", "tool", "frontier"),
    "toucan-15m-sft":          ("TRAIN", "tool", "frontier"),
    "cli-agent-sessions-sampler": ("TRAIN", "terminal", "frontier"),
    "opus48-pi-traces":        ("TRAIN", "terminal", "frontier"),
    "glaive-fc-v2":            ("TRAIN", "tool", "frontier"),
    "hermes-fc-multiturn":     ("TRAIN", "tool", "frontier"),
    "fireact-multitask":       ("TRAIN", "tool", "frontier"),
    # --- training trajectories: mid-size generator (failure-loop collapse) ---
    "r2e-gym-swe-agent-lm-traj": ("TRAIN", "swe", "mid"),
    "qwen35-9b-react-hotpot":    ("TRAIN", "search", "mid"),
    "dcagent3-financeagent-a3rl":("TRAIN", "terminal", "mid"),
    # --- training trajectories: human-demonstration (released as training) ---
    "agentnet-text":           ("TRAIN", "gui", "human_demo"),
    "agentnet-actions":        ("TRAIN", "gui", "human_demo"),
    "opencua-text":            ("TRAIN", "gui", "human_demo"),
    "cua-agentnet-gimp-text":  ("TRAIN", "gui", "human_demo"),
    "nnetnav-live":            ("TRAIN", "web", "frontier"),  # real-web rollouts
    "nnetnav-wa":              ("TRAIN", "web", "mid"),        # simulated webarena
    "agenttrek":               ("TRAIN", "web", "distill"),
    # --- training trajectories: distillation / SFT mixtures (template band) ---
    "agentgym-agenttraj-l":    ("TRAIN", "mixed", "distill"),
    "agentinstruct-all":       ("TRAIN", "mixed", "distill"),
    "agent-flan-all":          ("TRAIN", "mixed", "distill"),
    "code-act-codeact":        ("TRAIN", "mixed", "distill"),
    "toolbench-toolllama-dfs": ("TRAIN", "tool", "distill"),
    "apigen-mt-5k":            ("TRAIN", "tool", "distill"),
    "lumos-ground-iter":       ("TRAIN", "mixed", "distill"),
    "scienceworld-expert-traj":("TRAIN", "embodied", "distill"),
    "nemotron-rl-swe-pivot":   ("TRAIN", "swe", "distill"),
    "nemotron-rl-conv-tool-pivot": ("TRAIN", "tool", "distill"),
    "nemotron-sft-v2-tool":    ("TRAIN", "tool", "distill"),
    "agentbank-all":           ("TRAIN", "mixed", "distill"),
    "ko-agent-traj-train":     ("TRAIN", "tool", "distill"),
    "app1-agentic-safety-sft": ("TRAIN", "safety", "distill"),
    "rebel-alfworld-sft":      ("TRAIN", "embodied", "distill"),
    "rebel-alfworld-actions":  ("TRAIN", "embodied", "distill"),
    "factory-agent-task-rollouts": ("TRAIN", "tool", "distill"),
    "deep-research-sft-0406":  ("TRAIN", "search", "distill"),
    "fractal-deepresearch-sft":("TRAIN", "search", "distill"),
    "tiger-browseragent-sft":  ("TRAIN", "web", "distill"),
    "nebius-swe-agent-traj":   ("TRAIN", "swe", "distill"),
    "openthoughts-agent-v1-sft": ("TRAIN", "terminal", "distill"),
    "nl2bash-teacher-glm46":   ("TRAIN", "terminal", "frontier"),
    "nl2bash-teacher-qwen3coder480b": ("TRAIN", "terminal", "frontier"),
    "nl2bash-teacher-minimax-m27":    ("TRAIN", "terminal", "frontier"),
    "inferredbugs-teacher-glm46":     ("TRAIN", "swe", "frontier"),
    "inferredbugs-teacher-glm46-131k":("TRAIN", "swe", "frontier"),
    "inferredbugs-teacher-glm47":     ("TRAIN", "swe", "frontier"),
    "inferredbugs-teacher-minimax-m27":("TRAIN", "swe", "frontier"),
    "inferredbugs-teacher-qwen35-122b":("TRAIN", "swe", "frontier"),
    "inferredbugs-teacher-kimi25":    ("TRAIN", "swe", "frontier"),
    "inferredbugs-teacher-gpt5nano":  ("TRAIN", "swe", "frontier"),
    # --- excluded: 3-point fit artifacts + single-episode proxy ---
    "nemotron-agentic-v1":     ("EXCLUDED", "tool", "distill"),
    "rlenv-appworld-train":    ("EXCLUDED", "tool", "synth_task"),
    "aguvis-s2-androidctl-text":("EXCLUDED", "gui", "human_demo"),
    "saital-browser-reasoning-action": ("EXCLUDED", "web", "distill"),
    "saital-browser-action-only":      ("EXCLUDED", "web", "distill"),
    "taubench-sonnet-proxy":   ("EXCLUDED", "tool", "frontier"),
}

# beta (corpus-level) and Hurst joins, keyed by slug where measured.
BETA = {}  # agentic beta lives in gamma_beta_all.csv under non-reference rows;
# we only have corpus-level agentic beta for the prose/code reference set there,
# so beta is reported at the phase level in the paper, not per-row. Hurst is
# per-slug where measured:
HURST = {}
for r in csv.DictReader(open(ROOT / "data" / "hurst.csv")):
    HURST[r["slug"]] = float(r["hurst"])


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


rows = list(csv.DictReader(open(SRC)))
out_rows = []
for r in rows:
    slug = r["slug"]
    role, domain, source = CLASS.get(slug, ("UNCLASSIFIED", "?", "?"))
    out_rows.append({
        "slug": slug,
        "dataset": r["dataset"],
        "role": role,
        "domain": domain,
        "source": source,
        "alpha": fnum(r["alpha"]),
        "h_inf": fnum(r["h_inf"]),          # reference-exact (canonical)
        "bpc_32768": fnum(r["bpc_32768"]),  # supplementary directly-measured
        "hurst": HURST.get(slug, ""),
        "n_episodes": r["n_episodes"],
        "mean_turns": r["mean_turns"],
        "mean_doc_bytes": r["mean_doc_bytes"],
        "resolved": r.get("resolved", ""),
    })

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
    w.writeheader()
    w.writerows(out_rows)

# --- aggregates --------------------------------------------------------------
unclass = [r for r in out_rows if r["role"] == "UNCLASSIFIED"]
assert not unclass, f"unclassified slugs: {[r['slug'] for r in unclass]}"

active = [r for r in out_rows if r["role"] != "EXCLUDED"]


def summ(rows, key):
    vals = [r[key] for r in rows if r[key] == r[key]]
    if not vals:
        return "n/a"
    return f"n={len(vals):3d}  median={st.median(vals):.2f}  mean={st.fmean(vals):.2f}  [{min(vals):.2f},{max(vals):.2f}]"


print(f"\nTotal rows={len(out_rows)}  active(non-artifact)={len(active)}  excluded={len(out_rows)-len(active)}\n")
print("=== by ROLE ===")
for role in ["TRAIN", "EVAL_TASK", "EVAL_TRAJ"]:
    grp = [r for r in active if r["role"] == role]
    print(f"\n[{role}]  rows={len(grp)}")
    print(f"  H_inf (ref-exact): {summ(grp,'h_inf')}")
    print(f"  BPC@32K          : {summ(grp,'bpc_32768')}")
    print(f"  alpha            : {summ(grp,'alpha')}")

print("\n=== by SOURCE (generator) ===")
for src in ["human_task", "human_demo", "synth_task", "frontier", "mid", "distill"]:
    grp = [r for r in active if r["source"] == src]
    print(f"\n[{src}]  rows={len(grp)}")
    print(f"  H_inf (ref-exact): {summ(grp,'h_inf')}")
    print(f"  BPC@32K          : {summ(grp,'bpc_32768')}")

print("\n=== H_inf>0 (content-bearing) fraction by source ===")
for src in ["human_task", "human_demo", "synth_task", "frontier", "mid", "distill"]:
    grp = [r for r in active if r["source"] == src]
    if grp:
        pos = sum(1 for r in grp if r["h_inf"] > 0.3)
        print(f"  {src:11s}: {pos}/{len(grp)} have H_inf>0.3")

print(f"\nwrote {OUT}")
