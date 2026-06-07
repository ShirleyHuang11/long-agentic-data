# InferredBugs teacher panel — do teachers separate on the real-bug source?

*2026-06-07 — scores in `data/agentic_alpha_hinf.csv` (slugs `inferredbugs-teacher-*`, `inferredbugs-tasks-clean`), seed-σ in `data/seed_sigma.csv`, provenance sidecars per slug. Companion to `reports/openthoughts_agent_analysis.md`.*

## Context

OpenThoughts-Agent's SFT ablation kept two winning instruction sources: **NL2Bash** (synthetic shell micro-tasks) and **InferredBugs** (real Microsoft C#/.NET bugs). Their teacher ablation found a **GLM-4.6 teacher ≈ 2× downstream** over GPT-family teachers. On the NL2Bash source (iter 59) every teacher was statistically identical under our compression oracle (α 0.135–0.156, H∞ all 0.000) — a clean tie, the probe's resolution boundary (finding 11/19). NL2Bash is ~10k near-permutations of small templated tasks, so the tie was expected.

**Open question:** does the *real-bug* source — less templated, real .NET source code in every trace — let teachers separate? This panel scores 7 per-teacher trace dumps (all terminus-2 schema: `conversations` list of `{role,content}` + `model` field → `ser_conversations_auto`) plus the real-bug **task corpus** for a task-level reference.

## Results

### Teacher panel (offset-0 registry scores)

| teacher (HF `model` field) | dump | α | H∞ | mean_turns | mean_bytes |
| :-- | :-- | --: | --: | --: | --: |
| **GLM-4.6** (GLM-4.6-AWQ) | penfever/…-32ep-65k | 0.227 | **0.000** | 19.4 | 35,625 |
| GLM-4.6 (131k no-summ chunk) | DCAgent2/…-131k-chunk002 | 0.245 | 0.223 | 22.6 | 42,453 |
| **GLM-4.7** (vLLM) | DCAgent2/…-maxeps-131k | 0.227 | **0.000** | 16.5 | 59,084 |
| **gpt-5-nano** (terminus-2 canonical) | mlfoundations-dev/…-terminus-2 | 0.232 | **0.000** | 15.4 | 53,452 |
| **MiniMax-M2.7** (vLLM) | penfever/…-minimax-m27-131k | 0.255 | 0.169 | 8.0 | 38,785 |
| **Qwen3.5-122B** (vLLM) | penfever/…-qwen3.5-122b-32k | 0.253 | 0.202 | 9.7 | 51,245 |
| **Kimi-2.5** (vLLM) | DCAgent2/…-maxeps-32k | 0.258 | 0.279 | 18.8 | 51,486 |
| *task corpus* (real MS bugs) | ypguo/Clean_Microsoft_InferredBugs | 0.236 | **0.000** | 1.0 | 3,072 |

### Seed-σ robustness (3 disjoint slices/teacher, step=40, `data/seed_sigma.csv`)

| teacher | H∞ slice 0 | slice 40 | slice 80 | read |
| :-- | --: | --: | --: | :-- |
| GLM-4.6 (65k) | 0.000 | 0.000 | 0.000 | **σ≈0 — genuinely pinned at the template band** |
| gpt-5-nano | 0.000 | 0.024 | **0.168** | **σ enormous — composition-driven, NOT a stable 0** |
| MiniMax-M2.7 | 0.169 | 0.103 | 0.067 | positive, σ≈0.05, drifting |
| Qwen3.5-122B | 0.202 | 0.175 | 0.208 | **positive, tight σ≈0.015 — most stable separation** |
| Kimi-2.5 | 0.279 | 0.255 | 0.415 | consistently positive, σ≈0.08 |

## Verdict: a partial, σ-fragile separation — not a clean tie, not a clean ranking

**Teachers DO move off the floor on the real-bug source — unlike NL2Bash.** On NL2Bash all teachers were pinned at H∞=0.000; here the panel spans **0.000 → 0.279** at offset-0, and three teachers (Qwen3.5, Kimi-2.5, MiniMax-M2.7) sit reproducibly above zero across every slice. Real .NET source content in the traces supplies an incompressible sliver that synthetic shell permutations did not. **So the answer to the open question is a qualified yes: the real-bug source breaks the perfect tie.**

**But the separation does not survive as a teacher ranking, and it does not recover the 2× downstream result.** Per finding 14, composition σ on heterogeneous sets runs ~0.03–0.24, and the gpt-5-nano slices alone span 0.000→0.168 — meaning a single-seed H∞ for one teacher is, by itself, uninterpretable. The cross-teacher spread (0.0–0.28) is the **same order as the within-teacher composition σ**, so the apparent ordering (Kimi > Qwen > MiniMax > GLM/GPT) is partly which 130–230 real bugs landed in each dump, not teacher skill. Concretely:

- **GLM-4.6 (the downstream winner) reads as template-band (0.000, σ≈0).** The compression oracle still cannot see why it trains 2× better. The winning teacher is, if anything, the *most* compressible — exactly the iter-59 pattern.
- **gpt-5-nano (the canonical loser) is NOT cleanly below the others** — its slice-80 H∞ (0.168) exceeds GLM-4.6 entirely. The "0.000" at offset-0 is a composition artifact, not a teacher signature.
- The robustly-positive teachers (Qwen3.5, Kimi) are *not* the ones that won the ablation.

**The iter-59 boundary therefore holds on the real-bug source too: compression statistics select data *regimes*, they do not rank *recipes within* a regime.** The real-bug source nudges the whole panel toward the healthy band relative to NL2Bash, but the teacher-identity axis — the one carrying the 2× downstream effect — remains below the probe's resolution once σ is accounted for. This is the iter-59 finding, now stress-tested on the less-templated source and surviving.

### Comparison to the NL2Bash panel

| | NL2Bash panel (iter 59) | InferredBugs panel (this) |
| :-- | :-- | :-- |
| H∞ spread across teachers | all 0.000 (perfect tie) | 0.000–0.279 (panel lifts off floor) |
| α | 0.135–0.156 | 0.227–0.258 (higher LRD) |
| task corpus H∞ | (not isolated) | 0.000 (real bugs, but task text is short+metadata-templated) |
| teacher ranking recoverable? | no (boundary) | **no** (spread ≈ composition σ) |
| regime | template band | template→low-healthy edge |

The real-bug *traces* carry more incompressible content than NL2Bash traces (panel off the floor, α higher), consistent with InferredBugs being the less-templated of the two winning sources. But the **task corpus itself scores H∞=0.000** — the clean text export is short (mean 3 KB) bug-metadata + before/after methods, and the cross-bug recurrence (boilerplate .NET idioms, repeated `DOTNET_RESOURCE_LEAK`-class fixes) compresses it to the floor. The novelty that lifts the trace panel comes from the *teacher's reasoning/patch prose around* the bugs, not the bug specs.

## Caveats

- **Single offset-0 seed per registry row; cross-teacher H∞ spread (0.0–0.28) ≈ within-teacher composition σ (finding 14: 0.03–0.24 on heterogeneous sets).** The per-teacher H∞ ordering is therefore not interpretable as a teacher ranking; only the *panel-vs-NL2Bash* shift (floor → off-floor) is above σ.
- Dumps differ in context budget (32k/65k/131k), summarization (GLM-4.6 65k vs 131k-no-summ differ by 0.000 vs 0.223 — a **format**, not teacher, effect), and episode count (134–236) — these confounds are entangled with teacher identity and were not controlled.
- The GLM-4.6 65k vs 131k-no-summ gap (0.000 → 0.223) is direct evidence that **trace-format choices move H∞ as much as teacher identity does**, reinforcing that the panel spread is not a clean teacher axis.
- Task corpus uses the `ypguo/Clean_Microsoft_InferredBugs` text export (real before/after methods + bug metadata; abstracted `*_template` fields excluded). The OpenThoughts sandbox task dumps (`DCAgent/inferredbugs-sandboxes-verifier`, `mlfoundations-dev/inferredbugs-sandboxes`) store **gzipped tarballs** in a `task_binary` blob — not cleanly text-scoreable without unpacking, so the clean export is the task-level reference here.
- Downstream 2× claim is from OpenThoughts' blog, not reproduced; we score the data side only.

## Dumps found vs missing

**Found & scored (7 teachers + 1 task corpus):** GLM-4.6 (×2 budgets), GLM-4.7, gpt-5-nano (= terminus-2 canonical), MiniMax-M2.7, Qwen3.5-122B, Kimi-2.5; task corpus `ypguo/Clean_Microsoft_InferredBugs`. All per-teacher dumps share the terminus-2 schema and loaded cleanly via the standard hub loader (no CastError, no hf-json: fallback needed).

**Also present on HF but not scored (out of scope — downstream eval/RL artifacts, not SFT teacher traces):** the large `DCAgent*/dcagent-dev-set-71-tasks-…-inferredbugs-…` and `…terminal_bench_2…/swebench_verified…` families (these are *eval runs of trained students*, not teacher rollouts); `DCAgent/inferredbugs-sandboxes_glm_4.7_traces_jupiter` (already in registry as `dcagent-glm47-terminus2`, SWE-Gym-sourced not InferredBugs-sourced); RL dumps (`*a3_rl*`, `rl__24GPU_shaped__…`). No GPT-5 / GPT-5-mini *full* teacher InferredBugs dump found beyond the gpt-5-nano terminus-2 canonical; no standalone Qwen3-Coder-480B InferredBugs dump found (that teacher appears only on NL2Bash).
