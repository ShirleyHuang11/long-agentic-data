# Metric space for characterizing agentic corpora — brainstorm

**Goal context:** a *unified interpretation of the datasets independent of train/eval*.
We already showed role (train/eval) organizes almost nothing (H∞ η²: source 0.32 ≫
role 0.09; silhouette by role ≈ 0). The corpora form a **continuum** on a small
number of axes. This document brainstorms the *full* space of metrics relevant to
**constructing training data** and **benchmark evals**, then flags which are new,
near-independent axes worth adding to our current 5 (H∞, BPC@32K, α, β, Hurst).

Legend for measurability:
- 🟢 **byte-level** — computable from trajectory text alone (what we already do).
- 🟡 **structural** — needs trajectory parsing (turns, tool calls, roles).
- 🟠 **model-in-the-loop** — needs reference-model rollouts / a judge.
- 🔴 **label/oracle** — needs ground truth, human annotation, or env access.

---

## A. Information content / compressibility — *what we have*
| metric | captures | meas. |
|---|---|---|
| H∞ (cross-episode floor) | corpus-level content **diversity** | 🟢 |
| BPC@32K | within-window **density** | 🟢 |
| α (entropy-decay) | curve shape; redundant w/ H∞ (ρ=0.78) | 🟢 |
| β (correlation decay) | **repetition** structure (format, not content) | 🟢 |
| Hurst | long-range dependence (near-independent axis) | 🟢 |

## B. Length / scale — *one independent axis (ρ≈0 to content)*
- **byte/token length** per episode 🟢
- **# turns / steps** 🟡 — we have this; orthogonal to content.
- **# tool calls**, **# distinct tools** 🟡
- **context required to solve** (min tokens that must be in-window) 🟠 — distinct from raw length; a long log with a small relevant slice is "short" for the model.
- **nesting / call-stack depth** 🟡

## C. Difficulty — *the benchmark-critical family*
- **empirical solve rate** (pass@1 / pass@k of a reference model) 🟠 — the gold difficulty signal.
- **discrimination** = spread of pass rates *across model tiers* 🟠 — a good benchmark item separates strong from weak models; an item everyone passes or everyone fails carries no signal.
- **headroom** = 1 − best-model solve rate 🟠 — saturated items are dead weight.
- **branching / search factor** — # candidate actions explored before success 🟡🟠
- **backtracking count** — dead ends entered 🟡
- **subgoal breadth** — # distinct subgoals to decompose into 🟠

## D. Reasoning / logical depth — *the user's seed, expanded*
- **deduction-chain length** (logical-depth analog) — # inferential steps from premises to answer 🟠
- **reasoning-to-action ratio** = think-tokens / action-tokens 🟡 — cheap proxy for "how much thinking per move"; separates reasoning-SFT from pure action traces (ties to our β finding).
- **novel-inference density** — fraction of steps that are genuine inferences vs lookups/restatements 🟠
- **working-set size** — # intermediate results that must be held simultaneously 🟠
- **credit-assignment horizon** ⭐ — longest distance between an action and the observation that justifies/rewards it. This is *the* long-horizon RL metric and plausibly correlates with **Hurst** (long-range dependence). 🟡🟠

## E. Tool / environment interaction
- **action-space size**, **tool diversity** 🟡
- **environment stochasticity** (deterministic vs random) 🔴
- **observation entropy** (how informative each obs is) 🟢🟡
- **tool-error rate** (calls that fail / get corrected) 🟡
- **irreversibility** — presence of high-stakes irreversible actions 🔴 (safety-relevant)

## F. Verifiability / reward — *the RL-data-critical family*
- **verifiability** ⭐ — can the outcome be checked automatically? (binary; the single most important property for RL training data and for trustworthy evals) 🔴
- **reward sparsity** — steps between reward signals 🟡🔴
- **ground-truth availability / quality** 🔴
- **answer stability** — is the gold answer time-sensitive / ambiguous? 🔴

## G. Quality / hygiene — *corpus curation*
- **near-duplication rate** within corpus (dedup) 🟢 — partially captured by H∞ but a direct dup-rate is actionable.
- **contamination** vs pretraining/training set (n-gram / embedding overlap) ⭐ 🟢🟠 — decisive for *eval* validity; an item leaked into training is worthless as a benchmark.
- **label noise / annotation consistency** 🔴
- **scaffold/boilerplate fraction** 🟢 — the "pooling" driver we already study; measurable via shared-line stripping.

## H. Diversity / coverage — *corpus-level*
- **inter-episode diversity** = H∞ (have it) 🟢
- **skill/domain coverage breadth** + **distributional balance** (entropy over task types) 🟡🟠
- **novelty vs training distribution** (for evals: distance from train set) 🟠

## I. Cost / efficiency — *the user's seed*
- **generation cost** (frontier-API $ to produce the trajectory) 🟠
- **solve cost** (inference tokens a reference model spends) 🟠
- **solution efficiency** = optimal-path-length / actual-path-length 🟠🔴 — detects padded/inefficient demonstrations.
- **annotation cost** (human time) 🔴

---

## Synthesis — what's worth adding to the unified frame

Our current measurable axes reduce to **~3 independent ones**: content-richness
(H∞/BPC/α bundle), length (turns), and a within-window-density subaxis. The
highest-value *new, plausibly-independent, byte/structural* axes to measure next:

1. ⭐ **reasoning-to-action ratio** (🟡, cheap) — a content-orthogonal "cognitive
   load" axis; should cleanly separate reasoning-SFT from action traces and may
   explain β.
2. ⭐ **scaffold/boilerplate fraction** (🟢, cheap) — directly operationalizes the
   pooling mechanism behind H∞=0; turns a latent cause into a measured coordinate.
3. ⭐ **credit-assignment horizon** (🟡) — the long-horizon-specific axis; test
   whether it's truly captured by Hurst or is independent.
4. ⭐ **near-duplication / contamination** (🟢🟠) — the hygiene axis that gates
   whether a corpus is usable as train vs eval *regardless of its content score*.

The 🟠/🔴 families (empirical difficulty, verifiability, discrimination) are the
true gold metrics for **benchmark** construction, but they need model rollouts or
oracle access — out of reach for a pure byte-level pass. They define the boundary
of what compression-oracle analysis can and cannot say about a corpus.
