# A Compression-Oracle Survey of Long-Horizon Agentic Data: Merging Training and Evaluation Corpora by Pattern and Content

*Analysis paper, long-agentic-data project. Built from the `SAMPLES.md` registry
(98 active datasets / 104 scored rows, iters 1–70) and the metrics validated in
`reports/`. Figures in `figures/`; per-row merged table in
`data/merged_analysis.csv` (`scripts/build_merged_table.py`).*

---

## Abstract

We survey **98 long-horizon agentic corpora** — spanning model-trained SFT/RL
trajectories, human-written benchmark tasks, human demonstrations, and agent
rollouts collected on benchmarks — with a single cheap, tokenizer-free
compression oracle. For every corpus we measure a **pattern** axis (the
context-scaling exponent α, the token-correlation decay β, and the Hurst
exponent H) and a **content** axis (the reference-exact incompressible floor
H∞, validated against true neural oracles at Spearman 0.97, with directly
measured BPC@32K as a companion). Merging training and evaluation data into one
table yields a sharp dissociation:

- **Pattern is a property of the agentic *format*.** α is nearly constant
  across roles (median 0.24–0.34 for train / eval-task / eval-traj), and agentic
  data occupies a distinct correlation-decay phase (β ≈ 0.2–0.5) that does not
  overlap natural-language prose (β ≈ 1.1–1.4), with code/math as the bridge
  (β ≈ 0.5–0.8).
- **Content is set by *who authored the token stream*, not by whether the data
  is for training or testing.** Median H∞ falls monotonically with generator
  source — human task 1.22, human demo 1.13, frontier rollout 0.80, synthetic
  task 0.45, mid-size rollout 0.00, distilled SFT 0.00 — and this ordering cuts
  *across* the train/eval boundary: mid-size eval rollouts collapse to the
  template floor exactly as distilled training mixtures do.
- This exposes a **content gap**: the benchmark *tasks* we score agents against
  are human-authored and content-dense (median H∞ 1.22), while the bulk of the
  *training* data we feed agents is content-sparse (median 0.26). We test on
  human richness and train on machine boilerplate.

We connect these measurements to a learning-curve theory (α_D = γ/2β), to a
form-vs-content decomposition that is domain-dependent, and to a training
experiment that confirms SFT teaches trajectory *form*. We close with the one
method caveat that survived audit (H∞ is not identifiable on heavily pooled
corpora; the reference 3-point clamp is retained as canonical for cross-domain
comparability) and three forward experiments.

---

## 1. Introduction

Long-horizon agentic data — multi-turn trajectories where a model interacts
with tools, repositories, browsers, or operating systems — is now the central
fuel of frontier agent training. Yet it is curated almost entirely by intuition:
"frontier teacher × hard tasks × real environment" is folklore, not a measured
quantity. Two questions motivate this survey:

1. **What can a cheap, label-free probe tell us about an agentic corpus before
   we spend GPUs training on it?** We want a few seconds of CPU compression to
   separate a content-rich rollout from a boilerplate-pooled one.
2. **Is the data we *train* on statistically like the data we *evaluate* on?**
   Training corpora and benchmarks are studied in separate literatures. Merging
   them into one measurement table is the novel lens of this paper, and it
   reveals a gap that neither literature sees alone.

The instrument is a Lempel–Ziv (zstd) compression oracle that reproduces, bit
for bit, the protocol from a 358-dataset formal-math survey
(`reference/data_format.md`), where it was validated against a true LLM oracle
(LZ-derived H∞ vs neural H∞, Spearman 0.97). We keep that protocol **exactly**,
for cross-domain comparability, and add three companion statistics from the
recent literature (β, Hurst, seed-σ).

---

## 2. The merged corpus

The registry holds **104 scored rows**; 6 are 3-point fit artifacts (α<0 or
H∞≫1, plus one single-episode dump) and are dropped, leaving **98 active
corpora**. We classify each along three axes (`scripts/build_merged_table.py`,
output `data/merged_analysis.csv`):

| axis | values |
| :-- | :-- |
| **role** | `TRAIN` (SFT/RL training trajectories) · `EVAL_TASK` (benchmark task/problem corpora + human-demonstration benchmarks) · `EVAL_TRAJ` (agent rollouts collected on a benchmark) |
| **domain** | swe · web · gui · tool · search · terminal · safety · embodied · mixed |
| **source** | `human_task` · `human_demo` · `synth_task` · `frontier` (frontier-model rollout) · `mid` (mid-size-model rollout) · `distill` (GPT-4-class distillation/SFT mixture) |

Counts: **TRAIN 69, EVAL_TASK 13, EVAL_TRAJ 16**; by source, frontier 44,
distill 24, mid 13, human_demo 10, human_task 5, synth_task 2. The merge is
deliberate: human-demonstration datasets (Mind2Web, WebLINX, GUI-Odyssey,
AndroidControl, AgentNet, OpenCUA) straddle the train/eval line — they ship test
splits *and* are used as demonstration training data — so `source` is the axis
that carries the real signal, while `role` tests whether the train/eval label
predicts anything (it largely does not; §5).

**Classification caveats (honest).** The role label is fuzzy for a handful of
corpora: SWE-Gym and SWE-rebench are training *environments* whose released form
is a task corpus (labeled `EVAL_TASK`); AgentNet/OpenCUA are human demos
released as *training* data (labeled `TRAIN`, source `human_demo`). These calls
do not move the central result, which is organized by `source`, not `role`.

---

## 3. The metric toolbox

For a corpus of documents joined with blank-line separators (≤1500 docs / 8 MB),
we compress in independent n-byte chunks and read bits-per-character BPC(n),
fitting the scaling model **BPC(n) = H∞ + c·n^(−α)**.

- **α — pattern / long-range structure.** Larger α ⇒ more context-length
  structure an LZ compressor can exploit (templating, repeated scaffolds, or
  genuine long-range regularity).
- **H∞ — content / incompressible floor (canonical).** The reference-exact
  3-point analytic estimate at n = 128 / 2048 / 32768 (geometric r=16),
  **floored at 0**. The closed form is
  `H∞ = B₃ − (B₂−B₃)² / [(B₁−B₂) − (B₂−B₃)]`. H∞ ≈ 0 is the reference's *valid*
  "template-degenerate / spam" signal (e.g. formal-math TPTP ≈ 0), not a bug.
  This is the canonical content metric, kept bit-identical to the formal-math
  registry (cross-domain table: NL 2.6 / code 2.63 / formal-math 1.57).
- **BPC@32K — directly measured content companion.** The raw compressed rate at
  a 32 KB context — no fit, no clamp, no extrapolation. Validated on synthetic
  controls (random 2.47 / pure template 0.02 / scaffold+content 1.40) and shown
  robust to the 8K-vs-32K-token context choice (ranking invariant). We report it
  alongside H∞ as a sanity companion; where the two disagree, §6 explains why.
- **β — token-correlation decay (Cagnetta et al.).** ‖C(n)‖ ∝ n^(−β). The
  data-limited learning-curve exponent is α_D = γ/2β, so β is a *predictive*
  data statistic. Measured byte-level here as a proxy.
- **Hurst H — long-range organization (Alabdulmohsin et al.).** R/S on
  order-3 byte-n-gram surprisal increments; correlates with downstream accuracy
  in the *fixed-data, cross-model* setting.
- **seed-σ — reproducibility.** Re-scoring on disjoint slices to bound which
  differences are real (cluster-level ≫ σ; within-band <0.3 is noise).

---

## 4. Pattern analysis: the agentic format is its own statistical phase

**α is nearly invariant to role.** Median α is 0.24 (TRAIN), 0.34 (EVAL_TASK),
0.26 (EVAL_TRAJ) — a ±0.05 band. Whatever long-range structure the LZ probe
sees is a property of the multi-turn agentic *serialization* (JSON scaffolds,
repeated system prompts, turn templates), and it is present whether the corpus
is a training mixture or a benchmark. Pattern, in this sense, is the genre
signature; it does not tell training from evaluation apart.

**Agentic data occupies a distinct correlation-decay phase.** Measuring β with
the same byte-level protocol across 19 reference corpora plus the FineFineWeb
domains (`data/gamma_beta_all.csv`, `figures/fig6_gamma_beta.png`) gives three
non-overlapping bands:

| phase | β | examples |
| :-- | :-- | :-- |
| **agentic** | **0.2–0.5** | the trajectories in this survey |
| **code / math (bridge)** | 0.52–0.79 | Python code 0.74, Proof-Pile arxiv 0.52, open-web-math 0.79, The Pile 0.78 |
| **natural-language prose** | 1.1–1.37 | C4 1.32, FineWeb-Edu 1.35, WikiText 1.22, TinyStories 1.35 |

Through α_D = γ/2β, the low agentic β predicts an unusually *high* data-limited
learning exponent (α_D ≈ 0.3–1.0): agentic data should be sample-efficient to
fit — consistent with the finding that SFT drives trajectory *form* to ceiling
almost immediately (§7), and a hypothesis still awaiting a from-scratch training
test (§9).

**Hurst alone cannot grade agentic data.** Across 9 representative corpora
(`data/hurst.csv`, `figures/fig5_hurst_vs_content.png`), template/spin corpora
sit in the *same* Hurst band as healthy ones (APIGen 0.80, Ko-Agent 0.83 vs
JetBrains 0.78, Toucan 0.90, SWE-ZERO 0.93) — because *repetition itself is
long-range dependence*. Hurst conflates form-LRD with content-LRD and is nearly
orthogonal to H∞ in this domain (the highest-content corpus, WebLINX-actions
H∞ 1.95, has the *lowest* Hurst 0.67). The (H, H∞) pair separates them; H∞
alone does not.

---

## 5. Content analysis: H∞ tracks the generator, and cuts across train/eval

This is the paper's central result (`figures/fig_merge_content_source.png`).

**By source, median H∞ falls monotonically:**

| source | n | median H∞ | mean H∞ | fraction H∞ > 0.3 |
| :-- | --: | --: | --: | --: |
| human task (written problems) | 5 | **1.22** | 1.06 | 4/5 |
| human demo (action streams) | 10 | **1.13** | 0.92 | 6/10 |
| frontier-model rollout | 44 | **0.79** | 0.78 | 31/44 |
| synthetic task | 2 | 0.45 | 0.45 | 1/2 |
| mid-size-model rollout | 13 | **0.00** | 0.13 | 2/13 |
| distilled SFT mixture | 24 | **0.00** | 0.08 | 3/24 |

**By role, the same numbers re-sort to expose the gap:**

| role | n | median H∞ | reading |
| :-- | --: | --: | :-- |
| EVAL_TASK | 13 | **1.22** | human-authored tasks/demos → content-dense |
| TRAIN | 69 | **0.26** | bimodal: healthy frontier minority + collapsed majority |
| EVAL_TRAJ | 16 | **0.04** | model rollouts span the full range; mid-model runs collapse |

Three things follow.

**(i) Content is authored, not roled.** The high-content end is human
(written tasks + demonstrations); the low-content end is mid-size or distilled
*model* generation. The decisive variable is the generator, and it dominates
both train and eval. An eval rollout produced by a 7B–32B model on the
aider-polyglot benchmark (`EVAL_TRAJ`, source `mid`, H∞ ≈ 0) is statistically
indistinguishable from a distilled training mixture (`TRAIN`, source `distill`,
H∞ ≈ 0) — same template floor, same failure-loop length inflation (§6).

A controlled benchmark-rollout batch (iter 69) makes this concrete and adds a
domain modulation. We scored three mid-size (≈32B) eval rollouts on three named
benchmarks: Qwen3-32B on Terminal-Bench-2 (H∞ 0.00, 172 turns), R2EGym-32B on
GAIA-127 (H∞ 0.00, 200 turns), and CoderForge-32B on SWE-bench-Verified
(H∞ 0.83, 136 turns). The first two **collapse to the template floor with the
longest failure loops in their domains** — and the GAIA-127 result sits directly
against the *frontier* GAIA rollout already in the registry (ii-agent, H∞ 1.25):
**same benchmark, frontier 1.25 vs mid 0.00**, a clean generator contrast on
identical eval tasks. The SWE rollout stays healthy at 0.83 — not because the
32B generator is strong, but because **repository observations are content**
(finding 16, §6): on SWE the environment injects real code regardless of the
agent. So eval rollouts inherit the *generator's* content signature, *modulated
by whether the domain's observations are content (SWE) or boilerplate
(terminal/search)* — never by the train/eval label.

A follow-up (iter 70) adds frontier rollouts on Terminal-Bench-2 — GPT-5 and
Claude-Sonnet-4.5 — and exposes a *measurement* nuance worth stating plainly.
On this benchmark the canonical H∞ reads ≈ 0 for **everyone**, frontier
included (GPT-5 0.00, Sonnet-4.5 0.16, Qwen3-32B 0.00): the terminal domain's
heavy JSON scaffold and short shell commands put the corpus squarely in the
pooling-degenerate regime of §8, where the clamped 3-point H∞ cannot separate
generators. The frontier-vs-mid gap is still there — it just lives in the
*companion* signals: directly measured BPC@32K is 1.77 / 1.75 for the frontier
rollouts vs **0.97** for the mid one, and the frontier agents solve in **10–28
turns** vs the mid agent's **172-turn failure loop**. This is exactly why we
carry BPC@32K and turn-count alongside H∞ rather than instead of it: on
content-bearing domains (SWE, search) H∞ separates generators cleanly, but on
scaffold-heavy domains it saturates at the floor and the companions do the
separating. The thesis is unchanged — generator dominates, role does not — but
*which statistic reveals it* is domain-dependent.

**(ii) The content gap.** The benchmark *tasks* we measure agents against are
human-written and dense (median H∞ 1.22); the training data we feed agents is
mostly boilerplate (median 0.26). We evaluate on human richness and train on
machine repetition. The healthy training minority that closes this gap —
frontier rollouts in diverse, real environments, and human action streams — is
exactly the data the probe scores ≥ 0.6 in seconds.

**(iii) The probe is a regime selector, not a recipe selector.** Within the
frontier band H∞ does not separate teachers (DTap Opus/Sonnet/Gemini cluster at
0.71–0.81, inside seed-σ; finding 11/14), and `EVAL_TASK` BPC@32K is flat (1.71)
even where H∞ separates sharply. The probe cleanly tells frontier-or-human from
mid-or-distilled, but within a healthy band you still need a proxy training run.

**Signature clusters** (`figures/fig1_signature_refhinf.png`) make this
concrete: a *healthy long-trajectory* cluster (frontier generators, α 0.26–0.38,
H∞ 0.57–1.63), a *distillation template* cluster (α 0.05–0.26, H∞ ≈ 0), a
*mid-size failure-loop* cluster (longest episodes, H∞ 0–0.08 — long horizon is
*not* high information density; finding 12), and a *compact human-action* cluster
(α 0.40–0.49, H∞ 1.7–1.95, the densest data in the survey, shortest episodes).

---

## 6. Form vs content: a domain-dependent decomposition

The probe also dissects *within* a trajectory by who wrote each token slice
(`figures/fig3_view_decomposition.png`):

- **Web/GUI observations are form.** Stripping rendered HTML/DOM or VLM
  annotations *raises* H∞ (Mind2Web 1.70 full-obs → action view; AgentNet
  0.00 annotated → 1.43 action-only). The boilerplate is in the machine layer;
  the human action stream is pure content.
- **SWE observations are content.** Stripping repository observations *lowers*
  H∞ (JetBrains 1.63 → 0.72 assistant-only): file bodies, diffs, and test output
  *are* the signal. Data-cleaning policy must therefore be domain-specific.
- **Grading the action source.** Human demonstration 1.43 ≫ planner-generated
  0.43 ≫ observation-pooled 0.00 — content density is layered by who authored
  the actions, not by surface format.

This decomposition reconciles findings 15–17 into one statement: **healthy
agentic content = a frontier (or human) generator × scenario diversity × real
grounding**, all three necessary. The same frontier model collapses to the
template floor on a repeated scenario family (DTap deep-tail: Sonnet 0.81 → 0.08;
finding 17), and ×100 SFT dosage does not move the band (finding 13). The image
channel agrees: adjacent-frame pixel change is 2.6%/step on a single-app GUI
(H∞ 0) vs 26.9%/step on cross-app navigation (H∞ 1.55) — observation redundancy
is visible across modalities (finding 19).

---

## 7. Data value is student-relative — and a training test of it

Findings 20–21 reframe "good data" as **student-relative**. A trajectory carries
*form* (the low-β echo skeleton) and *choices* (the H∞ novelty). A weak model
lacks form, so a template-band corpus is the cheapest *form* curriculum; a
frontier model lacks only choices, so only the healthy band helps it. This
explains why OpenThoughts-Agent — a winning small-model SFT recipe — measures
content-low (BPC@32K 1.38, descaffold H∞ ≈ 0.75) yet trains the strongest model
at its scale: it is an excellent *form* curriculum, not an empty one.

We tested the form half directly (`reports/formchoice_results.md`,
`figures/fig7_formchoice_result.png`): Qwen2.5-1.5B SFT'd separately on a
template corpus (OpenThoughts) and a healthy corpus (GLM-4.7/JetBrains/SWE-ZERO),
matched at 30M tokens, evaluated on form vs decision.

| | base | template | healthy |
| :-- | --: | --: | --: |
| **FORM** | 0.78 | **0.99** | **0.96** |
| **DECISION** | 0.033 | 0.025 | 0.017 |

**FORM is confirmed** — SFT on either corpus drives the trajectory skeleton to
ceiling. **DECISION is inconclusive** — all three models, *including untrained
base*, floor at 2–3%, so there is no dynamic range to test healthy > template at
this scale/format. We report it as inconclusive per the pre-registered design.
(A first run was degenerate — a JSON-vs-fenced parser bug — and was caught only
because the *base-model control also floored*; the lesson is logged.)

A complementary filter, **LZ-Select** (`reports/lz_select_results.md`), uses the
same statistics for per-episode selection: on Kwai-Klear, raw H∞ 0.26 → strip
0.36 → select 0.41 → strip+select 0.51, and the kept/rejected split separates
(kept 0.41 vs rejected 0.24) — the probe is actionable at the episode level, not
only the corpus level.

---

## 8. A method note that survived audit

An earlier revision proposed replacing the extrapolated H∞ with directly
measured BPC@32K, after a user challenge ("OpenThoughts shouldn't score 0")
surfaced three real failure modes: a negative-fit clamp (all 37 "zeros" were
clamps), a non-flattening BPC curve (H∞ *not identifiable* in the measurable
window — the closed-form denominator is a noise-dominated second difference,
ill-conditioned), and cross-episode scaffold pooling (shared system prompts
cancel across concatenated episodes, measuring boilerplate density rather than
per-episode content).

The binding methodological decision is to **keep the reference paper's method
exactly**. The reference 3-point clamped H∞ is canonical because it is what was
validated against true neural oracles (Spearman 0.97) and what makes this
registry comparable to the 358-dataset formal-math survey. H∞ ≈ 0 is therefore
read as the reference's *valid* template-degenerate signal, and the pooling
effect is retained as a documented **caveat**, not a replacement: BPC@32K,
score_v3 (bounded 7-point least-squares with a `resolved` flag), and the
descaffold measurement remain **supplementary diagnostics**. The qualitative
structure (healthy vs template vs failure-loop) is invariant across all three
measurements; only the absolute boundary at "exactly 0" is softer than it looks.
**Lesson:** any clamped/extrapolated/pooled "0" must be falsified against a
directly measured quantity and synthetic controls before becoming a finding.

Other limitations: β and Hurst are byte-level proxies (not the token/model-level
definitions of their source papers); several `source` groups are small (human
task n=5, synth task n=2); and the train/eval `role` label is editorial for the
handful of straddling human-demo benchmarks.

---

## 9. Forward experiments

1. **α_D training validation.** Train small models (Cagnetta protocol) on
   healthy-band vs template-band corpora and check whether the measured
   learning-curve exponent matches γ/2β. Confirmation gives the first
   *predictive* theory of agentic data value.
2. **Short-but-fractal → length generalization.** Fix the token budget and
   contrast short high-density human-action data (WebLINX/AgentNet action views,
   1–2 KB/ep, H∞ 1.4–1.95) against long low-density failure loops (aider-flail
   322 KB/ep, H∞ 0) on long-task extrapolation — testing "statistics > window
   length."
3. **Verifiable synthetic code → cross-domain transfer.** Formally verified
   synthetic code is unlimited, zero-human, and tunable in long-range structure;
   test whether natural-language long-task generalization tracks a corpus's
   (Hurst, γ, β) rather than its domain or size.
4. **A real decision test for finding 20.** A ≥7B student and an *in-format,
   sandbox-verified* decision check (held-out next-command that must pass a
   held-out test) to give the metric the dynamic range §7 lacked. The harness
   (`scripts/formchoice/`) is ready to scale.

---

## 10. Conclusion

Merging training and evaluation corpora into one compression-oracle table
separates two axes that the literature usually conflates. **Pattern** (α, β,
Hurst) is the genre signature of the agentic format — a distinct, low-β
statistical phase, nearly constant across the train/eval boundary. **Content**
(reference-exact H∞) is governed by who authored the token stream, falling
monotonically from human task → human demo → frontier → mid/distilled, and this
ordering ignores the train/eval label entirely. The practical consequence is a
measurable **content gap**: we benchmark agents on human-authored richness
(median H∞ 1.22) and train them on machine boilerplate (median 0.26). The cheap
probe cannot pick a teacher inside the healthy band, but it identifies the
healthy band itself in seconds — which is exactly the selection step that turns
"collect more trajectories" into "collect the right ones."

---

### References

- Cagnetta et al., *Neural scaling laws from data correlations* (γ/β; α_D = γ/2β).
- Alabdulmohsin et al., 2402.01825, *Fractal patterns / Hurst and downstream accuracy*.
- *Intrinsic Entropy* (ICLR'26): optimal context length grows with data;
  Bayes-risk fit = the oracle equation used here.
- `reference/data_format.md`: the formal-math survey LZ-oracle protocol
  (358 datasets; LZ↔neural H∞ Spearman 0.97) reproduced bit-for-bit.

*Supporting material: `SAMPLES.md` (full registry + 21 cumulative findings),
`reports/` (openthoughts, formchoice, lz_select, inferredbugs, hinf_clamp),
`figures/FIGURES.md` (figure index), `data/merged_analysis.csv` (per-row table).*
