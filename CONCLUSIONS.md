# Conclusions — long-agentic-data

*A plain-language summary of what this project found, what was corrected, and what's next. Full detail in `SAMPLES.md`; reports in `reports/`; figures in `figures/` (`FIGURES.md` indexes them).*

---

## 1. What the project is

A measurement study of **long-horizon agentic data** on HuggingFace —
**133 active datasets/views** (CSV 139 rows) spanning training trajectories,
benchmark task corpora, human demonstrations, and benchmark eval rollouts,
scored with a cheap, tokenizer-free compression oracle plus supplementary
statistics (β correlation-decay, Hurst, seed-σ, image-channel). The training and
evaluation corpora are merged into one analysis in
**`paper/long_horizon_agentic_data.md`** (§7 below); this file is the
plain-language digest. Outputs: findings, figures, a candidate data filter
(LZ-Select), and a form-vs-choices training experiment.

## 2. The metric — and an important correction

**Original idea:** compress the data at growing context sizes, fit
`BPC(n) = H∞ + c·n^(−α)`, read off **H∞** (the irreducible content floor) and
**α** (long-range structure).

**The correction (driven by a user challenge, "OpenThoughts shouldn't score 0"):**
the extrapolated **H∞ was unreliable** — three compounding failures:
1. a clamp turned negative fits into a fake `0.000` (all 37 "zeros" were clamps);
2. the BPC curve often never flattens in the measurable window, so H∞ is
   *not identifiable* by any extrapolator (proved via the closed form
   `H∞ = B₃ − (B₂−B₃)²/[(B₁−B₂)−(B₂−B₃)]` — the denominator is a noise-dominated
   second difference → ill-conditioned);
3. pooling many episodes lets shared scaffolds cancel, measuring boilerplate
   density rather than content.

**Final reconciliation (the binding decision, iter 68 — "use the reference paper exactly"):**
the **reference-exact 3-point clamped H∞ is the canonical content metric**, kept
bit-for-bit identical to the 358-dataset formal-math survey it was validated
against (LZ↔neural H∞ Spearman 0.97). H∞ ≈ 0 is the reference's *valid*
"template-degenerate" signal, not a bug. The earlier "BPC@32K replaces H∞"
correction was an over-correction and was reverted; **`BPC@32K` (directly
measured) and `score_v3` (bounded fit + `resolved` flag) are retained as
*supplementary companions*** — most useful exactly where H∞ is pooling-confounded
(benchmark eval rollouts under a heavy shared harness; see the paper §5.3). On
clean corpora H∞ and BPC@32K agree (Spearman +0.56 overall, higher on clean
data); they diverge precisely on harness-pooled rollouts, which is what flags the
confound.

**Lesson (top of SAMPLES.md):** any "exactly 0" from a clamp, extrapolation, or
pooled measurement must be falsified against a directly-measured quantity
(BPC@32K), the turn-count, and synthetic controls before it becomes a finding.

## 3. The core findings (qualitative structure survived the correction)

1. **Length anti-correlates with content.** The longest episodes are failure
   loops with near-zero content; the densest data is often the shortest.
2. **Agentic data is a distinct statistical phase** (β ≈ 0.2–0.5 vs prose
   0.9–1.4); **code/math sits between** (0.5–0.8) — the LRD "bridge."
3. **Value is set by who wrote each token stream:** humans ≥ frontier ≫
   mid-size ≈ template; no SFT/RL dose mints novelty that isn't there.
4. **Form vs content decomposition flips by domain:** stripping observations
   heals web/GUI data but *removes* content from SWE (repo observations are content).
5. **Healthy data = frontier generator × scenario diversity × real grounding**
   — all three necessary (DTap deep-tail collapses even frontier models on
   repeated scenarios).
6. **Data value is student-relative** (finding 20): template data teaches
   *form* (valuable to a weak model), healthy data teaches *choices* (what moves
   a frontier model). OpenThoughts — a strong small-model recipe — is genuinely
   low-content (BPC@32K 1.38, descaffold 0.75) but high-form: an excellent form
   curriculum, not an empty one.

## 4. The training experiment (finding 21) — what we actually proved

Trained Qwen2.5-1.5B separately on a **template** corpus (OpenThoughts) and a
**healthy** corpus, matched at 30M tokens, then evaluated form vs decision.
See `figures/fig7_formchoice_result.png`.

| | base | template | healthy |
| :-- | --: | --: | --: |
| **FORM** | 0.78 | **0.99** | **0.96** |
| **DECISION** | 0.033 | 0.025 | 0.017 |

- **FORM half — CONFIRMED.** SFT on either corpus drives the trajectory
  skeleton to ceiling. Both corpora teach form, as predicted.
- **DECISION half — INCONCLUSIVE.** All three models *including untrained base*
  floor at ~2–3%, so there is no dynamic range to test healthy > template. Per
  the pre-registered design, reported as inconclusive — **not** support or
  refutation.

**Why decision floored:** 1.5B/30M is small; the decision check was deliberately
cross-format (depresses all models); only the deterministic eval subset ran
(sandbox-verifier check deferred); and the template arm isn't actually
content-empty (BPC@32K 1.38), so the contrast was muddier than designed.

**Honesty note:** the first eval run was degenerate (all models floored) — a
harness bug (parser expected ```fenced blocks``` but the format is JSON). It was
caught because the *base-model control also floored*, fixed, and rerun. Without
that control it would have produced a false "finding 20 refuted."

## 5. Bottom line

- The compression oracle is a useful, cheap **data-quality probe** — read the
  reference-exact H∞ for training corpora, and fall back to the directly-measured
  `BPC@32K` + turn-count where H∞ is harness-pooled (benchmark eval rollouts).
- Agentic data is **mostly low-content boilerplate**; the valuable minority
  (frontier rollouts in diverse real environments, human action streams) is
  identifiable in seconds.
- The framework's central hypothesis — **data value is student-relative
  (form vs choices)** — has its *form half empirically confirmed* and a working
  training+eval harness; its *decision half remains an open, now-runnable
  hypothesis*.

## 6. What would settle the open question

A larger student (≥7B) and/or more tokens, an **in-format, sandbox-verified
decision check** (held-out terminus-2 next-command that must pass a held-out
test — gives the metric dynamic range), and a genuinely content-empty template
arm (verified low BPC@32K *and* `resolved≈0`) vs a high-content arm. The harness
(`scripts/formchoice/`) is ready to scale up directly.

## 7. The merged train+eval analysis (`paper/long_horizon_agentic_data.md`)

Merging the training and evaluation corpora into one table (133 active) gives the
project's headline result, dissociating two axes the literature usually conflates:

- **Pattern is the agentic *format* signature.** α is role-invariant (median 0.25–0.34
  across train / eval-task / eval-traj), and agentic data is a distinct
  correlation-decay phase (β ≈ 0.2–0.5) vs prose (1.1–1.4), with code/math the
  bridge (0.5–0.8). β tracks the *model-generated* bulk; human demos decorrelate
  faster (Mind2Web β 0.77).
- **Content is set by the generator, not the train/eval role.** Median H∞ falls
  monotonically with source — human task 1.11 / human demo 1.13 / frontier 0.78 /
  synthetic 0.45 / mid 0.00 / distilled 0.00 — and a variance decomposition makes
  it quantitative: **source η² = 0.33 ≫ role 0.09 ≈ domain 0.07** (source explains
  ~3× more). α and H∞ co-vary (ρ +0.79, both depressed by templating); Hurst is
  the one statistic orthogonal to content (ρ −0.02).
- **The content gap.** Benchmark *tasks* are human-authored and dense (median H∞
  1.11); training data is mostly boilerplate (0.27). We test on human richness and
  train on machine repetition.
- **A measurement caveat (the sharpest contribution).** On benchmark eval rollouts,
  H∞ measures the *agent harness* as much as the generator: SWE-bench-Verified reads
  H∞ 0.00 / 0.83 / 1.0–1.2 across the mini-swe-agent / OpenHands / SWE-Router
  harnesses — the same tasks, the ladder tracking shared-prompt weight. Within a
  fixed harness, BPC@32K + turn-count recover the generator ranking where H∞ is flat.
  Multi-agent orchestration is the same effect (finding 23).

The cheap probe is a **regime selector, not a recipe selector**: it identifies the
healthy band in seconds but cannot rank teachers within it (that still needs a proxy
training run).

## 8. Two experiments run for this paper (the substantive validations)

Late in the project we ran the two experiments the analysis had been pointing to —
the loop's biggest contributions, both with full honesty about what does and doesn't hold.

### 8a. Does the metric measure what we claim? (LZ-vs-neural, §8 + Figure 8)

A Qwen2.5-0.5B neural oracle over 102 corpora gives **Spearman(neural bits/token, pooled
H∞) = 0.17 — the formal-math LZ↔neural 0.97 does NOT transfer to agentic data.** But the
divergence is diagnostic, not disqualifying: the neural estimate tracks the *finite-context*
LZ series (BPC@128/2048/32K = 0.49/0.59/0.54) and **collapses only against the ∞-context
extrapolated H∞**, with the per-corpus divergence scaling in the context-pooling drop. So
**H∞ measures *cross-episode* incompressibility** — the shared-scaffold repetition that
*defines* agentic template-degeneracy — which a context-bounded neural LM (like BPC@32K)
structurally cannot see. The poster case: `agent-flan` reads H∞ 0 yet the *highest* neural
bits/token measured (3.29) — varied episodes, one repeated scaffold. This **independently
confirms the §5.3 harness-pooling story** and refines the headline: the source effect is
specifically *cross-episode diversity* (within-window neural η²(source)=0.03; distill is
locally dense yet H∞ 0). H∞ = 0 means "cross-episode-repetitive," not "neurally empty per
episode" — and cross-episode repetition is exactly what makes training data low-value, so
H∞ measures the training-relevant axis.

### 8b. Does the metric *predict* training value? (α_D = γ/2β, §9 #1 + Figure 9)

From-scratch single-pass training of a 29M GPT on 4 corpora spanning predicted
α_D = α/(2β) (CoderForge 0.93 / SWE-ZERO 0.47 / JetBrains 0.34 / AgentNet 0.05) finds the
**measured data-limited loss-decay exponent rank-correlates with the prediction at
Spearman 0.80** — low-β agentic data is measurably *more sample-efficient*, the direction
the theory predicts. Absolute exponents are compressed (0.15–0.26 measured vs 0.05–0.93
predicted): a single small model over a narrow data range recovers the *ordering*, not the
theoretical scale, and **n=4 is directional, not conclusive**. Still: this is the first
evidence the pattern statistics are not just descriptive but *predictive* of learnability.

## 9. Reflections — the loop, and what it produced

This deliverable was built over ~165 self-paced loop iterations. The arc:

1. **Build (iters 1–~135):** grow the registry to 133 corpora, classify every slug by
   role/domain/source, and assemble the merged pattern+content analysis. The productive
   vein was niche named-org/benchmark searches; the recurring failure mode was per-add
   *number churn* across the paper, eventually tamed by a `paper_stats.py --check-paper`
   drift guard (now asserts every count, table cell, figure caption, and §-ref).
2. **Audit + mine (iters ~136–149):** at data saturation, value shifted from *adding* data
   to *interrogating* it — three cross-cut findings (turns-by-source inversion, H∞-vs-BPC
   discrimination, source-composition-by-role) and a four-confound robustness battery, all
   from columns already in hand. Several iterations caught the loop's *own* drift (sampling
   bias from over-adding content-rich exceptions; a within-domain overclaim a data-add had
   silently created) and turned each into a sharper, more honest claim.
3. **Validate (iters 150–166):** the two experiments above — the loop reaching past the
   exhausted descriptive space to *test its own instrument and its own prediction*.

**Methodological lessons that recurred:** (i) any clamped/pooled "0" is a hypothesis to
falsify, not a finding; (ii) the test for adding data is whether it's an *independent,
balance-improving* sample (skip same-pipeline twins and over-represented domains, even under
an "add more" mandate); (iii) regenerating a figure is not verifying it — *look* at it;
(iv) a finding earns its place by being coherent from the contribution statement down to the
figure caption; (v) when a result is surprising, follow it — the neural "non-transfer" became
the paper's mechanistic account of what H∞ measures. The paper is stronger for having
measured its metric's limits and predictions on the target domain rather than assuming them.
