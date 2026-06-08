# Conclusions — long-agentic-data

*A plain-language summary of what this project found, what was corrected, and what's next. Full detail in `SAMPLES.md`; reports in `reports/`; figures in `figures/` (`FIGURES.md` indexes them).*

---

## 1. What the project is

A measurement study of **long-horizon agentic training data** on HuggingFace.
~90 datasets/views scored with a cheap, tokenizer-free compression oracle, plus
supplementary statistics (β correlation-decay, Hurst, seed-σ, image-channel),
turned into findings, figures, and a candidate data filter (LZ-Select).

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

**Fixes adopted:**
- **`BPC@32K` (directly measured)** is now the canonical content metric —
  validated on synthetic controls (random 2.47 / template 0.02 / mixed 1.40),
  robust to the 8K-vs-32K-token context choice (ranking invariant).
- **`score_v3`** (7-point bounded least-squares + `resolved` flag + stderr)
  gives a *correct* H∞ where the curve converges: 39/92 datasets resolve to real
  positive floors (WebLINX 1.86, mind2web 1.68, SWE-bench 1.56, healthy
  trajectories 0.7–1.2); the rest flag `resolved=False` — itself a template signal.

**Lesson (now top of SAMPLES.md):** any "exactly 0" from a clamp, extrapolation,
or pooled measurement must be falsified against a directly-measured quantity and
synthetic controls before it becomes a finding.

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

- The compression oracle is a useful, cheap **data-quality probe** — once you
  use the directly-measured `BPC@32K` and stop trusting clamped/extrapolated H∞.
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
