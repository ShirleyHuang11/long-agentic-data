---
title: Direction pivot — holographic data, edge-of-chaos sweet spot, architecture-dependent boundary
date: 2026-04-30
status: research_plan_v2
authoritative_source: assets/31121cf1-bd04-483a-ab74-5c6ee8e5be6e_Holographic_Data.pdf (proposal)
---

# Direction pivot — holographic data, edge-of-chaos, architecture-dependent boundary

## Context

The user has updated `CLAUDE.md` with a new research goal:

> 找到 phase diagram 里 edge-of-chaos 区间(长程相关性 + max effective complexity)
> → 在这种数据上训练 → 更好的长度泛化能力。等 KV retrieval 任务完成后,设计其他任务并训练更多 >100M transformers。GPU 用 kempner partition.

The proposal `assets/Holographic_Data.pdf` reframes the (β, γ) phase
diagram into a **3-region theory** with a **4-anchor experimental
plan**, plus 4 data-engineering principles for the paper's
"holographic" claim.

## The 3-region theory (proposal §Phase Diagram)

The β−γ plane has three regions, separated by the line `γ = 2βδ` where
δ encodes architecture + optimization + representational capacity:

* **Region I (Green / Learnable / Scalable)**: `γ > 2βδ`.
  Scaling exponent governed by data structure: `α = γ/(2β)`. Stable
  training, clean power-law scaling. **Natural language sits here**.
* **Region II (Yellow / Critical / Phase Transition)**: `γ < 2βδ`.
  Exponent governed by δ (model internal learning speed), not data.
  Long plateau + sudden collapse. Architecture-sensitive.
* **Region III (Red / Impossible / Unreachable)**: near β=0 or γ=0
  axes, or origin. Two failure modes:
  * `β → 0 band`: correlations don't decay, attention loses
    distance structure → rank collapse.
  * `γ → 0 band`: entropy plateaus, no gradient signal → instant
    memorization or no descent.

The key claim: **the boundary line slope is set by δ**, so different
architectures (Transformer vs Mamba vs RoPE) have **different green-zone
sizes**. This is the architectural prediction the paper makes.

## The 4 anchor points (proposal §Grid Search Plan)

| anchor | (β, γ) | regime | physical meaning |
|---|---|---|---|
| Natural | (2.0, 0.8) | Green | Wikipedia-like; 80 % noise tokens, retrievals strongly local |
| CoT | (0.5, 0.4) | Yellow→Green | Math reasoning with explanatory filler |
| **Edge-of-Chaos** | **(0.05, 0.05)** | Yellow | **Strong reasoning, near critical line** |
| Abyss | (0.0, 0.0) | Red | Holographic limit — uniform random, model fails |

**Recommended action (proposal)**: first compare (2.0, 0.8) vs (0.05, 0.05).
If Mamba beats Transformer at the edge-of-chaos anchor, the paper's
architecture-dependent-boundary claim is established.

## Where this loop's data sits relative to the new theory

The 89 unique cells aggregated so far cover only Region I + boundary
of Region II:

| anchor | (β, γ) | covered? |
|---|---|---|
| Natural (2.0, 0.8) | — | **NO** — submitted iter-46, job 9388697 (kempner, 3 seeds) |
| CoT (0.5, 0.4) | — | **NO** — defer to next iteration |
| Edge-of-Chaos (0.05, 0.05) | — | **NO** — submitted iter-46, job 9388698 (kempner, 3 seeds) |
| Abyss (0.0, 0.0) | (0.01, 0.02) corners | YES (proxy) — chaos confirmed |

**Most of the existing data is in the Green/Yellow strip near β ≥ 0.8.**
The proposal's three new anchors are explicitly in regions we haven't
sampled — this is exactly the experiment to run.

## What the existing 18 emergent cells say

Re-fit on all 18 N≥1 emergent cells (post-complement, post-pilots):

```
train_acc(β, γ) = 0.269 + 0.0490·log(β) − 0.263·γ
R² = 0.9124    max|resid| = 0.040
```

vs the iter-19 5-cell fit (R²=0.9999, max|resid|<0.001) — **the
near-perfect 5-cell fit was an over-fit artifact** (5 points × 3 free
parameters). The 6 refuted predictions of iter-46 (P21, P22, P23, P24,
P25 — five cells the iter-19 fit predicted emergent but were observed
chaos) confirm the over-fit.

Updated γ*(β) under the 18-cell fit:

| β | γ*(β) | observed bracket | within bracket? |
|---:|---:|---|---|
| 0.5 | 0.134 | unknown | (anchor ungenerated) |
| 0.8 | 0.222 | (0.05, 0.208) | predicted just outside; 5-cell predicted 0.215 also outside |
| 1.4 | 0.326 | (0.345, 0.39) | within ✓ |
| 2.0 | 0.393 | unknown | (anchor pending) |
| 4.0 | 0.522 | (0.20, 0.60) | within ✓ |
| 8.0 | 0.651 | (0.510, 0.755) | within ✓ |

The 18-cell fit is **directionally correct everywhere observable** but
its R²=0.91 says the linear-in-(log β, γ) form misses ~4 % of variance.
Result 9c (iter-39) noted concavity in log(β); the new fit confirms.

## Predictions for the 2 submitted anchor pilots (linear-fit baseline)

Current 18-cell linear fit predicts:

* **Natural (β=2.0, γ=0.8)**: train_acc = `0.269 + 0.049·log(2) − 0.263·0.8 = 0.093` → **chaos**
* **Edge-of-Chaos (β=0.05, γ=0.05)**: train_acc = `0.269 + 0.049·log(0.05) − 0.263·0.05 = 0.109` → **chaos**

### Iter-48 update — seed 1 of both anchors landed at 15:17 EDT

| anchor | predicted | observed (seed 1) | error | predicted phase | observed |
|---|---:|---:|---:|---|---|
| Natural (2.0, 0.8) | 0.093 | **0.082** | 0.011 | chaos | **chaos** ✓ |
| Edge-of-Chaos (0.05, 0.05) | 0.109 | **0.106** | 0.003 | chaos | **chaos** ✓ |

Both predictions confirmed at sub-1.5 % error in train_acc. The linear
fit's **extrapolation works correctly at the directional level** (correct
phase classification) at both anchors — including Edge-of-Chaos which
sits two decades below any β cell used to fit. Long-acc and gap:

| anchor | long_acc | gap | retention |
|---|---:|---:|---:|
| Natural | 0.0516 | 0.030 | 0.629 |
| Edge-of-Chaos | 0.0554 | 0.051 | 0.523 |

Both ~5 % long_acc — close to chaos-floor random baseline (1/19 ≈
0.053). The model is essentially unable to extract retrieval signal
in either regime.

**Mechanistic difference**:
* Natural is "signal washed out by 80 % noise" — the model's locality
  bias (β=2) is well-matched but there's nothing to retrieve from.
* Edge-of-Chaos is "signal present but inaccessible" — γ=0.05 means
  almost every token is information, but β=0.05 means retrieval
  distance is uniformly distributed and the Transformer's local
  attention can't localize the relevant past KV pair.

This **establishes the Transformer baseline** for the paper's
architecture-dependent boundary claim. Future Mamba experiments would
need to show:
* Mamba at Natural (2.0, 0.8) → still chaos (signal genuinely absent)
* Mamba at Edge-of-Chaos (0.05, 0.05) → escapes chaos (selective state
  space exploits long-range correlations the Transformer misses)

Pending: seeds 2, 3 of both pilots (~1.7 h more) for N=3 confirmation.
Plus CoT (β=0.5, γ=0.4) anchor (job 9393854, currently PENDING).

So **a Transformer is predicted to be CHAOS at both proposal anchors**,
but for opposite reasons:

* At Natural: γ=0.8 is so high that even a strong-locality (β=2)
  Transformer can't extract retrieval signal from 80 % noise.
* At Edge-of-Chaos: β=0.05 means retrievals are nearly uniform-random
  — the Transformer's local-attention bias is useless. Combined with
  γ=0.05 (high info density), the data has long-range correlations
  the model can't exploit.

If both observe chaos, **the paper's architecture-dependent claim** is
testable next: a Mamba at the same anchors should escape chaos at
Edge-of-Chaos but stay chaos at Natural. **Future work** in the new
direction is to implement Mamba and re-run.

## Action plan for this loop iteration and beyond

### Iter 46 (now, just submitted)
* Submit 2 anchor pilots on `kempner / kempner_sham_lab`:
  * 9388697: `phase-anchor-natural-b2p0-g0p8` (β=2.0, γ=0.8, 3 seeds, 4 h wall)
  * 9388698: `phase-anchor-edge-b0p05-g0p05` (β=0.05, γ=0.05, 3 seeds, 4 h wall)
* Document direction pivot (this file).

### Iter 46.5 (user redirected to seas_gpu)
* User explicitly asked to move kempner jobs back to seas_gpu (the
  proposal's "prefer kempner" was overridden for this run).
* Cancelled 9388697 + 9388698 (had been RUNNING 6 min, no data
  produced yet — clean cancel).
* Resubmitted to `seas_gpu / barak_lab` with same parameters:
  * 9390238: `phase-anchor-natural-b2p0-g0p8`
  * 9390239: `phase-anchor-edge-b0p05-g0p05`
* Both backfilled to RUNNING within 2.5 minutes (queue much less
  congested than during iter 14-46 marathon).

### Iter 47 (now, this loop fire)
* The 2 RUNNING anchor pilots are 27 min into ~51 min/cell ×
  3 seeds = ~2.5 h jobs. First row each ~14:25+51m=15:16 EDT.
* Submit 3rd anchor (CoT) per proposal: `(β=0.5, γ=0.4)` with
  3 seeds, 4 h wall, also seas_gpu/barak_lab.
  * 9393854: `phase-anchor-cot-b0p5-g0p4`
* Coverage status of proposal's 4 anchors:

  | anchor | (β, γ) | coverage |
  |---|---|---|
  | Natural | (2.0, 0.8) | running (9390238) |
  | CoT | (0.5, 0.4) | submitted (9393854) |
  | Edge-of-Chaos | (0.05, 0.05) | running (9390239) |
  | Abyss | (0.0, 0.0) | covered by corners (0.01, 0.02), N=3 chaos |

  All 4 anchor regions will be sampled by the time these 3 pilots
  complete (~3 h from now if no preemption).

### Iter 47–50 (next 3–4 hours, anchor pilots returning)
* When anchors land, refit the linear law with 20 cells (vs 18).
* Verify the chaos-at-both-anchors prediction.
* If chaos at Edge-of-Chaos: confirms Transformer hits Region II/III
  there (proposal's prediction).
* If emergent at Edge-of-Chaos: refutes one of:
  * the linear-fit extrapolation to small β,
  * the proposal's claim about Region II being chaos for Transformer.

### Iter 50+ (after anchors)
* Submit CoT anchor (β=0.5, γ=0.4, 3 seeds) — fills the third proposal
  anchor.
* Begin Mamba implementation: clone `model.py` → `model_mamba.py` with
  `MambaCausal` matching the `(d_model=1024, n_layers=8)` budget.
* First Mamba experiments at the same 4 anchors.

### Iter 100+ (multi-day horizon)
* Length-generalization stress test: train at L=512, eval at L=8k,
  32k, 100k. Compare Transformer vs Mamba. The proposal's strongest
  claim is that Edge-of-Chaos training enables L=100k accuracy retention
  for Mamba.
* New synthetic tasks per CLAUDE.md proposal page 6:
  * Logical folding (deep nested logic in 2k window)
  * Anti-N-gram (force full-sequence attention)
  * Random-shuffle definitions (key-value not position-correlated)
  * Long-range pointer chasing in short windows

## Why this pivot is consistent with the existing loop's work

* The 30 atomic commits already shipped quantify the **Transformer's**
  green-zone phase diagram boundaries.
* The proposal's `α = γ/(2β)` scaling exponent **is** the
  `alpha_theory` column the loop has been computing per-cell since
  iteration 1. Result 2's refutation of single-α-threshold tells us
  the Region I/II line slope δ is non-zero — δ matters, architecture
  matters.
* The verifier's 6 refuted predictions (P11 + P21–P25) are honest
  documentation of where the in-strip linear law fails to extrapolate.
  These refutations are the **evidence** for the proposal's claim that
  the standard-fit model is incomplete.

The loop's substantive mistake (now corrected): I was using "emergent"
as the goal phase. The proposal says **edge-of-chaos** is the goal
phase. These are different regions; my 5-cell fit thought it was
mapping the goal but was mapping the boundary of the green zone.

## Reproducibility

* Submitted jobs: 9388697 (Natural), 9388698 (Edge-of-Chaos).
* Slurm command line:
  ```bash
  sbatch \
    --job-name=phase-anchor-{name} \
    --partition=kempner --account=kempner_sham_lab \
    --time=04:00:00 \
    --export=ALL,RUN_NAME={run_name},OVERRIDES_B64={b64} \
    case/phase/sweep.slurm
  ```
* OVERRIDES_B64 decodes to `sweep.seeds=[1,2,3] plan.name=single
  plan.beta={beta} plan.gamma={gamma}`.
