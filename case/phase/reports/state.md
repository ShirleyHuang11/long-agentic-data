# Phase experiment — periodic state snapshot

> Auto-updated on every /loop fire. One section per loop iteration.

## 2026-04-30 16:31 EDT (cron iter-1, 30-min loop)

### Queue

| job | name | partition | state | elapsed |
|---|---|---|---|---:|
| 9390238 | phase-anchor-natural-b2p0-g0p8 | seas_gpu | RUNNING | 2:06:12 |
| 9390239 | phase-anchor-edge-b0p05-g0p05 | seas_gpu | RUNNING | 2:06:12 |
| 9393854 | phase-anchor-cot-b0p5-g0p4 | seas_gpu | RUNNING | 0:57:43 |
| 9407296 | phase-mamba-smoke-edge | kempner_requeue | PENDING | 0:00 |

### Rows landed

| run dir | rows | predicted phase | observed (latest) |
|---|---:|---|---|
| `anchor_natural_b2p0_g0p8` | 2 | chaos (P26, ta≈0.100) | seed 1: 0.082 / seed 2: 0.087 — **chaos** ✓ |
| `anchor_edge_b0p05_g0p05` | 2 | chaos (P27, ta≈0.115) | seed 1: 0.106 / seed 2: 0.103 — **chaos** ✓ |
| `anchor_cot_b0p5_g0p4` | 1 | chaos (P28, ta≈0.135) | seed 1: 0.128 — **chaos** ✓ |
| `mamba_smoke_edge` | 0 | (smoke; not for science) | not started |

### Predictions scoreboard (27 total)

* confirmed = 16
* observed = 2
* pending = 3 (P7, P9, P16 — alpha_iso_0p1 / beta_axis_g0p3 high-β cells, never landed)
* refuted = 6 (P11, P21, P22, P23, P24, P25)

### Plan.md priorities

* Priority 1 (Mamba): **code complete (commit eb69c9f), smoke submitted (9407296)**
* Priority 2 (Logical Folding new task): not started
* Priority 3 (boundary refinement low-β probes): not submitted (waiting on Mamba smoke result before queue-stacking)

### Next loop fire (16:37 EDT or whenever cron next fires)

* Pull updated rows for 4 jobs
* If Mamba smoke completes → measure actual sequential-scan timing → decide on full-size Mamba submission cadence
* If anchor seed 3 of any pilot lands → re-run aggregate; expect P26/P27/P28 to upgrade from observed/seed-1-confirmed to N=3-confirmed

---

## Notes for whoever reads this later

The 27 predictions verifier file at
`case/phase/results/predictions_verification.md` is the authoritative
scoreboard. The cross-variant scatter and 3-panel continuous figures
at `case/phase/figures/{cross_variant_scatter,phase_continuous}.png`
are auto-regenerated each loop iteration; they currently show 92
unique cells with the weighted γ*(β) boundary fit (R²=0.947).

When seed 3 of all 3 anchors lands (~17:00 EDT), the linear fit will
have 4 anchors × 3 seeds of "true holdout" data outside the strip —
strongest test yet of the boundary's extrapolation behavior.

---

## 2026-04-30 16:47 EDT (cron iter-2)

### Mamba smoke completed (4:26 wall, 9407296)

* COMPLETED state, exit 0
* train_time = 205.5 s for 20 steps → **10.3 sec/step** at d=1024,
  L=32, batch=8, train_len=512
* Forward shape OK, full pipeline (training + eval + plotting) runs
  end-to-end
* train_acc=0.046, long_acc=0.043 (random baseline — only 20 steps,
  expected)

### Mamba full-size at Edge-of-Chaos submitted (9411019)

* `--partition=kempner --account=kempner_sham_lab --time=18:00:00`
  (per CLAUDE.md "GPU prefer kempner" guideline)
* Single cell: (β=0.05, γ=0.05), 1 seed, 5000 train_steps
* Projected wall ≈ 14 h (5000 × 10.3 s = 51,500 s)
* This is THE paper-headline experiment: if Mamba achieves
  train_acc ≥ 0.20 at this anchor, the architecture-dependent boundary
  claim is established (Transformer here is chaos at 0.106).
* PENDING on kempner queue at submission time.

### Queue snapshot

| job | name | partition | state | elapsed |
|---|---|---|---|---:|
| 9390238 | phase-anchor-natural-b2p0-g0p8 | seas_gpu | RUNNING | 2:22 |
| 9390239 | phase-anchor-edge-b0p05-g0p05 | seas_gpu | RUNNING | 2:22 |
| 9393854 | phase-anchor-cot-b0p5-g0p4 | seas_gpu | RUNNING | 1:13 |
| 9411019 | phase-mamba-edge-full | kempner | PENDING | 0:00 |

### Plan.md priority status

* Priority 1 (Mamba): code done; smoke verified; **full-size submitted**
* Priority 2 (Logical Folding new task): not started
* Priority 3 (boundary refinement at low β): deferred — would
  queue-stack on top of Mamba; submit once Mamba lands or one anchor
  pilot completes

### Next loop fire (~17:07 EDT)

* Anchor seed 3 of Natural+Edge should land (~17:00) → P26/P27 N=3
  confirmed
* If Mamba 9411019 not yet started → wait
* If Mamba started → estimate completion time

---

## 2026-04-30 17:17 EDT (cron iter-4)

### N=3 anchor confirmation (P26 + P27 fully observed)

| anchor | (β, γ) | N | train_acc mean | seed-std | predicted | classifier |
|---|---|---:|---:|---:|---:|---|
| Natural | (2.0, 0.8) | **3** | 0.0849 | 0.003 | 0.100 | **chaos** ✓ |
| Edge-of-Chaos | (0.05, 0.05) | **3** | 0.1050 | 0.002 | 0.115 | **chaos** ✓ |
| CoT | (0.5, 0.4) | 2 | 0.1269 | 0.001 | 0.135 | chaos ✓ (N≥1) |

**Inter-seed std 0.002–0.003 across all 3 anchors** — paper-grade
reproducibility. All 3 anchor cells confirmed chaos for the
Transformer; Transformer baseline established.

### Systematic under-prediction at anchors

The 18-cell weighted fit over-predicts train_acc at all 3 anchors by
0.008–0.015 (always positive error). This is consistent with the
iter-39 Result 9c finding that the linear-in-log(β) fit is **concave
in log(β)** — both directions of extrapolation over-predict.

The deepest under-prediction is at Natural (β=2.0, γ=0.8, off 0.015).
Even though the cell is CLASSIFIED correctly, the magnitude error
suggests the linear law's intercept is too high, or the γ-axis slope
is too gentle at large γ.

### New submission: boundary probe

Submitted `phase-probe-b0p2-g0p05` (job 9417079) at (β=0.2, γ=0.05),
3 seeds, seas_gpu, 4 h wall. Fills the missing low-β region between
the 5 N=3 strip cells (β=1.4) and the 1 N=3 corner (β=0.05). With
this point at N=3 in the fit, the log(β) concavity should become
quantifiable.

Picked seas_gpu (queue depth 18) over kempner (queue depth 52);
Natural+Edge just freed 2 seas_gpu slots so this should backfill
quickly.

### Queue snapshot

| job | name | partition | state | notes |
|---|---|---|---|---|
| 9393854 | phase-anchor-cot-b0p5-g0p4 | seas_gpu | RUNNING 1:43 | seed 3 ~17 min away |
| 9411019 | phase-mamba-edge-full | kempner | PENDING | kempner depth 52 |
| 9417079 | phase-probe-b0p2-g0p05 | seas_gpu | PENDING | just submitted |

### Plan.md priorities

* P1 Mamba: code done; smoke verified; full-size submitted (PENDING)
* P2 Logical Folding: not started
* P3 boundary probes: **first probe submitted (9417079)**

### Predictions scoreboard (27 total, unchanged)

confirmed=16, observed=2, pending=3, refuted=6

P26 + P27 still classified as confirmed (already were at seed 1, the
N=3 confirmation is just a stronger version).

---

## 2026-04-30 18:17 EDT (cron iter-8)

### CoT anchor done at N=3

| (β, γ) | N | ta_mean | seed-std | predicted (P28) |
|---|---:|---:|---:|---:|
| (0.5, 0.4) | **3** | 0.1262 | 0.0015 | 0.135 (off −0.009) |

3 anchors all complete at N=3, all chaos as predicted, all
under-predicted by linear fit by 0.008–0.015. Job 9393854 done.

### Boundary probe — first row at (β=0.2, γ=0.05)

| (β, γ) | seed | ta | la | predicted | error |
|---|---:|---:|---:|---:|---:|
| (0.2, 0.05) | 1 | **0.126** | 0.060 | **0.181** | **+0.055** |

The 18-cell weighted fit massively over-predicts at this low-β cell.
Observed ta=0.126, predicted 0.181 — **off by 0.055**, far worse
than the 0.008–0.015 anchor errors.

This **strongly validates Result 9c quantitatively**: the
linear-in-log(β) fit extrapolates too aggressively into low β.
Concrete numbers:

* fit's β-slope: +0.0475 per log(β)
* implied γ=0 intercept at β=0.2: 0.270 + 0.0475·log(0.2) = 0.194
* observed γ=0 implied intercept (using local fit slope -0.184):
  0.126 + 0.184·0.05 = 0.135
* gap of 0.06 in γ=0 intercept

The fit needs either:
* a 1/β saturation term, or
* a piecewise-linear-in-log(β) form with knee around β≈0.5

### Queue snapshot (iter-8)

| job | name | partition | state | elapsed |
|---|---|---|---|---:|
| 9411019 | phase-mamba-edge-full | kempner | PENDING | — |
| 9417079 | phase-probe-b0p2-g0p05 | seas_gpu | RUNNING 55 min | seed 2 in progress |

### Predictions scoreboard (27, unchanged)

confirmed=16, observed=2, pending=3, refuted=6

The probe wasn't pre-registered as a prediction (it's a follow-up
data probe rather than a hypothesis test). Result 9c implications
live in main_findings.md not the verifier scoreboard.

### Plan.md status

* P1 Mamba: full-size still PENDING on kempner (queue depth 52)
* P2 Logical Folding: not started
* P3 boundary probes: 1 of expected ~3 done (seed 1 at β=0.2; 2 more
  seeds + possible β=0.3, β=0.4 probes pending)

---

## 2026-04-30 19:17 EDT (cron iter-12)

### Mamba 9411019 FAILED (OOM on 40GB A100, 30 sec elapsed)

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 MiB.
GPU 0 has a total capacity of 39.49 GiB of which 1.25 MiB is free.
```

The sequential-scan implementation accumulates per-timestep activations
across all 32 layers × T=512 × batch=16. Memory blows up:
~512 MB per layer × 32 layers ≈ 16 GB of dA/dB tensors alone, plus
hidden-state sequence + standard activations → exceeds 40 GB.

### Resubmitted as 9436718 with batch_size 16 → 4

`model.train_batch_size=4 model.eval_batch_size=2` override. 4× less
memory in the inner scan tensors. Same kempner partition, 18h wall.

If batch=4 also OOMs, escalation options:
1. `--partition=kempner_h100` (80 GB H100, 2× headroom)
2. Reduce d_model or num_layers (changes 100M target)
3. Add `torch.utils.checkpoint` to MambaBlock.forward (more compute,
   less memory)

### Probe (β=0.2, γ=0.05) seed 2 landed

| seed | ta | la |
|---:|---:|---:|
| 1 | 0.1257 | 0.0597 |
| 2 | 0.1244 | 0.0612 |

N=2 mean = 0.1250, std = 0.0007. Inter-seed variance trivial.

**Result 9c quantitative confirmation at N=2**: linear fit predicted
0.181 here, observed 0.125 — gap 0.056. Same magnitude as seed-1
finding; not a fluke.

### Queue snapshot

| job | partition | state | elapsed |
|---|---|---|---:|
| 9417079 | seas_gpu | RUNNING 1:55 | probe seed 3 in ~17 min |
| 9436718 | kempner | PENDING | new mamba (bs=4) |

### Plan.md status

* P1 Mamba: smoke OK, full-size **OOMed**, **resubmitted with batch=4**
* P2 Logical Folding: not started
* P3 boundary probes: (β=0.2, γ=0.05) at N=2 confirms iter-8 finding

---

## 2026-04-30 20:17 EDT (cron iter-16)

### Probe (β=0.2, γ=0.05) N=3 complete

| seed | ta | la |
|---:|---:|---:|
| 1 | 0.1257 | 0.0597 |
| 2 | 0.1244 | 0.0612 |
| 3 | 0.1287 | 0.0596 |

**N=3 mean = 0.1263 ± 0.0022, predicted = 0.1809, error = −0.0546**

The fit's error at this cell is **~5× larger** than at the 3 anchors:

| cell | (β, γ) | error |
|---|---|---:|
| Natural | (2.0, 0.8) | −0.015 |
| Edge-of-Chaos | (0.05, 0.05) | −0.010 |
| CoT | (0.5, 0.4) | −0.008 |
| **Probe** | **(0.2, 0.05)** | **−0.055** |

(β=0.2, γ=0.05) sits **between** the β=1.4 strip cells (where slope was
fit) and the β=0.05 corner cell (the only N=3 anchor at low β). The
linear-in-log(β) fit interpolates linearly across this region, but the
true curve is concave — the linear interpolant over-shoots in the
middle. This is exactly the iter-39 Result 9c finding, now **quantified
at N=3 with an interior witness cell**.

For the paper: the boundary detection (γ\*(β) sign) is unaffected
because both fit and observed are firmly in chaos. But the magnitude
calibration is off by 30 % at this cell. A 1/β-saturating intercept
or a piecewise-linear-in-log(β) form would fix this; the existing fit
is honest about its R²=0.947 residuals.

### Queue snapshot

| job | partition | state | elapsed |
|---|---|---|---:|
| 9436718 | kempner | PENDING | (waiting on kempner queue depth ~49) |

Probe 9417079 done. seas_gpu now free for backfill. Mamba still
PENDING on kempner.

### Plan.md status

* P1 Mamba: bs=4 retry submitted (9436718), still queued
* P2 Logical Folding: not started
* P3 boundary probes: **(β=0.2, γ=0.05) N=3 done**, Result 9c
  quantitatively established. Could submit (β=0.3 or 0.4) similarly
  for a richer concavity-fit dataset, but diminishing returns until
  Mamba result lands or paper-narrative shifts.

---

## 2026-04-30 22:14 EDT (resume after pause)

### Second probe (β=0.3, γ=0.05) seed 1 — Result 9c bias is constant

| Cell | Predicted | Observed | Error | N |
|---|---:|---:|---:|---:|
| (β=0.2, γ=0.05) | 0.181 | 0.126 | **-0.055** | 3 |
| (β=0.3, γ=0.05) | 0.200 | **0.142** | **-0.058** | 1 |

The fit's over-prediction is constant at **-0.056 ± 0.002** across two
independent cells in the low-β regime. This is paper-quality
reproducibility of the bias.

Implications:
* The 18-cell linear fit's intercept is too high by ~0.056 in the
  β ∈ [0.05, 1.4] regime
* True train_acc(β, γ=0.05) curve is shifted DOWN ~0.056 from the
  linear extrapolation across the entire low-β region
* The bias is **consistent** (not increasing with β-distance from
  fit), suggesting an additive offset rather than a slope mismatch

Paper-grade fit candidate:

```
train_acc(β, γ) ≈ 0.214 + 0.0475·log(β) − 0.254·γ   for β < 1
                  0.270 + 0.0475·log(β) − 0.254·γ   for β ≥ 1
```

(intercept dropped 0.056 in the low-β regime). A knee at β=1 — which
is also Result 9c's β\* threshold for emergent strip — has the
intuition that "the model retrieves at β ≥ 1, doesn't at β < 1, and
those two regimes have different intercepts".

Will validate when β=0.3 N=3 lands (~2 hours).

### Queue snapshot

| job | partition | state | elapsed |
|---|---|---|---:|
| 9436718 | kempner | RUNNING | 1:50 (~13% of ~14h) |
| 9458760 | seas_gpu | RUNNING | 59 min (seed 2 incoming) |

---

## 2026-04-30 23:51 EDT (probe N=3 — concavity, not step)

### (β=0.3, γ=0.05) N=3 complete

| seed | ta | la |
|---:|---:|---:|
| 1 | 0.1417 | 0.0641 |
| 2 | 0.1371 | 0.0606 |
| 3 | 0.1447 | 0.0599 |

N=3 mean = 0.1412 ± 0.0038, predicted 0.200, **error -0.059**.

### Bias pattern across all N=3 cells at γ=0.05:

| β | observed intercept_at_γ=0 | fit intercept_at_γ=0 | bias |
|---:|---:|---:|---:|
| 0.05 (corner) | 0.118 | 0.128 | **-0.010** |
| 0.2 | 0.139 | 0.194 | **-0.055** |
| 0.3 | 0.154 | 0.213 | **-0.059** |
| 1.4 (strip) | 0.272 | 0.286 | **-0.014** |

**The bias is U-shaped, not a constant offset.** It vanishes at the
fit's anchor cells (β=0.05 corner, β=1.4 strip) and is maximal in the
interior. This **CORRECTS** the iter-24 piecewise-step hypothesis —
the right description is **concavity in log(β)**:

* The fit anchors at the two endpoints (β=0.05 and β=1.4) and
  linearly interpolates across the gap
* The true intercept(log β) is concave (rises rapidly out of the
  corner, levels at the strip)
* Linear interpolation **overshoots** the concave true curve in the
  middle — by ~0.06 at β∈[0.2, 0.3], near 0 at the anchors

Better model: a quadratic in log(β):

```
train_acc(β, γ) ≈ a + b·log(β) + c·log(β)² − 0.254·γ
```

or a saturating form like `1 - α·exp(-β/β_0)`. Either captures the
"diminishing returns at high β + slow build-up at low β" structure.

### Implication for paper

Result 9 (linear 2D fit, R²=0.9999 on 5 cells) was an over-fit
artifact — confirmed by these probes.

Result 9c (concavity in log(β)) is now QUANTIFIED with 4 N=3 cells
spanning β ∈ [0.05, 1.4]. The peak interior bias is 0.06 in train_acc
units, ~30 % of the fit's predicted value at those cells.

Honest paper framing:

* **Linear fit is good for classification** (chaos vs emergent
  prediction is correct at all 4 cells)
* **Linear fit is wrong for magnitude prediction by up to 30 %**
* **The true relation is concave in log(β)** — quadratic or
  saturating form needed for quantitative use

### Mamba 9436718 progress

3:27 elapsed (~25 % of ~14h). Still RUNNING. Will write its single
row at the very end after eval.
