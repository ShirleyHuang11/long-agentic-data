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
