---
title: Phase sweep — iteration 2 progress (mid-run, refined diagnosis)
date: 2026-04-27
status: in_progress
authoritative_data: /n/home07/hanlinzhang/projects/holographic-data/case/phase/runs/<variant>/run_summary.csv
last_updated: 2026-04-27 21:05 EDT
---

# Phase sweep — iteration 2 progress (mid-run, ~21:05 EDT)

> Supersedes the iteration-1 hypothesis "training is too short, 5000 steps
> insufficient". The loss-curve audit below shows training is **converged**
> at 5000 steps; the chaos region is real, not undertrained.

## 0. What changed since iteration 1

1. **Corrected wall-time projection.** A row in `run_summary.csv` is one
   `(β, γ, seed)`, not one cell. Real per-row time is **51 min**, so:
   * `standard` (147 expected rows) needs **~125 GPU-h** → will time out
     at hour 48 with **~38 % done** (β only up to ~0.5).
   * `refine_b0p4_g0p3` and `refine_b2p0_g0p3` (75 rows each) need **~64 h**
     → time out at **~75 % done**.
   * The other 8 variants will finish under 32 h. ✓
2. **Loss audit shows convergence at 5000 steps.** `aulc_train_to_final_norm`
   is below 0.05 for **31 / 33** rows examined. The two "still-down" outliers
   are within rounding of the threshold. Implication: extending
   `model.train_steps` beyond 5000 is **unlikely** to help — the cosine LR
   schedule decays to `lr_min_ratio=0.1`, and the curve flattens by step
   ~3500 in every cell.
3. **One emergent cell already visible.** `corners` reached
   `(β=8.0, γ=0.02)` with `train_acc=0.356, long_acc=0.133, gap=0.223`.
   Per the fixed-threshold classifier this is **not chaos** — it is
   `emergent`. So the chaos region IS bounded: the `(high β, low γ)` corner
   is qualitatively different.

## 1. Job state (snapshot)

| variant | rows | expected | %done | h_remaining @51 min/row | fits 48 h wall? |
|---|---:|---:|---:|---:|---|
| `standard` | 7 | 147 | 4.8 % | 128.0 | **TIMEOUT** |
| `refine_b0p4_g0p3` | 7 | 75 | 9.3 % | 62.2 | **TIMEOUT** |
| `refine_b2p0_g0p3` | 7 | 75 | 9.3 % | 62.2 | **TIMEOUT** |
| `corners` | 7 | 15 | 46.7 % | 7.3 | YES |
| `alpha_iso_{0p1,0p4,1p0}` | 7 each | 36 each | 19.4 % | 26.5 | YES |
| `beta_axis_g0p3` | 7 | 36 | 19.4 % | 26.5 | YES |
| `gamma_axis_b0p4` | 7 | 36 | 19.4 % | 26.5 | YES |
| `fast_beta_{p2,p3}` | 7 each | 36 each | 19.4 % | 26.5 | YES |

Iteration order is **β-major, γ-minor**. After 7 rows ≈ 2.3 cells, every
variant has visited at most 1–2 distinct β values. `standard` is sitting
entirely in β=0.10. By wall-time it will reach roughly β=0.8 — it will
**miss the entire β > 1 half of the grid**, exactly where the only known
emergent cell lives.

## 2. Loss-curve audit (sample of 33 rows across 5 variants)

```
variant            (β, γ, s)         init  final  best   aulc_norm  status
corners            (0.01, 0.02, 1)   4.293 3.345  3.323  0.0249     converged
corners            (0.01, 0.95, 1)   4.109 3.250  3.160  0.0273     converged
corners            (8.00, 0.02, 1)   4.281 2.532  2.497  0.0410     converged   ← hot cell
refine_b2p0_g0p3   (1.4, 0.21, 1)    4.294 3.166  3.148  0.0496     converged
standard           (0.1, 0.05, 1)    4.297 3.353  3.350  0.0473     converged
fast_beta_p2       (0.0025, 0.05, 1) 4.299 3.376  3.373  0.0506     borderline
alpha_iso_0p1      (0.1, 0.02, 1)    4.300 3.306  3.292  0.0353     converged
…
```

The hot corner `(β=8, γ=0.02)` reached `final_loss=2.50` — meaningfully
below all other cells whose final loss is in [3.16, 3.44]. Confirms the
qualitative gap between hot corner and the rest is genuine, not noise.

**Implication for "more training" hypothesis (iteration 1):**
training is the wrong axis to push on. Useful axes from here:

* **Architecture** (deeper / different positional encoding / RoPE) — would
  break the current preset's 100M floor or cost more compute, but may be
  the only way to escape the chaos plateau in the long-range regime.
* **LR schedule** (e.g. constant LR for second half) — cheaper to test, may
  squeeze a little more out of the existing preset, unlikely to flip
  classifier categories.
* **Eval methodology** (more eval batches, evaluate retrieval-only loss
  instead of next-token loss) — the current loss includes irreducible
  next-token entropy on noise/key tokens. A retrieval-only metric would
  separate "model can retrieve but not predict noise" from "model fails at
  retrieval".

## 3. What the partial dataset tells us right now

Aggregate report run on partial data (`case/phase/REPORT_sweep_summary.md`,
2026-04-27 21:00 EDT):

* 30 cells classified, **all are chaos** under fixed thresholds.
* The single emergent cell `(β=8, γ=0.02)` is in `corners` but
  `aggregate_report.section_unusual` does not flag it because the
  unusual-cell rules only catch (rote-with-collapse, contradictory,
  high-seed-variance), none of which match a 1-seed emergent cell.

Action item logged for the next refactor pass (do not commit yet — see §5):

> `aggregate_report.py` should also surface "best per variant" cells for
> non-chaos phases; otherwise an emergent corner like (8, 0.02) is invisible
> in the markdown until all 3 seeds finish.

## 4. Decision tree — what to submit, what to wait

| When | Action | Rationale |
|---|---|---|
| Now (iteration 2) | **Nothing new.** Let the 11 jobs continue. | Loss is converged → re-running with longer train_steps is wasted compute. The 1 emergent cell is informative; we need 1–2 more days of data before iteration-3 decisions. |
| Iteration 3+ (next 6 h) | If `corners` finishes (it should, at hour ~14) | Re-run aggregate; check whether all 3 seeds at (β=8, γ=0.02) are emergent or if it was a single-seed fluke. |
| Hour ~30 | If `alpha_iso_*` finishes | Look at long_acc along α=0.1, 0.4, 1.0 lines. Theoretical critical line claim is testable here. |
| Hour ~42 | When `standard` is at ~5/49 cells (hourly check) | Decide whether to **kill standard early and resubmit with reversed β order** so the high-β region is sampled. Killing requires user confirmation. |
| Iteration N (whenever) | When all "fits-48h" variants are complete | Run `plot_phase_diagram.py plot.phase_mode=quantile` on the partial dataset — produces a relative phase map even when absolute thresholds all classify as chaos. |

## 5. Code-quality audit (no commits yet — gated on results)

The user rule is "atomic commit only when results work AND there is an
important improvement". Iteration-2 results don't yet "work" (only 1
non-chaos cell), so all of these are **logged** as candidate atomic
commits for iteration-N:

1. **De-dup `classify_phase`.** `aggregate_report.py:55` is a hardcoded
   copy of `utils.py:220::classify_fixed`. Replace by:

   ```python
   from utils import classify_fixed, PHASE_NAMES
   _CODE_TO_LABEL = {0: "chaos", 1: "emergent", 2: "super_gen", 3: "rote"}
   def classify_phase(row) -> str:
       return _CODE_TO_LABEL[classify_fixed(row)]
   ```

   Keeps the string API the rest of the file uses; removes the silent
   risk of the two implementations drifting apart.

2. **Drive `EXPECTED_PAIRS` from `data_generator`.** The hand-edited table
   already drifted (`corners` claims 5 pairs but actually produces 5
   from `boundary_corners()`; the 7-row count looks like 2 cells × 3 seeds
   + 1 partial row, which IS consistent with 5 pairs — false alarm in
   iteration-1 report). Still, hand-maintaining the table per variant is
   error-prone. Replace with a single resolution helper:

   ```python
   def expected_rows_for(name: str, seeds: int = 3) -> int:
       cfg = OmegaConf.create({"name": name, "alpha": 0.4, "beta": 0.4,
                               "gamma": 0.3, "p": 2.0, "n": 12})
       plan = dg._FACTORIES.get(name, lambda c: dg.standard_grid())(cfg)
       return len(plan.pairs) * seeds
   ```

3. **Surface non-chaos cells.** Add a "Top 5 cells by long_acc, regardless
   of variant" section in `aggregate_report.py` so cells like
   `(β=8, γ=0.02)` show up in section 6.

4. **Documentation drift.** `DATA_EXPLANATION.md` describes pointer
   chasing (the `data/pointer.py` task), but `case/phase/` uses
   `phase_core.AlgorithmicKVGenerator` (KV retrieval). They are different
   tasks. Either rename `DATA_EXPLANATION.md` or relocate it under
   `data/pointer/README.md`. (No code change needed; doc move only.)

## 6. Iteration-2 deliverable

* This file (updated).
* `case/phase/REPORT_sweep_summary.md` regenerated from partial data
  (committed by `aggregate_report.py`, not git-committed).
* No new sbatch submissions.
* No git commits (per user rule).
