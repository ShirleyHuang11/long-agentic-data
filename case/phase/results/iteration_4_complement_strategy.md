---
title: Iteration 4 — high-β complement sweep, parameterized standard factory
date: 2026-04-27
status: submitted_pending
slurm_jobs: [9008462]
authoritative_data: case/phase/runs/standard_complement/run_summary.csv (will populate)
---

# Iteration 4 — high-β complement sweep

## 1. Why a complement?

Iteration-2 wall-time projection: the running `standard` job (8493171) started
2026-04-27 14:29 EDT, will time out at hour 48 with **~38 % of 147 expected
rows**. Iteration order is β-major from β=0.1, so by hour 48 it will have
covered roughly β ∈ {0.1, 0.2, 0.4} only. The high-β region — where
`corners` already showed the only emergent cell `(β=8, γ=0.02)` — will
**not be reached at all** in the original sweep.

Re-submitting `standard` with reversed iteration order would require either
killing the running job (destructive) or modifying iteration logic
(invasive). Submitting a complementary sweep that ONLY covers the high-β
half is non-destructive and runs in parallel.

## 2. Design: tile, don't duplicate

`standard_grid` uses `np.geomspace(beta_lo, beta_hi, n_beta)` and
`np.linspace(gamma_lo, gamma_hi, n_gamma)`. Verified:

```
plan.beta_range=[0.1, 6.4]  n_beta=7  →  β = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4]
plan.beta_range=[0.8, 6.4]  n_beta=4  →  β = [0.8, 1.6, 3.2, 6.4]
```

The 4 β values of the complement are **exactly the upper half** of the
running standard's 7-β grid. With identical γ values, the two sweeps' cells
are aligned on the same 7×7 lattice — no duplicate-and-discard waste,
**no overlap with what the running job will reach by timeout**.

Submitted (job 9008462):

```
plan.name=standard
plan.n_beta=4 plan.n_gamma=7
plan.beta_range=[0.8, 6.4] plan.gamma_range=[0.05, 1.0]
sweep.seeds=[1]
```

Cost: 28 cells × 1 seed × ~51 min/row = **~24 h on A100**. Fits with
margin in 2-day wall limit.

## 3. Combined coverage at hour 48 (projected)

| region | source | seeds |
|---|---|---:|
| β ∈ {0.1, 0.2, 0.4} × γ all 7 | running `standard` (8493171) | 3 (full) |
| β ∈ {0.8, 1.6, 3.2, 6.4} × γ all 7 | complement (9008462) | 1 |
| α=0.1, 0.4, 1.0 critical lines | running `alpha_iso_*` | 3 (full) |
| β log-sweep at γ=0.3 | running `beta_axis_g0p3` | 3 (full) |
| γ linear-sweep at β=0.4 | running `gamma_axis_b0p4` | 3 (full) |
| (β=8, γ=0.02), (β=0.01, γ=0.02), (β=0.01, γ=0.95) | running `corners` | 3 (full) |
| β=γ^p, p∈{2,3} | running `fast_beta_*` | 3 (full) |
| 5×5 around (0.4, 0.3) | running `refine_b0p4_g0p3` (~75 % at timeout) | 3 (partial) |
| 5×5 around (2.0, 0.3) | running `refine_b2p0_g0p3` (~75 % at timeout) | 3 (partial) |

The full 7×7 standard grid is recoverable (low-β with seeds, high-β
single-seed). Sufficient for a phase-diagram figure with error bars in the
chaos region and point estimates in the emergent region.

## 4. Code change required

Single-source change to `data_generator.py`:

```python
def _standard_from_cfg(c) -> SweepPlan:
    kwargs = {}
    if hasattr(c, "n_beta") and c.get("n_beta") is not None:
        kwargs["n_beta"] = int(c.n_beta)
    # ... same for n_gamma, beta_range, gamma_range
    return standard_grid(**kwargs)

_FACTORIES["standard"] = _standard_from_cfg
```

Backwards compatible: with no overrides, returns the same 49-pair grid as
before. Verified by smoke test:

```
$ python case/phase/data_generator.py plan.name=standard
=== standard_grid (49 pairs) ===
  βs (7): [0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4]
  γs (7): [0.05, 0.208, 0.367, 0.525, 0.683, 0.842, 1.0]

$ python case/phase/data_generator.py plan.name=standard \
    plan.n_beta=4 plan.n_gamma=7 'plan.beta_range=[0.8,6.4]'
=== standard_grid (28 pairs) ===
  βs (4): [0.8, 1.6, 3.2, 6.4]
  γs (7): [0.05, 0.208, 0.367, 0.525, 0.683, 0.842, 1.0]
```

## 5. Why not also seed the complement at 3?

3 seeds × 28 cells × 51 min = ~71 h. Exceeds wall limit. With 1 seed we
get full coverage in 24 h; if any high-β cell shows non-chaos behavior
(very likely given `corners` evidence), we add seeds for **just that cell**
in iteration N+1 — an order-of-magnitude cheaper than re-seeding all 28.

## 6. Risk and rollback

* If 9008462 fails at startup: `.err` file will show the cause; the change
  is contained to `_FACTORIES["standard"]` and reverts cleanly.
* If running `standard` completes faster than projected (e.g. node is
  faster than my 51 min/row estimate): mild duplicate compute on the upper
  half — wasteful but not destructive.
* If priority queue holds 9008462 too long (>24 h): the complement won't
  finish by the time iteration-N decisions need to be made. Mitigation:
  if pending past hour 36, demote priority of refine_* (lowest impact)
  via `scancel` — only with user authorization.
