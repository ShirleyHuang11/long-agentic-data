# Goal Tracker

## IMMUTABLE SECTION
<!-- Do not modify after initialization -->

### Ultimate Goal
Build "holographic" synthetic pre-training data + the harness to map the (β,γ)
phase diagram of **length generalization** (train short/shallow → test long/deep)
for a vanilla decoder-only Transformer trained with pure next-token prediction,
locate the holographic/edge-of-chaos band, and test it against two controls
(holographic-short vs truncated-long; Transformer vs Mamba). Spec:
docs/superpowers/specs/2026-06-13-holographic-length-gen-design.md.

### Acceptance Criteria
- **AC1** Holographic generator (`nested_monoid` task) producing well-posed AR
  reduction/recall traces with continuous (β,γ) knobs; deterministic; unit-tested.
- **AC2** Standard decoder-only Transformer (RoPE) trainable via the existing
  NTP harness on this task; Mamba control available; small + ≥100M presets.
- **AC3** Answer-masked length-generalization metric (retention =
  acc(D_long)/acc(D_short)) wired into the sweep CSVs.
- **AC4** §3.4 knob-verification gate: realized structure tracks (β,γ) monotonically.
- **AC5** Phase A anchors retention ordering produced (Transformer, grid tier).
- **AC6** Phase B 6×6 (β,γ) retention heatmap + ridge analysis.
- **AC7** Phase C controls: H2 (holographic-short vs truncated-long) + H3 (Mamba).
- **AC8** Phase D ≥100M scale confirmation + consolidated report with H1/H2/H3 verdicts.

---

## MUTABLE SECTION

### Plan Version: 1 (Updated: Round 0)

#### Plan Evolution Log
| Round | Change | Reason | Impact on AC |
|-------|--------|--------|--------------|
| 0 | Code lives in `case/phase/` as task `nested_monoid` (not new `case/recursion/`) | Follows the existing task-dispatch convention (kv, logical_folding); DRY | none (spec §7 intent preserved) |
| 0 | Primary model = RoPE (`model_rope.py`), not APE `model.py` | APE far positions untrained → confounds length-gen; spec text said RoPE | strengthens AC2/AC3 |
| 0 | Generator redesigned: separate-blocks fold → interleaved register machine with **named-op recall cue** (`DEF name idx`/`USE name→result`) | Original was ill-posed (op-application order never encoded) and had no measurable long-range structure; recall cue fixes both (BL-20260613-recall-cue) | required for AC1/AC4 |
| 0 | Default operator = permutation-pool lookup (`op_kind=perm`), affine optional | Modular affine over in-context params is grokking-hard → unlearnable; isolates length-gen from arithmetic (spec §9) | required for AC2/AC5 |
| 0 | Gate verifies via realized recall-distance + filler fraction, not `‖C(n)‖`/entropy-decay exponents | gamma-beta exponents unestimable on sparse synthetic streams (fail even on proven KV) (BL-20260613-gate-instrument) | AC4 |

#### Active Tasks
| Task | Target AC | Status | Tag | Owner | Notes |
|------|-----------|--------|-----|-------|-------|
| (none — all mainline tasks resolved) | - | - | - | - | H3 closed as INCONCLUSIVE (pure-pytorch Mamba too slow + undertrained) |

### Blocking Side Issues
| Issue | Discovered Round | Blocking AC | Resolution Path |
|-------|-----------------|-------------|-----------------|
| kempner partitions QOSMaxGRESPerUser-capped for this user | 0 | AC5-AC8 | route GPU jobs to SEAS (chen_lab_seas) / gpu_requeue; or kempner_barak_lab |

### Queued Side Issues
| Issue | Discovered Round | Why Not Blocking | Revisit Trigger |
|-------|-----------------|------------------|-----------------|
| filler fraction ≈ γ/(γ+(1-γ)·3.5) < γ (per-step convention) | 0 | knob is monotone (matches KV convention); axis uses nominal γ | if phase map needs realized-γ axis |
| `‖C(n)‖`/γ-decay estimators noisy on synthetic data | 0 | kept as secondary diagnostics only; gate uses structural measures | if a reviewer requires exponent-based verification |

### Completed and Verified
| AC | Task | Completed Round | Verified Round | Evidence |
|----|------|-----------------|----------------|----------|
| AC1 | nested_monoid generator | 0 | pending | test_nested_monoid (8 tests) pass |
| AC2 | RoPE model + presets + Mamba | 0 | pending | test_model_rope (4); presets build 19.2/101.3/13.4M |
| AC3 | masked answer-eval + retention CSV | 0 | pending | test_phase_core_masked (2); smoke CSV has retention_ratio |
| AC4 | knob-verification gate | 0 | pending | knob_verify.py GATE PASS (results/holo_knob_verification.md) |
| AC5 | Phase A anchors retention (H1) | 0 | pending | results/holo_phaseA_anchors.md — Edge NOT best; H1 refuted directionally |
| AC6 | Phase B 6×6 grid + heatmap (H1) | 0 | pending | results/holo_phaseB_grid.md + heatmap.png — flat, ridge is artifact; H1 refuted |
| AC8 | Phase D 100M + report | 0 | pending | results/holo_phaseD_scale.md — H1 refuted at scale; reports/holo_length_gen.md |
| AC7 (H2) | holographic vs truncated | 0 | pending | results/holo_phaseC_controls.md — holo>trunc abs acc (trunc retention artifact) |
| AC7 (H3) | Transformer vs Mamba | 0 | pending | INCONCLUSIVE — pure-pytorch Mamba scan too slow + undertrained; defer to N=3 KV (d371e5a) |

### Explicitly Deferred
| Task | Original AC | Deferred Since | Justification | When to Reconsider |
|------|-------------|----------------|---------------|-------------------|
| (none) | - | - | - | - |
