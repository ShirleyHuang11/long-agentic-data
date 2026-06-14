# Round 0 Summary — Holographic Length-Generalization

**State: implementation COMPLETE + validated; GPU campaign LAUNCHED; results PENDING.**
This round is not yet "plan complete" — Phases A–D experiments are running on the
cluster and analysis/reports (plan Tasks 11–14) follow as results land.

## What Was Implemented (plan Tasks 1–10, all committed)

Added `nested_monoid` as a third (β,γ) task in `case/phase/` (following the existing
kv / logical_folding convention — NOT a separate `case/recursion/`):
- `model_rope.py` — RoPE decoder-only Transformer (length-extrapolation-friendly),
  `transformer_rope` arch key.
- `data_nested_monoid.py` — interleaved affine/perm **register machine with named-op
  recall** (`DEF name idx` / `USE name → result`). β = Zipf recall recency, γ = filler.
  Default `op_kind=perm` (learnable permutation-pool lookup), `affine` optional.
  Modes: holographic / truncated (H2 control).
- `knob_verify.py` — realized-structure §3.4 gate + secondary ‖C(n)‖/entropy diagnostics.
- `phase_core.py` edits: make_generator(nested_monoid), masked_next_token_acc,
  answer-masked evaluate_at_length, transformer_rope branch.
- `phase_sweep.py` edit: configurable `min_non_embed_params` floor.
- `data_generator.py`: holo_anchors / holo_grid / holo_grid_at_beta plan factories.
- Configs: holo_small (RoPE 19.2M), holo_100m (101.3M), holo_mamba_small (13.4M).
- `holo.slurm` (flexible runner) + `submit_holo.sh` (campaign launcher).

## Files Changed
Created: model_rope.py, data_nested_monoid.py, knob_verify.py, holo.slurm,
submit_holo.sh, configs/models/holo_{small,100m,mamba_small}.yaml, tests/test_*.py (6),
results/holo_knob_verification.md, results/holo_probe_learnability.md.
Modified: phase_core.py, phase_sweep.py, data_generator.py, .humanize/bitlesson.md.

## Validation (evidence)
- **Unit tests: 24 pass** (`cd case/phase && PYTHONPATH=. python -m pytest tests -q`).
- **§3.4 knob gate: PASS** — median recall distance ↓ in β (24→12→7); filler ↑ in γ
  (0.016→0.163→0.527). (results/holo_knob_verification.md)
- **GPU learnability probe: PASS** — β=0.5,γ=0.2, holo_small, 2000 steps → train-length
  acc **1.000**; graceful length-gen decay (512:0.708, 1024:0.370, 2048:0.193).
  Retention measured on a fully-learned signal. (results/holo_probe_learnability.md)

### Two design bugs found and fixed (gate/smoke doing their job)
1. Ill-posed task (op-application order never encoded) → recall-cue fix (BL-20260613-recall-cue).
2. Unlearnable modular-affine operator (grokking-hard, acc=chance) → permutation-pool
   default; long-range retrieval preserved. Gate instrument corrected to verify realized
   structure, not gamma-beta exponents (BL-20260613-gate-instrument).

## Campaign status (Phases A–D) — routed to gpu_test (idle, separate QoS)
- RUNNING: Phase A anchors ×3 (22532601); grid β=0 row ×3 (22532602).
- DRAINING (background submitter through the submit cap): full grid ×1 seed,
  Phase C truncated-long (H2), Phase C Mamba (H3).
- DEFERRED: Phase D (≥100M) — OOMs on MIG-20GB at L=2048; needs an H100 slot.

## Remaining Items (next rounds, plan Tasks 11–14)
Phase A retention ordering (H1 directional); Phase B heatmap + ridge; Phase C H2/H3
verdicts; Phase D scale confirmation; consolidated report reports/holo_length_gen.md.
Blocker: kempner QOSMaxGRESPerUser + ~220 queued cron jobs (QOSMaxSubmitJobPerUserLimit).

## BitLesson Delta
Action: add
Lesson ID(s): BL-20260613-recall-cue, BL-20260613-gate-instrument
Notes: recall cue makes the compositional AR task well-posed + measurable; verify
synthetic-data (β,γ) knobs via realized structure, not gamma-beta exponents.
