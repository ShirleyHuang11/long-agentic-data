# Round 0 Summary — Holographic Length-Generalization

**State: COMPLETE — implementation + all experiments + analysis + report done.**
Headline: holographic/edge-of-chaos length-gen hypothesis (H1) is REFUTED for a
vanilla decoder-only Transformer across Phase A (anchors), Phase B (6×6 grid,
cheat-guarded), and Phase D (100M). H2 partially supports holo>trunc on absolute
accuracy. H3 (Mamba) inconclusive (pure-pytorch scan too slow). See
reports/holo_length_gen.md.

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

## Results (Phases A–D, all committed to results/ + reports/)
- **Phase A** (anchors N=3): retention Natural .250 ≈ Abyss .247 > Edge .200 > CoT .187 → Edge NOT best.
- **Phase B** (6×6 grid, cheat-guarded): flat retention ~.18–.24, no ridge; small-β/high-γ "ridge" is a
  low-train-acc artifact. Heatmap PNG saved.
- **Phase D** (100M N=3): same ordering, uniformly higher retention (.26–.33) → H1 refuted at scale.
- **H2** (holo vs truncated, matched budget): holo > trunc on absolute L=2048 acc (trunc barely learns;
  its high retention is the same artifact). Partial support for holo's "microcosm > slice".
- **H3** (Tx vs Mamba): INCONCLUSIVE — pure-pytorch Mamba scan too slow (3/12 in 5h) + undertrained;
  defer to prior clean N=3 KV result (commit d371e5a). GPU freed.

Routing solved: kempner QOSMaxGRESPerUser + ~220 cron jobs (submit cap) → used gpu_test (2-slot QoS)
+ kempner A100 (per user request; backfilled instantly, unblocked Phase D's 40GB need).

## Remaining / follow-ups (not blocking)
Grid is N=1 (anchors N=3) — could confirm ridge-absence at N=3; H3 needs mamba_ssm CUDA kernel + more
steps; affine op_kind and a second task family are future work.

## BitLesson Delta
Action: add
Lesson ID(s): BL-20260613-recall-cue, BL-20260613-gate-instrument,
BL-20260613-retention-guard, BL-20260613-mamba-scan-slow
Notes: recall cue makes the compositional AR task well-posed + measurable; verify
synthetic-data (β,γ) knobs via realized structure (not gamma-beta exponents);
cheat-guard retention with train_acc + absolute long-length acc (caught a false
holographic ridge); pure-pytorch Mamba scan too slow for long-eval sweeps.
