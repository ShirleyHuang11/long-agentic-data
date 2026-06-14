# BitLesson Knowledge Base

This file is project-specific. Keep entries precise and reusable for future rounds.

## Entry Template (Strict)

Use this exact field order for every entry:

```markdown
## Lesson: <unique-id>
Lesson ID: <BL-YYYYMMDD-short-name>
Scope: <component/subsystem/files>
Problem Description: <specific failure mode with trigger conditions>
Root Cause: <direct technical cause>
Solution: <exact fix that resolved the problem>
Constraints: <limits, assumptions, non-goals>
Validation Evidence: <tests/commands/logs/PR evidence>
Source Rounds: <round numbers where problem appeared and was solved>
```

## Entries

## Lesson: nested-monoid-needs-recall-cue
Lesson ID: BL-20260613-recall-cue
Scope: case/phase/data_nested_monoid.py (holographic nested_monoid task)
Problem Description: A compositional AR task where each step applies one of several previously-defined ops to a running register is ILL-POSED for next-token prediction if the sequence does not encode WHICH op is applied at each step. The model cannot predict the step result; train/eval accuracy is stuck at the marginal. Symptom also: gamma-beta correlation estimator ||C(n)|| ~ 0 for all (β,γ) because nothing in the stream repeats at the dependency distance.
Root Cause: The op-application order was sampled (β-Zipf) but never emitted as tokens; long-range dependency existed only as an unobserved latent. No token identity repeats across the dependency distance, so pairwise correlation is unmeasurable.
Solution: Emit a recall CUE token at each use — `USE <name>` then the result — exactly like AlgorithmicKVGenerator re-emits the recalled key at query time. The name token repeats (definition + use) at the β-distributed distance, making the task well-posed AND creating the long-range structure. Interleave DEF/USE (register-machine) so recall distances span a measurable range and grow with sequence length.
Constraints: Requires a name-token pool >= max ops per example (n_names=512 → vocab 552). Names unique within an example, reused across examples.
Validation Evidence: knob_verify gate (median recall-distance decreases monotonically with β; filler fraction & entropy increase with γ). Unit test_nested_monoid::test_answer_is_correct_affine_fold replays applied (a,b) on x0.
Source Rounds: 0

## Lesson: synthetic-data-gate-use-structure-not-corr-exponent
Lesson ID: BL-20260613-gate-instrument
Scope: case/phase/knob_verify.py (β,γ knob verification gate)
Problem Description: Verifying (β,γ) knobs on synthetic algorithmic streams by fitting ||C(n)||_op ~ n^{-β} (gamma-beta.pdf Eq.7) and H_n-H_inf ~ n^{-γ} gives noise: β̂ ~ 0 and inverted ordering even for the PROVEN KV task; γ̂ ~ 10 (nonsense) from high-order n-gram undersampling (vocab^order >> tokens).
Root Cause: The gamma-beta exponents are estimable on natural language (strong, clean correlations) but NOT on sparse synthetic streams dominated by structural/noise tokens; high-order plug-in entropy is undersampled.
Solution: Gate on the ground-truth structural quantities the knobs directly control: β via the realized recall-distance distribution (median/quantiles, monotone in β on a β-axis at fixed γ); γ via filler fraction (≈γ) and order-1/2 entropy LEVEL (monotone in γ on a γ-axis at fixed β). Isolate each knob on its own axis (anchors confound β and γ). Keep ||C(n)|| / entropy-decay estimators only as secondary reported diagnostics.
Constraints: Validates that knobs MOVE structure monotonically; does not claim the realized exponent equals the nominal one.
Validation Evidence: knob_verify._run_gate prints β-axis and γ-axis tables with PASS/FAIL on monotonicity.
Source Rounds: 0

## Lesson: retention-ratio-cheat-guard
Lesson ID: BL-20260613-retention-guard
Scope: any length-generalization sweep using retention = acc(L_long)/acc(L_short)
Problem Description: A (β,γ) cell showed "high retention" (0.30–0.80) suggesting great length
generalization, but it was an artifact: those cells had low train-length accuracy (0.18–0.31), so
the small denominator inflated the ratio while ABSOLUTE long-length accuracy was the WORST (0.08–0.20).
Trusting raw retention would have produced a false "holographic ridge" / false "truncated > holographic".
Root Cause: retention is a ratio; when the model never learned the task at train length, the ratio is
dominated by noise/low base rate, not by genuine extrapolation.
Solution: ALWAYS cheat-guard: (1) only trust retention where train_acc >= ~0.90; (2) always report
ABSOLUTE long-length accuracy alongside retention; (3) report train_acc as a gate. The genuine signal
lives in absolute long-length accuracy within the learnable region.
Constraints: threshold 0.90 is a heuristic; tune per task.
Validation Evidence: Phase B grid — guarded view removed the small-β/high-γ pseudo-ridge; results/holo_phaseB_grid.md.
Source Rounds: 0

## Lesson: pure-pytorch-mamba-scan-too-slow
Lesson ID: BL-20260613-mamba-scan-slow
Scope: case/phase/model_mamba.py (sequential-scan MambaCausal) in length-gen sweeps
Problem Description: Mamba eval at L>=1024/2048 with the pure-pytorch sequential scan is prohibitively
slow (3/12 runs in ~5h); a full 4-anchor×3-seed grid would need ~20h A100. Also undertrained at 2000
steps (acc@256=0.62<1.0), making Tx-vs-Mamba comparisons unreliable/contradictory across horizons.
Root Cause: O(L) python loop over timesteps × layers, no CUDA kernel; SSM also needs more steps to fit.
Solution: Do NOT run pure-pytorch Mamba in long-eval sweeps. For architecture comparisons either (a) cap
eval length and budget for more steps, or (b) use mamba_ssm CUDA kernels, or (c) cite the prior clean
N=3 KV result (commit d371e5a). Mark such a control INCONCLUSIVE rather than burning ~20h GPU.
Constraints: applies to the minimal pure-pytorch scan; CUDA-kernel Mamba is fine.
Validation Evidence: holo_C_mamba (eval≤2048) + holo_C_mamba_fast (eval≤1024) both stalled at 3/12; cancelled.
Source Rounds: 0
