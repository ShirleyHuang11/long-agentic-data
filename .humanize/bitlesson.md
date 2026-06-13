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
