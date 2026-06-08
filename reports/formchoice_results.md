# Form-vs-Choices results (finding-20 discriminator)

*2026-06-08. SFT: Qwen2.5-1.5B-Instruct, two corpora at matched ~30M tokens —
TEMPLATE (OpenThoughts-Agent-v1-SFT) vs HEALTHY (GLM-4.7/JetBrains/SWE-ZERO
thirds). Eval: `scripts/formchoice/eval.py` (deterministic JSON-aware form
parser + gold-tool exact-match decision). Raw: `data/formchoice_eval.json`.*

## Results (3×2)

| model | FORM | DECISION |
| :-- | --: | --: |
| base (no SFT) | 0.783 | 0.033 |
| template (OpenThoughts) | **0.987** | 0.025 |
| healthy (GLM/JetBrains/SWE-ZERO) | **0.960** | 0.017 |
| **Δ healthy − template** | −0.027 | −0.008 |

Form breakdown (fraction passing): trained models ~0.95–0.98 on every
sub-check (valid_json, has_commands, analysis+plan, well-formed keystrokes);
base 0.67–0.77. n_form=60, n_decision=120.

## Harness validation (run 1 was discarded)
Run 1 reported all three models floored (form ~0.25, incl. base) — a harness
bug: the parser looked for ```fenced blocks``` but terminus-2 emits JSON. Fixed
to a JSON-aware parser; run 2 (here) shows form now **discriminates** (trained
0.96–0.99 ≫ base 0.78), confirming the harness works. The base-model control
floored is what flagged the bug — do not trust a Δ when the control is also at floor.

## Verdict: FORM half CONFIRMED · DECISION half INCONCLUSIVE

- **Form (confirmed).** SFT on either corpus drives the trajectory skeleton to
  ceiling (base 0.78 → 0.96–0.99). The template corpus is marginally higher
  (0.99 vs 0.96) — directionally consistent with "template data is a pure form
  curriculum," though the 0.027 gap is within sampling noise at n=60.
- **Decision (inconclusive).** All three models — *including untrained base* —
  score ~2–3% (Δ = −0.008, noise). With no dynamic range above base, the eval
  **cannot test** whether healthy > template on choices. Per the pre-registered
  design: "if neither beats base on decisions, the 30M/1.5B scale is too small …
  report as inconclusive, do not over-claim." So we do not claim finding 20 is
  supported *or* refuted on its load-bearing decision claim.

## Why decision floored (and what would fix it)
1. **Scale.** 1.5B + 30M tokens is small; choosing the correct tool/command on
   novel tasks may simply not register.
2. **Cross-format decision check.** The terminus-2-trained models were asked
   APIGen tool-call-format questions (deliberately out-of-corpus for "no
   template overlap") — a format mismatch that depresses all models, not a pure
   decision signal.
3. **Deterministic subset only.** The design's strongest decision check
   (aider-polyglot, sandbox-verified) was deferred; the implemented exact-match
   gold-tool check is the weakest of the proposed set.

Fix for a real decision test: (a) a larger student (≥7B) and/or more tokens;
(b) an *in-format* decision check — held-out terminus-2 next-command with a
sandbox verifier (does the chosen command pass the held-out test), which gives
the metric dynamic range; (c) implement the deferred aider-verifier check.

## Caveat from the H∞ correction (finding-20 revision)
The "template" arm is **not** content-empty: OpenThoughts BPC@32K = 1.38
(descaffold H∞ ≈ 0.75, healthy band). So the template-vs-healthy contrast is
**muddier than the experiment assumed** — both arms carry real content, which
further limits what a form/decision split here can prove. A cleaner future
design would contrast a genuinely content-empty corpus (verified low BPC@32K
*and* resolved≈0) against a high-content one.

## Bottom line
The pipeline works end-to-end (matched corpora → SFT → discriminating form
eval). It confirms SFT teaches form. It does **not** resolve finding 20's core
claim — the decision metric had no dynamic range at this scale/format. Finding
20 remains a well-motivated hypothesis with a now-validated training+eval
harness ready for a larger, in-format rerun.
