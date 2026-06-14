# Progress — bridging holographic-data into the unified interpretation

**Date:** 2026-06-14
**Goal:** use `holographic-data/` (the companion generative/causal theory repo) to
extend the unified-interpretation framework (the observational/measurement side).
Shared coordinates: **β** (correlation-decay) and **γ**, plus the law **α = γ/2β**.

Steps requested: (1) place real corpora on the holographic (β,γ) phase diagram,
(2) import a predicted learnability axis, (3) widen β coverage, (4) sharpen the
α_D=γ/2β validation using the holographic contrast.

---

## Step 1 — place real corpora on the holographic phase diagram

Holographic empirical boundary (natural log): `γ*(β) = 0.274 + 0.265·ln β`,
emergent/learnable when `γ < γ*`. Predicted learnability (their R²=0.9999 fit):
`train_acc = 0.254 + 0.0523·ln β − 0.197·γ`. Script: `scripts/holographic_bridge.py`
→ `data/holographic_placement.csv`, `figures/fig_holographic_bridge.png`.

**Key finding — the naive bridge FAILS, and that is informative.**
Deriving `γ̂ = 2βα` (from α=γ/2β) gives the *wrong sign*:
- Spearman(γ̂, H∞) = **+0.49** (expected negative)
- Spearman(naive learnability, H∞) = **−0.67** (backwards — content-rich corpora look "high-noise/chaos")

Cause: **the two frameworks use "γ" for different things.** Holographic γ is a
*noise-token rate*; our paper's "γ" in α_D=γ/2β is the *LZ entropy-decay exponent*
(= our `alpha`), which **tracks content** (α↔H∞ = +0.78). So `2βα` makes dense
corpora look noisy. The shared `α=γ/2β` is a coincidence of **form, not a shared γ**.

**Principled bridge:** map holographic noise-γ → our measured noise/boilerplate
rate `scaffold_frac`. This restores the correct *sign* at small n (n=18:
learnability↔H∞ = +0.13) **but collapses to NULL at full coverage (n=85, after
Step 3):**
- Spearman(scaffold-γ, H∞) = **−0.08**, Spearman(learnability, H∞) = **−0.03**
- region × health shows **no separation**: emergent = 10 healthy / 5 pooled;
  chaos = 30 healthy / 27 pooled.

**Verdict: the holographic (β,γ) boundary does NOT transfer to predicting
real-corpus content/health** — even with the principled scaffold-γ mapping. The
+0.13 at n=18 was small-sample noise. Most agentic corpora cluster at low β (the
"agentic phase") on the chaos side, but their content health is unrelated to where
the boundary places them.

## Step 2 — import a predicted learnability axis

Applied `train_acc(β,γ)` with γ=scaffold_frac → per-corpus learnability score
(`data/holographic_placement.csv`). Given the weak bridge (Step 1), this axis is
**exploratory, not validated** — it ranks corpora by predicted length-gen but the
prediction's transfer to real data is not yet established.

## Step 3 — widen β coverage (DONE)

`measure_beta.py` only covered 16–22 corpora. Wrote `scripts/measure_beta_batch.py`
(resilient per-slug try/except + SKIP for loader-broken/OOM slugs, mirroring the
Hurst batch) auto-targeting REGISTRY corpora with α but no β. **Result: 74 added,
β coverage 22 → 90** (1 skipped). Re-running Steps 1–2 at n=85 gave the verdict
above: the principled scaffold-γ bridge is **null** — the +0.13 at n=18 did not
survive the power increase.

### Which proxy makes sense for γ? (the crux)
- ❌ **`2βα`** — wrong construct: it's the entropy-decay/content exponent, not a
  noise rate. Gives backwards results.
- 🤔 **`scaffold_frac`** — the most defensible *available* proxy (fraction of
  repeated task-irrelevant filler; anti-content), right construct — but yields a
  null bridge at n=85.
- ⚠️ A **content-derived γ** (e.g. 1 − normalized H∞) *would* "work" but only
  **tautologically** (healthy = dense = low-γ = emergent).
- 💡 Conclusion: holographic γ is a *generative knob* (signal-vs-noise token split)
  with **no clean, non-circular observational counterpart** in real corpora — which
  is the deeper reason the quantitative bridge does not transfer. A proper γ proxy
  would have to be *constructed* (a within-sequence signal-dilution measure: the
  fraction of context that fails to reduce next-token uncertainty), not repurposed
  from an existing axis.

## Step 4 — sharpen the α_D = γ/2β validation

Measured α_D (log-log slope of `data/alpha_d_*.json` curves) vs predicted α/2β:

| corpus | pred α/2β | measured α_D | H∞ |
|---|---|---|---|
| agentnet | 0.05 | 0.15 | 0.00 |
| apigen | 0.30 | 0.34 | 0.00 |
| coderforge | **0.93** | **0.22** | 0.83 |
| glaive | 0.52 | 0.25 | 1.05 |
| jetbrains | 0.34 | 0.17 | 1.63 |
| smolagents | 0.31 | 0.16 | 1.44 |
| swezero | 0.47 | 0.26 | 0.80 |
| taubench | 0.25 | 0.39 | 0.00 |
| weblinx | **0.93** | **0.29** | 1.95 |

Predicted range is **wide (0.05–0.93)**; measured is **compressed (0.15–0.39)**.
The biggest over-predictions (coderforge, weblinx, jetbrains, smolagents) are all
**content-rich** (high H∞). Holographic contrast explains the looseness:
1. **Controlled vs uncontrolled.** Their 100M-model synthetic experiments *fix*
   content and vary (β, γ_noise) cleanly → R²=0.9999. Our α_D uses a 29M model over
   a narrow D range (250K–3M tokens) on real corpora → exponents compress.
2. **Content inflates α (the formula's γ).** For content-rich corpora the LZ-α is
   driven by genuine content, not by the noise-decay the law assumes, so α/2β
   over-predicts the achievable data-scaling. This is the same pooling/content
   confound that the bridge's γ-conflation (Step 1) exposed.
(The paper's reported Spearman 0.50 used a proper L∞-offset scaling fit; the crude
no-offset slope here gives +0.12 — the *qualitative* compression + content-driven
discordance is the stable message, not the exact coefficient.)

---

## Summary

- The two repos genuinely study the **same (β, γ) structure**, but a naive
  quantitative bridge via the shared `α=γ/2β` formula **does not work** — the two
  γ's are different constructs (noise rate vs entropy-decay exponent).
- A **principled** bridge (holographic noise-γ ↔ measured `scaffold_frac`) had the
  right sign at n=18 but **collapsed to null at n=85** (learnability↔H∞ = −0.03).
  The holographic boundary does **not** transfer to predicting real-corpus health.
  Root cause: γ is a generative knob with no clean, non-circular real-corpus proxy.
- α_D=γ/2β is a **controlled-setting law**; real corpora violate the control
  (uncontrolled content, pooling, tiny model / narrow D), which **explains** the
  loose validation rather than excusing it.

## Reflection

The biggest lesson is a guard against **false unification**. It was tempting to
fuse the two frameworks through the shared `α=γ/2β` formula — and the naive bridge
*looked* runnable — but it produced a backwards result that, taken at face value,
would have claimed "content-rich corpora are unlearnable." Tracing *why* surfaced
the real structure: **β is a shared axis (correlation decay), but γ is not** — the
generative theory's noise-rate and our measurement framework's entropy-decay are
different things that happen to wear the same letter.

The *conceptual* unification still holds and is valuable: the (β-axis = correlation
structure) ⊥ (γ-axis = noise/content) split is exactly why we measured Hurst as
content-independent — our empirical orthogonality is a prediction of their theory.
But the *quantitative* bridge **does not transfer** (n=85 null), and Step 3 shows
this is not a power problem — it's that **γ has no clean, non-circular observational
counterpart**. Claiming a fused framework would be overreach. Honest status:
"shared vocabulary and one genuinely shared axis (β), a conceptual γ↔noise mapping
that explains existing findings (Hurst-independence) but fails as a predictor, and
**no validated quantitative transfer**." To revive it would require *constructing* a
proper within-sequence signal-dilution γ, then closed-loop synthetic experiments at
real corpora's measured (β, γ) — the opt-in GPU direction — not metric substitution.

## Next steps
- **Construct a proper γ** (within-sequence signal-dilution: fraction of context
  that fails to reduce next-token uncertainty) — the existing axes don't supply one.
  Only then is the (β,γ) placement a fair test.
- **Closed-loop validation** (generate synthetic holographic data at a corpus's
  measured (β,γ), train, check predicted phase) — the principled version of the
  GPU-difficulty direction; makes the bridge causal instead of correlational.
- Keep the **conceptual** unification (β-axis shared; γ↔noise explains
  Hurst-independence) — it's sound — but do **not** ship the quantitative bridge or
  the imported learnability axis as validated; they are not.
