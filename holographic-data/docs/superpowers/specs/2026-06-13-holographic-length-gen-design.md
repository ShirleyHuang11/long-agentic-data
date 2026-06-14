# Holographic Data for Length Generalization — Design Spec

**Date:** 2026-06-13
**Status:** approved design → ready for implementation plan
**Location:** `holographic-data/case/recursion/` (new module, parallel to `case/phase/`)

---

## 0. One-paragraph summary

We generate synthetic pre-training data that is **holographic** — short sequences
engineered so that the autoregressive next-token target *is* a length-invariant
operator (a fold over nested function composition), and whose statistical
structure (long-range correlation `β`, sustained information density `γ`) is tuned
to the regime where a **standard decoder-only Transformer** actually learns the
operator instead of a depth-specific shortcut. We map the `(β,γ)` phase diagram of
**length generalization** (train on short / shallow sequences, test on long / deep
ones) for a vanilla Transformer trained purely with next-token prediction, locate
the "holographic edge-of-chaos band," and verify with two controls that the effect
is a *data* effect (holographic-short beats truncated-long) and relate it to
architecture (does holographic data close the Transformer-vs-Mamba gap?).

---

## 1. Motivation & relationship to the four writeups

The four writeups in `assets/` combine as follows:

- **`holo.pdf`** — the *blueprint* (predictions only, no experiments). Thesis:
  length generalization can be achieved by **data design alone** (no RoPE change,
  no loss change). Short data must be a *microcosm* of long data, not a *slice*.
  Construct short sequences exhibiting `β→0` (correlations that do not decay with
  distance / fully-connected dependency) and slowly-decaying `γ` (sustained high
  information density), forcing the model to learn a scale-invariant **operator**
  `x_{t+1} = f(x_t, state)` rather than position-specific patterns. Provides the
  `(β,γ)` anchors and the implementation `P ∝ 1/d^(1+β)`.
- **`gamma-beta.pdf`** (Cagnetta et al., the cited "Paper A") — *rigorous*
  definitions: `γ` = next-token conditional-entropy decay exponent
  (`H_n − H_∞ ∼ n^(−γ)`), `β` = pairwise-correlation decay exponent
  (`‖C(n)‖_op ∼ n^(−β)`), and the data-limited scaling exponent
  `α_D = γ/(2β)` with usable horizon `n*(P) ∝ P^(1/2β)`. Supplies the
  **estimators** we use to verify our knobs hit their target exponents. It has no
  synthetic data and no length-gen protocol — those are our contribution.
- **`epiplexity.pdf`** (Finzi et al., the cited "Paper B") — a *measurable*
  structural complexity, **epiplexity ≈ area under the training-loss curve above
  the floor**, and the empirical edge-of-chaos: complexity peaks at Wolfram
  **Class-IV** automata, between order and chaos. Used as a light "band
  certification" metric.
- **`data-manifold.pdf`** (Zhang–Jin–Barak–Kakade, your group) — a *training-free
  oracle* `(H_∞, α̂)` read off one LZ77 pass, plus a synthetic L-system generator
  and the finding that language lives in an **intermittent long-memory** regime
  (`θ ≈ 0.57`). Optional descriptive overlay; not in the headline matrix.

**The crux this design resolves.** `holo.pdf` *predicts* long-range data helps a
vanilla Transformer length-generalize, but the project's own prior N=3 result
found Transformers length-generalize *worse* than Mamba exactly in structured
long-range regions. Resolution: **"holographic" is not merely long-range
correlation** — it is an **operator-trace** whose AR target is a length-invariant
program. The `(β,γ)` statistics then control *whether* a vanilla Transformer
learns the program (extrapolates) or a depth-specific shortcut (fails). That
disentanglement is what makes this a clean, falsifiable, vanilla-Transformer-only
study.

---

## 2. Core hypotheses

- **H1 (main).** A vanilla decoder-only Transformer trained with NTP on **short**
  reduction-traces of nested-monoid compositions generalizes to **unseen greater
  depths** far better in the holographic regime (`β→0`, low–moderate `γ`) than
  outside it. The retention surface `r(β,γ) = acc(D_test)/acc(D_train)` has a
  **ridge/peak** in `(β,γ)` space (the "holographic edge-of-chaos band").
- **H2 (data control).** At *matched token budget*, short **holographic** traces
  beat short **truncations of long** traces.
- **H3 (architecture bridge).** The Transformer-vs-Mamba length-gen gap from the
  prior N=3 grid **shrinks inside the band** — holographic data lets a vanilla
  Transformer recover SSM-like extrapolation.

---

## 3. Data generator (the core artifact)

### 3.1 Operator: affine maps over ℤ_p

Each named op is `nᵢ = (aᵢ, bᵢ)` acting `x ↦ aᵢ·x + bᵢ mod p` for a small prime
`p` (default `p = 31`). Composition is **associative and closed**
(`affine ∘ affine = affine`), so the fold rule is **identical at every depth** ⇒
the ground-truth operator is **provably depth-invariant**. This is precisely
`holo.pdf`'s `x_{t+1} = f(x_t, state)`. Answer is a single token in `ℤ_p`.

**Alternative (drop-in):** `Sₖ` permutations on `k` elements (group product as
composition). Kept as a fallback if affine values turn out too easy/degenerate.

### 3.2 Sequence layout — one example = 4 blocks, one AR stream

1. **Definition block** — `D` named ops `nᵢ = (aᵢ,bᵢ)`, interspersed with
   `γ`-controlled filler tokens (no-op bindings / comments / distractor defs).
2. **Expression block** — nested composition referencing names:
   `eval( n7 ( n3 ( n12 ( … ( x₀ ) … ) ) ) )`.
3. **Reduction trace (scratchpad)** — innermost-out:
   `x₀ → v₁ → v₂ → … → answer`, each intermediate value emitted as tokens.
4. **Answer token.**

NTP cross-entropy is taken over the whole sequence (genuine LM pre-training). The
designated **answer region** (block 4, and per-step values in block 3) is what we
score for length generalization.

### 3.3 Knob definitions

- **β = reference-distance / nesting skew.** The distance from a name's
  *definition* (block 1) to its *use* (block 2) is sampled
  `P(d) ∝ d^(−(1+β))`. `β→∞` ⇒ definitions just-in-time (local, short-range);
  `β→0` ⇒ definitions front-loaded, scale-free, the answer depends on a token
  thousands of positions back (maximal head-tail coupling). Matches `holo.pdf`'s
  `P ∝ 1/d^(1+β)`.
- **γ = filler rate.** Fraction of tokens that are semantically inert. `γ→0` ⇒
  every token a non-skippable computation step (high density); `γ→1` ⇒ mostly
  filler.

### 3.4 Knob verification (mandatory rigor gate, before training spend)

On generated corpora, **measure the realized exponents** using the
`gamma-beta.pdf` estimators and confirm `(β̂, γ̂)` track nominal `(β, γ)`:

- `γ̂` from `H_n − H_∞ ∼ n^(−γ)` (conditional-entropy decay; estimate `H_n` via
  n-gram / small-model conditional NLL).
- `β̂` from `‖C(n)‖_op ∼ n^(−β)` (top singular value of the two-point covariance
  `C_{μ,ν}(n) = P(Xᵢ=μ, X_{i+n}=ν) − P(Xᵢ=μ)P(X_{i+n}=ν)`).

If nominal and realized exponents diverge, fix the generator before any training.
The phase map is uninterpretable otherwise.

### 3.5 Length-generalization protocol

Train depths `D ≤ D_train` (short; total length `≤ L_train`). Test
`D ∈ {2,4,8,…} × D_train` (length `≫ L_train`, unseen). Hold the operator
distribution fixed across train/test; only depth changes.

---

## 4. Models

- **Primary:** vanilla decoder-only Transformer (GPT, RoPE) — reuse
  `case/phase/model.py`.
- **Control:** Mamba — reuse the `model_mamba.py` lineage from the prior N=3 work.
- **Scale (two-tier):**
  - *Grid tier:* small (~10–30M params) for the full `(β,γ)` sweep (many cells ×
    seeds, cheap).
  - *Headline tier:* **≥100M** Transformer on headline cells to confirm the band
    survives scale.
- **Compute:** primary GPU partition `-p kempner` (`-A kempner_sham_lab`, rotate
  per `acct.sh`); short smoke tests on `kempner_requeue` with tight `-t`.
  Checkpoints/large files to `$SCRATCH`, never `$HOME`.

---

## 5. Metrics

- **Primary:** answer exact-match accuracy vs depth `D`; **retention**
  `r = acc(D_test)/acc(D_train)` (same definition as the prior N=3 grid, so
  numbers are directly comparable).
- **Secondary:** per-reduction-step trace accuracy; train-length loss floor.
- **Band certification (light):** confirm the retention ridge co-locates with the
  *measured* complexity peak — epiplexity (`epiplexity.pdf` area-under-loss)
  and/or long-range-correlation strength. Minimal, since the full oracle study was
  scoped out.

---

## 6. Experimental matrix (staged)

| Phase | What | Arch | Model | Seeds | Purpose |
|---|---|---|---|---|---|
| **A — Anchors** | {Natural (2,0.8), CoT (0.5,0.4), Edge (0.05,0.05), Abyss (0,0)} | Transformer | small | 3–5 | establish retention ordering fast |
| **B — Grid** | β∈{0,0.05,0.2,0.5,1,2} × γ∈{0,0.05,0.2,0.4,0.6,0.8} (6×6) | Transformer | small | 3 | retention heatmap; locate ridge |
| **C — Controls** | (i) holographic-short vs truncated-long @ matched budget on headline cells; (ii) Mamba column on the 4 anchors | Tx + Mamba | small | 3 | H2, H3 |
| **D — Scale-up** | headline cells (band + Natural baseline) | Transformer | ≥100M | 3 | band survives scale |

Phase A gates Phase B (don't stack PENDING grid jobs until the ordering is
confirmed).

---

## 7. Codebase plan (KISS, reuse, no tech debt)

- New module `holographic-data/case/recursion/` parallel to `case/phase/`.
- New file `case/recursion/data/nested_monoid.py` — generator + §3.4 knob
  verification. Single source of truth for `(β,γ)` sampling.
- **Reuse, do not fork:** `phase_core.train_one_model` (extend the `architecture`
  switch already used for Mamba), `model.py`, `model_mamba.py`.
- Config-driven: `(β, γ, D_train, D_test, p, seed, arch, model_size)`.
- Outputs: CSVs → `runs/`, plots → `plots/`, reports → `reports/*.md`. Atomic git
  commit when results are good (per `CLAUDE.md`). Report takeaways in Chinese +
  emoji.

---

## 8. Success criteria / falsifiability

- **H1 ✅** if `r` shows a statistically significant ridge at small `β` +
  low–moderate `γ` and is low at Natural/Abyss. **❌** if `r` is flat across
  `(β,γ)` or peaks at Natural.
- **H2 ✅** if holographic-short > truncated-long at matched token budget
  (significant across seeds).
- **H3 ✅** if Transformer retention inside the band approaches Mamba's.
- **Abyss control:** `(0,0)` all-architectures-fail bounds the band (attention
  saturation / rank collapse).

---

## 9. Risks & mitigations

- **Vanilla Transformer fails to length-generalize anywhere** (consistent with
  the prior N=3 finding). The **scratchpad** format makes the task RASP-L-
  achievable; if it still fails everywhere, that is itself a clean publishable
  finding ("data alone insufficient for Transformers — the architecture matters"),
  and the Mamba control quantifies it.
- **Knobs miss their target exponents** → caught by the §3.4 verification gate
  before any training spend.
- **Edge cell (0.05,0.05) untrainable** → expected per `holo.pdf`; it is the
  boundary, not a bug. Abyss (0,0) is the explicit failure control.
- **Affine operator too easy** (model memorizes `a,b`) → swap to `Sₖ`
  permutations (§3.1) and/or increase `p`, op vocabulary, and distractor density.

---

## 10. Out of scope (YAGNI)

- Position-encoding ablation (RoPE/NoPE/APE) — explicitly excluded.
- Full training-free oracle study (LZ77 `(H_∞, α̂)` prediction of the peak) —
  only the light band-certification overlay is kept.
- Mini-interpreter / arithmetic-expression operator — possible later face-validity
  extension, not in v1.
- Compressed (no-scratchpad) "hard mode" and the trace-exposure knob — possible
  follow-up axis, not in v1.
