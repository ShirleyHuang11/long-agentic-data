# Long-Range Dependency in Neural Architectures and Neural Scaling-Law Theory

> Scope: the ML core of a data-theory project. Two questions: **(a)** how do
> neural sequence architectures capture *long-range, power-law* dependencies —
> the kind the project parameterizes by the token-correlation decay exponent
> β — and **(b)** what theory links *data statistics* to *scaling exponents*,
> in particular the data-limited law **α_D = γ/(2β)** that this project adopts.
>
> Web search was available; every reference below was checked against
> arXiv / OpenReview / PMLR / journal listings during writing, and arXiv IDs
> are given for traceability. Seminal works (2017–2022) and recent advances
> (2023–2026) are both covered.
>
> Companion reviews in this folder: `01_fractal_structures_in_data.md`
> (data-side complexity / fractality) and `02_length_generalization.md`
> (train-short/test-long). This file is the architecture + scaling-theory leg.

## Overview

The project sits at the intersection of two literatures that have, until very
recently, run in parallel:

1. **Architecture / long-range capacity.** A decade of work asks which
   sequence architectures can represent and learn long-range dependencies,
   and at what cost. The modern axis is **attention (quadratic, content-
   addressable, *unbounded* effective state) vs. recurrence / state-space
   models (linear, *fixed-size* compressed state)**. The central, repeatedly
   rediscovered fact: a fixed-size recurrent state is information-theoretically
   bottlenecked for *retrieval/copying*, while attention pays quadratic compute
   to keep an unbounded, exact cache. This is exactly the knob the project
   manipulates — Transformer vs. Mamba vs. RoPE on a KV-retrieval task whose
   long-range structure is set by β.

2. **Neural scaling-law theory.** A parallel line explains *why* test loss
   falls as a power law in data/parameters/compute, and what sets the
   exponent. Early phenomenology (Kaplan, Chinchilla) gave exponents without
   first-principles values; later theory tied the exponent to a *data*
   property — spectral decay (Maloney–Roberts–Sully), data-manifold dimension
   (Sharma–Kaplan), quantized skills (Michaud), or compositional hierarchy
   (Cagnetta–Wyart). The project's source paper (Cagnetta, Raventós, Ganguli,
   Wyart, "Deriving neural scaling laws from the statistics of natural
   language") closes the loop: it *measures* two language statistics — the
   conditional-entropy decay exponent γ and the token-correlation decay
   exponent β — and predicts the data-limited loss exponent **α_D = γ/(2β)**
   with no free parameters.

The synthesis the project pursues is to read both legs through one
(β, γ) plane: β controls the long-range structure the *architecture* must
capture, γ controls the predictive information available, and α_D = γ/(2β)
is the scaling exponent the data statistics dictate. The "edge-of-chaos /
maximum effective complexity" hypothesis is then an architecture-dependent
boundary (slope set by an architecture parameter δ) on that plane.

## State-space models & long-conv

**The HiPPO → S4 lineage.** Modern SSMs descend from a continuous-time
linear recurrence x'(t) = A x(t) + B u(t), y = C x. The breakthrough was
making this *trainable and efficient at long range*:

- **S4** (Gu, Goel & Ré 2022; arXiv:2111.00396) parameterizes A by a normal-
  plus-low-rank structure, diagonalizable into a Cauchy-kernel computation,
  giving a long convolution computable in near-linear time. S4 was the first
  architecture to *solve* Long Range Arena (LRA), including the Path-X task
  that every Transformer variant had failed. The HiPPO theory gives A a
  principled initialization that optimally compresses history onto orthogonal
  polynomial bases — i.e., a *structured* fixed-size memory.
- **S4D** (Gu et al. 2022; arXiv:2206.11893) shows a purely *diagonal* SSM,
  with the right initialization, matches S4 — simplifying the model and
  clarifying that diagonal SSMs are the practical core.
- **S5** (Smith, Warrington & Linderman 2023; arXiv:2208.04933) replaces the
  bank of independent single-input SSMs with one multi-input/multi-output SSM
  computed by a *parallel associative scan*, simplifying the architecture and
  improving LRA further. The scan formulation is what later makes Mamba's
  selective recurrence efficient.

**H3 and the recall gap.** **H3** (Dao, Fu et al. 2023; arXiv:2212.14052)
isolates *why* SSMs trailed attention in language: SSMs are weak at
**recalling earlier tokens** and **comparing tokens across the sequence**.
H3 adds shift- and diagonal-SSM stacks with multiplicative gating to mimic a
two-token "induction"-like comparison, plus a FlashConv kernel, narrowing the
LM gap. This is the first crisp statement of the *associative-recall* deficit
that the project's KV-retrieval task directly stresses.

**Long convolutions without recurrence.** **Hyena** (Poli et al. 2023;
arXiv:2302.10866) drops the SSM and uses *implicitly parameterized long
convolutions* + data-controlled gating as a sub-quadratic attention
replacement, reaching Transformer quality at 2K context with 20% less compute
and ~100× speedups at 64K. It clarifies that the key ingredients are (i) a
global, sequence-length filter and (ii) input-dependent gating — not the SSM
per se.

**Selectivity: Mamba / Mamba-2.** **Mamba** (Gu & Dao 2023;
arXiv:2312.00752) makes the SSM parameters (Δ, B, C) *input-dependent*
("selective"), so the model can choose what to remember or forget at each
step — recovering content-based addressing inside a linear-time recurrence.
With a hardware-aware parallel scan it gives 5× inference throughput and
million-token scaling, matching Transformers at small/medium scale. The
project's `model_mamba.py` is exactly this: a pure-PyTorch selective SSM with
HiPPO-style log-spaced A, input-dependent Δ/B/C, sequential scan. **Mamba-2 /
Structured State-Space Duality (SSD)** (Dao & Gu 2024; arXiv:2405.21060) is
the unifying result: a large class of SSMs is *equivalent* to a form of masked
(causal) linear attention via structured semiseparable matrices. This means
"**Transformers are SSMs**" in a precise algebraic sense — and conversely, the
fixed-state vs. unbounded-cache distinction is the *only* essential difference,
which is precisely what the project's architecture comparison probes.

**RetNet and gated linear attention.** **RetNet** (Sun et al. 2023;
arXiv:2307.08621) introduces *retention* with a fixed per-head decay, giving
three equivalent forms — parallel (training), recurrent (O(1) inference),
chunkwise (long-sequence). **Gated Linear Attention (GLA)** (Yang, Wang et al.
2024; arXiv:2312.06635) generalizes this to *data-dependent* gates with a
hardware-efficient chunkwise algorithm (FlashLinearAttention), competitive
with Mamba and LLaMA. Both are points on the linear-attention ↔ SSM spectrum
that SSD formalizes; both keep a fixed-size (matrix-valued) state.

**The fixed-state limit is fundamental.** Two results bound what *any* such
fixed-state model can do at retrieval:
- **Repeat After Me** (Jelassi, Brandfonbrener, Kakade & Malach 2024;
  arXiv:2402.01032): a two-layer Transformer can copy strings of *exponential*
  length, while generalized SSMs are limited by their fixed-size state; this
  gap holds for pretrained LLMs too. Direct evidence that β-structured
  retrieval favors attention's unbounded cache.
- **Zoology / MQAR** (Arora, Eyuboglu et al. 2023; arXiv:2312.04927):
  formalizes **Multi-Query Associative Recall** and shows 82% of the
  attention-vs-gated-conv perplexity gap is explained by recall; a 70M
  attention model beats a 1.4B gated-conv model on recall. MQAR is the
  canonical synthetic that the project's KV-retrieval task generalizes (β
  smoothly tunes the *distance distribution* of the queried key, which MQAR
  leaves uniform).

## Attention variants & expressivity

**Efficient / sparse / linear attention** attack the O(T²) cost so attention
can reach β-relevant long contexts:

- **Longformer** (Beltagy et al. 2020; arXiv:2004.05150): sliding-window +
  global tokens, O(T) attention.
- **BigBird** (Zaheer et al. 2020; arXiv:2007.14062): window + global +
  random attention; proves the sparse pattern is a universal approximator and
  Turing-complete.
- **Linformer** (Wang et al. 2020; arXiv:2006.04768): low-rank projection of
  keys/values, O(T) — but the fixed projection rank caps recall of distant
  detail.
- **Performer** (Choromanski et al. 2021; arXiv:2009.14794): FAVOR+ random
  features give an unbiased linear-time softmax-attention estimator.

These trade exact long-range content-addressing for compute; like SSMs, the
ones with a *fixed-size* summary (Linformer, linear/Performer) inherit a
recall bottleneck, while sparse-but-exact ones (Longformer/BigBird) keep it
at the cost of a hand-designed sparsity pattern.

**Expressivity limits.** A formal-language line bounds what attention can
compute regardless of width:
- Hard-attention / log-precision Transformers sit in the circuit class **TC⁰**
  (Merrill & Sabharwal 2023, "The Parallelism Tradeoff"; arXiv:2207.00729),
  so a fixed-depth Transformer provably cannot solve inherently sequential
  (NC¹-hard) problems like general state tracking.
- **The Illusion of State in State-Space Models** (Merrill, Petty &
  Sabharwal 2024; arXiv:2404.08819): *SSMs are no more expressive than
  Transformers* in this sense — also TC⁰ — so their "recurrent state" does not
  buy real state-tracking (e.g., permutation composition / S₅). A sharp
  caution: the project's *length-generalization* advantage for Mamba is about
  **memory retention of structured retrieval**, not about escaping the TC⁰
  state-tracking ceiling.
- **Induction heads** (Olsson et al. 2022; arXiv:2209.11895): the mechanistic
  origin of in-context copy/recall ([A][B]…[A]→[B]); induction heads emerge in
  a sharp phase change that coincides with the onset of in-context learning.
  This is the mechanism a Transformer uses to solve KV-retrieval, and the one
  H3/Mamba try to emulate in a recurrence.

## RNN-family

Classical RNNs/LSTMs (Hochreiter & Schmidhuber 1997) introduced gating to
combat the **vanishing/exploding gradient** problem (Bengio et al. 1994;
Pascanu et al. 2013), but their fixed hidden state and sequential training
limited long-range capacity and scale. The fixed-state memory-capacity
argument (you cannot losslessly store an unbounded prefix in O(1) memory) is
the RNN ancestor of the SSM recall bottleneck above.

- **RWKV** (Peng et al. 2023; arXiv:2305.13048): an "RNN-as-Transformer" with
  a linear-attention-like WKV operator, trainable in parallel, O(1) inference
  — an early modern linear-recurrent LM.
- **xLSTM** (Beck, …, Hochreiter 2024; arXiv:2405.04517): revives the LSTM
  with *exponential gating* (sLSTM) and a *matrix memory with covariance
  update* (mLSTM, parallelizable), competitive with Transformers and SSMs in
  scaling — evidence the recurrent family can be modernized, but it shares the
  fixed-state retrieval limits.

The recurrent family is the conceptual baseline for "what a compressed,
fixed-size state can and cannot hold," which is the quantity the L²M condition
(below) makes precise for long-context modeling.

## Benchmarks

- **Long Range Arena (LRA)** (Tay et al. 2021; arXiv:2011.04006): six tasks,
  1K–16K tokens (ListOps, byte text, retrieval, image, Pathfinder, Path-X);
  the standard efficient-architecture testbed. S4 was the model that finally
  cleared Path-X, making LRA the de facto SSM proving ground.
- **MQAR** (Arora et al. 2023, above): the diagnostic that *predicts* LM
  quality from a synthetic recall task — methodologically the closest prior
  art to the project's "train on a synthetic with a tunable statistic, read
  off architecture differences" design.
- **Induction / associative recall** (Olsson et al. 2022; H3 2023): the
  minimal in-context-recall probes; the project's KV task is a parameterized
  (β = retrieval-distance decay, γ = noise) generalization.

## Neural scaling-law theory

**Phenomenology.**
- **Kaplan et al. 2020** (arXiv:2001.08361): test loss is a power law in
  parameters N, data D, and compute C, over many orders of magnitude;
  introduced the compute-optimal frontier but *over-weighted N vs D*.
- **Hoffmann et al. 2022, "Chinchilla"** (arXiv:2203.15556): corrected the
  trade-off — N and D should scale ~equally; a 70B model trained on 1.4T
  tokens beats much larger under-trained models. Establishes the *data-limited*
  regime's practical importance, which is exactly the regime the project's
  α_D law targets.

**First-principles theory (what sets the exponent).**
- **Bahri, Dyer, Kaplan & Sharma 2024, "Explaining Neural Scaling Laws"**
  (arXiv:2102.06701; PNAS): four regimes — *variance-limited* (∝1/D or 1/N,
  from a well-behaved infinite-data/width limit) and *resolution-limited*
  (the power-law regime, where the model resolves a smooth data manifold and
  the exponent is set by the manifold/spectrum). The variance↔resolution split
  is the cleanest map for *which* exponent governs a given regime.
- **Sharma & Kaplan 2022** (arXiv:2004.10802): if the network is effectively
  doing regression on a **data manifold of intrinsic dimension d**, then
  α ≈ 4/d. Ties the exponent to a single *geometric* data property; the
  conceptual parent of "exponent = data statistic."
- **Maloney, Roberts & Sully 2022, "A Solvable Model of Neural Scaling Laws"**
  (arXiv:2210.16859): an analytically solvable random-feature model where a
  **power-law spectrum** of the data/feature covariance (exponent 1+α) is
  *translated* into a power-law test-loss decay, with a plateau when the
  spectral power law has finite extent. This is the cleanest mechanism
  for "data correlation spectrum → loss exponent," the same logic the project's
  β (correlation-decay) leg uses.
- **Michaud, Liu, Girit & Tegmark 2023, "The Quantization Model"**
  (arXiv:2303.13506): capabilities come in discrete **quanta**; if quanta are
  learned in decreasing frequency order and frequencies follow a power law
  (Zipf), the loss scales as a power law and *emergence* is the learning of a
  new quantum. Connects scaling to **emergence** — relevant to the project's
  "emergent strip."
- **Bordelon, Atanasov & Pehlevan 2024, "A Dynamical Model of Neural Scaling
  Laws"** (arXiv:2402.01092): a solvable model giving *time/compute*-resolved
  scaling and reconciling several exponents; useful if the project ever
  reports compute-limited (not just data-limited) curves.

## Data-statistics → scaling exponents (the γ/β line)

This is the theoretical heart for the project and deserves precise statement.

**Cagnetta, Raventós, Ganguli & Wyart, "Deriving neural scaling laws from the
statistics of natural language"** (arXiv:2602.07488; the project's
`assets/gamma-beta.pdf`). The claim: in the **data-limited** regime, the loss
exponent is predicted *with no free parameters* from two measurable
statistics of the corpus:

- **γ — conditional-entropy decay.** The next-token conditional entropy with
  context length n satisfies H_n − H_∞ ≍ n^{−γ}. (Measured: TinyStories
  γ ≈ 0.34, WikiText γ ≈ 0.27; γ is architecture-independent across GPT-2-APE,
  GPT-2-RoPE, LLaMA.)
- **β — token-correlation decay.** The top singular value of the two-point
  token covariance C(n) at lag n decays as ‖C(n)‖_op ≍ n^{−β}. (Measured:
  TinyStories β ≈ 0.88, WikiText β ≈ 0.94.)
- **Mechanism.** A signal-to-noise argument gives a *data-dependent prediction
  horizon* n*(P) ≍ P^{1/(2β)}: with P training tokens, correlations at lag n
  become resolvable only once P is large enough (‖C(n)‖_op ≳ P^{−1/2}). The
  loss decomposes into a horizon term H_{n*(P)} plus within-horizon excess
  terms; assuming fast within-horizon learning, the **horizon term dominates**,
  and substituting n*(P) ≍ P^{1/(2β)} into H_n − H_∞ ≍ n^{−γ} yields

  **L_AR(P) − H_∞ ≍ P^{−γ/(2β)} ⟹ α_D = γ/(2β).**

- **Validation.** TinyStories: α_D ≈ 0.34/(2·0.88) ≈ 0.19; WikiText:
  α_D ≈ 0.27/(2·0.94) ≈ 0.14 — matching trained GPT-2/LLaMA learning curves.
  The framework also predicts a **scaling collapse**: individual n-gram
  learning curves L_n(P), replotted as n^γ L_n vs P/n^{2β}, fall on one master
  curve — a stringent, parameter-free test.

This is the *exact* α_D = γ/2β the project's `main_findings.md` and source
paper adopt. Two upstream pillars it builds on:

- **Cagnetta & Wyart 2024 (Random Hierarchy Model, RHM)** (arXiv:2307.02129;
  PRX): a solvable hierarchical/compositional generative model of data where
  deep nets beat the curse of dimensionality with *polynomial* sample
  complexity; provides the "data has latent hierarchical structure recoverable
  from substrings + correlations" picture underlying the horizon argument.
- **L²M: Mutual-Information Scaling Law** (Chen, Mayné i Comas, Jin, Luo &
  Soljačić 2025; arXiv:2503.04725): the *bipartite* mutual information between
  two halves of a sequence grows as a **power law L^β** in natural language
  (multi-token, distinct from two-point MI). The **L²M condition** lower-bounds
  the required growth of a model's *history-state size* for long-context
  modeling — and shows fixed-state SSMs/RNNs/linear-attention *cannot* satisfy
  it with one fixed model, while attention's growing cache can. This is the
  information-theoretic bridge between the **β leg** (correlation/MI power law)
  and the **architecture leg** (fixed vs. growing state) — arguably the single
  most relevant external result for the project's architecture comparison.

A few adjacent data-distribution results worth tracking: Hutter 2021
(arXiv:2102.04074, infinite feature/Zipf toy model giving clean data-exponent
↔ loss-exponent maps) and the broader "solvable model" cluster cited by the
source paper (Spigler, Lin, Paquette).

## Methods / results to borrow (mapped to β, δ-architecture-dependence, α_D)

1. **Measure β and γ as *data* statistics, not fit parameters.** The source
   paper's protocol — estimate γ from the small-n decay of n-gram losses of a
   well-trained model, estimate β from the operator-norm decay of the lag-n
   token covariance — is directly portable to the project's
   `AlgorithmicKVGenerator`. The project already computes Rényi dimensions of
   n-gram statistics (`phase_core.compute_dataset_renyi_dimensions`); adding
   β̂(C(n)) and γ̂(H_n) makes the (β, γ) axes *measured*, closing the loop
   between the *designed* β (the power-law p(d) ∝ d^{−(β+1)} retrieval prior)
   and the *realized* correlation exponent.

2. **Scaling-collapse as a falsifiable test.** The n^γ L_n vs P/n^{2β}
   collapse is exactly the kind of committed-in-advance prediction the
   project's `verify_predictions.py` scoreboard rewards. A collapse (or its
   failure) at fixed β would be a clean, parameter-free check that α_D = γ/2β
   governs the synthetic task — strengthening the paper beyond the current
   train_acc linear laws.

3. **Architecture-dependence = the fast-learning assumption breaks.** The
   α_D = γ/2β derivation *assumes within-horizon learning is fast* — an
   explicitly **architecture-dependent** assumption (the source paper notes it
   should fail for shallow nets / kernels / n-gram models that suffer the curse
   of dimensionality at large n). This is the natural home for the project's
   δ: the architecture-dependent boundary slope is a statement about *which
   architectures realize the fast-learning assumption at which n*. Mamba's
   measured 2.0–2.6× length-generalization retention vs. Transformer at the
   strip (main_findings Result 11) is the empirical signature: a fixed-state
   model retains *structured* retrieval better when β concentrates the queried
   key at short-to-moderate lag, but collapses at edge-of-chaos (β≈0.05) where
   there is no structure to compress — consistent with both Repeat-After-Me
   (attention wins exact long copy) and L²M (fixed state fails when MI grows).

4. **Recall as the discriminating axis.** Borrow MQAR/Zoology's framing: report
   architecture differences on a *recall* metric (the project's length-gen
   ratio r = acc(2048)/acc(512) is a good analogue) rather than train_acc —
   main_findings Result 11 already shows the architecture effect is *entirely*
   in length-generalization, invisible to train_acc, exactly the Zoology lesson.

5. **Expressivity ceiling as a scope guard.** Cite Illusion-of-State /
   Parallelism-Tradeoff to bound claims: the project's Mamba advantage is
   memory-retention of β-structured retrieval, **not** a state-tracking
   capability gain (both are TC⁰). Prevents over-claiming.

## Relevance to this project

The project's (β, γ) phase diagram is, in one sentence, **the Cagnetta–
Raventós–Ganguli–Wyart scaling theory turned into a controlled experiment, with
architecture added as a third axis.** Concretely:

- **β ↔ long-range capacity.** β is the project's retrieval-distance decay and
  the source paper's token-correlation exponent — the *same object*. The
  architecture literature says: smaller β (heavier long-range tails, more MI
  at distance) increasingly favors attention's unbounded cache over a fixed
  SSM state (Repeat-After-Me, L²M, MQAR). The project's job is to map *where*
  on the β axis this crossover happens for its task, and that crossover is the
  δ-dependent boundary.

- **α_D = γ/(2β) is the scaling theory behind the diagram.** The data-limited
  exponent the project inherits is derived, not assumed: horizon n* ≍ P^{1/2β}
  (set by β) feeding the entropy decay H_n − H_∞ ≍ n^{−γ} (set by γ). The
  "edge of chaos / maximum effective complexity" target is a region of the
  (β, γ) plane where structured long-range dependence exists (β small enough
  to demand long memory) yet is *learnable* (γ large enough that predictive
  information is actually present) — i.e., where α_D and the architecture's
  fast-learning assumption are both favorable. This is a sharper, theory-
  grounded restatement of the project's stated goal.

- **The architecture comparison tests the one assumption the theory leaves
  open.** Because α_D = γ/2β rests on an architecture-dependent fast-learning
  assumption, comparing Transformer / Mamba / RoPE on the *same* (β, γ) data
  is precisely the experiment that isolates δ. The recommended additions —
  measured β̂/γ̂, the n^γ L_n scaling collapse, and the recall-style length-gen
  metric — would let the paper state the architecture effect *in the source
  paper's own variables*, which is the strongest possible bridge between the
  two literatures this review spans.

## References

(Author, year, title — arXiv ID / venue — 1–2 line summary. Verified against
arXiv/OpenReview/PMLR/journal listings during writing.)

**Architectures — SSM / long-conv / linear-attention**
1. Gu, Goel & Ré 2022 — *Efficiently Modeling Long Sequences with Structured
   State Spaces (S4)* — arXiv:2111.00396 (ICLR'22). NPLR-parameterized SSM;
   first to solve Long Range Arena incl. Path-X.
2. Gu, Goel, Gupta & Ré 2022 — *On the Parameterization and Initialization of
   Diagonal State Space Models (S4D)* — arXiv:2206.11893. Diagonal SSM matches
   S4 with proper init; simplifies the core.
3. Smith, Warrington & Linderman 2023 — *Simplified State Space Layers for
   Sequence Modeling (S5)* — arXiv:2208.04933 (ICLR'23). MIMO SSM via parallel
   associative scan; the scan that Mamba reuses.
4. Dao, Fu, Saab, Thomas, Rudra & Ré 2023 — *Hungry Hungry Hippos (H3)* —
   arXiv:2212.14052 (ICLR'23). Diagnoses SSM recall/comparison deficit; gated
   SSM stack narrows the LM gap.
5. Poli, Massaroli, Nguyen, Fu, Dao, Baccus, Bengio, Ermon & Ré 2023 —
   *Hyena Hierarchy* — arXiv:2302.10866 (ICML'23). Implicit long convolutions
   + gating as sub-quadratic attention replacement.
6. Gu & Dao 2023 — *Mamba: Linear-Time Sequence Modeling with Selective State
   Spaces* — arXiv:2312.00752. Input-dependent (selective) SSM + hardware scan;
   the project's `model_mamba.py` baseline.
7. Dao & Gu 2024 — *Transformers are SSMs: Structured State Space Duality
   (Mamba-2)* — arXiv:2405.21060 (ICML'24). SSMs ≡ a form of causal linear
   attention; unifies the two families.
8. Sun, Dong, Huang, Ma, Xia, Xue, Wang & Wei 2023 — *Retentive Network
   (RetNet)* — arXiv:2307.08621. Retention with fixed decay; parallel /
   recurrent / chunkwise equivalence.
9. Yang, Wang, Shen, Panda & Kim 2024 — *Gated Linear Attention Transformers
   with Hardware-Efficient Training (GLA)* — arXiv:2312.06635 (ICML'24).
   Data-dependent gated linear attention; FlashLinearAttention.

**Attention variants & expressivity**
10. Vaswani et al. 2017 — *Attention Is All You Need* — arXiv:1706.03762.
    The Transformer; quadratic content-addressable attention.
11. Beltagy, Peters & Cohan 2020 — *Longformer* — arXiv:2004.05150.
    Sliding-window + global tokens; O(T) attention.
12. Zaheer et al. 2020 — *Big Bird* — arXiv:2007.14062. Window+global+random
    sparse attention; universal approximator / Turing-complete.
13. Wang, Li, Khabsa, Fang & Ma 2020 — *Linformer* — arXiv:2006.04768.
    Low-rank key/value projection; O(T) but fixed-rank recall cap.
14. Choromanski et al. 2021 — *Rethinking Attention with Performers* —
    arXiv:2009.14794 (ICLR'21). FAVOR+ random-feature linear-time softmax.
15. Merrill & Sabharwal 2023 — *The Parallelism Tradeoff: Limitations of
    Log-Precision Transformers* — arXiv:2207.00729 (TACL). Log-precision
    Transformers ⊆ TC⁰; cannot solve NC¹-hard tasks at fixed depth.
16. Merrill, Petty & Sabharwal 2024 — *The Illusion of State in State-Space
    Models* — arXiv:2404.08819 (ICML'24). SSMs also ⊆ TC⁰; no real
    state-tracking advantage over Transformers.
17. Olsson et al. 2022 — *In-context Learning and Induction Heads* —
    arXiv:2209.11895 (Anthropic). Induction heads as the copy/recall mechanism;
    emerge in a sharp phase change.
18. Jelassi, Brandfonbrener, Kakade & Malach 2024 — *Repeat After Me:
    Transformers are Better than SSMs at Copying* — arXiv:2402.01032 (ICML'24).
    Attention copies exp-length strings; fixed-state SSMs cannot.
19. Arora, Eyuboglu, Timalsina, Johnson, Poli, Zou, Rudra & Ré 2023 —
    *Zoology: Measuring and Improving Recall in Efficient LMs (MQAR)* —
    arXiv:2312.04927 (ICLR'24). Recall explains 82% of the attention-vs-conv
    gap; defines MQAR.

**RNN family**
20. Peng et al. 2023 — *RWKV: Reinventing RNNs for the Transformer Era* —
    arXiv:2305.13048 (EMNLP findings). Parallel-trainable linear-recurrent LM,
    O(1) inference.
21. Beck, Pöppel, Spanring, Auer, …, Hochreiter 2024 — *xLSTM: Extended Long
    Short-Term Memory* — arXiv:2405.04517 (NeurIPS'24). Exponential gating +
    matrix memory; modernized LSTM competitive with SSM/Transformer.

**Benchmarks**
22. Tay, Dehghani et al. 2021 — *Long Range Arena* — arXiv:2011.04006
    (ICLR'21). 1K–16K-token efficient-architecture benchmark; the SSM testbed.

**Scaling-law theory**
23. Kaplan et al. 2020 — *Scaling Laws for Neural Language Models* —
    arXiv:2001.08361. Power laws in N, D, C; compute-optimal frontier.
24. Hoffmann et al. 2022 — *Training Compute-Optimal LLMs (Chinchilla)* —
    arXiv:2203.15556 (NeurIPS'22). N and D should scale equally; data-limited
    regime matters.
25. Bahri, Dyer, Kaplan & Sharma 2024 — *Explaining Neural Scaling Laws* —
    arXiv:2102.06701 (PNAS). Variance- vs resolution-limited regimes; four
    scaling laws.
26. Sharma & Kaplan 2022 — *A Neural Scaling Law from the Dimension of the
    Data Manifold* — arXiv:2004.10802 (JMLR). α ≈ 4/d from data-manifold
    dimension.
27. Maloney, Roberts & Sully 2022 — *A Solvable Model of Neural Scaling Laws*
    — arXiv:2210.16859. Data spectral power law → test-loss power law; finite
    extent → plateau.
28. Michaud, Liu, Girit & Tegmark 2023 — *The Quantization Model of Neural
    Scaling* — arXiv:2303.13506 (NeurIPS'23). Power-law-frequency quanta give
    power-law loss + emergence.
29. Bordelon, Atanasov & Pehlevan 2024 — *A Dynamical Model of Neural Scaling
    Laws* — arXiv:2402.01092 (ICML'24). Compute/time-resolved solvable scaling.

**Data-statistics → exponents (the γ/β line)**
30. Cagnetta, Raventós, Ganguli & Wyart 2026 — *Deriving Neural Scaling Laws
    from the Statistics of Natural Language* — arXiv:2602.07488 (project's
    `gamma-beta.pdf`). Measures γ (H_n−H_∞ ≍ n^{−γ}) and β (‖C(n)‖_op ≍ n^{−β});
    derives **α_D = γ/(2β)** parameter-free via horizon n*(P) ≍ P^{1/(2β)}.
31. Cagnetta, Petrini, Tomasini, Favero & Wyart 2024 — *How Deep Neural
    Networks Learn Compositional Data: The Random Hierarchy Model* —
    arXiv:2307.02129 (PRX). Solvable hierarchical data; polynomial sample
    complexity beats the curse of dimensionality.
32. Chen, Mayné i Comas, Jin, Luo & Soljačić 2025 — *L²M: Mutual Information
    Scaling Law for Long-Context Language Modeling* — arXiv:2503.04725.
    Bipartite MI grows as L^β; **L²M condition** lower-bounds history-state size
    — fixed-state SSMs/RNNs/linear-attn cannot satisfy it; attention can.
33. Hutter 2021 — *Learning Curve Theory* — arXiv:2102.04074. Infinite-feature
    Zipf toy model giving clean data-exponent ↔ loss-exponent maps; adjacent to
    the γ/β line.
