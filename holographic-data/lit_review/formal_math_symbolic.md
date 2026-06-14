# Formal Languages, Algorithmic Information & Length/Compositional Generalization in Symbolic Reasoning

> Scope: a method-focused, cross-disciplinary review of formal-language theory,
> algorithmic information theory (Kolmogorov complexity, **logical depth**,
> computational depth, **epiplexity / time-bounded entropy**), and
> compositional / length generalization on symbolic tasks. Written to import
> transferable methods into the holographic-data project: the **(β, γ) phase
> diagram**, **α_D = γ/2β**, **length generalization** (logical depth ≫ input
> length), and the "holographic compression" of deep computation into short
> context. See `## Relevance to this project`.
>
> **Web search was available** (WebSearch + WebFetch). References below were
> checked against arXiv / OpenReview / MDPI / ACL / NeurIPS listings during the
> review; arXiv IDs / venues are given for traceability. A few very recent
> (2026) IDs surfaced by search are flagged inline as needing a final check.

## Overview

Three research traditions independently formalize what the holographic project
calls "deep logical content in a short input":

1. **Formal-language statistics.** The Chomsky hierarchy is not only about *what*
   a grammar can generate but about the *statistical signature* it leaves —
   especially how mutual information between two symbols decays with their
   separation. Regular grammars force **exponential** decay; context-free /
   recursive grammars permit **power-law** decay (long-range correlation). This
   is the closest formal analogue of the project's **β** knob (the exponent of
   the recall-distance / correlation-distance distribution).

2. **Algorithmic information theory (AIT).** Kolmogorov complexity measures
   *incompressibility*; it cannot distinguish a random string from a deeply
   organized one. **Bennett's logical depth** and its resource-bounded cousins
   (**computational depth**, Antunes–Fortnow) repair this by charging for
   *decompression time* — the number of computational steps to reconstruct the
   object from its shortest description. This is the formal analogue of the
   project's **logical depth ≫ input length** (a 33-token jump table that takes
   path_length≫33 sequential steps to "unfold"). **Epiplexity** (Finzi et al.
   2026, the project's own anchor paper) is the learning-theoretic descendant:
   structural information a *computationally bounded* observer extracts, with
   random/time-bounded-entropy content explicitly subtracted — the analogue of
   subtracting γ-filler from learnable content.

3. **Length / compositional generalization.** A body of empirical and theoretical
   work (Chomsky-hierarchy probes, RASP-L, automata-shortcut theory, group word
   problems, compositional benchmarks) characterizes *which* symbolic tasks a
   transformer can learn to extrapolate to longer inputs / deeper compositions,
   and *why*. This directly informs the project's train-short / test-long design
   and the question of whether "edge-of-chaos" data improves length generalization.

## Formal languages & statistical structure

**Mutual-information decay as a grammar fingerprint (the β analogue).**
Lin & Tegmark (2017, *Entropy*; arXiv:1606.06737, "Critical Behavior in Physics
and Probabilistic Formal Languages") prove the central result the project needs:
in **any probabilistic regular grammar (= hidden Markov process)** the mutual
information I(X_i; X_{i+d}) between two symbols decays **exponentially** in the
separation d, governed by the spectral gap of the transition matrix; whereas a
**probabilistic context-free grammar (PCFG)** can produce **power-law** decay
I(d) ∝ d^{-α}. Natural language, music, and genomes empirically show power-law
decay, which is why Markov/regular models fail to capture them. They give a
constructive model class — recursive / tree-like "deep dynamics" — that
generically yields the critical (power-law) behavior. **Method to borrow:**
measure I(d) on generated corpora as a direct, grammar-agnostic readout of the
correlation structure that β controls; the exponent α of I(d) is the empirical
counterpart of the project's β.

**Neural nets vs the Chomsky hierarchy (architecture × correlation class).**
Delétang et al. (2023, ICLR; arXiv:2207.02098, "Neural Networks and the Chomsky
Hierarchy") run 2200 models × ~16 tasks and show that *out-of-distribution
(longer-input) generalization* tracks the Chomsky level of the task and the
memory structure of the net: plain RNN/transformer handle (sub)regular tasks,
LSTMs reach counter languages, and only stack/tape-augmented nets reach
context-free / context-sensitive tasks — with stark negative results (unlimited
data never induces the algorithm). **Method to borrow:** a task-difficulty axis
(Chomsky level / required memory) orthogonal to length, useful for situating the
holographic tasks (pointer-chasing and nested-monoid are essentially
*state-tracking* / NC¹ tasks, harder than regular).

**What transformers can express (expressivity bounds).**
Strobl et al. (2024, TACL, "What Formal Languages Can Transformers Express? A
Survey") synthesize circuit-complexity results: soft-attention transformers with
fixed depth sit inside TC⁰ / are characterized by logics like C-RASP, so tasks
requiring more than constant parallel depth (e.g. unbounded sequential
composition) are inherently hard to length-generalize without chain-of-thought.
This bounds *what the architecture can hold*, complementing the data-side
correlation view.

## Algorithmic information, logical depth & epiplexity

**Kolmogorov complexity (the baseline that fails).**
K(x) = length of the shortest program that outputs x (Kolmogorov 1965; Chaitin
1969; Li & Vitányi text). It is incompressibility, and it is *maximized by random
noise* — so it cannot separate "deep structure" from "γ-filler." Both extremes
(trivial and random) must be excluded to capture organized complexity.

**Bennett's logical depth (the project's core import).**
Bennett (1988, "Logical Depth and Physical Complexity") defines depth as the
**computation time** of a near-minimal-length program that reproduces x (more
precisely, the least time for any program within b bits of K(x)). Crucially,
depth is **low for both random and trivial strings** and **high only for
organized objects** whose plausible origin is a long causal/computational
process. This is exactly the holographic data's defining property: a 33-token
input with **small K but large logical depth** (path_length steps of forced
sequential pointer-chasing). **The single most transferable concept in this
review.**

**Computational depth (resource-bounded, measurable).**
Antunes, Fortnow, van Melkebeek & Vinodchandran (2006, *TCS*; "Computational
Depth: Concept and Applications") operationalize depth as a **difference of
Kolmogorov measures**, e.g. K^t(x) − K(x) (time-bounded minus plain). This is
small for both easy and fully-random strings and large for "useful/non-random"
structure — a computable-in-spirit proxy for Bennett depth, and the bridge to
the project's loss-curve estimators (area under loss curve above the floor).

**Epiplexity / time-bounded entropy (the project's anchor).**
Finzi, Qiu, Jiang, Izmailov, Kolter & Wilson (2026, arXiv:2601.03220, "From
Entropy to Epiplexity") define **epiplexity** as the **structural information a
*computationally bounded* observer can extract** — the information in the model
that minimizes description length under compute constraints — explicitly
*subtracting* **time-bounded entropy** (random, unpredictable content:
PRNG/CSPRNG output, chaotic micro-state, hashes/API-keys). Three "paradoxes"
they resolve (information can be created by deterministic computation; it depends
on factorization/order; likelihood modeling exceeds distribution matching) all
hinge on the same move as logical depth: charge for computation, separate random
from structural content. They give practical estimators (area under the loss
curve above the final loss; cumulative teacher–student KL) and show epiplexity
**correlates with OOD generalization**. **Direct mapping:** γ (filler / noise
rate) ↔ time-bounded entropy to be subtracted; learnable holographic structure ↔
epiplexity; "good data" = high-epiplexity, which the project hypothesizes lives
at an edge-of-chaos band of (β, γ).

**Sophistication / algorithmic statistics (the structure–noise split).**
Vereshchagin & Vitányi (algorithmic statistics; Kolmogorov's "structure
function") formalize splitting K(x) into a **model part (sophistication)** and a
**noise part** — the AIT formalization of exactly the "structural vs random"
decomposition epiplexity and γ encode. Provides the rigorous vocabulary for
"maximum effective complexity at the edge of chaos."

## Compositional & length generalization (symbolic)

**RASP-L: a learnability predictor for length generalization.**
Zhou et al. (2024, ICLR; arXiv:2310.16028, "What Algorithms can Transformers
Learn? A Study in Length Generalization") propose the **RASP-Generalization
Conjecture**: a transformer length-generalizes on a task iff a *short* RASP-L
program (RASP restricted to operations a transformer can realize, no arbitrary
index arithmetic) solves it at all lengths. Confirmed empirically: counting,
mode, sort, copy-without-repeats generalize; addition, parity, copy-with-repeats
do not — unless the **input format is changed** to remove index arithmetic.
**Method to borrow:** before running, ask whether each holographic task has a
short length-invariant program; the nested-monoid fold *does* (a depth-invariant
recall-and-apply rule), which predicts it *should* length-generalize — a falsifiable
design check.

**Automata shortcuts and the depth/sequentiality wall.**
Liu, Ash, Goel, Krishnamurthy & Zhang (2023, ICLR; arXiv:2210.10749,
"Transformers Learn Shortcuts to Automata") show a fixed-depth transformer can
simulate any **solvable** automaton in O(1) depth and any automaton in O(log T)
depth via the algebra of its transformation semigroup — but these *parallel
shortcuts are brittle OOD* and only the autoregressive (sequential) solution
length-generalizes. This is the theoretical core of the holographic "attention
saturation" intuition: deep sequential dependence cannot be compressed into
constant parallel depth without losing extrapolation.

**Group word problems = the hardest state tracking.**
Merrill, Sabharwal et al. (and "Illusion of State in SSMs", arXiv:2404.08819)
use the **word problem for A₅ / S₅** (NC¹-complete permutation composition) as
the canonical hard state-tracking benchmark; standard transformers and linear
SSMs provably cannot solve it at fixed depth. The project's **nested-monoid
(perm) task is exactly this family** — a fold over permutations of Z_p — placing
it on the hardest rung of the depth hierarchy, which is why long-range recall +
depth is the right stress test.

**CoT as the length-generalization unlock.**
"Transformers Provably Learn Chain-of-Thought Reasoning with Length
Generalization" (2025, arXiv:2511.07378 — *verify final ID*) and looped/recurrent
transformers (Fan et al. 2024, arXiv:2409.15647, "Looped Transformers for Length
Generalization") show that *re-emitting intermediate state* (scratchpad / loop)
converts a parallel-depth-bounded task into a length-generalizable one — the
mechanism behind the project's autoregressive "recall-and-apply trace" emission.

**Compositional generalization benchmarks (systematicity & productivity).**
- **SCAN** (Lake & Baroni 2018, ICML; arXiv:1711.00350): seq2seq nets fit
  training commands but fail on systematic recombination and on **longer**
  command sequences (the length split) — the original demonstration that
  in-distribution fit ≠ compositional/length generalization.
- **COGS** (Kim & Linzen 2020, EMNLP): synthetic semantic parsing with 21
  generalization types; transformers hit 96–99% in-distribution but ~35% on the
  generalization set — a clean structural-generalization gap.
- **CFQ** (Keysers et al. 2020, ICLR; arXiv:1912.09713): introduces **compound
  divergence** (maximize novel *combinations* while matching atom frequencies) —
  a principled, *tunable* train/test split metric directly reusable to construct
  hard generalization splits for the holographic tasks.

**Arithmetic length generalization (format & positional fixes).**
McLeish et al. (2024, NeurIPS; arXiv:2405.17399, "Transformers Can Do Arithmetic
with the Right Embeddings") introduce **Abacus embeddings** (encode digit
position-within-number, not absolute token position) + input injection/recurrence
to reach 6× length extrapolation (train 20-digit, test 120-digit). Cho et al.
(2024, NeurIPS; arXiv:2405.20671, "Position Coupling") tie positions to task
structure for similar gains. **Method to borrow:** length generalization is
recovered by aligning positional information with the *task's intrinsic
coordinate* (the recursion depth / register, not raw token index) — relevant to
how the holographic tasks present recall distance.

## Controlled data generation methods

- **PCFGs with tunable depth/branching (the canonical knob).** A PCFG's
  production rules give *direct, principled control* over hierarchical depth,
  recursion, and — via the spectral gap of the induced process — the
  mutual-information decay exponent (Lin & Tegmark above). Allen-Zhu & Li
  ("Physics of Language Models, Part 1: Context-Free Grammar"; arXiv:2305.13673)
  generate deep synthetic CFGs and show transformers learn the latent hierarchy;
  Cagnetta, Petrini, Wyart et al. (arXiv:2406.00048; 2505.07067) give a
  *learning-curve theory* for hierarchically compositional data with power-law
  features — directly tying generation parameters to sample complexity.
- **Recursive / tree "deep dynamics" models** (Lin & Tegmark) reproduce critical
  power-law correlation by construction — a generator template for "long-range
  correlated" data at tunable α (= β analogue).
- **Automaton / group-word generators** (A₅/S₅ composition, semi-automaton
  simulation à la Liu et al.) give *exact* control over logical depth (= number
  of composition steps) independent of input length — the holographic
  pointer/nested-monoid generators are instances of this template.
- **Compound-divergence splitting** (CFQ) is a *data-curation* method: hold
  atoms fixed, vary combination novelty — a ready recipe for constructing the
  "test-long / test-deeper" evaluation sets.
- **Hierarchical latent-variable generators** (arXiv:2603.06592, "Hierarchical
  Latent Structures…" — *verify ID*) parametrize emergence-of-capability via
  latent depth, another tunable-complexity corpus generator.

## Methods to borrow (mapped to γ / logical-depth, length-gen, task design)

| Project need | Borrowed method / concept | Source |
|---|---|---|
| Quantify **logical depth ≫ input length** | Bennett logical depth (decompression time); computational depth K^t−K | Bennett 1988; Antunes et al. 2006 |
| Separate **learnable structure from γ-noise** | Epiplexity (extractable structure) − time-bounded entropy; sophistication / structure function | Finzi et al. 2026; Vereshchagin–Vitányi |
| Measure what **β** controls | Mutual-information-vs-distance curve I(d); exponential (regular) vs power-law (CFG) decay | Lin & Tegmark 2017 |
| Predict whether a task **length-generalizes** | RASP-L short-program existence test | Zhou et al. 2024 |
| Explain the **depth/parallel-compute wall** | Automata-shortcut theory (O(1) solvable, O(log T) general, brittle OOD); TC⁰/C-RASP bounds | Liu et al. 2023; Strobl et al. 2024 |
| Hardest **state-tracking** stress test | A₅/S₅ group word problem (NC¹-complete) | Merrill et al.; "Illusion of State" 2024 |
| **Unlock** length-gen via emitting state | CoT / looped-transformer scratchpads | Fan et al. 2024; CoT-LG 2025 |
| **Generate** tunable-correlation/-depth data | PCFG depth/branching; recursive deep dynamics; learning-curve theory | Allen-Zhu & Li; Cagnetta et al.; Lin & Tegmark |
| Construct hard **generalization splits** | Compound divergence (CFQ) | Keysers et al. 2020 |
| Estimate complexity **from training** | Area-under-loss-curve-above-floor / teacher-student KL | Finzi et al. 2026; computational-depth analogue |

## Relevance to this project

- **Logical depth ↔ holographic compression.** Bennett depth is the precise
  formal statement of "tiny input, deep computation": the holographic data are
  *shallow in K, deep in time*. The project can report **depth (decompression
  steps = path_length / fold count)** as a first-class data statistic alongside
  K, and frame holographic compression as *maximizing depth at fixed input
  length* — a clean, citable framing.
- **β ↔ MI-decay exponent.** Lin & Tegmark give the project a direct experimental
  readout: plot I(X_i; X_{i+d}) for generated corpora. The (β, γ) generators
  predict a tunable α; verifying exponential→power-law as β decreases would
  *ground the phase diagram in formal-language theory* and validate α_D = γ/2β
  against a measured correlation exponent. **Recommended concrete experiment.**
- **γ ↔ time-bounded entropy.** Epiplexity's explicit subtraction of random
  content is the theoretical justification for treating γ-filler as
  non-structural; "edge of chaos / maximum effective complexity" is naturally
  read as **maximizing epiplexity (or sophistication) over (β, γ)** — exactly
  the band the project hypothesizes improves generalization. This is a testable
  claim with the paper's own estimators.
- **Length generalization predictions.** RASP-L + automata-shortcut theory make
  *a priori* predictions: the nested-monoid fold has a short depth-invariant
  program (should length-generalize once state is emitted autoregressively);
  pure parallel-depth solutions will be brittle OOD (matches "attention
  saturation"). The A₅/S₅ result certifies the task is genuinely hard
  (NC¹-complete), not a toy.
- **Task design.** PCFG/automaton/group-word generators are the principled
  templates for the project's controlled synthetic tasks; compound-divergence
  splitting is the principled way to build the test-deeper sets.

## References

1. **Lin, H. W. & Tegmark, M. (2017).** *Critical Behavior in Physics and
   Probabilistic Formal Languages.* Entropy 19(7):299; arXiv:1606.06737. —
   Regular grammars give exponential MI decay; CFGs give power-law decay; the
   foundational link between grammar class and long-range correlation (β analogue).
2. **Delétang, G. et al. (2023).** *Neural Networks and the Chomsky Hierarchy.*
   ICLR; arXiv:2207.02098. — 2200-model study: OOD/length generalization tracks
   Chomsky level × memory structure; strong negative results.
3. **Strobl, L., Merrill, W., Weiss, G., Chiang, D. & Angluin, D. (2024).** *What
   Formal Languages Can Transformers Express? A Survey.* TACL. — Circuit-complexity
   bounds (TC⁰, C-RASP) on transformer expressivity; the depth wall.
4. **Bennett, C. H. (1988).** *Logical Depth and Physical Complexity.* In *The
   Universal Turing Machine: A Half-Century Survey.* — Depth = decompression time
   of a near-minimal program; low for random & trivial, high for organized; the
   core import.
5. **Antunes, L., Fortnow, L., van Melkebeek, D. & Vinodchandran, N. V. (2006).**
   *Computational Depth: Concept and Applications.* Theoretical Computer Science
   354(3):391–404. — Depth as difference of (time-bounded) Kolmogorov measures;
   computable-in-spirit proxy for Bennett depth.
6. **Finzi, M., Qiu, S., Jiang, Y., Izmailov, P., Kolter, J. Z. & Wilson, A. G.
   (2026).** *From Entropy to Epiplexity: Rethinking Information for
   Computationally Bounded Intelligence.* arXiv:2601.03220. — Epiplexity =
   extractable structural information minus time-bounded entropy; loss-curve
   estimators; correlates with OOD generalization (project anchor; γ ↔ entropy).
7. **Vereshchagin, N. & Vitányi, P. (2004).** *Kolmogorov's Structure Functions
   and Model Selection.* IEEE Trans. Info. Theory. — Algorithmic statistics:
   sophistication = structure part of K vs noise part; rigorous structure–noise split.
8. **Li, M. & Vitányi, P. (2008).** *An Introduction to Kolmogorov Complexity and
   Its Applications* (3rd ed.). Springer. — Standard reference for K, randomness
   deficiency, depth.
9. **Zhou, H. et al. (2024).** *What Algorithms can Transformers Learn? A Study in
   Length Generalization.* ICLR; arXiv:2310.16028. — RASP-L generalization
   conjecture: short length-invariant RASP-L program ⇒ length generalization.
10. **Liu, B., Ash, J. T., Goel, S., Krishnamurthy, A. & Zhang, C. (2023).**
    *Transformers Learn Shortcuts to Automata.* ICLR; arXiv:2210.10749. — O(1)
    depth for solvable, O(log T) general; parallel shortcuts brittle OOD, only
    autoregressive solutions extrapolate.
11. **Merrill, W., Petty, J. & Sabharwal, A. (2024).** *The Illusion of State in
    State-Space Models.* ICML; arXiv:2404.08819. — A₅/S₅ word problem
    (NC¹-complete) as the hard state-tracking benchmark; SSM/transformer limits.
12. **Lake, B. & Baroni, M. (2018).** *Generalization without Systematicity:
    Compositional Skills of Seq2Seq Networks (SCAN).* ICML; arXiv:1711.00350. —
    Foundational compositional + length split; in-dist fit ≠ generalization.
13. **Kim, N. & Linzen, T. (2020).** *COGS: A Compositional Generalization
    Challenge Based on Semantic Interpretation.* EMNLP. — 21 generalization types;
    96–99% in-dist vs ~35% OOD.
14. **Keysers, D. et al. (2020).** *Measuring Compositional Generalization: A
    Comprehensive Method on Realistic Data (CFQ).* ICLR; arXiv:1912.09713. —
    Compound-divergence split metric; tunable hard generalization splits.
15. **McLeish, S. et al. (2024).** *Transformers Can Do Arithmetic with the Right
    Embeddings (Abacus).* NeurIPS; arXiv:2405.17399. — Position-within-number
    embeddings + recurrence ⇒ 6× length extrapolation in addition.
16. **Cho, H., Cha, J. et al. (2024).** *Position Coupling: Improving Length
    Generalization of Arithmetic Transformers Using Task Structure.* NeurIPS;
    arXiv:2405.20671. — Tie positions to task structure for length generalization.
17. **Fan, Y. et al. (2024).** *Looped Transformers for Length Generalization.*
    arXiv:2409.15647. — Recurrence/looping recovers length generalization on
    algorithmic tasks (state re-emission mechanism).
18. **Allen-Zhu, Z. & Li, Y. (2023).** *Physics of Language Models, Part 1:
    Learning Hierarchical Language Structures (CFG).* arXiv:2305.13673. —
    Deep synthetic PCFGs; transformers learn the latent hierarchy; tunable
    generation template.
19. **Cagnetta, F., Petrini, L., Tomasini, U. M., Favero, A. & Wyart, M. (2024).**
    *Towards a Theory of How the Structure of Language Is Acquired by Deep Neural
    Networks.* arXiv:2406.00048 (and learning-curve theory, arXiv:2505.07067). —
    Sample complexity vs hierarchical-generation parameters (power-law features).
20. **Bingbin Liu / Cyril Zhang et al. (2024).** *Simulating Weighted Automata
    over Sequences and Trees with Transformers.* arXiv:2403.09728. — Construction
    bounds for weighted-automaton simulation; controlled depth generation.
21. **Zhang, S. D., Lin, H. W. & Tegmark, M. (2017).** *Do Neural Nets Learn
    Statistical Laws behind Natural Language? (power-law MI).* arXiv:1707.04848.
    — Empirical follow-up confirming learned models reproduce power-law MI decay.
22. **Yang, K., Swope, A. et al. / DeepSeek-Prover (2024).** *miniF2F &
    DeepSeek-Prover: Advancing Theorem Proving in LLMs.* arXiv:2405.14333;
    miniF2F (Zheng et al. 2022, arXiv:2109.00110). — Formal-proof benchmark;
    proof-length / autoformalization scaling — the math-reasoning analogue of
    depth-vs-length (longer proofs = deeper logical chains; marginal relevance).
23. **(2025).** *Transformers Provably Learn Chain-of-Thought Reasoning with
    Length Generalization.* arXiv:2511.07378 — *verify final ID.* — Theory that
    emitting intermediate state yields provable length generalization.

> Notes: items 19 (2505.07067), 23 (2511.07378), and the hierarchical-latent
> generator (2603.06592) are recent search-surfaced IDs; confirm exact arXiv
> numbers before citing in a paper. miniF2F original ID (2109.00110) inferred
> from standard usage — verify. All other IDs were directly confirmed in search.
