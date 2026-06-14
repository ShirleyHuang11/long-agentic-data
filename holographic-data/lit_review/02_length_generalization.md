# Length Generalization in Sequence Models and Transformers: A Literature Review

> Scope: train-short / test-long generalization in transformers and SSMs, with
> emphasis on positional encoding, algorithmic/reasoning tasks, theory of
> learnability, architecture effects (transformer vs SSM/RoPE), and 2024–2025
> advances. Written to support the holographic pointer-chasing / nested-monoid
> length-generalization project (see `## Relevance to this project`).
>
> Web search was available; references below were checked against arXiv /
> OpenReview / ACL / NeurIPS listings during the review. arXiv IDs are given
> for traceability.

## Overview

**Length generalization (LG)** is the ability of a model trained on sequences
up to length *L* to maintain task performance on sequences longer than *L* at
test time. It is a sharp probe of whether a model has learned the *algorithm*
underlying a task versus a *length-specific pattern*. Three findings recur
across the literature and frame everything below:

1. **LG is fragile and highly contingent.** Whether a transformer
   length-generalizes depends jointly on (i) positional encoding, (ii) data
   format / task presentation, (iii) the algorithmic structure of the task,
   and (iv) even random seed and data order (Zhou et al. 2024b). There is no
   single architectural fix; success is a *conjunction* of conditions.

2. **The bottleneck is usually positional, not capacity.** Standard positional
   encodings (APE, RoPE, ALiBi) push positional features out-of-distribution
   (OOD) at test length. Removing them (NoPE), randomizing them, or coupling
   them to task structure are the most reliable interventions
   (Kazemnejad et al. 2023; Ruoss et al. 2023; McLeish et al. 2024).

3. **There is now a learnability theory.** The RASP-L conjecture (Zhou et al.
   2024a) and the identifiability framework of Huang et al. (2024) make
   *a priori* predictions about which tasks length-generalize, validated
   empirically. The headline: tasks expressible without arbitrary index
   arithmetic generalize; addition/parity/copy-with-repeats do not, unless
   the task is re-presented to remove that requirement.

## Positional encodings and extrapolation

**Absolute / sinusoidal (APE).** Vaswani et al. (2017) fixed sinusoidal
embeddings; learned APE generalizes poorly because unseen positions are OOD.

**RoPE** (Su et al., RoFormer, 2021) rotates query/key by position-dependent
angles, encoding *relative* position in the dot product. It is the de facto
standard in modern LLMs but extrapolates poorly past training length without
intervention. Three families fix this **post hoc by rescaling frequencies**:
- **Position Interpolation (PI)** (Chen et al. 2023): linearly compress
  position indices into the trained range; ~1000 steps of fine-tuning extends
  LLaMA to 32k.
- **NTK-aware scaling** (bloc97, 2023, community): scale low frequencies more
  than high, preserving high-frequency resolution lost by PI.
- **YaRN** (Peng et al. 2024): "NTK-by-parts" plus attention-logit temperature
  scaling; the first rigorous academic synthesis of community RoPE-extension
  tricks. Note Xu et al. (2024, "Base of RoPE Bounds Context Length") show the
  RoPE base frequency itself imposes a context-length ceiling.

**ALiBi** (Press et al. 2022): adds a linear distance penalty to attention
logits, no learned position vectors; strong train-short/test-long
*perplexity* extrapolation, but weaker on downstream algorithmic LG
(Kazemnejad et al. 2023).

**NoPE** (Kazemnejad et al. 2023): decoder-only transformers with *no*
explicit positional encoding implicitly learn position via the causal mask,
and **outperform** APE/RoPE/ALiBi on algorithmic LG. A key, slightly
counterintuitive result: explicit PE is not necessary and can hurt LG.

**Randomized PE** (Ruoss et al. 2023): sample position IDs from a range much
larger than training length, so test-length positions are in-distribution.
+12% average test accuracy over 15 algorithmic tasks; combinable with any PE.

**Abacus embeddings** (McLeish et al. 2024, NeurIPS): per-digit positional
embeddings aligning digits of equal significance. Trained on ≤20-digit
addition, generalizes to 120 digits (**6× LG**, vs prior 2.5×); with input
injection + recurrence reaches ~99% on 100-digit addition.

**Position coupling** (Cho et al. 2024): manually assign shared position IDs to
tokens that must interact (e.g. aligned digits), injecting task structure into
PE; provable and strong LG on addition and related tasks.

**FIRE** (Li et al. 2024, Functional Interpolation for Relative PE): a learned,
input-length-normalized relative bias; one ingredient of the best-known
addition recipe (Zhou et al. 2024b).

## Algorithmic length generalization

Algorithmic tasks isolate the algorithm-vs-pattern question:

- **Addition / arithmetic.** The canonical LG testbed. Jelassi et al. (2023)
  show relative PE enables 5→15-digit addition but fails for multiplication
  (fixed by "train-set priming" with a few long examples). Zhou et al. (2024b,
  "…But Not Robustly") give the best recipe — FIRE + randomized positions +
  reversed digit order + index hints — reaching 2.5× extrapolation, but with
  **large seed-to-seed variance**: LG is achievable yet *fragile*.
- **Parity.** A classic failure case; Anil et al. (2022) trace failure to
  "distracting tokens" in the input, and find scratchpad alone insufficient.
- **Copying.** Generalizes only when inputs avoid n-gram repetition (Jelassi
  et al. 2024); duplicated n-grams break the induction-style copy mechanism.
- **Counting, mode, sorting.** Generalize readily — they have natural RASP-L
  programs (Zhou et al. 2024a).
- **Pointer chasing / variable binding / state tracking.** Inherently
  sequential, requiring depth (or CoT steps) that scales with logical depth;
  hard for constant-depth transformers (see Theory).
- **Induction heads** (Olsson et al. 2022): the two-layer match-and-copy
  circuit ([…A B…A]→B) underlying in-context learning; extrapolate well in SSMs
  but degrade in transformers as positional features go OOD.

**Scratchpad / chain-of-thought.** Decomposing into intermediate steps
(Nye et al. 2021; Anil et al. 2022) helps but is not sufficient for LG on its
own; the *format* and PE still gate generalization. CoT provably raises
expressivity (more below), turning a parallel-depth bottleneck into a
serial-token-budget one.

## Theory (what is learnable)

**RASP-L conjecture** (Zhou et al. 2024a, "What Algorithms can Transformers
Learn?"): transformers length-generalize *exactly* on tasks expressible as
short programs in **RASP-L**, a restricted RASP fragment that forbids arbitrary
index arithmetic (only order comparisons and successor/predecessor on indices)
and is built from elementwise ops + a causal-attention primitive (`kqv`).
Counting/mode/copy-unique/sort have RASP-L programs (→ LG); addition/parity/
copy-with-repeats do not (→ no LG). This converts "will it generalize?" into a
question about program existence.

**Identifiability framework** (Huang et al. 2024, ICLR 2025, "A Formal
Framework…"): for causal transformers with learnable APE under a norm-based
regularizer and an idealized inference scheme, a function length-generalizes
iff it is **identifiable in the limit** from sufficiently long inputs. Gives
provable yes/no predictions matching empirics across algorithmic and
formal-language tasks — the most rigorous current account.

**Circuit-complexity ceiling.** A fixed transformer forward pass lies in
log-uniform **TC⁰** (Merrill & Sabharwal; Strobl et al. survey). Hard-attention
variants are weaker (AC⁰, cannot do PARITY). Consequence: state tracking,
graph connectivity, and unbounded sequential reasoning are *not* solvable by a
constant-depth transformer on unbounded inputs — a fundamental LG limit. Two
escapes: (i) depth growing as O(log n), (ii) **chain-of-thought**, where
constant-depth transformers with O(log n) embedding width can simulate any
size-T circuit using T CoT steps (Merrill & Sabharwal 2024). Lower bounds on
required CoT length also exist (Amiri et al. 2025).

**Sparsity** (Sabbaghi/Bhojanapalli et al. 2025, "The Role of Sparsity for
LG"): sparse/local attention structure is a key enabler of LG — long-range
dense attention is the part that goes OOD.

## Architecture effects: transformer vs SSM/RoPE

**Mamba / selective SSMs** (Gu & Dao 2023): linear-time recurrent models with
input-dependent state transitions. On synthetic copying and induction-head
tasks Mamba can extrapolate to >1M tokens (4000× training length) where
transformers fail; Mamba-130M solves 16k passkey retrieval where a comparable
Pythia transformer fails entirely. The recurrent state means *no positional
features to go OOD* — a structural LG advantage.

**Caveats on SSM LG.** The advantage is not universal:
- DeciMamba (Ben-Kish et al. 2024) shows vanilla Mamba's *effective* context
  is bounded and must be explicitly extended.
- Mamba Modulation (2025) studies why Mamba LG still degrades and how to fix
  the state-transition spectrum.
- SSMs trade off **copying / in-context retrieval** capacity vs transformers
  (Jelassi et al. 2024; the recall-vs-memory tension). Recall-with-Reasoning
  (2025) uses CoT distillation to patch Mamba long-context recall.

**RoPE vs alternatives for LG.** Within transformers, the empirical LG ranking
on algorithmic tasks is roughly NoPE ≳ randomized/FIRE/Abacus ≳ ALiBi ≳ RoPE ≳
APE (Kazemnejad et al. 2023; Ruoss et al. 2023). RoPE's strength is *trained*
in-context performance and cheap post-hoc extension (PI/YaRN), not native
extrapolation.

## Recent advances (2024–2025)

- **Best addition recipe + fragility** (Zhou et al. 2024b): 2.5× LG but
  seed-fragile — caution for any single-seed LG claim.
- **Abacus & position coupling** (McLeish 2024; Cho 2024): task-structured PE
  pushes arithmetic LG to 6×+ and 100-digit addition.
- **Looped transformers for LG** (Fan et al. 2024): weight-tied recurrence with
  adaptive step count generalizes on algorithmic tasks by decoupling depth from
  length.
- **Formal/identifiability theory matures** (Huang et al. 2024 ICLR'25;
  RASP-L ICLR'24) — LG predictions are now derivable, not just measured.
- **Sparsity as enabler** (Bhojanapalli et al. 2025).
- **SSM length-generalization deep-dives** (DeciMamba 2024; Mamba Modulation
  2025; Recall-with-Reasoning 2025) — the SSM LG advantage is real but bounded
  and architecture-detail-dependent.
- **Provable CoT + LG** (Transformers Provably Learn CoT Reasoning with Length
  Generalization, 2025) — training-dynamics-level account of when CoT yields LG.
- **Pretraining shapes LG** (Born a Transformer — Always a Transformer?, 2025):
  pretraining endows/limits specific algorithmic LG abilities.

## Open questions

1. **What single quantity predicts LG?** RASP-L and identifiability give binary
   predictions; we lack a *graded* predictor that says how much LG a given
   (task, model, data distribution) yields — exactly what a data-complexity
   axis like (β, γ) could supply.
2. **Data structure vs architecture vs PE — which dominates, and when?**
   Most papers vary one axis. Controlled factorial studies are rare.
3. **Is the SSM LG advantage about state recurrence or about avoiding OOD
   positions?** Disentangling these would tell us whether a positionless
   transformer (NoPE) closes the gap.
4. **Logical depth vs sequence length.** Tasks where logical depth ≫ input
   length (deep pointer chasing on short inputs) are under-studied; classic LG
   is length-driven, not depth-driven.
5. **Robustness of LG claims.** Seed fragility (Zhou 2024b) means many LG
   results need multi-seed replication and absolute (not ratio) metrics.

## Relevance to this project

This project trains 100M-param transformers (and Mamba/RoPE variants) on
algorithmic tasks parametrized by correlation-decay **β** and entropy/noise
**γ**, measuring LG. Concrete ties:

- **Holographic pointer-chasing (33-token input, logical depth 128–1024).**
  This is a *depth-driven* LG task, not the *length-driven* setting most of the
  literature studies — it probes Open Question #4 directly. The TC⁰ ceiling
  and the CoT-expressivity results explain *why* a constant-depth transformer
  saturates with logical depth (the "attention saturation" in
  `DATA_EXPLANATION.md`): deep sequential pointer chasing is exactly the
  state-tracking class that needs depth O(log n) or CoT steps. This predicts
  scratchpad/CoT should rescue deep chasing where raw next-token cannot — a
  testable extension of the H1/H3 campaign.

- **(β, γ) KV-retrieval phase diagram and the LG ratio acc(2048)/acc(512).**
  β (power-law retrieval-distance decay) is, mechanistically, a *sparsity /
  locality* knob — high β ⇒ local/recent retrieval, low β ⇒ long-range. This
  is precisely the variable Bhojanapalli et al. (2025) identify as the LG
  enabler, and it matches the project's own finding (`holo_length_gen.md` H1)
  that **vanilla RoPE transformers extrapolate better under large β
  (local dependence)** and fail when test length exceeds trained retrieval
  distance at small β. The literature thus *predicts the sign* of the H1
  effect and reframes the "edge-of-chaos / max-complexity ridge" hypothesis:
  RASP-L/identifiability theory suggests LG is gated by whether the retrieval
  is expressible without OOD index arithmetic, not by a complexity sweet spot —
  consistent with H1 being **refuted** (no edge-of-chaos optimum; retention
  flat/monotone in β).

- **The retention "cheat-guard."** The project's insistence that
  retention = acc(long)/acc(short) is only trustworthy when train_acc is high
  is *independently validated* by Zhou et al. (2024b): LG ratios are
  seed-fragile and dominated by the denominator at low absolute accuracy. The
  project's negative result (the "holographic ridge" being a low-denominator
  artifact) is methodologically aligned with the field's move toward absolute
  long-length accuracy + multi-seed reporting.

- **Transformer vs Mamba (H3, inconclusive in `holo_length_gen.md`; clean
  N=3 KV result: Mamba 2.0–2.6× retention).** Directly tests the Gu & Dao /
  DeciMamba picture: SSM state-recurrence avoids OOD positional features, so
  Mamba retention ≈ 1.0 at the strip is exactly the predicted SSM LG
  advantage — *and* the predicted failure at the chaos/edge anchor matches
  DeciMamba's "bounded effective context" caveat. The project's main-findings
  Result 11 (Mamba advantage largest where data has structured long-range
  dependence, collapses at edge-of-chaos) is a clean empirical instance of the
  SSM-LG-is-real-but-bounded literature.

- **Nested-monoid task (DEF/USE register machine + named-operator recall).**
  This is a *variable-binding / state-tracking* task — the class the
  circuit-complexity theory flags as the hard core of LG. Comparing `op_kind`
  =perm (table lookup, RASP-L-friendly) vs affine (mod arithmetic, found
  grokking-hard/unlearnable in the project) is a direct, in-house instance of
  the RASP-L learnability boundary: index-arithmetic-like tasks resist LG.

**Suggested literature-grounded next steps for the project:** (1) add a NoPE
and a randomized-PE transformer arm — theory predicts both should beat the
RoPE baseline on LG and could change the H1 verdict; (2) test scratchpad/CoT
on deep pointer-chasing to probe the TC⁰/CoT-expressivity prediction;
(3) frame β explicitly as the sparsity/locality axis of Bhojanapalli et al.
(2025) rather than (only) a correlation-decay knob.

## References

1. Vaswani et al. (2017). *Attention Is All You Need.* arXiv:1706.03762 —
   introduces transformers and sinusoidal absolute positional encoding.
2. Su et al. (2021). *RoFormer: Enhanced Transformer with Rotary Position
   Embedding.* arXiv:2104.09864 — RoPE; relative position via rotation.
3. Press, Smith, Lewis (2022). *Train Short, Test Long: Attention with Linear
   Biases (ALiBi).* arXiv:2108.12409 — distance-penalty bias enabling
   perplexity extrapolation.
4. Olsson et al. (2022). *In-context Learning and Induction Heads.*
   arXiv:2209.11895 — match-and-copy circuit underlying ICL; relevant to copy
   and pointer mechanisms.
5. Nye et al. (2021). *Show Your Work: Scratchpads for Intermediate
   Computation with Language Models.* arXiv:2112.00114 — scratchpad for
   multi-step algorithmic tasks.
6. Anil et al. (2022). *Exploring Length Generalization in Large Language
   Models.* NeurIPS 2022; arXiv:2207.04901 — finetune/prompt/scratchpad LG
   trade-offs; distracting-token failure on PARITY.
7. Kazemnejad et al. (2023). *The Impact of Positional Encoding on Length
   Generalization in Transformers.* NeurIPS 2023; arXiv:2305.19466 — NoPE
   beats APE/RoPE/ALiBi on algorithmic LG.
8. Ruoss et al. (2023). *Randomized Positional Encodings Boost Length
   Generalization of Transformers.* ACL 2023; arXiv:2305.16843 — sample
   positions from a large range; +12% over 15 tasks.
9. Jelassi et al. (2023). *Length Generalization in Arithmetic Transformers.*
   arXiv:2306.15400 — relative PE for addition LG; train-set priming for
   multiplication.
10. Zhou et al. (2024a). *What Algorithms can Transformers Learn? A Study in
    Length Generalization.* ICLR 2024; arXiv:2310.16028 — the RASP-L
    conjecture; program-existence predicts LG.
11. Chen et al. (2023). *Extending Context Window of Large Language Models via
    Positional Interpolation.* arXiv:2306.15595 — PI; LLaMA to 32k with light
    fine-tuning.
12. Peng et al. (2024). *YaRN: Efficient Context Window Extension of Large
    Language Models.* ICLR 2024; arXiv:2309.00071 — NTK-by-parts + logit
    scaling RoPE extension.
13. Gu & Dao (2023). *Mamba: Linear-Time Sequence Modeling with Selective
    State Spaces.* arXiv:2312.00752 — selective SSM; extrapolates copy/induction
    to >1M tokens.
14. Zhou et al. (2024b). *Transformers Can Achieve Length Generalization But
    Not Robustly.* arXiv:2402.09371 — FIRE + randomized + reversed + index
    hints → 2.5× addition LG; seed-fragile.
15. McLeish et al. (2024). *Transformers Can Do Arithmetic with the Right
    Embeddings (Abacus).* NeurIPS 2024; arXiv:2405.17399 — per-digit PE; 6× LG,
    100-digit addition.
16. Cho et al. (2024). *Position Coupling: Improving Length Generalization of
    Arithmetic Transformers Using Task Structure.* NeurIPS 2024;
    arXiv:2405.20671 — shared position IDs encode task structure; provable LG.
17. Li et al. (2024). *Functional Interpolation for Relative Positions Improves
    Long Context Transformers (FIRE).* ICLR 2024; arXiv:2310.04418 — learned
    length-normalized relative bias.
18. Huang et al. (2024). *A Formal Framework for Understanding Length
    Generalization in Transformers.* ICLR 2025; arXiv:2410.02140 —
    identifiability-in-the-limit predicts LG; provable yes/no.
19. Merrill & Sabharwal (2023/2024). *The Parallelism Tradeoff: Limitations of
    Log-Precision Transformers* (TC⁰) and *The Expressive Power of Transformers
    with Chain of Thought.* arXiv:2207.00729 / arXiv:2310.07923 — TC⁰ ceiling;
    CoT raises expressivity to size-T circuits with T steps.
20. Strobl et al. (2024). *What Formal Languages Can Transformers Express? A
    Survey.* TACL — expressivity / circuit-class limits incl. hard-attention
    AC⁰ and PARITY.
21. Bhojanapalli/Sabbaghi et al. (2025). *The Role of Sparsity for Length
    Generalization in Transformers.* arXiv:2502.16792 — sparse/local attention
    structure as a key LG enabler.
22. Fan et al. (2024). *Looped Transformers for Length Generalization.*
    arXiv:2409.15647 — weight-tied adaptive recurrence decouples depth from
    length.
23. Ben-Kish et al. (2024). *DeciMamba: Exploring the Length Extrapolation
    Potential of Mamba.* arXiv:2406.14528 — Mamba's effective context is
    bounded; method to extend it.
24. (2025). *Mamba Modulation: On the Length Generalization of Mamba.*
    arXiv:2509.19633 — analyzes/fixes Mamba LG via state-transition spectrum.
25. (2025). *Recall with Reasoning: Chain-of-Thought Distillation for Mamba's
    Long-Context Memory and Extrapolation.* arXiv:2505.03320 — CoT distillation
    to patch SSM long-context recall.
26. Jelassi et al. (2024). *Repeat After Me: Transformers are Better than State
    Space Models at Copying.* arXiv:2402.01032 — copy/recall advantage of
    transformers over SSMs; copy LG needs no n-gram repetition.
27. Xu et al. (2024). *Base of RoPE Bounds Context Length.* arXiv:2405.14591 —
    RoPE base frequency caps achievable context length.
28. (2025). *Born a Transformer — Always a Transformer? On the Effect of
    Pretraining on Architectural Abilities.* arXiv:2505.21785 — pretraining
    shapes which algorithmic LG abilities are available.
</content>
</invoke>
