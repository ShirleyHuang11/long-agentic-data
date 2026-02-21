# Report: Zipf Four-Phase LLM Manifold Experiment

## 1. Motivation

Understanding how sequence models generalize across **context length** and **data regime** is central to building reliable long-context systems. Real-world data is not homogeneous: Wikipedia-style text exhibits strong locality and Zipf-like statistics, while chain-of-thought (CoT) reasoning mixes local and long-range dependencies, and pathological regimes (e.g., uniform random access) stress both attention and state-space architectures.

This experiment treats different **data regimes as physical phases** parameterized by:

- **β (beta)**: Controls the *pointer jump distribution* when the model must retrieve a value from memory. For β > 0, retrieval distances follow a Zipf-like power law \( P(d) \propto 1/d^{\beta+1} \), so “recent” memories are accessed more often. For β → 0, the distribution becomes uniform over all stored memories (“the abyss”).
- **γ (gamma)**: Fraction of tokens that are *pure noise* (filler). High γ mimics natural text with many non-reasoning tokens; low γ yields denser logical structure.

Goals:

1. **Phase characterization**: Compare model behavior across four anchor regimes—Natural (Wikipedia-like), CoT (reasoning), Edge of Chaos (sweet spot), and The Abyss (topological collapse).
2. **Length generalization**: Train at a fixed short length (256) and evaluate at 256, 512, 1024, 2048, 4096, and 8192 to see how **task loss** and **accuracy** scale with sequence length in each phase.
3. **Architecture comparison**: Contrast a **causal Transformer** (2-layer) with a **Mamba-style SSM** (6-layer) under identical data and training, to see which regime favors which inductive bias.

---

## 2. Task and Data

- **Task**: Key–value associative recall. The sequence is a mix of:
  - **Noise tokens** (vocab 1–19), with probability γ.
  - **Writes**: key K (20–39) followed by value V (40–59), forming a memory slot (K → V).
  - **Reads**: a key K is presented; the model must predict the *correct value V* from the current memory (only at read positions is loss computed).

- **Memory access**: On a read, the “target” memory slot is chosen by a pointer jump of distance \( d \) (1 = most recent). For β > 0.01, \( d \) is drawn from a Zipf-style distribution with exponent β+1; for β ≤ 0.01 (Abyss), \( d \) is uniform over all stored slots.

- **Regimes**:
  - **Natural (Wikipedia)** (β=2.0, γ=0.8): Strong recency bias, lots of filler — similar to natural text.
  - **CoT (reasoning with explanations)** (β=0.5, γ=0.4): Softer recency, moderate filler.
  - **Edge of Chaos (o1/sweet spot)** (β=0.05, γ=0.05): Very weak recency, little noise.
  - **The Abyss (topological collapse zone)** (β=0.0, γ=0.0): Uniform random access, no filler — worst case for structure.

---

## 3. Experimental Setup

- **Training**: Sequence length \( L = 256 \); 501 steps per regime; AdamW (lr 3e-3) with cosine annealing; batch size 64. Separate training per regime for both Transformer and Mamba (no mixed training).
- **Evaluation**: Same β, γ as training; lengths \( L \in \{256, 512, 1024, 2048, 4096, 8192\} \). Batch size 64 when possible, reduced (e.g., to 8 for long sequences) on OOM.
- **Metrics**: **T** = Transformer, **M** = Mamba. For each (regime, length): **loss** (masked cross-entropy at read positions) and **accuracy** (fraction of correct value predictions at read positions).

---

## 4. Results Summary

### 4.1 Natural (Wikipedia) — β=2.0, γ=0.8

| Length | T Loss | T Acc  | M Loss | M Acc  |
|--------|--------|--------|--------|--------|
| 256    | 1.27   | 65.64% | 0.58   | 85.95% |
| 512    | 2.36   | 36.50% | 0.59   | 85.08% |
| 8192   | 3.20   |  7.08% | 0.64   | 84.23% |

- **Mamba** keeps **M Acc ~84–86%** across all lengths; loss stays low (~0.58–0.66).
- **Transformer** starts strong at L=256 (65.64%) but **degrades sharply** with length (36.5% → 21.6% → … → 7.08%); loss rises to ~3.2.
- **Interpretation**: Zipf-heavy, recency-dominated access is well matched by the SSM’s recurrent state; the Transformer’s quadratic cost and fixed context may be less efficient for this long-tail, length-scaled regime.

### 4.2 CoT (reasoning with explanations) — β=0.5, γ=0.4

| Length | T Loss | T Acc  | M Loss | M Acc  |
|--------|--------|--------|--------|--------|
| 256    | 2.07   | 37.07% | 1.57   | 51.71% |
| 8192   | 3.07   |  6.20% | 1.93   | 43.81% |

- Both models sit between Natural and Edge-of-Chaos: **M Acc** ~52% → ~44%; **T Acc** ~37% → ~6%.
- Mamba again keeps much higher accuracy at long lengths; Transformer accuracy collapses by L=8192.

### 4.3 Edge of Chaos (o1/sweet spot) — β=0.05, γ=0.05

| Length | T Loss | T Acc  | M Loss | M Acc  |
|--------|--------|--------|--------|--------|
| 256    | 2.43   | 23.82% | 2.16   | 33.50% |
| 8192   | 3.00   |  6.27% | 2.73   | 21.19% |

- Harder for both: weak recency, dense logic. **M Acc** ~33% → ~21%; **T Acc** ~24% → ~6%.
- Mamba still clearly better than Transformer at every length, but absolute performance is modest.

### 4.4 The Abyss (topological collapse) — β=0.0, γ=0.0

| Length | T Loss | T Acc  | M Loss | M Acc  |
|--------|--------|--------|--------|--------|
| 256    | 2.64   | 15.38% | 2.65   | 15.49% |
| 8192   | 2.99   |  5.55% | 3.00   |  5.72% |

- **T and M almost coincide**: same loss and accuracy within rounding. At L=256, ~15% acc (near random for the value set); by L=8192, ~5.5–6%.
- **Interpretation**: With no Zipf structure and uniform random access, neither architecture has a useful inductive bias; both approach the same “topological collapse” and behave like random guessers.

---

## 5. Conclusions

1. **Phase structure**: The four regimes behave as distinct phases. Natural (high β, high γ) is easiest and most discriminative; Abyss (β=γ=0) is hardest and architecture-agnostic; CoT and Edge of Chaos sit in between.
2. **Length generalization**: Training at L=256 only, **Mamba** generalizes length much better than the **Transformer** in every non-Abyss regime (e.g., M Acc ~84% vs T Acc ~7% at L=8192 in Natural). The Transformer’s accuracy drops sharply as length increases.
3. **Inductive bias**: Zipf-like, recency-biased access (Natural, then CoT) favors the SSM’s recurrent state; when that structure is removed (Abyss), both models perform equally poorly.
4. **Practical implication**: For long-context, Wikipedia-like or CoT-like data, Mamba-style SSMs can maintain high accuracy out to 8K+ tokens under this synthetic setup, while a small Transformer trained at 256 fails to generalize. The Abyss regime is a useful stress test where no architecture should be expected to do well.

---

## 6. References to Experiment Artifacts

- **Script**: `case/scratchpad/zipf_4phase.py` — data generation (Zipf pointer jumps, write/read, noise), training/eval loop, and four-regime setup.
- **Run output**: `case/scratchpad/logs/holo_61310517.out` — full table of T/M loss and accuracy per regime and length.
