# Agentic LZ-vs-neural validation (paper §9 experiment #5) — RESULT

**Run:** Qwen2.5-0.5B oracle (mean bits/token) over **102 registry corpora** with cached
text samples (full H∞ range [0.00–1.95]); GPU job 20678563 on gpu_requeue.
Data: `data/neural_oracle_bpc.csv`. Script: `scripts/neural_oracle_validation.py`.

## Headline

| correlation (Spearman, n=102) | value |
| :-- | --: |
| neural bits/token ↔ **BPC@32K** (within-corpus zstd) | **0.54** |
| neural bits/token ↔ **3-point pooled H∞** | **0.17** |
| neural bits/token ↔ H∞, **LZ-nonzero corpora only** (n=61) | 0.40 |
| H∞ ↔ BPC@32K (consistency check, paper reports 0.60) | 0.58 ✓ |

## Interpretation

The formal-math LZ↔neural agreement (Spearman 0.97) does **not** transfer cleanly to
agentic data. The reason is diagnostic, not disqualifying:

- A context-bounded neural LM measures **within-window** density — and it tracks the
  directly-measured **BPC@32K** (also within-corpus) at 0.54.
- The 3-point **pooled** H∞ additionally compresses **across episodes**, capturing the
  shared-scaffold repetition that *defines* agentic template-degeneracy (§5.3). A
  per-window neural LM cannot see that, so it diverges from H∞ (0.17), most of the gap
  coming from the floor/pooled corpora (excluding them lifts the correlation to 0.40).

So on agentic data, **H∞ = 0 means "cross-episode-repetitive," not "neurally empty per
episode."** Poster case: `agent-flan-all` reads **H∞ 0** yet the **highest** neural
bits/token measured (**3.29**) — diverse individual episodes, one repeated SFT scaffold.

This **reinforces** the §5.3 harness-pooling story (the neural oracle independently
confirms the pooling is cross-episode) while honestly bounding the metric: the LZ H∞ is a
cross-episode incompressibility measure, complementary to — not a proxy for — per-window
neural content (which BPC@32K is the registry's stand-in for).
