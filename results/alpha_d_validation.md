# α_D = γ/2β training validation (paper §9 #1) — RESULT

Single-pass from-scratch 29M GPT on each corpus's GPT-2 token stream (data-limited regime); val loss logged at D=0.25–3M tokens. Job 20718622 (gpu_requeue). Data: `data/alpha_d_*.json`; scripts: `alpha_d_dataprep.py`, `alpha_d_train.py`.

| corpus | β | α(γ) | predicted α_D=α/2β | measured exponent |
| :-- | --: | --: | --: | --: |
| CoderForge | 0.15 | 0.28 | 0.93 | 0.22 |
| SWE-ZERO | 0.28 | 0.26 | 0.47 | 0.26 |
| JetBrains | 0.52 | 0.35 | 0.34 | 0.17 |
| AgentNet | 1.30 | 0.14 | 0.05 | 0.15 |

**Spearman(predicted α_D, measured exponent) = 0.80 (n=4).** The low-β agentic corpora (CoderForge, SWE-ZERO) show steeper data-limited loss decay — more sample-efficient — than high-β AgentNet, the direction α_D=γ/2β predicts. Absolute exponents are compressed (0.15–0.26 measured vs 0.05–0.93 predicted): a single small model over a narrow D range (0.25–3M) recovers the *ordering*, not the theoretical scale. **n=4 is directional/suggestive, not conclusive** — a fuller test (more corpora, larger models, wider D) is forward work.
