# α_D = γ/2β training validation (paper §9 #1) — RESULT

Single-pass from-scratch 29M GPT on each corpus's GPT-2 token stream (data-limited regime); val loss logged at D=0.25–3M tokens. Job 20718622 (gpu_requeue). Data: `data/alpha_d_*.json`; scripts: `alpha_d_dataprep.py`, `alpha_d_train.py`.

| corpus | β | α(γ) | predicted α_D=α/2β | measured exponent |
| :-- | --: | --: | --: | --: |
| WebLINX | 0.26 | 0.49 | 0.93 | 0.61 |
| CoderForge | 0.15 | 0.28 | 0.93 | 0.22 |
| Glaive-FC | 0.27 | 0.28 | 0.52 | 0.53 |
| SWE-ZERO | 0.28 | 0.26 | 0.47 | 0.26 |
| JetBrains | 0.52 | 0.35 | 0.34 | 0.17 |
| smolagents-GAIA | 0.49 | 0.30 | 0.31 | 0.16 |
| APIGen | 0.30 | 0.18 | 0.30 | 0.34 |
| tau-bench | 0.34 | 0.17 | 0.25 | 0.39 |
| AgentNet | 1.30 | 0.14 | 0.05 | 0.15 |

**Spearman(predicted α_D, measured exponent) = 0.50 (n=9).** Positive but modest: the prediction's *direction* holds — low-β agentic data tends to be more sample-efficient (WebLINX, Glaive, SWE-ZERO decay faster than high-β AgentNet) — but the quantitative α_D=γ/2β prediction is far from precise at this scale, with notable discordant points (CoderForge predicted 0.93 but measured 0.22; tau-bench predicted 0.25 but measured 0.39). An earlier n=4 subset gave a stronger Spearman 0.80; the larger sample reveals that was optimistic. Absolute exponents are also compressed (measured 0.15–0.61 vs predicted 0.05–0.93): a single small model over a narrow D range (0.25–3M) recovers a weak ordering, not the theoretical scale. **Honest read: preliminary, modest positive support for the pattern statistics as a predictive signal — not a confirmation.** A fuller test (larger models, wider D, β/α measured on the exact training serialization) is forward work.
