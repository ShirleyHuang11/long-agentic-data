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

## Which statistic actually carries the signal? (predictor decomposition)

Rank-correlating the measured data-scaling exponent against each candidate predictor isolates where the directional success comes from:

| predictor | Spearman vs measured exponent |
| :-- | --: |
| **β alone** | **−0.67** |
| predicted α_D = α/(2β) | 0.50 |
| H∞ (content) | 0.27 |
| α alone (γ-proxy) | 0.18 |
| BPC@32K (content) | −0.08 |

**β alone out-predicts the full α/(2β) composite** (|−0.67| > 0.50): the entire directional signal lives in the correlation-decay exponent β (lower β → faster loss decay → more sample-efficient), and folding in the noisy γ-proxy α (uninformative on its own, 0.18) *dilutes* it. The two **content** metrics are orthogonal to data-efficiency (H∞ 0.27, BPC@32K −0.08) — a clean dissociation in the expected direction: sample-efficiency is a property of a corpus's repetition structure (a *pattern* statistic), not of its incompressible content. So the honest, sharpened claim is that **β is the predictive pattern statistic** for agentic-data learnability; the specific α_D = γ/2β functional form is not yet supported at this scale.

**Robustness — is −0.67 an artifact of the curve-fit?** The measured exponent comes from an L∞ grid-fit, so the β↔efficiency correlation could be fit-dependent. Re-running it against *fit-free* efficiency proxies confirms the **direction is robust but the magnitude shrinks**:

| efficiency measure | Spearman(β, ·) |
| :-- | --: |
| fitted exponent (figure value) | −0.67 |
| raw log-log slope (no L∞) | −0.45 |
| total loss drop, D = 0.25→3M | −0.38 |
| relative drop per log-D | −0.45 |

All four keep the expected negative sign, so β genuinely predicts data-efficiency regardless of how efficiency is operationalized — but the honest magnitude is a **range ≈ −0.4 to −0.7**, with the −0.67 figure value sitting at the optimistic (fitted) end. The signal is carried mainly by the slow-decaying high-β corpora (AgentNet 1.30, JetBrains 0.52, smolagents 0.49 have the lowest slopes); the low-β end is noisy (CoderForge β = 0.15 does *not* have the highest slope). Take the predictive claim as directional, not as a precise coefficient.
