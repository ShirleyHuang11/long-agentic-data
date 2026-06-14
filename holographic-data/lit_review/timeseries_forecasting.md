# Long-Horizon Forecasting & Horizon/Length Extrapolation in Time-Series Modeling

*Method-focused literature review, assembled to import length-generalization (train-short / test-long) ideas
into the holographic-data project (algorithmic tasks parametrized by correlation-decay β and entropy-decay γ;
Transformer / Mamba / RoPE comparison; retention r = acc(L_long)/acc(L_short)).*

Web search **was available** (June 2026); references were verified against arXiv/OpenReview where possible.

---

## Overview

"Long-horizon forecasting" in time series and "length generalization" in algorithmic sequence modeling are
two faces of the same problem: **how does prediction quality degrade as you ask a model to operate over a
span longer than its comfortable/trained regime?** The time-series community has converged on a small set of
recurring mechanisms that buy robustness over long spans, each of which maps onto a knob the holographic
project already varies:

1. **Patching / tokenizing sub-series** — shorten the effective sequence so positional and attention
   structure has to extrapolate less (PatchTST, TimesFM, MOMENT).
2. **Series decomposition (trend/seasonal, Fourier, multi-rate)** — separate the slowly-varying long-memory
   component (low β / long-range) from the fast/noise component (high γ), and model each at its own scale
   (Autoformer, FEDformer, N-HiTS, Pyraformer, TimesNet).
3. **Direct (multi-output) vs. recursive (autoregressive) multi-step** — the central trade between
   error-accumulation (recursive) and horizon-specific bias (direct); directly relevant to why retention
   collapses at 4× length.
4. **Linear / scale-invariant baselines** — DLinear shows a *channel-independent linear map from a fixed
   lookback to the whole horizon* is hard to beat, i.e. much "long-horizon" gain is really good handling of
   trend, not long-range attention.
5. **Long-memory limits** — classical ARFIMA/Hurst theory gives the *information-theoretic ceiling* on
   multi-step forecastability as a function of the long-range-dependence exponent — a direct analogue of β.

A consistent empirical finding across the deep-forecasting literature (Zeng et al.; the foundation-model
benchmarks) is that **architectural complexity helps less than (a) decomposition/patching pre-processing and
(b) the intrinsic long-memory structure of the data** — which is exactly the holographic project's negative
result that data shaping < model size < architecture (Mamba).

---

## Deep long-horizon forecasting

The "LTSF" (long-term time-series forecasting) line trains on a fixed lookback window and predicts a fixed
(long) horizon, almost always with a **direct multi-output head**. Key models, in rough chronological order:

- **Informer** (Zhou et al., AAAI 2021, *Best Paper*; arXiv:2012.07436). ProbSparse self-attention +
  self-attention distilling bring complexity to O(L log L); a *generative-style decoder emits the whole long
  horizon in one forward pass* (one-shot direct forecasting) instead of step-by-step, explicitly to avoid
  recursive error accumulation. First to make L≈10³ horizons tractable.
- **Autoformer** (Wu et al., NeurIPS 2021; arXiv:2106.13008). Moves *series decomposition inside* the block
  as a recurring operator, and replaces dot-product attention with an **Auto-Correlation** mechanism that
  aggregates sub-series at the dominant periods (found via FFT). Decomposition isolates the trend
  (long-memory) channel from the seasonal channel.
- **Pyraformer** (Liu et al., ICLR 2022; OpenReview 0EXmFzUn5I). **Pyramidal attention**: a tree of
  coarser-to-finer scales so the *maximum signal-traversal path is O(1) in L* while compute is O(L). A
  multi-resolution inductive bias for long-range dependence — directly analogous to making long-range
  retrieval (small β) reachable in few hops.
- **FEDformer** (Zhou et al., ICML 2022). Frequency-Enhanced Decomposition: attention in a randomly-sampled
  sparse Fourier basis → linear complexity; assumes the series has a *sparse spectral representation*. Strong
  prior that long-range structure lives in a few low frequencies.
- **DLinear / NLinear** (Zeng et al., "Are Transformers Effective for Time Series Forecasting?", AAAI 2023;
  arXiv:2205.13504). A *one-layer linear map* (with Autoformer-style trend/seasonal decomposition for DLinear)
  from lookback to horizon beats all the above transformers on most LTSF benchmarks. The field's most cited
  cautionary result: **much apparent long-horizon skill is trend extrapolation, not long-range attention** —
  permutation-invariant self-attention can even *destroy* the temporal ordering that a linear map keeps.
- **PatchTST** (Nie et al., ICLR 2023; arXiv:2211.14027). Two ideas that *rescued* transformers post-DLinear:
  (i) **patching** — group adjacent steps into patch tokens, cutting the token count ~P× so attention and
  positions extrapolate over far fewer positions; (ii) **channel independence** — share one model across
  variates. ~21% MSE reduction over the best prior transformer. Patching is the single most transferable
  "make it length-robust" trick.
- **iTransformer** (Liu et al., ICLR 2024 Spotlight; arXiv:2310.06625). *Inverts* the axes: each **variate
  becomes a token**, attention models cross-variate correlation, the FFN models per-variate temporal
  dynamics. Notably improves **utilization of arbitrarily long lookback windows** (vanilla transformers
  degrade as lookback grows) — a length-robustness result.
- **TimesNet** (Wu et al., ICLR 2023; arXiv:2210.02186). Reshapes a 1D series into a stack of **2D tensors,
  one per discovered period** (FFT-selected), so intra- and inter-period variation become 2D convolutions.
  Multi-periodicity as an explicit multiscale inductive bias.
- **N-BEATS** (Oreshkin et al., ICLR 2020; arXiv:1905.10437). Deep stack of fully-connected **doubly-residual
  blocks** producing backcast+forecast; basis-expansion blocks (trend polynomial, seasonal Fourier) give
  interpretability. Pure MLP, no attention/recurrence — strong evidence that long-horizon skill needn't come
  from a sequence model at all.
- **N-HiTS** (Challu et al., AAAI 2023; arXiv:2201.12886). Adds **multi-rate input sampling + hierarchical
  interpolation**: each block predicts at a different temporal resolution, summed into the final forecast.
  ~25% better than N-BEATS on long horizons at a fraction of the compute — the cleanest "decompose the
  horizon by scale" recipe.

---

## Time-series foundation models & horizon generalization

These are pre-trained once on huge heterogeneous corpora and used **zero-shot** at arbitrary context and
horizon lengths — the closest external analogue of the project's train-short/test-long question. How each
handles *variable/longer* context & horizon:

- **TimeGPT-1** (Garza et al., Nixtla, 2023; arXiv:2310.03589). First TS foundation model; transformer
  encoder–decoder trained on >100B points. Zero-shot across granularities/horizons; the decoder rolls out to
  the requested (variable) horizon.
- **TimesFM** (Das et al., Google, ICML 2024; arXiv:2310.10688). **Decoder-only, patched** (200M params,
  ~100B+ points). **Dynamic/random patch masking during training** is the explicit mechanism for context-
  length generalization; the autoregressive decoder *naturally extrapolates to longer horizons than any
  single training horizon* by predicting after a variable number of input patches. Most directly mirrors the
  project's decoder-only setup; suggests **mask/length-curriculum training** as a length-gen lever.
- **Chronos** (Ansari et al., Amazon, 2024; arXiv:2403.07815). **Tokenizes** values by scaling+quantizing
  into a fixed vocabulary, then trains a vanilla **T5** with cross-entropy — i.e. treats forecasting as
  language modeling. Augments with synthetic Gaussian-process data (KernelSynth). Shows that *generic LM
  machinery + tokenization* transfers to TS; horizons handled by autoregressive rollout.
- **Moirai** (Woo et al., Salesforce, ICML 2024; arXiv:2402.02592). **Masked-encoder** universal forecaster;
  **any-variate attention** and *multiple patch-size projections* to handle different frequencies; trained on
  LOTSA (27B+ obs). Patch-size choice is its scale/horizon-adaptation mechanism.
- **MOMENT** (Goswami et al., CMU, ICML 2024; arXiv:2402.03885). **Masked patch reconstruction** (BERT-style)
  on the "Time-Series Pile"; a general-purpose encoder for forecasting/classification/anomaly/imputation.
  Fixed input length via patching + padding/masking.
- **Lag-Llama** (Rasul et al., 2023; arXiv:2310.08278). Decoder-only **LLaMA-style** model whose tokens are
  **lag features** (values at a set of lags) — bakes multi-scale temporal structure into the input
  representation rather than the architecture.
- **Timer-XL / Towards Long-Context TSFMs** (Liu et al., 2024; arXiv:2410.04803, 2409.13530). Explicitly
  target **long context**: extend the usable context window of TS foundation models and study how forecast
  quality scales with context length — the most on-topic "length generalization for TS" work.

**Cross-cutting caveat** (Foundation Models for Time Series surveys, 2024–2025; e.g. arXiv:2510.00742
"How Foundational are Foundation Models…"): zero-shot length/horizon generalization is **good on
periodic/structured series and poor on irregular real-world data** — i.e. extrapolation quality is governed
by the *data's* statistical structure, echoing the project's finding that retention depends on β/γ, not just
the model.

---

## Classical long-memory forecasting (ARFIMA)

- **ARFIMA(p, d, q)** (Granger & Joyeux 1980; Hosking 1981). Generalizes ARIMA by allowing a **fractional**
  differencing order d∈(−0.5, 0.5). The autocorrelation decays as a **power law** ρ(k) ∼ k^(2d−1) (long
  memory) rather than the geometric decay of ARMA — this is *precisely the project's β knob*:
  power-law correlation decay p(d) ∝ d^{−(β+1)}.
- **Hurst exponent ↔ d**: H = d + 1/2. H > 1/2 (d > 0) ⇒ persistent long-range dependence; H = 1/2 ⇒
  short-memory. Small β (the project's "strip") corresponds to **high H / strong long memory**.
- **Implication for multi-step forecastability**: long memory means past information remains predictive at
  *long* lead times, so ARFIMA can beat ARMA at **long horizons** (the gain is largest there, often
  negligible/negative at short horizons). But there is a ceiling: the *forecast-error variance grows with
  horizon* and the long-memory advantage is bounded by d. "On Long Memory Origins and Forecast Horizons"
  (Vera-Valdés, arXiv:1712.08057) analyzes how the optimal horizon for exploiting long memory depends on its
  origin. **Takeaway for the project:** the β that makes retrieval long-range (small β) is exactly the regime
  where long-horizon extrapolation is *theoretically possible but information-limited* — consistent with the
  observed flat, modest retention rather than a magic ridge.

---

## Multi-step strategies & error growth

The choice of *how* to produce a multi-step (long) prediction is the most transferable concept, because
"length generalization" is essentially "how does the multi-step error envelope grow past the trained span."

- **Recursive / iterated**: one one-step model fed its own outputs. Low variance, but **errors compound** —
  forecast quality degrades roughly monotonically and often *accelerates* with horizon once predictions drift
  off the training manifold. This is the canonical model for *why retention falls at 4× length* in an
  autoregressive (decoder-only / Mamba) model.
- **Direct (multi-horizon)**: a separate (or multi-output) head per horizon. **No accumulation**, but each
  head sees less data and ignores inter-step dependency; LTSF transformers (Informer→PatchTST) and
  N-BEATS/N-HiTS are all *direct* one-shot predictors — partly *why* they hold up at long horizons.
- **DirRec / hybrids** (Taieb & Hyndman): chain direct models to keep dependency awareness without full
  accumulation. Marcellino, Stock & Watson (2006) and Taieb et al. (NN5 review, 2012) give the canonical
  empirical comparisons.
- **Modern reappraisal** (Epistemic Error Decomposition…, 2025; arXiv:2511.11461): the textbook
  "recursive = high-bias/low-variance, direct = low-bias/high-variance" picture is too simple; recursive can
  match or beat direct, and the right decomposition is epistemic. Relevant when comparing the project's
  autoregressive (recursive-like) decoder against any direct read-out.

**Mapping to the project:** the project's decoder-only Transformer/Mamba forecasts next-token autoregressively
(recursive regime), so the 60% drop at 2048 is exactly recursive error growth diluted over 4× more positions.
A *direct / patched* read-out (predict block at offset, à la PatchTST/TimesFM) is the untried lever.

---

## Methods to borrow (mapped to length-gen, β/γ)

| TS method | Mechanism | What it buys for length-gen | Project mapping (β/γ, retention) |
|---|---|---|---|
| **Patching** (PatchTST, TimesFM, Moirai, MOMENT) | group adjacent steps into patch tokens | fewer positions ⇒ positional/attention extrapolation is gentler at test length | reduces the "4× more positions than trained" gap that crushes retention; test as a tokenization knob |
| **Random/dynamic patch masking & length curriculum** (TimesFM) | train over variable context lengths | model never overfits one length ⇒ better extrapolation | a *length-curriculum* (mix L during training) is a direct fix for train-short/test-long |
| **Series / multi-scale decomposition** (Autoformer, FEDformer, N-HiTS, Pyraformer, TimesNet) | split trend/seasonal/multi-rate, model per scale | the long-memory (small-β) channel is modeled separately from γ-noise | matches (β,γ) being two separable axes; decompose retrieval-distance scale from noise rate |
| **Relative/scale-invariant position handling** (Pyraformer O(1) path; iTransformer lookback robustness) | hop-count independent of L | long-range link reachable regardless of test length | analogue to RoPE/ALiBi length-extrapolation; explains Mamba's edge (no positional length dependence) |
| **Direct one-shot horizon head** (Informer, N-BEATS/N-HiTS, DLinear) | predict whole horizon at once | no recursive accumulation | untried in the project; would isolate "fitting" from "rollout drift" |
| **Linear baseline** (DLinear/NLinear) | one linear map | reveals how much gain is trend vs. true long-range | a DLinear-style control on the algorithmic task would test if retention is "free structure" |
| **Long-memory ceiling** (ARFIMA / Hurst H=d+½) | power-law ρ(k)∼k^(2d−1) | sets the *information limit* on long-horizon skill | small β ⇔ high H: long-horizon predictability exists but is bounded ⇒ predicts the *flat, non-magic* retention the project found |

---

## Relevance to this project

1. **β is a Hurst/long-memory exponent in disguise.** The project's p(d) ∝ d^{−(β+1)} retrieval-distance law
   is the discrete-token analogue of ARFIMA's power-law autocorrelation (H = d + 1/2). The classical theory
   *predicts* the project's empirical shape: small β (high H, long memory) makes long-horizon prediction
   *possible but information-limited* — so a flat, modest retention strip with **no sharp "edge-of-chaos"
   ridge** is exactly what long-memory theory expects, supporting the project's REFUTED H1.

2. **The most transferable single method is patching + a length curriculum (TimesFM-style dynamic masking).**
   It directly attacks the failure mode "test length ≫ trained length" by (a) shrinking the position count and
   (b) training over mixed lengths. This is a concrete, low-risk experiment the project has not run.

3. **Decomposition explains why (β,γ) are separable.** The forecasting field treats trend (long-memory) and
   seasonal/noise as separable channels; the project's clean separation of a β (correlation) axis and a γ
   (entropy/noise) axis is the same factorization, and decomposition architectures are the principled way to
   exploit it.

4. **Recursive error growth is the right model for the 60% drop at 4× length.** The decoder-only setup is a
   recursive multi-step forecaster; a **direct/patched read-out** is the natural ablation to separate
   "learned retrieval quality" from "autoregressive rollout drift," sharpening the retention diagnostic.

5. **DLinear is the missing control.** A linear/decomposition baseline on the algorithmic task would quantify
   how much of any retention is cheap structure vs. genuine long-range generalization — methodologically
   aligned with the project's cheat-guard philosophy (report absolute long-range accuracy, not just ratios).

6. **Foundation-model caveat backs the project's data-vs-model finding.** TS foundation models length-
   generalize well only on structured data — i.e. extrapolation is governed by data statistics, echoing
   "data shaping < model size < architecture (Mamba)."

---

## References

1. **Zhou et al., 2021** — *Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting.*
   AAAI 2021 (Best Paper); arXiv:2012.07436. ProbSparse attention + one-shot generative decoder for O(L log L)
   long-horizon forecasting; first to make ~10³ horizons practical and to avoid recursive rollout.
2. **Wu et al., 2021** — *Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series
   Forecasting.* NeurIPS 2021; arXiv:2106.13008. Series decomposition as an inner block + period-based
   Auto-Correlation; isolates trend/long-memory from seasonal.
3. **Liu et al., 2022** — *Pyraformer: Low-Complexity Pyramidal Attention for Long-Range Time Series Modeling
   and Forecasting.* ICLR 2022 (Oral); OpenReview 0EXmFzUn5I. Tree of scales giving O(1) signal-path length,
   O(L) compute — a multi-resolution long-range inductive bias.
4. **Zhou et al., 2022** — *FEDformer: Frequency Enhanced Decomposed Transformer for Long-Term Series
   Forecasting.* ICML 2022; arXiv:2201.12740. Sparse-Fourier attention (linear complexity) + decomposition;
   assumes long-range structure is spectrally sparse.
5. **Zeng et al., 2023** — *Are Transformers Effective for Time Series Forecasting?* AAAI 2023;
   arXiv:2205.13504. DLinear/NLinear linear baselines beat transformer LTSF models — much "long-horizon skill"
   is trend extrapolation, not attention. Field's key cautionary result.
6. **Nie et al., 2023** — *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers (PatchTST).*
   ICLR 2023; arXiv:2211.14027. Patching + channel independence; ~21% MSE gain; patching is the most
   transferable length-robustness trick.
7. **Liu et al., 2024** — *iTransformer: Inverted Transformers Are Effective for Time Series Forecasting.*
   ICLR 2024 (Spotlight); arXiv:2310.06625. Variate-as-token inversion; improves use of arbitrarily long
   lookback windows (a length-robustness result).
8. **Wu et al., 2023** — *TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis.*
   ICLR 2023; arXiv:2210.02186. Period-folded 1D→2D tensors; multi-periodicity as explicit multiscale bias.
9. **Oreshkin et al., 2020** — *N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series
   Forecasting.* ICLR 2020; arXiv:1905.10437. Pure-MLP doubly-residual basis-expansion stack; direct
   one-shot forecaster — long-horizon skill without attention/recurrence.
10. **Challu et al., 2023** — *N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting.*
    AAAI 2023; arXiv:2201.12886. Multi-rate sampling + hierarchical interpolation; decomposes the horizon by
    temporal scale; ~25% better than N-BEATS at long horizons.
11. **Garza et al., 2023** — *TimeGPT-1.* Nixtla; arXiv:2310.03589. First TS foundation model (encoder-decoder,
    >100B points); zero-shot across granularities and variable horizons.
12. **Das et al., 2024** — *A Decoder-Only Foundation Model for Time-Series Forecasting (TimesFM).* ICML 2024;
    arXiv:2310.10688. Patched decoder-only; **dynamic patch masking** for context-length generalization;
    autoregressive rollout extrapolates beyond any single training horizon. Closest analogue to the project.
13. **Ansari et al., 2024** — *Chronos: Learning the Language of Time Series.* arXiv:2403.07815. Scale+quantize
    tokenization → T5 + cross-entropy; synthetic GP augmentation; forecasting-as-language-modeling.
14. **Woo et al., 2024** — *Unified Training of Universal Time Series Forecasting Transformers (Moirai).*
    ICML 2024; arXiv:2402.02592. Masked-encoder universal forecaster; any-variate attention + multiple
    patch-size projections for cross-frequency/horizon adaptation; LOTSA (27B+ obs).
15. **Goswami et al., 2024** — *MOMENT: A Family of Open Time-Series Foundation Models.* ICML 2024;
    arXiv:2402.03885. Masked patch reconstruction (BERT-style) on the Time-Series Pile; general-purpose TS
    encoder.
16. **Rasul et al., 2023** — *Lag-Llama: Towards Foundation Models for Probabilistic Time Series Forecasting.*
    arXiv:2310.08278. Decoder-only LLaMA-style model with **lag-feature tokens** — multiscale temporal
    structure built into the input representation.
17. **Liu et al., 2024** — *Timer-XL: Long-Context Transformers for Unified Time Series Forecasting.*
    arXiv:2410.04803 (see also *Towards Long-Context TSFMs*, arXiv:2409.13530). Explicitly extends usable
    context length and studies forecast scaling with context — the most on-topic "length generalization for TS."
18. **Granger & Joyeux, 1980; Hosking, 1981** — *Fractional Differencing / ARFIMA.* J. Time Series Analysis /
    Biometrika. Power-law autocorrelation ρ(k)∼k^(2d−1); long memory for d>0; H = d + 1/2. The classical
    parametrization of the project's β knob.
19. **Vera-Valdés, 2017/2021** — *On Long Memory Origins and Forecast Horizons.* arXiv:1712.08057
    (Intl. J. Forecasting). How the optimal forecast horizon for exploiting long memory depends on its origin;
    information-limited long-horizon predictability under long-range dependence.
20. **Marcellino, Stock & Watson, 2006** — *A Comparison of Direct and Iterated Multistep AR Methods for
    Forecasting Macroeconomic Time Series.* J. Econometrics. Canonical direct-vs-recursive empirical study.
21. **Taieb et al., 2012** — *A Review and Comparison of Strategies for Multi-Step-Ahead Time Series
    Forecasting (NN5 competition).* Expert Systems with Applications. Recursive / direct / DirRec / MIMO
    taxonomy and comparison.
22. **(Epistemic Error Decomposition), 2025** — *Epistemic Error Decomposition for Multi-Step Time Series
    Forecasting: Rethinking Bias–Variance in Recursive and Direct Strategies.* arXiv:2511.11461. Modern
    reappraisal: the textbook recursive/direct bias-variance story is incomplete; recursive can match/beat
    direct.
23. **(Survey), 2025** — *How Foundational are Foundation Models for Time Series Forecasting?* arXiv:2510.00742.
    Zero-shot length/horizon generalization is good on structured/periodic data and weak on irregular
    real-world data — extrapolation is governed by data statistics (supports the project's data-vs-model
    finding).
