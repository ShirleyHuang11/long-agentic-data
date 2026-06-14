# Long Memory, Multifractality, and Rough/Fractional Models in Econophysics & Quantitative Finance — A Method-Focused Literature Review

*Web search was available (June 2026); references below are verified against publisher / arXiv / SSRN pages. Author/year/venue checked from search results.*

## Overview

This review imports **measurement methods** from econophysics and quantitative finance into the (β, γ) phase-diagram project (`holographic-data/case/phase`), where:

- **β** = correlation-decay sharpness (retrieval index power law `p(d) ∝ d^{-(β+1)}`),
- **γ** = entropy/noise-decay rate,
- **α_D = γ/2β** = a derived "theory" exponent,
- **Hurst H** is treated as a *near-independent* axis,
- and **Rényi / multifractal diagnostics** (`D_q`) already separate the emergent strip from chaos (Result 8: D_q=1 emergent 0.625 vs chaos 0.659, ~3.4σ).

Finance has spent 30+ years measuring exactly these objects — long-range dependence (LRD), multifractal spectra, cross-correlation scaling, and roughness — on heavy-tailed correlated series, and (crucially) cataloguing the **biases and spurious-LRD traps** in those estimators. The central transferable lessons are: (1) **returns are ~uncorrelated but |returns| have long memory** — the signal lives in a *nonlinear* transform of the series, exactly the absolute/squared analogue of our entropy-side γ axis; (2) the **width of the multifractal singularity spectrum Δα** is a single scalar that summarizes "effective complexity / heterogeneity of scaling" — a natural candidate for the project's "Maximum Effective Complexity" target at the edge of chaos; (3) **apparent long memory is routinely manufactured by structural breaks or by genuinely-rough short-memory processes** — a non-negotiable caution before claiming any β/H estimate is "real."

## Long memory & stylized facts

The canonical catalogue is **Cont (2001)**: heavy tails, gain/loss asymmetry, aggregational Gaussianity, intermittency, **volatility clustering**, **slow decay of autocorrelation in |returns|**, and the "leverage" effect. The defining fact for us: linear autocorrelation of returns decays to noise within minutes, yet the ACF of |r_t| and r_t² decays **hyperbolically** (∼ k^{-α}, α < 1) over weeks — "absence of linear autocorrelation but long-range dependence in volatility." **Cont (2005)** is the dedicated method survey of LRD: it distinguishes *self-similarity* (a sample-path/distributional property) from *long memory* (a second-order/ACF property), warns they are conflated, and reviews R/S, DFA, periodogram/GPH, and wavelet estimators. The empirical regularity that returns are unpredictable while their magnitudes are strongly persistent is the finance mirror of the project's separation between the **β (correlation) axis** and the **γ (entropy) axis**: the second-order structure of the raw signal and of its magnitude/entropy live on different axes.

Mechanistically, **Granger–Ding** style results and the multiscaling literature (e.g. multiscaling of generalized cumulative absolute returns) show |returns| exhibit *multi*scaling, not mono-fractal scaling — motivating the multifractal models below.

## Multifractal models

**Mandelbrot, Fisher & Calvet (1997) — MMAR (Multifractal Model of Asset Returns).** Log-price = Brownian motion *subordinated* (time-changed) by a multifractal trading-time measure built from a multiplicative cascade. Delivers multifractality, scale-consistency, long memory in volatility, and heavy tails **without** infinite variance. Key idea for us: a single *time-deformation* with a multiplicative cascade simultaneously generates LRD + heavy tails + multiscaling — i.e. one generative knob produces correlated structure across many scales.

**Bacry, Delour & Muzy (2001) — Multifractal Random Walk (MRW).** The first stationary-increment, continuous-scale multifractal process (no preferred scale ratio, unlike grid cascades). It is a stochastic-volatility model with a **log-correlated** volatility: log-vol covariance ∼ λ² log(T/τ). Parameterized by an **intermittency coefficient λ²** that *directly* controls both the multifractal-spectrum width and the volatility-correlation strength. This is the cleanest "knob → spectrum width" mapping in the field and the most direct analogue of a tunable β/γ generator.

**Calvet & Fisher (2001, 2004) — Markov-Switching Multifractal (MSM).** Volatility = product of k̄ multiplicative components with **heterogeneous Markovian durations** (geometric time scales). Parsimonious (states grow but parameters don't), estimable by ML/GMM, and forecasts realized vol better than GARCH/FIGARCH/MS-GARCH at 10–50 day horizons. Important caveat (see below): **MSM only *approximates* LRD** — its ACF mimics power-law decay over a finite range while being short-memory asymptotically.

## Cross-correlation methods (DCCA / MF-DCCA)

**Podobnik & Stanley (2008) — Detrended Cross-Correlation Analysis (DCCA).** Generalizes DFA to *two* non-stationary series: integrate both, split into windows, locally detrend (polynomial fit), compute the **detrended covariance** per window, and look for power-law scaling F²_{xy}(s) ∼ s^{2λ}. The cross-correlation exponent λ measures long-range *coupling*. **Zebende (2011)** added the DCCA *coefficient* ρ_DCCA(s) ∈ [−1,1] — a scale-resolved, detrending-robust alternative to Pearson correlation.

**Zhou (2008) — MF-DCCA (a.k.a. MF-X-DFA).** Merges MF-DFA + DCCA: compute the q-dependent fluctuation `F_q(s) = {(1/N)Σ[F²_{xy}(s)]^{q/2}}^{1/q}` and extract a **q-dependent cross-correlation exponent h_xy(q)**. q-dependence ⇒ *multifractal* cross-correlation; q-independence ⇒ monofractal coupling. **Detrended partial-cross-correlation analysis (DPCCA / DPXA)** removes common-driver confounds — the cross-correlation analogue of partial correlation. **Local/rolling DCCA** (Kristoufek; Wang et al.) and detrended moving-average cross-correlation (DMCA) give time-resolved coupling.

For the project these are the tools to measure **cross-axis dependence** directly: treat the raw token-statistic series and the magnitude/entropy series (or two diagnostics like the β-side ACF and the γ-side entropy) as the (x, y) pair and ask whether they are coupled with a power-law scale law, and whether that coupling is mono- or multifractal.

## Rough/fractional models

**Gatheral, Jaisson & Rosenbaum (2018, "Volatility is rough", QF 18(6); arXiv:1410.3394).** From high-frequency realized vol, **log-volatility behaves like fractional Brownian motion with H ≈ 0.1** (much rougher than the H=1/2 semimartingale assumption). Their **RFSV** model fits data and forecasts better than long-memory models. The estimator is itself a borrowable method: regress `m(q,Δ) = E|log σ_{t+Δ} − log σ_t|^q` against log Δ; the slope gives qH, and **linearity in q of the scaling slope tests mono- vs multi-fractality of the increments**. The Hurst exponent here is estimated from the *q-th moment of increments of a derived (volatility/entropy) process*, not the raw signal — a template for putting H on its own axis as the project does.

**Bayer, Friz & Gatheral (2016) — rough Bergomi (rBergomi).** A 3-parameter non-Markovian SV model driven by fractional kernels; reproduces the implied-vol smile, esp. short maturities. Relevant as the generative side: rough kernels (Riemann–Liouville fBm) are a minimal way to inject tunable roughness/correlation-decay into a synthetic generator.

**⚠️ The rough-volatility / spurious-LRD link (critical):** GJR explicitly show that **classical long-memory detectors (R/S, GPH) wrongly report long memory on data simulated from their *rough, short-memory* RFSV model.** Roughness (small H) and long memory are *different* properties that standard estimators conflate.

## Estimation debates (spurious LRD vs breaks)

This section is the most important import — it is the field's accumulated immune system against false positives.

- **Lo (1991, Econometrica) — modified R/S.** Classical R/S over-detects LRD because short-range dependence inflates the range. Lo normalizes by a HAC (Newey–West-style) standard deviation S_q that absorbs the first q autocovariances. Result: once short memory is accounted for, **daily/monthly US stock returns show *no* robust long memory.** Caveat (Teverovsky–Taqqu–Mandelbrot 1999, "A critical look at Lo's modified R/S"): the test is sensitive to the bandwidth q and can over-*reject* true LRD.
- **Granger & Hyung (2004, J. Empirical Finance 11) — occasional structural breaks.** A series with **occasional level/variance breaks generates slowly-decaying ACF and an apparent fractional integration order d** indistinguishable from genuine I(d). On S&P 500 absolute returns the break model fits as well as the I(d) model — and (their second perplexity) a true long-memory series can spuriously trigger break detection. Diebold–Inoue and Mikosch–Stărică make the same point.
- **Calvet–Fisher / Liu–Di Matteo–Lux (arXiv:0704.1338) — "true vs apparent scaling".** MSM (a short-memory regime model) sits arbitrarily close to LRD over any finite sample; "the proximity of MSM to long-range dependence" means scaling-exponent estimates cannot by themselves certify asymptotic long memory.
- **General estimator-bias literature:** R/S, DFA, and GPH are all biased under trends, heavy tails, short memory, and breaks. Robust practice: use **multiple estimators (DFA + GPH + wavelet + modified R/S)**, surrogate/shuffled-data nulls, and finite-size/confidence bands; report the *range* over which scaling holds rather than a point H.

## Methods to borrow (mapped to β / γ / Hurst)

| Method | What it measures | Maps to (β, γ, H) in this project |
|---|---|---|
| **MFDFA** (Kantelhardt et al. 2002, Physica A 316) | q-dependent Hurst h(q) → mass exponent τ(q) → singularity spectrum f(α); **width Δα = α_max − α_min** | Δα is a single-scalar "effective complexity / multiscaling heterogeneity" diagnostic. Compute on the token / entropy series per (β,γ) cell; **hypothesis: Δα is maximized at the edge-of-chaos strip** (max effective complexity). Complements existing D_q (Rényi) by adding the *spread* of local scaling, not just one Rényi dimension. |
| **Singularity-spectrum width Δα** (from MFDFA) | Degree of multifractality (monofractal ⇒ Δα→0) | **Most transferable scalar.** Directly operationalizes "Maximum Effective Complexity." Track Δα across the (β,γ) plane as a 3rd diagnostic alongside train_acc and length-gen ratio r. |
| **DCCA exponent λ + ρ_DCCA(s)** (Podobnik–Stanley; Zebende) | Scale-resolved long-range coupling between two series | Quantify **cross-axis dependence** (β-side vs γ-side signals; or signal vs |signal|) at each scale; ρ_DCCA(s) replaces Pearson where series are non-stationary. |
| **MF-DCCA exponent h_xy(q)** (Zhou 2008) | Whether cross-correlation is mono- or multi-fractal | Tests if the coupling between two diagnostics is itself scale-heterogeneous; q-spread = cross-multifractality. |
| **Rough-vol q-moment estimator** (GJR 2018) | H of a *derived* (vol/entropy) process via `E|Δ_τ X|^q ∝ τ^{qH}`; q-linearity = monofractal | Estimator template for placing **H on its own axis**: estimate H from increments of the entropy/magnitude process, not the raw token stream — exactly the "near-independent Hurst axis" stance. |
| **Modified R/S + multi-estimator panel** (Lo 1991; Cont 2005) | Robust LRD test discounting short memory | Use as the **guard** before claiming any measured β/H is "real" long memory rather than short-range structure. |
| **Break-vs-LRD model comparison** (Granger–Hyung 2004) | Whether slow ACF decay is from breaks not LRD | Run shuffled/surrogate and break-injected nulls per cell; report Δα and H *with* a spurious-LRD caveat. |
| **MRW intermittency λ²** (Bacry–Delour–Muzy 2001) | One knob → spectrum width + vol-correlation | Reference *generator*: a principled way to synthesize data with tunable, jointly-controlled correlation strength and multifractal width — a clean baseline/ablation against the KV generator. |

## Relevance to this project

1. **Multifractal-spectrum width Δα is the headline import.** The project already uses Rényi `D_q` (a single dimension); MFDFA gives the *entire* f(α) curve and its width Δα, which is precisely a "heterogeneity of scaling / effective complexity" scalar. The edge-of-chaos / max-effective-complexity hypothesis becomes a *falsifiable measurement*: **does Δα(β,γ) peak on the emergent strip?** This slots directly into the existing data-side diagnostic story (Result 8) and the falsifiability scoreboard.

2. **The returns-vs-|returns| dichotomy validates the β/γ separation.** Finance's central stylized fact — linear structure dead, magnitude structure long-memory — is the empirical precedent for treating correlation-decay (β) and entropy/magnitude-persistence (γ) as different axes. The same diagnostics should be run on both the raw token series and a magnitude/entropy transform.

3. **The spurious-LRD caution is mandatory.** GJR (rough but spurious-LRD), Lo (short memory faking LRD), Granger–Hyung (breaks faking LRD), and MSM (finite-sample proximity to LRD) jointly imply: **a single Hurst/β estimate is not evidence of long memory.** Before any "α_D = γ/2β" claim is read as a real long-range exponent, run a multi-estimator panel + surrogate/break nulls, and report the scaling range. This protects the paper from the most common econophysics referee objection.

4. **DCCA/MF-DCCA give a principled cross-axis-dependence test.** The project asserts Hurst is "near-independent" of (β,γ). DCCA's ρ_DCCA(s) and MF-DCCA's h_xy(q) let that independence be *measured* at each scale rather than asserted.

## References

1. **Cont, R. (2001).** *Empirical properties of asset returns: stylized facts and statistical issues.* Quantitative Finance 1(2), 223–236. — Canonical stylized-facts catalogue; volatility clustering and slow ACF decay of |returns| vs uncorrelated returns. https://www.stat.rice.edu/~dobelman/courses/texts/stylized.cont.2001.pdf
2. **Cont, R. (2005).** *Long range dependence in financial markets.* In *Fractals in Engineering*, Springer. — Method survey separating self-similarity from long memory; reviews R/S, DFA, GPH, wavelet estimators. http://rama.cont.perso.math.cnrs.fr/pdf/FE05.pdf
3. **Mandelbrot, B., Fisher, A. & Calvet, L. (1997).** *A Multifractal Model of Asset Returns.* Cowles Foundation DP 1164. — MMAR: Brownian motion subordinated to a multifractal trading time; multifractality + long memory + heavy tails without infinite variance. https://users.math.yale.edu/~bbm3/web_pdfs/Cowles1164.pdf
4. **Bacry, E., Delour, J. & Muzy, J.-F. (2001).** *Multifractal random walk.* Physical Review E 64, 026103 (arXiv:cond-mat/0005405). — Stationary-increment, continuous-scale multifractal; intermittency λ² controls spectrum width + log-correlated volatility. https://arxiv.org/abs/cond-mat/0005405
5. **Pochart, B. & Bouchaud, J.-P. / Bacry et al. (2002).** *The skewed multifractal random walk with applications to option smiles.* arXiv:cond-mat/0204047. — Asymmetric MRW capturing return–volatility (leverage) correlation. https://arxiv.org/abs/cond-mat/0204047
6. **Calvet, L. & Fisher, A. (2001).** *Forecasting multifractal volatility.* Journal of Econometrics 105(1), 27–58. — Foundations of the discretized multifractal / MSM approach.
7. **Calvet, L. & Fisher, A. (2004).** *How to forecast long-run volatility: regime switching and the estimation of multifractal processes.* Journal of Financial Econometrics 2(1), 49–83. — MSM: heterogeneous-duration multiplicative volatility; ML/GMM estimation; beats GARCH/FIGARCH at 10–50 day horizons.
8. **Liu, R., Di Matteo, T. & Lux, T. (2007).** *True and apparent scaling: the proximity of the Markov-switching multifractal model to long-range dependence.* arXiv:0704.1338. — Short-memory MSM mimics LRD in finite samples; scaling estimates can't certify asymptotic long memory. https://arxiv.org/pdf/0704.1338
9. **Kantelhardt, J. et al. (2002).** *Multifractal detrended fluctuation analysis of nonstationary time series.* Physica A 316(1–4), 87–114. — MFDFA: q-order h(q) → τ(q) → f(α) singularity spectrum; the workhorse for spectrum width Δα.
10. **Ihlen, E. (2012).** *Introduction to multifractal detrended fluctuation analysis in Matlab.* Frontiers in Physiology 3, 141. — Practical MFDFA recipe; explicit Hq→τq→hq→Dq and width computation. https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2012.00141/full
11. **Podobnik, B. & Stanley, H. E. (2008).** *Detrended cross-correlation analysis: a new method for analyzing two nonstationary time series.* Physical Review Letters 100, 084102. — DCCA: detrended-covariance power-law scaling between two series.
12. **Zebende, G. F. (2011).** *DCCA cross-correlation coefficient: quantifying level of cross-correlation.* Physica A 390(4), 614–618. — ρ_DCCA(s) ∈ [−1,1], scale-resolved, detrending-robust correlation coefficient.
13. **Zhou, W.-X. (2008).** *Multifractal detrended cross-correlation analysis for two nonstationary signals.* Physical Review E 77, 066211. — MF-DCCA: q-dependent cross-correlation exponent h_xy(q); mono- vs multifractal coupling. (see review arXiv:1805.04750)
14. **Jiang, Z.-Q., Xie, W.-J., Zhou, W.-X. & Sornette, D. (2019).** *Multifractal analysis of financial markets: a review.* Reports on Progress in Physics 82, 125901 (arXiv:1805.04750). — Comprehensive review of MFDFA, MF-DCCA, partial/DPXA variants, spurious-multifractality cautions. https://arxiv.org/pdf/1805.04750
15. **Gatheral, J., Jaisson, T. & Rosenbaum, M. (2018).** *Volatility is rough.* Quantitative Finance 18(6), 933–949 (arXiv:1410.3394). — Log-vol ≈ fBm with H≈0.1; q-moment-of-increments H estimator; **classical LRD tests spuriously report long memory on rough short-memory data.** https://arxiv.org/abs/1410.3394
16. **Bayer, C., Friz, P. & Gatheral, J. (2016).** *Pricing under rough volatility.* Quantitative Finance 16(6), 887–904. — rough Bergomi (rBergomi): 3-parameter fractional-kernel SV model; minimal tunable-roughness generator. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2554754
17. **Lo, A. W. (1991).** *Long-term memory in stock market prices.* Econometrica 59(5), 1279–1313. — Modified R/S with HAC normalization; no robust LRD in US stock returns once short memory is discounted. https://ideas.repec.org/a/ecm/emetrp/v59y1991i5p1279-313.html
18. **Teverovsky, V., Taqqu, M. & Mandelbrot, B. (1999).** *A critical look at Lo's modified R/S statistic.* Journal of Statistical Planning and Inference 80(1–2), 211–227. — Lo's test is bandwidth-sensitive and can over-reject *true* LRD. https://www.researchgate.net/publication/222452893
19. **Granger, C. W. J. & Hyung, N. (2004).** *Occasional structural breaks and long memory with an application to the S&P 500 absolute stock returns.* Journal of Empirical Finance 11(3), 399–421. — Breaks manufacture apparent fractional integration; break model fits S&P |returns| as well as I(d). https://www.sciencedirect.com/science/article/abs/pii/S0927539804000131
20. **Di Matteo, T. (2007).** *Multi-scaling in finance.* Quantitative Finance 7(1), 21–36. — Generalized-Hurst-exponent (GHE) multiscaling estimator H(q); practical and widely used for spectrum estimation. (see arXiv:1509.05471 for application)
21. **Buonocore, R., Aste, T. & Di Matteo, T. (2016).** *Measuring multiscaling in financial time-series.* Chaos, Solitons & Fractals (arXiv:1509.05471). — Robustly estimating the q-dependence of scaling and testing genuine vs apparent multiscaling. https://arxiv.org/pdf/1509.05471
22. **Lux, T. (2006/2008).** *Financial power laws: empirical evidence, models, and mechanisms.* Kiel Working Paper. — Survey of power-law tails and LRD in finance; mechanisms and estimation pitfalls. https://warwick.ac.uk/fac/soc/wbs/subjects/finance/research/wpaperseries/wf06-255.pdf
