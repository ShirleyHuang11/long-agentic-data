# Methods Toolbox: Measuring Fractal Structure, Long-Range Dependence, and Entropy/Complexity in Sequences

*A cross-disciplinary estimator catalog (time-series analysis, signal processing, econophysics, nonlinear dynamics) for an ML data-theory project.*

Web search **was available**; citations below were checked against primary sources / standard references. Where a specific numerical relation is stated, it is the standard textbook form.

---

## Overview

This project parameterizes sequence data by three exponents and treats them as (near-)independent axes:

- **β** — token-correlation power-law decay exponent. The autocorrelation / mutual-information of the symbol stream decays as a power law, and β controls its sharpness. In `phase_core.AlgorithmicKVGenerator`, retrieval distance is drawn from `p(d) ∝ d^{-(β+1)}`, so the *induced* long-range structure has a tunable decay exponent.
- **γ** — conditional-entropy power-law decay exponent: `Hₙ ∼ n^{−γ}`, where `Hₙ` is the per-symbol conditional entropy (entropy rate estimate at block length n). In the codebase γ is also overloaded as a noise-token rate; this document treats γ in its measurement sense (the exponent of conditional-entropy convergence).
- **α_D = γ / 2β** — a derived "data dimension"-like ratio.
- **Hurst H** — a near-independent long-memory axis.

The codebase already implements two measurements: **Rényi/generalized dimension D_q** of n-gram statistics (`estimate_renyi_dimensions_from_tokens`: fits `H_q(m) ≈ a_q·m + b_q`, defines `D_q = a_q / log(vocab)`), and **Hurst via rescaled-range (R/S)** elsewhere. The goal here is a rigorous menu of *all* the standard estimators, with formulas, assumptions, pitfalls, and an explicit mapping to β, γ, H.

A unifying fact worth stating up front: for self-similar / long-memory processes, **the Hurst exponent H, the power-spectral exponent (1/f^a), the DFA exponent α, and the autocorrelation-decay exponent are all linked by simple algebra** (see "Exponent relationships"). So "β / γ / H" are not three unrelated knobs — they are partly redundant views of the same scaling, which is exactly why cross-checking estimators is valuable.

---

## Long-memory / Hurst estimators

A stationary process has **long memory** (long-range dependence, LRD) if its autocovariance is non-summable: `ρ(k) ∼ c·k^{-(2-2H)}` as `k→∞` with `1/2 < H < 1`, equivalently a spectral pole at the origin `f(λ) ∼ |λ|^{-2d}` with `d = H − 1/2`. The estimators below all target H (or the fractional-integration parameter `d`).

### Rescaled-range (R/S) analysis — Hurst (1951), Mandelkar/Mandelbrot–Wallis
For windows of size n, compute the range of the cumulative-deviation series divided by the standard deviation, `R(n)/S(n)`. Then `E[R(n)/S(n)] ∼ c·n^H`; H is the slope of `log(R/S)` vs `log n`.
- **Assumptions:** weak stationarity; finite variance.
- **Pitfalls:** strongly biased in finite samples and **sensitive to short-range autocorrelation and trends** (the classic over-estimate of H near 0.5→0.7). Lo's (1991) modified R/S adds a Newey–West-style correction for short memory. Generally **least accurate** of the modern methods; keep it only as a fast sanity check (this is what the codebase currently uses).

### Detrended Fluctuation Analysis (DFA) — Peng et al. (1994)
1. Integrate the (mean-subtracted) series into a profile `Y(i)=Σ(xₖ − x̄)`.
2. Split into windows of size s; in each window fit and subtract a polynomial trend (order ℓ ⇒ "DFA-ℓ"; DFA-1 = linear).
3. RMS of residuals = fluctuation function `F(s)`.
4. `F(s) ∼ s^α`; α is the slope of `log F(s)` vs `log s`.
- **Mapping:** for a stationary LRD series α = H (and `α = d + 1/2`). α=0.5 ⇒ white noise; 0.5<α<1 ⇒ persistent LRD; α>1 ⇒ nonstationary (fBm-like). The polynomial detrending makes DFA **robust to slow trends**, its main advantage over R/S.
- **Pitfalls:** spurious **crossovers** from periodic/sinusoidal trends; finite-size curvature at small s; DFA-ℓ removes only polynomial trends of order ≤ℓ; unreliable for short series. (Bryce & Sprague 2012, "Revisiting DFA"; Hu et al. 2001, "Effect of Trends on DFA".)

### Detrending Moving Average (DMA) — Alessio, Carbone et al. (2002); Vandewalle–Ausloos (1998)
Like DFA but detrend with a moving average instead of a per-window polynomial: `F²(n) = (1/N)Σ[Y(i) − Ȳ_n(i)]²`, where `Ȳ_n` is an n-point moving average; `F(n) ∼ n^H`.
- **Trade-off vs DFA:** comparable accuracy; centered-window DMA performs similarly to DFA-1, slightly different bias profile. No per-window fits ⇒ cheaper. Generalizes to MFDMA (multifractal).
- **Pitfalls:** choice of window position (backward/centered/forward) shifts the estimate; same crossover issues.

### Wavelet-based estimators (Abry–Veitch 1998)
Use the discrete wavelet transform: the variance of detail coefficients at scale `j` scales as `Var(dⱼ) ∼ 2^{j(2H−1)}` (for fGn) so a weighted log-regression of `log₂(mean dⱼ²)` vs `j` gives H. The "logscale diagram."
- **Strengths:** wavelets *whiten* LRD and kill polynomial trends up to the number of vanishing moments ⇒ **statistically efficient, robust to trends**, near-unbiased, with tractable confidence intervals. Strong default for clean H estimation.
- **Pitfalls:** boundary effects; need enough octaves; pick vanishing moments ≥ polynomial trend order.

### Periodogram / GPH log-periodogram regression — Geweke & Porter-Hudak (1983)
Semiparametric, frequency-domain. Regress the log-periodogram on log-frequency at the lowest K Fourier frequencies:
`log I(λₖ) = a − d·log{4 sin²(λₖ/2)} + εₖ`; the OLS slope estimates `d`, and `H = d + 1/2`.
- **Bandwidth:** `K ≈ N^{0.5}` (GPH default; `N^{0.45}`–`N^{0.5}` common). Only this family has clean **known asymptotic distribution** for confidence intervals.
- **Pitfalls:** bias if short-range dynamics leak into the low-frequency band; bandwidth choice is the key knob (bias–variance trade-off).

### Local Whittle / Whittle MLE — Künsch (1987), Robinson (1995)
Maximize a local Whittle (quasi-)likelihood using only the first `m ≪ N` periodogram ordinates near the origin. More **efficient than GPH** (likelihood-based) with the same semiparametric robustness. Consistent and asymptotically normal for `d ∈ (−1/2, 1/2)`; extensions (Velasco; exact/2-step LW) reach `d` up to ~3/4 and the nonstationary range.
- **Pitfalls:** sensitive to deterministic trends/level shifts (use detrended or "exact" LW variants); `m` selection matters.

### ARFIMA / fractionally-integrated parametric models — Granger–Joyeux (1980), Hosking (1981)
ARFIMA(p,d,q): `(1−L)^d` fractional differencing with `d ∈ (−0.5, 0.5)` gives long memory; `H = d + 1/2`. Fit by exact/Whittle MLE.
- **Strengths:** full generative model; separates short-range (p,q) from long-range (d) ⇒ removes the short-memory contamination that biases R/S.
- **Pitfalls:** model-order misspecification; assumes the *parametric* form; heavier to fit. Good when you want a model, not just an exponent.

---

## Multifractal methods

When a single H is insufficient (scaling depends on the *moment order* q), one needs a **spectrum** of exponents. Output is the generalized Hurst exponent `h(q)`, the mass exponent `τ(q)`, and the singularity spectrum `f(α)` (Legendre transform). Monofractal ⇔ `h(q)` constant; multifractal ⇔ `h(q)` decreasing in q.

### Multifractal DFA (MFDFA) — Kantelhardt et al. (2002)
Generalize DFA step 3 to a q-th order fluctuation function:
`F_q(s) = { (1/N_s) Σ [F²(ν,s)]^{q/2} }^{1/q} ∼ s^{h(q)}` (with a log-average for q=0).
- `h(2)` = classical Hurst H. `τ(q) = q·h(q) − 1`; `f(α)` via Legendre transform `α = dτ/dq`, `f(α) = qα − τ(q)`.
- **Most practical multifractal tool** for finite, noisy, nonstationary data. Default choice.
- **Pitfalls:** finite-size and LRD can create **spurious (apparent) multifractality** in truly monofractal signals; restrict q range (e.g. |q|≤5) and validate against shuffled/surrogate data; small-s and large-s windows unreliable.

### Wavelet Transform Modulus Maxima (WTMM) — Muzy, Bacry, Arneodo (1991–1995)
Track the *maxima lines* of the continuous wavelet transform across scales; build a partition function `Z(q,a) = Σ |WT|^q ∼ a^{τ(q)}` over maxima lines; Legendre-transform to `f(α)`.
- **Strengths:** wavelet vanishing moments remove polynomial trends; resolves local Hölder/singularity exponents; historically the gold standard for turbulence/DNA.
- **Pitfalls:** more complex to implement and tune than MFDFA; maxima-chaining is finicky; MFDFA usually preferred now for robustness/simplicity.

### Structure-function (multiscaling) method — Frisch–Parisi multifractal formalism
Compute moments of increments: `S_q(r) = ⟨|x(t+r) − x(t)|^q⟩ ∼ r^{ζ(q)}`. `ζ(q)/q` plays the role of `h(q)`; nonlinear concave `ζ(q)` ⇒ multifractality (intermittency). Classical in turbulence (Kolmogorov `ζ(q)=q/3` is the monofractal baseline).
- **Pitfalls:** works for *stationary-increment* (fBm-like) data; positive-q biased by extremes; negative-q ill-defined for increments (use WTMM/MFDFA there).

---

## Fractal dimension

### Box-counting dimension D₀
Cover the support with boxes of size ε; `N(ε) ∼ ε^{−D₀}`; `D₀ = lim log N(ε)/log(1/ε)`. The q=0 member of the Rényi family.
- **Pitfalls:** needs a wide scaling range; biased at small/large ε; for *sequences* one works on a reconstructed phase-space embedding, not raw symbols.

### Correlation dimension D₂ — Grassberger & Procaccia (1983)
Embed the series (Takens delay embedding, dimension m), compute the correlation sum `C(ε) = (2 / N(N−1)) Σ_{i<j} Θ(ε − ‖xᵢ − xⱼ‖)`; `C(ε) ∼ ε^{D₂}`; `D₂` = slope of `log C` vs `log ε` in the scaling region.
- **Strengths:** cheaper/more robust than box-counting; the standard chaos/attractor-dimension estimator.
- **Pitfalls:** embedding parameters (m, delay τ); needs long, low-noise series; spurious low D from autocorrelation/oversampling (Theiler correction); choose scaling plateau carefully.

### Rényi / generalized dimensions D_q
`D_q = (1/(q−1)) · lim_{ε→0} [ log Σᵢ pᵢ(ε)^q / log ε ]`, with the q=1 (information-dimension) limit `D₁ = lim Σ pᵢ log pᵢ / log ε`.
- D₀ = box-counting, D₁ = information dimension, D₂ = correlation dimension. `D_q` non-increasing in q; constant ⇔ monofractal, spread ⇔ multifractal. Related to `τ(q) = (q−1)D_q`.
- **This is what the codebase already computes** on symbolic n-gram statistics: rather than a geometric ε→0 limit, it uses **n-gram block size m as the "resolution"** and fits Rényi block entropy `H_q(m)` linearly in m, normalizing the slope by `log(vocab)` to get `D_q ∈ [0,1]`. This is a sound symbolic analogue (it is essentially an entropy-*rate* per symbol; see caveat under "Recommended methods").

---

## Correlation- & MI-decay estimation (the β axis)

For **symbolic / token sequences** (no natural metric), linear autocorrelation is the wrong tool; use information-theoretic two-point functions.

### Symbolic autocorrelation
Map symbols to indicators and compute `C(k) = ⟨1[xₜ=xₜ₊ₖ]⟩ − baseline`, or per-symbol-pair correlations. Fit `C(k) ∼ k^{−β}` on a log-log plot. Cheap, but loses higher-order structure and is dominated by the most frequent symbols.

### Mutual-information decay — Lin & Tegmark (2017), "Criticality in Formal Languages"
Estimate `I(Xₜ ; Xₜ₊ₖ)` (two-point mutual information between symbols separated by k) and fit its decay vs k.
- **Key theoretical fact (directly relevant to this project):** **Markov / hidden-Markov / regular-grammar sources give exponential `I(k)` decay; power-law `I(k) ∼ k^{−β}` ("critical") requires deeper / context-free / recursive generative structure.** Natural language, music, and genomes empirically show power-law MI decay. This is the clean operational definition of the **β axis** and a built-in test of whether the data generator actually produces long-range structure (the KV generator's `p(d)∝d^{−(β+1)}` is designed to).
- **Estimation pitfalls:** plug-in MI is **positively biased** at finite samples, with a `1/N` leading bias term (Miller–Madow / Paninski corrections; or NSB / bootstrap). Bias grows with alphabet size and k. The apparent power law can flatten into a noise floor at large k once `I(k)` drops below the bias ⇒ restrict the fit to the reliable mid-range; use surrogates to set the floor.

### Power-spectral-density (1/f^a) scaling
Welch/periodogram PSD; fit `S(f) ∼ f^{−a}` at low frequencies (log-log). For symbol streams, run on an indicator/embedded signal. `a` ties directly into H (next section). PSD is the Fourier dual of autocorrelation (Wiener–Khinchin), so it and DFA carry the same information via an integral transform.
- **Pitfalls:** spectral leakage (taper/window); pick the low-frequency scaling band; aliasing.

---

## Entropy / complexity & predictive information (the γ axis)

### Block (Shannon) entropy & entropy rate
Block entropy `H(L) = −Σ p(w) log p(w)` over length-L words. Per-symbol **conditional entropy** `h_μ(L) = H(L) − H(L−1)` is the finite-L entropy-rate estimate; it **decreases monotonically** to the true entropy rate `h_μ = lim h_μ(L)`.
- **γ axis mapping:** the project's `Hₙ ∼ n^{−γ}` is the *rate of convergence* of `h_μ(n)` toward its limit. Equivalently, the **excess entropy** `E = Σ_{L≥1} [h_μ(L) − h_μ]` measures the *total* sub-extensive correction; γ is the *power* governing how that sum's terms decay. Slow (power-law, small γ) convergence ⇔ strong long-range predictive structure.
- **Estimation pitfalls:** block entropy is **severely undersampled** for moderate L (need `≫ vocab^L` tokens). The plug-in estimate is downward-biased; use Miller–Madow, NSB, or **Context-Tree Weighting (CTW)** / Lempel–Ziv-based estimators, which extend reach in L. Finite-N corrections to `h_μ(L)` scale like `(log N)/N^δ`; do NOT fit the γ power law into the undersampled tail.

### Excess entropy / predictive information — Crutchfield–Feldman (2003); Bialek, Nemenman, Tishby (2001)
`E = I(past ; future)` = mutual information between the two semi-infinite halves of the sequence; equivalently `H(L) ≈ E + h_μ·L` (the intercept of the linear block-entropy asymptote). Bialek et al.'s **predictive information** `I_pred(L) = I(past-of-length-L ; future)` and its growth rate is the canonical "complexity" measure: bounded ⇒ simple, log-divergent ⇒ critical/finite-complexity, power-law-divergent ⇒ infinite predictive information.
- **Why it matters here:** this is the rigorous version of the project's "maximum effective complexity / edge-of-chaos" intuition. Predictive-information growth `I_pred(L) ∼ L^{(1−γ-like)}` and MI-decay exponent β are two faces of the same long-range structure (`I_pred(L) = Σ_{k≤L} I(k)` roughly), so **β and γ are linked through the predictive-information integral**, not fully independent.

### Permutation entropy — Bandt & Pompe (2002)
Entropy of the distribution of ordinal patterns (rank-orderings) of length-D windows. Fast, robust to monotonic transforms and noise, no binning.
- **Pitfalls:** ordinal — needs an ordered alphabet (good for embedded/continuous signals, less natural for nominal tokens); choose embedding D so `D! ≪ N`.

### Sample / approximate entropy — Pincus (1991, ApEn); Richman & Moorman (2000, SampEn)
Regularity statistics: probability that sequences similar for m points stay similar for m+1. SampEn fixes ApEn's self-match bias and is less length-dependent.
- **Pitfalls:** tolerance `r` and `m` choice; designed for short physiological series; amplitude-based (needs a metric).

### Lempel–Ziv complexity — Lempel & Ziv (1976); LZ77/LZ78
Count distinct phrases in an incremental parse; normalized LZ complexity estimates the entropy rate of the source and is an **algorithmic** compressibility measure. Pairs naturally with permutation entropy in the complexity–entropy plane to separate stochastic from deterministic-chaotic.
- **Strengths:** parameter-light, directly on symbols, asymptotically consistent for entropy rate. Excellent quick structure probe for token streams.
- **Pitfalls:** converges slowly; sensitive to alphabet coding; a single number, not a spectrum.

---

## Exponent relationships & pitfalls

For self-similar / long-memory processes the four families collapse onto **one** scaling, with these standard conversions (state assumptions explicitly when using them):

| From → To | Relation | Regime |
|---|---|---|
| DFA α ↔ Hurst H | `α = H` | stationary LRD (fGn) |
| DFA α ↔ fractional d | `α = d + 1/2`, `H = d + 1/2` | LRD |
| Spectral a ↔ DFA α | `a = 2α − 1` | general |
| Spectral a ↔ Hurst H | fGn: `a = 2H − 1`; fBm: `a = 2H + 1` | noise vs motion |
| Autocorr decay ↔ H | `ρ(k) ∼ k^{−γ_c}`, `γ_c = 2 − 2H` | stationary LRD |
| Rényi: D_q monotone | `τ(q) = (q−1)D_q`, `α=dτ/dq` | multifractal Legendre pair |

**Critical caveat — fGn vs fBm.** The most common error is applying the *noise* relation to *motion* data (or vice-versa). If `α>1` (or spectral `a>1` strongly), you are in the nonstationary fBm regime; differentiate first (or use the fBm relation). DFA's integration step is precisely what lets it span both.

**Project-specific note on β and γ.** In the project, β (correlation/MI decay) and γ (conditional-entropy convergence) are conceptually related to the *same* long-range structure: `I(k) ∼ k^{−β}` (two-point) and `Σ_k I(k)` ≈ predictive information whose finite-L correction is the γ power law. They are **not algebraically identical** (β is two-point, γ is full-block), so measuring both and checking consistency is meaningful — but treat "near-independent" with care. Hurst H is the most genuinely separate axis (it captures *signed/persistent* increment correlation, which symbol-level MI does not).

**General pitfalls across all estimators:**
- **Nonstationarity / trends** masquerade as long memory. Always run DFA-ℓ / wavelet / detrending variants and compare.
- **Crossovers**: two scaling regimes; report the band and the crossover scale, don't fit one slope across it.
- **Finite-size & undersampling**: entropy/MI plug-ins are biased (`O(1/N)`, alphabet-dependent); never fit power laws into the undersampled/noise-floor tail. Use surrogate/shuffled data to locate the floor.
- **Spurious multifractality**: LRD + finite size produce apparent `h(q)` spread even for monofractals — validate with shuffled and phase-randomized surrogates.
- **Short-range contamination**: R/S and GPH over-estimate H when short memory is present; prefer ARFIMA/local-Whittle/wavelet.
- **Embedding sensitivity**: D₂/permutation entropy depend on (m, τ); report robustness.

---

## Recommended methods for THIS project

A concrete shortlist mapped to the three axes, prioritizing **symbolic-sequence-appropriate** estimators and reuse of existing code.

### Measuring β (token-correlation / MI decay)
1. **Two-point mutual-information decay `I(k) ∼ k^{−β}`** (Lin–Tegmark) — *primary*. It is the native definition for symbolic data, directly tests whether the generator produces power-law (vs exponential, Markov) structure, and gives β as a log-log slope. Use Miller–Madow / NSB bias correction; restrict the fit to the reliable mid-range; set the noise floor with shuffled surrogates.
2. **Symbolic autocorrelation `C(k) ∼ k^{−β}`** — cheap cross-check; expect `β_corr ≈ β_MI` for near-Gaussian-ish structure but trust MI for heavy multi-symbol dependence.
3. **PSD `S(f) ∼ f^{−a}` on an embedded/indicator signal** — Fourier cross-check; convert via the table if a Hurst comparison is wanted.

### Measuring γ (conditional-entropy convergence `Hₙ ∼ n^{−γ}`)
1. **Conditional / block entropy convergence `h_μ(n) = H(n) − H(n−1)`** — *primary*; fit the approach `h_μ(n) − h_μ ∼ n^{−γ}`. This is *exactly* the project's definition.
2. **CTW or LZ-based entropy-rate estimator** to push the usable block length n further before undersampling kills the estimate (plug-in block entropy saturates fast at vocab=60).
3. **Excess entropy E / predictive information `I_pred(L)`** (Crutchfield–Feldman; Bialek et al.) as the integrated complementary view — and as the principled "**maximum effective complexity / edge-of-chaos**" target the project is hunting for (peak predictive information ⇔ richest exploitable long-range structure).

### Measuring Hurst H (near-independent long-memory axis)
1. **DFA-1/DFA-2** — *primary* upgrade over the current R/S. Robust to trends, directly gives `α=H`, trivial to implement, handles the nonstationary regime via its integration step. Replace/augment the existing R/S call.
2. **Wavelet (Abry–Veitch) logscale estimator** — most efficient/unbiased with confidence intervals; use as the accuracy reference.
3. **GPH or local-Whittle** — frequency-domain cross-check with known asymptotic CIs; flags short-range contamination if it disagrees with DFA.
4. Keep **R/S** only as a legacy sanity check (document its known upward bias).

### Multifractal layer (optional but cheap, and aligned with existing D_q code)
- **MFDFA** to get `h(q)` and `f(α)` — the natural multifractal extension of the recommended DFA, and a richer companion to the **Rényi D_q** the codebase already computes. If `h(q)` is flat / `D_q` is q-independent, the data is monofractal and a single H/β/γ suffices; spread signals genuine multiscaling. **Always validate spread against shuffled surrogates** (LRD + finite-size fake multifractality).

### Caveat on the existing D_q implementation
`estimate_renyi_dimensions_from_tokens` fits Rényi block entropy *linearly in block size m* and normalizes by `log(vocab)`. This measures a **per-symbol Rényi entropy *rate*** (normalized to [0,1]), which is a legitimate and useful "structure" scalar — but note it is conceptually an **entropy-rate / D_q-of-the-symbol-process** quantity, *not* the geometric box-counting `D_q` of an attractor. For the project's purposes it most naturally connects to the **γ axis** (entropy-rate scaling) rather than to a phase-space fractal dimension; label it accordingly to avoid the fGn-vs-fBm-style category confusion.

---

## References

1. **Hurst, H.E. (1951).** Long-term storage capacity of reservoirs. *Trans. Am. Soc. Civ. Eng.* 116:770–808. — Origin of R/S analysis and the Hurst exponent.
2. **Mandelbrot, B.B. & Wallis, J.R. (1969).** Robustness of the rescaled range R/S in the measurement of noncyclic long-run statistical dependence. *Water Resources Res.* 5:967–988. — Foundational R/S treatment of long memory.
3. **Lo, A.W. (1991).** Long-term memory in stock market prices. *Econometrica* 59:1279–1313. — Modified R/S correcting short-range autocorrelation bias.
4. **Peng, C.-K. et al. (1994).** Mosaic organization of DNA nucleotides. *Phys. Rev. E* 49:1685. — Introduces DFA; the standard LRD estimator robust to trends.
5. **Hu, K., Ivanov, P.Ch., Chen, Z., Carpena, P., Stanley, H.E. (2001).** Effect of trends on detrended fluctuation analysis. *Phys. Rev. E* 64:011114. (arXiv:physics/0103018) — Catalogs DFA trend/crossover artifacts.
6. **Bryce, R.M. & Sprague, K.B. (2012).** Revisiting detrended fluctuation analysis. *Sci. Rep.* 2:315. — Critical review of DFA finite-size curvature and detrending limits.
7. **Vandewalle, N. & Ausloos, M. (1998).** Crossing of two mobile averages: a method for measuring the roughness exponent. *Phys. Rev. E* 58:6832. — Origin of moving-average (DMA) detrending.
8. **Alessio, E., Carbone, A., Castelli, G., Frappietro, V. (2002).** Second-order moving average and scaling of stochastic time series. *Eur. Phys. J. B* 27:197–200. — Establishes the DMA Hurst estimator (variance ∼ n^{2H}).
9. **Abry, P. & Veitch, D. (1998).** Wavelet analysis of long-range-dependent traffic. *IEEE Trans. Inf. Theory* 44:2–15. — Efficient, trend-robust wavelet H estimator with confidence intervals.
10. **Geweke, J. & Porter-Hudak, S. (1983).** The estimation and application of long memory time series models. *J. Time Ser. Anal.* 4:221–238. — GPH log-periodogram regression for d (H=d+1/2).
11. **Künsch, H.R. (1987).** Statistical aspects of self-similar processes. *Proc. 1st World Congress Bernoulli Soc.* — Introduces the local Whittle approach.
12. **Robinson, P.M. (1995).** Gaussian semiparametric estimation of long range dependence. *Ann. Statist.* 23:1630–1661. — Local Whittle consistency/asymptotic normality; efficient semiparametric d.
13. **Granger, C.W.J. & Joyeux, R. (1980).** An introduction to long-memory time series models and fractional differencing. *J. Time Ser. Anal.* 1:15–29. — ARFIMA / fractional integration.
14. **Hosking, J.R.M. (1981).** Fractional differencing. *Biometrika* 68:165–176. — ARFIMA theory; `(1−L)^d`, H=d+1/2.
15. **Kantelhardt, J.W. et al. (2002).** Multifractal detrended fluctuation analysis of nonstationary time series. *Physica A* 316:87–114. — MFDFA: generalized Hurst h(q), τ(q), f(α). The default multifractal tool.
16. **Muzy, J.F., Bacry, E., Arneodo, A. (1991).** Wavelets and multifractal formalism for singular signals. *Phys. Rev. Lett.* 67:3515. — WTMM multifractal formalism.
17. **Arneodo, A., Bacry, E., Muzy, J.F. (1995).** The thermodynamics of fractals revisited with wavelets. *Physica A* 213:232–275. — Full WTMM partition-function / f(α) method.
18. **Frisch, U. & Parisi, G. (1985).** On the singularity structure of fully developed turbulence. In *Turbulence and Predictability...* (Ghil et al., eds.), North-Holland. — Structure-function multifractal formalism; ζ(q) and the Legendre transform to f(α).
19. **Mandelbrot, B.B. & Van Ness, J.W. (1968).** Fractional Brownian motions, fractional noises and applications. *SIAM Rev.* 10:422–437. — fBm/fGn; the basis of all H↔spectral↔autocorrelation conversions.
20. **Grassberger, P. & Procaccia, I. (1983).** Characterization of strange attractors. *Phys. Rev. Lett.* 50:346. — Correlation dimension D₂ from the correlation sum.
21. **Grassberger, P. & Procaccia, I. (1983).** Measuring the strangeness of strange attractors. *Physica D* 9:189–208. — Generalized D_q / correlation-integral scaling.
22. **Hentschel, H.G.E. & Procaccia, I. (1983).** The infinite number of generalized dimensions of fractals and strange attractors. *Physica D* 8:435–444. — Rényi generalized dimensions D_q and the D₀/D₁/D₂ hierarchy.
23. **Takens, F. (1981).** Detecting strange attractors in turbulence. *Lecture Notes in Math.* 898:366–381. — Delay embedding underlying D₂/permutation/sample-entropy on scalar series.
24. **Lin, H.W. & Tegmark, M. (2017).** Critical behavior in physics and probabilistic formal languages. *Entropy* 19:299. (arXiv:1606.06737) — Power-law MI decay ⇔ long-range structure; Markov/regular ⇒ exponential, context-free/recursive ⇒ power law. Core reference for the β axis.
25. **Crutchfield, J.P. & Feldman, D.P. (2003).** Regularities unseen, randomness observed: levels of entropy convergence. *Chaos* 13:25–54. (arXiv:cond-mat/0102181) — Block-entropy convergence, entropy rate, excess entropy E = Σ[h_μ(L)−h_μ]. Core reference for the γ axis.
26. **Bialek, W., Nemenman, I., Tishby, N. (2001).** Predictability, complexity, and learning. *Neural Computation* 13:2409–2463. — Predictive information I_pred(L)=I(past;future); the principled "complexity"/edge-of-chaos measure.
27. **Bandt, C. & Pompe, B. (2002).** Permutation entropy: a natural complexity measure for time series. *Phys. Rev. Lett.* 88:174102. — Ordinal-pattern entropy; fast, robust.
28. **Richman, J.S. & Moorman, J.R. (2000).** Physiological time-series analysis using approximate and sample entropy. *Am. J. Physiol.* 278:H2039–H2049. — SampEn (and ApEn, Pincus 1991) regularity statistics.
29. **Lempel, A. & Ziv, J. (1976).** On the complexity of finite sequences. *IEEE Trans. Inf. Theory* 22:75–81. — LZ algorithmic complexity / entropy-rate estimator for symbol streams.
30. **Miller, G. (1955); Paninski, L. (2003).** Note on the bias of information estimates (Miller); Estimation of entropy and mutual information (Paninski, *Neural Computation* 15:1191–1253). — Finite-sample bias of plug-in entropy/MI and corrections; essential for honest β/γ fits.
31. **Eke, A., Herman, P., Kocsis, L., Kozak, L.R. (2002).** Fractal characterization of complexity in temporal physiological signals. *Physiol. Meas.* 23:R1–R38. — Practical guide to fGn-vs-fBm classification and choosing among H estimators (the category-error caveat).
