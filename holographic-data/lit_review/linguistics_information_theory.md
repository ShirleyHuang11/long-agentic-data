# Information Theory & Statistical Linguistics of Natural Language: Entropy Rate, Excess Entropy, Hilberg's Law, and Long-Range Dependence

*A method-focused literature review supporting the γ (next-token conditional-entropy decay) and β (token-correlation decay) exponents of the holographic-data project.*

Web search **was available**; all citations below were checked against primary sources (arXiv / journal pages / canonical PDFs). Numerical values quoted are as reported by the cited authors.

---

## Overview

The project's anchor paper — Cagnetta, Raventós, Ganguli & Wyart, *"Deriving neural scaling laws from the statistics of natural language"* (arXiv:2602.07488, 2026; the `gamma-beta.pdf` asset) — defines two dataset-level exponents and a derived scaling exponent:

- **γ**: the power-law decay of the next-token **conditional entropy** with conditioning context length, `Hₙ − H∞ ∼ n^{−γ}` (their Eq. 6). `Hₙ = −E[log P(x_{n+1} | x_{1:n})]` is the n-gram (block) conditional entropy of the *data*, lower-bounded empirically by the n-gram cross-entropy of a well-trained LM.
- **β**: the power-law decay of **token–token correlations** with temporal separation, `‖C(n)‖_op ∼ n^{−β}` (their Eq. 7), where `C(n)` is the two-point co-occurrence covariance matrix at lag n.
- **α_D = γ / (2β)**: the data-limited neural-scaling exponent, `L(P) − H∞ ∼ P^{−α_D}`.

Both γ and β are *statistical properties of the language*, not of the model. This review collects the half-century of information-theoretic and statistical-linguistics work that (i) **defines** these quantities (entropy rate, excess entropy, mutual-information decay, Hilberg's law), (ii) supplies **practical estimators** for them, and (iii) documents the **long-range-dependence evidence** in text that makes the power-law (rather than exponential) ansatz credible. The single most important conceptual link: γ is the *decay rate of the conditional entropy toward the entropy rate*, and the **excess entropy / predictive information** E = Σₙ(Hₙ − H∞) is finite iff γ > 1 — so γ separates "weak" (summable, finite-memory) from "strong" (non-summable, Hilberg-type) long-range dependence. β is the language-statistics counterpart of the Lin–Tegmark / Ebeling–Pöschel mutual-information-decay exponent.

---

## Entropy of language & entropy-rate estimation

The entropy rate `h = lim_{n→∞} Hₙ` of English has been estimated repeatedly; the *rate of convergence* `Hₙ → h` is exactly the project's γ.

- **Shannon (1951)** introduced the operational definition and the first estimates: single-letter, digram, trigram models give 4.03 / 3.32 / 3.1 bits/letter; his human-prediction "guessing game" pushed the bound to ~1.0–1.3 bits/letter and gave upper/lower bounds from the rank distribution of guesses. This is the origin of both the *entropy-rate* and the *conditional-entropy-vs-context-length* curve. **Method to borrow:** the next-letter / next-token prediction protocol is the direct human analog of evaluating an LM's `Hₙ` at varying n.
- **Cover & King (1978)** replaced guessing with **sequential gambling**: a subject bets a fraction of capital on each next symbol; capital growth rate gives a *convergent* (not just bounded) entropy estimate, `Ĥ = (1 − (1/n)log_{27} S_n)·log₂27`. Estimate ≈ 1.3 bits/char. Key insight: optimal bets equal the conditional probabilities, tying the estimator to the data's `Hₙ`.
- **Brown et al. (1992)** gave a model-based **upper bound** of 1.75 bits/char via a word-trigram model's cross-entropy on a balanced corpus — the prototype of using an LM's cross-entropy as an upper bound on `Hₙ` (exactly the project's strategy: `L_n(P, M) ≥ Hₙ`).
- **Plug-in / block-entropy, LZ, and CTW estimators** (see Methods section) are the non-parametric route. **Schürmann & Grassberger (1996)** characterize the finite-sample bias and propose the `h_N = h∞ + c·(log N)/N^γ` finite-size ansatz central to extrapolating `Hₙ`.
- **Takahira, Tanaka-Ishii & Dębowski (2016)** and **Takahashi & Tanaka-Ishii (2018)** extrapolate compressed-corpus and neural-LM cross-entropies to the infinite-data / infinite-context limit, estimating English entropy rate ≈ 1.44 bits/char (~20% below earlier estimates) and showing cross-entropy obeys power-law decay in both data size *and* context length — directly the γ phenomenon and the project's `L(P)` scaling.

---

## Hilberg's law, excess entropy & long memory

- **Hilberg (1990)** reinterpreted Shannon's data and conjectured **sub-extensive (power-law) block-entropy growth**: `H(n) ≈ A·n^β_H` with β_H ≈ 0.5 (β_H < 1), equivalently mutual information between adjacent length-n blocks growing as a power law `E(n) ∼ n^{β_H}`. This is the foundational "long memory in text" claim; β_H here is a *block-entropy* exponent and relates to the project's γ via `Hₙ − H∞ ∼ n^{−γ}` (the marginal/derivative form of Hilberg growth).
- **Excess entropy / predictive information** `E = Σₙ (Hₙ − H∞) = I(past; future)`. If `Hₙ − H∞ ∼ n^{−γ}`, then E is **finite iff γ > 1** and **diverges (power-law, Hilberg-type) iff γ ≤ 1**. This is the precise dividing line between weak and strong long-range dependence and gives γ a model-free interpretation. (Crutchfield & Feldman 2003 give the canonical treatment of excess entropy / predictive information convergence.)
- **Dębowski** built the rigorous theory: *Excess entropy in natural language: present state and perspectives* (2011) reviews estimators and evidence; he proved the **"theorem of the facts and words"** linking power-law mutual information to Herdan/Heaps vocabulary growth, and constructed **Santa Fe / Oracle processes** — stationary sources with provable power-law block-entropy growth — and later **multiperiodic processes** (ergodic, vanishing entropy rate, still Hilberg-satisfying). His monograph *Information Theory Meets Power Laws* (Wiley, 2021) is the definitive reference; it formalizes excess entropy, conditional MI, maximal-repetition, and the facts-vs-words link. **Most relevant theory** for interpreting what γ ≤ 1 vs γ > 1 means about the data-generating process.
- **Ebeling & Pöschel (1994)** measured, on *Moby Dick* and Grimm's tales: mutual information between letters decaying as a **power law**, per-letter entropy `Hₙ` decaying roughly as `n^{−1/2}` (i.e. γ ≈ 0.5), and sub-word counts growing as a stretched exponential. This is an early *direct measurement* of the γ-curve in real text.
- **Hilberg exponents** (Dębowski 2014) formalize multiple "Hilberg-type" exponents (for entropy, for maximal repetition, for the "facts") as new measures of long memory.

---

## Zipf / Heaps & scaling

- **Zipf's law** (rank-frequency `f(r) ∼ r^{−z}`, z ≈ 1) and **Heaps'/Herdan's law** (vocabulary `V(n) ∼ n^{ν}`, 0 < ν < 1) are the most robust text power laws. They are mathematically linked: **Heaps follows from Zipf**, with `ν ≈ 1/z` (asymptotically) — see **Lü, Zhang & Zhou (2010)** "Zipf's Law Leads to Heaps' Law" and **Font-Clos et al. (2013)** "A scaling law beyond Zipf's law." This matters for the project because Zipf-distributed tokens are the substrate of recent neural-scaling-from-data-statistics work (Michaud et al. quanta; Kunstner & Bach bigram scaling).
- **Dębowski's facts-and-words theorem** turns Zipf/Heaps into a *statement about long memory*: a Heaps-type power-law vocabulary growth co-occurs with power-law growth of mutual information (the project's strong-LRD regime). The recent **arXiv:2512.13491** ("From Zipf's Law to Neural Scaling through Heaps' Law and Hilberg's Hypothesis", 2025) explicitly chains Zipf → Heaps → Hilberg → neural scaling, the closest published cousin to the project's γ/β framework.

---

## Mutual-information decay in text

This is the literature most directly tied to **β** (and to the strong/weak LRD distinction).

- **Lin & Tegmark (2017)**, *Critical Behavior in Physics and Probabilistic Formal Languages* (Entropy): the **canonical β reference**. In texts, music, and genomes the mutual information `I(X_i; X_{i+n})` between two symbols decays roughly as a **power law** in the separation n. They prove that *regular grammars / (hidden) Markov chains give exponential decay*, whereas *context-free / recursive (tree-like) generators can give power-law decay* — explaining why text needs deep, not Markov, models. β is precisely this two-point MI/correlation decay exponent.
- **Ebeling–Pöschel (1994)**, **Altmann, Cristadoro & Degli Esposti (2012)** *On the origin of long-range correlations in texts* (PNAS): LRD in text is driven by **burstiness** of semantically loaded (topical) words; correlations propagate from high linguistic levels (topics) down to letters. Explains *why* β exists and is broad-spectrum (the covariance matrix has many singular directions, as the project's `gamma-beta.pdf` Fig. 3 notes).
- **Futrell, Gibson & Levy (2020)** *Lossy-Context Surprisal* (Cognitive Science) + **Futrell (2019)** *Information-theoretic locality* + **Hahn, Futrell et al.** *Information locality as an inductive bias*: formalize **information locality** — processing/learning is easiest when mutually-informative tokens are close. Their "memory–surprisal trade-off" curve is the integral of the conditional MI decay; the decay exponent is a cousin of γ/β and provides a cognitively grounded estimator of how predictive information is distributed over distance. **Method to borrow:** the memory–surprisal trade-off curve as an alternative readout of the same decay.
- **Tanaka-Ishii & Bunde / Tanaka-Ishii & Takahashi**: long-range-correlation analysis of text (rare-word interval series), **Taylor's law** (`σ ∝ μ^ζ`, ζ ∈ [0.5,1]; ζ>0.5 ⇒ LRD/clustering), and **correlation dimension** of text and of LLMs in a statistical manifold (PRR 2024) — fractal-geometric measures of the same self-similarity, and a measured-complexity axis analogous to the project's Rényi-D diagnostic.

---

## Methods to borrow (mapped to γ and β, with pitfalls)

**Estimating γ (conditional-entropy convergence `Hₙ − H∞ ∼ n^{−γ}`):**

1. **Neural-LM cross-entropy at varying context n** (the project's & Cagnetta et al.'s method, after Brown 1992 / Takahashi–Tanaka-Ishii 2018). Train/evaluate an expressive LM and read `L_n(P) = −E[log P̂(x_{n+1}|x_{1:n})]` as a function of n; it upper-bounds `Hₙ`. Fit a power law to the small-n decay. **Most directly useful estimator for γ.** *Pitfalls:* the bound is loose unless the model is well-trained and P (data) is large — undertrained or data-limited models inflate `Hₙ` and bias γ; must verify the curve has *converged* in P (Cagnetta et al. show `L_n(P)` converging from above as P grows) before fitting; positional-encoding/architecture effects are second-order (they verify γ is architecture-independent across APE/RoPE/LLaMA).
2. **Block (plug-in) entropy** `Hₙ = H(n+1) − H(n)` from empirical n-gram counts. *Pitfall:* severe downward (Miller–Madow) bias once `n` makes the context count exceed the sample — "reliable frequency estimates require prohibitive sample sizes" (the reason Cagnetta et al. *avoid* it). Use Schürmann–Grassberger / Bayesian (Nemenman–Shafee–Bialek) bias corrections; reliable only for small n.
3. **Compression-based** (LZ77/LZ78, **CTW**, PPM). CTW converges faster and is the most accurate non-parametric entropy-rate estimator; combine with the **Schürmann–Grassberger finite-size ansatz** `h_N = h∞ + c·(log N)/N^γ` to *extract γ as the convergence exponent itself*. *Pitfall:* LZ has high variance; CTW convergence varies across partitions — average over partitions and bootstrap for error bars.

**Estimating β (two-point correlation decay `‖C(n)‖ ∼ n^{−β}`):**

4. **Token co-occurrence covariance matrix** `C(n)_{μν} = P(X_i=μ, X_{i+n}=ν) − P(X_i=μ)P(X_{i+n}=ν)`; take top singular value `‖C(n)‖_op` (or Frobenius norm) vs n and fit a power law (Cagnetta et al. Eq. 3/7, Fig. 3). **Most directly useful estimator for β.** *Pitfalls:* finite-sample noise floor `‖Ĉ(n) − C(n)‖ = O(P^{−1/2})` sets the largest usable n; correlations can be **broken power laws** (WikiText shows a short-lag and long-lag regime — fit the *short-lag* stage relevant to the prediction); choice of norm (op vs Frobenius) should be cross-checked (they track each other).
5. **Symbol mutual information** `I(X_i; X_{i+n})` vs n (Lin–Tegmark, Ebeling–Pöschel) — the information-theoretic sibling of (4); β_MI and the covariance-β coincide for weakly-coupled symbols. *Pitfall:* MI estimation has its own positive finite-sample bias (same NSB/plug-in caveats as γ).
6. **Long-range-correlation / DFA / Taylor's-law analyses on a symbol-indicator or rare-word-interval series** (Tanaka-Ishii, Altmann) — robust to trends, give a Hurst-type exponent convertible to β via the standard `autocorr ∼ k^{−(2−2H)}` relation.

**Cross-check / theory:** verify `α_D = γ/(2β)` by the **scaling-collapse** test (Cagnetta et al.): rescale `P → P/n^{2β}` and `L_n → n^γ L_n`; all n-gram curves should collapse onto one master curve. Excess-entropy finiteness (`E = Σ(Hₙ−H∞)` finite ⇔ γ>1) is the model-free check on whether the data is weak- or strong-LRD.

---

## Relevance to this project

- **Direct definitional grounding.** The project's γ and β are *exactly* the entropy-rate convergence exponent (Shannon/Hilberg/Ebeling–Pöschel/Dębowski) and the two-point MI/correlation decay exponent (Lin–Tegmark/Ebeling–Pöschel). This review supplies the canonical definitions, the `E = Σ(Hₙ−H∞)` excess-entropy interpretation, and the weak-vs-strong-LRD threshold at γ=1.
- **Estimator menu with pitfalls** is portable to the `phase_core` diagnostics. The neural-LM-cross-entropy-vs-n method (for γ) and the co-occurrence-covariance-singular-value method (for β) are the two highest-value, most-standard estimators and are exactly what the anchor paper validated; the plug-in/CTW/Schürmann–Grassberger alternatives give independent cross-checks with documented bias corrections.
- **Synthetic-data design.** The project's `AlgorithmicKVGenerator` draws retrieval distance from `p(d) ∝ d^{−(β+1)}`, *engineering* a power-law correlation — Dębowski's Santa Fe / multiperiodic processes are the rigorous precedents for "stationary process with tunable power-law block entropy," and Lin–Tegmark explains why a recursive/long-range generator (not a Markov one) is needed to get power-law (not exponential) MI. The "edge-of-chaos / maximum effective complexity" goal maps onto the **excess-entropy / predictive-information maximum** and the γ≈1 (Hilberg) boundary.
- **Caveats to import.** Finite-sample bias (plug-in entropy, MI), broken power laws (fit the right regime), convergence-in-P before fitting γ, and noise floors on `C(n)` are all live threats already visible in the project's undertrained-Mamba (H3 inconclusive) and small-cell-count fits.

---

## References

1. **Shannon, C. E. (1951).** *Prediction and Entropy of Printed English.* Bell System Technical Journal 30:50–64. — First entropy-rate estimates and the human next-symbol prediction protocol; origin of the `Hₙ`-vs-context curve.
2. **Cover, T. M. & King, R. C. (1978).** *A Convergent Gambling Estimate of the Entropy of English.* IEEE Trans. Information Theory 24(4):413–421. — Sequential-betting estimator giving a convergent ≈1.3 bits/char estimate; optimal bets = conditional probabilities.
3. **Brown, P. F., Della Pietra, S. A., Della Pietra, V. J., Lai, J. C. & Mercer, R. L. (1992).** *An Estimate of an Upper Bound for the Entropy of English.* Computational Linguistics 18(1):31–40. — 1.75 bits/char upper bound via word-trigram cross-entropy; prototype of LM-cross-entropy-as-upper-bound-on-Hₙ.
4. **Hilberg, W. (1990).** *Der bekannte Grenzwert der redundanzfreien Information in Texten — eine Fehlinterpretation der Shannonschen Experimente?* Frequenz 44:243–248. — Conjectures sub-extensive power-law block-entropy growth `H(n) ∼ n^{β_H}` (β_H≈0.5); foundational long-memory-in-text claim.
5. **Ebeling, W. & Pöschel, T. (1994).** *Entropy and Long-Range Correlations in Literary English.* Europhysics Letters 26(4):241–246. — Direct measurement: power-law letter MI, `Hₙ ∼ n^{−1/2}` (γ≈0.5), stretched-exponential subword growth in *Moby Dick* / Grimm.
6. **Crutchfield, J. P. & Feldman, D. P. (2003).** *Regularities Unseen, Randomness Observed: Levels of Entropy Convergence.* Chaos 13:25–54. — Canonical treatment of excess entropy / predictive information and `Hₙ → h` convergence; the `E = Σ(Hₙ−H∞)` finiteness ⇔ γ>1 framework.
7. **Dębowski, Ł. (2011).** *Excess Entropy in Natural Language: Present State and Perspectives.* Chaos 21:037105 (arXiv:1105.1306). — Reviews excess-entropy estimators/evidence; states the facts-and-words link to power-law MI.
8. **Dębowski, Ł. (2014).** *Hilberg Exponents: New Measures of Long Memory in the Process.* IEEE Trans. Information Theory (arXiv:1403.1757). — Formalizes multiple Hilberg-type exponents (entropy, maximal repetition, facts) as long-memory measures.
9. **Dębowski, Ł. (2021).** *Information Theory Meets Power Laws: Stochastic Processes and Language Models.* Wiley. — Definitive monograph: excess entropy, conditional MI, Santa Fe/multiperiodic processes, maximal repetition, facts-vs-words theorem.
10. **Lin, H. W. & Tegmark, M. (2017).** *Critical Behavior in Physics and Probabilistic Formal Languages.* Entropy 19(7):299 (arXiv:1606.06737). — Canonical β reference: power-law two-symbol MI decay in text/music/genomes; Markov⇒exponential, context-free/recursive⇒power-law.
11. **Altmann, E. G., Cristadoro, G. & Degli Esposti, M. (2012).** *On the Origin of Long-Range Correlations in Texts.* PNAS 109(29):11582–11587. — LRD in text driven by topical-word burstiness; correlations flow from semantics down to letters.
12. **Futrell, R., Gibson, E. & Levy, R. P. (2020).** *Lossy-Context Surprisal: An Information-Theoretic Model of Memory Effects in Sentence Processing.* Cognitive Science 44(3):e12814. — Information locality / memory–surprisal trade-off; cognitively grounded readout of conditional-MI decay over distance.
13. **Futrell, R. (2019).** *Information-Theoretic Locality Properties of Natural Language.* (Quasy/SyntaxFest). — Formalizes information locality: high-MI elements tend to be linearly close; estimator of how predictive information is distributed.
14. **Takahashi, S. & Tanaka-Ishii, K. (2018).** *Cross Entropy of Neural Language Models at Infinity — A New Bound of the Entropy Rate.* Entropy 20(11):839. — Extrapolates neural-LM cross-entropy in data size and context length (power-law decay); English entropy rate ≈1.44 bits/char.
15. **Takahira, R., Tanaka-Ishii, K. & Dębowski, Ł. (2016).** *Entropy Rate Estimates for Natural Language — A New Extrapolation of Compressed Large-Scale Corpora.* Entropy 18(10):364. — Compression-based entropy-rate extrapolation across 6 languages; finite-size extrapolation methodology.
16. **Schürmann, T. & Grassberger, P. (1996).** *Entropy Estimation of Symbol Sequences.* Chaos 6:414–427. — Finite-sample bias analysis and the `h_N = h∞ + c·(log N)/N^γ` convergence ansatz used to extract the entropy-rate convergence exponent.
17. **Tanaka-Ishii, K. & Bunde, A. (2016).** *Long-Range Memory in Literary Texts: On the Universal Clustering of the Rare Words.* PLOS ONE 11(11):e0164658. — Rare-word interval series shows universal long-range correlation/clustering across languages.
18. **Tanaka-Ishii, K. & Kobayashi, T. (2018).** *Taylor's Law for Human Linguistic Sequences.* ACL 2018 (arXiv:1804.07893). — Taylor's law `σ ∝ μ^ζ` (ζ∈[0.5,1]); ζ>0.5 signals long-range clustering; corpus-scale measurement.
19. **Du, X. & Tanaka-Ishii, K. (2024).** *Correlation Dimension of Natural Language in a Statistical Manifold.* Phys. Rev. Research 6:L022028. — Fractal correlation-dimension measure of text/LLM self-similarity; a measured-complexity axis analogous to the project's Rényi-D.
20. **Lü, L., Zhang, Z.-K. & Zhou, T. (2010).** *Zipf's Law Leads to Heaps' Law: Analyzing Their Relation in Finite-Size Systems.* PLOS ONE 5(12):e14139. — Derives Heaps from Zipf with `ν ≈ 1/z`; finite-size corrections.
21. **Font-Clos, F., Boleda, G. & Corral, Á. (2013).** *A Scaling Law Beyond Zipf's Law and Its Relation to Heaps' Law.* New J. Phys. 15:093033. — Generalized Zipf scaling and its Heaps consequence; corpus-size dependence of exponents.
22. **Cagnetta, F., Raventós, A., Ganguli, S. & Wyart, M. (2026).** *Deriving Neural Scaling Laws from the Statistics of Natural Language.* arXiv:2602.07488. — Project anchor: defines γ (`Hₙ−H∞∼n^{−γ}`), β (`‖C(n)‖∼n^{−β}`), `α_D=γ/2β`; validates via scaling collapse on TinyStories/WikiText.
23. **Anonymous/2025 (arXiv:2512.13491).** *From Zipf's Law to Neural Scaling through Heaps' Law and Hilberg's Hypothesis.* — Chains Zipf→Heaps→Hilberg→neural-scaling; closest published cousin to the γ/β framework. *(verify authorship/venue.)*
24. **Nemenman, I., Shafee, F. & Bialek, W. (2002).** *Entropy and Inference, Revisited.* NeurIPS 14. — NSB Bayesian entropy estimator; the standard bias correction for plug-in entropy/MI in the undersampled regime (γ/β finite-sample pitfalls).
25. **Dębowski, Ł. (2018).** *Is Natural Language a Perigraphic Process? The Theorem about Facts and Words Revisited.* Entropy 20(2):85. — Rigorous facts-and-words theorem linking power-law vocabulary growth to power-law mutual information (Heaps ↔ Hilberg).
