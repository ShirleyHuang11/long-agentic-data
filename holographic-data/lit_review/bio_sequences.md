# Long-Range Structure, Fractality, and Length-Generalization Methods in Biological Sequences

*A method-focused literature review for importing biology's sequence-complexity tooling into the (β, γ) phase-diagram / length-generalization project.*

Web search **was available**; references below carry verified arXiv IDs / DOIs / venues. Where a classic 1990s paper has no stable open ID, the venue + year is given and was cross-checked against secondary sources.

---

## Overview

Biological sequences (DNA, RNA, protein, and the 3D genome) have been studied for ~35 years as **long-range-correlated, fractal/multifractal symbolic series**. This community independently invented and battle-tested almost exactly the measurement toolkit this project needs: estimators for power-law correlation decay, for the scaling of conditional entropy with context, and for the fractal/Hurst character of a sequence — the same objects the project calls **β** (token–token correlation decay exponent), **γ** (conditional-entropy decay exponent, Hₙ ∼ n^−γ), and the Rényi/effective-dimension diagnostics already in `phase_core.py`.

Three threads are relevant:

1. **Statistical-physics genomics (1992–2010):** DNA-walk, DFA, multifractal DFA (MFDFA), mutual-information decay, power-spectrum 1/f analysis, block entropy, Zipf/Heaps linguistics, isochore/mosaic structure. These give *estimators* for β-like and γ-like exponents and a Hurst exponent H directly transferable to the project's correlation and entropy diagnostics.
2. **3D genome / Hi-C (2009–):** the **fractal globule** — a concrete physical generative model whose contact probability decays as a clean power law P(s) ∝ s^−1, a striking real-world analogue of a tunable power-law-decay data generator and a candidate "edge-of-chaos" structured-but-not-frozen regime.
3. **Genomic & protein language models (2021–2026):** Enformer, HyenaDNA, Nucleotide Transformer, DNABERT-2, Evo/Evo2 (StripedHyena), Caduceus (bi-directional Mamba), ESM-2. These contribute concrete **length-generalization and long-range-dependency architectures** — sub-quadratic implicit convolutions and selective SSMs that zero-shot extrapolate 10–100× beyond training length — directly informing the project's Transformer-vs-Mamba length-gen finding (Result 11).

The deepest conceptual bridge between the two worlds is **Shen 2019** (mutual-information scaling and expressive power), which proves attention captures power-law MI that RNNs cannot, and **L²M 2025** (MI scaling law for long-context) — these connect the project's β/γ correlation-decay framing to architecture expressivity in the *same language* biology uses for genomes.

---

## Long-range correlations & fractal structure in genomes

**The founding observation (Peng et al. 1992; Voss 1992; Li & Kaneko 1992).** Mapping a nucleotide sequence to a ±1 "DNA walk" (purine/pyrimidine) and measuring the mean-square fluctuation F(ℓ) of the walk over windows of length ℓ revealed F(ℓ) ∝ ℓ^H with H > 1/2 in non-coding/intron-rich DNA — i.e. **persistent long-range correlations** spanning thousands of bases, where an uncorrelated sequence would give H = 1/2. Voss independently found power-law 1/f^α power spectra (α ≈ 0.5–0.85). Li & Kaneko tied the effect to repetitive elements and heterogeneous "memory."

**Detrended Fluctuation Analysis — DFA (Peng et al. 1994).** Because raw DNA walks are non-stationary (local compositional bias, patchiness), Peng et al. introduced **DFA**: integrate the series, split into windows, fit and subtract a local polynomial trend in each window, then measure the RMS of the residuals F(n) ∝ n^α. The scaling exponent **α is a Hurst-like exponent**: α = 0.5 uncorrelated, 0.5 < α < 1 long-range positively correlated, α > 1 non-stationary/super-diffusive. DFA's removal of polynomial trends makes it robust on the kind of non-stationary symbolic data the project generates. This is the single most-transferable estimator (see Methods to borrow).

**Mutual-information decay (Li, Marr & Kaneko 1994; Grosse et al. 2000).** A model-free alternative to the DNA-walk: compute the average mutual information M(d) between symbols separated by distance d. In non-coding DNA, M(d) decays as a **power law** (∝ d^−β-like) extending to ≥ hundreds of bases, vs. exponential decay for short-memory/Markov sources. This is *exactly* the project's β object — M(d) is a direct, alphabet-native estimator of token–token correlation decay needing no walk embedding.

**Power spectrum / 1/f and DNA walks (Buldyrev, Goldberger, Havlin, Stanley 1990s; Lévy-walk model).** The Fourier power spectrum S(f) ∝ 1/f^α at low f is the spectral dual of the autocorrelation power law; the generalized **Lévy-walk** generative model reproduces both the long-range correlations and the long biased-walk subregions of non-coding DNA — a candidate parametric generator with a single tunable correlation exponent.

**Multifractal DFA — MFDFA (Kantelhardt et al. 2002, applied widely to genomes).** Generalizes DFA by computing the q-dependent fluctuation Fq(n) ∝ n^h(q). The **generalized Hurst exponent h(q)**: if h(q) is constant the series is *mono*fractal; if h(q) decreases with q the series is **multifractal** — different scaling for large vs. small fluctuations. Genomes (maize, soybean, human) are robustly multifractal. MFDFA yields a *spectrum* of exponents and a singularity width Δα that quantifies heterogeneity of structure — a richer descriptor than a single β or γ.

**Block entropy, Zipf & Heaps linguistics (Mantegna et al. 1994; Schmitt & Herzel).** Treating k-mers as "words": the **block entropy** Hₖ (Shannon entropy of length-k blocks) and its increments hₖ = Hₖ − Hₖ₋₁ (the *entropy rate* / conditional entropy of the next symbol given k−1 prior symbols) measure how fast uncertainty about the next token falls as context grows — **this is the project's γ object measured directly on symbols.** Mantegna et al. further showed non-coding DNA follows a **Zipf law** (rank-frequency power law) more closely than coding DNA, and obeys Heaps' law (vocabulary growth). These are textbook complexity statistics ML can compute on token streams.

**Isochore / mosaic / CpG structure.** Genomes are mosaics of long (>200 kb) compositionally homogeneous **isochores**; CpG-island placement shows long-range autocorrelations out to ~10 Mb. This is the biological origin of much of the measured long-range correlation: *patchiness at many scales*, a natural picture for "structured but not periodic" data sitting between order and randomness.

---

## 3D genome / long-range contacts

**Hi-C and the fractal globule (Lieberman-Aiden et al. 2009, Science).** Genome-wide chromosome-conformation-capture (Hi-C) measures contact frequency P(s) between loci separated by genomic distance s. The headline result: **P(s) ∝ s^−1.08** over 500 kb–7 Mb, matching the **fractal globule** (crumpled, knot-free, P(s) ∝ s^−1) rather than the equilibrium globule (s^−3/2). The fractal globule is a *non-equilibrium, self-similar, space-filling (Peano-like) polymer state* that packs densely yet unfolds locally without entanglement.

**Why this matters as a generative analogy.** The fractal globule is a real physical system whose long-range dependency structure is a **single clean power law with a tunable exponent**, produced by a self-similar folding process — a concrete, biologically validated instance of the kind of power-law-correlation data generator the project builds with β. The exponent of P(s) is a direct, interpretable analogue of β; the difference between s^−1 (fractal, structured, evolvable) and s^−3/2 (equilibrium, "frozen") is suggestive of the project's edge-of-chaos / maximum-effective-complexity hypothesis. Long-range enhancer–promoter contacts are exactly the dependencies genomic LMs (next section) must learn.

---

## Genomic & protein language models (long-context & length-gen methods)

**Enformer (Avsec et al. 2021, Nature Methods, doi:10.1038/s41592-021-01252-x).** Transformer + convolutional trunk predicting gene expression / chromatin from DNA, integrating enhancers up to ~100 kb away (200 kb receptive field, 5× prior CNNs). Demonstrates that *self-attention captures long-range regulatory dependencies* that dilated convolutions miss — direct evidence that the architecture's correlation-modeling reach is the binding constraint, echoing the project's correlation-decay framing.

**HyenaDNA (Nguyen et al. 2023, NeurIPS; arXiv:2306.15794).** Replaces attention with the **Hyena operator** (implicit long convolutions + gating), sub-quadratic in length, enabling **single-nucleotide context up to 1 M tokens** (≈500× longer than dense-attention DNA models) and ~160× faster training. The key length-gen lesson: long implicit convolutions give global context at every layer without quadratic cost — an alternative to attention for long-range dependency.

**Nucleotide Transformer (Dalla-Torre et al. 2023/2025).** Family of DNA transformers pretrained on 850+ species; replaces overlapping k-mers with non-overlapping tokens to shorten sequences. Establishes the multi-species pretraining + downstream-benchmark paradigm; its 18-dataset suite is the comparison baseline HyenaDNA and others report against.

**DNABERT-2 (Zhou et al. 2023; ICLR 2024; arXiv:2306.15006).** Swaps k-mer tokenization for **Byte-Pair Encoding (BPE)**, removing k-mer sample/compute inefficiencies, and ships the **GUE benchmark** (28 tasks). Relevant as a tokenization-effects study — how the segmentation of a symbolic stream changes what long-range structure a model can see.

**Evo / Evo2 (Nguyen et al. 2024; Brixi et al. 2025, Nature, doi:10.1038/s41586-026-10176-5).** Genome foundation models on **StripedHyena / StripedHyena 2** — *striped* hybrids interleaving Hyena long-convolution blocks with sparse attention, byte/single-nucleotide resolution, near-linear compute in length. Evo: 7 B params, ~300 B tokens; **Evo2: 1 M-token context, 9.3 T bp across all domains of life.** The hybrid conv+attention recipe is the current frontier for long-context genomics and a strong architecture template.

**Caduceus (Schiff et al. 2024, ICML; arXiv:2403.03234).** First **bi-directional, reverse-complement-equivariant Mamba** DNA model: BiMamba (Mamba on sequence + its reverse, tied weights) + MambaDNA RC-equivariance. On long-range variant-effect prediction it **beats models 10× larger**. Directly relevant: it is the genomics instantiation of the project's Transformer→Mamba swap (Result 11), showing selective-SSM inductive bias + a structural symmetry prior wins on long-range tasks.

**ESM-2 (Lin et al. 2023, Science; bioRxiv 2022.07.20.500902).** Protein masked-LM, 8 M→15 B params. Unsupervised **long-range contact prediction** from a sparse linear combination of attention heads; long-range contact precision rises 0.16→0.54 with scale. Shows attention maps *encode* the true long-range (3D-contact) dependency structure of the sequence — an interpretability handle: the model's internal correlation structure can be read out and compared to the data's.

**Length generalization in recurrent/SSM models (2024–2026).** Recent work ("Understanding and Improving Length Generalization in Recurrent Models," arXiv:2507.02782; long-context-generalization studies) finds SSMs **zero-shot extrapolate to 10–100× training length** while attention "collapses" beyond training length unless explicitly mitigated — the general-ML statement of the project's empirical Mamba-retention advantage, and a source of concrete mitigations (state-passing / unrolling tricks) to try.

---

## Methods to borrow (explicit, with mapping to β / γ / Hurst)

| Method | What it computes | Maps to project quantity | How to apply / port |
|---|---|---|---|
| **DFA (Peng 1994)** | scaling exponent α of detrended RMS fluctuation F(n) ∝ n^α | **Hurst H = α**; relates to correlation decay via β = 2 − 2α (for 0.5<α<1) | Embed token stream as a ±1 (or multi-level) walk, run DFA. Gives a *trend-robust* Hurst estimate on each (β,γ) cell's sequences; cross-check against the analytic β. |
| **MFDFA (Kantelhardt 2002)** | generalized Hurst h(q), singularity spectrum f(α), width Δα | mono- vs multi-fractality; Δα = a *new* heterogeneity axis beyond single β | Detects whether a (β,γ) cell's data is mono- or multi-fractal. Hypothesis: edge-of-chaos / max-effective-complexity cells have **maximal Δα** (richest multi-scale structure). Cheap to add, strong candidate diagnostic. |
| **Mutual-information decay M(d) (Li-Marr-Kaneko 1994; Grosse 2000)** | symbol-pair MI vs. separation d | **β directly** — M(d) ∝ d^−β is the token–token correlation decay | Compute M(d) on generated token streams; the fitted power-law exponent is an *empirical β* to validate the generator's nominal β and to measure β a model's *outputs* exhibit. Alphabet-native, no walk embedding. |
| **Block entropy Hₖ / entropy rate hₖ (Mantegna 1994; Schmitt-Herzel)** | Shannon entropy of length-k blocks; increments hₖ = Hₖ − Hₖ₋₁ | **γ directly** — hₖ ∼ k^−γ is conditional-entropy decay with context | hₖ is the model-free version of the project's Hₙ. Estimate γ from data without training a model; compare to a trained model's conditional-entropy curve to see if it recovers the true γ. Pairs naturally with the existing Rényi-D diagnostic. |
| **Rényi / generalized dimension Dq** | q-scaling of n-gram measure | already in `phase_core` (`renyi_D_rate_q*`) | Biology computes Dq from box-counting on the multifractal measure; the genome-analysis derivations justify and extend the project's existing Dq diagnostic and link it to MFDFA's f(α). |
| **Power spectrum 1/f^α (Voss 1992)** | low-frequency spectral slope | spectral dual of β (Wiener–Khinchin: α_spectral = 1 − β) | FFT of the embedded walk; fast cross-check of the correlation exponent and a detector of periodicity vs. true power-law structure. |
| **Zipf / Heaps (Mantegna 1994)** | rank-frequency & vocabulary-growth exponents | linguistic-complexity axes orthogonal to β/γ | Cheap "is this data language-like / structured?" sanity checks on token streams; distinguishes near-random (γ→1 noise) from richly structured cells. |
| **Fractal-globule P(s) ∝ s^−1 (Lieberman-Aiden 2009)** | contact-probability power law | β analogue from a *physical* generator; s^−1 vs s^−3/2 = structured vs. frozen | Conceptual template: an explicit self-similar generative process whose single exponent controls long-range dependency — and whose "fractal vs equilibrium" distinction parallels edge-of-chaos. |

**Practical note on estimators:** DFA, MFDFA, M(d), block-entropy, and power-spectrum all share the same finite-size-effect pitfalls flagged repeatedly in this literature (window-length bias, undersampling at large d/k, spurious crossovers). The genomics papers on finite-size effects are worth following for confidence intervals on β̂ and γ̂.

---

## Relevance to this project

1. **Validate the (β, γ) generator empirically.** The project currently *sets* β and γ. Biology supplies the *inverse estimators*: run M(d) → β̂ and block-entropy hₖ → γ̂ on `AlgorithmicKVGenerator` output to confirm the realized exponents match the nominal knobs, and to measure the β/γ a *trained model's generations* exhibit (a held-out diagnostic of what the model learned).

2. **A new edge-of-chaos / max-complexity axis: MFDFA singularity width Δα.** The project seeks the cell with maximum effective complexity. Δα from MFDFA is a principled, single-number multi-scale-heterogeneity measure. Testable hypothesis: the emergent strip / edge-of-chaos region coincides with **maximal Δα and maximal multifractality** — a data-side diagnostic complementing the existing Rényi-D (Result 8) and length-gen-ratio (Result 7) diagnostics, and requiring no model training.

3. **Length generalization is the genomics frontier and corroborates Result 11.** Caduceus (Bi-Mamba) beating 10×-larger models on long-range variant effect, plus SSMs' documented 10–100× zero-shot length extrapolation vs. attention collapse, independently support the project's finding that Mamba retains length-gen far better than the Transformer at the structured (Strip/CoT) cells. The hybrid conv+attention recipe (StripedHyena/Evo2) and RC-equivariance/state-passing tricks are concrete next architectures to test.

4. **Attention maps encode the dependency structure (ESM-2).** Suggests reading out the project Transformer's attention to check whether its internal correlation reach matches the data's β — a mechanistic test of *why* it fails to length-generalize at high β.

5. **The fractal globule grounds "structured-but-evolvable."** A physically real power-law system whose s^−1 (fractal) vs s^−3/2 (equilibrium) distinction mirrors the project's structured-vs-frozen contrast, lending biological credibility to the edge-of-chaos framing.

6. **The MI-expressivity bridge (Shen 2019 / L²M 2025) is the theory link.** It states the project's thesis in biology's language: data with power-law (not exponential) correlation decay requires architectures whose own MI scaling is non-exponential — i.e., β/γ structure *and* architecture choice (Transformer vs. Mamba) are two sides of one expressivity coin. This is the cleanest citation tying the data-theory and architecture halves of the project together.

---

## References

1. **Peng, C.-K., Buldyrev, S.V., Goldberger, A.L., Havlin, S., Sciortino, F., Simons, M., Stanley, H.E. (1992).** "Long-range correlations in nucleotide sequences." *Nature* 356:168–170. — First demonstration of power-law long-range correlations in DNA via the DNA-walk; H > 1/2 in non-coding DNA.

2. **Voss, R.F. (1992).** "Evolution of long-range fractal correlations and 1/f noise in DNA base sequences." *Phys. Rev. Lett.* 68:3805. — Power-spectrum 1/f^α analysis across species; spectral signature of fractal structure.

3. **Li, W., Kaneko, K. (1992).** "Long-range correlation and partial 1/f^α spectrum in a noncoding DNA sequence." *Europhys. Lett.* 17:655. — Linked long-range correlation to repetitive elements; α ≈ 0.5–0.85.

4. **Peng, C.-K., Buldyrev, S.V., Havlin, S., Simons, M., Stanley, H.E., Goldberger, A.L. (1994).** "Mosaic organization of DNA nucleotides." *Phys. Rev. E* 49:1685. — Introduces **Detrended Fluctuation Analysis (DFA)**; the canonical trend-robust Hurst estimator.

5. **Li, W., Marr, T.G., Kaneko, K. (1994).** "Understanding long-range correlations in DNA sequences." *Physica D* 75:392–416. — **Mutual-information-decay** method; power-law M(d) to ≥800 bp in non-coding DNA. Direct β estimator.

6. **Mantegna, R.N., Buldyrev, S.V., Goldberger, A.L., Havlin, S., Peng, C.-K., Simons, M., Stanley, H.E. (1994).** "Linguistic features of noncoding DNA sequences." *Phys. Rev. Lett.* 73:3169. — **Zipf law + block-entropy/redundancy** for DNA; noncoding more language-like than coding.

7. **Buldyrev, S.V., Goldberger, A.L., Havlin, S., et al. (1993–1998).** "Fractal landscapes and molecular evolution / Generalized Lévy-walk model for DNA." *Biophys. J.* and *Phys. Rev. E*. — Generative **Lévy-walk** model reproducing long-range correlations + biased subwalks in non-coding DNA.

8. **Grosse, I., Herzel, H., Buldyrev, S.V., Stanley, H.E. (2000).** "Species independence of mutual information in coding and noncoding DNA." *Phys. Rev. E* 61:5624. — Robust species-independent MI-decay methodology; coding vs noncoding contrast.

9. **Kantelhardt, J.W., Zschiegner, S.A., Koscielny-Bunde, E., Havlin, S., Bunde, A., Stanley, H.E. (2002).** "Multifractal detrended fluctuation analysis of nonstationary time series." *Physica A* 316:87–114. — Defines **MFDFA**, generalized Hurst h(q), singularity spectrum; the standard multifractal estimator now applied to genomes.

10. **Multifractal analysis of maize and soybean DNA (2024).** *PMC11082218.* — Recent MFDFA application showing genomes are multifractal with persistent long-range correlations; template for porting MFDFA to a new symbolic domain.

11. **Lieberman-Aiden, E., van Berkum, N.L., et al. (2009).** "Comprehensive mapping of long-range interactions reveals folding principles of the human genome." *Science* 326:289–293, doi:10.1126/science.1181369. — **Hi-C** + **fractal globule**; P(s) ∝ s^−1.08 power-law contact decay. Physical power-law generative model.

12. **Avsec, Ž., Agarwal, V., Visentin, D., et al. (2021).** "Effective gene expression prediction from sequence by integrating long-range interactions." *Nature Methods* 18:1196–1203, doi:10.1038/s41592-021-01252-x. — **Enformer**; transformer integrates ~100 kb enhancer context (200 kb receptive field).

13. **Nguyen, E., Poli, M., Faizi, M., et al. (2023).** "HyenaDNA: Long-range genomic sequence modeling at single nucleotide resolution." *NeurIPS 2023*; arXiv:2306.15794. — **Hyena** implicit long-convolution operator; up to **1 M-token** single-nucleotide context, sub-quadratic.

14. **Dalla-Torre, H., Gonzalez, L., Mendoza-Revilla, J., et al. (2023/2025).** "The Nucleotide Transformer: building and evaluating robust foundation models for human genomics." *Nature Methods.* — Multi-species DNA transformers; non-overlapping tokenization; 18-task benchmark.

15. **Zhou, Z., Ji, Y., Li, W., Dutta, P., Davuluri, R., Liu, H. (2023).** "DNABERT-2: Efficient foundation model and benchmark for multi-species genome." *ICLR 2024*; arXiv:2306.15006. — **BPE tokenization** for genomes + GUE 28-task benchmark.

16. **Nguyen, E., Poli, M., Durrant, M.G., et al. (2024).** "Sequence modeling and design from molecular to genome scale with Evo." *Science.* — **Evo**, 7 B-param **StripedHyena** genome model, byte-level, near-linear scaling.

17. **Brixi, G., Durrant, M.G., et al. (2025).** "Genome modelling and design across all domains of life with Evo 2." *Nature*, doi:10.1038/s41586-026-10176-5. — **Evo 2**: StripedHyena 2, **1 M-token context**, 9.3 T bp; hybrid conv+attention frontier.

18. **Schiff, Y., Kao, C.-H., Gokaslan, A., Dao, T., Gu, A., Kuleshov, V. (2024).** "Caduceus: Bi-directional equivariant long-range DNA sequence modeling." *ICML 2024*; arXiv:2403.03234. — **Bi-Mamba + RC-equivariance**; beats 10×-larger models on long-range variant effect. Genomics analogue of the project's Transformer→Mamba swap.

19. **Lin, Z., Akin, H., Rao, R., et al. (2023).** "Evolutionary-scale prediction of atomic-level protein structure with a language model (ESM-2)." *Science* 379:1123–1130; bioRxiv 2022.07.20.500902. — Protein LM; **unsupervised long-range contact prediction from attention heads**, precision 0.16→0.54 with scale.

20. **Shen, H. (2019).** "Mutual information scaling and expressive power of sequence models." arXiv:1905.04271. — Proves RNN/LSTM MI decays **exponentially** while attention captures **power-law** MI; the expressivity↔correlation-decay bridge. **Most directly relevant theory paper.**

21. **Chen, Z., et al. (2025).** "L²M: Mutual information scaling law for long-context language modeling." arXiv:2503.04725. — Modern MI-scaling law formalizing long-context capacity vs. power-law correlation; updates Shen 2019 for LLMs.

22. **(2025).** "Understanding and improving length generalization in recurrent models." arXiv:2507.02782. — SSMs zero-shot extrapolate 10–100× training length; attention collapses past training length without mitigation. General-ML statement of the project's Mamba length-gen result.

23. **Bernardi, G., et al. (1985; reviews to 2012).** Isochore theory — "The mosaic genome of warm-blooded vertebrates" and isochore-mapping work (e.g. *PLoS ONE* 2012; *PMC4662427* segmentation). — Long (>200 kb) compositional **isochore/mosaic** structure; multi-scale patchiness underlying long-range correlation.

24. **Long-range autocorrelations of CpG islands in the human genome.** *PMC3256200.* — CpG-island placement correlated to ~10 Mb; concrete multi-megabase long-range genomic structure.

25. **Schmitt, A.O., Herzel, H. (1997).** "Estimating the entropy of DNA sequences." *J. Theor. Biol.* 188:369–377. — **Block-entropy / entropy-rate** estimation with finite-sample corrections; the model-free γ estimator with explicit bias handling.
