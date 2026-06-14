# Physics & Statistical Mechanics of Long-Range Dependence, Fractals, Criticality, and the Renormalization Group — A Method-Focused Literature Review for the (β, γ) Data-Theory Project

> Scope: a cross-disciplinary, method-first survey of the physics of correlation decay, scaling exponents, criticality, multifractality, and the renormalization group (RG), curated to import concrete *measurement* and *interpretation* tools into the project's (β, γ) phase-diagram program. Web search **was available**; arXiv IDs / DOIs below were checked against live search results.

---

## Overview

The project (Cagnetta, Raventós, Ganguli & Wyart, arXiv:2602.07488) characterizes a text corpus by two power-law exponents and derives a data-limited scaling exponent:

- **β** — decay of the two-point token covariance, `||C(n)||_op ∝ n^{-β}` (spatial/temporal correlation decay along the sequence).
- **γ** — decay of the next-token conditional entropy with context length, `H_n − H_∞ ∝ n^{-γ}` (how fast the prediction problem "saturates" with horizon).
- **α_D = γ / 2β** — the predicted data-limited loss exponent, `L(P) − H_∞ ∝ P^{−γ/2β}`, via a data-dependent prediction horizon `n*(P) ∝ P^{1/2β}`.

These objects are *exactly* the quantities statistical mechanics has studied for 60 years under different names: β is a **correlation-function critical exponent** (η-like / Hurst-like); γ governs an **information/entropy-rate convergence**; the master-curve "scaling collapse" the paper observes is the canonical **finite-size-scaling (FSS) data collapse**; and the architecture-independence of γ is exactly the physics notion of a **universality class**. The paper itself flags this in its conclusions ("phase transitions are characterized by universality classes... which architectures and training dynamics give rise to the scaling exponent we propose?").

The transferable payoff of the physics literature is fourfold:
1. **Exponent estimators** that are robust to finite samples and trends (operator-norm decay fits, MFDFA, structure functions, wavelet leaders) — directly improve β and γ measurement.
2. **Finite-size scaling** — a mature theory of how observables depend on system size L, which maps *directly* onto **length generalization** (train length T → eval length 2048) and onto data-size P scaling.
3. **RG** — a principled theory of *what survives coarse-graining*, giving a mechanistic reading of n*(P), the horizon-limited regime, and α_D as an RG flow toward a fixed point.
4. **Edge-of-chaos / criticality** — quantitative criteria (Lyapunov exponent ≈ 0, diverging correlation length, peak in predictive information / effective complexity) for the "edge of chaos" region the project hunts for, where long-range dependence and maximal effective complexity coincide.

---

## Criticality, correlation length & finite-size scaling

At a continuous (second-order) phase transition the **correlation length ξ diverges**, `ξ ∝ |T − T_c|^{−ν}`, and the order-parameter correlation function loses its scale and becomes a pure power law, `G(r) ∝ r^{−(d−2+η)}` — the *only* functional form with no characteristic scale. The exponent η is the physics twin of β: both measure the algebraic decay of two-point correlations *at criticality*. Off-criticality, correlations decay exponentially with scale ξ; the appearance of a clean power-law decay in token correlations is itself a *signature of proximity to criticality*. Stanley's review (1999) and Cardy's textbook (1996) give the canonical accounts; the cluster of exponents (α, β_order, γ_susc, δ, ν, η) obey **scaling relations** (Fisher, Rushbrooke, Widom, hyperscaling) so that only two are independent — a strong hint that β and γ here may not be independent either, and that an α_D = γ/2β-type relation is the analog of a scaling law.

**Finite-size scaling (FSS)** is the single most transferable idea. On a finite system of linear size L, ξ cannot exceed L, so the true divergence is *rounded and shifted*; any observable obeys
`P(t, L) = L^{ρ/ν} · Q(t · L^{1/ν})`,
i.e. curves for different L **collapse** onto one master curve Q when axes are rescaled by powers of L. Binder's Monte-Carlo FSS methodology (1981) and the Binder cumulant give a recipe to *extract critical points and exponents from systems too small to be critical* — exactly the regime of a model trained at length 512 but evaluated at 2048. The mapping is: **L ↔ sequence/context length T (or dataset size P)**; **the collapse exponent ↔ β (or α_D)**; **failure of collapse ↔ failure of length generalization**. The project's observed `L_n(P)` collapse under `P → P/n^{2β}`, `L_n → n^γ L_n` *is* a two-variable FSS collapse, and FSS theory tells you how to read off and error-bar the exponents, and how to detect corrections-to-scaling (the concave-in-log(β) deviations seen in `main_findings.md` are textbook *corrections to scaling*).

---

## Renormalization group (and RG ↔ deep learning)

The **renormalization group** (Kadanoff block-spin 1966; Wilson 1971–75, Nobel 1982) explains *why* power laws and universality appear: coarse-graining (integrate out short-scale degrees of freedom, rescale) defines a flow in the space of Hamiltonians. **Fixed points** of this flow are scale-invariant (critical) theories; the *relevant* eigen-directions of the linearized flow give the critical exponents (ν from the thermal eigenvalue, etc.), and the basin of attraction of a fixed point is a **universality class** — microscopically different systems flow to the same fixed point and share exponents. This is the precise sense in which the project's claim "γ is architecture-independent" is a universality statement.

The RG ↔ deep learning bridge is a live research program and the richest source of mechanistic analogy for α_D and n*(P):

- **Mehta & Schwab (2014)** construct an *exact mapping* between Kadanoff's variational (real-space) RG and stacked Restricted Boltzmann Machines: successive layers ≈ successive RG coarse-graining steps that distill *relevant* features. This is the founding analogy: **depth ↔ scale**, **learned representation ↔ RG-relevant operators**.
- **Koch-Janusz & Ringel (2018, Nature Physics)** invert the idea: maximize a **real-space mutual information** objective to *learn* the optimal coarse-graining and recover Ising exponents — operationalizing "which degrees of freedom carry the long-range information," a direct tool for asking *which tokens at separation n carry the slow, predictive information* (i.e. a constructive version of n*(P)).
- **Lenggenhager et al. / Gökmen et al. — "Optimal RG transformation from information theory" (PRX 2020)** prove information-theoretic optimality properties of such coarse-grainings (does not lengthen interaction range), giving guarantees relevant to coarse-graining sequences.
- **Roberts, Yaida & Hanin (2022), *The Principles of Deep Learning Theory*** develop an effective-field-theory / RG flow over network *depth*, treating layer-to-layer evolution as an RG and predicting feature-learning and criticality-at-initialization conditions.
- **Lin, Tegmark & Rolnick (2017), "Why does deep and cheap learning work so well?"** argue hierarchical/Markov structure in data (a generative RG-like hierarchy) is *why* shallow-cheap deep nets succeed — directly relevant to the latent-hierarchy view the source paper cites (Cagnetta et al. context-free grammars).

**Reading for the project:** n*(P) ∝ P^{1/2β} is an RG length scale that grows as you "add data/resolution"; α_D = γ/2β is the rate at which the loss flows toward the H_∞ fixed point. The horizon-limited regime is "RG-relevant directions dominate"; the within-horizon excess is "irrelevant operators." Distinct architectures sharing γ but differing in prefactor = same fixed point, different bare couplings.

---

## Multifractals & turbulence

A single exponent (Hurst H, or β) assumes **monofractal** scaling. Real long-range-correlated signals are often **multifractal**: scaling exponents depend on the moment order q. The toolkit:

- **Structure functions** `S_q(ℓ) = ⟨|x(t+ℓ) − x(t)|^q⟩ ∝ ℓ^{ζ(q)}`. For Kolmogorov 1941 turbulence ζ(q) = q/3 is linear; **intermittency** makes ζ(q) *concave* (Frisch, *Turbulence*, 1995; Kolmogorov 1962 lognormal correction; She–Lévêque 1994). Concavity of ζ(q) ⇔ multifractality ⇔ fat-tailed, bursty fluctuations.
- **Multifractal spectrum f(α)** (Legendre transform of (q−1)D_q, the generalized/Rényi dimensions) — width of f(α) measures the *degree* of multifractality. The project already computes Rényi dimensions D_q of n-gram statistics (`main_findings.md` Result 8); the f(α) width is the natural single-number "how multifractal / how structured is this corpus" knob, and could be a cleaner edge-of-chaos diagnostic than D_{q=1} alone.
- **MFDFA — Multifractal Detrended Fluctuation Analysis** (Kantelhardt et al. 2002) is the workhorse for *nonstationary* series: build the profile (cumulative sum), segment, locally detrend (remove polynomial trends), compute the q-dependent fluctuation function `F_q(s) ∝ s^{h(q)}`, and read the **generalized Hurst exponent h(q)**. h(2) recovers the standard Hurst exponent (β = 2H−1 for the correlation exponent of the increments). MFDFA is robust to trends that contaminate naive correlation-function fits — a likely improvement over the operator-norm power-law fit currently used for β, especially in the broken-power-law WikiText case the source paper noted. Mature implementations exist (Python `MFDFA`, R `MFDFA`), so this is low-cost to adopt.
- **Wavelet leaders / WTMM** (Muzy, Bacry, Arneodo) — an alternative multifractal estimator with better statistical properties than box-counting.

---

## 1/f noise & self-organized criticality

**1/f (flicker) noise** — power spectral density `S(f) ∝ 1/f^a`, a ≈ 1 — is ubiquitous and is the frequency-domain face of long-range temporal correlations (a = 1 − β-type relations connect spectral slope, autocorrelation decay, and Hurst exponent via the Wiener–Khinchin theorem). It signals the absence of a single characteristic timescale — exactly the "long-range dependence" the project wants its edge-of-chaos data to have. **Practical tool:** a 1/f power spectrum of the token/embedding sequence is an independent, FFT-cheap estimator of correlation decay, complementary to β from `C(n)`; the spectral exponent and the correlation exponent must be consistent, providing a cross-check.

**Self-organized criticality (SOC)** — Bak, Tang & Wiesenfeld (1987, PRL) — shows that *driven dissipative* systems (the sandpile) can flow to a critical state *without parameter tuning*, producing scale-free avalanches (power-law size/duration distributions) and 1/f noise as generic, robust outputs. The conceptual import: a "good" edge-of-chaos data regime might be an *attractor* of a generative process rather than a fine-tuned point — relevant to designing data generators that sit at criticality by construction (the project's `AlgorithmicKVGenerator` β-knob is essentially a tunable-criticality dial; SOC suggests generators that self-tune). Jensen's *Self-Organized Criticality* (1998) is the standard monograph; Sornette's *Critical Phenomena in Natural Sciences* (2006) is the best single cross-disciplinary toolbox (power-law fitting, fractals, FSS, SOC, heavy tails) for someone importing these methods.

---

## Edge of chaos / dynamical criticality

The "edge of chaos" is *dynamical* criticality: the boundary between an **ordered/frozen phase** (perturbations die — no memory, can't transmit information) and a **chaotic phase** (perturbations explode — memory corrupted). Computation is maximized *at the boundary*.

- **Langton (1990), "Computation at the edge of chaos"** — cellular automata; introduced the λ parameter and showed information transmission, storage, and modification peak near the order/chaos transition.
- **Bertschinger & Natschläger (2004, Neural Computation)** — recurrent threshold networks perform complex real-time computation *only near the critical line*; quantified via a separation/Lyapunov criterion. Foundational for **reservoir computing at criticality**.
- **Lyapunov exponent λ_max** is the order parameter: λ_max < 0 ordered, λ_max > 0 chaotic, **λ_max ≈ 0 at the edge** ("edge of stability"). Maximal memory capacity and mutual information peak there (Mutual Information & Edge of Chaos in Reservoir Computers, arXiv:1906.03186; Boedecker et al.). This gives the project a *model-side* criticality probe: estimate the effective Lyapunov / Jacobian sensitivity of the trained Transformer's hidden state to input perturbations; the edge-of-chaos data region should produce λ_max ≈ 0 dynamics.
- **Criticality in living/neural systems** — Mora & Bialek (2011), "Are biological systems poised at criticality?", and Tkačik et al. (2015, PNAS), "Thermodynamics and signatures of criticality in a network of neurons," tie criticality to maximum-entropy models, **Zipf's law**, and a *peak in predictive information / specific heat*. Schwab, Nemenman & Mehta (2014, PRL) caution that Zipf's law and apparent criticality can arise from latent variables "without fine-tuning" — an important null hypothesis when interpreting power laws in token statistics. Beggs & Plenz (2003) neuronal avalanches give the canonical experimental SOC signature.
- **Maximum effective complexity** — predictive information / excess entropy (Crutchfield & Feldman 2003; Bialek, Nemenman & Tishby 2001) is maximized near criticality, formalizing the project's "maximum effective complexity" target: complexity = the **mutual information between past and future** of the sequence, which diverges (sub-extensively) exactly when correlations decay as a slow power law.

---

## Methods to borrow (mapped to β / γ / α_D / length-gen)

| Physics method | What it measures | Maps to | Why adopt it |
|---|---|---|---|
| **Finite-size scaling + data collapse** (Binder; Cardy) | observables vs system size L, exponent + master curve | **length generalization (T) and data scaling (P)**; α_D and β as collapse exponents | Turns "does it length-generalize?" into "do eval-length curves collapse under L^{1/ν} rescaling?" Gives error bars + corrections-to-scaling. The project's `L_n(P)` collapse already *is* an FSS collapse. |
| **Corrections-to-scaling** (Wegner) | sub-leading `(1 + b L^{−ω})` terms | the **concave-in-log(β)** deviation in `main_findings.md` | Explains why a single linear/power law fails at the edges; principled way to fit the curvature instead of going piecewise. |
| **MFDFA / generalized Hurst h(q)** (Kantelhardt 2002) | q-dependent scaling of nonstationary series | robust **β** (β = 2H−1) and multifractal width | Detrending beats naive `C(n)` fits when there are trends / broken power laws (the WikiText case). |
| **Structure functions ζ(q) + f(α) spectrum** (Frisch; turbulence) | moment-order-dependent exponents | upgrade D_q (Result 8) to full **multifractal spectrum**; "effective complexity" via f(α) width | Single-number "how structured / how intermittent is this corpus," cleaner edge-of-chaos diagnostic. |
| **1/f power-spectrum slope** (Wiener–Khinchin) | spectral exponent a | cross-check on **β** | FFT-cheap independent estimator; consistency test against `C(n)`-derived β. |
| **RG flow / fixed points** (Wilson; Mehta–Schwab; Koch-Janusz–Ringel) | what survives coarse-graining; relevant vs irrelevant | mechanistic reading of **n*(P)**, **α_D**, universality of **γ** | α_D = flow rate to H_∞ fixed point; n*(P) = emergent RG length; same-γ-different-prefactor = same universality class. Real-space MI directly identifies which tokens carry slow predictive info. |
| **Predictive information / excess entropy** (Bialek–Nemenman–Tishby; Crutchfield–Feldman) | mutual info between past & future | **maximum effective complexity** target; relation to γ | Formal definition of the project's "max complexity" region; peaks at criticality. |
| **Lyapunov exponent ≈ 0 criterion** (Bertschinger–Natschläger) | dynamical order/chaos boundary | **model-side edge-of-chaos** probe | Measure hidden-state sensitivity of the trained Transformer; edge-of-chaos data ⇒ λ_max ≈ 0. |
| **Scaling relations** (Fisher/Widom/hyperscaling) | only 2 of 6 exponents independent | are **β and γ independent**? | Suggests searching for a constraint linking β, γ — would strengthen α_D = γ/2β into a derived scaling law, not just a measured ratio. |
| **Binder cumulant / crossing point** | dimensionless ratio that crosses at T_c | locating the **phase boundary γ*(β)** without compute at the boundary | A dimensionless, size-independent crossing observable could pin γ*(β) more cleanly than the train_acc < 0.20 cutoff. |

---

## Relevance to this project

1. **Length generalization is finite-size scaling.** This is the headline transfer. Training at T=512 and evaluating at 1024/2048 is exactly probing a system below vs above its correlation length. Reframe the length-gen ratio `r = acc(2048)/acc(512)` as an FSS-collapse test: if the data is at criticality (slow power-law correlations, β small enough that ξ ≳ T), curves at different eval-lengths should *collapse* under the appropriate `L^{1/ν}` rescaling, and the collapse exponent is predictable from β. Failure to collapse ⇒ the relevant correlations live beyond the trained context (ξ > T) ⇒ length-gen breaks. This gives a *principled* explanation for the observed `r ≈ 0.39` emergent-strip value and the Mamba-vs-Transformer gap (Result 11): an architecture with longer effective ξ (state-space memory) collapses better.

2. **α_D = γ/2β is a scaling-relation / RG-flow rate.** The physics framing predicts α_D should be *universal* (architecture-independent) within a universality class — testable, and consistent with the source paper's architecture-independence of γ. Search for a scaling relation linking β and γ (analogous to hyperscaling) would be a genuine theoretical contribution.

3. **Edge of chaos = the project's target region, made quantitative.** "Long-range dependence + maximum effective complexity" is precisely the critical point: ξ → large (small β), predictive information / excess entropy peaks, Lyapunov ≈ 0. The project can locate this region with *three independent criticality probes* — (a) data-side: multifractal spectrum width / predictive information peak; (b) model-side: hidden-state Lyapunov ≈ 0; (c) FSS: best length-gen collapse. Convergence of all three is strong evidence of true criticality (cf. the project's existing multi-diagnostic philosophy).

4. **Better β/γ estimators.** Replace/augment the naive operator-norm power-law fit with MFDFA (handles nonstationarity, broken power laws) and a 1/f spectral cross-check; upgrade Rényi D_q (already computed) to the full f(α) multifractal spectrum as the "effective complexity" axis.

5. **Caveat — apparent criticality without fine-tuning.** Schwab–Nemenman–Mehta and the SOC/latent-variable literature warn that Zipf laws and power-law correlations can be artifacts of latent heterogeneity, not genuine criticality. The project should adopt their null tests before claiming its edge-of-chaos region is "truly critical."

---

## References

1. **Cagnetta, F., Raventós, A., Ganguli, S., Wyart, M. (2026).** *Deriving neural scaling laws from the statistics of natural language.* arXiv:2602.07488. — The source paper: defines β (token-correlation decay), γ (conditional-entropy decay), and α_D = γ/2β; observes scaling collapse; explicitly invokes physics universality classes.
2. **Stanley, H. E. (1999).** *Scaling, universality, and renormalization: Three pillars of modern critical phenomena.* Rev. Mod. Phys. 71, S358. — Canonical pedagogical review of critical exponents, scaling laws, universality.
3. **Cardy, J. (1996).** *Scaling and Renormalization in Statistical Physics.* Cambridge Univ. Press. — Standard textbook: correlation functions, exponents, RG, finite-size scaling.
4. **Kadanoff, L. P. (1966).** *Scaling laws for Ising models near T_c.* Physics 2, 263. — Block-spin construction; intuition for RG and universality.
5. **Wilson, K. G. (1975).** *The renormalization group: Critical phenomena and the Kondo problem.* Rev. Mod. Phys. 47, 773. — Foundational RG; fixed points, relevant/irrelevant operators, universality classes.
6. **Binder, K. (1981).** *Finite size scaling analysis of Ising model block distribution functions.* Z. Phys. B 43, 119. — FSS methodology + Binder cumulant; extracting exponents from finite systems (↔ length generalization).
7. **Fisher, M. E. & Barber, M. N. (1972).** *Scaling theory for finite-size effects in the critical region.* Phys. Rev. Lett. 28, 1516. — Original finite-size scaling theory.
8. **Mehta, P. & Schwab, D. J. (2014).** *An exact mapping between the Variational Renormalization Group and Deep Learning.* arXiv:1410.3831. — Stacked RBMs ≡ Kadanoff real-space RG; depth ↔ scale, representation ↔ relevant operators.
9. **Koch-Janusz, M. & Ringel, Z. (2018).** *Mutual information, neural networks and the renormalization group.* Nature Physics 14, 578. arXiv:1704.06279. — Learn optimal coarse-graining by maximizing real-space mutual information; recovers Ising exponents.
10. **Gökmen, D. E., Ringel, Z., Huber, S. D., Koch-Janusz, M. (2020).** *Optimal Renormalization Group Transformation from Information Theory.* Phys. Rev. X 10, 011037. arXiv:1809.09632. — Information-theoretic optimality of RG coarse-graining.
11. **Roberts, D. A., Yaida, S. & Hanin, B. (2022).** *The Principles of Deep Learning Theory.* Cambridge Univ. Press. arXiv:2106.10165. — Effective-field-theory / RG-over-depth treatment of neural networks; criticality at initialization.
12. **Lin, H. W., Tegmark, M. & Rolnick, D. (2017).** *Why does deep and cheap learning work so well?* J. Stat. Phys. 168, 1223. arXiv:1608.08225. — Hierarchical/Markov (RG-like) data structure explains deep-net efficiency.
13. **Mehta, P. et al. (2019).** *A high-bias, low-variance introduction to machine learning for physicists.* Physics Reports 810, 1. arXiv:1803.08823. — Bridges statistical-mechanics and ML language; RG/energy-landscape framing.
14. **Frisch, U. (1995).** *Turbulence: The Legacy of A. N. Kolmogorov.* Cambridge Univ. Press. — Structure functions, K41 vs intermittency, multifractal model of turbulence.
15. **Kolmogorov, A. N. (1941; 1962).** *Local structure of turbulence...* / *Refinement (lognormal intermittency).* Dokl. Akad. Nauk SSSR. — Origin of structure-function scaling ζ(q) and intermittency corrections.
16. **She, Z.-S. & Lévêque, E. (1994).** *Universal scaling laws in fully developed turbulence.* Phys. Rev. Lett. 72, 336. — Modern intermittency model; concave ζ(q).
17. **Kantelhardt, J. W. et al. (2002).** *Multifractal detrended fluctuation analysis of nonstationary time series.* Physica A 316, 87. — MFDFA algorithm; generalized Hurst exponent h(q); the workhorse multifractal estimator.
18. **Peng, C.-K. et al. (1994).** *Mosaic organization of DNA nucleotides.* Phys. Rev. E 49, 1685. — Detrended fluctuation analysis (DFA), the monofractal precursor; long-range correlations in symbolic sequences (directly analogous to token sequences).
19. **Muzy, J. F., Bacry, E. & Arneodo, A. (1991).** *Wavelet transform modulus-maxima (WTMM) for multifractal analysis.* Phys. Rev. Lett. 67, 3515. — Wavelet-based multifractal spectrum estimator.
20. **Bak, P., Tang, C. & Wiesenfeld, K. (1987).** *Self-organized criticality: An explanation of 1/f noise.* Phys. Rev. Lett. 59, 381. — SOC; scale-free avalanches and 1/f noise as generic critical outputs without fine-tuning.
21. **Jensen, H. J. (1998).** *Self-Organized Criticality.* Cambridge Univ. Press. — Standard SOC monograph.
22. **Sornette, D. (2006).** *Critical Phenomena in Natural Sciences: Chaos, Fractals, Self-organization and Disorder* (2nd ed.). Springer. — Best cross-disciplinary toolbox: power-law fitting, fractals, FSS, SOC, heavy tails.
23. **Langton, C. G. (1990).** *Computation at the edge of chaos: Phase transitions and emergent computation.* Physica D 42, 12. — λ parameter; information processing peaks at the order/chaos boundary.
24. **Bertschinger, N. & Natschläger, T. (2004).** *Real-time computation at the edge of chaos in recurrent neural networks.* Neural Computation 16, 1413. — Complex computation only near criticality; Lyapunov/separation criterion; reservoir computing.
25. **Boedecker, J., Obst, O., Lizier, J. T., Mayer, N. M. & Asada, M. (2012).** *Information processing in echo state networks at the edge of chaos.* Theory in Biosciences 131, 205. — Memory capacity and mutual information peak at λ_max ≈ 0.
26. **Mora, T. & Bialek, W. (2011).** *Are biological systems poised at criticality?* J. Stat. Phys. 144, 268. — Maximum-entropy models, Zipf's law, specific-heat peak as criticality signatures.
27. **Tkačik, G. et al. (2015).** *Thermodynamics and signatures of criticality in a network of neurons.* PNAS 112, 11508. — Operational thermodynamic criticality diagnostics for high-dim data.
28. **Schwab, D. J., Nemenman, I. & Mehta, P. (2014).** *Zipf's law and criticality in multivariate data without fine-tuning.* Phys. Rev. Lett. 113, 068102. arXiv:1310.0448. — Null hypothesis: latent variables can mimic criticality/Zipf — essential caveat.
29. **Bialek, W., Nemenman, I. & Tishby, N. (2001).** *Predictability, complexity, and learning.* Neural Computation 13, 2409. — Predictive information / excess entropy as a complexity measure; sub-extensive divergence ⇔ long-range structure (the project's "max effective complexity").
30. **Crutchfield, J. P. & Feldman, D. P. (2003).** *Regularities unseen, randomness observed: Levels of entropy convergence.* Chaos 13, 25. — Entropy-rate convergence h(n) → h_∞ (the γ exponent's information-theoretic home); excess entropy = predictive information.
31. **Beggs, J. M. & Plenz, D. (2003).** *Neuronal avalanches in neocortical circuits.* J. Neuroscience 23, 11167. — Experimental SOC signature (power-law avalanche sizes) in neural systems.
32. **Muñoz, M. A. (2018).** *Colloquium: Criticality and dynamical scaling in living systems.* Rev. Mod. Phys. 90, 031001. arXiv:1712.04499. — Modern synthesis of criticality, edge of chaos, and dynamical scaling across living systems.
