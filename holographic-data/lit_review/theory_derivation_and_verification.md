# Theory Derivation and Citation Verification: α_D = γ/(2β), Finite-Size Scaling ↔ Length Generalization, and Architecture (δ) Dependence

*Compiled 2026-06-14 for the (β, γ) phase-diagram / edge-of-chaos project.*
*Web search WAS available. All five core citations were verified live against arXiv abstract pages and the HTML full text where reachable. The exact derivation of Eq. (8) below was extracted from the arXiv HTML of 2602.07488v1.*

---

## Citation verification table

| Claim / role in project | As-cited in prompt | Verified? | Corrected / confirmed reference |
|---|---|---|---|
| Source of α_D = γ/(2β) | "Cagnetta/Raventós/Ganguli/Wyart, *Deriving neural scaling laws from the statistics of natural language*, arXiv:2602.07488 (ID may be placeholder)" | ✅ **CONFIRMED** — ID, authors, and core claim all correct | Francesco **Cagnetta**, Allan **Raventós**, Surya **Ganguli**, Matthieu **Wyart**, *Deriving Neural Scaling Laws from the statistics of natural language*, **arXiv:2602.07488** (2026). Core claim verified: derives data-limited scaling exponent **ℒ_AR(P) − H_∞ ≍ P^(−γ/(2β))** (their Eq. 8) with **no free parameters**; matched against GPT-2 / LLaMA-style models on TinyStories and WikiText. The arXiv ID is real, not a placeholder. |
| Epiplexity | "arXiv:2601.03220? confirm" | ✅ **CONFIRMED** | Marc **Finzi** et al. (incl. J. Zico **Kolter**, Andrew Gordon **Wilson**), *From Entropy to Epiplexity: Rethinking Information for Computationally Bounded Intelligence*, **arXiv:2601.03220** (Jan 2026). Core claim verified: introduces *epiplexity*, a compute-bounded information measure; language data has far higher epiplexity per token than images (CIFAR-5M), rationalizing text-pretraining transfer. Note: this is **not** itself a γ/(2β) paper; it is the complexity/measure companion. |
| L²M mutual-information scaling | "arXiv:2503.04725" | ✅ **CONFIRMED** | Zhuo **Chen** et al., *L²M: Mutual Information Scaling Law for Long-Context Language Modeling*, **arXiv:2503.04725** (submitted 6 Mar 2025; NeurIPS 2025). Core claim verified: bipartite mutual information grows as a **power law I ∼ L^β** in natural language (PG19), distinct from two-point MI; "L²M condition" lower-bounds required history-state (KV/SSM) growth; validated on Transformers and SSMs. |
| Random Hierarchy Model | "Cagnetta–Wyart, arXiv:2307.02129" | ⚠️ **CONFIRMED with author correction** | **arXiv:2307.02129** is correct, but the byline is **not** "Cagnetta–Wyart" alone: Francesco **Cagnetta, Leonardo Petrini, Umberto M. Tomasini, Alessandro Favero, Matthieu Wyart**, *How Deep Neural Networks Learn Compositional Data: The Random Hierarchy Model*, arXiv:2307.02129 (2023); published **Phys. Rev. X 14, 031001 (2024)**. The analytic power-law *correlation* result the project leans on is in the companion **Cagnetta & Wyart, arXiv:2406.00048** (NeurIPS 2024) — that is the genuinely two-author paper. Use 2307.02129 for the RHM definition + sample complexity, and 2406.00048 for the analytic token-correlation power law. |
| Maloney–Roberts–Sully solvable model | "arXiv:2210.16859" | ✅ **CONFIRMED** | Alexander **Maloney**, Daniel A. **Roberts**, James **Sully**, *A Solvable Model of Neural Scaling Laws*, **arXiv:2210.16859** (30 Oct 2022). Core claim verified: a power law in dataset *spectrum* (random-feature covariance) is mapped through a nonlinear random-feature map into a power-law test-loss scaling, with a plateau set by the finite extent of the spectral power law. |

**Net verdict:** 4 of 5 fully clean. The only correction is the RHM byline (it is a five-author PRX paper, not "Cagnetta–Wyart"); the two-author "Cagnetta–Wyart" paper the project means for the *analytic correlation exponent* is **2406.00048**, a distinct and also-real reference. No fabricated IDs.

---

## Deriving α_D = γ/(2β) (step by step)

The source paper (2602.07488) frames autoregressive loss at context length T as a sum over per-position excess losses, then shows the exponent is set entirely by two data statistics. The derivation has three moving parts.

**(1) The two data exponents.**
- **Conditional-entropy decay (γ).** Define H_n = next-token conditional entropy given the previous n tokens, and H_∞ its asymptotic floor (the irreducible entropy rate). The paper's Eq. (6):
  > H_n − H_∞ ≍ n^(−γ).

  This is the *value* of knowing more context: the predictable-information gain from extending the conditioning window from n to ∞ shrinks as a power law with exponent γ. (This is the differential form of the Hilberg/Dębowski excess-entropy statement.)
- **Correlation decay (β).** Let C(n) be the cross-covariance (correlation) matrix between tokens n apart. Eq. (7):
  > ‖C(n)‖_op ≍ n^(−β).

  This is the *detectability* of a dependence at separation n: the signal strength of an order-n correlation falls off as n^(−β).

**(2) The dataset-limited prediction horizon n*(P).**
With a finite training set of P tokens, an order-n correlation can only be *estimated* up to statistical noise. The empirical correlation estimator has noise of order O(P^(−1/2)) (standard error of a sample covariance). A correlation at separation n is **learnable** only when its true signal exceeds that noise floor:
$$\|C(n)\| \gtrsim P^{-1/2} \;\;\Longleftrightarrow\;\; \|C(n)\|^2 \gtrsim 1/P.$$
The paper's Eq. (26) writes the sample size needed to resolve separation n as
> P_{n*} ≡ c² / ‖C(n)‖²_op.

Substituting ‖C(n)‖_op ≍ n^(−β):
$$P_{n^*} \;\asymp\; n^{2\beta} \quad\Longrightarrow\quad n^*(P) \;\asymp\; P^{1/(2\beta)}.$$

**This is where the factor of 2 comes from.** It is **not** a kinetics/feature-learning factor — it is the **square in a signal-to-noise comparison**: noise scales as P^(−1/2), signal as n^(−β), and matching *signal²* to *variance* (equivalently |C| to P^(−1/2)) puts a square on the correlation, turning β into 2β. Detection of a weak correlation is variance-limited, and variance is the square of the amplitude.

**(3) Loss = conditional entropy at the achievable horizon.**
The model can exploit context only out to n*(P); beyond that, correlations are buried in sampling noise. Under the paper's assumption that within-horizon excess losses ℰ_n(P) decay "sufficiently fast" and that the trained context T ≫ n*(P), the residual loss above the entropy floor is dominated by the conditional entropy *at the horizon* — i.e. the value of the context the model could not yet resolve:
$$\mathcal{L}_{AR}(P) - H_\infty \;\asymp\; H_{n^*(P)} - H_\infty \;\asymp\; \big(n^*(P)\big)^{-\gamma} \;\asymp\; \big(P^{1/(2\beta)}\big)^{-\gamma} \;=\; P^{-\gamma/(2\beta)}.$$
Hence **α_D = γ/(2β)** (Eq. 8).

**The hidden assumption that buries δ.** The step "loss ≈ H at the horizon" presupposes that *once a correlation at separation n ≤ n*(P) is statistically resolvable, the model actually learns it* — the **fast within-horizon learning** assumption. All optimization/representational/architectural difficulty is assumed to vanish: the model is treated as a perfect statistical estimator that saturates the data-detectability bound. Any architecture-dependent slowdown (a Transformer needing depth/heads to compose an order-n dependency, an SSM's finite state truncating it) lives precisely in the gap between "resolvable in the data" and "learned by *this* model." That gap is the project's δ. So α_D = γ/(2β) is the **data-optimal** (architecture-free) exponent; the realized exponent is α_D − (architecture penalty), and the project's δ is exactly the leakage term the derivation assumes away.

---

## Random Hierarchy Model route

The α_D derivation takes β and γ as *measured* inputs. The RHM (2307.02129; analytic correlations in 2406.00048) is the route to deriving them **from first principles** for a generative grammar, closing the loop.

- **Generative model.** The RHM is an ensemble of probabilistic context-free grammars: a class label expands through L levels of composition rules, each non-terminal choosing among m equivalent "synonym" production rules over a vocabulary of v symbols with branching factor s. This is a controlled, tunable instance of the hierarchical-source mechanism that Lin–Tegmark (2017) identified as the *generic origin of power-law correlations* (regular grammars give exponential decay; context-free/hierarchical grammars give power laws).
- **Power-law correlations are a theorem, not a fit.** In the RHM, token–token correlations can be computed **analytically** and decay as a **power law in separation** (2406.00048), with the exponent set by the grammar parameters (multiplicity m, vocabulary v, branching s). This is exactly a first-principles **β**.
- **Finite data ⇒ effective correlation range.** A finite training set resolves the grammar's hidden variables only up to a depth/separation that **grows with dataset size** — the same "data-limited horizon" idea as n*(P), but derived inside a generative model rather than assumed. This makes the RHM the natural sandbox to *test whether the realized scaling exponent equals γ/(2β)* with both exponents known in closed form, and to measure the architecture penalty δ directly (CNNs matched to the generative locality scale faster than Transformers — Sclocchi–Favero–Wyart 2025, arXiv:2505.07070).
- **Tie to this project.** The KV-retrieval generator's p(d) ∝ d^(−(β+1)) is a *shallow, single-level* analogue of the RHM's deep recursion: both synthesize data with a tunable correlation exponent. The RHM is the route to upgrade the project from "we set β by hand" to "β and γ both follow from a generative grammar, and α_D = γ/(2β) is a falsifiable prediction with no free knobs."

---

## Finite-size scaling ↔ length generalization (formalized)

Borrow the statistical-physics finite-size-scaling (FSS) apparatus. Near a critical point a system has a **correlation length ξ** that diverges; observables on a system of finite size L collapse onto a universal curve when plotted against the scaling variable L/ξ.

**Mapping.** For a sequence model, the "system size" is the context window, and the role of ξ is played by the **data-limited horizon / effective correlation range**. Two distinct lengths matter:
- **Train length T** — the horizon over which the model *was optimized* to compose dependencies.
- **Test length T′** (e.g. 2T) — the horizon at which we *evaluate*.

Length generalization is then literally the FSS question: **does an observable measured at size T predict the same observable at size T′?** Formalize the "train at T, test at 2T" protocol as probing the data **below vs. above the correlation length ξ**:

- If **ξ ≪ T** (data has a finite memory shorter than the train window): the model has seen all the relevant correlation structure during training. Test at 2T probes only separations already represented; the length-generalization ratio r = acc(2T)/acc(T) → 1, and curves at different T **collapse**.
- If **ξ ≳ T** (power-law / critical data, no finite memory — the edge-of-chaos strip): the train window *truncates* a correlation structure that genuinely extends to 2T. Test at 2T probes separations the model never resolved; r < 1, and **collapse fails** unless plotted against the scaling variable T/ξ (or T·n*^{−1}).

**Collapse exponent.** Under the data-optimal picture, the resolvable horizon scales as n* ∼ P^{1/(2β)} in *data*, but along the *length* axis the controlling exponent is β itself: the excess loss above the floor at evaluated length ℓ should scale as ℒ(ℓ) − H_∞ ∼ ℓ^{−γ} once ℓ exceeds what training resolved, so the natural collapse variable is **ℓ/T** with the curves rescaled by T^{−γ} (equivalently, plot (ℒ(ℓ)−H_∞)·T^{γ} vs ℓ/T). The correlation/length exponent governing *where* collapse breaks is β (it sets ξ and n*); the *magnitude* of the residual is governed by γ. A clean collapse of r(T, ℓ) vs ℓ/T across several train lengths T is the falsifiable signature that the data sits in the power-law regime; a sharp, T-independent r means ξ is finite (off the critical strip).

**When collapse fails.** (i) Off-critical data (finite ξ): trivially collapses to r ≈ 1, no scaling. (ii) Architecture-limited (δ large): the model never reached the data-optimal horizon even at training length, so the *model's* effective ξ_model < ξ_data and the collapse variable must use ξ_model, not the data ξ — the failure mode the project sees as Transformer r ≈ 0.39 (broken) vs Mamba r ≈ 1.0 (collapsed) at the strip (main_findings Result 11).

---

## Where δ (architecture) enters

δ is the **architecture-dependent boundary slope** — the term that separates the realized scaling/length-generalization exponent from the data-optimal α_D = γ/(2β).

1. **In the derivation:** δ lives in the "fast within-horizon learning" assumption. The data permits resolving separations up to n*(P) ∼ P^{1/(2β)}; whether the model *realizes* that horizon depends on whether its inductive bias can cheaply compose order-n dependencies. Write the realized horizon as n*_model(P) ∼ P^{1/(2β + δ)} (or a multiplicative penalty); δ = 0 recovers the data-optimal bound.

2. **L²M makes δ concrete (2503.04725):** to model dependencies out to length ℓ, a model's **history state must grow at least as fast as I ∼ ℓ^β**. An architecture whose state *cannot* grow that fast is hard-capped:
   - **Fixed-window / vanilla Transformer:** attention is O(T²) but the *usable* composition depth is bounded by layer count; beyond the train window the positional regime is unseen → r collapses (the project's r ≈ 0.39).
   - **RoPE Transformer:** length generalization is gated by rotary-frequency extrapolation; δ is dominated by the positional encoding's out-of-distribution behavior, not by the data exponent. Isolating RoPE vs NoPE vs ALiBi at *fixed* (β, γ) data measures the *positional* component of δ.
   - **Mamba / SSM:** a recurrent state that grows (or is content-selective) can in principle satisfy the L²M condition, giving r ≈ 1 (project Result 11: Mamba r = 1.001 vs Transformer r = 0.385 at the strip). The state dimension d_state caps the realizable β before truncation.

3. **How a Transformer/Mamba/RoPE comparison *isolates* δ:** hold the data generator fixed (same β, γ, same n-gram statistics — the project already verifies train_acc at L=512 is matched across architectures, so the *fitting* of in-window structure is identical) and vary **only** the architecture. Any difference in the length-generalization ratio r(2T/T) or in the realized scaling exponent is then attributable to δ alone, because α_D = γ/(2β) is architecture-free by construction. This is exactly the project's controlled design: the architectural effect is "essentially all in length-generalization, not in fitting the training-length distribution" (Result 11 notes). The slope of r vs (β, γ) for each architecture *is* δ(architecture); the gap between the Mamba curve and the data-optimal r = 1 measures residual SSM truncation, and the gap between Transformer and Mamba measures the state-growth penalty.

---

## Open theoretical questions

1. **Does α_D = γ/(2β) survive when δ ≠ 0?** The source derivation gives the data-optimal exponent. Is the realized exponent α_D − f(δ), and is f additive in the denominator (P^{1/(2β+δ)}) or multiplicative? The RHM with controlled architectures is the cleanest testbed.
2. **Reconciling with the project's Result 2 (no single α* threshold).** main_findings.md *refutes* a single α_theory = γ/(2β) threshold as the phase boundary in the KV task. Is this because (a) the KV generator is single-level (not the deep hierarchy the derivation assumes), (b) δ varies across the (β, γ) plane and dominates the boundary, or (c) the boundary is genuinely a soft crossover (their finding) rather than a critical line? Distinguishing these is the central theory question for the paper.
3. **Is the factor of 2 robust to estimator choice?** It comes from O(P^{−1/2}) sample-covariance noise. Feature-learning / kernel regimes (Maloney–Roberts–Sully, Bahri et al.) can change the effective noise scaling and thus the 2. Does a feature-learning correction shift 2β → (1+a)β?
4. **What sets ξ_model vs ξ_data?** A first-principles expression for an architecture's effective correlation length (state dim, depth, positional scheme) would turn δ from a measured slope into a predicted one and directly predict the Mamba-vs-Transformer r gap.
5. **Epiplexity as the right complexity scalar for the strip.** Does the edge-of-chaos / maximal-effective-complexity strip coincide with maximal *epiplexity* (compute-bounded extractable structure, 2601.03220) rather than maximal Shannon excess entropy? This would connect the project's "maximal effective complexity" hypothesis to a defined measure.

---

## Most promising next step

**Run the RHM (or the KV generator extended to multi-level hierarchy) with β and γ both known in closed form, sweep architecture (Transformer / RoPE-Transformer / Mamba) at fixed data, and test whether the realized data-limited scaling exponent equals γ/(2β) and whether the length-generalization collapse variable is ℓ/T with residual ∼ T^{−γ}.** This simultaneously (a) validates the source paper's formula in a setting where it should hold exactly, (b) *measures* δ as the per-architecture deviation from the data-optimal exponent, and (c) tests whether the project's soft boundary (Result 2) is a δ-driven artifact or a genuine crossover — turning the empirical (β, γ) phase diagram into a derivation-grounded one.

---

## References

1. **Cagnetta, F., Raventós, A., Ganguli, S. & Wyart, M. (2026).** *Deriving Neural Scaling Laws from the statistics of natural language.* arXiv:2602.07488. — Source of α_D = γ/(2β) (Eq. 8); H_n − H_∞ ≍ n^{−γ} (Eq. 6); ‖C(n)‖_op ≍ n^{−β} (Eq. 7); P_{n*} ≡ c²/‖C(n)‖²_op (Eq. 26) ⇒ n*(P) ≍ P^{1/(2β)}.
2. **Finzi, M., Kolter, J.Z., Wilson, A.G. et al. (2026).** *From Entropy to Epiplexity: Rethinking Information for Computationally Bounded Intelligence.* arXiv:2601.03220. — Compute-bounded information measure; language ≫ images in epiplexity per token.
3. **Chen, Z. et al. (2025).** *L²M: Mutual Information Scaling Law for Long-Context Language Modeling.* arXiv:2503.04725 (NeurIPS 2025). — Bipartite MI ∼ L^β; L²M condition lower-bounds history-state growth; the data→architecture-cost link that localizes δ.
4. **Cagnetta, F., Petrini, L., Tomasini, U.M., Favero, A. & Wyart, M. (2023/2024).** *How Deep Neural Networks Learn Compositional Data: The Random Hierarchy Model.* arXiv:2307.02129; Phys. Rev. X 14, 031001 (2024). — RHM definition + sample complexity (note: five-author PRX paper, not "Cagnetta–Wyart").
5. **Cagnetta, F. & Wyart, M. (2024).** *Towards a theory of how the structure of language is acquired by deep neural networks.* arXiv:2406.00048 (NeurIPS 2024). — Analytic power-law token correlations in a PCFG; finite data ⇒ effective correlation range growing with dataset size. The genuine "Cagnetta–Wyart" first-principles β.
6. **Maloney, A., Roberts, D.A. & Sully, J. (2022).** *A Solvable Model of Neural Scaling Laws.* arXiv:2210.16859. — Spectral data power law → test-loss power law through a random-feature map; plateau from finite spectral extent.
7. **Sclocchi, A., Favero, A. & Wyart, M. (2025).** *Scaling Laws and Representation Learning in Simple Hierarchical Languages: Transformers vs. Convolutional Architectures.* arXiv:2505.07070. — Architecture-matched-to-locality scales faster; direct measurement of an architecture penalty (δ analogue).
8. **Lin, H.W. & Tegmark, M. (2017).** *Critical Behavior in Physics and Probabilistic Formal Languages.* Entropy 19(7):299; arXiv:1606.06737. — Hierarchical grammars ⇒ power-law correlations (origin of β); edge-of-chaos framing.
