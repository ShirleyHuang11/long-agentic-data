# A Unified Information-Theoretic Framework for Length Generalization and Holographic Data

*Synthesis draft — information-theory / scaling-law / logical-depth lens for the (β, γ) phase-diagram program.*

*Status: derivation draft. Grounds entirely in the repo's read-first materials and the two readable PDFs (`gamma-beta.pdf` = Cagnetta–Raventós–Ganguli–Wyart arXiv:2602.07488, readable; `epiplexity.pdf` = Finzi et al. arXiv:2601.03220, readable). Conjectural extensions beyond those sources are flagged.*

---

## 0. Notation and objects

Let a stationary source emit tokens $x_1, x_2, \dots$ over a finite alphabet $\mathcal{V}$, $|\mathcal V| = V$. Write $x_{1:n} = (x_1,\dots,x_n)$. All logarithms base 2 (bits).

| symbol | object | role |
|---|---|---|
| $H_n$ | next-token conditional entropy given $n$ tokens of context | data, γ-axis |
| $H_\infty$ | entropy rate $\lim_{n\to\infty} H_n$ | irreducible floor |
| $\gamma$ | conditional-entropy convergence exponent | data |
| $C(n)$ | lag-$n$ token cross-covariance matrix | data, β-axis |
| $\beta$ | correlation / bipartite-MI decay exponent | data |
| $E$ | excess entropy $=I(\text{past};\text{future})$ | data complexity |
| $I_{\mathrm{pred}}(\ell)$ | predictive information of an $\ell$-window | data complexity |
| $D$ | logical depth / epiplexity (deep-computation measure) | data |
| $P$ | training tokens (dataset size) | resource |
| $T$ | train context length; $\ell$ test/eval length | resource |
| $S(\ell)$ | model history-state capacity at length $\ell$ (bits) | architecture |
| $\delta$ | architecture penalty exponent | architecture |
| $\alpha_D$ | data-limited loss exponent | derived |

We treat $\gamma, \beta, D$ as **data** statistics, $S(\cdot), \delta$ as **architecture** statistics, and $P, T, \ell$ as **resources**.

---

## 1. Definitions and the clean relations among $H_\infty$, $\gamma$, $\beta$

### 1.1 The entropy-rate convergence exponent $\gamma$

Define the conditional entropy of the next token given $n$ tokens of context,
$$
H_n \;=\; H(x_{n+1}\mid x_{1:n}) \;=\; -\,\mathbb{E}\!\left[\log P(x_{n+1}\mid x_{1:n})\right],
\tag{1}
$$
which is the finite-$n$ entropy-rate estimate. $H_n$ is non-increasing and converges to the entropy rate $H_\infty$. The **convergence exponent** $\gamma$ is defined by the power law (Cagnetta et al. Eq. 6; Ebeling–Pöschel; Hilberg):
$$
\boxed{\,H_n - H_\infty \;\sim\; A\,n^{-\gamma}, \qquad \gamma > 0.\,}
\tag{2}
$$
$H_n - H_\infty$ is the *marginal predictable-information gain* from extending context $n\to\infty$. Large $\gamma$: context value vanishes quickly (shallow predictive structure). Small $\gamma$: distant context keeps paying off (deep predictive structure).

### 1.2 Excess entropy $E$ and the **finite-iff-$\gamma>1$** threshold

The **excess entropy** (= predictive information = past–future mutual information; Crutchfield–Feldman, Bialek–Nemenman–Tishby) is the total sub-extensive correction:
$$
E \;=\; \sum_{n\ge 1}\big(H_n - H_\infty\big) \;=\; I(\overleftarrow{X};\overrightarrow{X}).
\tag{3}
$$
Substituting (2), $E \sim A\sum_n n^{-\gamma}$ is a $p$-series, giving the sharp dichotomy:
$$
\boxed{\;E < \infty \iff \gamma > 1 \qquad\text{(weak LRD)},\qquad E = \infty \iff \gamma \le 1 \quad\text{(strong / Hilberg LRD)}.\;}
\tag{4}
$$
This is the model-free location of the weak/strong long-range-dependence boundary. The growth of the predictive information of a finite window is the partial sum,
$$
I_{\mathrm{pred}}(\ell) \;=\; \sum_{n=1}^{\ell}\big(H_n - H_\infty\big)
\;\sim\;
\begin{cases}
E - \dfrac{A}{\gamma-1}\,\ell^{-(\gamma-1)} \to E, & \gamma > 1,\\[2mm]
A\log \ell, & \gamma = 1,\\[2mm]
\dfrac{A}{1-\gamma}\,\ell^{\,1-\gamma}, & 0<\gamma<1.
\end{cases}
\tag{5}
$$
$I_{\mathrm{pred}}(\ell)$ is the **predictive-information budget** an ideal predictor must hold to be optimal at horizon $\ell$ — the central quantity for length generalization (§3).

### 1.3 The correlation / bipartite-MI exponent $\beta$

Let $C(n)_{\mu\nu} = P(x_i{=}\mu, x_{i+n}{=}\nu) - P(x_i{=}\mu)P(x_{i+n}{=}\nu)$ be the lag-$n$ covariance. The **correlation exponent** $\beta$ is (Cagnetta et al. Eq. 7; Lin–Tegmark MI form):
$$
\boxed{\,\|C(n)\|_{\mathrm{op}} \;\sim\; B\,n^{-\beta}, \qquad \beta > 0.\,}
\tag{6}
$$
The two-point mutual information obeys $I(x_i;x_{i+n}) \sim n^{-\beta}$ for weak coupling. The **bipartite** (block-block) mutual information of L²M (Chen et al.) is the integrated cousin: $I_{\mathrm{bi}}(\ell) \sim \ell^{\,\beta}$ when $\beta<1$ (non-summable two-point MI). Note $I_{\mathrm{bi}}$ and $I_{\mathrm{pred}}$ are the same family of quantity viewed two ways; we use $I_{\mathrm{pred}}$ from (5) as the canonical state-requirement (5)↔(L²M) below.

**Origin of power laws (not exponentials).** Markov / regular sources give *exponential* decay in (2),(6); power-law $\gamma,\beta$ require recursive / context-free / hierarchical generators (Lin–Tegmark; RHM). This is why holographic data must be generated by deep recursion, not a finite Markov chain.

### 1.4 Logical depth $D$ / epiplexity — the "deep computation in short context" measure

Kolmogorov complexity $K(x)$ measures the *length* of the shortest program for $x$; it cannot see that a string is the product of long computation. **Bennett's logical depth** measures exactly that:
$$
D_t(x) \;=\; \min\{\,t(p) : |p| \le K(x)+c,\; U(p)=x\,\},
\tag{7}
$$
the running time of (near-)shortest programs. Both random strings ($K\approx|x|$, trivial program, shallow) and trivial strings (constant program, shallow) have *low* depth; only organized-but-compressed strings are deep. **Holographic data** = data that is short in $K$ but large in $D$: a short prompt that is the compressed trace of a deep computation.

**Epiplexity** (Finzi et al., `epiplexity.pdf` Def. 8) is the compute-bounded operationalization: the structural information a computationally bounded observer extracts to its weights, excluding *time-bounded entropy* (apparent randomness with no learnable structure — PRNGs, chaos). Its practical estimator is the **area under the loss curve above the final loss**,
$$
\boxed{\;D \;\approx\; \mathrm{Epx} \;=\; \int_0^{\infty}\!\big(\mathcal L_t - \mathcal L_\infty\big)\,dt
\quad\text{(equivalently cumulative teacher$\to$student KL).}\;}
\tag{8}
$$
This is the rigorous version of the repo's "maximum effective complexity," and the operationalization of logical depth used downstream. Note $\mathrm{Epx}$ is *not* $E$: $E$ is Shannon predictive information (compute-unbounded); $D\approx\mathrm{Epx}$ is what a bounded learner can actually convert into reusable structure. The repo's AULC diagnostic (Result 10: emergent cells have *higher* AULC) is, up to normalization, a measurement of (8).

---

## 2. Master learnability / length-generalization equation

### 2.1 Data-limited loss (the $\alpha_D = \gamma/2\beta$ law)

With $P$ training tokens, a lag-$n$ correlation of true strength $\|C(n)\|$ is detectable only above the sample-covariance noise floor $O(P^{-1/2})$. Matching *signal²* to *variance* (the source of the factor 2 — it is a variance-limited detection, not kinetics):
$$
\|C(n^*)\|^2 \;\gtrsim\; \frac{1}{P}
\;\Longrightarrow\;
P_{n^*} \asymp \frac{c^2}{\|C(n^*)\|_{\mathrm{op}}^2} \asymp (n^*)^{2\beta}
\;\Longrightarrow\;
\boxed{\,n^*(P) \;\asymp\; P^{1/(2\beta)}.\,}
\tag{9}
$$
$n^*(P)$ is the **data-limited prediction horizon**: the largest lag whose correlation is resolvable from $P$ tokens. The residual loss is the conditional entropy *at* that horizon (within-horizon structure is learned, beyond-horizon structure is buried in noise):
$$
\mathcal L_{\mathrm{AR}}(P) - H_\infty \;\asymp\; H_{n^*(P)} - H_\infty \;\asymp\; \big(n^*(P)\big)^{-\gamma} \;=\; P^{-\gamma/(2\beta)},
$$
$$
\boxed{\;\mathcal L_{\mathrm{AR}}(P) - H_\infty \;\sim\; P^{-\alpha_D},\qquad \alpha_D = \frac{\gamma}{2\beta}.\;}
\tag{10}
$$
(Cagnetta et al. Eq. 8.) Equation (10) is the **data-optimal** exponent: it assumes the model *learns* every within-horizon correlation (the "fast within-horizon learning" assumption). That assumption is where architecture enters — §4.

### 2.2 The information-theoretic length-generalization condition

Train context $T$, test horizon $\ell \gg T$. An ideal predictor optimal at $\ell$ must *hold* the predictive information of an $\ell$-window in its state across the autoregressive roll-out. The **state-capacity condition** for length generalization is:
$$
\boxed{\;S(\ell) \;\ge\; I_{\mathrm{pred}}(\ell)\;\;\big(\sim I_{\mathrm{bi}}(\ell)\big)\qquad\text{for all }\ell \le \ell_{\max}.\;}
\tag{11}
$$
This is the repo's reading of the **L²M condition**: a model's history-state size must grow at least as fast as the data's bipartite mutual information. Using (5):

- **Weak LRD ($\gamma>1$):** $I_{\mathrm{pred}}(\ell)\to E<\infty$. A *finite* state $S=E$ suffices for arbitrary $\ell$. **Length generalization is information-theoretically possible with a fixed-state model.** This is the regime where extrapolation to $\ell \gg T$ is feasible in principle.
- **Strong LRD ($\gamma\le 1$):** $I_{\mathrm{pred}}(\ell)$ diverges ($\log\ell$ at $\gamma=1$, $\ell^{1-\gamma}$ below). **No fixed-state model can be optimal at all $\ell$**; the required state grows without bound. A growing cache (attention) is mandatory; even it must pay $\Omega(I_{\mathrm{bi}}(\ell))$.

The **length-generalization ratio** $r(\ell) = \mathrm{acc}(\ell)/\mathrm{acc}(T)$ degrades once the test horizon exceeds what training resolved, $\ell > n^*(P_{\mathrm{train}})$ and/or $\ell$ exceeds the state's reach. The excess loss above floor at evaluated length $\ell$ collapses as
$$
\big(\mathcal L(\ell) - H_\infty\big)\cdot T^{\gamma} \;=\; \Phi(\ell/T),
\tag{12}
$$
a finite-size-scaling data collapse with collapse variable $\ell/T$, residual scale $T^{-\gamma}$, and the *location* of breakdown set by $\beta$ (which fixes $n^*$ and the correlation length $\xi$). Clean collapse ⇒ power-law (critical) data; sharp $T$-independent $r$ ⇒ finite $\xi$ (off-critical).

---

## 3. Architecture term $\delta$

$\delta$ is the gap between **data-resolvable** information and **model-realized** information. Two equivalent encodings:

**(a) Horizon penalty (exponent form).** The data permits horizon $n^*\sim P^{1/2\beta}$; a given architecture realizes only
$$
n^*_{\mathrm{model}}(P) \;\sim\; P^{1/(2\beta + \delta)},
\qquad
\alpha_D^{\mathrm{realized}} \;=\; \frac{\gamma}{2\beta+\delta},
\tag{13}
$$
with $\delta=0$ recovering the data-optimal bound (10). $\delta>0$ is any architecture-dependent slowdown in composing an order-$n$ dependency (Transformer depth/head budget; SSM finite state). $\delta$ lives precisely in the "fast within-horizon learning" assumption the derivation (10) makes silently.

**(b) State-capacity penalty (L²M form).** Define the architecture's effective correlation length $\xi_{\mathrm{model}}$ as the largest $\ell$ satisfying the capacity condition (11):
$$
\xi_{\mathrm{model}} \;=\; \sup\{\,\ell : S(\ell) \ge I_{\mathrm{pred}}(\ell)\,\}.
\tag{14}
$$
- **Fixed-state (Mamba / SSM / linear attention):** $S(\ell) = S_0$ constant (state dim $d_{\mathrm{state}}$). Then $\xi_{\mathrm{model}}$ is *finite* whenever $I_{\mathrm{pred}}$ diverges (strong LRD), and is $\infty$ when $\gamma>1$ provided $S_0 \ge E$. Capacity caps the realizable $\beta$ before state truncation.
- **Growing-cache (attention):** $S(\ell) = O(\ell)$ (KV cache) can meet $I_{\mathrm{bi}}(\ell)\sim\ell^\beta$ for $\beta\le 1$; $\xi_{\mathrm{model}}$ limited instead by positional extrapolation (RoPE/ALiBi/NoPE) and depth, not by state size.

The collapse variable in (12) must use $\min(\xi_{\mathrm{data}}, \xi_{\mathrm{model}})$. The repo's Result 11 (Mamba $r=1.001$ vs Transformer $r=0.385$ at the strip) is read as: at the strip the data is *weak enough* in effective LRD that Mamba's fixed state satisfies (11) and length-generalizes ($\xi_{\mathrm{model}}\ge\ell$), while the vanilla Transformer's *positional* $\delta$ (unseen positions beyond $T$) dominates and breaks $r$. A clean Transformer/Mamba/RoPE comparison at *fixed* $(\beta,\gamma)$ isolates $\delta$ because (10) is architecture-free by construction; any difference in $r$ or in the realized exponent (13) is $\delta$ alone.

---

## 4. Holographic / edge-of-chaos design principle

### 4.1 The optimization

Design data to maximize length generalization subject to learnability. Formalize as a constrained program over $(\beta,\gamma,D)$ for a fixed architecture (given $S(\cdot),\delta$) and budgets $(P,T)$:
$$
\boxed{\;
\max_{\beta,\gamma,D}\;\; r_{\infty} \;=\; \lim_{\ell\to\infty}\frac{\mathrm{acc}(\ell)}{\mathrm{acc}(T)}
\quad\text{s.t.}\quad
\underbrace{S(\ell)\ge I_{\mathrm{pred}}(\ell)\;\forall \ell}_{\text{(11) length-gen feasible}},\;\;
\underbrace{\alpha_D^{\mathrm{realized}}=\tfrac{\gamma}{2\beta+\delta}>0}_{\text{learnable in }P},\;\;
\underbrace{D \ge D_{\min}}_{\text{nontrivial}}.
\;}
\tag{15}
$$

### 4.2 The predicted optimum (tension between two forces)

Two opposing pressures act on $\gamma$:
1. **Learnability / state feasibility wants large $\gamma$:** by (4)–(5), $\gamma>1$ makes $E$ finite, so a fixed-state model *can* length-generalize (condition (11) satisfiable with finite $S$). Larger $\gamma$ also raises $\alpha_D$ (faster learning per token).
2. **Useful long-range content wants small $\gamma$ and small $\beta$:** a target that is too predictable ($\gamma\to\infty$, $H_n\to H_\infty$ immediately) is Markov — there is *nothing long-range to generalize*, and depth $D\to 0$. Holography requires distant context to matter.

The reconciliation is that the design optimum is **not** at maximal raw long-range dependence (small $\gamma$, the naive "edge of chaos"). It sits at the **feasibility frontier** — the largest predictive-information content that the model's state can still hold:
$$
\boxed{\;\gamma^\star \;\approx\; 1^{+}\quad(\text{excess entropy } E \text{ just finite}),\qquad
\beta^\star \;=\; \sup\{\beta : S(\ell)\ge I_{\mathrm{bi}}(\ell)\}\;\text{(largest learnable }\beta).\;}
\tag{16}
$$
$\gamma^\star\approx 1^+$ keeps $I_{\mathrm{pred}}(\ell)$ as large as possible (logarithmically near-divergent, maximal $D$/epiplexity per (8)) while still bounded so a finite-state model is feasible. This is the operative restatement of "maximum effective complexity": **peak compute-bounded structure (epiplexity), not peak Shannon LRD.**

### 4.3 Reconciliation with the REFUTED edge-of-chaos H1

The repo refuted (H1) the claim that *minimal* $\beta$ / classic edge-of-chaos ($\beta\approx0.05$) maximizes length generalization (`holo_length_gen.md`: ordering Natural ≈ Abyss > Edge > CoT; no ridge; Mamba advantage *collapses* at Edge-of-Chaos, $r=0.653\pm0.235$). The framework explains *why*:

- At $\beta\to 0$ (uniform retrieval) the correlation is **non-localizable** and $I_{\mathrm{bi}}(\ell)$ grows fastest — condition (11) is *violated* for any finite state, so $\xi_{\mathrm{model}}<\ell$ and $r$ collapses. There is structure but it is **not learnable / not holdable**. Depth $D$ is high in $E$ but low in epiplexity (8) — apparent complexity the bounded learner cannot extract (exactly Finzi et al.'s *time-bounded entropy*).
- The refuted H1 conflated $E$ (Shannon) with $D$ (epiplexity). The genuine optimum (16) is at *moderate, learnable* $\beta$ and $\gamma\approx1^+$, consistent with the empirical "Natural / Strip" optimum and the literature's "higher-$\beta$ locality helps length-gen" (RASP-L; Bhojanapalli). **Edge-of-chaos in the naive small-$\beta$ sense is refuted; edge-of-*feasibility* (16) replaces it.**

So the role of logical depth $D$ is to *correct* the objective: maximize **extractable** structure (epiplexity, $D$) not raw predictive information ($E$). Data that is deep-but-incompressible-to-a-bounded-learner (small $\beta$, chaos edge) looks complex by $E$ but is useless for generalization — the refutation in one line.

---

## 5. Falsifiable predictions and measurement recipes

### 5.1 Predictions

- **P-A (feasibility optimum).** Length-gen $r_\infty$ is maximized near $\gamma\approx1^+$ at the largest $\beta$ satisfying (11), **not** at minimal $\beta$. Sweeping $\gamma$ across 1 should show $r$ peaking just above $\gamma=1$ and falling for $\gamma\le1$ (state overflow) and $\gamma\gg1$ (no long-range content). *Refutes H1's small-$\beta$ ridge; testable on the KV/RHM generator with measured $\gamma$.*
- **P-B (state-capacity collapse).** For a fixed-state model, $r(\ell)\to$ const iff $\gamma>1$ (finite $E\le S_0$) and $\to 0$ once $\gamma\le1$ or $S_0<E$. The transition occurs at $S_0 = E$ — increasing $d_{\mathrm{state}}$ should push the breakdown $\gamma$ downward. *Quantitative L²M test.*
- **P-C (FSS collapse).** Excess loss obeys (12): $(\mathcal L(\ell)-H_\infty)T^\gamma = \Phi(\ell/T)$ collapses across train lengths $T$ for power-law data, with breakdown located by $\beta$. Failure of collapse ⇒ $\xi$ finite (off-critical) or $\delta$-dominated ($\xi_{\mathrm{model}}<\xi_{\mathrm{data}}$).
- **P-D (epiplexity ↔ generalization).** The cell maximizing epiplexity (8) (AULC above final loss) coincides with the cell maximizing $r_\infty$, and *both* differ from the cell maximizing Shannon $E$. *Directly tests $D$ vs $E$ as the right objective; the repo's Result 10 AULC data is a first measurement.*
- **P-E (architecture $\delta$).** At fixed $(\beta,\gamma)$, $\alpha_D^{\mathrm{realized}}=\gamma/(2\beta+\delta)$ with $\delta_{\mathrm{Mamba}}>\delta_{\mathrm{attn}}$ in strong-LRD cells and $\delta\approx0$ for both in weak-LRD cells. The realized-vs-data exponent gap *is* $\delta$.

### 5.2 Measurement recipes (zero training cost for data-side quantities)

| quantity | recipe | pitfalls |
|---|---|---|
| $\gamma$ | fit $H_n-H_\infty\sim n^{-\gamma}$ from **neural-LM cross-entropy $\mathcal L_n(P)$ vs context $n$** (well-trained, $P$-converged); cross-check block entropy $h_\mu(n)=H(n){-}H(n{-}1)$ + CTW | undertrained LM inflates $H_n$; plug-in undersamples for $n$ large; verify $P$-convergence first |
| $\beta$ | fit $\|C(n)\|_{\mathrm{op}}\sim n^{-\beta}$ from lag-$n$ token covariance top singular value; cross-check two-point MI $I(x_i;x_{i+n})$ | noise floor $O(P^{-1/2})$ caps usable $n$; broken power laws — fit short-lag stage; NSB/Miller–Madow bias correction for MI |
| $E$, $I_{\mathrm{pred}}(\ell)$ | $E=\sum_n(H_n-H_\infty)$ from the $\gamma$-curve; finite iff $\gamma>1$ | dominated by small-$n$ terms; tail sensitivity near $\gamma=1$ |
| $D$ (epiplexity) | **area under loss curve above final loss** (8), or cumulative teacher→student KL; the repo's `aulc_*` column | normalize by steps; needs converged $\mathcal L_\infty$; separates structure from time-bounded entropy |
| $\alpha_D$ / collapse | rescale $\mathcal L_n\!\to\! n^\gamma\mathcal L_n$, $P\!\to\!P/n^{2\beta}$ → master curve; for length, $T^\gamma(\mathcal L(\ell)-H_\infty)$ vs $\ell/T$ | collapse failure diagnoses $\delta$ or finite $\xi$ |
| $S(\ell)$, $\delta$ | data-side $I_{\mathrm{bi}}(\ell)\sim\ell^\beta$ vs model state bits; $\delta$ = realized minus data-optimal exponent at fixed $(\beta,\gamma)$ | hold generator fixed, vary only architecture |

---

## 6. Assumptions and limits

1. **Stationarity & single power law.** (2),(6) assume stationary sources with clean (un-broken) power laws. Real corpora show broken power laws (WikiText short/long-lag regimes); fit the prediction-relevant stage. Latent-variable mixing can fake power laws (Schwab–Nemenman–Mehta) — run surrogate nulls before claiming criticality.
2. **The factor-2 / variance-limited detection.** (9) assumes $O(P^{-1/2})$ sample-covariance noise (variance-limited regime). Feature-learning / kernel regimes can shift $2\beta\to(1+a)\beta$; the exponent in (10) is regime-dependent.
3. **$\delta$ functional form is conjectural.** Whether $\delta$ enters additively in the denominator (13) or multiplicatively is not derived here; the RHM with controlled architectures is the testbed. $\xi_{\mathrm{model}}$ in (14) is defined operationally, not yet predicted from first principles (open: express $\xi_{\mathrm{model}}$ from $d_{\mathrm{state}}$, depth, positional scheme).
4. **$E$ vs $D$ identification.** Equating the design objective with epiplexity (8) rather than Shannon $E$ is a *hypothesis* (P-D), motivated by Finzi et al. and the H1 refutation, not proven. The $\gamma^\star\approx1^+$ optimum (16) is a prediction of this hypothesis.
5. **Expressivity ceiling.** Both attention and SSMs are TC⁰ (Illusion of State; Parallelism Tradeoff). The Mamba length-gen advantage is memory-retention of structured retrieval, **not** a state-tracking capability gain. The framework governs *information retention*, not computational class.
6. **Single-level vs deep hierarchy.** The repo's KV generator is single-level ($p(d)\propto d^{-(\beta+1)}$); $\alpha_D=\gamma/2\beta$ is derived for deep hierarchies (RHM). The repo's Result 2 (no single $\alpha^\star$ threshold) may reflect (a) single-level data, (b) $\delta$ varying across the plane, or (c) a genuine soft crossover — distinguishing these is the central open question.
7. **Synthetic ↔ natural gap.** All quantitative claims are for the controlled generator; transfer to natural language (where $\gamma\approx0.3$, $\beta\approx0.9$ per Cagnetta et al., i.e. weak LRD, $E$ finite) is a separate validation.

---

## Appendix: one-screen summary

- **Master scaling:** $\displaystyle \mathcal L_{\mathrm{AR}}(P)-H_\infty \sim P^{-\alpha_D}$, $\alpha_D=\dfrac{\gamma}{2\beta+\delta}$ (data-optimal $\delta{=}0$).
- **Length-gen condition:** $S(\ell)\ge I_{\mathrm{pred}}(\ell)$; feasible with finite state iff $\gamma>1$ (since $E=\sum(H_n-H_\infty)<\infty\iff\gamma>1$).
- **$\delta$ enters** as the data-resolvable↔model-realized gap: horizon $P^{1/(2\beta+\delta)}$ and state reach $\xi_{\mathrm{model}}=\sup\{\ell:S(\ell)\ge I_{\mathrm{pred}}(\ell)\}$.
- **Logical depth $D$ / epiplexity** corrects the objective: maximize extractable structure (AUC-above-final-loss), not Shannon $E$; this is what makes holographic (short-context-deep-computation) data *learnable* vs merely complex.
- **Design optimum:** $\gamma^\star\approx1^+$ (excess entropy just finite, epiplexity near-maximal), $\beta^\star$ = largest state-feasible $\beta$ — the *edge of feasibility*, replacing the refuted small-$\beta$ edge-of-chaos.
