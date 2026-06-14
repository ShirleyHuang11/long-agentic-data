# A Unified Framework for Holographic Data & Length Generalization

**Goal.** Predict — and *design data for* — length generalization: a model trained at
context length $T$ that must perform at context $\ell \gg T$. The framework ties the
data's structural exponents ($\beta,\gamma$), its information content ($H_\infty$,
logical depth $D$), the model architecture ($\delta$), and the length-generalization
outcome into one scaling picture. It is the synthesis of a 12-section cross-disciplinary
review (`literature_review.md`); two independent derivations (physics/RG and
information-theory) agree on every core equation. Rigor is labeled per result.

---

## 1. Primitives (measurable properties of the data)

| symbol | definition | how to measure |
|---|---|---|
| $\beta$ | token–token correlation decay, $\lVert C(n)\rVert_{\mathrm{op}}\sim n^{-\beta}$ | top singular value of lag-$n$ covariance vs $n$ |
| $\gamma$ | conditional-entropy decay, $H_n-H_\infty\sim n^{-\gamma}$ | neural-LM cross-entropy $\mathcal L_n(P)$ at varying context |
| $H_\infty$ | irreducible entropy rate | extrapolated floor of $\mathcal L_n$ |
| $E$ | excess entropy $=\sum_n (H_n-H_\infty)$ | **finite $\iff \gamma>1$** (weak/strong-LRD divide) |
| $H$ | Hurst exponent (near-independent axis) | **DFA/MFDFA** (replaces R/S) |
| $D$ | logical depth / **epiplexity** = extractable structure | area under the loss curve above $H_\infty$ (Finzi et al., 2601.03220) |

These are *not* new quantities: $\beta$ is a critical correlation exponent, $\gamma$ the
Hilberg/entropy-rate-convergence exponent, $E$ the predictive-information divide — see
`literature_review.md`.

## 2. Core scaling law (rigorous; Cagnetta–Raventós–Ganguli–Wyart, arXiv:2602.07488)

**Resolvable horizon.** With $P$ training tokens, correlations are detectable only above
the $O(P^{-1/2})$ sample-covariance noise floor; matching signal $\lVert C(n)\rVert^2$ to
$1/P$ gives the data-resolvable horizon

$$\boxed{\;\xi(P) \;=\; n^*(P)\;\sim\;P^{1/(2\beta)}\;}$$

(the factor **2** is the square of a *variance-limited detection* comparison, not kinetics.)

**Master collapse.** Loss at horizon $n$ obeys a finite-size-scaling form with
correlation-length exponent $\nu=1/(2\beta)$:

$$\boxed{\;\big(\mathcal L_n(P)-H_\infty\big)\,n^{\gamma} \;=\; \ell\!\big(P\,n^{-2\beta}\big)\;}$$

Evaluated at the resolvable horizon $n=\xi(P)$ this yields the **data-limited scaling exponent**

$$\boxed{\;\mathcal L(P)-H_\infty \sim P^{-\alpha_D},\qquad \alpha_D=\frac{\gamma}{2\beta}\;}$$

## 3. Length generalization (the unification — two equivalent statements)

Training at $T$ and testing at $\ell$ probes the system below vs. above its correlation
length. The framework's central claim:

$$\boxed{\;\text{generalizes to length }\ell \iff \ell \;\lesssim\; \min\!\big(T,\;\xi(P)\big)\;}$$

i.e. the relevant correlations must fit inside *both* the trained context and the
data-resolvable horizon. Equivalently, plot the **FSS collapse**
$(\mathcal L(\ell)-H_\infty)\,T^{\gamma}=\Phi(\ell/T)$; collapse holds when $\xi\lesssim T$
and *fails* (retention drops) when $\xi>T$.

**Information-theoretic dual.** A model length-generalizes only if its state can hold the
predictive information required at the test horizon (the L²M condition, Chen et al.
arXiv:2503.04725, with bipartite MI $I_{\rm bi}(\ell)\sim \ell^{\beta}$):

$$\boxed{\;S_{\text{model}} \;\ge\; I_{\mathrm{pred}}(\ell)\;}$$

Since $I_{\mathrm{pred}}(\ell)\!\to\!E<\infty$ iff $\gamma>1$, a **fixed-state model
(Mamba/SSM) can length-generalize only in the weak-LRD regime $\gamma>1$**; strong LRD
($\gamma<1$) requires a *growing* cache (attention).

## 4. Architecture term $\delta$ (proposed ansatz — not yet derived)

Real models fall short of the data-resolvable horizon by an architecture-dependent gap
$\delta\ge 0$ ("fast within-horizon learning" is imperfect):

$$\xi_{\text{model}}(P)\sim P^{1/(2\beta+\delta)},\qquad \alpha_{\text{realized}}=\frac{\gamma}{2\beta+\delta}.$$

$\delta$ has two sources: **(i) positional extrapolation** (RoPE goes out-of-distribution
at $\ell>T$; NoPE/randomized PE reduce it) and **(ii) state capacity** ($S_{\text{model}}\ge\ell^{\beta}$;
fixed-state caps the realizable $\beta$). Holding $(\beta,\gamma)$ fixed and swapping
Transformer/RoPE/Mamba **measures $\delta$ directly** — and accounts for the repo's
Result 11 (Mamba $r\!=\!1.00$ vs Transformer $r\!=\!0.39$ at matched train accuracy).

## 5. Design principle (the actionable goal) — and why edge-of-chaos was refuted

Naive hypothesis (H1): push data to the **edge of chaos** ($\beta\to0$, long-range) for
maximal complexity. **The repo refuted H1**, and the framework explains why: at $\beta\to0$,
$\xi(P)=P^{1/2\beta}\!\to\!\infty\gg T$, so the long-range structure is *unlearnable in
context* — high Shannon excess entropy $E$ but low **extractable** structure (epiplexity).

Corrected principle — maximize *extractable* structure subject to feasibility:

$$\boxed{\;\max_{\beta,\gamma,D}\; D(\beta,\gamma)\quad\text{s.t.}\quad \xi_{\text{data}}\approx\xi_{\text{model}}\approx T\;,\;\; \gamma\gtrsim 1\;}$$

Both derivations land on the same optimum: **$\gamma^\star\approx 1^{+}$** (the excess-entropy
boundary, where predictive information is just finite/holdable yet epiplexity is near-maximal)
and **$\beta^\star=$ the largest state-learnable $\beta$**, i.e. **match the data's correlation
length to the architecture's horizon** — *not* chase criticality. "Holographic" data =
maximize logical depth $D$ at fixed input length, staying within $\xi\!\le\!T$.

## 6. Falsifiable predictions

1. **FSS collapse:** $(\mathcal L(\ell)-H_\infty)T^{\gamma}=\Phi(\ell/T)$ collapses across $T$; the collapse exponent equals the measured $\gamma$.
2. **Horizon condition:** retention $r=\mathrm{acc}(2T)/\mathrm{acc}(T)$ drops sharply exactly when $\xi(P)=P^{1/2\beta}$ crosses $T$ (sweep $P$ at fixed $\beta$).
3. **$\gamma{=}1$ wall for fixed-state models:** Mamba length-generalizes for $\gamma>1$ and collapses for $\gamma<1$; attention does not show this wall.
4. **$\delta$ ordering:** at fixed $(\beta,\gamma)$, $\alpha_{\text{realized}}$ orders NoPE/Mamba $>$ RoPE; the deficit $=\delta$.
5. **Design optimum:** length-gen peaks at $\gamma\!\approx\!1^{+}$, moderate $\beta$ — **not** at $\beta\to0$ (the refuted edge-of-chaos), and tracks epiplexity $D$, not Shannon $E$.

## 7. Measurement recipes
- $\beta$: $\lVert C(n)\rVert_{\mathrm{op}}$ vs $n$ (short-lag power-law fit; watch the $P^{-1/2}$ noise floor).
- $\gamma$: neural-LM cross-entropy $\mathcal L_n(P)$ vs $n$ (confirm convergence in $P$ before fitting; plug-in block entropy is biased).
- $H_\infty,E$: floor + $\sum(H_n-H_\infty)$; check $\gamma>1$.
- $D$: epiplexity AUC above final loss. $H$: DFA. $\xi$: fit $n^*(P)$.
- Cross-check: $P\!\to\!P/n^{2\beta}$, $\mathcal L\!\to\! n^\gamma\mathcal L$ scaling collapse.

## 8. Assumptions & honest limits
- $\alpha_D=\gamma/2\beta$ (§2) is rigorous for the data-optimal estimator; **the $\delta$ extension (§4) is a modeling ansatz**, to be validated by the Transformer/Mamba/RoPE experiment.
- Power laws are necessary-not-sufficient for criticality — require a **multi-estimator panel + surrogate/structural-break nulls + MLE fits** before any "critical" claim (econophysics/neuro caution).
- $\beta$ and $\gamma$ are not fully independent (two faces of long-range structure); **Hurst is the genuinely separate axis** — consistent with the companion repo's $|\rho(\text{Hurst},\text{content})|\!\approx\!0$.
- The cleanest open theory step: **derive $\alpha_D$ and measure $\delta$ in the Random Hierarchy Model** (Cagnetta et al. PRX 2024, arXiv:2307.02129; analytic correlations arXiv:2406.00048), where $\beta,\gamma$ are known in closed form.

---

*Drafts merged from `lit_review/framework_draft_physics.md` and `lit_review/framework_draft_infotheory.md`; grounded in `literature_review.md` and the primary sources (arXiv:2602.07488, 2601.03220, 2503.04725).*
