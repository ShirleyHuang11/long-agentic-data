# A Finite-Size-Scaling / RG Framework for Length Generalization on (β, γ)-Structured Data

*Draft synthesis. Physics / renormalization-group / finite-size-scaling (FSS) lens.*
*Grounded in: `gamma-beta.pdf` (Cagnetta–Raventós–Ganguli–Wyart 2026, arXiv:2602.07488 — fully readable, Eqs. 3–9 + App. A Eqs. 15–21 transcribed), `theory_derivation_and_verification.md`, `physics_statmech.md`, `case/phase/results/main_findings.md`, `case/phase/reports/holo_length_gen.md`.*

We treat a sequence model trained at context `T` on `P` tokens as a **finite-size critical system**. The data supplies two correlation/entropy exponents `(β, γ)`; the architecture supplies a penalty `δ`. Length generalization is the FSS question: *does an observable measured at size `T` collapse onto the universal curve at size `T′ ≫ T`?* The answer is governed by a single emergent length, the prediction horizon `n*(P)`.

---

## 1. Definitions (precise, with units)

Tokens are dimensionless symbols indexed by integer position; **all "lengths" (`n`, `T`, `ξ`, `n*`) are in units of tokens**. `P` is a token count; `H`, `L` are in nats/token.

- **Correlation-decay exponent β** *(dimensionless)*. With the two-point co-occurrence covariance matrix (PDF Eq. 3)
  $$C_{\mu\nu}(n) = \Pr\{X_i{=}\mu,\,X_{i+n}{=}\nu\} - \Pr\{X_i{=}\mu\}\Pr\{X_{i+n}{=}\nu\},$$
  the **signal strength** at separation `n` is the top singular value `‖C(n)‖_op`, and
  $$\boxed{\;\|C(n)\|_{\mathrm{op}} \asymp n^{-\beta}\;}\qquad\text{(PDF Eq. 7).}$$
  Large `β` = short memory (local); small `β` = long-range / "critical".

- **Conditional-entropy-decay exponent γ** *(dimensionless)*. With `H_n = H(X_{n+1}\mid X_{1:n})` the next-token conditional entropy and `H_∞ = \lim_{n} H_n` the irreducible entropy rate (PDF Eq. 18),
  $$\boxed{\;H_n - H_\infty \asymp n^{-\gamma}\;}\qquad\text{(PDF Eq. 6/19).}$$
  `γ` is the **value of context**: the predictable-information gained by extending the window. `H_n` is non-increasing in `n`, so `γ>0`.

- **Hurst exponent H** *(dimensionless, `0<H<1`)*. Self-affinity of the embedded/symbolic series, `⟨|x_{t+ℓ}-x_t|^2⟩ ∝ ℓ^{2H}`; equivalently DFA-1 fluctuation `F(s)∝s^H`. For the *increment* correlation it maps to `β = 2 - 2H` (fGn convention). **Empirically the genuinely independent axis** — the repo finds `|ρ(Hurst, content)| ≈ 0` (`MEMORY.md`); `(β, γ)` and `H` should be reported as separate coordinates.

- **Logical depth D** *(units: compute steps)*. Bennett depth = the runtime of the near-shortest program that prints the string. "**Holographic**" data = high `D` at fixed, short input length (deep computation packed into a short context). Low for both random and trivial strings; the project's `epiplexity` (arXiv:2601.03220) is the compute-bounded operationalization.

- **Effective correlation length ξ(P) / prediction horizon n*(P)** *(units: tokens)*. The maximal separation whose correlation rises above the finite-sample noise floor. The empirical covariance has standard error `O(P^{-1/2})`, so an order-`n` dependence is *detectable* iff (PDF Eq. 4)
  $$\|C(n)\|_{\mathrm{op}} \gtrsim P^{-1/2}\;\Longleftrightarrow\; \|C(n)\|_{\mathrm{op}}^2 \gtrsim 1/P.$$
  Substituting `n^{-β}` and solving for `n`:
  $$\boxed{\;\xi(P) \equiv n^*(P) \asymp P^{\,1/(2\beta)}\;}\qquad\text{(PDF "Pₙ\* ≍ n^{2β}").}$$
  **The factor 2 is the square in a signal²-vs-variance comparison** — detection is variance-limited — *not* a kinetics constant. This is the RG length that grows as resolution (`P`) increases; `H_∞` is the fixed point it flows toward.

---

## 2. Master scaling law (FSS data collapse) + length-generalization condition

**(a) Loss decomposition (PDF Eq. 5).** Split the autoregressive loss above the floor into a horizon term and within-horizon suboptimality:
$$\mathcal{L}_{\mathrm{AR}}(P,T) - H_\infty \;\asymp\; \underbrace{\big(H_{n^*(P)} - H_\infty\big)}_{\text{unresolved context}} \;+\; \underbrace{\sum_{n\le n^*(P)} \mathcal{E}_n(P)}_{\text{within-horizon error}}, \qquad T \gtrsim n^*(P).$$

**(b) Data-limited scaling exponent.** Under the **fast-within-horizon-learning** assumption (`Σℰ_n` subdominant), the residual equals the conditional entropy *at the horizon*:
$$\mathcal{L}_{\mathrm{AR}}(P) - H_\infty \asymp \big(n^*(P)\big)^{-\gamma} \asymp P^{-\gamma/(2\beta)}\;\Longrightarrow\;\boxed{\;\alpha_D = \dfrac{\gamma}{2\beta}\;}\quad\text{(PDF Eq. 8).}$$

**(c) The FSS data collapse (the unifying form).** The per-horizon `n`-gram loss family collapses (PDF Eq. 9):
$$\boxed{\;\mathcal{L}_n(P) \;=\; n^{-\gamma}\,\ell\!\left(\frac{P}{n^{2\beta}}\right)\;}\qquad\Longleftrightarrow\qquad \mathcal{L}_n\,n^{\gamma} = \ell\!\left(P\,n^{-2\beta}\right).$$
This is **exactly Fisher–Barber/Binder FSS**: `n` plays the role of system size `L`, the scaling variable is `P/n^{2β} = (n/ξ(P))^{-2β}`, the magnitude prefactor exponent is `γ`, and the correlation-length exponent (the physics `ν`) is `1/(2β)`. **Reading the dictionary:**

| Statistical physics | This framework |
|---|---|
| system size `L` | horizon `n` (or eval length `ℓ`) |
| correlation length `ξ ∝ |t-t_c|^{-ν}` | prediction horizon `ξ(P) = n*(P) ∝ P^{1/(2β)}` |
| correlation-length exponent `ν` | `1/(2β)` |
| scaling variable `L/ξ` | `n/ξ(P) = (P/n^{2β})^{-1/(2β)}` |
| anomalous-dimension `η` | `β` |
| order-parameter exponent ratio `ρ/ν` | `γ` (vertical rescaling of the master curve) |
| RG flow to fixed point | loss flow `L_AR → H_∞` |
| universality class | architectures sharing `γ` (PDF Fig. 2: γ same for APE/RoPE/LLaMA) |

**(d) Length-generalization condition.** Train at `T`, evaluate at `ℓ`. The model resolved correlations only up to `min(T, n*(P))`. Generalization to `ℓ` holds iff the eval length probes no *new* unresolved structure:
$$\boxed{\;\text{length-generalizes} \iff \ell \lesssim \xi_{\mathrm{model}}(P,T) \approx \min\big(T,\;n^*(P)\big).}$$
- **Off-critical data (`ξ(P) ≪ T`, large β):** all relevant structure is inside `T`; the ratio `r(ℓ)=\mathrm{acc}(ℓ)/\mathrm{acc}(T)\to 1` and curves at different `T` collapse trivially — **clean length-gen**.
- **Critical data (`ξ(P) ≳ T`, small β):** training *truncates* a structure that extends past `T`; evaluating at `ℓ>T` probes never-resolved separations, so `r<1` and collapse fails **unless** plotted against `ℓ/T` with residual rescaled by `T^{-γ}`:
  $$\big(\mathcal{L}(\ell) - H_\infty\big)\,T^{\gamma} \;=\; g(\ell/T),\qquad g\equiv\text{universal length-gen master curve.}$$
  A clean collapse of `r(T,ℓ)` vs `ℓ/T` across several `T` is the **falsifiable signature of the power-law (critical) regime**; a sharp `T`-independent `r` means `ξ` is finite (off-critical). This matches the repo's `L_n(P)→L_n n^γ`, `P→P/n^{2β}` collapse already observed.

---

## 3. Where the architecture penalty δ enters

`α_D = γ/2β` is **data-optimal**: it assumes the model is a perfect estimator that saturates the data-detectability bound (`Σℰ_n` negligible). Real architectures lag. **δ lives in the gap between "resolvable in the data" and "learned by this model."** Two equivalent encodings:

1. **Horizon penalty (multiplicative in the exponent).** The *model's* realized horizon falls short:
   $$n^*_{\mathrm{model}}(P) \asymp P^{1/(2\beta + \delta)},\qquad \alpha_D^{\mathrm{realized}} = \frac{\gamma}{2\beta+\delta},\qquad \xi_{\mathrm{model}} \le \xi_{\mathrm{data}}.$$
   `δ=0` recovers the bound. Length-gen uses `ξ_model`, not `ξ_data`.

2. **State-capacity floor (L²M, arXiv:2503.04725).** To model dependencies out to `ℓ`, the history state must grow at least as `I ∼ ℓ^{β}` (bipartite MI). This hard-caps fixed-state models and pins δ to architecture:
   - **Vanilla / RoPE Transformer:** KV cache is unbounded, *but* usable composition depth is capped by layer count, and **positional generalization gates length-gen**. RoPE δ is dominated by rotary-frequency extrapolation OOD behavior — isolating RoPE/NoPE/ALiBi at fixed `(β,γ)` measures the *positional* component of δ. Repo: Transformer `r ≈ 0.385` at the strip (`main_findings` Result 11), i.e. broken collapse.
   - **Mamba / selective SSM:** a recurrent state that is content-selective can satisfy the L²M condition up to its `d_state`; repo: Mamba `r = 1.001 ± 0.017` at the strip (Result 11) — `ξ_model ≈ ξ_data`, collapse holds. The `d_state` caps the realizable `β` before truncation.

**Isolation recipe (the controlled δ experiment).** Hold the generator fixed (same `β, γ`, matched train_acc at `T`), vary only architecture. Because `α_D = γ/2β` is architecture-free, any residual gap in `r(ℓ/T)` *is* δ. Repo confirms the precondition: train_acc at `T=512` is matched Transformer-vs-Mamba (0.230 vs 0.230 at the strip), so "the architectural difference is essentially all in length-generalization, not in fitting" (Result 11). The Mamba-to-`r=1` gap measures residual SSM truncation; the Transformer-to-Mamba gap measures the state-growth penalty.

---

## 4. Design principle: max-effective-complexity as constrained optimization

We want data that maximizes the *reach* of length generalization while remaining *learnable* from finite data and finite context.

**Objective.** Length-gen reach is the resolvable horizon `ξ(P) ∝ P^{1/(2β)}` (monotone **decreasing in β**: small β ⇒ long reach). Learnability is the data-limited convergence rate `α_D = γ/2β` (monotone **increasing in γ**, **decreasing in β**) *plus* a hard finite-sample/finite-context constraint: the structure at the horizon must actually clear the noise floor within budget `(P, T)`. Formally,
$$\max_{\beta,\gamma}\;\; \underbrace{\xi(P)=P^{1/(2\beta)}}_{\text{reach}} \quad \text{s.t.}\quad \underbrace{\xi(P)\le T}_{\text{within trained context}}\;\;\text{and}\;\; \underbrace{\alpha_D=\tfrac{\gamma}{2\beta}\ \text{large enough to converge in }P\text{ steps}}_{\text{learnable}}.$$
The **excess-entropy / predictive-information** budget `E = Σ_n (H_n - H_∞)` is *finite iff γ>1* (Bialek–Nemenman–Tishby; Crutchfield–Feldman): the candidate "maximum effective complexity" point sits near **γ ≈ 1** (sub-extensive divergence boundary), not at γ→0.

**Predicted optimum.** The constraint `ξ(P) ≤ T` binds. Saturating it gives the **edge-of-budget**, not the edge-of-chaos:
$$\beta^\star(P,T) = \frac{\log P}{2\log T},\qquad \gamma^\star \approx 1.$$
For training-scale `P, T` this is **moderate-to-large β** (locality matched to the trained window), with γ near the excess-entropy boundary — **not** β→0.

**Reconciliation with the REFUTED H1 (edge-of-chaos is NOT best).** The naive "edge of chaos" hypothesis equates max complexity with **β→0** (ξ→∞, true criticality). Our objective shows that is a *category error under a finite budget*: pushing β→0 sends `ξ(P) ≫ T`, which **violates the binding constraint** — the data's relevant structure lies beyond the trained (and often beyond the data-resolvable) horizon, so it is *unlearnable / does not length-generalize*. This is exactly what the repo found:
- `holo_length_gen.md`: retention ordering **Natural > Abyss > Edge > CoT**; "Edge-of-chaos 非最优" (NOT optimal), flat ridge `r ≈ 0.18–0.24`, no peak; small-β×high-γ "high retention" was a **low-denominator cheat artifact** (low train_acc).
- `main_findings.md` Result 2: the single-`α* = γ/2β` threshold is **refuted** — the chaos `α` range `[0.025, 192]` *encloses* the emergent range `[0.001, 0.123]`. There is no critical-α line; the boundary is a **soft crossover** `γ*(β) ≈ 0.274 + 0.265 log β`.
- RASP-L / locality literature (`literature_review.md` caveat): better length-gen at **high β** (sparsity/locality), consistent with the refutation.

So the corrected design principle is **"match the data's correlation length to the architecture's effective horizon"** (`ξ_data ≈ ξ_model ≈ T`), maximizing γ (toward the γ≈1 excess-entropy boundary) at the **largest β still long enough to be non-trivial** — a *budget-matched* optimum, not a critical point. "Holographic" learnability follows the same rule: deep logical depth `D` is learnable only when its correlation footprint `ξ` fits inside `min(T, n*(P))`.

---

## 5. Falsifiable predictions + measurement recipes

| # | Prediction | Measurement recipe |
|---|---|---|
| **P1** | **Collapse exponent is `1/(2β)`.** `L_n(P)` curves collapse under `P→P/n^{2β}`, `L_n→n^γ L_n`; collapse *quality* peaks at the true β and degrades for wrong β. | Measure β via `‖C(n)‖_op` power-law fit (PDF Eq. 7) and γ via small-`n` fit of `L_n(P)` for largest `P` (Eq. 6). Test data collapse residual vs assumed β (FSS goodness-of-collapse). |
| **P2** | **Length-gen threshold `ξ(P)≈min(T,n*)`.** `r(ℓ)=acc(ℓ)/acc(T)` stays ≈1 for `ℓ<ξ_model` and drops for `ℓ>ξ_model`; the knee location scales as `P^{1/(2β+δ)}`. | Sweep eval length `ℓ`; locate the `r`-knee; sweep `P` and fit knee vs `P` on log-log to extract `2β+δ`; compare to data β. |
| **P3** | **δ is positional + state-capacity.** At fixed `(β,γ)`: Mamba `r`(ℓ/T) collapses (`ξ_model≈ξ_data`); RoPE-Transformer breaks; the gap closes as `d_state` ↑ (Mamba) or via ALiBi/NoPE (Transformer). | Controlled swap at matched train_acc(T) (repo Result 11 protocol); plot `r` vs `ℓ/T` per architecture; sweep `d_state` and positional scheme. |
| **P4** | **Optimum at budget-matched β, γ≈1 — NOT β→0.** Max length-gen reach (subject to learnability) occurs near `β★=log P/(2 log T)`, `γ★≈1`; β→0 gives `ξ≫T` and *fails*. | Phase sweep over `(β,γ)`; report **absolute** acc(ℓ) + train_acc gate (cheat-guard); locate max of learnable reach; verify it is interior, not at β→0. |
| **P5** | **Max effective complexity ⇔ γ≈1 (excess-entropy boundary), not Lyapunov-zero criticality.** The repo's "emergent strip" coincides with finite-but-large predictive information `E=Σ(H_n-H_∞)`, peaking near γ→1⁺, *not* with β→0 / diverging ξ. | Compute `E` and predictive information on generated data; compute MFDFA Δα and multiscale entropy (peaked complexity probes); correlate their peak location with the emergent strip and with γ≈1. |

---

## 6. Assumptions & limits (explicit)

1. **Fast-within-horizon learning.** The master law assumes `Σℰ_n` is subdominant (`δ=0` baseline). The PDF itself shows `δ_n > γ/2β` empirically (Fig. 6), i.e. within-horizon losses decay *faster* than the AR loss — so the horizon term dominates, supporting the assumption *for deep nets*, but it **fails for shallow nets / kernels / n-gram models** (PDF §3, §6). All architecture difficulty is pushed into `δ`.
2. **Single emergent length.** We assume one correlation length `ξ(P)` controls collapse (monofractal). Real data is often **multifractal** (broken power law — PDF notes WikiText β has two stages; repo notes train_acc is "concave in log β" = corrections-to-scaling). Then a single β/ξ is an effective coarse-grained description; use the short-lag stage (PDF convention) and report MFDFA `h(q)`.
3. **β and γ may not be independent.** Physics hyperscaling implies only 2 of 6 exponents are free; whether `(β,γ)` obey a constraint (making `α_D=γ/2β` a genuine *scaling relation*) is open. Hurst H *is* empirically separate (`ρ≈0`).
4. **Variance-limited detection sets the factor 2.** Feature-learning / kernel regimes (Maloney–Roberts–Sully) can change the effective noise scaling and shift `2β → (1+a)β`. The factor 2 is robust only under `O(P^{-1/2})` sample-covariance noise.
5. **Single-level vs deep hierarchy.** The repo's `AlgorithmicKVGenerator` is single-level `p(d)∝d^{-(β+1)}`; the derivation assumes a deep latent hierarchy (RHM). This may be *why* Result 2 sees a soft crossover rather than a sharp critical line — distinguishing "single-level artifact" from "genuine soft crossover" from "δ varies across the plane" is the central open theory question.
6. **Finite-range probe.** Like the PDF, all tests probe finite `P_max, T`; many horizons `n` are not yet in the saturation/bending regime, so exponents are *effective*, valid in the probed window.
7. **Criticality caveat.** Power-law correlations are necessary-not-sufficient for criticality (Schwab–Nemenman–Mehta: latent mixing fakes Zipf/criticality). Demand multiple independent signatures (MFDFA Δα, predictive-information peak, model-side Lyapunov≈0, FSS collapse) before any "critical" claim.

---

### One-paragraph summary
A model trained at context `T` on `P` tokens is a finite-size critical system whose emergent length is the data-resolvable horizon `ξ(P)=n*(P) ∝ P^{1/(2β)}` (factor 2 = signal²-vs-variance). The loss collapses as `L_n(P)=n^{-γ}ℓ(P/n^{2β})` — textbook FSS with `ν=1/(2β)`, anomalous dimension `β`, vertical exponent `γ` — giving the data-optimal exponent `α_D=γ/(2β)`. **Length generalization holds iff `ℓ ≲ min(T, n*(P))`** (equivalently `ξ ≲ T`). The architecture penalty `δ` is the gap between data-resolvable and model-learned horizon, `n*_model ∝ P^{1/(2β+δ)}`, dominated by positional extrapolation (RoPE) and state capacity (L²M `state ≳ ℓ^β`); it is *all* of the Transformer-vs-Mamba length-gen gap. The design optimum **maximizes reach `ξ(P)` subject to `ξ ≤ T` and learnability `α_D`**, which binds at **moderate-to-large β** (`β★≈log P/2log T`) and **γ≈1** (excess-entropy boundary) — explaining the repo's **refuted H1**: β→0 (true edge-of-chaos) pushes `ξ≫T`, making the long-range structure unlearnable, exactly the observed flat/no-ridge, Edge<Natural retention, and the enclosed-α (no `α*`) soft crossover.
