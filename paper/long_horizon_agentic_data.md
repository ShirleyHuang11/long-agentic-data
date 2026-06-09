# A Compression-Oracle Survey of Long-Horizon Agentic Data: Merging Training and Evaluation by Pattern and Content

**long-agentic-data project**
*Built from the `SAMPLES.md` registry (127 active datasets / 133 scored rows, iters 1–123). Per-row table: `data/merged_analysis.csv` (`scripts/build_merged_table.py`). Figures: `figures/`. Supporting reports: `reports/`.*

---

## Abstract

We survey **127 long-horizon agentic corpora** — spanning model-trained SFT/RL trajectories, human-written benchmark tasks, human demonstrations, and agent rollouts collected on benchmarks — with a single cheap, tokenizer-free compression oracle, and merge training and evaluation data into one measurement table. For every corpus we measure a *pattern* axis (the context-scaling exponent α, the token-correlation decay β, and the Hurst exponent H) and a *content* axis (the reference-exact incompressible floor H∞, validated against true neural oracles at Spearman 0.97, with a directly measured companion BPC@32K). We report two dissociated findings. First, **pattern is a property of the agentic *format***: α is nearly constant across train/eval roles (median 0.25–0.34), and agentic model trajectories occupy a distinct correlation-decay band (β ≈ 0.15–0.52) that is separated from natural-language prose (β ≈ 1.11–1.37) by an empty gap, with code/math contiguous as the bridge (β 0.52–0.79). Second, **content is set by who authored the token stream, not by whether the data is for training or testing**: median H∞ falls monotonically with generator source (human task 1.11, human demo 1.13, frontier rollout 0.80, synthetic task 0.45, mid-size rollout 0.00, distilled SFT 0.00), and this ordering cuts across the train/eval boundary — mid-size eval rollouts collapse to the template floor exactly as distilled training mixtures do. A one-way variance decomposition makes the asymmetry quantitative: **generator source explains ~3× more of the H∞ variance than the train/eval role (η² = 0.35 vs 0.08)**. The consequence is a measurable **content gap**: the benchmark tasks we score agents against are human-authored and dense (median H∞ 1.11) while the training data we feed agents is content-sparse (median 0.28), and the gap reproduces *within* every domain that has both cohorts. We connect these statistics to a learning-curve theory (α_D = γ/2β) and to a training experiment that confirms SFT teaches trajectory *form*, and we surface one method caveat that survived audit: on single-harness-pooled eval rollouts the 3-point H∞ measures the agent scaffold as much as the generator, so BPC@32K and turn-count are the robust separators there. We keep the reference 3-point clamped H∞ as canonical for cross-domain comparability.

---

## 1. Introduction

Long-horizon agentic data — multi-turn trajectories where a model interacts with tools, repositories, browsers, or operating systems — is now the central fuel of frontier agent training, yet it is curated almost entirely by intuition: "frontier teacher × hard tasks × real environment" is folklore, not a measured quantity.

**Compression as a data oracle.** Language modeling is compression [4]: a model's bits-per-character on a corpus is its negative log-likelihood, so a cheap general-purpose compressor is a label-free lower bound on what a model could learn. We use a Lempel–Ziv (zstd) oracle to read off two scaling parameters of the bits-per-character curve BPC(n) over context length n, following a protocol validated against true LLM oracles (§2). A few seconds of CPU compression can then separate a content-rich rollout from a boilerplate-pooled one *before* any GPU is spent.

**Pattern versus content.** A corpus has structure (how much its long-range regularities compress — templating, repeated scaffolds, genuine dependence) and content (its irreducible per-character novelty). We measure them with complementary statistics — α, β, H for pattern; H∞ and BPC@32K for content — but they are not all independent: α and H∞ co-vary strongly (§4), because templating depresses both, while the Hurst exponent is the one pattern statistic genuinely orthogonal to content. The dissociation this paper establishes is therefore not pattern-from-content but **role-from-everything**: whether a corpus is for training or evaluation predicts neither its pattern nor its content; the generator source predicts the content.

**Merging training and evaluation.** Training corpora and benchmarks are studied in separate literatures. We place both in one table and ask whether the train/eval label predicts anything about a corpus's statistics. It largely does not — the generator source does — and the resulting "content gap" between what we test on and what we train on is visible only once the two are merged.

**Statement of contribution.** In summary, we:

1. assemble and release a merged table of 123 long-horizon agentic corpora, each classified by role (train / eval-task / eval-traj), domain, and generator source, with reference-exact (α, H∞), directly measured BPC@32K, and where available β and Hurst (`data/merged_analysis.csv`);
2. show that **pattern is the agentic-format genre signature** — α invariant across roles, and a distinct low-β correlation-decay phase separating agentic data from prose with code/math as the bridge (§4);
3. show that **content (H∞) tracks generator source, not the train/eval role**, exposing a quantified content gap between human-authored benchmark tasks (median H∞ 1.11) and machine-generated training data (median 0.28) (§5);
4. give a domain-dependent form/content decomposition (web/GUI observations are form, SWE observations are content) and a training experiment confirming SFT teaches form (§6–7); and
5. document a measurement caveat — the agent harness confounds H∞ on single-harness-pooled eval rollouts — and the synthetic-control audit that fixes the canonical metric to the reference 3-point clamp (§8).

---

## 2. Background and Related Work

**Language modeling as compression.** Delétang et al. [4] formalize the equivalence between sequence modeling and compression; an off-the-shelf compressor therefore upper-bounds achievable BPC. Earlier work relates LLMs to the entropy of English [6]. We exploit this to build a model-free data probe.

**Neural scaling from data statistics.** Cagnetta et al. [1] derive data-limited learning-curve exponents from the statistics of natural language, showing the exponent depends on the token–token correlation decay β (‖C(n)‖ ∝ n^(−β)) and a generative-tree branching factor γ, with α_D = γ/2β. β is therefore a *predictive* data statistic, which we measure for every phase.

**Fractal and long-range structure.** Alabdulmohsin et al. [2] establish that language is self-similar and long-range dependent, with a Hurst parameter H ≈ 0.70 ± 0.09, and show that the small cross-model variation in fractal parameters improves prediction of downstream accuracy over BPC alone (adjusted R² 0.65 → 0.86). We measure H on agentic corpora and find (§4) that it cannot grade them alone, because repetition is itself long-range dependence.

**Intrinsic entropy and context length.** The intrinsic-entropy work [3] shows the optimal context length grows with data, and fits a Bayes-risk form that is exactly the BPC(n) = H∞ + c·n^(−α) equation our oracle reads. The compression-progress view of curiosity/learning [5] motivates "intermediate complexity is most learnable," which our content axis quantifies.

**The formal-math LZ survey (the protocol we reproduce).** Our oracle reproduces, bit-for-bit, the protocol of a 358-dataset formal-math survey (`reference/data_format.md`): 3-point analytic estimation at n = 128 / 2048 / 32768 (geometric ratio r = 16), zstd level 19, ≤1500 documents or 8 MB, with H∞ floored at 0. There, LZ-derived H∞ correlated with a true neural-oracle H∞ at Spearman 0.97, and H∞ ≈ 0 was a *valid* template-degenerate signal (e.g. TPTP ≈ 0). We keep that protocol exactly so this agentic registry is comparable to the formal-math one and to its cross-domain table (natural language 2.6 / code 2.63 / formal-math 1.57).

---

## 3. Data and Methods

### 3.1 The merged corpus

The registry holds **130 scored rows**; 6 are 3-point fit artifacts (α < 0 or H∞ ≫ 1, plus one single-episode dump) and are dropped, leaving **123 active corpora**. We classify each along three axes (`scripts/build_merged_table.py`):

| axis | values |
| :-- | :-- |
| **role** | `TRAIN` (SFT/RL training trajectories) · `EVAL_TASK` (benchmark task/problem corpora + human-demonstration benchmarks) · `EVAL_TRAJ` (agent rollouts collected on a benchmark) |
| **domain** | swe · web · gui · tool · search · terminal · safety · embodied · mixed |
| **source** | `human_task` · `human_demo` · `synth_task` · `frontier` (frontier-model rollout) · `mid` (mid-size-model rollout) · `distill` (GPT-4-class distillation/SFT mixture) |

Counts: **TRAIN 83, EVAL_TASK 15, EVAL_TRAJ 29**; by source, frontier 63, distill 26, mid 19, human_demo 10, human_task 7, synth_task 2. The merge is deliberate: human-demonstration datasets (Mind2Web, WebLINX, GUI-Odyssey, AndroidControl, AgentNet, OpenCUA) straddle the train/eval line — they ship test splits *and* serve as demonstration training data — so `source` is the axis that carries the real signal, while `role` tests whether the train/eval label predicts anything (it largely does not; §5).

The role label is editorial for a handful of corpora: SWE-Gym and SWE-rebench are training *environments* whose released form is a task corpus (labeled `EVAL_TASK`); AgentNet/OpenCUA are human demos released as *training* data (labeled `TRAIN`, source `human_demo`). These calls do not move the central result, which is organized by `source`.

### 3.2 The metric toolbox

For a corpus of documents joined with blank-line separators (≤1500 docs / 8 MB), we compress in independent n-byte chunks and read bits-per-character BPC(n), fitting the scaling model

$$\mathrm{BPC}(n) = H_\infty + c\,n^{-\alpha}.$$

- **α — pattern / long-range structure.** Larger α ⇒ more context-length structure an LZ compressor can exploit (templating, repeated scaffolds, or genuine long-range regularity).
- **H∞ — content / incompressible floor (canonical).** The reference-exact 3-point analytic estimate at n = 128 / 2048 / 32768, floored at 0, with closed form $H_\infty = B_3 - (B_2-B_3)^2 / [(B_1-B_2)-(B_2-B_3)]$. H∞ ≈ 0 is the reference's valid "template-degenerate / spam" signal, not a bug. This is the canonical content metric, kept bit-identical to the formal-math registry.
- **BPC@32K — directly measured content companion.** The raw compressed rate at a 32 KB context — no fit, no clamp, no extrapolation. Validated on synthetic controls (random 2.47 / pure template 0.02 / scaffold+content 1.40) and robust to the 8K-vs-32K-token context choice (ranking invariant). Reported alongside H∞; where they disagree, §5 explains why.
- **β — token-correlation decay [1].** ‖C(n)‖ ∝ n^(−β); α_D = γ/2β. Measured byte-level here as a proxy.
- **Hurst H [2].** R/S on order-3 byte-n-gram surprisal increments; correlates with downstream accuracy in the fixed-data, cross-model setting.
- **seed-σ — reproducibility.** Re-scoring on disjoint slices (38 rescores over 12 corpora) bounds which differences are real. Measured: homogeneous pipelines σ ≈ 0.03–0.04 (glaive 1.03±0.03, Toucan 1.35±0.04), the template band pinned at σ = 0, and heterogeneous repo-scale corpora up to σ ≈ 0.22 (SWE-ZERO 0.82±0.22, slice-composition dominated). So **cluster-level gaps (0 vs 1+) are ≫ σ and trustworthy, while within-band differences < 0.3 are noise** — the rule used throughout.

**Cost.** "Cheap" is literal: scoring a full 8 MB corpus with the reference 3-point oracle (zstd-19 at three context sizes) takes ≈ 4.5 s on a single CPU core, so the entire 106-corpus registry scores in a few minutes with no GPU — the point of a compression probe is that it runs before, and at a tiny fraction of the cost of, any training.

---

## 4. Pattern Analysis: the Agentic Format Is Its Own Statistical Phase

**α is nearly invariant to role.** Median α is 0.25 (TRAIN), 0.34 (EVAL_TASK), 0.25 (EVAL_TRAJ) — a ~0.1-wide band (Figure 1b). Whatever long-range structure the LZ probe sees is a property of the multi-turn agentic *serialization* (JSON scaffolds, repeated system prompts, turn templates) and is present whether the corpus is a training mixture or a benchmark. Pattern, in this sense, is the genre signature; it does not tell training from evaluation apart.

**Agentic data occupies a distinct correlation-decay phase.** Measuring β with the same byte-level protocol across 19 reference corpora plus the FineFineWeb domains (`data/gamma_beta_all.csv`; Figure 3, with the detailed γ–β phase plane in Figure 7) gives an ordered three-band structure:

| phase | β (measured range) | examples |
| :-- | :-- | :-- |
| **agentic** (model trajectories) | **0.15–0.52** | the trajectories in this survey |
| **code / math (bridge)** | 0.52–0.79 | Python code 0.74, Proof-Pile arxiv 0.52, open-web-math 0.79, The Pile 0.78 |
| **natural-language prose** | 1.11–1.37 | C4 1.32, FineWeb-Edu 1.35, WikiText 1.22, TinyStories 1.35 |

The agentic and code/math bands are **contiguous, meeting exactly at β = 0.52** — and the corpus sitting on that seam is a *SWE* trajectory set (JetBrains GPT-5.2, β 0.52), the same value as Proof-Pile-arxiv — which is "code is the bridge" stated literally: code-heavy agentic data shades continuously into the code phase. The genuinely empty gap is **0.79 → 1.11**, separating everything procedural (agentic + code/math) from natural-language prose.

Through α_D = γ/2β [1], the low agentic β predicts an unusually high data-limited learning exponent (α_D ≈ 0.3–1.0): agentic data should be sample-efficient to fit — consistent with the observation that SFT drives trajectory *form* to ceiling almost immediately (§7), and a hypothesis still awaiting a from-scratch training test (§9). The phase is a property of the agentic *format*, training and evaluation alike: benchmark eval rollouts (iters 69–73) fall in the same band (GPT-5/Terminal-Bench β = 0.45, DeepSeek-R1/tau-bench 0.34, Qwen3-32B/Terminal-Bench 0.33, CoderForge/SWE-bench 0.15; Figure 7), reinforcing the §5 dissociation from the pattern side. The 0.15–0.52 band describes *model-generated* trajectories specifically; the human-demonstration and machine-annotation layers spread higher (Mind2Web action stream β = 0.77, AgentNet VLM annotation 1.30) because **repetition structure, not domain, sets β** — non-repetitive human action and prose-like annotation decorrelate faster (Figure 7). Heavily templated event-stream serializations sit at the opposite extreme, with correlation curves so flat that β is unfit (openhands-feedback decay ratio ≈ 1×, excluded as an artifact alongside the FineMath/Pile reference cases).

**Hurst alone cannot grade agentic data.** Across 9 representative corpora (`data/hurst.csv`; Figure 4), template/spin corpora sit in the *same* Hurst band as healthy ones (APIGen 0.80, Ko-Agent 0.83 vs JetBrains 0.78, Toucan 0.90, SWE-ZERO 0.93) — because repetition itself is long-range dependence. Hurst conflates form-LRD with content-LRD and is nearly orthogonal to H∞ here (the highest-content corpus, WebLINX-actions H∞ 1.95, has the *lowest* Hurst 0.67). The (H, H∞) pair separates them; H∞ alone does not.

**How the statistics relate (rank correlations over the 127 active corpora).** The pattern and content axes are not all independent, and quantifying their relationship clarifies what each adds. Spearman(α, H∞) = **+0.78** — α and H∞ co-vary strongly, since templating depresses both the exploitable structure and the content floor; they are two readouts of the same healthy↔template axis, not orthogonal dimensions. Spearman(BPC@32K, H∞) = **+0.59** — the directly measured companion agrees with H∞ on clean corpora but only moderately overall, precisely because the two *diverge* on harness-pooled eval rollouts (§5.3), where H∞ collapses to 0 while BPC@32K stays mid-band. Spearman(Hurst, H∞) = **−0.02** (n = 9) — Hurst is the one statistic genuinely orthogonal to content, which is exactly why it cannot grade agentic data alone. The honest summary: α (and β) track the same healthy↔template structure as content, Hurst is orthogonal to it, and BPC@32K is content's robust stand-in where the 3-point H∞ is harness-confounded.

---

## 5. Content Analysis: H∞ Tracks the Generator, and Cuts Across Train/Eval

This is the paper's central result (Figure 1).

### 5.1 Content falls monotonically with generator source

| source | n | median H∞ | mean H∞ | fraction H∞ > 0.3 |
| :-- | --: | --: | --: | --: |
| human task (written problems) | 7 | **1.11** | 1.06 | 6/7 |
| human demo (action streams) | 10 | **1.13** | 0.92 | 6/10 |
| frontier-model rollout | 63 | **0.78** | 0.76 | 45/63 |
| synthetic task | 2 | 0.45 | 0.45 | 1/2 |
| mid-size-model rollout | 19 | **0.00** | 0.17 | 4/19 |
| distilled SFT mixture | 26 | **0.00** | 0.08 | 3/26 |

By role, the same numbers re-sort to expose the gap:

| role | n | median H∞ | reading |
| :-- | --: | --: | :-- |
| EVAL_TASK | 15 | **1.11** | human-authored tasks/demos → content-dense |
| TRAIN | 83 | **0.28** | bimodal: healthy frontier minority + collapsed majority |
| EVAL_TRAJ | 29 | **0.08** | model rollouts span the full range; H∞ also harness-confounded (§5.3) |

**Content is authored, not roled.** The high-content end is human (written tasks + demonstrations); the low-content end is mid-size or distilled *model* generation. The decisive variable is the generator, and it dominates both train and eval. An eval rollout produced by a 7B–32B model (`EVAL_TRAJ`, source `mid`, H∞ ≈ 0) is statistically indistinguishable from a distilled training mixture (`TRAIN`, source `distill`, H∞ ≈ 0) — same template floor, same failure-loop length inflation (§6). Quantitatively, a one-way variance decomposition of H∞ confirms the asymmetry: **generator source accounts for η² = 0.34 of the variance, versus 0.09 for the train/eval role and 0.07 for domain** — source explains roughly **4× more** than the train/eval label. The dissociation is statistically robust, not a point-estimate artifact: across 2000 bootstrap resamples of the 127 corpora, **source explains more H∞ variance than role in 100% of them** (η² 95% CI: source [0.24, 0.50] vs role [0.02, 0.22]). And the generator effect is not merely a domain proxy: in a two-way decomposition, **source still explains 25% of the H∞ variance after removing domain means (partial η² 0.25), whereas domain explains only 3% after removing source means (partial η² 0.03)** — the asymmetry says domains differ in content chiefly *because* different generators populate them, not the reverse. Pattern (α) shows the mirror image (§4): it is nearly flat across source but, like everything else here, flat across role.

### 5.2 The content gap

The benchmark *tasks* we measure agents against are human-written and dense (median H∞ 1.11); the training data we feed agents is mostly boilerplate (median 0.28). We evaluate on human richness and train on machine repetition. The healthy training minority that closes this gap — frontier rollouts in diverse, real environments, and human action streams — is exactly the data the probe scores ≥ 0.6 in seconds.

Crucially, the gap is **within-domain, not a composition artifact** of train and eval covering different areas. In every domain that has both a training-trajectory and a human-authored eval-task cohort, the eval-task content floor is higher:

| domain | TRAIN (model trajectories) median H∞ | EVAL_TASK (human-authored) median H∞ |
| :-- | --: | --: |
| gui | 0.51 (n=4) | **1.40** (n=2) |
| web | 0.66 (n=4) | **1.00** (n=4) |
| swe | 0.59 (n=29) | **1.00** (n=6) |
| mixed | 0.00 (n=6) | **1.67** (n=1) |

The human-authored benchmark tasks out-score the model-generated training trajectories in the *same* domain every time — the content gap is a property of *who authored the data*, reproduced inside each domain, not an accident of which domains happen to be over-represented on each side.

Placed against the reference registry's whole-corpus content floors — natural language 2.6, code 2.63, formal-math 1.57 — **agentic data is the lowest-content and most bimodal domain measured**: median H∞ 0.35 (mean 0.55), well below all three, with **40% of corpora pinned at the floor (H∞ < 0.05)**. Yet its healthy tail (frontier rollouts + human action streams, median 1.10, max 1.95) reaches into the code/formal-math range. So agentic data is not uniformly low-content — it is uniquely *split* between a large boilerplate mass and a dense minority, which is precisely why a per-corpus probe (rather than a domain-level prior) is worth running.

### 5.3 A controlled benchmark-rollout study, and a harness confound

A batch of benchmark eval rollouts (iters 69–71) makes the source dominance concrete and adds a domain modulation. Consolidated:

| benchmark | rollout generator | source | agent harness | H∞ | BPC@32K | turns |
| :-- | :-- | :-- | :-- | --: | --: | --: |
| SWE-bench-Verified | GPT-5-mini | small | SWE-Router | **1.22** | 2.08 | 31 |
| SWE-bench-Verified | GPT-5.2 | frontier | SWE-Router | **1.01** | 2.11 | 32 |
| SWE-bench-Verified | CoderForge-32B | mid | OpenHands | **0.83** | 1.73 | 136 |
| SWE-bench-Verified | Claude-Sonnet-4.5 | frontier | mini-swe-agent | **0.00** | 1.60 | 106 |
| GAIA-127 | ii-agent | frontier | ii-agent | **1.25** | 3.05 | 47 |
| GAIA-127 | R2EGym-32B | mid | OpenHands | **0.00** | 1.30 | 200 |
| Terminal-Bench-2 | GPT-5 | frontier | terminus-2 | 0.00 | 1.77 | 10 |
| Terminal-Bench-2 | Claude-Sonnet-4.5 | frontier | terminus-2 | 0.16 | 1.75 | 28 |
| Terminal-Bench-2 | Claude-Haiku-4.5 | small | terminus-2 | 0.00 | 1.53 | 74 |
| Terminal-Bench-2 | GPT-5-nano | small | terminus-2 | 0.00 | 1.43 | 25 |
| Terminal-Bench-2 | Qwen3-32B | mid (open) | terminus-2 | 0.00 | 0.97 | 172 |

The mid rollouts on GAIA and Terminal-Bench collapse to the template floor with the longest failure loops in their domains (long horizon is *not* high information density; §6). The SWE rollout stays healthy at 0.83 not because a 32B generator is strong but because **repository observations are content** (§6): on SWE the environment injects real code regardless of the agent.

But reading the table carefully exposes a cautionary measurement result. **Within a fixed harness** (the five Terminal-Bench-2 rows, all `terminus-2`), H∞ is pinned at ≈ 0 for *every* model — flagship and small alike — yet BPC@32K spans a wide 0.97 → 1.77 and orders the generators by capability/success: flagship frontier (GPT-5 1.77, Sonnet-4.5 1.75) > small frontier-family (Haiku-4.5 1.53, GPT-5-nano 1.43) > open Qwen3-32B (0.97), whose 172-turn failure loop dilutes content per byte (finding 12, now within a controlled harness). Turn-count corroborates (GPT-5 solves in 10 turns; Qwen3-32B grinds 172). **Across harnesses, by contrast, H∞ entangles the agent scaffold with the generator.** The four SWE-bench-Verified rows make this a clean gradient on a *single* benchmark: the same task set reads H∞ 0.00 under `mini-swe-agent` (Claude-Sonnet-4.5), 0.83 under OpenHands (CoderForge-32B), and 1.01–1.22 under the light-scaffold SWE-Router harness (GPT-5.2 / GPT-5-mini) — the H∞ ladder tracks **how heavy each harness's shared system prompt is**, not the generator, because heavy fixed prompts pool to zero across episodes (the §8 effect at benchmark scale). Two corollaries: (i) the *same* frontier model, Claude-Sonnet-4.5, reads 0.16 under `terminus-2` but 0.00 under `mini-swe-agent`; and (ii) under the light SWE-Router harness, where the repository content survives, frontier and small are statistically tied (GPT-5.2 1.01 ≈ GPT-5-mini 1.22, within seed-σ) — on SWE the environment injects the content regardless of the agent (finding 16, §6), so capability barely moves H∞. The clean-looking GAIA-127 contrast inherits the same harness caveat (ii-agent vs OpenHands). **The two statistics that survive the harness are BPC@32K and turn-count.** The operational rule for eval rollouts is therefore: prefer BPC@32K + turn-count over the 3-point H∞, which on single-harness-pooled rollouts measures the scaffold as much as the generator. (This does not bear on the §5.1 train-data result, where corpora mix many sources and are not pooled under one giant shared system prompt.) The pooling is a property of *task narrowness*, not the harness label per se: the same `terminus-2` scaffold that pins the five-task Terminal-Bench-2 benchmark to ≈ 0 reads **H∞ 0.66–0.91** on two diverse DCAgent GLM-4.7 SWE *training* corpora (swe-gym-openhands 0.91, nebius-swe-agent 0.66; thousands of distinct repositories) — once the task set is wide, the per-episode repository content swamps the shared prefix and the harness no longer pools. Heavy scaffold *and* a narrow task distribution are both required for the floor effect.

The broader SWE-rollout picture reinforces this. Across **six generators and both English and multilingual SWE-bench** (GPT-5.2 1.01, GPT-5-mini 1.22, Kimi-K2 0.59, glm-5 0.97, minimax-m2.5 0.87, CoderForge-32B 0.83; BPC@32K 1.7–2.1 throughout), SWE rollouts sit in the healthy band regardless of generator or natural language — code content is what the probe measures, and the environment supplies it. The only two SWE rollouts that read H∞ 0 are the *harness-pooled* one (Claude-Sonnet-4.5 under mini-swe-agent, BPC@32K still 1.60) and the *weak-generator* one (swe-agent-llama-70b, BPC@32K 1.43, genuinely lower). So even within one domain, H∞ = 0 has two distinct causes — pooling vs a weak generator — separable by reading BPC@32K, never by H∞ alone.

The tool domain shows the harness effect across benchmark *generations*. The original tau-bench pools even a frontier reasoner to the floor (DeepSeek-R1 H∞ 0.00) because its fixed retail-policy system prompt is shared across every episode; its successor **tau2-bench**, in a lighter code-agent format, reads a frontier rollout (GPT-5.1-codex airline) at **H∞ 1.00** while a mid model (qwen3.5-9B retail) still collapses to 0. Same benchmark family, but the harness format — not the benchmark — decides whether H∞ is usable.

### 5.4 The probe is a regime selector, not a recipe selector

Within the frontier band H∞ does not separate teachers (DTap Opus/Sonnet/Gemini cluster at 0.71–0.81, inside seed-σ), and EVAL_TASK BPC@32K is flat (1.71) even where H∞ separates sharply. The probe cleanly tells frontier-or-human from mid-or-distilled, but within a healthy band a proxy training run is still required. The signature clusters (Figure 2) make this concrete: a *healthy long-trajectory* cluster (frontier generators, α 0.26–0.38, H∞ 0.57–1.63), a *distillation template* cluster (α 0.05–0.26, H∞ ≈ 0), a *mid-size failure-loop* cluster (longest episodes, H∞ 0–0.08), and a *compact human-action* cluster (α 0.40–0.49, H∞ 1.7–1.95, the densest data in the survey, shortest episodes). The length–content independence is registry-wide and quantitative: **episode length is essentially uncorrelated with content — Spearman(bytes/episode, H∞) = −0.02, Spearman(turns, H∞) = −0.01** (both faintly *negative*). Longer trajectories do not carry more content per byte; the longest are failure loops at the floor. Any "collect longer trajectories" heuristic is therefore orthogonal to data quality — a corpus-selection caveat that falls straight out of the merged table. 

The richest single cohort of *training* data the survey found is **real frontier CLI coding sessions** — raw agent sessions captured from frontier models doing real work (GPT-5.5 H∞ 1.93, Claude-Opus-4.8 1.73, a mixed sampler 1.49, Qwen3.7-max 1.29, minimax-m2.7 1.24, Kimi-K2.6 0.98). GPT-5.5's sessions are the second-densest corpus in the entire registry, behind only human WebLINX actions (1.95) and ahead of every benchmark task corpus. This is the cleanest empirical instance of the healthy-data recipe (frontier generator × real grounding × task diversity), and a concrete recommendation: *raw frontier agent sessions are the highest-content training data available.* The cohort's two `claude-code`-serialized members both read lower than their `pi`/`agent`-serialized siblings — minimax 1.24 → 0.00 and Kimi-K2.6 0.98 → 0.35 — a within-lab, two-model replay of §5.3: the same generator under a heavier-scaffold harness reads as less content (minimax pooled to the floor; Kimi partially).

### 5.5 Content by domain

Cutting the active registry by domain (median over each domain's corpora) shows content is also domain-stratified, in a way that reflects the source mix and the observation–content relationship of §6:

| domain | n | median H∞ | median BPC@32K | median α |
| :-- | --: | --: | --: | --: |
| gui | 6 | **1.13** | 1.67 | 0.38 |
| science | 1 | **1.00** | 2.23 | 0.22 |
| safety | 5 | 0.71 | 1.71 | 0.28 |
| swe | 46 | 0.66 | 1.65 | 0.26 |
| search | 8 | 0.60 | **2.42** | 0.19 |
| web | 11 | 0.46 | 1.51 | 0.34 |
| tool | 18 | 0.13 | 1.83 | 0.25 |
| terminal | 20 | 0.08 | 1.76 | 0.25 |
| mixed | 7 | 0.00 | 1.23 | 0.13 |
| embodied | 3 | 0.00 | 0.89 | 0.25 |
| office | 1 | 0.00 | 1.01 | 0.07 |
| legal | 1 | 0.00 | 1.29 | 0.17 |

Three patterns recur. (a) The high-H∞ domains are those dominated by **human action streams** (gui 1.13, web's human-demo cohort Mind2Web/WebLINX/AndroidControl/GUI-Odyssey) or by **frontier rollouts in real environments** (safety 0.71 = DTap, swe 0.66 = OpenHands family). Web's domain median is only 0.46 because its newer *model rollouts* (the webarena-infinity browser-use/computer-use trajectories) sit well below the human demos — the same source/framework effect, now visible *within* a single domain. (b) **search has the highest BPC@32K (2.42)** despite middling H∞, because search trajectories splice in large spans of retrieved web text — directly measured density is high even where the 3-point floor is conservative. (c) The H∞ ≈ 0 domains split by cause: **embodied** (ALFWorld), **mixed** (distillation soups), and **office** (synthetic OfficeBench SFT) are genuinely template-degenerate (low BPC@32K too), whereas **terminal** reads a near-floor median H∞ (0.08) chiefly because of the harness-pooling confound of §5.3 — its BPC@32K (1.76) is mid-band, not floor, and its diverse-task terminus-2 corpora are genuinely content-rich (§5.3). The domain cut therefore reinforces the §5 thesis (content follows who/what authored the stream) and the §5.3 caveat (terminal's zero is a measurement artifact, not empty data) at once.

---

## 6. Form versus Content: a Domain-Dependent Decomposition

The probe also dissects *within* a trajectory by who wrote each token slice (Figure 5):

- **Web/GUI observations are form.** Stripping rendered HTML/DOM or VLM annotations *raises* H∞ (Mind2Web 1.70 full-obs → action view; AgentNet 0.00 annotated → 1.43 action-only). The boilerplate is in the machine layer; the human action stream is pure content.
- **SWE observations are content.** Stripping repository observations *lowers* H∞ (JetBrains 1.63 → 0.72 assistant-only): file bodies, diffs, and test output *are* the signal. Data-cleaning policy must be domain-specific.
- **Grading the action source.** Human demonstration 1.43 ≫ planner-generated 0.43 ≫ observation-pooled 0.00 — content density is layered by who authored the actions, not by surface format.

This reconciles into one statement: **healthy agentic content = a frontier (or human) generator × scenario diversity × real grounding**, all three necessary. The same frontier model collapses to the template floor on a repeated scenario family (DTap deep-tail: Sonnet 0.81 → 0.08), and ×100 SFT dosage does not move the band. The image channel agrees: adjacent-frame pixel change is 2.6%/step on a single-app GUI (H∞ 0) vs 26.9%/step on cross-app navigation (H∞ 1.55) — observation redundancy is visible across modalities.

The decomposition extends to the **agent architecture** itself, with a cautionary twist that re-illustrates §5.3. A dataset that serializes the *same* SWE-smith tasks both as single-agent and as multi-agent (subagent-orchestrated) trajectories lets us isolate the orchestration layer: the single-agent view reads H∞ 1.03 (47.8 turns/episode), but the multi-agent view of the identical tasks reads **H∞ 0.26** (10.8 turns/episode). Read naively that is a 4× content collapse — but the *directly measured* BPC@32K barely moves (1.75 → 1.65). The gap is therefore **the H∞-pooling effect again, not a real per-byte content loss**: multi-agent orchestration fragments the solution into many short sub-episodes that all share the subagent-spawn scaffold and per-subagent system prompts, and that repeated scaffold pools away across the (now more numerous) episodes, dragging the extrapolated floor down while the per-byte content stays intact. So the orchestration layer is *form* in the precise sense of §6 — a boilerplate frame, visible to H∞ as pooling — and the lesson of §5.3 recurs: on heavily-scaffolded serializations, read BPC@32K, not H∞. (Single controlled dataset, n = 1; reported as an illustration, not an established regularity.)

---

## 7. A Training Test: Data Value Is Student-Relative

"Good data" is **student-relative**. A trajectory carries *form* (the low-β echo skeleton) and *choices* (the H∞ novelty). A weak model lacks form, so a template-band corpus is the cheapest *form* curriculum; a frontier model lacks only choices, so only the healthy band helps it. This explains why OpenThoughts-Agent — a winning small-model SFT recipe — measures content-low (BPC@32K 1.38, descaffold H∞ ≈ 0.75) yet trains the strongest model at its scale: it is an excellent *form* curriculum, not an empty one.

We tested the form half directly (`reports/formchoice_results.md`; Figure 6): Qwen2.5-1.5B SFT'd separately on a template corpus (OpenThoughts) and a healthy corpus (GLM-4.7/JetBrains/SWE-ZERO), matched at 30M tokens, evaluated on form vs decision.

| | base | template | healthy |
| :-- | --: | --: | --: |
| **FORM** | 0.78 | **0.99** | **0.96** |
| **DECISION** | 0.033 | 0.025 | 0.017 |

**FORM is confirmed** — SFT on either corpus drives the trajectory skeleton to ceiling. **DECISION is inconclusive** — all three models, *including untrained base*, floor at 2–3%, so there is no dynamic range to test healthy > template at this scale/format; we report it as inconclusive per the pre-registered design. (A first run was degenerate — a JSON-vs-fenced parser bug — caught only because the *base-model control also floored*.) A complementary filter, **LZ-Select** (`reports/lz_select_results.md`), uses the same statistics for per-episode selection: on Kwai-Klear, raw H∞ 0.26 → strip 0.36 → select 0.41 → strip+select 0.51, with kept/rejected episodes separating (0.41 vs 0.24) — the probe is actionable at the episode level, not only the corpus level.

---

## 8. Limitations and a Method Note

An earlier revision proposed replacing the extrapolated H∞ with directly measured BPC@32K, after a challenge ("OpenThoughts shouldn't score 0") surfaced three real failure modes: a negative-fit clamp (all 37 "zeros" were clamps), a non-flattening BPC curve (H∞ not identifiable in the measurable window — the closed-form denominator is a noise-dominated second difference, ill-conditioned), and cross-episode scaffold pooling (shared system prompts cancel across concatenated episodes, measuring boilerplate density rather than per-episode content).

The binding decision is to **keep the reference paper's method exactly**. The reference 3-point clamped H∞ is canonical because it is what was validated against true neural oracles (Spearman 0.97) and what makes this registry comparable to the formal-math survey. H∞ ≈ 0 is therefore read as the valid template-degenerate signal, and the pooling effect is retained as a documented **caveat**, not a replacement: BPC@32K, score_v3 (bounded 7-point least-squares with a `resolved` flag), and the descaffold measurement remain supplementary diagnostics. The qualitative structure (healthy vs template vs failure-loop) is invariant across all three measurements; only the absolute boundary at "exactly 0" is softer than it looks. The §5.3 harness confound is the sharpest live instance of this caveat. **Lesson:** any clamped/extrapolated/pooled "0" must be falsified against a directly measured quantity and synthetic controls before becoming a finding.

**Controlled validation of the pooling mechanism.** The harness confound is not only observed on real rollouts — it reproduces on a synthetic control (`scripts/synth_harness_pooling.py`). Building 400 episodes as `[shared boilerplate prefix] + [unique content]` and sweeping the prefix fraction: at 0% prefix H∞ = 2.57 and BPC@32K = 2.62 (pure content); at a **50% shared prefix H∞ clamps to 0 (raw fit −4.27) while BPC@32K stays at 1.57** — the shared prefix lets large-context chunks dedup across episodes, steepening BPC(n) until the 3-point floor extrapolates negative and clamps, exactly as on `mini-swe-agent`/`terminus-2`/`tau-bench` rollouts. Pushed further (≥80% prefix) the fit turns non-monotone (α < 0, H∞ → spurious large values), which is precisely the failure mode of the six excluded artifact rows (giant repeated system prompts, e.g. the Aguvis/saital cases). BPC@32K, by contrast, declines monotonically and stays positive and ordered throughout. This is a controlled demonstration that **H∞ measures shared-scaffold pooling, and BPC@32K is the robust companion**, on data where the ground-truth content is held fixed.

Other limitations: β and Hurst are byte-level proxies (not the token/model-level definitions of [1, 2]); several `source` groups are small (human task n = 7, synth task n = 2); and the train/eval `role` label is editorial for the straddling human-demo benchmarks.

**Representativeness.** The registry is a convenience sample of what is publicly released as loadable text on HuggingFace, not a balanced design, and it skews accordingly: **SWE is ~36% of corpora and frontier-model rollouts are ~50%** (the rest spread over tool, terminal, web, search, mixed, gui, safety, embodied, and the singleton office/science/legal domains; by source, ~22% distill, ~15% mid, ~15% human demo/task). New loadable trajectories are now concentrated in SWE, and several domains of interest (data-science, SQL, scientific-discovery agents) have no text-trajectory releases at all (only screenshots or metadata for GUI/OSWorld). Two guards keep the headline claims from being artifacts of this skew: the content gap is reproduced **within** each domain that has both cohorts (§5.2), and the source-vs-role dissociation is **bootstrap-robust** (§5.1) — both control for composition. Absolute medians (e.g. the overall agentic H∞ ≈ 0.35) do reflect the sampled mix and should be read as "this corpus of corpora," not a population estimate.

### 8.1 Practical guidance: which statistic to trust

The survey reduces to a short decision procedure for anyone applying the probe to agentic data:

| You have… | Trust | Because |
| :-- | :-- | :-- |
| a training corpus (mixed sources) | reference-exact **H∞** | validated (Spearman 0.97) and cross-domain comparable; many sources ⇒ no single shared prompt to pool away |
| a benchmark **eval rollout** (one harness) | **BPC@32K + turn-count** | H∞ is harness-confounded — a heavy shared system prompt pools it to 0 regardless of generator (§5.3) |
| to compare **generators under a fixed harness** | **BPC@32K + turn-count** | orders by capability/success even when H∞ is flat at the floor (terminus-2 ladder, §5.3) |
| to rank **teachers within the healthy band** | a **proxy training run** | the probe is a regime selector, not a recipe selector; within-band gaps sit inside seed-σ (§5.4) |
| to assess **long-range organization** | **Hurst, paired with H∞** | Hurst alone conflates repetition with content (ρ ≈ 0 vs H∞; §4) |
| to predict **data-limited learnability** | **β** (→ α_D = γ/2β) | the one statistic with a learning-curve theory behind it [1] |

The single most important rule: **a clamped/extrapolated/pooled "0" is a hypothesis, not a finding** — falsify it against a directly measured quantity (BPC@32K), the turn-count, and synthetic controls before reporting it.

---

## 9. Forward Experiments

1. **α_D training validation.** Train small models (Cagnetta protocol [1]) on healthy-band vs template-band corpora and check whether the measured learning-curve exponent matches γ/2β. Confirmation gives the first *predictive* theory of agentic data value.
2. **Short-but-fractal → length generalization.** Fix the token budget and contrast short high-density human-action data (WebLINX/AgentNet action views, 1–2 KB/ep, H∞ 1.4–1.95) against long low-density failure loops (aider-flail 322 KB/ep, H∞ 0) on long-task extrapolation — testing "statistics > window length."
3. **Verifiable synthetic code → cross-domain transfer.** Formally verified synthetic code is unlimited, zero-human, and tunable in long-range structure; test whether natural-language long-task generalization tracks a corpus's (Hurst, γ, β) rather than its domain or size.
4. **A real decision test.** A ≥7B student and an *in-format, sandbox-verified* decision check (held-out next-command that must pass a held-out test) to give the §7 metric dynamic range. The harness (`scripts/formchoice/`) is ready to scale.

---

## 10. Conclusion

Merging training and evaluation corpora into one compression-oracle table separates two axes the literature usually conflates. **Pattern** (α, β, Hurst) is the genre signature of the agentic format — a distinct, low-β statistical phase, nearly constant across the train/eval boundary. **Content** (reference-exact H∞) is governed by who authored the token stream, falling monotonically from human task → human demo → frontier → mid/distilled, and this ordering ignores the train/eval label entirely. The practical consequence is a measurable **content gap**: we benchmark agents on human-authored richness (median H∞ 1.11) and train them on machine boilerplate (median 0.28). The cheap probe cannot pick a teacher inside the healthy band, but it identifies the healthy band itself in seconds (≈4.5 s/corpus) — the selection step that turns "collect more trajectories" into "collect the right ones." The dissociation is statistically robust (source explains more H∞ variance than role in 100% of bootstraps) and holds *within* every domain, not just in aggregate.

A second, methodological contribution is cautionary and transfers beyond agentic data: on single-harness eval rollouts the canonical 3-point H∞ measures the agent harness's shared scaffold as much as the generator — the *same* SWE-bench-Verified tasks span H∞ 0.00 → 1.22 across three harnesses — a confound we reproduce on a synthetic control (a 50% shared prefix clamps H∞ to 0 while BPC@32K holds). The general lesson: a compression floor read off a corpus with a large shared prefix reports the prefix, so directly-measured BPC@32K and turn-count are the readouts to trust there. Used with that caveat, the oracle is a practical, GPU-free first filter on long-horizon agentic data.

---

## Figures

**Figure 1.** Merged train+eval content map (`figures/fig_merge_content_source.png`, n = 99). (a) reference-exact H∞ by generator source, split by role — content tracks source, not the train/eval label; medians fall monotonically. (b) the α × H∞ plane — α is ~constant while H∞ spreads with source.

**Figure 2.** Signature clusters in the α × H∞ plane (`figures/fig1_signature_refhinf.png`): healthy long-trajectory, distillation-template, mid-size failure-loop, and compact human-action clusters, plus the benchmark-eval-rollout cohort (magenta), which splits between the H∞≈0 harness-pooled zone (terminal/tool/GAIA-mid) and the healthy ~1.0 SWE zone (SWE-Router) — a visual of the §5.3 harness effect.

**Figure 3.** γ–β phase plane (`figures/fig6_gamma_beta.png`): the ordered β bands — agentic model trajectories 0.15–0.52, code/math 0.52–0.79 (contiguous with agentic at the SWE seam), prose 1.11–1.37 (separated by an empty gap).

**Figure 4.** Hurst vs content (`figures/fig5_hurst_vs_content.png`): template and healthy corpora share a Hurst band; H∞ separates them.

**Figure 5.** View decomposition (`figures/fig3_view_decomposition.png`): stripping observations recovers content on web/GUI but removes it on SWE.

**Figure 6.** Form-vs-choices training result (`figures/fig7_formchoice_result.png`): FORM confirmed (base 0.78 → 0.96–0.99), DECISION inconclusive (all ~2–3%).

**Figure 7.** γ–β phase plane (`figures/fig9_gamma_beta_all.png`): agentic trajectories (diamonds), benchmark eval rollouts (✚, iters 69–73), and reference code/math/prose corpora under one byte-level protocol, colored by reference-exact H∞; contours are α_D = γ/2β. Agentic train and eval data share the low-β phase; code/math is the bridge to prose.

---

## References

[1] F. Cagnetta et al. *Deriving Neural Scaling Laws from the Statistics of Natural Language.* arXiv:2602.07488. (γ, β; α_D = γ/2β.)

[2] I. Alabdulmohsin, V. Q. Tran, M. Dehghani. *Fractal Patterns May Illuminate the Success of Next-Token Prediction.* arXiv:2402.01825, 2024. (Self-similarity, Hurst, downstream prediction.)

[3] *Intrinsic Entropy of Context Length Scaling in LLMs.* (Optimal context length grows with data; Bayes-risk fit = the oracle equation.)

[4] G. Delétang et al. *Language Modeling Is Compression.* arXiv:2309.10668. (Sequence modeling ⇔ compression.)

[5] J. Schmidhuber. *Driven by Compression Progress.* (Formal theory of curiosity; intermediate complexity is most learnable.)

[6] *Large Language Models and the Entropy of English.*

[7] *Reference protocol:* `reference/data_format.md` — the 358-dataset formal-math LZ-oracle survey (3-point clamped H∞; LZ↔neural H∞ Spearman 0.97) reproduced bit-for-bit.

*Supporting material: `SAMPLES.md` (full registry + cumulative findings), `reports/` (openthoughts, formchoice, lz_select, inferredbugs, hinf_clamp), `figures/FIGURES.md` (figure index), `data/merged_analysis.csv` (per-row table), and **Appendix A** (`paper/appendix_corpus_table.md`) — the complete per-corpus listing of all 104 active corpora with role, source, domain, α, H∞, BPC@32K, and turns.*
