# A unified interpretation of the agentic corpora — independent of train/eval

**Thesis:** The train/eval/benchmark *role* is not a property of the data — it is a
deployment decision. When you measure the corpora directly, role organizes almost
nothing. The 136 active corpora form a **continuum** on a small number of content
axes, each governed by a *different* property of the data (who made it, what it is
about), and the train/eval label is a near-invisible coordinate on that continuum.

All numbers below are on the 136 active corpora (`data/merged_analysis.csv`,
`EXCLUDED` dropped); reproduced by `scripts/unified_structure.py`. Centerpiece:
`figures/fig_unified_interpretation.png`.

---

## 1. Role does not structure the data

η² (variance of each metric explained by each label):

| metric | role (train/eval) | domain | source |
|---|---|---|---|
| α | 0.071 | 0.209 | **0.254** |
| H∞ | 0.090 | 0.067 | **0.324** |
| BPC@32K | 0.015 | **0.333** | 0.106 |
| length (log turns) | **0.285** | 0.233 | 0.420 |

**Role is the weakest organizer of every content metric.** The only thing role
tracks is episode length (eval trajectories run longer). For *what the data
contains*, role explains 1.5–9%.

Confirmation that this isn't an artifact of the η² estimator: the **silhouette**
of each label partition in the standardized 4-metric space is
- role **−0.017**, source **+0.005**, domain **−0.264**.

No label carves the space into clusters — least of all role. The corpora are a
**continuum**, and labels are weak coordinates, not partitions.

## 2. The continuum has ~3 near-independent axes

Spearman correlations among the metrics:

|  | α | H∞ | BPC@32K | length |
|---|---|---|---|---|
| α | 1.00 | 0.78 | 0.13 | 0.00 |
| H∞ | | 1.00 | 0.57 | −0.03 |
| BPC@32K | | | 1.00 | −0.03 |
| length | | | | 1.00 |

PCA of the standardized metric space:

| | PC1 (47%) | PC2 (25%) | PC3 (24%) |
|---|---|---|---|
| α | +0.54 | +0.54 | −0.36 |
| H∞ | +0.70 | +0.06 | +0.06 |
| BPC@32K | +0.45 | −0.54 | +0.54 |
| length | −0.13 | +0.64 | +0.76 |

Reading the axes:
- **Axis 1 — content richness (47%).** α, H∞, BPC@32K load together positively.
  This is the "how much genuine information per corpus" axis. H∞ is its purest
  marker; α is a near-redundant curve-shape correlate (ρ=0.78); BPC@32K is the
  within-window companion (ρ=0.57).
- **Axis 2 — length (25%).** Episode length is **orthogonal** to content
  (ρ≈0 to α, H∞, BPC). *A corpus being long tells you nothing about whether it is
  content-rich.* This is the single most counter-intuitive structural fact.
- **Axis 3 — within-window density vs length (24%).** BPC@32K trades off against
  length: short-but-dense vs long-but-dilute episodes.

Hurst (n=25 subset) is a fourth, weakly-coupled axis: ρ(Hurst, H∞) = −0.15,
ρ(Hurst, α) = −0.26 — long-range dependence is largely its own thing.

## 3. Each axis is governed by a different property — none of them role

- **Content richness ← SOURCE.** Who generated the trajectory sets its
  cross-episode diversity (H∞ η²(source)=0.32). Human-authored > frontier-model >
  distilled/templated.
- **Within-window density ← DOMAIN.** What the task is about sets the local
  information rate (BPC@32K η²(domain)=0.33). SWE/search are dense; scaffold-heavy
  terminal is dilute.
- **Length ← role + source.** The one place role appears — and even here source
  matters more (η² 0.42 vs 0.29).

So the "train vs eval" question dissolves: the meaningful questions are **who made
it** (→ content) and **what domain** (→ density), with length as a free third knob.

## 4. The real "kinds" of agentic data cut across role

Unsupervised k-means (k=4) on the metric space produces groups that **mix
train/eval freely** (every cluster contains TRAIN and EVAL corpora). The natural
kinds, by median signature:

| kind | α | H∞ | BPC@32K | turns | typical domains |
|---|---|---|---|---|---|
| **pooled / degenerate** | 0.18 | 0.00 | 1.53 | 18.8 | swe, tool, terminal |
| **low-density** | 0.32 | 0.12 | 0.93 | 14 | web, swe, mixed |
| **short + dense** | 0.31 | 1.05 | 2.00 | 3 | swe, tool, web |
| **long + content-rich** | 0.30 | 1.01 | 1.91 | 40 | swe, terminal, search |

These four kinds — defined by **content × length × density**, not by train/eval —
are the honest taxonomy of the corpus collection.

## 5. What this frame can and cannot claim (held-out check)

The associational picture above is robust. But a stronger reading — "these metrics
*replace* a provenance label" — fails a held-out test. Leave-one-corpus-out source
classification (`scripts/validate_source_attribution.py`): a **domain-only**
baseline (balanced acc 0.57 coarse) beats **H∞+BPC@32K** (0.39); H∞ alone is at
chance. **Association ≠ separability**: source shifts H∞ measurably, but not enough
to out-classify cheap domain metadata out-of-sample. The defensible claim is
*"content richness is governed by source"*, not *"H∞ is a source classifier"*.

## 6. Where the frame is incomplete — candidate axes to add

The current axes are all byte-level. The unified frame would be sharpened by a few
*structural* axes that are plausibly independent and still cheaply measurable
(full space in `reports/metric_space_brainstorm.md`):

1. ⚠️ **reasoning-to-action ratio** — *attempted, marker-based version infeasible*:
   explicit `Thought:`/`Action:` markers occur in only 12%/7% of corpora, so a
   marker ratio would be format-heterogeneity noise. Measured a **format-free
   proxy instead** — `structure_density` = non-alphabetic char fraction (high in
   JSON/code/tool serializations, low in reasoning prose; `scripts/measure_
   structure_density.py`, `data/structure_density.csv`, n=102; figure
   `figures/fig_structure_density.png`). **It is a genuinely new near-independent
   axis**: ρ(structure, H∞)=**+0.08** (orthogonal to content), and no label
   organizes it (η² role/domain/source = 0.09/0.10/0.11 — *not* a domain proxy).
   A suggestive negative link to β (ρ=−0.53) is **underpowered (n=12)** so the
   "explains β" idea stays open. Verdict: a real serialization-symbol-heaviness
   axis, orthogonal to content — but its "reasoning vs action" reading is loose.
2. ✅ **scaffold/boilerplate fraction** — *measured* (`scripts/measure_scaffold.py`,
   `data/scaffold_frac.csv`, n=102 active corpora; byte share of lines shared
   across ≥2 cached episodes; figure `figures/fig_scaffold_pooling.png`). It
   operationalizes the pooling cause of H∞=0 as a coordinate: the **H∞=0 pooled
   cluster carries 2.5× the scaffold of healthy corpora** (median 0.246 vs 0.100),
   it is **orthogonal to length** (ρ=−0.05), and it retains signal among content-
   dense corpora (ρ(scaffold, H∞)=−0.32 at BPC@32K>1) — i.e. it explains some
   H∞=0 *beyond* low within-window density. It is a **modest** standalone predictor
   (ρ(scaffold, H∞)=−0.34 overall) and is estimated from only 3 cached episodes, so
   it is a directional coordinate, not a precise one. Verdict: a real, partially-
   independent pooling axis — it makes the pooled/degenerate "kind" (Sec 4) a
   *measured* region rather than an inferred one.
3. **credit-assignment horizon** — the long-horizon-specific axis; test whether it
   is genuinely captured by Hurst or independent. *(not yet measured)*
4. ✅ **near-duplication** — *measured* (`scripts/measure_neardup.py`,
   `data/neardup.csv`, n=102; mean pairwise word-5-gram Jaccard across cached
   episodes). **It is not a new axis — it collapses onto scaffold** (ρ=+0.84), but
   it is the *stronger estimator* of the redundancy/pooling dimension
   (ρ(neardup, H∞)=−0.48 vs scaffold's −0.34; ⊥ length at +0.03). So redundancy is
   one robust axis, best measured by shingle-Jaccard. (Contamination *vs an external
   training set* — the eval-validity sense — remains unmeasured; needs a reference
   corpus.)

## 7. The consolidated frame — ~5 measured dimensions

Putting all measured axes together (`scripts/extended_axes.py`,
`figures/fig_extended_axes_corr.png`, n=102 corpora with all axes), a PCA shows
**5 dimensions capture 90% of the variance**. The Spearman structure resolves them:

| dimension | markers | independence |
|---|---|---|
| 1. content richness | H∞, α (ρ=0.78) | the primary axis |
| 2. within-window density | BPC@32K | partly separate from #1 (α–BPC only 0.13) |
| 3. length | turns | **fully orthogonal** (all \|ρ\|≤0.21) |
| 4. structure / serialization | structure_density | near-orthogonal (max \|ρ\|=0.29) |
| 5. redundancy / pooling | neardup ≈ scaffold (ρ=0.84) | anti-correlated w/ content, distinct |

This is the unified interpretation in compact form: **agentic corpora occupy a
~5-dimensional content/format space — and the train/eval role is not one of its
dimensions.** Role re-enters only as a weak loading on #3 (length). The same four
"kinds" of Sec 4 are simply dense regions of this space, defined by content ×
length × density × redundancy, with structure as a near-free fifth knob.

The gold *benchmark* metrics (empirical difficulty, discrimination, verifiability)
require model rollouts or oracle access — they mark the boundary of what a
compression-oracle analysis can say.

---

**One-sentence interpretation:** Agentic corpora are not "training data" or
"benchmarks" — they are points on a content-richness × length × density continuum,
where richness is set by *who generated the data* and density by *what domain it
covers*, and the train/eval label is a downstream choice that the data itself does
not encode.
