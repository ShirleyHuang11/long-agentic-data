# paper/ — merged long-horizon agentic data analysis

- **`long_horizon_agentic_data.md`** — the paper. *A Compression-Oracle Survey of
  Long-Horizon Agentic Data: Merging Training and Evaluation by Pattern and Content.*
- **`appendix_corpus_table.md`** — Appendix A: auto-generated per-corpus table
  (all active corpora × role/source/domain/α/H∞/BPC@32K/turns).

## Headline result

Merging training and evaluation corpora (133 active / 139 scored rows) dissociates
two axes:
- **Pattern** (α, β, Hurst) is the agentic-*format* genre signature — role-invariant
  α (median 0.25–0.34), a distinct low-β phase (0.2–0.5 vs prose 1.1–1.4, code/math
  bridge 0.5–0.8).
- **Content** (reference-exact H∞) tracks the **generator source, not the train/eval
  role** — η² 0.33 vs 0.09, robust in 100% of 2000 bootstraps — exposing a
  within-domain **content gap** (human-authored eval-task 1.11 vs model-generated
  training 0.26).
- **Caveat (validated):** on single-harness eval rollouts H∞ measures the agent
  scaffold (pooling), not the generator; use BPC@32K + turn-count there. Demonstrated
  on a synthetic control (`scripts/synth_harness_pooling.py`).

## Data and reproduction

| Artifact | Source of truth | Regenerate with |
| :-- | :-- | :-- |
| per-row classified table | `data/merged_analysis.csv` | `python scripts/build_merged_table.py` |
| canonical (α, H∞) scores | `data/agentic_alpha_hinf.csv` | `scripts/lz_oracle.py` (reference-exact 3-point, zstd-19) |
| β / Hurst | `data/gamma_beta*.csv`, `data/hurst.csv` | `scripts/measure_beta*.py`, `scripts/measure_hurst.py` |
| Figure 1 (merge) | — | `python figures/make_merge_figure.py` |
| Figure 7 (γ–β plane) | — | `python scripts/make_phase_all.py` |
| Figure 2 (signature map) | — | `python scripts/make_figures_refhinf.py` |
| Figure 8 (neural validation) | `data/neural_oracle_bpc.csv` | `sbatch scripts/run_neural_oracle.sh` (GPU) → `python scripts/make_neural_oracle_figure.py` |
| Appendix A | — | `python scripts/make_appendix_table.py` |
| η² bootstrap | — | `python scripts/bootstrap_eta2.py` |

The canonical content metric is the reference-exact 3-point clamped H∞
(`reference/data_format.md`, validated against true neural oracles at Spearman 0.97);
BPC@32K and `score_v3` are supplementary companions used where H∞ is harness-pooled.

Full registry, cumulative findings, and the iteration log live in `../SAMPLES.md`;
the plain-language digest is `../CONCLUSIONS.md`; figures are indexed in
`../figures/FIGURES.md`.
