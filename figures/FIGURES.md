# Figure index (canonical set — corrected metric BPC@32K, 2026-06-07)

All content-axis figures use **directly-measured BPC@32768** (bits/char at 32 KB
context). The extrapolated H∞ was deprecated (ill-conditioned; see
`reports/hinf_clamp_fix.md` and SAMPLES.md correction block). Old H∞-based
figures are archived in `deprecated_hinf/`.

- **fig_merge_content_source** — *(paper centerpiece, reference-exact H∞)* merged train+eval (n=98): (a) H∞ by generator source split by role — content tracks source not train/eval role; (b) α×H∞ plane — α ~constant, H∞ spreads with source. See `paper/long_horizon_agentic_data.md`.
- **fig1_signature_refhinf / fig2_ranking_refhinf** — canonical reference-exact H∞ signature map + content ranking (per iter-68; BPC@32K versions below are supplementary finer-resolution views).
- **fig1_signature_map** — BPC@32K (content) × α (structure), all datasets by category
- **fig2_content_ranking** — datasets ranked by content density (BPC@32K)
- **fig3_view_decomposition** — strip observations: content recovered (web/GUI) vs lost (SWE)
- **fig4_horizon_vs_content** — bytes/episode vs content: long ≠ rich
- **fig5_hurst_vs_content** — Hurst vs content: Hurst alone can't rate data
- **fig6_gamma_beta** — γ–β phase plane, colored by content density
- **fig9_gamma_beta_all** — *(canonical, reference-exact H∞; paper Figure 7; `scripts/make_phase_all.py`)* detailed γ–β phase plane: agentic trajectories (◆) + benchmark eval rollouts (✚, iters 69–73) + reference code/math/prose corpora under one byte-level protocol; contours α_D=γ/2β. **Regenerate every loop iteration** as new samples gain β (`scripts/measure_beta_new.py` → `data/gamma_beta.csv`). Old BPC@32K-era copies archived in `deprecated_hinf/`.
- **fig10_openthoughts** — OpenThoughts reproduction (3-panel; note: H∞ panel pre-correction)
- **fig7_formchoice_result** — form-vs-choices training result (finding 21): FORM confirmed, DECISION inconclusive
