# Figure index (canonical set — corrected metric BPC@32K, 2026-06-07)

All content-axis figures use **directly-measured BPC@32768** (bits/char at 32 KB
context). The extrapolated H∞ was deprecated (ill-conditioned; see
`reports/hinf_clamp_fix.md` and SAMPLES.md correction block). Old H∞-based
figures are archived in `deprecated_hinf/`.

- **fig1_signature_map** — BPC@32K (content) × α (structure), all datasets by category
- **fig2_content_ranking** — datasets ranked by content density (BPC@32K)
- **fig3_view_decomposition** — strip observations: content recovered (web/GUI) vs lost (SWE)
- **fig4_horizon_vs_content** — bytes/episode vs content: long ≠ rich
- **fig5_hurst_vs_content** — Hurst vs content: Hurst alone can't rate data
- **fig6_gamma_beta** — γ–β phase plane, colored by content density
- **fig10_openthoughts** — OpenThoughts reproduction (3-panel; note: H∞ panel pre-correction)
