# H∞ measurement failure & correction (2026-06-07)

Triggered by the user's challenge: *"OpenThoughts should not score 0."* It shouldn't, and chasing why exposed that the H∞ metric was unreliable across the registry.

## The metric (as it was)
Concatenate episodes → one corpus. Measure BPC(n) = bits/char when zstd-19 compresses independent n-byte chunks, at n=128/2048/32768. Fit `BPC(n)=H∞ + c·n^(−α)` analytically over the 3 points; report `H∞ = max(fit, 0)`.

## Three compounding failures
1. **Negative-clamp.** For compressible data BPC is still falling steeply at 32 KB (the largest n measured), so the power-law floor extrapolates **negative** and `max(·,0)` reports a fake `0.000`. **All 37 "H∞=0" rows are negative clamps** (raw −0.04 to −17), none a measured zero.
2. **Non-convergent curve.** Extending to 524 KB, many corpora (e.g. swe-zero: 5.68→4.05→3.15→2.57→1.93→0.95→0.54) **never flatten** — the n→∞ floor is not in the observable window, so *any* extrapolator is fitting an unseen asymptote. My attempted fix `score_v2` (least-squares floor scan) then **pegs at its scan bound** (swe-zero → −1.0, a healthy dataset) — a new artifact.
3. **Cross-episode pooling.** A large chunk spans many episodes; the shared system-prompt/JSON scaffold repeats every episode and zstd crushes it across episodes. The pooled floor measures **dataset-level boilerplate density**, not per-episode content. OpenThoughts pooled ≈ 0; **descaffolded H∞ = 0.75 (healthy band)** — the content was always there.

## Synthetic-control validation
Built corpora with known structure:

| control | truth | BPC@32K pooled | BPC@32K stripped |
| :-- | :-- | --: | --: |
| random words | high content | **2.47** ✓ | 2.47 |
| identical template | ~0 | **0.02** ✓ | **8.00** ✗ (inverts) |
| scaffold+content | medium | 1.40 | 2.36 |

- **Directly-measured BPC@32K is correct** on all three.
- **Line-stripping is unsafe**: on pure template it *inverts* (removing all shared lines leaves incompressible fragments → 8.0). So strip-based numbers (incl. the 0.75 above, and LZ-Select's H∞ gains) are directionally suggestive but not canonical.

## Decision
- **Deprecate** extrapolated H∞ (clamp/peg/non-convergence) and line-stripping (inversion).
- **Adopt directly-measured `BPC@32768`** as the content-density metric: no fitting, no clamp, no extrapolation, far less pooling contamination (a 32 KB chunk spans ~1–3 episodes, not ~40), already in the CSV for all rows, passes all controls. Lower = more templated, higher = more content.
- Keep `score_v2` only as a diagnostic; it is **not** the canonical metric.

## Real-registry effect
BPC@32K ranks sensibly — most-templated lowest (aider-7B 0.50, agentgym 0.67, agentbank 0.55), most-content highest (ii-agent-GAIA 3.05, GDPval 2.87, Opus-4.8 2.57). **5 datasets reported H∞=0 actually carry high content**: nemotron-tool 2.54, deep-research-sft 2.45, qwen35-react 2.38, nemotron-conv-pivot 2.30, fractal 2.29. The "template band = H∞=0" cluster was partly a measurement artifact.

## Findings to revise (not discard — ordering mostly survives)
- **4 / 8 / 17**: restate "H∞=0 template band" as "low BPC@32K"; the band is a smear, not a point at 0; boundary fuzzier.
- **20 (most affected)**: OpenThoughts is **not** content-empty (descaffold 0.75). "Template recipe = pure form" premise weakened; the running form-vs-choices SFT contrast is muddier than designed (both arms carry content) — interpret its result with this caveat.

## Lesson
Any "exactly 0" produced by a clamp, an extrapolation, or a pooled measurement must be falsified against a **directly measured quantity** and **synthetic controls** before it becomes a finding. We reported 37 of them.

## TODO
- Re-derive §V signature clusters and figs 1/4/8/9 on BPC@32K (content axis) instead of H∞.
- Re-examine whether per-episode (un-pooled) BPC is a better content metric than pooled BPC@32K for short-episode datasets.
