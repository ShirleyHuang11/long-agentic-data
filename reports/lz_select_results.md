# LZ-Select pilot results (Kwai-Klear mini-swe 66k)

End-to-end run of the 3-stage LZ-Select pipeline (`scripts/lz_select.py`) on
`Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k`
(slug `kwai-klear-mini-swe-66k`), first **n=300** streamed episodes.

Pipeline stages:
1. **STRIP** — drop duplicate system turn(s) (modal leading `\n\n`-block shared
   by ≥50% of episodes) + drop lines appearing verbatim in >50% of episodes.
2. **SELECT** — per-episode `template_gain` (sibling-dictionary compression) and
   `loop_gain` (2nd-half | 1st-half conditional compression); drop the worst
   quartile of *each* (reused from `episode_filter_pilot.py`).
3. **GATE** — score the corpus with `lz_oracle` → (alpha, H_inf).

## Comparison table

| stage         | episodes (corpus) | alpha | **H_inf** | total MB | scored bytes |
|---------------|------------------:|------:|----------:|---------:|-------------:|
| raw           | 300               | 0.234 | 0.262     | 17.2     | 8.0 MB (cap) |
| strip-only    | 300               | 0.256 | 0.355     | 14.5     | 8.0 MB (cap) |
| select-only   | 177               | 0.243 | 0.406     | 10.7     | 8.0 MB (cap) |
| **strip+select** | **176**        | **0.264** | **0.505** | 9.0  | 8.0 MB (cap) |

Matched-size note: every stage's corpus exceeds the oracle's 8 MB cap, so all
four (alpha, H_inf) numbers are measured on a **full 8 MB scored corpus** — a
clean matched-size comparison. Re-running with each stage truncated to the same
176 episodes gives identical numbers (raw 0.262 → strip+select 0.505),
confirming the H_inf gain comes from boilerplate removal + quality selection,
not from packing more episodes into the cap.

## Prediction

Standing prediction: **strip+select reaches H_inf ≥ 0.6.**
Result: **NOT met** — strip+select H_inf = **0.505**.

It nearly doubled the raw floor (0.262 → 0.505, +0.243) and cleared the prior
pilot's best (select-only kept H_inf ≈ 0.41), but fell ~0.10 short of 0.6.

## Stage attribution

- STRIP alone: 0.262 → 0.355 (+0.093). Removes the identical `[system]` turn and
  the large verbatim task-instruction block inside the first `[user]` turn
  (`# Task Instructions`, format examples, "CRITICAL REQUIREMENTS", etc.). Drops
  ~16% of bytes (17.2 → 14.5 MB).
- SELECT alone: 0.262 → 0.406 (+0.144). Drops worst-quartile template/loop
  episodes (177 of 300 kept).
- The two stages are **roughly additive** (+0.093 and +0.144 → combined +0.243),
  i.e. they target largely independent redundancy: STRIP removes
  *within-episode cross-episode* boilerplate; SELECT removes whole episodes that
  are *template-heavy or self-looping*.

## Caveats (honest)

- **Single dataset, n=300.** Only Kwai-Klear was run; SWE-smith trajectories are
  unusually boilerplate-heavy (fixed system prompt + a long fixed instruction
  block), so the STRIP gain here is likely an upper end. Generalization unknown.
- **8 MB cap dominates corpus size**, not episode count — fine for a matched-size
  read, but it means H_inf reflects the *first* ~9–17 MB of streamed (unshuffled)
  episodes, not a random sample of the 66k corpus.
- **STRIP operates on rendered text**, not raw turn objects: a "turn" = a
  `\n\n`-block and a "line" = a `\n`-line, matching the registry serializers.
  Near-duplicate system turns are matched on whitespace-normalized text only
  (no fuzzy/edit-distance), so heavily templated-but-not-identical system turns
  with per-episode interpolation would not be caught.
- **Line dedup uses per-episode line *sets*** (a line repeated within one episode
  counts once toward the >50% cross-episode threshold), so genuine in-episode
  loops are preserved for the SELECT stage to judge via `loop_gain`.

## Artifacts

- CLI: `scripts/lz_select.py`
- Stage comparison rows: `data/lz_select_pilot.csv`
