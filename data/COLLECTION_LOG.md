# Collection Log — long-horizon agentic data

Every collection step, data source, and protocol decision is recorded here.
Per-dataset machine-readable provenance (pinned HF revision SHA, sampling
params, oracle params, raw scores, UTC timestamp) lives in
`data/provenance/<slug>.json`. Raw serialized episode excerpts (first 3
episodes per dataset, exactly as fed to the oracle) live in
`samples_cache/<slug>.txt`.

## Protocol (fixed before any scoring)

- **Template:** mirrors the formal-math survey `data_format.md` — registry
  tables of `(数据集, 类别, 规模, 特点, α, H∞, HF 镜像)` accumulated over
  iterations, with an LZ data oracle.
- **Oracle:** `scripts/lz_oracle.py` — zstd level 19, independent n-byte
  chunks at n₁=128 / n₂=2048 / n₃=32768 (geometric, r=16), per-chunk header
  overhead subtracted; 3-point analytical fit of BPC(n) = H∞ + c·n^(−α):
  α = log₁₆((B₁−B₂)/(B₂−B₃)), H∞ = B₃ − (B₂−B₃)/(16^α − 1).
- **Corpus per dataset:** up to 1500 episodes or 8 MB (whichever first),
  episodes joined with `\n\n`. One *document* = one serialized episode
  (full trajectory), turns rendered as `[role]\ntext` blocks.
- **Sampling:** HF `streaming=True`, first-N episodes per split, no shuffle
  (single-seed pilot; multi-seed σ deferred to full collection, as in the
  template's iter-22 protocol).
- **Real data:** all rows are streamed live from the HF Hub at collection
  time; nothing is synthesized or imputed.

## Step log

| # | date (UTC) | step | tool/script | artifact |
| :- | :--- | :--- | :--- | :--- |
| 1 | 2026-06-05 | Read template `data_format.md` (4171 lines), extracted registry format + oracle protocol | manual | — |
| 2 | 2026-06-05 | Environment: python 3.12 venv, `datasets==5.0.0`, `zstandard`; verified HF Hub reachable | `.venv/` | — |
| 3 | 2026-06-05 | Implemented LZ oracle (port of the survey's 3-point protocol) | `scripts/lz_oracle.py` | — |
| 4 | 2026-06-05 | Oracle sanity checks: random printable ASCII → H∞=6.64 (≈log₂94 ✓, incompressible); templated code → H∞→0 (template-degenerate signature ✓); `data_format.md` itself → α=0.20, H∞=1.40 (NL-range ✓) | inline | — |
| 5 | 2026-06-05 | Schema probe of 6 candidate datasets (1 streamed row each); fixed split/config names for SWE-smith (`xml`) and AgentInstruct (`default` config, per-env splits) | inline | — |
| 6 | 2026-06-05 | Pilot registry + serializers + provenance sidecars | `scripts/score_agentic_datasets.py` | `data/provenance/*.json` |
| 7 | 2026-06-05 | Pilot scoring run, part 1: nebius-swe-agent-traj, agentinstruct-all, agentgym-agenttraj-l, swe-bench-verified scored OK (provenance sidecars written) | `scripts/score_agentic_datasets.py` | `data/provenance/*.json`, `samples_cache/*.txt` |
| 8 | 2026-06-05 | **Incident:** `/n/home12` hit 100% (95G/95G, pre-existing user data — not this session); CSV write + Mind2Web stream died with `ENOSPC`. Cleared session HF cache (~700M) from home; still 0 free | manual | — |
| 9 | 2026-06-05 | Relocated working area + `HF_HOME` to `/n/netscratch/chen_lab_seas/Lab/shirleyhuang/long-agentic-data` (rsync of repo files, excl. `.git`/`.venv`); repo in home left intact, to be synced back when space frees | rsync | netscratch working copy |
| 10 | 2026-06-05 | Fixed SWE-smith serializer (`messages` is a JSON-encoded string, not a list) and scored it; Mind2Web rescored from netscratch | `scripts/score_agentic_datasets.py` | `data/provenance/{swe-smith-traj-xml,mind2web-actions}.json` |
| 11 | 2026-06-05 | Rebuilt CSV from provenance sidecars (sidecar = source of truth after incident) | `scripts/rebuild_csv.py` | `data/agentic_alpha_hinf.csv` |
| 12 | 2026-06-05 | Sample registry entries written for review before full collection | manual | `SAMPLES.md` |

## Data sources (pilot)

| source | what it is | config/split used | serialization | license note |
| :--- | :--- | :--- | :--- | :--- |
| [`nebius/SWE-agent-trajectories`](https://huggingface.co/datasets/nebius/SWE-agent-trajectories) | 80k SWE-agent issue-fixing trajectories (Nebius) | `train` | join `trajectory[*]` as `[role]\ntext` | CC-BY-4.0 (per card) |
| [`SWE-bench/SWE-smith-trajectories`](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories) | 5k trajectories used to train SWE-agent-LM-32B | `xml` split | join `messages[*]` as `[role]\ncontent` | MIT (per card) |
| [`THUDM/AgentInstruct`](https://huggingface.co/datasets/THUDM/AgentInstruct) | AgentTuning's 1.8k GPT-4 trajectories over 6 envs (OS, DB, ALFWorld, WebShop, Mind2Web, KG) | splits `os+db+alfworld+webshop+mind2web+kg` | join `conversations[*]` as `[from]\nvalue` | research-only (per card) |
| [`AgentGym/AgentTraj-L`](https://huggingface.co/datasets/AgentGym/AgentTraj-L) | AgentGym's 14k trajectories across 14 envs | `train` | join `conversations[*]` as `[from]\nvalue` | per card |
| [`princeton-nlp/SWE-bench_Verified`](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified) | 500 human-verified SWE-bench instances (benchmark, not trajectories) | `test` | `problem_statement + hints + patch + test_patch` | per card |
| [`osunlp/Mind2Web`](https://huggingface.co/datasets/osunlp/Mind2Web) | 2,350 web tasks / 137 sites | `train` | `confirmed_task + action_reprs` (compact view; raw `cleaned_html` observations **excluded** — MB-scale per step would let 2–3 episodes fill the 8 MB corpus) | OPEN-RAIL (per card) |

### Deferred to full collection (with reasons)

| source | reason deferred |
| :--- | :--- |
| `gaia-benchmark/GAIA` | gated — needs HF token + terms acceptance |
| WebArena / VisualWebArena | environment on GitHub, no canonical HF trajectory release; needs trajectory generation or third-party dumps |
| OSWorld / AgentNet | needs schema work (screenshots; multimodal) |
| `SWE-Gym/*` trajectory releases, OpenHands trajectories, `ToolBench`, `AgentBank`, tau-bench, AppWorld, TheAgentCompany | full-collection sweep, same pipeline |

## Honest caveats (pilot)

1. Single seed, first-N sampling (no shuffle) — ordering bias possible if the
   upstream dataset is sorted; template's protocol calls for 5-seed σ at
   publication quality.
2. Mind2Web scored on the *compact action view*, not raw HTML observations —
   α/H∞ not comparable to a hypothetical full-observation slice.
3. Oracle is a re-implementation from the protocol description in
   `data_format.md` (original `compute-free/hurst/lempel-ziv.py` not available
   in this repo); absolute α/H∞ may shift vs the original, relative ranking
   within this registry is internally consistent.
