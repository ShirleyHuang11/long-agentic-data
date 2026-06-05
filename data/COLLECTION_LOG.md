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
| 7 | 2026-06-05 | Pilot scoring run (6 datasets, streaming) | `scripts/score_agentic_datasets.py` | `data/agentic_alpha_hinf.csv`, `samples_cache/*.txt` |
| 8 | 2026-06-05 | Sample registry entries written for review before full collection | manual | `SAMPLES.md` |
| 8a | 2026-06-05 | (from parallel pilot session, merged at push time) Pilot scoring part 1 OK; then **incident**: `/n/home12` hit 100% (95G/95G, pre-existing user data) — CSV write + Mind2Web stream died with `ENOSPC`; session HF cache (~700M) cleared | manual | — |
| 8b | 2026-06-05 | (parallel session) Working area + `HF_HOME` relocated to `/n/netscratch/chen_lab_seas/Lab/shirleyhuang/long-agentic-data`; SWE-smith serializer fixed there (`messages` = JSON-encoded string) and Mind2Web rescored; CSV rebuilt from provenance sidecars via `scripts/rebuild_csv.py` | rsync + `scripts/rebuild_csv.py` | netscratch copy, `data/agentic_alpha_hinf.csv` |
| 8c | 2026-06-05 | History note: that session pushed its pilot as 20c1308; this branch (loop iters 1–18) was rebased on top at push time. For the 6 overlapping files the loop branch's versions were kept (supersets); swe-smith/mind2web score deltas between the two single-seed runs are within ordering-bias expectations (both pinned in provenance) | git rebase | — |
| 9 | 2026-06-05 | Loop iter 1: SAMPLES.md created (step 8 had been logged but file was never written); re-scored the 2 missing pilot entries (swe-smith-traj-xml α=0.285 H∞=0.555, 88 eps × 64.2 turns × 96 KB; mind2web-actions α=0.420 H∞=1.696); CSV rebuilt from provenance sidecars (6 rows) | `scripts/score_agentic_datasets.py --only` | `SAMPLES.md`, `data/agentic_alpha_hinf.csv` |
| 10 | 2026-06-05 | Hub existence check for 6 backlog candidates (SWE-Gym, SWE-rebench, code-act, AgentBank, WebLINX, Lumos) — all exist; revision SHAs recorded in SAMPLES.md backlog | `huggingface_hub` | `SAMPLES.md` |
| 11 | 2026-06-05 | Loop iter 2: schema probes (SWE-Gym + SWE-rebench are *task* sets, not trajectories; R2E-Gym is task-generation data) + scored 8 new entries incl. 3 OpenHands trajectory dumps. New serializers `ser_trajectory_auto`/`ser_messages_auto`/`ser_conversations_auto` (auto-detect role/content vs from/value vs role/text keys). `swe-rebench-oh-traj` is the new longest-horizon entry: 127.9 turns, 196 KB/ep, α=0.294, H∞=0.673. NB: one run crashed at interpreter exit (PyGILState, pyarrow streaming teardown) *after* writing provenance+CSV — artifacts verified intact. NNetNav-WA caveat: rows are per-step SFT samples (overlapping prefixes), H∞=0.46 reads low for that reason | `scripts/score_agentic_datasets.py --only` ×8 | `data/provenance/*.json` (14), `data/agentic_alpha_hinf.csv` (14 rows) |
| 12 | 2026-06-05 | Loop iter 3: scored 7 more (SWE-Hero 125 turns/141 KB·ep⁻¹; AgentBank 19-config merge; Lumos; nnetnav-live; OpenHands Sampled + Verifier; WebLINX group-by-`demo` compact action view). Script gains multi-config iteration (`cfg` may be a list) and consecutive-row episode grouping (`group_key`, 6th registry tuple element). Findings: success-filtering ablation Δα=0.015 (Sampled vs SFT); WebLINX H∞=1.948 highest in registry; nnetnav-live H∞=1.37 vs nnetnav-wa 0.46 (real web ≫ simulated WebArena diversity). Registry: 21 datasets | `scripts/score_agentic_datasets.py --only` ×7 | `data/provenance/*.json` (21), `data/agentic_alpha_hinf.csv` |
| 13 | 2026-06-05 | Loop iter 4 (tool-use): probed 6 candidates, scored 3, registered 1. `taubench-traces-jkazdan` → registry (α=0.263, H∞=0, n=50 low-sample caveat). Excluded with raw scores kept in CSV/provenance: `taubench-sonnet-proxy` (whole dump = ONE proxy session → 1 episode after thread_id grouping, 29KB corpus < 32KB oracle chunk, fit invalid) and `rlenv-appworld-train` (90 near-identical task prompts, not rollouts; 3-point fit artifact α=−0.166/H∞=9.94 — same artifact class the math survey's anomaly filter catches). ToolBench mirrors rejected: Adorg (split format mismatch, unloadable), Maurus (retrieval corpus: query+api_list+embedding, not trajectories). New serializer `ser_taubench_proxy` (HTTP-log → conversation via thread_id grouping + longest-body heuristic) | `scripts/score_agentic_datasets.py --only` ×3 | `data/provenance/*.json` (24), CSV 24 rows; registry 22 valid |
| 14 | 2026-06-05 | Loop iter 5: scored ToolLLaMA-DFS (α=0.192, H∞=0, 860 eps sampled) and OpenHands/openhands-feedback (α=0.361, H∞=0.745, 123 eps × 61.8 turns × 205,677 B/ep — longest bytes/ep in registry; new `ser_oh_events` for {action,content,extras} event streams). Rejected after content checks: yoonholee/terminalbench-trajectories (30/30 rows steps=null) and harithoppil/terminal-bench-2-trajectories (pass config 20/20 rows response="") — metadata-only shells. OS-Genesis gated (needs HF token); Aguvis deferred (image decode required). Registry: 24 valid | `scripts/score_agentic_datasets.py --only` ×2 | `data/provenance/*.json` (26), CSV 26 rows |
| 15 | 2026-06-05 | Loop iter 6 (full-observation protocol): `ser_mind2web_fullobs` (per-step cleaned_html capped at HTML_CAP=8192 chars — raw is 37–240KB/step, uncapped 2-3 episodes would fill the 8MB corpus) and `ser_weblinx_fullobs` (pruned DOM ~1.6-4.3KB/turn, included whole). Scored: mind2web-fullobs α=0.427/H∞=0.297 (129 eps), weblinx-fullobs α=0.203/H∞=0.000 (116 eps). Key finding: folding observations in collapses H∞ (Mind2Web 1.70→0.30, WebLINX 1.95→0.00) — web observations are near-pure long-range template redundancy; compact-action vs full-obs views are not comparable. Discovered Nemotron agentic family + APIGen-MT-5k → queue. Registry: 26 valid | `scripts/score_agentic_datasets.py --only` ×2 | `data/provenance/*.json` (28), CSV 28 rows |
| 16 | 2026-06-05 | Loop iter 7 (Nemotron family + APIGen-MT): 7 runs, 4 registered (SFT-v2-interactive α=0.167/H∞=0.26; RL-SWE-Pivot α=0.207/H∞=0.59, 90 turns/step-level; RL-Conv-Tool-Pivot α=0.048/H∞=0 near-pure template; APIGen-MT-5k α=0.178/H∞=0). Failed/excluded: SFT-v2 `search` split (CastError, heterogeneous schema upstream), SFT-v2 `tool_calling` split (JSON decode error upstream), Nemotron-Agentic-v1 (fit artifact α=−0.079/H∞=15.17, same class as rlenv-appworld). New `ser_nemotron_rl` (responses_create_params.input + expected_action). Registry: 30 valid | `scripts/score_agentic_datasets.py --only` ×7 | `data/provenance/*.json` (33), CSV 33 rows |
| 17 | 2026-06-05 | Loop iter 8: +3 registered — agent-flan-all (α=0.113/H∞=0; 1500-ep cap hit inside first split, react-split-dominated sample), scienceworld-expert-traj (repo `lclan/webshop_expert_trajectories` is mislabeled — rows are ScienceWorld episodes; α=0.251/H∞=0, 45.9 turns), fireact-multitask (α=0.397/H∞=1.802 — 2nd-highest H∞ in registry). Rejected: OpenWebVoyager-IL (SplitsNotFoundError, broken repo). weblinx-browsergym ReadTimeout — retry queued. New `ser_conversation_auto` (singular `conversation` field). Registry: 33 valid | `scripts/score_agentic_datasets.py --only` ×3 | `data/provenance/*.json` (36), CSV 36 rows |
| 18 | 2026-06-05 | Loop iters 9–10: §V summary section written (horizon ranking, α×H∞ signature clusters, 5 cumulative findings). +1 registered: r2e-gym-swe-agent-lm-traj (α=0.232, H∞=0.007, 30.4 turns/71KB — 32B self-rollout lands in the template-degenerate cluster, supporting the generator-scale contrast vs frontier OpenHands dumps; new `ser_r2e_steps`). Rejected: annon124816/tau_bench (parquet checksum manifest, not data), MoatlessTools-Sampled (eval metadata only), ZeonLap/OpenHands-Trajectories (webdataset tar, empty logs). weblinx-browsergym shelved after 2 hangs. Registry: 34 valid | `scripts/score_agentic_datasets.py --only` ×1 | `data/provenance/*.json` (37), CSV 37 rows |
| 19 | 2026-06-05 | Loop iter 11: hit unauthenticated HF 429 (browsergym tree listing drained the 500req/300s window; one back-off cycle), then registered r2e-gym-lite-tasks (α=0.381, H∞=0.890, 382 eps × 22KB — `ser_r2e_task` renders problem_statement + commit hunks, whole-file contents excluded). Synthetic-issue H∞=0.89 < human SWE-bench-Verified 1.57, ≈ SWE-rebench 0.84. Registry: 35 valid | `scripts/score_agentic_datasets.py --only` ×1 | `data/provenance/*.json` (38), CSV 38 rows |
| 20 | 2026-06-05 | Loop iters 12–13 (final, user-requested stop): HF token configured (auth as shirleyhuang1; written to ~/.cache/huggingface/token from ~/.env). Still blocked: GAIA / OS-Genesis / xlam-60k (gated, need per-dataset access request), weblinx-browsergym (ReadTimeout even authenticated). +2 registered: glaive-fc-v2 (α=0.280/H∞=1.050) and hermes-fc-multiturn (α=0.259/H∞=0.777) — finding 8: short FC dialogues keep healthy H∞ while trajectory-style tool sets sit at H∞=0, so repeated per-turn observations (not tool-calling itself) are what kill H∞. SAMPLES.md §V refreshed to closing state. Final registry: 37 valid entries; CSV 40 rows (3 documented exclusions: taubench-sonnet-proxy, rlenv-appworld-train, nemotron-agentic-v1); 40 provenance sidecars. Deferred to future work: 5-seed σ, multimodal (Aguvis/OSWorld), gated sets pending user access, Nemotron-SFT-v2 search/tool_calling upstream fixes | manual + `scripts/score_agentic_datasets.py --only` ×2 | `SAMPLES.md`, `data/agentic_alpha_hinf.csv`, `data/provenance/*.json` |
| 21 | 2026-06-05 | Loop iters 14–18 (5-round extension, final): (a) gated re-check — GAIA/OS-Genesis/xlam still blocked, no access requests made; (b) Aguvis text-side slice scored then excluded (fit artifact α=−0.031/H∞=34.6: single-step rows + repeated giant system prompt); (c) category sweep found II-Agent GAIA validation traces (α=0.197/H∞=1.254, 46.6 turns — ungated GAIA trajectory proxy), deep-research-sft-0406 (α=0.179/H∞=0) and Fractal DeepResearch-SFT (α=0.180/H∞=0, 118KB single-doc reports); (d) Nemotron-SFT-v2 search+tool_calling rescued via raw-JSONL streaming (loader's schema inference was the failure, bad_lines=0): search α=0.261/H∞=1.366 (24.8 turns/76KB), tool_calling α=0.157/H∞=0; CSV rows appended manually in matching format, provenance sidecars written with serializer noted as jsonl_direct. Findings 9 (domain doesn't determine signature — generation pipeline does) and 10 (loader failure ≠ data corruption) added to §V. FINAL STATE: 42 valid registry entries; CSV/provenance/samples all 46; 4 documented exclusions (taubench-sonnet-proxy, rlenv-appworld-train, nemotron-agentic-v1, aguvis-s2-androidctl-text) | inline jsonl_direct scorer + `scripts/score_agentic_datasets.py --only` ×4 | `SAMPLES.md`, `data/agentic_alpha_hinf.csv` (46), `data/provenance/*.json` (46) |

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
