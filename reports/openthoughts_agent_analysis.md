# OpenThoughts-Agent reproduction & analysis

*2026-06-07 — scores in `data/agentic_alpha_hinf.csv` (slugs `openthoughts-agent-v1-sft`, `nl2bash-teacher-*`), provenance sidecars per slug.*

## 1. What OpenThoughts-Agent is

Open consortium (Stanford/Berkeley/LAION/mlfoundations et al.) finding **data recipes for small agentic models** (terminal agents, OpenThinker-Agent-v1 = Qwen-8B-class). Two-stage recipe:

- **SFT**: ablated 15 instruction sources → winners **NL2Bash** (synthetic shell tasks) + **InferredBugs** (real C#/Java bugs); traces by a teacher running terminus-2 in sandboxes; **teacher ablation: GLM-4.6 ≈ 2× downstream over GPT-family teachers**; ~15k traces (v1-SFT).
- **RL**: GPT-5-Mini permutes NL2Bash tasks + writes tests; sandbox-verified; tasks where GPT-5-Codex gets zero reward are dropped (10k → ~700).
- Result: best-in-size on Terminal-Bench 2.0; RL adds +1–2%.

The `DCAgent`/`DCAgent2`/`DCAgent3`/`penfever`/`mlfoundations-dev` HF orgs are this project's experimental exhaust — our registry had already scored several of its artifacts (GLM-4.7 terminus traces, nemotron-terminal SFT-dose pair, aider-polyglot generator panel, finance-terminal RL traces) before identifying the source.

## 2. Reproduction: our oracle on their recipe

| dataset (identical NL2Bash task source unless noted) | α | H∞ |
| :-- | :-- | :-- |
| OpenThoughts-Agent-v1-SFT (released winning blend) | 0.137 | **0.000** |
| NL2Bash × GLM-4.6 teacher (their winner) | 0.135 | 0.000 |
| NL2Bash × Qwen3-Coder-480B teacher | 0.156 | 0.000 |
| NL2Bash × MiniMax-M2.7 teacher | 0.152 | 0.000 |
| NL2Bash × GPT-5-Nano teacher | repo emptied upstream |

Two findings, both honest:

**(a) The winning recipe is template-band data.** 10k near-permutations of small synthetic shell tasks → scenario-instance repetition → H∞=0, exactly as finding 17 predicts. Yet it trains the best small terminal model. **H∞=0 data is not worthless — it is worthless *for what frontier models lack*.**

**(b) The teacher ablation (2× downstream!) is invisible to compression.** All teachers produce statistically indistinguishable traces on these tasks (α 0.135–0.156, H∞ all 0). The probe's resolution boundary, already suspected from finding 11, is now training-validated: **compression statistics select data *regimes*; they cannot rank recipes *within* a regime.** Within-regime choices (teacher identity, format details) still require proxy-training ablations — which is precisely the human-intensive part of their pipeline.

## 3. The resolution: data value is student-relative (finding 20)

Combining with the registry's echo/novelty decomposition:

- A trajectory = **form** (low-β recurrence: schemas, loops, observation echo) + **choices** (H∞ novelty).
- A *small base model* lacks the **form** — it cannot yet hold a terminus loop together. Template-band data is the cheapest possible curriculum for form, and form is most of what Terminal-Bench-2-at-8B measures. Hence their result.
- A *frontier model* has long since saturated the form (every mid-size model in our registry produces flawless trajectory skeletons around zero progress). Its deficit is the choice sliver — which is exactly what H∞ measures and what their recipe's data does not contain.
- Their own two-stage design is this decomposition operationalized: **SFT teaches the echo, RL buys decisions** (+1–2%, small because ~700 verified tasks is a tiny novelty budget).

**Finding 20: the value of agentic data is the match between its recurrence/novelty profile and the student's deficit.** Template-band data is a form-curriculum (valuable at small scale, inert at frontier); healthy-band data is a decision-curriculum (the only thing that moves frontier agents). One probe, two regimes of use.

## 4. Where the humans actually are in their recipe

Stripping the automation away, human supervision concentrates at exactly three points:

1. **Source taste** — choosing which 15 instruction sources to try, and accepting the 2 winners. (Judgment about *where tasks come from*.)
2. **The eval set** — 70 dev tasks hand-curated by volunteers under expert supervision; every automated loop optimizes against this human artifact.
3. **Recipe selection via ablation** — humans design/launch the grid (teachers, sources, filters) and read the results.

Everything else is already automated: generation (GPT-5-Mini permutation), verification (sandbox tests), filtering (zero-reward removal), judging (4o-mini). The expensive human inputs are *taste, ground truth, and search* — not labeling.

## 5. Scaling without humans: an algorithmic substitute for each leverage point

| human input | algorithmic substitute | grounding in our results |
| :-- | :-- | :-- |
| **Source taste** | The (H∞, β, horizon) probe as automatic source ranker — seconds per source vs a training run; prunes regime-level search (would have flagged all-NL2Bash-teachers as one equivalence class instantly, saving 3 of their 4 teacher runs). Within-regime ties broken by small proxy trainings (their α_D is high — proxies are cheap by construction). | registry; §2b boundary |
| **Eval ground truth** | Replace hand-curated dev tasks with **verifier-minted evals**: formally verified synthetic code/terminal tasks (tests = ground truth) at controlled LRD and dialable difficulty; difficulty calibrated by solve-rate of reference models (their zero-reward filter, inverted). | finding 8/17 (grounded synthetic stays healthy); Toucan pattern; verified-code proposal |
| **Recipe search** | Close the loop: generate → verify → **score (H∞/template-gain/loop-gain) → filter → proxy-train → promote**, with the diversity controller (reject batches drifting toward scenario repetition: per finding 17, monitor instance-level repetition, not domain count) and per-domain stream stripping (finding 16) as fixed stages. The compression panel acts as the cheap outer-loop fitness; proxy training as the expensive inner confirmation, run only on regime survivors. | LZ-Select pilot (kept 0.41 vs rejected 0.24); episode filter; findings 15–17 |
| **The novelty budget itself** | The one thing no filter can create. Automatic sources, in our measured ordering: real environments (execution outputs, real repos/web), formal verifiers (novelty = the verifier's verdict structure propagated into traces), and frontier/human session exhaust (CLI sessions measured H∞ 1.49–1.73). Scale plan: synthetic *tasks* with real *grounding* — permute like they do, but ground every permutation in a verifier and an environment with real content, and meter H∞ continuously so permutation never silently converts novelty to recurrence. | findings 8/9/17; fig 9 code-bridge band |

The compact statement: **their pipeline already automates labor; what remains human is taste, ground truth, and search — and each has a measurable, theory-backed substitute: compression statistics for taste, verifiers for ground truth, and probe-pruned proxy-training loops for search.** The student-relative principle (finding 20) sets the curriculum: template-band form data early (free to synthesize), healthy-band decision data late (budget-constrained, verifier-minted), with the H∞ meter deciding when a student has graduated from one to the other.

## 6. Caveats

- Teacher-panel H∞ ties are on *one* task source (NL2Bash); InferredBugs-per-teacher dumps exist and could differ.
- GPT-5-Nano traces were removed upstream — the losing teacher is unmeasured.
- "Form vs choices" student-relativity (finding 20) is an interpretation consistent with their results and ours; the direct test is SFT-ing a small model on healthy-band vs template-band data at matched budget and watching form-vs-decision benchmarks separately.
- Their downstream numbers come from their blog (Terminal-Bench 2.0 claims), not independently reproduced here; we reproduced the *data side* only.
