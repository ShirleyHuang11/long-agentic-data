# Form-vs-Choices Eval Design (finding 20 discriminator)

*Companion to `data/formchoice_manifest.json` and `scripts/formchoice/`. The
experiment SFT-s Qwen2.5-1.5B-Instruct separately on a TEMPLATE corpus
(H∞=0, OpenThoughts-Agent-v1-SFT) and a HEALTHY corpus (H∞ 0.80–1.63, equal
thirds DCAgent-GLM4.7 / JetBrains / SWE-ZERO-12M), at matched ~30M-token budget.*

## Hypothesis (finding 20, operationalized)

Template-band data teaches **FORM** (trajectory skeleton: valid action syntax,
loop discipline, well-formed tool calls) but not **CHOICES** (selecting the
correct command/action for a novel task). Healthy-band data teaches choices.

Predicted result pattern:

| | form-bound checks | decision-bound checks |
| :-- | :-- | :-- |
| TEMPLATE model | **high** (its whole signal is form) | low / no lift over base |
| HEALTHY model | high (form is cheap, present in healthy too) | **higher than TEMPLATE** |

The discriminating signal is the **interaction**: TEMPLATE ≈ HEALTHY on form,
but HEALTHY > TEMPLATE on decisions. If TEMPLATE matches HEALTHY on *both*,
finding 20 is wrong (template data also carries choices). If neither beats the
base model on decisions, the 30M budget / 1.5B scale is too small to register
choices and the experiment is inconclusive (report as such, do not over-claim).

## Protocol

- 3 models compared: `base` (Qwen2.5-1.5B-Instruct, no SFT), `template`,
  `healthy`.
- Greedy decoding, max 2048 new tokens per turn, identical system prompt =
  the terminus-2 instruction header (so format is in-distribution for both).
- Each check = (prompt source, pass criterion). Form checks are scored by a
  deterministic parser; decision checks by an oracle (gold command / verifier /
  exact-match on a held-out gold action). No LLM judge — keeps scoring honest.
- **No template overlap rule for decision checks**: prompts are drawn from
  registry datasets that appear in NEITHER training corpus (see candidate pool
  below), and from synthetic single-shot tasks we author here.

---

## A. Form-bound checks (12)

Scored on the model's *raw emission* for a single held-out agent turn. These
ask only "is the output a well-formed terminus-2 / tool step", not "is it the
right step". Prompts are held-out task headers (not in the 30M sampled bytes:
take OpenThoughts-Agent-v1-SFT rows *beyond* the 7,787 episodes consumed by the
build, and DCAgent rows beyond those consumed).

1. **Valid terminus-2 action block** — prompt ends after an observation; pass =
   emission contains exactly one fenced action block in the terminus-2 schema
   the corpus uses (parseable command region). Parser: regex for the corpus's
   action delimiter.
2. **Single action per turn** — pass = exactly one action block emitted (not
   zero, not multiple) — loop discipline.
3. **No premature termination** — given an observation that clearly is not a
   success state, pass = model does NOT emit the "task complete"/submit token.
4. **Correct termination** — given an observation showing the goal met, pass =
   model emits the submit/finish action.
5. **Valid JSON tool call** — for a tool-call-style held-out prompt, pass =
   emitted tool call parses as JSON with required keys present (`name`,
   `arguments`). Drawn from `Salesforce/APIGen-MT-5k` prompt (tool schema given
   in context) — NOT in either training mix.
6. **Argument-key well-formedness** — pass = tool-call `arguments` keys are a
   subset of the schema-declared params (no hallucinated keys). Schema-checked.
7. **Shell-command syntax validity** — pass = emitted command parses under
   `bash -n` (syntactic only, not executed).
8. **Observation echo discipline** — pass = model does NOT verbatim-copy >50%
   of the preceding observation into its action (template models sometimes echo).
9. **Turn-role formatting** — pass = emission stays in assistant role, no
   spurious `<|im_start|>user` / fake observation injected by the model.
10. **Multi-turn loop continuation** — feed a 3-turn prefix, pass = model emits
    a 4th well-formed action (loop held together across turns).
11. **Stop-token emission** — pass = generation ends with the proper EOS/turn
    delimiter rather than running to max tokens.
12. **No degenerate repetition** — pass = emitted action has <30% 4-gram
    repetition (catches template-collapse babble).

Form score = fraction of the 12 passing, micro-averaged over ~50 held-out
prompts per check.

---

## B. Decision-bound checks (12)

Each has a **gold action / command** and success = the model chooses it (exact
match on the load-bearing token, or verifier pass). Prompts have NO template
overlap with either training corpus.

Candidate pool (registry datasets in NEITHER training mix):
- `Salesforce/APIGen-MT-5k` (tau-bench-style airline/retail tool calls; gold =
  recorded correct API call).
- `jkazdan/taubench_traces_training_data` (tau-bench airline/retail; gold = next
  correct tool/action).
- `Yhyu13/ToolBench_toolllama_G123_dfs` (multi-step API selection; gold = next
  API in the DFS solution path).
- `DCAgent2/aider_polyglot_*-traces` (aider-polyglot coding exercises; gold =
  the edit/command that passes the held-out unit test — verifier-scored).
- Authored single-shot terminal tasks with a unique correct command (see 11–12).

1. **Tool selection (APIGen)** — multi-tool context, one correct tool; pass =
   model's tool name == gold tool name. ~80 prompts.
2. **Tool argument correctness (APIGen)** — given correct tool, pass = required
   argument *values* match gold (e.g. correct flight id, correct user id).
3. **tau-bench next-action** — mid-trajectory state from `jkazdan/taubench`;
   pass = next action == gold next action (tool name + key arg).
4. **tau-bench policy decision** — a state where policy forbids an action; pass
   = model does NOT take the forbidden action (refuses / asks). Gold = refusal.
5. **ToolBench API path** — given a partial DFS path, pass = next API == gold
   next API. ~60 prompts.
6. **aider-polyglot fix selection** — given a failing test + file, pass = the
   model's proposed edit makes the held-out unit test pass (sandbox verifier).
   This is the strongest decision check (real ground truth).
7. **Disambiguation** — task with two plausible commands, only one correct
   given the stated constraint; pass = correct one. Authored, 30 prompts.
8. **Error-recovery choice** — observation = an error trace; pass = model picks
   the action that addresses the *named* error (not a generic retry). Gold =
   labeled corrective action from held-out OpenHands trajectories not in mix.
9. **Stop-vs-continue decision** — ambiguous near-goal state; pass = model's
   stop/continue choice matches gold (does it know when it's done).
10. **Constraint adherence** — task says "do X without using command Y"; pass =
    chosen command avoids Y while achieving X. Authored, 30 prompts.
11. **Authored terminal task, unique gold command** — e.g. "count lines in
    every .py under src/ and write the total to count.txt"; pass = a command
    whose execution produces the gold file content (verifier). ~25 tasks.
12. **Authored multi-step plan first action** — task requiring a specific first
    step (e.g. must `git checkout -b` before editing); pass = correct first
    action. ~25 tasks.

Decision score = fraction passing, micro-averaged per check.

---

## Scoring & reporting

- Report a 3×2 table: {base, template, healthy} × {form score, decision score}.
- Primary statistic: `Δdecision = healthy − template` and
  `Δform = healthy − template`. Finding 20 supported iff
  `Δdecision > 0` meaningfully while `Δform ≈ 0`.
- Bootstrap 95% CIs over prompts (per-prompt resampling). With ~50–80 prompts
  per check the per-check CI is wide; rely on the pooled form/decision scores
  (≈600 prompts each) for the headline contrast.
- Honesty caveats to carry into any writeup: (i) 1.5B + 30M tokens is a small
  regime; a null on decisions may be scale-limited, not evidence against
  finding 20. (ii) The healthy mix is 3 SWE/terminal sources; decision checks
  drawn from tool-calling (tau/APIGen) test *transfer*, not in-domain recall —
  by design (no template overlap), but means decision lift, if present, is a
  strong result and absence is weaker evidence.

## Build status of the eval set

This document is the **design**. The held-out prompt extraction + verifier
harness are not yet built (out of scope for the prep/smoke task). Next step
after full training: implement `scripts/formchoice/eval.py` realizing checks
A1–A12 (deterministic parsers) and B1–B12 (gold/verifier matching), drawing
prompts from the candidate pool above with the no-overlap filter enforced by
hashing against the 30M-token build.
