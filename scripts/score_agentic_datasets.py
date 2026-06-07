"""Score long-horizon agentic datasets with the LZ (alpha, H_inf) oracle.

Pilot registry of common long-horizon agentic benchmarks / trajectory corpora
on HuggingFace. Each entry defines how to serialize one episode (trajectory)
into a single text document: turns rendered as `[role]\ntext` blocks joined by
blank lines, matching the doc-per-episode protocol of the formal-math survey
(~1500 docs or 8 MB cap per dataset, see lz_oracle.py).

Usage:
    python scripts/score_agentic_datasets.py            # score all pilot entries
    python scripts/score_agentic_datasets.py --only nebius/SWE-agent-trajectories
Outputs:
    data/agentic_alpha_hinf.csv   (one row per dataset)
    samples_cache/<slug>.txt      (first 3 serialized episodes, for inspection)
"""

import argparse
import csv
import datetime
import itertools
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(__file__))
import lz_oracle
from datasets import load_dataset
from huggingface_hub import HfApi


def _turns_chatml(msgs, role_key, text_key):
    return "\n\n".join(f"[{m.get(role_key, '?')}]\n{m.get(text_key) or ''}" for m in msgs)


def ser_nebius(row):
    return _turns_chatml(row["trajectory"], "role", "text"), len(row["trajectory"])


def ser_swesmith(row):
    msgs = row["messages"]
    if isinstance(msgs, str):  # stored as a JSON-encoded string
        msgs = json.loads(msgs)
    return _turns_chatml(msgs, "role", "content"), len(msgs)


def ser_conversations(row):
    return _turns_chatml(row["conversations"], "from", "value"), len(row["conversations"])


def _msgs_auto(msgs):
    # {role,content} / {from,value} / {role,text} item styles.
    if not msgs:
        return "", 0
    keys = msgs[0].keys()
    rk = "role" if "role" in keys else "from"
    tk = "content" if "content" in keys else ("value" if "value" in keys else "text")
    return _turns_chatml(msgs, rk, tk), len(msgs)


def ser_trajectory_auto(row):
    return _msgs_auto(row["trajectory"])


def ser_nemotron_inj(row):
    # Injection task corpus: the attack payload lives in `environment` (e.g.
    # chart_notes carrying the injected instruction), so serialize input
    # messages + environment JSON — prompt-only view is near-pure template.
    doc, n = _msgs_auto(row["responses_create_params"]["input"])
    env = json.dumps(row.get("environment") or {}, ensure_ascii=False)
    return f"{doc}\n\n[environment]\n{env}", n + 1


def ser_agentnet(row):
    # AgentNet (OSWorld-family computer-use): traj steps carry full textual
    # observation/thought/action/reflection + pyautogui code; images are
    # filename strings (excluded). First true desktop-GUI text-side entry.
    parts = [f"[instruction]\n{row.get('instruction') or ''}"]
    for s in row["traj"]:
        v = s.get("value") or {}
        body = "\n".join(f"[{k}]\n{v[k]}" for k in
                         ("observation", "thought", "action", "code", "reflection")
                         if v.get(k))
        parts.append(body)
    return "\n\n".join(parts), len(row["traj"])


def ser_agentnet_actions(row):
    # AgentNet compact action view: instruction + pyautogui code per step only
    # (VLM-generated observation/thought/reflection stripped) — within-dataset
    # ablation against ser_agentnet, mirroring the mind2web/weblinx pair.
    parts = [f"[instruction]\n{row.get('instruction') or ''}"]
    parts += [(s.get("value") or {}).get("code") or "" for s in row["traj"]]
    return "\n\n".join(parts), len(row["traj"])


def ser_rebel_actions(row):
    # Counter-test for finding 15: action-only view of a machine-generated
    # trajectory set. ALFWorld expert actions come from a rule-based planner,
    # so unlike AgentNet's human action stream we expect H_inf to stay low.
    steps = json.loads(row["steps"])
    parts = [f"[task]\n{row.get('task') or ''}"]
    parts += [s.get("ground_truth_action") or "" for s in steps]
    return "\n\n".join(parts), len(steps)


def _assistant_only(msgs):
    if not msgs:
        return "", 0
    keys = msgs[0].keys()
    rk = "role" if "role" in keys else "from"
    tk = "content" if "content" in keys else ("value" if "value" in keys else "text")
    kept = [m for m in msgs if m.get(rk) in ("assistant", "gpt", "agent")]
    return _turns_chatml(kept, rk, tk), len(kept)


def ser_messages_assistant(row):
    # Agent-text-only view: keep assistant turns, drop env observations —
    # completes the finding-15 decomposition for frontier SWE rollouts.
    msgs = row["messages"]
    if isinstance(msgs, str):
        msgs = json.loads(msgs)
    return _assistant_only(msgs)


def ser_trajectory_assistant(row):
    return _assistant_only(row["trajectory"])


def ser_gdpval(row):
    # GDPval: human-written professional task prompts (reference/deliverable
    # files are binary and excluded) — section IV task corpus.
    return (f"[sector]\n{row.get('sector') or ''}\n\n[occupation]\n"
            f"{row.get('occupation') or ''}\n\n[prompt]\n{row.get('prompt') or ''}"), 1


def ser_cua_gimp(row):
    # cua-dev AgentNet-GIMP (multimodal): text channel = instruction + turn
    # messages + raw pyautogui actions; image channel (~10 PNG/ep, 1920x1080)
    # is dropped via drop_cols and recorded descriptively in the registry row.
    doc, n = _msgs_auto(row.get("messages") or [])
    acts = "\n".join(row.get("raw_actions") or [])
    return (f"[instruction]\n{row.get('instruction') or ''}\n\n{doc}\n\n"
            f"[raw_actions]\n{acts}"), int(row.get("num_steps") or n)


def ser_gui_odyssey(row):
    # GUI-Odyssey (multimodal mobile GUI): steps is a Python-repr list of step
    # dicts; screenshots are filename strings (excluded). Text channel =
    # instruction + per-step action/info.
    import ast
    steps = ast.literal_eval(row["steps"])
    parts = [f"[instruction]\n{row.get('instruction') or ''}"]
    for s in steps:
        parts.append(f"[{s.get('action', '?')}]\n{s.get('info', '')}")
    return "\n\n".join(parts), len(steps)


def ser_android_control(row):
    # AndroidControl (multimodal): human demos; text channel = goal + per-step
    # human instruction + structured action; screenshots_b64 dropped upstream.
    parts = [f"[goal]\n{row.get('goal') or ''}"]
    instrs = row.get("step_instructions") or []
    acts = row.get("actions") or []
    for i, a in enumerate(acts):
        ins = instrs[i] if i < len(instrs) else ""
        astr = " ".join(f"{k}={v}" for k, v in (a or {}).items() if v is not None)
        parts.append(f"[instruction]\n{ins}\n[action]\n{astr}")
    return "\n\n".join(parts), len(acts)


def ser_opencua(row):
    # OpenCUA (multimodal): messages is a JSON string with interleaved
    # image-ref/text content items; keep the text items only.
    msgs = json.loads(row["messages"])
    parts = []
    for m in msgs:
        content = m.get("content")
        if isinstance(content, list):
            text = "\n".join(c.get("text") or "" for c in content
                             if c.get("type") == "text")
        else:
            text = content or ""
        parts.append(f"[{m.get('role', '?')}]\n{text}")
    return "\n\n".join(parts), len(msgs)


def ser_rebel_steps(row):
    # ReBel ALFWorld: `steps` is a JSON-encoded list of step dicts (idx + obs/
    # action/... fields); render each non-idx field as its own labelled line.
    steps = json.loads(row["steps"])
    parts = []
    for s in steps:
        body = "\n".join(f"[{k}]\n{v}" for k, v in s.items() if k != "idx" and v)
        parts.append(body)
    return "\n\n".join(parts), len(steps)


def ser_messages_auto(row):
    return _msgs_auto(row["messages"])


def ser_conversations_auto(row):
    # `conversations` list with either {from,value} or {role,content} items.
    msgs = row["conversations"]
    if msgs and "role" in msgs[0]:
        return _turns_chatml(msgs, "role", "content"), len(msgs)
    return _turns_chatml(msgs, "from", "value"), len(msgs)


def ser_swebench(row):
    doc = (f"[problem_statement]\n{row['problem_statement']}\n\n"
           f"[hints]\n{row['hints_text']}\n\n"
           f"[patch]\n{row['patch']}\n\n[test_patch]\n{row['test_patch']}")
    return doc, 1


def ser_weblinx_episode(rows):
    # Episode = consecutive rows sharing `demo`; compact action view (turn +
    # action only). Raw `clean_html`/`candidates` are ~100KB-scale per turn and
    # excluded for the same corpus-fill reason as Mind2Web observations.
    parts = [f"[turn {r['turn']}]\n{r['action']}" for r in rows]
    return "\n\n".join(parts), len(rows)


def ser_taubench_proxy(rows):
    # sammshen/taubench-sonnet-traces: raw HTTP proxy logs grouped by
    # thread_id. Reconstruct the conversation from the request row whose JSON
    # body carries the most messages (the final API call holds the full
    # dialogue). User-sim and agent requests share a thread; we keep whichever
    # side ends up longest.
    best = []
    for r in rows:
        if r.get("type") != "request" or not r.get("body"):
            continue
        try:
            msgs = json.loads(r["body"]).get("messages") or []
        except (ValueError, AttributeError):
            continue
        if len(msgs) > len(best):
            best = msgs
    return _msgs_auto(best)


def ser_oh_events(row):
    # OpenHands feedback dump: trajectory = event stream of
    # {action, content, extras} items (not chat turns).
    evs = row["trajectory"]
    parts = [f"[{e.get('action', '?')}]\n{e.get('content') or ''}" for e in evs]
    return "\n\n".join(parts), len(evs)


HTML_CAP = 8192  # per-step observation cap for full-obs slices


def ser_mind2web_fullobs(row):
    # Full-observation slice: per-step cleaned_html capped at HTML_CAP chars
    # (raw is 37-240KB/step; uncapped, 2-3 episodes would fill the 8MB corpus
    # and cross-episode statistics would degenerate). Episode stays one doc.
    steps = []
    for a in row["actions"]:
        op = a.get("operation") or {}
        steps.append(f"[step op={op.get('op', '?')} value={op.get('value', '')}]\n"
                     f"{(a.get('cleaned_html') or '')[:HTML_CAP]}")
    return f"[task]\n{row['confirmed_task']}\n\n" + "\n\n".join(steps), len(row["actions"])


def ser_weblinx_fullobs(rows):
    # WebLINX chat-config clean_html is already a pruned DOM (~1.6-4.3KB/turn);
    # include it whole, plus the action. Episode = consecutive rows per demo.
    parts = [f"[turn {r['turn']}]\n{(r.get('clean_html') or '')[:HTML_CAP]}\n"
             f"[action] {r['action']}" for r in rows]
    return "\n\n".join(parts), len(rows)


def ser_conversation_auto(row):
    # Agent-FLAN: field is `conversation` (singular), role/content items.
    return _msgs_auto(row["conversation"])


def ser_r2e_steps(row):
    # AxT-dev r2e-gym trajectories: trajectory_steps = list of step dicts
    # (step_idx, thought, action, observation, ...). Render every string field
    # per step to stay schema-robust.
    steps = row["trajectory_steps"]
    parts = []
    for s in steps:
        fields = [f"[{k}]\n{v}" for k, v in s.items()
                  if isinstance(v, str) and v]
        parts.append("\n".join(fields))
    doc = f"[problem_statement]\n{row['problem_statement']}\n\n" + "\n\n".join(parts)
    return doc, len(steps)


def ser_aguvis_texts(row):
    # Aguvis stage-2: `texts` = list of {system,user,assistant} GUI steps
    # (think/code format). Image observations are skipped — text-side slice.
    parts = []
    for t in row["texts"]:
        for k in ("system", "user", "assistant"):
            if t.get(k):
                parts.append(f"[{k}]\n{t[k]}")
    return "\n\n".join(parts), len(row["texts"])


def ser_ii_gaia(row):
    # II-Agent GAIA validation runs: `trace` = JSON list of tool_call /
    # message events. Tool inputs rendered as JSON, capped per event.
    try:
        evs = json.loads(row["trace"])
    except (ValueError, TypeError):
        evs = []
    parts = []
    for e in evs:
        body = e.get("tool_input", e.get("content", ""))
        if not isinstance(body, str):
            body = json.dumps(body, ensure_ascii=False)
        parts.append(f"[{e.get('type', '?')} {e.get('tool_name', '')}]\n{body[:HTML_CAP]}")
    doc = (f"[question]\n{row['Question']}\n\n" + "\n\n".join(parts)
           + f"\n\n[prediction]\n{row['prediction']}")
    return doc, max(len(evs), 1)


def ser_fractal_dr(row):
    return (f"[system]\n{row['system_prompt']}\n\n[prompt]\n{row['prompt']}\n\n"
            f"[report]\n{row['deepresearch_report']}"), 1


def ser_glaive_fc(row):
    # glaive-FC-v2: raw text turns ("USER: ... ASSISTANT: ..." in `chat`).
    doc = f"{row['system']}\n\n{row['chat']}"
    n_turns = row["chat"].count("USER:") + row["chat"].count("ASSISTANT:")
    return doc, max(n_turns, 1)


def ser_r2e_task(row):
    # R2E-Gym task sets: synthetic issue + parsed commit. Render hunks only
    # (old/new_file_content are whole files and would dwarf the task text);
    # cap per-file hunk text like the full-obs slices.
    pc = json.loads(row["parsed_commit_content"])
    diffs = []
    for fd in pc.get("file_diffs", []):
        fname = (fd.get("plus_file") or {}).get("path", "?")
        diffs.append(f"[file {fname}]\n{str(fd.get('hunks'))[:HTML_CAP]}")
    doc = (f"[problem_statement]\n{row['problem_statement']}\n\n"
           f"[commit_message]\n{pc.get('commit_message', '')}\n\n"
           + "\n\n".join(diffs)
           + f"\n\n[expected_tests]\n{row['expected_output_json']}")
    return doc, 1


def ser_nemotron_rl(row):
    # Nemotron RL pivot sets: per-*step* rows; context = responses_create_params
    # .input (messages), target = expected_action. Prefix-overlap caveat applies
    # (same class as nnetnav per-step SFT rows).
    doc, n = _msgs_auto(row["responses_create_params"]["input"])
    return f"{doc}\n\n[expected_action]\n{row.get('expected_action')}", n + 1


def ser_mind2web(row):
    # Compact action-trajectory view: task + per-step action representations.
    # (Raw `actions[*].cleaned_html` observations are ~MB-scale per step and
    # would let 2-3 episodes fill the whole 8 MB corpus; excluded in pilot.)
    steps = "\n".join(f"[step {i}] {a}" for i, a in enumerate(row["action_reprs"]))
    return f"[task]\n{row['confirmed_task']}\n\n{steps}", len(row["action_reprs"])


# (hf_path, config, split(s), serializer, slug)
REGISTRY = [
    ("nebius/SWE-agent-trajectories", None, ["train"], ser_nebius, "nebius-swe-agent-traj"),
    ("SWE-bench/SWE-smith-trajectories", None, ["xml"], ser_swesmith, "swe-smith-traj-xml"),
    ("THUDM/AgentInstruct", "default",
     ["os", "db", "alfworld", "webshop", "mind2web", "kg"],
     ser_conversations, "agentinstruct-all"),
    ("AgentGym/AgentTraj-L", None, ["train"], ser_conversations, "agentgym-agenttraj-l"),
    ("princeton-nlp/SWE-bench_Verified", None, ["test"], ser_swebench, "swe-bench-verified"),
    ("osunlp/Mind2Web", None, ["train"], ser_mind2web, "mind2web-actions"),
    # --- loop iter 2: SWE tasks + code-act trajectories ---
    ("SWE-Gym/SWE-Gym", None, ["train"], ser_swebench, "swe-gym-tasks"),
    ("nebius/SWE-rebench", None, ["test"], ser_swebench, "swe-rebench-test"),
    ("xingyaoww/code-act", None, ["codeact"], ser_conversations_auto, "code-act-codeact"),
    # --- loop iter 2b: OpenHands trajectory dumps + web-agent trajectories ---
    ("nebius/SWE-rebench-openhands-trajectories", None, ["train"],
     ser_trajectory_auto, "swe-rebench-oh-traj"),
    ("SWE-Gym/OpenHands-SFT-Trajectories", None, ["train.success.oss"],
     ser_messages_auto, "swe-gym-oh-sft-traj"),
    ("nvidia/SWE-Zero-openhands-trajectories", None, ["train"],
     ser_trajectory_auto, "swe-zero-oh-traj"),
    ("xlangai/AgentTrek", None, ["train"], ser_messages_auto, "agenttrek"),
    # NB: nnetnav-wa rows are per-*step* SFT examples (messages = context up to
    # that step), not whole episodes — overlapping prefixes inflate redundancy;
    # interpret (alpha, H_inf) accordingly.
    ("stanfordnlp/nnetnav-wa", None, ["train"], ser_messages_auto, "nnetnav-wa"),
    # --- loop iter 3 ---
    ("nvidia/SWE-Hero-openhands-trajectories", None, ["train"],
     ser_trajectory_auto, "swe-hero-oh-traj"),
    ("Solaris99/AgentBank",
     ["alfred", "alfworld", "apps", "gsm8k", "hotpotqa", "humaneval",
      "intercode_bash", "intercode_sql", "iqa", "math", "mathqa", "mbpp",
      "mbpp_before", "mind2web", "rearrange", "strategyqa", "triviaqa",
      "webarena", "webshop"],
     ["train"], ser_conversations, "agentbank-all"),
    ("ai2lumos/lumos_unified_ground_iterative", None, ["train"],
     ser_messages_auto, "lumos-ground-iter"),
    # nnetnav-live: per-step SFT rows, same prefix-overlap caveat as nnetnav-wa.
    ("stanfordnlp/nnetnav-live", None, ["train"], ser_messages_auto, "nnetnav-live"),
    ("SWE-Gym/OpenHands-Sampled-Trajectories", None, ["train.raw"],
     ser_messages_auto, "swe-gym-oh-sampled-traj"),
    # Verifier set: judge conversations over trajectories (verifier training
    # data), not agent rollouts themselves.
    ("SWE-Gym/OpenHands-Verifier-Trajectories", None, ["train.mixture"],
     ser_messages_auto, "swe-gym-oh-verifier-traj"),
    # WebLINX chat config: per-turn rows -> episodes grouped by `demo`;
    # compact action view (see ser_weblinx_episode).
    ("McGill-NLP/weblinx", "chat", ["train"], ser_weblinx_episode,
     "weblinx-actions", "demo"),
    # --- loop iter 4: tool-use trajectories ---
    # Unofficial tau-bench SFT traces; tool_calls payloads are dropped when
    # message content is None (approximate text view).
    ("jkazdan/taubench_traces_training_data", None, ["train"],
     ser_messages_auto, "taubench-traces-jkazdan"),
    # Unofficial tau-bench proxy logs -> conversations via thread_id grouping.
    ("sammshen/taubench-sonnet-traces", None, ["train"],
     ser_taubench_proxy, "taubench-sonnet-proxy", "thread_id"),
    # AppWorld RL env: rows look like task prompts (messages), maybe not
    # rollouts — mean_turns will tell.
    ("hamishivi/rlenv-appworld-train", None, ["train"],
     ser_messages_auto, "rlenv-appworld-train"),
    # --- loop iter 5 ---
    # ToolBench's ToolLLaMA DFS decision-tree trajectories (unofficial mirror).
    ("Yhyu13/ToolBench_toolllama_G123_dfs", None, ["train"],
     ser_conversations, "toolbench-toolllama-dfs"),
    # Official OpenHands user-session feedback dump (event-stream format).
    ("OpenHands/openhands-feedback", None, ["train"],
     ser_oh_events, "openhands-feedback"),
    # --- loop iter 6: full-observation long-context slices ---
    ("osunlp/Mind2Web", None, ["train"], ser_mind2web_fullobs, "mind2web-fullobs"),
    ("McGill-NLP/weblinx", "chat", ["train"], ser_weblinx_fullobs,
     "weblinx-fullobs", "demo"),
    # --- loop iter 7: Nemotron agentic family + APIGen-MT ---
    ("nvidia/Nemotron-SFT-Agentic-v2", None, ["interactive_agent"],
     ser_messages_auto, "nemotron-sft-v2-interactive"),
    ("nvidia/Nemotron-SFT-Agentic-v2", None, ["search"],
     ser_messages_auto, "nemotron-sft-v2-search"),
    ("nvidia/Nemotron-SFT-Agentic-v2", None, ["tool_calling"],
     ser_messages_auto, "nemotron-sft-v2-tool"),
    ("nvidia/Nemotron-Agentic-v1", None, ["interactive_agent", "tool_calling"],
     ser_messages_auto, "nemotron-agentic-v1"),
    ("nvidia/Nemotron-RL-Agentic-SWE-Pivot-v1", None, ["train"],
     ser_nemotron_rl, "nemotron-rl-swe-pivot"),
    ("nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1", None,
     ["train"], ser_nemotron_rl, "nemotron-rl-conv-tool-pivot"),
    ("Salesforce/APIGen-MT-5k", "dataset", ["train"],
     ser_conversations, "apigen-mt-5k"),
    # --- loop iter 8 ---
    ("internlm/Agent-FLAN", None,
     ["agent_instruct_react", "agent_instruct_tflan",
      "toolbench_instruct_j1s1_3k", "toolbench_negative", "toolbench_react_10p",
      "toolbench_tflan_60p_r10r5u7", "toolbench_tflan_cot_30p"],
     ser_conversation_auto, "agent-flan-all"),
    # NB: despite the repo name, rows are ScienceWorld episodes
    # ("You are an agent for science world"), not WebShop.
    ("lclan/webshop_expert_trajectories", None, ["test"],
     ser_conversations_auto, "scienceworld-expert-traj"),
    ("zwhe99/FireAct", None, ["multitask_multimethod"],
     ser_messages_auto, "fireact-multitask"),
    # --- loop iter 10 ---
    ("AxT-dev/swe-agent-lm-32b-r2e-gym-trajectories", None, ["train"],
     ser_r2e_steps, "r2e-gym-swe-agent-lm-traj"),
    # --- loop iter 11 ---
    ("R2E-Gym/R2E-Gym-Lite", None, ["train"], ser_r2e_task, "r2e-gym-lite-tasks"),
    # --- loop iter 12: single/multi-turn function calling (breadth) ---
    ("glaiveai/glaive-function-calling-v2", None, ["train"],
     ser_glaive_fc, "glaive-fc-v2"),
    ("NousResearch/hermes-function-calling-v1", "func_calling", ["train"],
     ser_conversations, "hermes-fc-multiturn"),
    # --- loop iter 14: Aguvis text-side slice (images skipped) ---
    ("smolagents/aguvis-stage-2", "android_control", ["train"],
     ser_aguvis_texts, "aguvis-s2-androidctl-text"),
    # --- loop iter 15: GAIA-proxy + deep-research trajectories ---
    ("Intelligent-Internet/ii-agent_gaia-benchmark_validation", None, ["train"],
     ser_ii_gaia, "ii-agent-gaia-traj"),
    ("kylemontgomery/deep-research-sft-0406", None, ["train"],
     ser_messages_auto, "deep-research-sft-0406"),
    ("FractalAIResearch/DeepResearch-SFT", None, ["train"],
     ser_fractal_dr, "fractal-deepresearch-sft"),
    # --- loop iter 19: new SWE rollout dumps + Toucan MCP tool-agentic ---
    ("AlienKevin/SWE-ZERO-12M-trajectories", None, ["train"],
     ser_messages_auto, "swe-zero-12m-traj"),
    ("Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k", None, ["train"],
     ser_messages_auto, "kwai-klear-mini-swe-66k"),
    # Toucan messages are a JSON-encoded string -> ser_swesmith handles both.
    ("Agent-Ark/Toucan-1.5M", "Kimi-K2", ["train"],
     ser_swesmith, "toucan-15m-kimi-k2"),
    # --- loop iter 20 ---
    # mini-swe-agent rollouts (GPT-5.2/5-mini mixture) on SWE-bench test-verified.
    ("JetBrains-Research/agent-trajectories-swe-bench-test-minus-verified", None,
     ["train"], ser_messages_auto, "jetbrains-swe-test-minus-verified"),
    # --- loop iter 22 ---
    # terminus-2 agent + GLM-4.7 traces on SWE-Gym sampled tasks.
    ("DCAgent/neulab-swe-gym-openhands-sampled-trajectories-sandboxes_glm_4.7_traces_jupiter",
     None, ["train"], ser_conversations_auto, "dcagent-glm47-terminus2"),
    # Korean tool-calling dialogues (glm-5.1 generated) — first non-English entry.
    ("taejoon89/Ko-Agent-Trajectories-1.0", "train", ["train"],
     ser_messages_auto, "ko-agent-traj-train"),
    # --- loop iter 23: mid-size generator contrast on identical polyglot tasks ---
    ("DCAgent2/aider_polyglot_Qwen3_Coder_30B_A3B_Instruct_20260430_164230-traces",
     None, ["train"], ser_conversations_auto, "aider-polyglot-qwen3coder30b"),
    ("DCAgent2/aider_polyglot_SWE_agent_LM_7B_20260429_173705-traces",
     None, ["train"], ser_conversations_auto, "aider-polyglot-sweagentlm7b"),
    ("DCAgent2/aider_polyglot_R2EGym_32B_Agent_20260505_060450-traces",
     None, ["train"], ser_conversations_auto, "aider-polyglot-r2egym32b"),
    # --- loop iter 24: MiroVerse-style aggregated search/multihop agentic SFT ---
    ("WaltonFuture/agentic-sft-new", None, ["train"],
     ser_messages_auto, "miroverse-agentic-sft-new"),
    # --- loop iter 25: SFT-corpus-scale ablation (same model/tasks, 1k vs 100k) ---
    ("DCAgent3/aider_polyglot_nemotron_terminal_corpus_unified_1000__Qwen3_32B_20260520_085722",
     None, ["train"], ser_conversations_auto, "aider-polyglot-qwen32b-ntc1k"),
    ("DCAgent3/aider_polyglot_nemotron_terminal_corpus_unified_100000__Qwen3_32B_20260520_100037",
     None, ["train"], ser_conversations_auto, "aider-polyglot-qwen32b-ntc100k"),
    ("AI45Research/APP1-Agentic-Safety-SFT-Data", None, ["train"],
     ser_messages_auto, "app1-agentic-safety-sft"),
    # --- loop iter 26 ---
    ("Decix/ReBel-ALFWorld-SFT-Trajectories", None, ["train"],
     ser_rebel_steps, "rebel-alfworld-sft"),
    # Korean factory-operations agent rollouts; JSONL direct-read (CastError on
    # the hub loader), task_rollouts = the long-horizon split. n=70 caveat.
    ("hf-json:SeongryongJung/factory-agent-rollouts", None, ["task_rollouts"],
     ser_messages_auto, "factory-agent-task-rollouts"),
    # --- loop iter 27: smolagents GAIA traces + Toucan teacher contrast ---
    ("smolagents/gaia-traces", None, ["train"],
     ser_messages_auto, "smolagents-gaia-traces"),
    ("Agent-Ark/Toucan-1.5M", "OSS", ["train"], ser_swesmith, "toucan-15m-oss"),
    ("Agent-Ark/Toucan-1.5M", "Qwen3", ["train"], ser_swesmith, "toucan-15m-qwen3"),
    # --- loop iter 29: complete the Toucan teacher family ---
    ("Agent-Ark/Toucan-1.5M", "SFT", ["train"], ser_swesmith, "toucan-15m-sft"),
    # --- loop iter 31: adversarial indirect-prompt-injection task corpus ---
    # Rows are RL task definitions (input messages + environment + attack
    # metadata), not rollouts -> section IV. Environment carries the injected
    # payload, so it is part of the document (see ser_nemotron_inj).
    ("nvidia/Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1", None, ["train"],
     ser_nemotron_inj, "nemotron-rl-injection-v1"),
    # --- loop iter 32: first desktop computer-use entry (text side) ---
    ("xlangai/AgentNet", None, ["train"], ser_agentnet, "agentnet-text"),
    # --- loop iter 40: multimodal in scope (text channel scored; image
    # channel recorded descriptively) ---
    ("OpenGVLab/GUI-Odyssey", None, ["all"], ser_gui_odyssey,
     "gui-odyssey-actions"),
    ("mlfoundations-cua-dev/agentnet-gimp-trajectories", None, ["train"],
     ser_cua_gimp, "cua-agentnet-gimp-text", None, ["images"]),
    # --- loop iter 41 ---
    ("smolagents/android-control", None, ["train"],
     ser_android_control, "android-control-text", None, ["screenshots_b64"]),
    ("cua-lite/OpenCUA", None, ["train"],
     ser_opencua, "opencua-text", None, ["images"]),
    # --- loop iter 49 ---
    ("TIGER-Lab/BrowserAgent-Data", "sft", ["train"],
     ser_messages_auto, "tiger-browseragent-sft"),
    # MiniWoB BrowserGym SFT pair: reasoning+action vs action-only — a
    # ready-made annotation ablation (finding 15) on synthetic web tasks.
    ("saital/browser-agent-phase1-sft-reasoning-action", None, ["train"],
     ser_messages_auto, "saital-browser-reasoning-action"),
    ("saital/browser-agent-phase1-sft-action-only", None, ["train"],
     ser_messages_auto, "saital-browser-action-only"),
    # --- loop iter 50: mid-size generator point in the search domain ---
    # CastError on the hub loader (schema drift across shards) -> JSONL
    # direct-read, finding-10 pattern.
    ("hf-json:KermitCO/qwen3.5-9B-react-hotpotqa-traces", None,
     ["generated_traces_judged"], ser_messages_auto, "qwen35-9b-react-hotpot"),
    # --- loop iter 54: RL-FT mid-size checkpoint on finance-terminal tasks ---
    ("DCAgent3/financeagent_terminal_a3_rl_laion_exp_rpt_methods2test_large_v2_20260606_225633",
     None, ["train"], ser_conversations_auto, "dcagent3-financeagent-a3rl"),
    # --- loop iter 33: annotation-stripped action view (within-dataset ablation) ---
    ("xlangai/AgentNet", None, ["train"], ser_agentnet_actions, "agentnet-actions"),
    # --- loop iter 34: action-origin counter-test (planner-generated actions) ---
    ("Decix/ReBel-ALFWorld-SFT-Trajectories", None, ["train"],
     ser_rebel_actions, "rebel-alfworld-actions"),
    # --- loop iter 35: frontier agent-text-only views (finding-15 trichotomy) ---
    ("JetBrains-Research/agent-trajectories-swe-bench-test-minus-verified", None,
     ["train"], ser_messages_assistant, "jetbrains-swe-assistant-only"),
    ("nebius/SWE-rebench-openhands-trajectories", None, ["train"],
     ser_trajectory_assistant, "swe-rebench-oh-assistant-only"),
    # --- loop iter 36: real-vs-synthetic task contrast + GDPval task corpus ---
    ("JetBrains-Research/agent-trajectories-swesmith-random-subset", None,
     ["train"], ser_messages_auto, "jetbrains-swesmith-subset"),
    ("openai/gdpval", None, ["train"], ser_gdpval, "gdpval-tasks"),
]


def iter_docs(path, cfg, splits, ser, group_key=None, drop_cols=None):
    cfgs = cfg if isinstance(cfg, list) else [cfg]
    for c in cfgs:
        for split in splits:
            if path.startswith("hf-json:"):
                # Raw JSONL direct-read (bypasses datasets schema cast — loop
                # iters 17/26): split names the .jsonl file inside the repo.
                repo = path[len("hf-json:"):]
                ds = load_dataset(
                    "json", data_files=f"hf://datasets/{repo}/{split}.jsonl",
                    split="train", streaming=True)
            else:
                ds = load_dataset(path, c, split=split, streaming=True)
            if drop_cols:
                # Multimodal text-channel scoring: drop image columns BEFORE
                # iteration so streaming never decodes the screenshots.
                present = [col for col in drop_cols
                           if col in (ds.column_names or drop_cols)]
                ds = ds.remove_columns(present)
            if group_key is None:
                for row in ds:
                    yield ser(row)
            else:
                # Episode = run of consecutive rows sharing row[group_key].
                buf, key = [], None
                for row in ds:
                    k = row[group_key]
                    if buf and k != key:
                        yield ser(buf)
                        buf = []
                    key = k
                    buf.append(row)
                if buf:
                    yield ser(buf)


def score_entry(path, cfg, splits, ser, slug, seed_offset=0, group_key=None,
                drop_cols=None):
    docs, turn_counts = [], []
    size = 0
    it = iter_docs(path, cfg, splits, ser, group_key=group_key,
                   drop_cols=drop_cols)
    if seed_offset:
        it = itertools.islice(it, seed_offset, None)
    for doc, n_turns in it:
        docs.append(doc)
        turn_counts.append(n_turns)
        size += len(doc)
        if len(docs) >= lz_oracle.MAX_DOCS or size >= lz_oracle.MAX_BYTES:
            break
    res = lz_oracle.score(docs)
    cfg_str = "+".join(cfg) if isinstance(cfg, list) else (cfg or "")
    res.update(
        dataset=path, config=cfg_str, splits="+".join(splits), slug=slug,
        mean_turns=round(statistics.mean(turn_counts), 1) if turn_counts else 0,
        mean_doc_bytes=round(statistics.mean(len(d) for d in docs)) if docs else 0,
        n_episodes=len(docs),
    )
    os.makedirs("samples_cache", exist_ok=True)
    with open(f"samples_cache/{slug}.txt", "w", encoding="utf-8") as f:
        f.write("\n\n========== EPISODE BREAK ==========\n\n".join(docs[:3]))

    # Provenance sidecar: exact source revision + sampling protocol, so every
    # number in the registry is traceable to a pinned upstream snapshot.
    try:
        sha = HfApi().dataset_info(path.removeprefix("hf-json:")).sha
    except Exception:
        sha = "unresolved"
    prov = {
        "hf_dataset": path,
        "hf_url": f"https://huggingface.co/datasets/{path}",
        "hf_revision_sha": sha,
        "config": cfg_str or "default",
        "group_key": group_key,
        "splits": splits,
        "serializer": ser.__name__,
        "serializer_doc": (ser.__doc__ or "").strip(),
        "sampling": {
            "mode": "streaming, first-N episodes per split (no shuffle)",
            "max_docs": lz_oracle.MAX_DOCS,
            "max_bytes": lz_oracle.MAX_BYTES,
            "seed_offset": seed_offset,
        },
        "oracle": {
            "script": "scripts/lz_oracle.py",
            "compressor": f"zstd level {lz_oracle.ZSTD_LEVEL}",
            "context_points": list(lz_oracle.N_POINTS),
        },
        "collected_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "result": {k: res[k] for k in
                   ("alpha", "h_inf", "bpc_128", "bpc_2048", "bpc_32768",
                    "n_episodes", "mean_turns", "mean_doc_bytes", "n_bytes")},
    }
    os.makedirs("data/provenance", exist_ok=True)
    with open(f"data/provenance/{slug}.json", "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2, ensure_ascii=False)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    fields = ["dataset", "config", "splits", "slug", "alpha", "h_inf",
              "bpc_128", "bpc_2048", "bpc_32768",
              "n_episodes", "mean_turns", "mean_doc_bytes", "n_bytes"]
    rows = []
    for entry in REGISTRY:
        path, cfg, splits, ser, slug = entry[:5]
        group_key = entry[5] if len(entry) > 5 else None
        drop_cols = entry[6] if len(entry) > 6 else None
        if args.only and args.only not in (path, slug):
            continue
        print(f"-> scoring {path} ({slug}) ...", flush=True)
        try:
            res = score_entry(path, cfg, splits, ser, slug, group_key=group_key,
                              drop_cols=drop_cols)
            rows.append({k: res.get(k, "") for k in fields})
            print(f"   alpha={res['alpha']:.3f} H_inf={res['h_inf']:.3f} "
                  f"episodes={res['n_episodes']} mean_turns={res['mean_turns']} "
                  f"mean_bytes={res['mean_doc_bytes']}", flush=True)
        except Exception as e:
            print(f"   FAIL: {type(e).__name__}: {e}", flush=True)

    os.makedirs("data", exist_ok=True)
    out = "data/agentic_alpha_hinf.csv"
    write_header = not os.path.exists(out) or not args.only
    mode = "w" if not args.only else "a"
    with open(out, mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
