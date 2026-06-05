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
]


def iter_docs(path, cfg, splits, ser, group_key=None):
    cfgs = cfg if isinstance(cfg, list) else [cfg]
    for c in cfgs:
        for split in splits:
            ds = load_dataset(path, c, split=split, streaming=True)
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


def score_entry(path, cfg, splits, ser, slug, seed_offset=0, group_key=None):
    docs, turn_counts = [], []
    size = 0
    it = iter_docs(path, cfg, splits, ser, group_key=group_key)
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
        sha = HfApi().dataset_info(path).sha
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
        if args.only and args.only not in (path, slug):
            continue
        print(f"-> scoring {path} ({slug}) ...", flush=True)
        try:
            res = score_entry(path, cfg, splits, ser, slug, group_key=group_key)
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
