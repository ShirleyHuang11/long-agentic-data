"""Assemble two matched SFT corpora for the form-vs-choices experiment (finding 20).

TEMPLATE mix (predictable, H_inf=0): open-thoughts/OpenThoughts-Agent-v1-SFT.
HEALTHY mix (H_inf 0.8-1.6): equal thirds from
  - DCAgent/...glm_4.7_traces_jupiter            (conversations, H_inf=0.91)
  - JetBrains-Research/...test-minus-verified     (messages,       H_inf=1.63)
  - AlienKevin/SWE-ZERO-12M-trajectories          (messages,       H_inf=0.80)

All episodes converted to uniform {"messages":[{"role","content"},...]} JSONL.
Token estimate = bytes/4 (UTF-8 content bytes of all messages, sum per episode).
The larger corpus is truncated (whole episodes) so the two token counts match
within 2%.

Outputs:
  $SCRATCH/long-agentic-data/formchoice/{template,healthy}.jsonl
  data/formchoice_manifest.json
"""
import json
import os
import datetime
from datasets import load_dataset

OUT_DIR = "/n/netscratch/chen_lab_seas/Lab/shirleyhuang/long-agentic-data/formchoice"
REPO = "/n/home12/shirleyhuang/long-agentic-data"
TARGET_TOKENS = 30_000_000          # ~30M tokens per corpus (bytes/4)
TARGET_BYTES = TARGET_TOKENS * 4    # ~120 MB content bytes per corpus
MATCH_TOL = 0.02                    # truncate larger corpus to within 2%

# (slug, hf_name, field, revision_sha). field is the list-of-turns column.
TEMPLATE_SRC = [
    ("openthoughts-agent-v1-sft",
     "open-thoughts/OpenThoughts-Agent-v1-SFT", "conversations",
     "c5dc896981f4e3b7c5382669b1d1be0bc4b6a1a6"),
]
HEALTHY_SRC = [
    ("dcagent-glm47-terminus2",
     "DCAgent/neulab-swe-gym-openhands-sampled-trajectories-sandboxes_glm_4.7_traces_jupiter",
     "conversations", None),
    ("jetbrains-swe-test-minus-verified",
     "JetBrains-Research/agent-trajectories-swe-bench-test-minus-verified",
     "messages", "dd79e2540cab4c0cc2c5bab6ea71ddd7ac6c1775"),
    ("swe-zero-12m-traj",
     "AlienKevin/SWE-ZERO-12M-trajectories", "messages",
     "44e028077c55e7255c328516c8bd76080fbb3840"),
]


def normalize(turns):
    """Coerce a conversations/messages list to [{role,content:str}, ...].

    Items use {role,content} or {from,value}; content is always a plain string
    in these four sources. Drop empty-content turns.
    """
    if isinstance(turns, str):
        turns = json.loads(turns)
    out = []
    for m in turns:
        role = m.get("role") or m.get("from") or "user"
        content = m.get("content")
        if content is None:
            content = m.get("value")
        if content is None:
            continue
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        if content == "":
            continue
        out.append({"role": role, "content": content})
    return out


def episode_bytes(messages):
    return sum(len(m["content"].encode("utf-8")) for m in messages)


def collect(src_list, byte_cap):
    """Stream sources, equal byte-budget per source, until byte_cap reached.

    Returns (episodes, per_source_stats). Each episode is a list of messages.
    """
    per_src_cap = byte_cap // len(src_list)
    episodes = []
    stats = []
    for slug, name, field, sha in src_list:
        ds = load_dataset(name, split="train", streaming=True, revision=sha)
        s_eps = 0
        s_bytes = 0
        s_turns = 0
        for row in ds:
            msgs = normalize(row[field])
            if not msgs:
                continue
            b = episode_bytes(msgs)
            if b == 0:
                continue
            episodes.append(msgs)
            s_eps += 1
            s_bytes += b
            s_turns += len(msgs)
            if s_bytes >= per_src_cap:
                break
        stats.append({
            "slug": slug, "hf_dataset": name, "field": field,
            "hf_revision_sha": sha, "episodes": s_eps, "bytes": s_bytes,
            "tokens_est": s_bytes // 4, "turns": s_turns,
            "byte_budget": per_src_cap,
        })
        print(f"  {slug}: {s_eps} eps, {s_bytes:,} B (~{s_bytes//4:,} tok)")
    return episodes, stats


def write_jsonl(path, episodes):
    with open(path, "w") as f:
        for msgs in episodes:
            f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")


def total_bytes(episodes):
    return sum(episode_bytes(e) for e in episodes)


def truncate_to(episodes, target_bytes):
    """Drop whole episodes from the end until <= target_bytes."""
    kept = []
    acc = 0
    for e in episodes:
        b = episode_bytes(e)
        if acc + b > target_bytes:
            break
        kept.append(e)
        acc += b
    return kept


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Collecting TEMPLATE mix...")
    tmpl_eps, tmpl_stats = collect(TEMPLATE_SRC, TARGET_BYTES)
    print("Collecting HEALTHY mix...")
    heal_eps, heal_stats = collect(HEALTHY_SRC, TARGET_BYTES)

    tmpl_b = total_bytes(tmpl_eps)
    heal_b = total_bytes(heal_eps)
    print(f"\nPre-truncation: template={tmpl_b:,} B  healthy={heal_b:,} B")

    # Truncate the larger to match the smaller within 2%.
    truncation = {"applied_to": None, "target_bytes": None}
    smaller = min(tmpl_b, heal_b)
    target = smaller  # match larger down to the smaller corpus size
    if tmpl_b > heal_b * (1 + MATCH_TOL):
        tmpl_eps = truncate_to(tmpl_eps, target)
        truncation = {"applied_to": "template", "target_bytes": target}
    elif heal_b > tmpl_b * (1 + MATCH_TOL):
        heal_eps = truncate_to(heal_eps, target)
        truncation = {"applied_to": "healthy", "target_bytes": target}

    tmpl_b = total_bytes(tmpl_eps)
    heal_b = total_bytes(heal_eps)
    diff = abs(tmpl_b - heal_b) / max(tmpl_b, heal_b)
    print(f"Post-truncation: template={tmpl_b:,} B  healthy={heal_b:,} B  "
          f"diff={diff*100:.2f}%")

    write_jsonl(os.path.join(OUT_DIR, "template.jsonl"), tmpl_eps)
    write_jsonl(os.path.join(OUT_DIR, "healthy.jsonl"), heal_eps)

    manifest = {
        "experiment": "formchoice (finding 20): form vs choices SFT discriminator",
        "built_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "format": '{"messages": [{"role","content"}, ...]} per line',
        "token_estimate_rule": "bytes/4 over UTF-8 content of all messages",
        "target_tokens_per_corpus": TARGET_TOKENS,
        "match_tolerance": MATCH_TOL,
        "truncation": truncation,
        "template": {
            "path": os.path.join(OUT_DIR, "template.jsonl"),
            "h_inf_band": "0.0 (template)",
            "episodes": len(tmpl_eps),
            "bytes": tmpl_b,
            "tokens_est": tmpl_b // 4,
            "sources": tmpl_stats,
        },
        "healthy": {
            "path": os.path.join(OUT_DIR, "healthy.jsonl"),
            "h_inf_band": "0.80-1.63 (healthy)",
            "episodes": len(heal_eps),
            "bytes": heal_b,
            "tokens_est": heal_b // 4,
            "sources": heal_stats,
        },
        "token_match_diff_pct": round(diff * 100, 3),
    }
    with open(os.path.join(REPO, "data", "formchoice_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("\nManifest written to data/formchoice_manifest.json")
    print(f"template: {len(tmpl_eps)} eps, ~{tmpl_b//4:,} tok")
    print(f"healthy:  {len(heal_eps)} eps, ~{heal_b//4:,} tok")


if __name__ == "__main__":
    main()
