"""Data prep for the α_D = γ/2β training validation (paper §9 #1).

For 4 corpora spanning predicted α_D (toucan β0.20→0.96, swe-zero 0.28→0.47,
jetbrains 0.52→0.34, agentnet 1.30→0.05): fetch ~16MB serialized text, GPT-2
tokenize, save train/val token arrays to $SCRATCH/alpha_d/. The from-scratch
training harness (alpha_d_train.py) then measures the loss-vs-data exponent per
corpus and compares to the predicted α/(2β).
"""
import json, os, sys
import numpy as np
from datasets import load_dataset
from transformers import GPT2TokenizerFast

OUT = os.path.join(os.environ["SCRATCH"], "alpha_d")
os.makedirs(OUT, exist_ok=True)
_tok = GPT2TokenizerFast.from_pretrained("gpt2")
def encode(text):
    # chunk to avoid the fast-tokenizer's overlong-sequence warning/cost
    out = []
    for i in range(0, len(text), 200_000):
        out.extend(_tok.encode(text[i:i+200_000]))
    return out
TARGET_BYTES = 16 * 1024 * 1024  # ~16MB serialized text → ~4M tokens

# predicted α_D = α/(2β): coderforge 0.93, swezero 0.47, jetbrains 0.34, agentnet 0.05
CORPORA = {
    "coderforge":("togethercomputer/CoderForge-Preview-32B-SWE-Bench-Verified-Evaluation-trajectories", None),
    "swezero":  ("AlienKevin/SWE-ZERO-12M-trajectories", None),
    "jetbrains":("JetBrains-Research/agent-trajectories-swe-bench-test-minus-verified", None),
    "agentnet": ("xlangai/AgentNet", None),
}

def ser_row(row):
    # conversations/messages — may be a list OR a JSON string
    for col in ("conversations", "messages"):
        if col in row and row[col] is not None:
            ms = row[col]
            if isinstance(ms, str):
                try: ms = json.loads(ms)
                except Exception: ms = None
            if isinstance(ms, list):
                out = [f"{m.get('from') or m.get('role')}: {m.get('value') or m.get('content','')}"
                       for m in ms if isinstance(m, dict)]
                if out: return "\n".join(out)
    # agentnet: 'traj' list of step dicts + instruction text
    if isinstance(row.get("traj"), list):
        pre = " ".join(str(row.get(k, "")) for k in ("instruction", "natural_language_task", "actual_task"))
        steps = []
        for st in row["traj"]:
            steps.append(" ".join(str(v) for v in st.values() if isinstance(v, (str, int, float)))
                         if isinstance(st, dict) else str(st))
        return pre + "\n" + "\n".join(steps)
    for col in ("text", "content", "instruction"):
        if isinstance(row.get(col), str):
            return row[col]
    return ""

def fetch_text(repo, config):
    # stream rows incrementally (no whole-shard download)
    try:
        ds = load_dataset(repo, config, split="train", streaming=True) if config \
             else load_dataset(repo, split="train", streaming=True)
    except Exception:
        ds = load_dataset(repo, split="train", streaming=True)
    buf, total = [], 0
    for i, r in enumerate(ds):
        if i > 200_000:  # safety: never stream a whole huge dataset
            break
        t = ser_row(r)
        if t:
            buf.append(t); total += len(t.encode())
            if total >= TARGET_BYTES:
                break
    return "\n\n".join(buf)

for name, (repo, config) in CORPORA.items():
    if os.path.exists(f"{OUT}/{name}_train.npy"):
        print(f"{name}: exists, skip", flush=True); continue
    print(f"{name}: fetching {repo} ...", flush=True)
    text = fetch_text(repo, config)
    ids = np.array(encode(text), dtype=np.uint16)
    n = len(ids); cut = int(n * 0.9)
    np.save(f"{OUT}/{name}_train.npy", ids[:cut])
    np.save(f"{OUT}/{name}_val.npy", ids[cut:])
    print(f"{name}: {n:,} tokens ({n/1e6:.2f}M)  train {cut:,} / val {n-cut:,}", flush=True)

print("done; arrays in", OUT, flush=True)
