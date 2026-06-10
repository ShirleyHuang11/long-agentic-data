"""Data prep for the α_D = γ/2β training validation (paper §9 #1).

For 4 corpora spanning predicted α_D (toucan β0.20→0.96, swe-zero 0.28→0.47,
jetbrains 0.52→0.34, agentnet 1.30→0.05): fetch ~16MB serialized text, GPT-2
tokenize, save train/val token arrays to $SCRATCH/alpha_d/. The from-scratch
training harness (alpha_d_train.py) then measures the loss-vs-data exponent per
corpus and compares to the predicted α/(2β).
"""
import json, os, sys
import numpy as np
from huggingface_hub import HfApi, hf_hub_download
import pandas as pd
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

CORPORA = {
    "toucan":   ("Agent-Ark/Toucan-1.5M", "Kimi-K2"),
    "swezero":  ("AlienKevin/SWE-ZERO-12M-trajectories", None),
    "jetbrains":("JetBrains-Research/agent-trajectories-swe-bench-test-minus-verified", None),
    "agentnet": ("xlangai/AgentNet", None),
}

def ser_row(row):
    # generic: conversations/messages list, or a text field
    for col in ("conversations", "messages"):
        if col in row and row[col] is not None:
            ms = row[col]
            return "\n".join(
                f"{m.get('from') or m.get('role')}: {m.get('value') or m.get('content','')}"
                for m in ms if isinstance(m, dict))
    for col in ("text", "content"):
        if col in row and isinstance(row[col], str):
            return row[col]
    return ""

def fetch_text(repo, config):
    api = HfApi()
    files = sorted(s.rfilename for s in api.dataset_info(repo).siblings
                   if s.rfilename.endswith(".parquet"))
    if config:  # filter shards by config dir if present
        cf = [f for f in files if config.lower().replace("-", "").replace("_", "") in f.lower().replace("-", "").replace("_", "")]
        files = cf or files
    buf, total = [], 0
    for f in files:
        df = pd.read_parquet(hf_hub_download(repo, f, repo_type="dataset"))
        for _, r in df.iterrows():
            t = ser_row(r.to_dict())
            if t:
                buf.append(t); total += len(t.encode())
                if total >= TARGET_BYTES:
                    return "\n\n".join(buf)
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
