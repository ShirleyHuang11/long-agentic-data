"""From-scratch training harness for the α_D = γ/2β validation (paper §9 #1).

Trains a small GPT from scratch on one corpus's GPT-2 token stream (single pass
over the data = the data-limited regime), logging held-out val loss at token
milestones D. The loss-vs-D curve's power-law exponent is the *measured* α_D;
compared across corpora to the predicted α/(2β).

Usage: python alpha_d_train.py --corpus toucan [--out data/alpha_d_<corpus>.json]
A small fixed model + fixed compute is used for ALL corpora (the only thing that
varies is the data), so differences in the loss-vs-D exponent are corpus effects.
"""
import argparse, json, math, os
import numpy as np
import torch, torch.nn as nn
from torch.nn import functional as F

p = argparse.ArgumentParser()
p.add_argument("--corpus", required=True)
p.add_argument("--out", default=None)
p.add_argument("--block", type=int, default=256)
p.add_argument("--batch", type=int, default=32)
p.add_argument("--n_layer", type=int, default=4)
p.add_argument("--n_head", type=int, default=4)
p.add_argument("--n_embd", type=int, default=256)
p.add_argument("--lr", type=float, default=3e-4)
a = p.parse_args()
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
DATA = os.path.join(os.environ["SCRATCH"], "alpha_d")
train = np.load(f"{DATA}/{a.corpus}_train.npy")
val = np.load(f"{DATA}/{a.corpus}_val.npy")
print(f"{a.corpus}: train {len(train):,} tok / val {len(val):,} tok on {DEV}", flush=True)

class Block(nn.Module):
    def __init__(s, ne, nh):
        super().__init__()
        s.ln1 = nn.LayerNorm(ne); s.ln2 = nn.LayerNorm(ne)
        s.attn = nn.MultiheadAttention(ne, nh, batch_first=True)
        s.mlp = nn.Sequential(nn.Linear(ne, 4*ne), nn.GELU(), nn.Linear(4*ne, ne))
    def forward(s, x, mask):
        h = s.ln1(x); x = x + s.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        return x + s.mlp(s.ln2(x))

class GPT(nn.Module):
    def __init__(s, vocab, block, nl, nh, ne):
        super().__init__()
        s.tok = nn.Embedding(vocab, ne); s.pos = nn.Embedding(block, ne)
        s.blocks = nn.ModuleList([Block(ne, nh) for _ in range(nl)])
        s.lnf = nn.LayerNorm(ne); s.head = nn.Linear(ne, vocab, bias=False)
        s.block = block
    def forward(s, idx):
        T = idx.size(1)
        x = s.tok(idx) + s.pos(torch.arange(T, device=idx.device))
        mask = torch.triu(torch.full((T, T), float("-inf"), device=idx.device), 1)
        for b in s.blocks: x = b(x, mask)
        return s.head(s.lnf(x))

model = GPT(50257, a.block, a.n_layer, a.n_head, a.n_embd).to(DEV)
nparam = sum(p.numel() for p in model.parameters())
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)
print(f"model {nparam/1e6:.1f}M params", flush=True)

def get_batch(arr, n):
    ix = torch.randint(len(arr) - a.block - 1, (n,))
    x = torch.stack([torch.from_numpy(arr[i:i+a.block].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(arr[i+1:i+1+a.block].astype(np.int64)) for i in ix])
    return x.to(DEV), y.to(DEV)

@torch.no_grad()
def val_loss():
    model.eval(); losses = []
    for _ in range(40):
        x, y = get_batch(val, a.batch)
        logits = model(x)
        losses.append(F.cross_entropy(logits.view(-1, 50257), y.view(-1)).item())
    model.train(); return sum(losses) / len(losses)

# single pass: step through the train stream sequentially; log val loss at token milestones
toks_per_step = a.batch * a.block
milestones = [0.25e6, 0.5e6, 1e6, 2e6, 3e6]
curve = []
seen, mi = 0, 0
max_tokens = min(len(train) - a.block - 1, int(milestones[-1] * 1.05))
model.train()
while seen < max_tokens:
    x, y = get_batch(train, a.batch)
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, 50257), y.view(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    seen += toks_per_step
    if mi < len(milestones) and seen >= milestones[mi]:
        vl = val_loss()
        curve.append({"D": milestones[mi], "val_loss": vl})
        print(f"  D={milestones[mi]/1e6:.2f}M  val_loss={vl:.4f}", flush=True)
        mi += 1

out = a.out or f"data/alpha_d_{a.corpus}.json"
json.dump({"corpus": a.corpus, "n_params": nparam, "curve": curve}, open(out, "w"), indent=2)
print(f"wrote {out}", flush=True)
