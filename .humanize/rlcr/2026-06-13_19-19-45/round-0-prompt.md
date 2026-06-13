Read and execute below with ultrathink

## Goal Tracker Setup (REQUIRED FIRST STEP)

Before starting implementation, you MUST initialize the Goal Tracker:

1. Read @/n/home07/hanlinzhang/projects/holographic-data/.humanize/rlcr/2026-06-13_19-19-45/goal-tracker.md
2. If the "Ultimate Goal" section says "[To be extracted...]", extract a clear goal statement from the plan
3. If the "Acceptance Criteria" section says "[To be defined...]", define 3-7 specific, testable criteria
4. Populate the "Active Tasks" table with MAINLINE tasks from the plan, mapping each to an AC and filling Tag/Owner
5. Record any already-known side issues in either "Blocking Side Issues" or "Queued Side Issues"
6. Write the updated goal-tracker.md

## Round Contract Setup (REQUIRED BEFORE CODING)

Before starting implementation, create @/n/home07/hanlinzhang/projects/holographic-data/.humanize/rlcr/2026-06-13_19-19-45/round-0-contract.md with:

1. **One mainline objective** for this round
2. **Target ACs** (1-2 ACs only)
3. **Blocking side issues in scope** for this round
4. **Queued side issues out of scope** for this round
5. **Round success criteria**

Use this contract to keep the round focused. Do NOT let non-blocking bugs or cleanup work replace the mainline objective.

**IMPORTANT**: The IMMUTABLE SECTION can only be modified in Round 0. After this round, it becomes read-only.

---

## Implementation Plan

For all tasks that need to be completed, please use the Task system (TaskCreate, TaskUpdate, TaskList).

Every task MUST start with exactly one lane tag:
- `[mainline]` for plan-derived work that directly advances the round objective
- `[blocking]` for issues that prevent the mainline objective from succeeding safely
- `[queued]` for non-blocking bugs, cleanup, or follow-up work

Rules:
- `[mainline]` tasks are the primary success condition for the round
- `[blocking]` tasks may be resolved in the round only if they truly block mainline progress
- `[queued]` tasks must NOT become the round objective and do NOT need to be cleared before moving on
- If a new issue is not blocking the current objective, tag it `[queued]` and keep moving on the mainline

## Task Tag Routing (MUST FOLLOW)

Each task must have one routing tag from the plan: `coding` or `analyze`.

- Tag `coding`: Claude executes the task directly.
- Tag `analyze`: Claude must execute via `/humanize:ask-codex`, then integrate Codex output.
- Keep Goal Tracker "Active Tasks" columns **Tag** and **Owner** aligned with execution (`coding -> claude`, `analyze -> codex`).
- If a task has no explicit tag, default to `coding` (Claude executes directly).

# Holographic Data for Length Generalization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "holographic" nested-monoid reduction-trace data generator and the eval/plumbing needed to map the (β,γ) phase diagram of *length generalization* (train short/shallow → test long/deep) for a vanilla decoder-only Transformer trained with pure next-token prediction.

**Architecture:** Extend the existing `case/phase/` experiment harness (it already does NTP training, multi-length eval, retention CSVs, and dispatches by `task`/`architecture` via config). We add: a third `(β,γ)` task `nested_monoid` (sibling to `kv` and `logical_folding`), a RoPE decoder-only model (so length-gen isn't confounded by APE's untrained far positions), answer-masked accuracy, a configurable param floor (for the small grid tier), two `(β,γ)` plan factories, and a knob-verification module (measure realized β̂,γ̂). Then run staged sweeps A→D.

**Tech Stack:** Python, PyTorch, NumPy, OmegaConf (config), Slurm (FASRC kempner/seas_gpu), pytest. Reuses `phase_core.train_one_model`, `phase_sweep.py`, `model_mamba.py`, `utils.py`, `plot_phase_diagram.py`.

**Spec:** `docs/superpowers/specs/2026-06-13-holographic-length-gen-design.md`

---

## Deliberate decisions (resolving two spec tensions)

1. **Code location.** The spec proposed a new `case/recursion/` module. The codebase already adds `(β,γ)` tasks as siblings inside `case/phase/` (KV in `phase_core.py`, logical-folding in `data_logical_folding.py`, both dispatched by `phase_core.make_generator`). To stay DRY and follow convention (spec §7: "reuse, do not fork; extend the architecture switch"), **all new code lives in `case/phase/`** as task `nested_monoid`. Reports/results go in `case/phase/results/`.
2. **Positional encoding.** The spec text says "vanilla Transformer (GPT, RoPE)" but `case/phase/model.py` (`TinyCausalTransformer`) uses **learned absolute PE** whose positions beyond `train_len` are untrained — structurally hostile to length extrapolation and a confound for a *data* claim. We implement a **RoPE** decoder-only model (`model_rope.py`, arch key `transformer_rope`) as the primary, since length generalization is the entire point. `TinyCausalTransformer` (APE) stays available for comparison but is not the headline.

---

## File map

**New files (all in `case/phase/`):**
- `model_rope.py` — `RoPECausalTransformer` (decoder-only, rotary PE, SDPA causal attn). Param count reuses `model.count_non_embedding_params`.
- `data_nested_monoid.py` — `NestedMonoidGenerator` (the core artifact).
- `knob_verify.py` — `estimate_beta`, `estimate_gamma` + CLI to run the §3.4 verification gate.
- `configs/models/holo_small.yaml` — grid-tier preset (RoPE, ~15M, lowered floor).
- `configs/models/holo_100m.yaml` — scale-tier preset (RoPE, ≥100M).
- `configs/models/holo_mamba_small.yaml` — Mamba control preset (grid-tier).

**Edited files:**
- `phase_core.py` — `make_generator` (+`nested_monoid`); new pure helper `masked_next_token_acc`; `evaluate_at_length` (thread mask, return masked `acc` when `eval_answer_masked`); `train_one_model` (+`transformer_rope` arch; pass eval flag).
- `phase_sweep.py` — `build_config`: configurable floor via `model.min_non_embed_params`; `transformer_rope` uses `count_non_embedding_params`.
- `data_generator.py` — add `holo_anchors` + `holo_grid` plan factories; register in `_FACTORIES`.

**Reused unchanged:** `model_mamba.py`, `utils.py`, `plot_phase_diagram.py`, `aggregate_report.py`, `sweep.slurm`, `sweep.sh`.

**Test files (new):** `case/phase/tests/test_model_rope.py`, `test_nested_monoid.py`, `test_knob_verify.py`, `test_phase_core_masked.py`, `test_build_config_floor.py`, `test_holo_plans.py`.

> All commands assume CWD = repo root `/n/home07/hanlinzhang/projects/holographic-data` and the env: `source /n/netscratch/kempner_sham_lab/Lab/hanlinzhang/envs/flow/bin/activate`. Run pytest with `PYTHONPATH=case/phase` so bare-module imports (`import model`, etc.) resolve like the harness does.

---

## Task 1: RoPE decoder-only model

**Files:**
- Create: `case/phase/model_rope.py`
- Test: `case/phase/tests/test_model_rope.py`

- [ ] **Step 1: Write the failing test**

```python
# case/phase/tests/test_model_rope.py
import torch
from model_rope import RoPECausalTransformer
from model import count_non_embedding_params


def _small():
    return RoPECausalTransformer(vocab_size=40, d_model=32, nhead=4,
                                 ff_mult=4, num_layers=2, dropout=0.0)


def test_forward_shape():
    m = _small().eval()
    x = torch.randint(0, 40, (3, 17))
    y = m(x)
    assert y.shape == (3, 17, 40)


def test_causal_future_does_not_leak():
    # Changing the LAST token must not change outputs at earlier positions.
    torch.manual_seed(0)
    m = _small().eval()
    x = torch.randint(0, 40, (2, 12))
    with torch.no_grad():
        a = m(x)
        x2 = x.clone()
        x2[:, -1] = (x2[:, -1] + 1) % 40
        b = m(x2)
    assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-5)


def test_length_extrapolation_runs():
    # RoPE has no learned position table → forward at a length never "configured".
    m = _small().eval()
    for T in (8, 64, 256):
        y = m(torch.randint(0, 40, (1, T)))
        assert y.shape == (1, T, 40)


def test_param_count_matches_helper():
    d, V = 32, 40
    m = _small()
    non_embed = sum(p.numel() for n, p in m.named_parameters()
                    if not n.startswith("embedding."))
    expected = count_non_embedding_params(d_model=d, nhead=4, ff_mult=4,
                                          num_layers=2, vocab_size=V) + 2 * d
    assert non_embed == expected  # +2d for the final LayerNorm
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd case/phase && PYTHONPATH=. python -m pytest tests/test_model_rope.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'model_rope'`.

- [ ] **Step 3: Write the implementation**

```python
# case/phase/model_rope.py
#!/usr/bin/env python3
"""RoPE decoder-only causal Transformer.

Same forward signature as model.TinyCausalTransformer and model_mamba.MambaCausal:
    forward(x: LongTensor[B, T]) -> FloatTensor[B, T, vocab]

Uses rotary position embeddings (no learned position table) so the model can be
evaluated at sequence lengths longer than any seen in training — the right
default for a length-generalization study. Non-embedding param count equals
model.count_non_embedding_params(...) + 2*d_model (the extra final LayerNorm).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _rope_cache(seq_len: int, head_dim: int, device, dtype):
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)            # (T, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)     # (T, head_dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(q, k, cos, sin):
    # q,k: (B, H, T, Dh); cos,sin: (T, Dh)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    return q, k


class _Block(nn.Module):
    def __init__(self, d_model: int, nhead: int, ff_mult: int, dropout: float) -> None:
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.proj = nn.Linear(d_model, d_model, bias=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, ff_mult * d_model, bias=True)
        self.fc2 = nn.Linear(ff_mult * d_model, d_model, bias=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        cos, sin = _rope_cache(T, self.head_dim, x.device, x.dtype)
        q, k = _apply_rope(q, k, cos, sin)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.drop(self.proj(a))
        h = self.ln2(x)
        h = self.fc2(self.drop(F.gelu(self.fc1(h))))
        return x + self.drop(h)


class RoPECausalTransformer(nn.Module):
    def __init__(self, vocab_size: int = 40, d_model: int = 256, nhead: int = 8,
                 ff_mult: int = 4, num_layers: int = 6, dropout: float = 0.1,
                 **_kwargs) -> None:  # ignore max_ctx_tokens etc. for drop-in compat
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [_Block(d_model, nhead, ff_mult, dropout) for _ in range(num_layers)]
        )
        self.norm_f = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embedding(x)
        for blk in self.blocks:
            h = blk(h)
        return self.fc(self.norm_f(h))
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd case/phase && PYTHONPATH=. python -m pytest tests/test_model_rope.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add case/phase/model_rope.py case/phase/tests/test_model_rope.py
git commit -m "feat(holo): RoPE decoder-only model for length-gen study"
```

---

## Task 2: NestedMonoidGenerator — core (holographic mode)

**Files:**
- Create: `case/phase/data_nested_monoid.py`
- Test: `case/phase/tests/test_nested_monoid.py`

- [ ] **Step 1: Write the failing test**

```python
# case/phase/tests/test_nested_monoid.py
import numpy as np
import torch
from data_nested_monoid import NestedMonoidGenerator


def test_shape_and_vocab():
    g = NestedMonoidGenerator(p=31, n_filler=6)
    assert g.vocab_size == 31 + 3 + 6
    x, y, m = g.generate_batch(batch_size=4, seq_len=128, beta=1.0, gamma=0.2)
    assert x.shape == (4, 128) and y.shape == (4, 128) and m.shape == (4, 128)
    assert x.max().item() < g.vocab_size and x.min().item() >= 0


def test_determinism():
    g = NestedMonoidGenerator(p=31, n_filler=6)
    np.random.seed(123)
    x1, _, _ = g.generate_batch(2, 96, beta=0.5, gamma=0.3)
    np.random.seed(123)
    x2, _, _ = g.generate_batch(2, 96, beta=0.5, gamma=0.3)
    assert torch.equal(x1, x2)


def test_answer_is_correct_affine_fold():
    g = NestedMonoidGenerator(p=31, n_filler=6)
    np.random.seed(7)
    x, y, m, meta = g.generate_batch(8, 160, beta=0.8, gamma=0.1, return_meta=True)
    for bi, md in enumerate(meta):
        v = md["x0"]
        for j in md["order"]:           # application order pi
            v = (md["a"][j] * v + md["b"][j]) % g.p
        assert v == md["answer"]
        # ANS token is the last masked position and equals the answer value token
        ans_pos = int(np.where(m[bi].numpy() > 0)[0][-1])
        assert int(x[bi, ans_pos].item()) == md["answer"]
        assert int(y[bi, ans_pos].item()) == md["answer"]


def test_mask_marks_only_value_positions():
    g = NestedMonoidGenerator(p=31, n_filler=6)
    np.random.seed(1)
    x, y, m, meta = g.generate_batch(4, 160, beta=0.8, gamma=0.1, return_meta=True)
    masked_tokens = x[m > 0]
    assert int(masked_tokens.max().item()) < g.p   # only value tokens are masked
    for bi, md in enumerate(meta):
        assert int(m[bi].sum().item()) == md["depth"] + 1  # trace values + answer
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd case/phase && PYTHONPATH=. python -m pytest tests/test_nested_monoid.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'data_nested_monoid'`.

- [ ] **Step 3: Write the implementation**

```python
# case/phase/data_nested_monoid.py
"""Nested-monoid (affine mod p) reduction-trace generator parametrized by (β, γ).

Third (β, γ) task for case/phase (sibling to AlgorithmicKVGenerator and
LogicalFoldingGenerator). Realizes the "holographic" construct from
assets/holo.pdf: ONE deep affine composition per example, emitted as an
autoregressive reduction trace, so a vanilla decoder-only Transformer trained
with next-token prediction on SHORT (shallow) sequences can be tested on LONGER
(deeper) ones.

Operator (monoid): affine maps over Z_p, v -> (a*v + b) mod p, closed under
composition => the per-step fold rule is identical at every depth (provably
depth-invariant ground truth).

Vocab (modulus p, n_filler filler symbols; vocab_size = p + 3 + n_filler):
    0..p-1      value tokens (Z_p; also a,b params, x0, trace values, answer)
    p           OP   (precedes each (a,b) op definition)
    p+1         RUN  (PROG -> reduction trace)
    p+2         ANS  (precedes final answer)
    p+3..       filler tokens (semantically inert; injected at rate gamma)

Layout (one example):
    [filler pad] OP a1 b1 ... OP aD bD x0 RUN v1 ... vD ANS answer
Trace applies ops in a beta-sampled order pi (v_k = op_pi(k)(v_{k-1})); the op
chosen at each step has reach ~ Zipf(beta) over |write_pos - op_pos| (same
convention as the KV / logical-folding tasks). answer = v_D depends on ALL ops.

Modes:
    'holographic' : depth D auto-scaled so the complete composition fits seq_len.
    'truncated'   : build depth=source_depth (> seq_len), keep only the last
                    seq_len tokens (a "slice" of long data) — the H2 control.

generate_batch(batch_size, seq_len, beta, gamma) -> (x, y, m) matches the
sibling generators. y = target value at masked positions; m marks trace values
+ final answer. return_meta=True (tests only) also returns ground-truth dicts.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch


class NestedMonoidGenerator:
    def __init__(self, p: int = 31, n_filler: int = 6,
                 mode: str = "holographic", source_depth: int = 256) -> None:
        if p < 2:
            raise ValueError(f"p must be >= 2, got {p}")
        if n_filler < 1:
            raise ValueError(f"n_filler must be >= 1, got {n_filler}")
        if mode not in ("holographic", "truncated"):
            raise ValueError(f"mode must be 'holographic'|'truncated', got {mode!r}")
        self.p = int(p)
        self.n_filler = int(n_filler)
        self.mode = str(mode)
        self.source_depth = int(source_depth)
        self.OP = self.p
        self.RUN = self.p + 1
        self.ANS = self.p + 2
        self.filler_tokens = np.arange(self.p + 3, self.p + 3 + self.n_filler)
        self.vocab_size = self.p + 3 + self.n_filler

    def _depth_for_len(self, seq_len: int, gamma: float) -> int:
        # non-filler budget per op ~ 4 (OP,a,b in PROG + 1 trace value); +4 overhead
        usable = max(1.0, seq_len * (1.0 - min(gamma, 0.95)) - 4.0)
        return max(1, int(usable // 4))

    def _assemble(self, D: int, beta: float, gamma: float
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        a = np.random.randint(1, self.p, size=D)
        b = np.random.randint(0, self.p, size=D)
        x0 = int(np.random.randint(0, self.p))

        toks: List[int] = []
        op_pos = np.zeros(D, dtype=np.int64)
        for i in range(D):
            if i > 0 and np.random.rand() < gamma:
                toks.append(int(np.random.choice(self.filler_tokens)))
            op_pos[i] = len(toks)
            toks.extend([self.OP, int(a[i]), int(b[i])])
        toks.append(x0)
        toks.append(self.RUN)

        remaining = list(range(D))
        order: List[int] = []
        trace_pos: List[int] = []
        v = x0
        for _ in range(D):
            write_pos = len(toks)
            reach = np.abs(write_pos - op_pos[remaining]).astype(np.float64)
            reach = np.maximum(reach, 1.0)
            if beta <= 1e-3:
                probs = np.ones_like(reach)
            else:
                probs = 1.0 / np.power(reach, beta + 1.0)
            probs = probs / probs.sum()
            sel = int(np.random.choice(len(remaining), p=probs))
            j = remaining.pop(sel)
            order.append(j)
            v = int((a[j] * v + b[j]) % self.p)
            if np.random.rand() < gamma:
                toks.append(int(np.random.choice(self.filler_tokens)))
            trace_pos.append(len(toks))
            toks.append(v)
        toks.append(self.ANS)
        ans_pos = len(toks)
        toks.append(v)

        arr = np.array(toks, dtype=np.int64)
        m = np.zeros(len(arr), dtype=np.float32)
        y = np.zeros(len(arr), dtype=np.int64)
        for pos in trace_pos:
            m[pos] = 1.0
            y[pos] = arr[pos]
        m[ans_pos] = 1.0
        y[ans_pos] = v
        meta = {"depth": D, "x0": x0, "a": a, "b": b, "order": order, "answer": v}
        return arr, y, m, meta

    @staticmethod
    def _fit(arr, y, m, seq_len, left_window: bool):
        if len(arr) > seq_len:
            arr, y, m = arr[-seq_len:], y[-seq_len:], m[-seq_len:]
        elif len(arr) < seq_len:
            pad = seq_len - len(arr)
            arr = np.concatenate([np.zeros(pad, dtype=np.int64), arr]) if not left_window else arr
        return arr, y, m

    def _build_one(self, seq_len: int, beta: float, gamma: float):
        if self.mode == "truncated":
            arr, y, m, meta = self._assemble(self.source_depth, beta, gamma)
            arr, y, m = arr[-seq_len:], y[-seq_len:], m[-seq_len:]
        else:
            D = self._depth_for_len(seq_len, gamma)
            while True:
                arr, y, m, meta = self._assemble(D, beta, gamma)
                if len(arr) <= seq_len or D <= 1:
                    break
                D -= 1
            if len(arr) > seq_len:        # D==1 still overflows tiny seq_len
                arr, y, m = arr[-seq_len:], y[-seq_len:], m[-seq_len:]
        if len(arr) < seq_len:
            pad = seq_len - len(arr)
            fill = np.random.choice(self.filler_tokens, size=pad)
            arr = np.concatenate([fill, arr])
            m = np.concatenate([np.zeros(pad, dtype=np.float32), m])
            y = np.concatenate([np.zeros(pad, dtype=np.int64), y])
        return arr, y, m, meta

    def generate_batch(self, batch_size: int, seq_len: int, beta: float,
                       gamma: float, return_meta: bool = False):
        xb = np.zeros((batch_size, seq_len), dtype=np.int64)
        yb = np.zeros((batch_size, seq_len), dtype=np.int64)
        mb = np.zeros((batch_size, seq_len), dtype=np.float32)
        metas: List[Dict] = []
        for bi in range(batch_size):
            arr, y, m, meta = self._build_one(seq_len, beta, gamma)
            xb[bi], yb[bi], mb[bi] = arr, y, m
            if return_meta:
                metas.append(meta)
        x, y_t, m_t = torch.tensor(xb), torch.tensor(yb), torch.tensor(mb)
        if return_meta:
            return x, y_t, m_t, metas
        return x, y_t, m_t
```

> Note: the unused `_fit` static method above is dead — delete it; it was a sketch. Final file must not include `_fit`. (Self-review will catch leftover dead code.)

- [ ] **Step 4: Remove the dead `_fit` method, then run tests**

Delete the `_fit` static method (it is not called). Then:
Run: `cd case/phase && PYTHONPATH=. python -m pytest tests/test_nested_monoid.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add case/phase/data_nested_monoid.py case/phase/tests/test_nested_monoid.py
git commit -m "feat(holo): nested-monoid reduction-trace generator (holographic mode)"
```

---

## Task 3: NestedMonoidGenerator — knob behavior (β reach, γ density, depth↔length)

**Files:**
- Modify: `case/phase/tests/test_nested_monoid.py` (append)

- [ ] **Step 1: Append the failing tests**

```python
# append to case/phase/tests/test_nested_monoid.py

def _filler_fraction(g, x):
    return float((x >= g.filler_tokens.min()).float().mean().item())


def test_gamma_controls_filler_fraction():
    g = NestedMonoidGenerator(p=31, n_filler=6)
    np.random.seed(0)
    x_lo, _, _ = g.generate_batch(16, 512, beta=1.0, gamma=0.1)
    x_hi, _, _ = g.generate_batch(16, 512, beta=1.0, gamma=0.6)
    assert _filler_fraction(g, x_lo) < _filler_fraction(g, x_hi)
    assert _filler_fraction(g, x_hi) > 0.30   # high gamma => filler-dominated


def test_beta_controls_reach():
    # Smaller beta => the application order reaches farther-back ops on average.
    g = NestedMonoidGenerator(p=31, n_filler=6)

    def mean_first_reach(beta):
        np.random.seed(5)
        _, _, _, meta = g.generate_batch(32, 320, beta=beta, gamma=0.0, return_meta=True)
        # order[0] is the op index applied first; smaller index = farther front.
        return float(np.mean([md["order"][0] for md in meta]))

    far = mean_first_reach(0.05)   # holographic: tends to grab front (small idx)
    near = mean_first_reach(3.0)   # local: tends to grab end (large idx)
    assert far < near


def test_depth_scales_with_seq_len():
    g = NestedMonoidGenerator(p=31, n_filler=6)
    np.random.seed(2)
    _, _, _, m_short = g.generate_batch(8, 128, beta=1.0, gamma=0.1, return_meta=True)
    _, _, _, m_long = g.generate_batch(8, 1024, beta=1.0, gamma=0.1, return_meta=True)
    assert np.mean([d["depth"] for d in m_long]) > 4 * np.mean([d["depth"] for d in m_short])


def test_truncated_mode_returns_window():
    g = NestedMonoidGenerator(p=31, n_filler=6, mode="truncated", source_depth=200)
    x, y, m = g.generate_batch(4, 128, beta=1.0, gamma=0.1)
    assert x.shape == (4, 128)
    # A deep (200) composition truncated to 128 tokens cannot contain all ops:
    n_ops_in_window = int((x == g.OP).sum(dim=1).float().mean().item())
    assert n_ops_in_window < 200
```

- [ ] **Step 2: Run to verify the new tests pass** (the generator from Task 2 already implements these)

Run: `cd case/phase && PYTHONPATH=. python -m pytest tests/test_nested_monoid.py -q`
Expected: PASS (8 passed). If `test_beta_controls_reach` fails, the reach-sampling sign is wrong — verify smaller β yields flatter `probs` (more mass on far/front ops); the implementation in Task 2 is correct.

- [ ] **Step 3: Commit**

```bash
git add case/phase/tests/test_nested_monoid.py
git commit -m "test(holo): β reach / γ density / depth↔length knob behavior"
```

---

## Task 4: phase_core integration — masked answer accuracy + nested_monoid + RoPE

**Files:**
- Modify: `case/phase/phase_core.py`
- Test: `case/phase/tests/test_phase_core_masked.py`

- [ ] **Step 1: Write the failing test**

```python
# case/phase/tests/test_phase_core_masked.py
import torch
from phase_core import masked_next_token_acc, make_generator


def test_masked_accuracy_counts_only_masked_positions():
    # vocab=3, T=4. logits argmax = [2,2,2,2] for both rows.
    logits = torch.zeros(1, 4, 3)
    logits[..., 2] = 1.0
    targets = torch.tensor([[2, 0, 2, 0]])    # correct at positions 0 and 2
    mask = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    acc = masked_next_token_acc(logits, targets, mask)
    assert abs(acc - 1.0) < 1e-6              # both masked positions correct
    mask2 = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    acc2 = masked_next_token_acc(logits, targets, mask2)
    assert abs(acc2 - 0.5) < 1e-6             # 2 of 4 correct


def test_make_generator_nested_monoid():
    g = make_generator({"task": "nested_monoid", "monoid_p": 31, "monoid_filler": 6})
    x, y, m = g.generate_batch(2, 64, beta=1.0, gamma=0.2)
    assert x.shape == (2, 64)
    assert g.vocab_size == 40
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd case/phase && PYTHONPATH=. python -m pytest tests/test_phase_core_masked.py -q`
Expected: FAIL — `ImportError: cannot import name 'masked_next_token_acc'`.

- [ ] **Step 3: Edit `phase_core.py`**

(3a) In `make_generator` (currently ends with the `logical_folding` branch and a `raise ValueError`), add a `nested_monoid` branch **before** the `raise`:

```python
    if task == "nested_monoid":
        from data_nested_monoid import NestedMonoidGenerator
        return NestedMonoidGenerator(
            p=int(config.get("monoid_p", 31)),
            n_filler=int(config.get("monoid_filler", 6)),
            mode=str(config.get("gen_mode", "holographic")),
            source_depth=int(config.get("source_depth", 256)),
        )
    raise ValueError(f"unknown task={task!r} (expected 'kv', 'logical_folding', or 'nested_monoid')")
```

(3b) Add a pure helper after `next_token_loss_and_acc`:

```python
def masked_next_token_acc(logits: torch.Tensor, targets: torch.Tensor,
                          target_mask: torch.Tensor) -> float:
    """Next-token accuracy over positions where target_mask == 1 (answer/trace)."""
    pred = logits.argmax(-1)
    correct = (pred == targets).float() * target_mask
    denom = float(target_mask.sum().item())
    if denom <= 0:
        return 0.0
    return float(correct.sum().item() / denom)
```

(3c) Replace `evaluate_at_length` so it can score the answer mask. The mask `m` from `generate_batch` is aligned to `x`; predicting `x[:,1:]` uses `m[:,1:]` as the target mask:

```python
def evaluate_at_length(
    model: nn.Module,
    gen,
    beta: float,
    gamma: float,
    seq_len: int,
    eval_batches: int,
    batch_size: int,
    device: torch.device,
    answer_masked: bool = False,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct_all = 0.0
    total_count_all = 0.0
    masked_correct = 0.0
    masked_count = 0.0

    with torch.no_grad():
        for _ in range(eval_batches):
            x, _, m = gen.generate_batch(batch_size, seq_len, beta, gamma)
            x = x.to(device)
            if x.size(1) < 2:
                continue
            inp = x[:, :-1]
            y = x[:, 1:]
            logits = model(inp)
            loss, acc = next_token_loss_and_acc(logits, y)
            total_loss += float(loss.item())
            total_correct_all += float(acc.item()) * float(y.numel())
            total_count_all += float(y.numel())
            if answer_masked:
                tmask = m[:, 1:].to(device)
                pred = logits.argmax(-1)
                masked_correct += float(((pred == y).float() * tmask).sum().item())
                masked_count += float(tmask.sum().item())

    acc_all = total_correct_all / max(total_count_all, EPS)
    out = {"loss": total_loss / max(eval_batches, 1), "acc_all": acc_all}
    if answer_masked:
        acc_ans = masked_correct / max(masked_count, EPS)
        out["acc_answer"] = acc_ans
        out["acc"] = acc_ans      # primary metric = answer accuracy
    else:
        out["acc"] = acc_all
    return out
```

(3d) In `train_one_model`, add the RoPE architecture branch. Replace the `if arch == "mamba": ... else: TinyCausalTransformer(...)` block with:

```python
    arch = str(config.get("architecture", "transformer"))
    if arch == "mamba":
        from model_mamba import MambaCausal
        model = MambaCausal(
            vocab_size=int(config["vocab_size"]),
            d_model=int(config["d_model"]),
            num_layers=int(config["num_layers"]),
            d_state=int(config.get("d_state", 16)),
            d_conv=int(config.get("d_conv", 3)),
            dropout=float(config["dropout"]),
        ).to(device)
    elif arch == "transformer_rope":
        from model_rope import RoPECausalTransformer
        model = RoPECausalTransformer(
            vocab_size=int(config["vocab_size"]),
            d_model=int(config["d_model"]),
            nhead=int(config["nhead"]),
            ff_mult=int(config["ff_mult"]),
            num_layers=int(config["num_layers"]),
            dropout=float(config["dropout"]),
        ).to(device)
    else:
        model = TinyCausalTransformer(
            vocab_size=int(config["vocab_size"]),
            d_model=int(config["d_model"]),
            nhead=int(config["nhead"]),
            ff_mult=int(config["ff_mult"]),
            num_layers=int(config["num_layers"]),
            max_ctx_tokens=max_len,
            dropout=float(config["dropout"]),
        ).to(device)
```

(3e) In `train_one_model`, pass the eval flag through the eval loop (the `for L in eval_lengths` block):

```python
    answer_masked = bool(config.get("eval_answer_masked", False))
    eval_by_len: Dict[int, Dict[str, float]] = {}
    for L in eval_lengths:
        eval_by_len[int(L)] = evaluate_at_length(
            model=model, gen=gen, beta=beta, gamma=gamma, seq_len=int(L),
            eval_batches=int(config["eval_batches"]),
            batch_size=int(config["eval_batch_size"]),
            device=device, answer_masked=answer_masked,
        )
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd case/phase && PYTHONPATH=. python -m pytest tests/test_phase_core_masked.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Regression — existing kv/logical_folding eval still returns `acc`**

Run: `cd case/phase && PYTHONPATH=. python -c "from phase_core import make_generator; g=make_generator({'task':'kv','vocab_size':60}); import torch; print(g.generate_batch(2,64,1.0,0.2)[0].shape)"`
Expected: `torch.Size([2, 64])` (no exception — `acc` still present for unmasked path).

- [ ] **Step 6: Commit**

```bash
git add case/phase/phase_core.py case/phase/tests/test_phase_core_masked.py
git commit -m "feat(holo): masked answer-accuracy + nested_monoid + RoPE in phase_core"
```

---

## Task 5: phase_sweep — configurable param floor + RoPE param count

**Files:**
- Modify: `case/phase/phase_sweep.py`
- Test: `case/phase/tests/test_build_config_floor.py`

- [ ] **Step 1: Write the failing test**

```python
# case/phase/tests/test_build_config_floor.py
import pytest
from omegaconf import OmegaConf
from phase_sweep import build_config


def _model_cfg(**over):
    base = dict(architecture="transformer_rope", vocab_size=40, d_model=128,
                nhead=4, ff_mult=4, num_layers=4, dropout=0.1, train_len=128,
                eval_lengths=[128, 512], train_steps=10, train_batch_size=8,
                eval_batch_size=4, eval_batches=2, lr=3e-4, grad_clip=1.0,
                renyi_seq_len=2000, renyi_qs=[1.0], renyi_orders=[1, 2])
    base.update(over)
    return OmegaConf.create(base)


def test_small_model_rejected_by_default_floor():
    with pytest.raises(ValueError):
        build_config(_model_cfg(), long_len=512, task="nested_monoid")


def test_small_model_passes_with_lowered_floor():
    cfg = build_config(_model_cfg(min_non_embed_params=100_000),
                       long_len=512, task="nested_monoid")
    assert cfg["task"] == "nested_monoid"
    assert cfg["architecture"] == "transformer_rope"
    assert cfg["_non_embed_params"] > 100_000
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd case/phase && PYTHONPATH=. python -m pytest tests/test_build_config_floor.py -q`
Expected: FAIL — `test_small_model_passes_with_lowered_floor` raises ValueError (floor still hard-coded), or `transformer_rope` hits the `else` param-count branch (which is actually correct) but the floor blocks it.

- [ ] **Step 3: Edit `build_config` in `phase_sweep.py`**

Replace the arch param-count + floor block. The `transformer_rope` arch uses the same `count_non_embedding_params` as the APE transformer (identical block structure):

```python
    arch = str(cfg.get("architecture", "transformer"))
    if arch == "mamba":
        from model_mamba import count_mamba_params
        n_non_embed = count_mamba_params(
            d_model=int(cfg["d_model"]),
            num_layers=int(cfg["num_layers"]),
            vocab_size=int(cfg["vocab_size"]),
            d_state=int(cfg.get("d_state", 16)),
            d_conv=int(cfg.get("d_conv", 3)),
        )
    else:  # 'transformer' (APE) and 'transformer_rope' share the count
        n_non_embed = count_non_embedding_params(
            d_model=int(cfg["d_model"]),
            nhead=int(cfg["nhead"]),
            ff_mult=int(cfg["ff_mult"]),
            num_layers=int(cfg["num_layers"]),
            vocab_size=int(cfg["vocab_size"]),
        )
    floor = int(cfg.get("min_non_embed_params", MIN_NON_EMBED_PARAMS))
    if n_non_embed < floor:
        raise ValueError(
            f"non-embedding params = {n_non_embed:,} (~{n_non_embed/1e6:.2f}M) "
            f"< floor {floor:,}. Bump model.d_model / model.num_layers, or lower "
            f"model.min_non_embed_params for a small-scale grid sweep."
        )
    cfg["_non_embed_params"] = int(n_non_embed)
    return cfg
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd case/phase && PYTHONPATH=. python -m pytest tests/test_build_config_floor.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add case/phase/phase_sweep.py case/phase/tests/test_build_config_floor.py
git commit -m "feat(holo): configurable param floor + RoPE arch in build_config"
```

---

## Task 6: (β,γ) plan factories — holo_anchors + holo_grid

**Files:**
- Modify: `case/phase/data_generator.py`
- Test: `case/phase/tests/test_holo_plans.py`

- [ ] **Step 1: Write the failing test**

```python
# case/phase/tests/test_holo_plans.py
from data_generator import _FACTORIES
from omegaconf import OmegaConf


def test_holo_anchors_pairs():
    pl = _FACTORIES["holo_anchors"](OmegaConf.create({}))
    pairs = {(round(b, 3), round(g, 3)) for b, g in pl.pairs}
    assert pairs == {(2.0, 0.8), (0.5, 0.4), (0.05, 0.05), (0.0, 0.0)}


def test_holo_grid_shape():
    pl = _FACTORIES["holo_grid"](OmegaConf.create({}))
    betas = sorted({round(b, 3) for b, _ in pl.pairs})
    gammas = sorted({round(g, 3) for _, g in pl.pairs})
    assert betas == [0.0, 0.05, 0.2, 0.5, 1.0, 2.0]
    assert gammas == [0.0, 0.05, 0.2, 0.4, 0.6, 0.8]
    assert len(pl.pairs) == 36
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd case/phase && PYTHONPATH=. python -m pytest tests/test_holo_plans.py -q`
Expected: FAIL — `KeyError: 'holo_anchors'`.

- [ ] **Step 3: Edit `data_generator.py`**

Add two factory functions near the other plan builders (before the `_FACTORIES` dict), reusing the existing `SweepPlan` dataclass:

```python
def holo_anchors(_cfg=None) -> "SweepPlan":
    """The four holo.pdf anchors for the holographic length-gen study."""
    pairs = ((2.0, 0.8), (0.5, 0.4), (0.05, 0.05), (0.0, 0.0))
    return SweepPlan(name="holo_anchors",
                     description="Natural / CoT / Edge-of-chaos / Abyss anchors",
                     pairs=pairs)


def holo_grid(_cfg=None) -> "SweepPlan":
    """6x6 (β,γ) grid for the holographic length-gen phase map."""
    betas = [0.0, 0.05, 0.2, 0.5, 1.0, 2.0]
    gammas = [0.0, 0.05, 0.2, 0.4, 0.6, 0.8]
    pairs = tuple((b, g) for b in betas for g in gammas)
    return SweepPlan(name="holo_grid",
                     description="6×6 β×γ grid (holographic length-gen)",
                     pairs=pairs)
```

Then register both in the `_FACTORIES` dict (factories take the plan cfg node; these ignore it):

```python
    "holo_anchors": lambda c: holo_anchors(c),
    "holo_grid": lambda c: holo_grid(c),
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd case/phase && PYTHONPATH=. python -m pytest tests/test_holo_plans.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add case/phase/data_generator.py case/phase/tests/test_holo_plans.py
git commit -m "feat(holo): holo_anchors + holo_grid (β,γ) plan factories"
```

---

## Task 7: Config presets (grid tier, scale tier, Mamba control)

**Files:**
- Create: `case/phase/configs/models/holo_small.yaml`
- Create: `case/phase/configs/models/holo_100m.yaml`
- Create: `case/phase/configs/models/holo_mamba_small.yaml`

- [ ] **Step 1: Write `holo_small.yaml`** (RoPE, grid tier ~15M, lowered floor)

```yaml
# Grid-tier RoPE preset for the holographic length-gen sweep.
# ~15M non-embed params; min floor lowered so small models pass build_config.
architecture: transformer_rope
vocab_size: 40            # p=31 + 3 specials + 6 filler  (NestedMonoidGenerator)
d_model: 512
nhead: 8
ff_mult: 4
num_layers: 6
dropout: 0.1
min_non_embed_params: 1000000

# ---- task knobs ----
task: nested_monoid
monoid_p: 31
monoid_filler: 6
gen_mode: holographic
source_depth: 256
eval_answer_masked: true

# ---- training ----
train_len: 256
eval_lengths: [256, 512, 1024, 2048]
train_steps: 4000
train_batch_size: 64
eval_batch_size: 16
eval_batches: 16
lr: 3e-4
warmup_steps: 100
lr_min_ratio: 0.1
grad_clip: 1.0
deterministic: false

# ---- diagnostics ----
renyi_seq_len: 12000
renyi_qs: [0.5, 1.0, 2.0]
renyi_orders: [1, 2, 3]
```

- [ ] **Step 2: Write `holo_100m.yaml`** (RoPE, scale tier ≥100M)

```yaml
# Scale-tier RoPE preset (≥100M non-embed) for headline-cell confirmation.
# d_model=1024, num_layers=8, ff_mult=4 → same ~100.8M as the KV 100m preset.
architecture: transformer_rope
vocab_size: 40
d_model: 1024
nhead: 16
ff_mult: 4
num_layers: 8
dropout: 0.1
# no min_non_embed_params override → default 100M floor enforced

task: nested_monoid
monoid_p: 31
monoid_filler: 6
gen_mode: holographic
source_depth: 256
eval_answer_masked: true

train_len: 256
eval_lengths: [256, 512, 1024, 2048]
train_steps: 6000
train_batch_size: 32
eval_batch_size: 8
eval_batches: 16
lr: 3e-4
warmup_steps: 100
lr_min_ratio: 0.1
grad_clip: 1.0
deterministic: false

renyi_seq_len: 12000
renyi_qs: [0.5, 1.0, 2.0]
renyi_orders: [1, 2, 3]
```

- [ ] **Step 3: Write `holo_mamba_small.yaml`** (Mamba control, grid tier)

```yaml
# Grid-tier Mamba control for the H3 architecture comparison.
architecture: mamba
vocab_size: 40
d_model: 512
num_layers: 16
d_state: 16
d_conv: 3
nhead: 8            # kept for build_config back-compat (Mamba ignores)
ff_mult: 4
dropout: 0.0
min_non_embed_params: 1000000

task: nested_monoid
monoid_p: 31
monoid_filler: 6
gen_mode: holographic
source_depth: 256
eval_answer_masked: true

train_len: 256
eval_lengths: [256, 512, 1024, 2048]
train_steps: 4000
train_batch_size: 32     # Mamba scan is slower; smaller batch
eval_batch_size: 8
eval_batches: 16
lr: 3e-4
warmup_steps: 100
lr_min_ratio: 0.1
grad_clip: 1.0
deterministic: false

renyi_seq_len: 12000
renyi_qs: [0.5, 1.0, 2.0]
renyi_orders: [1, 2, 3]
```

- [ ] **Step 4: Verify configs load + a model builds at each preset (CPU, no training)**

Run:
```bash
cd /n/home07/hanlinzhang/projects/holographic-data
source /n/netscratch/kempner_sham_lab/Lab/hanlinzhang/envs/flow/bin/activate
PYTHONPATH=case/phase python -c "
from utils import load_config
from phase_sweep import build_config
for preset in ['holo_small','holo_100m','holo_mamba_small']:
    cfg = load_config([f'model_preset={preset}','task=nested_monoid'])
    c = build_config(cfg.model, None, task='nested_monoid')
    print(preset, 'arch=', c['architecture'], 'non_embed=', f\"{c['_non_embed_params']/1e6:.1f}M\")
"
```
Expected: prints three lines; `holo_100m` ≥ 100.0M, the two small presets ~10–20M, no exceptions.

- [ ] **Step 5: Commit**

```bash
git add case/phase/configs/models/holo_small.yaml case/phase/configs/models/holo_100m.yaml case/phase/configs/models/holo_mamba_small.yaml
git commit -m "feat(holo): config presets (grid/scale RoPE + Mamba control)"
```

---

## Task 8: End-to-end CPU smoke (anchors, 5 steps)

**Files:** none new (uses `phase_sweep.py`)

- [ ] **Step 1: Run a tiny smoke sweep on CPU**

Run:
```bash
cd /n/home07/hanlinzhang/projects/holographic-data
source /n/netscratch/kempner_sham_lab/Lab/hanlinzhang/envs/flow/bin/activate
PYTHONPATH=case/phase python case/phase/phase_sweep.py \
    plan.name=holo_anchors task=nested_monoid model_preset=holo_small \
    sweep.seeds=[1] sweep.device=cpu sweep.no_tqdm=true sweep.no_plot=true \
    model.train_steps=5 model.train_len=96 model.eval_lengths=[96,192] \
    model.train_batch_size=4 model.eval_batch_size=2 model.eval_batches=2 \
    model.renyi_seq_len=2000 model.min_non_embed_params=10000 \
    sweep.out_dir=case/phase/runs/holo_smoke
```
Expected: completes in <2 min; prints `[done] 4 runs`; writes `case/phase/runs/holo_smoke/run_summary.csv`.

- [ ] **Step 2: Verify the CSV has answer-retention columns**

Run:
```bash
PYTHONPATH=case/phase python -c "
import csv
rows=list(csv.DictReader(open('case/phase/runs/holo_smoke/run_summary.csv')))
print('rows', len(rows))
print('cols', 'retention_ratio' in rows[0], 'long_acc' in rows[0])
for r in rows: print(r['beta'], r['gamma'], 'train_acc=%.3f'%float(r['train_acc']), 'ret=%.3f'%float(r['retention_ratio']))
"
```
Expected: 4 rows; `retention_ratio` and `long_acc` present; values are finite floats (learning quality irrelevant at 5 steps — this is a plumbing check). `train_acc` here is the **answer-masked** accuracy because `eval_answer_masked: true`.

- [ ] **Step 3: Clean up the smoke output (it is throwaway)**

Run: `rm -rf case/phase/runs/holo_smoke`

- [ ] **Step 4: Commit** (no code changed; record the verified smoke command in the runbook — see Task 10). Skip commit if nothing changed.

---

## Task 9: Knob-verification module + §3.4 gate

**Files:**
- Create: `case/phase/knob_verify.py`
- Test: `case/phase/tests/test_knob_verify.py`

- [ ] **Step 1: Write the failing test**

```python
# case/phase/tests/test_knob_verify.py
import numpy as np
from knob_verify import estimate_beta, estimate_gamma


def test_iid_has_fast_decay_low_correlation():
    rng = np.random.default_rng(0)
    toks = rng.integers(0, 8, size=40000)
    beta = estimate_beta(toks, vocab_size=8, lags=[1, 2, 4, 8, 16, 32])
    # iid => correlation amplitude tiny at all lags => steep (large) beta or ~0 corr
    assert beta["corr_at_lag1"] < 0.05


def test_long_range_block_repeat_has_slower_decay_than_iid():
    rng = np.random.default_rng(1)
    iid = rng.integers(0, 8, size=40000)
    block = rng.integers(0, 8, size=200)
    rep = np.tile(block, 200)                 # strong long-range periodic structure
    b_iid = estimate_beta(iid, vocab_size=8, lags=[1, 2, 4, 8, 16, 32, 64])
    b_rep = estimate_beta(rep, vocab_size=8, lags=[1, 2, 4, 8, 16, 32, 64])
    assert b_rep["corr_at_lag32"] > b_iid["corr_at_lag32"]


def test_gamma_estimator_monotone_entropy():
    rng = np.random.default_rng(2)
    toks = rng.integers(0, 8, size=40000)
    g = estimate_gamma(toks, vocab_size=8, max_order=4)
    # conditional entropies are non-increasing in context order (within noise)
    hs = g["cond_entropy_by_order"]
    assert hs[0] >= hs[-1] - 0.05
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd case/phase && PYTHONPATH=. python -m pytest tests/test_knob_verify.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'knob_verify'`.

- [ ] **Step 3: Write `knob_verify.py`**

```python
# case/phase/knob_verify.py
#!/usr/bin/env python3
"""Realized (β̂, γ̂) estimators for the holographic length-gen study (spec §3.4).

β̂  — pairwise-correlation decay exponent: build the two-point covariance
      C(n)[u,v] = P(X_t=u, X_{t+n}=v) - P(u)P(v); correlation strength = its top
      singular value ||C(n)||_op; fit ||C(n)|| ~ n^{-β} (gamma-beta.pdf Eq. 7).
γ̂  — conditional-entropy decay exponent: entropy-rate-difference estimate of the
      next-token conditional entropy H_n, fit H_n - H_inf ~ n^{-γ} (Eq. 6).

CLI runs the gate: generate a long stream per anchor, report β̂/γ̂, and check the
knob moves β̂ monotonically across the anchors.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

EPS = 1e-12


def _counts_entropy(counts: np.ndarray) -> float:
    p = counts.astype(np.float64)
    p = p / max(p.sum(), EPS)
    p = np.clip(p, EPS, 1.0)
    return float(-np.sum(p * np.log(p)))


def _mgram_entropy(tokens: np.ndarray, order: int, vocab_size: int) -> float:
    n = len(tokens) - order + 1
    if n <= 0:
        return 0.0
    ids = np.zeros(n, dtype=np.int64)
    for k in range(order):
        ids = ids * vocab_size + tokens[k:k + n]
    _, counts = np.unique(ids, return_counts=True)
    return _counts_entropy(counts)


def estimate_gamma(tokens: np.ndarray, vocab_size: int, max_order: int = 6) -> Dict:
    """H_n via entropy-rate differences H(m+1-gram) - H(m-gram); fit n^{-γ}."""
    tokens = np.asarray(tokens, dtype=np.int64)
    hs: List[float] = []
    for m in range(1, max_order + 1):
        hs.append(_mgram_entropy(tokens, m + 1, vocab_size)
                  - _mgram_entropy(tokens, m, vocab_size))
    hs_arr = np.array(hs, dtype=np.float64)
    h_inf = float(hs_arr.min())
    excess = np.clip(hs_arr - h_inf, EPS, None)
    n = np.arange(1, len(hs) + 1, dtype=np.float64)
    if len(hs) >= 2:
        slope, _ = np.polyfit(np.log(n), np.log(excess), 1)
        gamma_hat = float(-slope)
    else:
        gamma_hat = 0.0
    return {"gamma_hat": gamma_hat, "cond_entropy_by_order": hs,
            "h_inf": h_inf}


def estimate_beta(tokens: np.ndarray, vocab_size: int,
                  lags: Sequence[int] = (1, 2, 4, 8, 16, 32, 64)) -> Dict:
    """||C(n)||_op vs lag n; fit n^{-β}. Returns β̂ and per-lag corr strengths."""
    tokens = np.asarray(tokens, dtype=np.int64)
    V = int(vocab_size)
    p1 = np.bincount(tokens, minlength=V).astype(np.float64)
    p1 = p1 / max(p1.sum(), EPS)
    strengths: Dict[str, float] = {}
    vals: List[float] = []
    used: List[int] = []
    for n in lags:
        if len(tokens) - n < 100:
            continue
        a = tokens[:-n]
        b = tokens[n:]
        joint = np.zeros((V, V), dtype=np.float64)
        np.add.at(joint, (a, b), 1.0)
        joint /= max(joint.sum(), EPS)
        C = joint - np.outer(p1, p1)
        top = float(np.linalg.svd(C, compute_uv=False)[0])
        strengths[f"corr_at_lag{n}"] = top
        vals.append(top)
        used.append(n)
    if len(vals) >= 2:
        x = np.log(np.array(used, dtype=np.float64))
        yv = np.log(np.clip(np.array(vals), EPS, None))
        slope, _ = np.polyfit(x, yv, 1)
        beta_hat = float(-slope)
    else:
        beta_hat = 0.0
    out = {"beta_hat": beta_hat}
    out.update(strengths)
    return out


def _run_gate() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data_nested_monoid import NestedMonoidGenerator

    anchors = {"natural": (2.0, 0.8), "cot": (0.5, 0.4),
               "edge": (0.05, 0.05), "abyss": (0.0, 0.0)}
    g = NestedMonoidGenerator(p=31, n_filler=6)
    print(f"{'anchor':10s} {'nom_b':>6s} {'nom_g':>6s} {'beta_hat':>9s} {'gamma_hat':>9s}")
    betas = {}
    for name, (b, gm) in anchors.items():
        np.random.seed(100000)
        x, _, _ = g.generate_batch(1, 16000, beta=b, gamma=gm)
        toks = x[0].numpy()
        bh = estimate_beta(toks, vocab_size=g.vocab_size)["beta_hat"]
        gh = estimate_gamma(toks, vocab_size=g.vocab_size)["gamma_hat"]
        betas[name] = bh
        print(f"{name:10s} {b:6.2f} {gm:6.2f} {bh:9.3f} {gh:9.3f}")
    ok = betas["natural"] > betas["edge"]   # larger nominal β => steeper decay
    print(f"\nGATE: realized β̂ tracks nominal β (natural > edge): {ok}")


if __name__ == "__main__":
    _run_gate()
```

- [ ] **Step 4: Run to verify the tests pass**

Run: `cd case/phase && PYTHONPATH=. python -m pytest tests/test_knob_verify.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Run the §3.4 gate and record results**

Run:
```bash
cd /n/home07/hanlinzhang/projects/holographic-data
source /n/netscratch/kempner_sham_lab/Lab/hanlinzhang/envs/flow/bin/activate
PYTHONPATH=case/phase python case/phase/knob_verify.py | tee case/phase/results/holo_knob_verification.txt
```
Expected: a 4-row table; the GATE line prints `True` (realized β̂ for `natural` > `edge`). **If the gate prints False**, the β knob is not separating the anchors — STOP and revisit the reach-sampling in `data_nested_monoid._assemble` before spending GPU. Save the table verbatim into `case/phase/results/holo_knob_verification.md` with a one-paragraph interpretation (Chinese + emoji per CLAUDE.md).

- [ ] **Step 6: Commit**

```bash
git add case/phase/knob_verify.py case/phase/tests/test_knob_verify.py case/phase/results/holo_knob_verification.*
git commit -m "feat(holo): (β̂,γ̂) knob-verification module + §3.4 gate results"
```

---

## Task 10: Runbook + module README

**Files:**
- Create: `case/phase/README_holo.md`

- [ ] **Step 1: Write the runbook** documenting the task and the exact Phase A–D submit/analyze commands.

```markdown
# Holographic length-gen (nested_monoid task) — runbook

Task `nested_monoid`: one deep affine-mod-p composition per example, emitted as
an AR reduction trace. (β,γ) knobs per spec; eval is answer-masked; retention =
acc(long)/acc(short). See docs/superpowers/specs/2026-06-13-holographic-length-gen-design.md.

## Local checks
- Unit tests:   `cd case/phase && PYTHONPATH=. python -m pytest tests -q`
- Knob gate:    `PYTHONPATH=case/phase python case/phase/knob_verify.py`

## Phase A — anchors (Transformer, grid tier)
sbatch --account=kempner_sham_lab --partition=kempner --gres=gpu:1 -t 1-00:00 \
  --export=ALL,RUN_NAME=holo_anchorsA,OVERRIDES="plan.name=holo_anchors task=nested_monoid model_preset=holo_small sweep.seeds=[1,2,3]" \
  case/phase/sweep.slurm
# output: case/phase/runs/holo_anchorsA/{run_summary.csv,raw_metrics.csv}

## Phase B — 6×6 grid (Transformer, grid tier)
sbatch --account=kempner_sham_lab --partition=kempner --gres=gpu:1 -t 2-00:00 \
  --export=ALL,RUN_NAME=holo_gridB,OVERRIDES="plan.name=holo_grid task=nested_monoid model_preset=holo_small sweep.seeds=[1,2,3]" \
  case/phase/sweep.slurm
# heatmap: PYTHONPATH=case/phase python case/phase/plot_phase_diagram.py plot.in_summary=case/phase/runs/holo_gridB/run_summary.csv plot.out_dir=case/phase/runs/holo_gridB

## Phase C(i) — holographic-short vs truncated-long (matched budget)
# holographic:
sbatch ... OVERRIDES="plan.name=holo_anchors task=nested_monoid model_preset=holo_small model.gen_mode=holographic sweep.seeds=[1,2,3]" RUN_NAME=holo_C_holo ...
# truncated (same train_len/steps = matched token budget; train on slices of depth-256 sequences):
sbatch ... OVERRIDES="plan.name=holo_anchors task=nested_monoid model_preset=holo_small model.gen_mode=truncated model.source_depth=256 sweep.seeds=[1,2,3]" RUN_NAME=holo_C_trunc ...
# Both eval on the SAME holographic ladder (eval always rebuilds via gen; for a
# clean test, evaluate the truncated-trained checkpoint on holographic data —
# set model.gen_mode=holographic at eval by running a holographic eval sweep that
# loads weights; for v1 compare retention of the two training regimes directly.)

## Phase C(ii) — Mamba column (anchors)
sbatch ... OVERRIDES="plan.name=holo_anchors task=nested_monoid model_preset=holo_mamba_small sweep.seeds=[1,2,3]" RUN_NAME=holo_C_mamba ...

## Phase D — scale-up (≥100M Transformer, headline cells)
sbatch ... OVERRIDES="plan.name=holo_anchors task=nested_monoid model_preset=holo_100m sweep.seeds=[1,2,3]" RUN_NAME=holo_D_100m -p kempner_h100 ...

## Account/partition rotation
Use ~/acct.sh to route to the highest-fairshare account before each batch; prefer
-p kempner (A100) for grid tier, -p kempner_h100 for the 100M tier; large files
to $SCRATCH, never $HOME.
```

- [ ] **Step 2: Run the full unit-test suite once more (regression gate)**

Run: `cd case/phase && PYTHONPATH=. python -m pytest tests -q`
Expected: all tests pass (model_rope 4, nested_monoid 8, phase_core_masked 2, build_config_floor 2, holo_plans 2, knob_verify 3 = 23 passed).

- [ ] **Step 3: Commit**

```bash
git add case/phase/README_holo.md
git commit -m "docs(holo): runbook + Phase A–D submit/analyze commands"
```

---

## Task 11: Phase A — anchors sweep (Transformer, grid tier)

**Files:** outputs only — `case/phase/runs/holo_anchorsA/`

- [ ] **Step 1: Route account, submit Phase A**

```bash
cd /n/home07/hanlinzhang/projects/holographic-data
~/acct.sh show
sbatch --account=kempner_sham_lab --partition=kempner --gres=gpu:1 -t 1-00:00 \
  --export=ALL,RUN_NAME=holo_anchorsA,OVERRIDES="plan.name=holo_anchors task=nested_monoid model_preset=holo_small sweep.seeds=[1,2,3]" \
  case/phase/sweep.slurm
```
Expected: a job id; 12 runs (4 anchors × 3 seeds).

- [ ] **Step 2: Wait for completion, then inspect retention ordering**

Run:
```bash
PYTHONPATH=case/phase python -c "
import csv, collections
rows=list(csv.DictReader(open('case/phase/runs/holo_anchorsA/run_summary.csv')))
agg=collections.defaultdict(list)
for r in rows: agg[(r['beta'],r['gamma'])].append(float(r['retention_ratio']))
import statistics as st
for k in sorted(agg): print(k, 'ret=%.3f ± %.3f'%(st.mean(agg[k]), st.pstdev(agg[k])))
"
```
Expected: per-anchor mean retention. **Success check (H1 directional):** Edge/holographic anchors (small β) show higher retention than Natural (β=2.0); Abyss (0,0) is the lowest (failure boundary).

- [ ] **Step 3: Record findings** in `case/phase/results/holo_phaseA_anchors.md` (Chinese + emoji; mean ± std retention table; the H1 directional verdict). Commit the results file:

```bash
git add case/phase/results/holo_phaseA_anchors.md
git commit -m "results(holo): Phase A anchor retention ordering"
```

---

## Task 12: Phase B — 6×6 grid + heatmap

**Files:** outputs only — `case/phase/runs/holo_gridB/`

- [ ] **Step 1: Submit the grid** (gate on Phase A confirming the directional ordering — do not run B if A is flat)

```bash
sbatch --account=kempner_sham_lab --partition=kempner --gres=gpu:1 -t 2-00:00 \
  --export=ALL,RUN_NAME=holo_gridB,OVERRIDES="plan.name=holo_grid task=nested_monoid model_preset=holo_small sweep.seeds=[1,2,3]" \
  case/phase/sweep.slurm
```
Expected: 108 runs (36 cells × 3 seeds).

- [ ] **Step 2: Render the retention heatmap**

```bash
PYTHONPATH=case/phase python case/phase/plot_phase_diagram.py \
  plot.in_summary=case/phase/runs/holo_gridB/run_summary.csv \
  plot.out_dir=case/phase/runs/holo_gridB
```
Expected: phase-diagram PNG(s) in `runs/holo_gridB/`. **Success check (H1):** a retention ridge at small β + low–moderate γ; low retention at Natural and Abyss corners.

- [ ] **Step 3: Record findings** in `case/phase/results/holo_phaseB_grid.md` (embed the heatmap path; state whether H1's ridge is present and where). Commit:

```bash
git add case/phase/results/holo_phaseB_grid.md case/phase/runs/holo_gridB/*.png
git commit -m "results(holo): Phase B 6×6 retention heatmap + ridge analysis"
```

---

## Task 13: Phase C — controls (holographic-vs-truncated, Mamba)

**Files:** outputs only — `case/phase/runs/holo_C_*`

- [ ] **Step 1: Submit the H2 control (matched-budget data ablation)**

```bash
# holographic training data
sbatch --account=kempner_sham_lab --partition=kempner --gres=gpu:1 -t 1-00:00 \
  --export=ALL,RUN_NAME=holo_C_holo,OVERRIDES="plan.name=holo_anchors task=nested_monoid model_preset=holo_small model.gen_mode=holographic sweep.seeds=[1,2,3]" \
  case/phase/sweep.slurm
# truncated-long training data (matched train_len/steps)
sbatch --account=kempner_sham_lab --partition=kempner --gres=gpu:1 -t 1-00:00 \
  --export=ALL,RUN_NAME=holo_C_trunc,OVERRIDES="plan.name=holo_anchors task=nested_monoid model_preset=holo_small model.gen_mode=truncated model.source_depth=256 sweep.seeds=[1,2,3]" \
  case/phase/sweep.slurm
```
Expected: two run dirs. **Success check (H2):** holographic-trained retention > truncated-trained retention at matched budget.

- [ ] **Step 2: Submit the H3 control (Mamba column)**

```bash
sbatch --account=kempner_sham_lab --partition=kempner --gres=gpu:1 -t 1-00:00 \
  --export=ALL,RUN_NAME=holo_C_mamba,OVERRIDES="plan.name=holo_anchors task=nested_monoid model_preset=holo_mamba_small sweep.seeds=[1,2,3]" \
  case/phase/sweep.slurm
```
Expected: one run dir. **Success check (H3):** compare RoPE-Transformer retention (Phase A) vs Mamba retention per anchor; does the gap shrink inside the band?

- [ ] **Step 3: Record findings** in `case/phase/results/holo_phaseC_controls.md` (H2 table: holo vs trunc; H3 table: Transformer vs Mamba per anchor). Commit:

```bash
git add case/phase/results/holo_phaseC_controls.md
git commit -m "results(holo): Phase C controls (H2 data ablation, H3 Mamba)"
```

---

## Task 14: Phase D — scale-up (≥100M) + final report

**Files:** outputs only — `case/phase/runs/holo_D_100m/`; report `case/phase/reports/holo_length_gen.md`

- [ ] **Step 1: Submit the ≥100M headline-cell confirmation** (route to H100)

```bash
~/acct.sh show
sbatch --account=kempner_sham_lab --partition=kempner_h100 --gres=gpu:1 -t 2-00:00 \
  --export=ALL,RUN_NAME=holo_D_100m,OVERRIDES="plan.name=holo_anchors task=nested_monoid model_preset=holo_100m sweep.seeds=[1,2,3]" \
  case/phase/sweep.slurm
```
Expected: 12 runs at ≥100M. **Success check:** the band/ridge ordering from grid tier persists at scale.

- [ ] **Step 2: Write the consolidated report** `case/phase/reports/holo_length_gen.md` covering: the knob-verification gate (Task 9), Phase A ordering, Phase B heatmap/ridge, Phase C H2+H3 verdicts, Phase D scale persistence — each claim backed by the specific CSV/figure, with the H1/H2/H3 ✅/❌ verdicts from the spec §8. Chinese + emoji for takeaways.

- [ ] **Step 3: Final regression + commit**

```bash
cd /n/home07/hanlinzhang/projects/holographic-data
PYTHONPATH=case/phase python -m pytest case/phase/tests -q
git add case/phase/results/holo_phaseD_scale.md case/phase/reports/holo_length_gen.md case/phase/runs/holo_D_100m/*.csv
git commit -m "results(holo): Phase D scale-up + consolidated length-gen report"
```

---

## Self-review (completed during planning)

**Spec coverage:**
- §3.1 affine-mod-p operator → Task 2. §3.2 4-block layout → Task 2. §3.3 β/γ knobs → Tasks 2–3. §3.4 knob verification → Task 9. §3.5 length protocol (train_len/eval_lengths) → Tasks 7, 11. §4 RoPE + Mamba + two-tier scale → Tasks 1, 7, 12, 14. §5 metrics (answer-masked retention) → Task 4. §6 matrix A–D → Tasks 11–14. §7 codebase reuse → Tasks 4–6 (edits, not forks). §8 success criteria → embedded as "Success check" in Tasks 11–14. §9 risks (knob gate, untrainable Edge) → Task 9 gate + Abyss as control.
- Band-certification overlay (§5 light) → epiplexity area-under-loss is **already** computed by `phase_core` (`aulc_train_to_final`) and lands in `run_summary.csv`; no extra task needed — the report (Task 14) correlates the ridge with `aulc`. (Noted so it is not a gap.)

**Placeholder scan:** none — all code blocks are complete; the one sketch (`_fit`) is explicitly flagged for deletion in Task 2 Step 4.

**Type/name consistency:** `make_generator` config keys (`monoid_p`, `monoid_filler`, `gen_mode`, `source_depth`, `eval_answer_masked`) are identical across phase_core (Task 4), presets (Task 7), and runbook (Task 10). `evaluate_at_length(..., answer_masked=...)` matches its caller in `train_one_model`. `count_non_embedding_params` reused for `transformer_rope` in both phase_core (model build) and phase_sweep (floor). Plan factory keys `holo_anchors`/`holo_grid` match across data_generator, tests, and runbook.

---

## BitLesson Selection (REQUIRED FOR EACH TASK)

Before executing each task or sub-task, you MUST:

1. Read @/n/home07/hanlinzhang/projects/holographic-data/.humanize/bitlesson.md
2. Run `bitlesson-selector` for each task/sub-task to select relevant lesson IDs
3. Follow the selected lesson IDs (or `NONE`) during implementation

Include a `## BitLesson Delta` section in your summary with:
- Action: none|add|update
- Lesson ID(s): NONE or comma-separated IDs
- Notes: what changed and why (required if action is add or update)

Reference: @/n/home07/hanlinzhang/projects/holographic-data/.humanize/bitlesson.md

---

## Goal Tracker Rules

Throughout your work, you MUST maintain the Goal Tracker:

1. **Before starting a round**: Re-anchor on the original plan and current round contract
2. **Before starting a task**: Mark the relevant mainline task as "in_progress" in Active Tasks
   - Confirm Tag/Owner routing is correct before execution
3. **Active Tasks** are MAINLINE tasks only - side issues do not belong there
4. **Blocking Side Issues** are reserved for issues that truly stop mainline progress
5. **Queued Side Issues** are non-blocking and must not take over the round
6. **After completing a mainline task**: Move it to "Completed and Verified" with evidence (but mark as "pending verification")
7. **If you discover the plan has errors**:
   - Do NOT silently change direction
   - Add entry to "Plan Evolution Log" with justification
   - Explain how the change still serves the Ultimate Goal
8. **If you need to defer a task**:
   - Move it to "Explicitly Deferred" section
   - Provide strong justification
   - Explain impact on Acceptance Criteria
9. **If you discover new issues**:
   - Add to "Blocking Side Issues" only if mainline progress is blocked
   - Otherwise add to "Queued Side Issues" or keep them as `[queued]` tasks/backlog

---

Note: You MUST NOT try to exit `start-rlcr-loop` loop by lying or edit loop state file or try to execute `cancel-rlcr-loop`

After completing the work, please:
0. If you have access to the `code-simplifier` agent, use it to review and optimize the code you just wrote
1. Finalize @/n/home07/hanlinzhang/projects/holographic-data/.humanize/rlcr/2026-06-13_19-19-45/goal-tracker.md (this is Round 0, so you are initializing it - see "Goal Tracker Setup" above)
2. Write your round contract into @/n/home07/hanlinzhang/projects/holographic-data/.humanize/rlcr/2026-06-13_19-19-45/round-0-contract.md
3. Commit your changes with a descriptive commit message
4. Write your work summary into @/n/home07/hanlinzhang/projects/holographic-data/.humanize/rlcr/2026-06-13_19-19-45/round-0-summary.md
