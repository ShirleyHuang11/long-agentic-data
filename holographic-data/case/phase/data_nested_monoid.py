"""Nested-monoid (affine mod p) register-machine generator parametrized by (β, γ).

Third (β, γ) task for case/phase (sibling to AlgorithmicKVGenerator and
LogicalFoldingGenerator). Realizes the "holographic" construct from
assets/holo.pdf: a compositional fold over named affine ops, emitted as an
autoregressive recall-and-apply trace, so a vanilla decoder-only Transformer
trained with next-token prediction on SHORT sequences can be tested on LONGER
ones (more fold steps + longer recall distances).

Operator (monoid), selected by op_kind:
    'perm'  (default) — a small pool of n_perms FIXED permutations of Z_p
            (seeded). A DEF binds a name to a pool index; USE applies that
            permutation to the register: reg <- perm[idx][reg]. The per-step op
            is a learnable LOOKUP (not arithmetic), so a vanilla Transformer can
            fit the train-length task and the experiment isolates the actual
            research question (LENGTH GENERALIZATION via long-range retrieval of
            the name->idx binding) instead of grokking modular arithmetic.
    'affine' — affine maps over Z_p, v -> (a*v + b) mod p. Compact but the
            per-step op is modular arithmetic over in-context params, which is
            grokking-hard; kept as an option, not the default.
Both are closed under composition => the per-step fold rule is identical at
every depth (provably depth-invariant ground truth).

Design (well-posed + measurable long-range structure): an interleaved register
machine, like AlgorithmicKVGenerator (which re-emits the recalled key). At each
step we either DEFINE a new named op or USE (recall) a previously-defined op by
emitting its NAME, then apply it to the running register and emit the result:

    x0 DEF n0 a0 b0 DEF n1 a1 b1 USE n0 r0 [filler] USE n1 r1 DEF n2 ... ANS ans

The NAME token repeats (definition + use) at the recall distance, so (a) the
next-token target is well-posed (the name says which op to apply) and (b) the
long-range dependency is real and required (the op's (a,b) live at the
definition and are NOT restated at use — the model must retrieve them).

Knobs:
    β  — recall recency. USE picks the d-th most-recently defined op with
         d ~ Zipf(β): P(d) ∝ d^{-(1+β)} (β<=1e-3 => uniform). β→∞ local recalls
         (recent ops), β→0 long-range recalls (old ops, distance grows with
         sequence length). Same convention as the KV / logical-folding tasks.
    γ  — filler rate. Fraction of semantically inert filler tokens.

Vocab (p values, n_names names, 3 specials, n_filler filler;
       vocab_size = p + n_names + 3 + n_filler):
    0..p-1                          value tokens (Z_p; also a,b params, x0,
                                    results, answer)
    p..p+n_names-1                  name tokens (op identifiers)
    p+n_names                       DEF
    p+n_names+1                     USE
    p+n_names+2                     ANS
    p+n_names+3..                   filler tokens

Modes:
    'holographic' : a self-contained program built to fit seq_len.
    'truncated'   : build a program of token length source_len (> seq_len) and
                    keep only the last seq_len tokens (a "slice" of long data
                    with dangling recalls) — the H2 matched-budget control.

generate_batch(batch_size, seq_len, beta, gamma) -> (x, y, m) matches the
sibling generators. y = target value at masked positions; m marks USE results +
the final answer. return_meta=True (tests only) also returns ground-truth dicts
with keys: x0, applied (list of (a,b) in use order), recencies, use_lags (token
distance from each recall back to its definition), n_uses, answer.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch


class NestedMonoidGenerator:
    _PERM_SEED = 20260613

    def __init__(self, p: int = 31, n_names: int = 512, n_filler: int = 6,
                 mode: str = "holographic", source_len: int = 1024,
                 p_def: float = 0.5, op_kind: str = "perm",
                 n_perms: int = 16) -> None:
        if p < 2:
            raise ValueError(f"p must be >= 2, got {p}")
        if n_names < 2:
            raise ValueError(f"n_names must be >= 2, got {n_names}")
        if n_filler < 1:
            raise ValueError(f"n_filler must be >= 1, got {n_filler}")
        if mode not in ("holographic", "truncated"):
            raise ValueError(f"mode must be 'holographic'|'truncated', got {mode!r}")
        if op_kind not in ("perm", "affine"):
            raise ValueError(f"op_kind must be 'perm'|'affine', got {op_kind!r}")
        if op_kind == "perm" and not (2 <= n_perms <= p):
            raise ValueError(f"n_perms must be in [2, p]={p}, got {n_perms}")
        self.p = int(p)
        self.n_names = int(n_names)
        self.n_filler = int(n_filler)
        self.mode = str(mode)
        self.source_len = int(source_len)
        self.p_def = float(p_def)
        self.op_kind = str(op_kind)
        self.n_perms = int(n_perms)
        self.def_len = 3 if op_kind == "perm" else 4  # DEF name idx | DEF name a b
        # Fixed permutation pool, shared across all examples (seeded).
        rng = np.random.default_rng(self._PERM_SEED)
        self.perms = np.stack([rng.permutation(self.p) for _ in range(self.n_perms)])
        self.name_base = self.p
        self.DEF = self.p + self.n_names
        self.USE = self.p + self.n_names + 1
        self.ANS = self.p + self.n_names + 2
        self.filler_tokens = np.arange(self.p + self.n_names + 3,
                                       self.p + self.n_names + 3 + self.n_filler)
        self.vocab_size = self.p + self.n_names + 3 + self.n_filler

    def _build(self, target_len: int, beta: float, gamma: float
               ) -> Tuple[List[int], List[float], List[int], Dict]:
        p = self.p
        reg = int(np.random.randint(0, p))
        x0 = reg
        toks: List[int] = [reg]
        m: List[float] = [0.0]
        y: List[int] = [0]
        mem: List[Tuple] = []                   # (name_token, payload) in def order
        def_pos: Dict[int, int] = {}            # name_token -> token index at DEF
        names_used = 0
        applied: List = []                      # payloads in recall order
        recencies: List[int] = []
        use_lags: List[int] = []
        limit = target_len - 2                  # reserve ANS + answer

        while len(toks) < limit:
            room = limit - len(toks)
            if len(toks) > 1 and room >= 1 and np.random.rand() < gamma:
                toks.append(int(np.random.choice(self.filler_tokens)))
                m.append(0.0)
                y.append(0)
                continue
            can_def = names_used < self.n_names and room >= self.def_len
            can_use = len(mem) > 0 and room >= 3
            if can_def and (not can_use or np.random.rand() < self.p_def):
                name = self.name_base + names_used
                names_used += 1
                toks.append(self.DEF); m.append(0.0); y.append(0)
                name_idx = len(toks)
                toks.append(name); m.append(0.0); y.append(0)
                if self.op_kind == "perm":
                    payload = int(np.random.randint(0, self.n_perms))
                    toks.append(payload); m.append(0.0); y.append(0)  # idx as value token
                else:
                    payload = (int(np.random.randint(1, p)), int(np.random.randint(0, p)))
                    toks.append(payload[0]); m.append(0.0); y.append(0)
                    toks.append(payload[1]); m.append(0.0); y.append(0)
                def_pos[name] = name_idx
                mem.append((name, payload))
            elif can_use:
                k = len(mem)
                if beta <= 1e-3:
                    d = int(np.random.randint(1, k + 1))
                else:
                    idx = np.arange(1, k + 1)
                    probs = 1.0 / np.power(idx, beta + 1.0)
                    probs = probs / probs.sum()
                    d = int(np.random.choice(idx, p=probs))
                name, payload = mem[k - d]
                if self.op_kind == "perm":
                    reg = int(self.perms[payload, reg])
                else:
                    a, b = payload
                    reg = int((a * reg + b) % p)
                toks.append(self.USE); m.append(0.0); y.append(0)
                use_idx = len(toks)
                toks.append(name); m.append(0.0); y.append(0)
                toks.append(reg); m.append(1.0); y.append(reg)
                applied.append(payload)
                recencies.append(d)
                use_lags.append(use_idx - def_pos[name])
            else:
                break

        toks.append(self.ANS); m.append(0.0); y.append(0)
        toks.append(reg); m.append(1.0); y.append(reg)
        meta = {"x0": x0, "applied": applied, "recencies": recencies,
                "use_lags": use_lags, "n_uses": len(applied), "answer": reg,
                "op_kind": self.op_kind}
        return toks, m, y, meta

    def _fit_to_len(self, toks, m, y, seq_len, window_tail):
        arr = np.array(toks, dtype=np.int64)
        ma = np.array(m, dtype=np.float32)
        ya = np.array(y, dtype=np.int64)
        if window_tail and len(arr) > seq_len:
            arr, ma, ya = arr[-seq_len:], ma[-seq_len:], ya[-seq_len:]
        elif len(arr) > seq_len:                # holographic never overshoots, but guard
            arr, ma, ya = arr[-seq_len:], ma[-seq_len:], ya[-seq_len:]
        if len(arr) < seq_len:
            pad = seq_len - len(arr)
            fill = np.random.choice(self.filler_tokens, size=pad)
            arr = np.concatenate([fill, arr])
            ma = np.concatenate([np.zeros(pad, dtype=np.float32), ma])
            ya = np.concatenate([np.zeros(pad, dtype=np.int64), ya])
        return arr, ma, ya

    def _build_one(self, seq_len: int, beta: float, gamma: float):
        if self.mode == "truncated":
            toks, m, y, meta = self._build(self.source_len, beta, gamma)
            arr, ma, ya = self._fit_to_len(toks, m, y, seq_len, window_tail=True)
        else:
            toks, m, y, meta = self._build(seq_len, beta, gamma)
            arr, ma, ya = self._fit_to_len(toks, m, y, seq_len, window_tail=False)
        return arr, ya, ma, meta

    def generate_batch(self, batch_size: int, seq_len: int, beta: float,
                       gamma: float, return_meta: bool = False):
        xb = np.zeros((batch_size, seq_len), dtype=np.int64)
        yb = np.zeros((batch_size, seq_len), dtype=np.int64)
        mb = np.zeros((batch_size, seq_len), dtype=np.float32)
        metas: List[Dict] = []
        for bi in range(batch_size):
            arr, ya, ma, meta = self._build_one(seq_len, beta, gamma)
            xb[bi], yb[bi], mb[bi] = arr, ya, ma
            if return_meta:
                metas.append(meta)
        x, y_t, m_t = torch.tensor(xb), torch.tensor(yb), torch.tensor(mb)
        if return_meta:
            return x, y_t, m_t, metas
        return x, y_t, m_t
