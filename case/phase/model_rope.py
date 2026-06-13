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
