#!/usr/bin/env python3
"""TinyCausalTransformer model + parameter accounting.

Decoupled from phase_core / data generation so that the model can be
re-used (or swapped for an alternative) without pulling in the training
loop, Rényi diagnostics, or the (β, γ) data generator.

Public surface:
    MIN_NON_EMBED_PARAMS        — hard floor enforced by phase_sweep.build_config
    count_non_embedding_params  — closed-form param count, used to validate configs
    TinyCausalTransformer       — the nn.Module
"""

from __future__ import annotations

import torch
import torch.nn as nn


MIN_NON_EMBED_PARAMS = 100_000_000


def count_non_embedding_params(
    d_model: int,
    nhead: int,
    ff_mult: int,
    num_layers: int,
    vocab_size: int,
) -> int:
    """Closed-form count of non-embedding params for TinyCausalTransformer.

    Excludes token embedding and positional embedding; includes transformer
    blocks (attn + FFN + LayerNorms) and the output projection.
    """
    d, m, L, V = int(d_model), int(ff_mult), int(num_layers), int(vocab_size)
    per_layer = (4 + 2 * m) * d * d + (9 + m) * d
    head = d * V + V
    return L * per_layer + head


class TinyCausalTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 60,
        d_model: int = 1024,
        nhead: int = 16,
        ff_mult: int = 4,
        num_layers: int = 12,
        max_ctx_tokens: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, max_ctx_tokens, d_model) * 0.01)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_mult * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
        self.register_buffer(
            "mask",
            torch.nn.Transformer.generate_square_subsequent_mask(max_ctx_tokens),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        emb = self.embedding(x) + self.pos_emb[:, :seq_len, :]
        causal_mask = self.mask[:seq_len, :seq_len]
        out = self.transformer(emb, mask=causal_mask, is_causal=True)
        return self.fc(out)
