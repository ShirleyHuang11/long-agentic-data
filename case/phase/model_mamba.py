#!/usr/bin/env python3
"""Minimal Mamba-style causal model for the (β, γ) phase experiments.

Selective state-space block in pure pytorch. Sequential scan loop —
slow but minimal (no CUDA kernels, no mamba_ssm dependency).

Same forward signature as `model.TinyCausalTransformer`:
    forward(x: LongTensor[B, T]) -> FloatTensor[B, T, vocab]

Drop-in via the `architecture: mamba` preset; the constructor accepts
the same kwargs as TinyCausalTransformer (extras are silently ignored).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

MIN_NON_EMBED_PARAMS = 100_000_000


def count_mamba_params(
    d_model: int,
    num_layers: int,
    vocab_size: int,
    d_state: int = 16,
    d_conv: int = 3,
) -> int:
    """Closed-form non-embedding param count for MambaCausal."""
    d, N, L, V = int(d_model), int(d_state), int(num_layers), int(vocab_size)
    # per block: in_proj(2d²) + out_proj(d²) + conv(d·d_conv + d) +
    #            x_proj(d·(2N+1)) + dt_proj((1+1)·d) + A_log(d·N) + D(d) + 2·LayerNorm(2d)
    per_block = 3 * d * d + d * d_conv + d + d * (2 * N + 1) + 2 * d + d * N + d + 4 * d
    head = d * V + V + 2 * d  # output proj + final LayerNorm
    return L * per_block + head


class MambaBlock(nn.Module):
    """One selective-SSM block. Sequential scan (loop over T). Slow but minimal."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 3) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.conv = nn.Conv1d(
            d_model, d_model, d_conv, padding=d_conv - 1, groups=d_model, bias=True
        )
        # x_proj produces dt_raw, B, C concatenated (input-dependent)
        self.x_proj = nn.Linear(d_model, 2 * d_state + 1, bias=False)
        self.dt_proj = nn.Linear(1, d_model)
        # A: HiPPO-style log-spaced negatives, learned in log
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1).float()).expand(d_model, -1).clone()
        )
        self.D = nn.Parameter(torch.ones(d_model))
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, D)
        B, T, D = x.shape
        N = self.d_state
        xz = self.in_proj(x)
        x_in, gate = xz.chunk(2, dim=-1)
        # depthwise causal conv: pad on left, truncate to T
        x_in = self.conv(x_in.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x_in = F.silu(x_in)
        # input-dependent dt, B, C
        dBC = self.x_proj(x_in)
        dt_raw = dBC[:, :, :1]
        Bp = dBC[:, :, 1 : 1 + N]
        Cp = dBC[:, :, 1 + N :]
        delta = F.softplus(self.dt_proj(dt_raw))  # (B, T, D)
        A = -torch.exp(self.A_log.float())  # (D, N), negative for stability
        # discretized recurrence: h_t = dA * h_{t-1} + dB * x_t
        h = torch.zeros(B, D, N, device=x.device, dtype=x.dtype)
        outs = []
        for t in range(T):
            dA = torch.exp(delta[:, t, :, None] * A[None, :, :])  # (B, D, N)
            dB = delta[:, t, :, None] * Bp[:, t, None, :]  # (B, D, N)
            h = dA * h + dB * x_in[:, t, :, None]
            y = (h * Cp[:, t, None, :]).sum(dim=-1) + self.D * x_in[:, t, :]
            outs.append(y)
        y = torch.stack(outs, dim=1)
        return self.out_proj(y * F.silu(gate))


class MambaCausal(nn.Module):
    """Causal Mamba-style LM. Same forward signature as TinyCausalTransformer."""

    def __init__(
        self,
        vocab_size: int = 60,
        d_model: int = 1024,
        num_layers: int = 32,
        d_state: int = 16,
        d_conv: int = 3,
        dropout: float = 0.0,
        **_kwargs,  # silently ignore Transformer-specific kwargs (nhead, ff_mult, max_ctx_tokens)
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "norm": nn.LayerNorm(d_model),
                        "ssm": MambaBlock(d_model, d_state, d_conv),
                    }
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_f = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embedding(x)
        for blk in self.blocks:
            h = h + blk["ssm"](blk["norm"](h))  # pre-norm residual
        return self.fc(self.norm_f(h))


if __name__ == "__main__":  # smoke check
    import sys

    n = count_mamba_params(d_model=1024, num_layers=32, vocab_size=60)
    print(f"non-embed params @ d=1024, L=32, vocab=60, N=16: {n:,} (~{n/1e6:.2f}M)")
    print(f"  meets 100M floor: {n >= MIN_NON_EMBED_PARAMS}")
    if "--smoke" in sys.argv:
        m = MambaCausal(vocab_size=60, d_model=128, num_layers=2, d_state=8)
        x = torch.randint(0, 60, (2, 32))
        y = m(x)
        print(f"smoke: input {x.shape} → output {y.shape} (expected (2, 32, 60))")
        assert y.shape == (2, 32, 60), "shape mismatch"
        print("smoke OK")
