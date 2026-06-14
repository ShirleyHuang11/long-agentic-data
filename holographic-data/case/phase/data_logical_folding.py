"""Logical-folding token-stream generator parametrized by (β, γ).

Sibling task to ``AlgorithmicKVGenerator`` for cross-task universality tests of
the (β, γ) phase boundary.

Vocab layout (vocab_size=60, identical to the KV task so model size stays
fixed):
    0..9   — noise tokens
    10..29 — 20 unary ops; each is a fixed permutation of the value set
    30..59 — 30 value tokens (state space)

A sequence is built by maintaining a "fold stack" of past states. At each
step:
  • with prob γ — emit a noise token (state stack unchanged)
  • else with prob 0.5 (or empty stack) — WRITE: emit (op, base_value) pair;
    push the composed state op(base_value) onto the stack
  • else — FOLD: pick depth d ~ Zipf(β) over the stack (same convention as
    the KV generator), apply a fresh op to stack[-d], emit (op, result), push

β controls the heavy-tail of how far back the model must reach to predict
the next value; γ controls the noise dilution. The interface
``generate_batch(batch_size, seq_len, beta, gamma)`` matches
``AlgorithmicKVGenerator`` so the trainer is task-agnostic.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


_OP_PERM_SEED = 12345


class LogicalFoldingGenerator:
    """Compositional state-machine generator with (β, γ) knobs."""

    def __init__(self, vocab_size: int = 60) -> None:
        if vocab_size != 60:
            raise ValueError(
                f"LogicalFoldingGenerator requires vocab_size=60, got {vocab_size}"
            )
        self.vocab_size = vocab_size
        self.noise_tokens = np.arange(0, 10)
        self.op_tokens = np.arange(10, 30)
        self.value_tokens = np.arange(30, 60)
        n_vals = len(self.value_tokens)
        # Fixed permutations shared across all (batch, seed) so the task is
        # well-defined: every op is a deterministic function on values.
        rng = np.random.default_rng(_OP_PERM_SEED)
        self.op_perms = np.stack(
            [rng.permutation(n_vals) for _ in self.op_tokens], axis=0
        )

    def generate_batch(
        self, batch_size: int, seq_len: int, beta: float, gamma: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_batch = np.zeros((batch_size, seq_len), dtype=np.int64)
        y_batch = np.zeros((batch_size, seq_len), dtype=np.int64)
        m_batch = np.zeros((batch_size, seq_len), dtype=np.float32)

        n_ops = len(self.op_tokens)
        n_vals = len(self.value_tokens)

        for b in range(batch_size):
            stack: list[int] = []  # value indices ∈ [0, n_vals)
            t = 0
            while t < seq_len:
                if np.random.rand() < gamma:
                    x_batch[b, t] = int(np.random.choice(self.noise_tokens))
                    t += 1
                    continue

                if len(stack) == 0 or np.random.rand() < 0.5:
                    if t + 1 >= seq_len:
                        t += 1
                        continue
                    op_idx = int(np.random.randint(n_ops))
                    base_val_idx = int(np.random.randint(n_vals))
                    new_val_idx = int(self.op_perms[op_idx, base_val_idx])
                    x_batch[b, t] = int(self.op_tokens[op_idx])
                    x_batch[b, t + 1] = int(self.value_tokens[new_val_idx])
                    y_batch[b, t + 1] = int(self.value_tokens[new_val_idx])
                    m_batch[b, t + 1] = 1.0
                    stack.append(new_val_idx)
                    t += 2
                else:
                    if beta <= 1e-3:
                        d = int(np.random.randint(1, len(stack) + 1))
                    else:
                        idx = np.arange(1, len(stack) + 1)
                        probs = 1.0 / np.power(idx, beta + 1.0)
                        probs = probs / probs.sum()
                        d = int(np.random.choice(idx, p=probs))

                    if t + 1 >= seq_len:
                        t += 1
                        continue
                    base_val_idx = stack[len(stack) - d]
                    op_idx = int(np.random.randint(n_ops))
                    new_val_idx = int(self.op_perms[op_idx, base_val_idx])
                    x_batch[b, t] = int(self.op_tokens[op_idx])
                    x_batch[b, t + 1] = int(self.value_tokens[new_val_idx])
                    y_batch[b, t + 1] = int(self.value_tokens[new_val_idx])
                    m_batch[b, t + 1] = 1.0
                    stack.append(new_val_idx)
                    t += 2

        return torch.tensor(x_batch), torch.tensor(y_batch), torch.tensor(m_batch)
