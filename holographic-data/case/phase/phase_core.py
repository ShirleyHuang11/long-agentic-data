#!/usr/bin/env python3
"""Core components for beta-gamma algorithmic phase sweeps.

Designed for reproducible large sweeps used in paper-grade experiments.
"""

import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import (  # noqa: F401 — re-exported for callers
    MIN_NON_EMBED_PARAMS,
    TinyCausalTransformer,
    count_non_embedding_params,
)
from utils import set_all_seeds  # noqa: F401 — re-exported for callers

EPS = 1e-8


def _renyi_entropy_from_counts(counts: np.ndarray, q: float) -> float:
    probs = counts.astype(np.float64)
    probs = probs / max(probs.sum(), EPS)
    probs = np.clip(probs, EPS, 1.0)
    if abs(q - 1.0) < 1e-12:
        return float(-np.sum(probs * np.log(probs)))
    return float((1.0 / (1.0 - q)) * np.log(np.sum(np.power(probs, q))))


def _ngram_ids(tokens: np.ndarray, order: int, vocab_size: int) -> np.ndarray:
    n = len(tokens) - order + 1
    if n <= 0:
        return np.zeros((0,), dtype=np.int64)
    ids = np.zeros((n,), dtype=np.int64)
    for k in range(order):
        ids = ids * vocab_size + tokens[k : k + n]
    return ids


def estimate_renyi_dimensions_from_tokens(
    tokens: np.ndarray,
    vocab_size: int,
    orders: Sequence[int],
    qs: Sequence[float],
) -> Dict[str, float]:
    """Estimate generalized Rényi dimensions from symbolic n-gram statistics.

    For each q, we fit:
      H_q(m) ~= a_q * m + b_q
    and define:
      D_q = a_q / log(vocab_size)
    """
    out: Dict[str, float] = {}
    orders = [int(o) for o in orders if int(o) >= 1]
    if len(orders) == 0:
        return out

    h_by_q: Dict[float, List[float]] = {float(q): [] for q in qs}
    used_orders: List[int] = []

    for m in orders:
        ids = _ngram_ids(tokens, m, vocab_size=vocab_size)
        if ids.size == 0:
            continue
        _, counts = np.unique(ids, return_counts=True)
        used_orders.append(m)
        for q in qs:
            h = _renyi_entropy_from_counts(counts, float(q))
            h_by_q[float(q)].append(h)

    if len(used_orders) == 0:
        return out

    log_vocab = np.log(max(vocab_size, 2))
    x = np.array(used_orders, dtype=np.float64)

    for q in qs:
        qf = float(q)
        hs = np.array(h_by_q[qf], dtype=np.float64)
        if hs.size == 0:
            continue
        if hs.size == 1:
            slope = hs[0]
            intercept = 0.0
        else:
            slope, intercept = np.polyfit(x, hs, 1)
        d_rate = float(slope / max(log_vocab, EPS))
        d_clipped = float(np.clip(d_rate, 0.0, 1.0))

        qtag = str(qf).replace("-", "m").replace(".", "p")
        out[f"renyi_D_rate_q{qtag}"] = d_rate
        out[f"renyi_D_rate_clip_q{qtag}"] = d_clipped
        out[f"renyi_H_order{int(used_orders[-1])}_q{qtag}"] = float(hs[-1])
        out[f"renyi_num_orders_q{qtag}"] = float(len(hs))

    return out


def make_generator(config: Dict[str, object]):
    """Build the (β, γ)-parametrized data generator selected by ``config['task']``.

    Supported tasks:
      ``kv`` (default) — ``AlgorithmicKVGenerator``
      ``logical_folding`` — ``LogicalFoldingGenerator``
    """
    task = str(config.get("task", "kv"))
    vocab_size = int(config.get("vocab_size", 60))
    if task == "kv":
        return AlgorithmicKVGenerator(vocab_size=vocab_size)
    if task == "logical_folding":
        from data_logical_folding import LogicalFoldingGenerator
        return LogicalFoldingGenerator(vocab_size=vocab_size)
    if task == "nested_monoid":
        from data_nested_monoid import NestedMonoidGenerator
        return NestedMonoidGenerator(
            p=int(config.get("monoid_p", 31)),
            n_names=int(config.get("monoid_names", 512)),
            n_filler=int(config.get("monoid_filler", 6)),
            mode=str(config.get("gen_mode", "holographic")),
            source_len=int(config.get("source_len", 1024)),
            op_kind=str(config.get("op_kind", "perm")),
            n_perms=int(config.get("n_perms", 16)),
        )
    raise ValueError(
        f"unknown task={task!r} (expected 'kv', 'logical_folding', or 'nested_monoid')")


def compute_dataset_renyi_dimensions(
    beta: float,
    gamma: float,
    seed: int,
    config: Dict[str, object],
) -> Dict[str, float]:
    """Generate a diagnostic sequence and compute Rényi-based generalized dimensions."""
    seq_len = int(config.get("renyi_seq_len", 12000))
    qs = [float(x) for x in config.get("renyi_qs", [0.5, 1.0, 2.0])]
    orders = [int(x) for x in config.get("renyi_orders", [1, 2, 3])]
    vocab_size = int(config.get("vocab_size", 60))

    # Use a deterministic offset so diagnostic stats are stable and separated from training RNG path.
    set_all_seeds(seed + 100_000, deterministic=bool(config.get("deterministic", False)))
    gen = make_generator(config)
    x, _, _ = gen.generate_batch(batch_size=1, seq_len=seq_len, beta=beta, gamma=gamma)
    tokens = x[0].cpu().numpy().astype(np.int64)
    return estimate_renyi_dimensions_from_tokens(tokens, vocab_size=vocab_size, orders=orders, qs=qs)


class AlgorithmicKVGenerator:
    """Key-value retrieval with controllable long-range access and noise."""

    def __init__(self, vocab_size: int = 60) -> None:
        self.vocab_size = vocab_size
        self.noise_tokens = np.arange(1, 20)
        self.keys = np.arange(20, 40)
        self.values = np.arange(40, 60)

    def generate_batch(self, batch_size: int, seq_len: int, beta: float, gamma: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_batch = np.zeros((batch_size, seq_len), dtype=np.int64)
        y_batch = np.zeros((batch_size, seq_len), dtype=np.int64)
        m_batch = np.zeros((batch_size, seq_len), dtype=np.float32)

        for b in range(batch_size):
            memory = []  # list[(key,value)]
            t = 0
            while t < seq_len:
                if np.random.rand() < gamma:
                    x_batch[b, t] = int(np.random.choice(self.noise_tokens))
                    t += 1
                    continue

                if len(memory) == 0 or np.random.rand() < 0.5:
                    if t + 1 < seq_len:
                        k = int(np.random.choice(self.keys))
                        v = int(np.random.choice(self.values))
                        x_batch[b, t] = k
                        x_batch[b, t + 1] = v
                        memory.append((k, v))
                        t += 2
                    else:
                        t += 1
                else:
                    if beta <= 1e-3:
                        d = int(np.random.randint(1, len(memory) + 1))
                    else:
                        idx = np.arange(1, len(memory) + 1)
                        probs = 1.0 / np.power(idx, beta + 1.0)
                        probs = probs / probs.sum()
                        d = int(np.random.choice(idx, p=probs))

                    target_idx = len(memory) - d
                    k, v = memory[target_idx]

                    x_batch[b, t] = k
                    y_batch[b, t] = v
                    m_batch[b, t] = 1.0
                    t += 1

        return torch.tensor(x_batch), torch.tensor(y_batch), torch.tensor(m_batch)


def next_token_loss_and_acc(logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Standard next-token prediction objective used in pre-training."""
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="mean")
    pred = logits.argmax(-1)
    acc = (pred == targets).float().mean()
    return loss, acc


def masked_next_token_acc(logits: torch.Tensor, targets: torch.Tensor,
                          target_mask: torch.Tensor) -> float:
    """Next-token accuracy over positions where target_mask == 1 (answer/trace)."""
    pred = logits.argmax(-1)
    correct = (pred == targets).float() * target_mask
    denom = float(target_mask.sum().item())
    if denom <= 0:
        return 0.0
    return float(correct.sum().item() / denom)


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


def train_one_model(
    beta: float,
    gamma: float,
    seed: int,
    config: Dict[str, float],
    device: torch.device,
) -> Dict[str, object]:
    """Train once and evaluate at multiple sequence lengths.

    Returns dict with:
      - train_time_sec
      - eval_by_len: {length: {acc, loss}}
    """
    set_all_seeds(seed, deterministic=bool(config.get("deterministic", False)))

    train_len = int(config["train_len"])
    eval_lengths = [int(x) for x in config["eval_lengths"]]
    max_len = max([train_len] + eval_lengths)

    gen = make_generator(config)
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

    lr = float(config["lr"])
    train_steps = int(config["train_steps"])
    train_batch_size = int(config["train_batch_size"])
    grad_clip = float(config["grad_clip"])
    warmup_steps = int(config.get("warmup_steps", 0) or 0)
    lr_min_ratio = float(config.get("lr_min_ratio", 0.1) or 0.0)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    # Linear warmup over `warmup_steps`, then cosine decay from 1.0× to
    # `lr_min_ratio`× of peak lr across the remaining steps.
    def _lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        decay_steps = max(1, train_steps - warmup_steps)
        progress = (step - warmup_steps) / decay_steps
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return lr_min_ratio + (1.0 - lr_min_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_lambda)

    t0 = time.time()
    train_loss_curve: List[float] = []
    for _ in range(train_steps):
        model.train()
        x, _, _ = gen.generate_batch(train_batch_size, train_len, beta, gamma)
        x = x.to(device)
        if x.size(1) < 2:
            continue

        inp = x[:, :-1]
        y = x[:, 1:]
        logits = model(inp)
        loss, _ = next_token_loss_and_acc(logits, y)
        train_loss_curve.append(float(loss.item()))

        opt.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        scheduler.step()

    train_time = float(time.time() - t0)
    if len(train_loss_curve) == 0:
        initial_train_step_loss = 0.0
        final_train_step_loss = 0.0
        best_train_step_loss = 0.0
        aulc_train_to_final = 0.0
        aulc_train_to_final_norm = 0.0
    else:
        initial_train_step_loss = float(train_loss_curve[0])
        final_train_step_loss = float(train_loss_curve[-1])
        best_train_step_loss = float(np.min(train_loss_curve))
        delta = np.abs(np.array(train_loss_curve, dtype=np.float64) - final_train_step_loss)
        aulc_train_to_final = float(np.trapz(delta, dx=1.0))
        if len(train_loss_curve) > 1:
            aulc_train_to_final_norm = float(aulc_train_to_final / float(len(train_loss_curve) - 1))
        else:
            aulc_train_to_final_norm = float(aulc_train_to_final)

    answer_masked = bool(config.get("eval_answer_masked", False))
    eval_by_len: Dict[int, Dict[str, float]] = {}
    for L in eval_lengths:
        eval_by_len[int(L)] = evaluate_at_length(
            model=model,
            gen=gen,
            beta=beta,
            gamma=gamma,
            seq_len=int(L),
            eval_batches=int(config["eval_batches"]),
            batch_size=int(config["eval_batch_size"]),
            device=device,
            answer_masked=answer_masked,
        )

    renyi_diag = compute_dataset_renyi_dimensions(beta=beta, gamma=gamma, seed=seed, config=config)

    return {
        "train_time_sec": train_time,
        "objective": "next_token_pretraining",
        "initial_train_step_loss": initial_train_step_loss,
        "final_train_step_loss": final_train_step_loss,
        "best_train_step_loss": best_train_step_loss,
        "aulc_train_to_final": aulc_train_to_final,
        "aulc_train_to_final_norm": aulc_train_to_final_norm,
        "eval_by_len": eval_by_len,
        "renyi_diag": renyi_diag,
    }


