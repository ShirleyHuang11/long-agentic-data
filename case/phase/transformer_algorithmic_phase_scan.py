#!/usr/bin/env python3
"""Train a small Transformer on algorithmic reasoning tasks over (beta, gamma) data regimes.

Goal:
- Control training data distribution with beta/gamma.
- Scan a 2D grid and obtain empirical phase diagram from model behavior.
- Summarize regularities from train/extrapolation metrics.
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.colors import ListedColormap

EPS = 1e-8


class AlgorithmicKVGenerator:
    """Key-value retrieval with controllable long-range access and noise.

    - beta: controls retrieval distance distribution (power-law over memory depth).
      Smaller beta => heavier tail => longer-range retrieval.
    - gamma: controls filler noise probability.
    """

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
            memory: List[Tuple[int, int]] = []
            t = 0
            while t < seq_len:
                # Filler noise.
                if np.random.rand() < gamma:
                    x_batch[b, t] = np.random.choice(self.noise_tokens)
                    t += 1
                    continue

                # Write/read operations.
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
                        # Uniform over memory positions (max long-range disorder)
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


class TinyCausalTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 60,
        d_model: int = 64,
        nhead: int = 4,
        ff_mult: int = 2,
        num_layers: int = 2,
        max_ctx_tokens: int = 1024,
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
        self.register_buffer("mask", torch.nn.Transformer.generate_square_subsequent_mask(max_ctx_tokens))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        emb = self.embedding(x) + self.pos_emb[:, :seq_len, :]
        causal_mask = self.mask[:seq_len, :seq_len]
        out = self.transformer(emb, mask=causal_mask, is_causal=True)
        return self.fc(out)


def masked_ce_loss(logits: torch.Tensor, y: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    losses = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none")
    return (losses * m.reshape(-1)).sum() / (m.sum() + EPS)


def evaluate(
    model: nn.Module,
    gen: AlgorithmicKVGenerator,
    beta: float,
    gamma: float,
    seq_len: int,
    eval_batches: int,
    batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_count = 0.0

    with torch.no_grad():
        for _ in range(eval_batches):
            x, y, m = gen.generate_batch(batch_size, seq_len, beta, gamma)
            x = x.to(device)
            y = y.to(device)
            m = m.to(device)
            logits = model(x)
            loss = masked_ce_loss(logits, y, m)

            pred = logits.argmax(-1)
            correct = ((pred == y) * m).sum().item()
            count = m.sum().item()

            total_loss += float(loss.item())
            total_correct += float(correct)
            total_count += float(count)

    return {
        "loss": total_loss / max(eval_batches, 1),
        "acc": total_correct / max(total_count, EPS),
    }


def classify_phase_static(acc_train: float, acc_long: float, ratio: float) -> int:
    """Return empirical phase code.

    0: Chaos/Saturation (low train + low long)
    1: Emergent (moderate train + moderate long)
    2: Super-Generalization (high long + strong retention)
    3: Rote Memorization (high train but poor long generalization)
    """
    if acc_train < 0.35 and acc_long < 0.25:
        return 0
    if acc_train >= 0.70 and acc_long < 0.40:
        return 3
    if acc_long >= 0.70 and ratio >= 0.80:
        return 2
    return 1


def phase_name(code: int) -> str:
    return {
        0: "Chaos/Saturation",
        1: "Emergent",
        2: "Super-Generalization",
        3: "Rote Memorization",
    }[code]


def assign_relative_phase_labels(rows: List[Dict[str, float]]) -> None:
    """Assign phase labels from distribution quantiles of one scan result.

    This is useful for small-model / CPU runs where absolute accuracy thresholds
    can be too strict but relative structure is still informative.
    """
    if not rows:
        return

    train_acc = np.array([float(r["train_acc"]) for r in rows], dtype=float)
    long_acc = np.array([float(r["long_acc"]) for r in rows], dtype=float)
    gaps = np.array([float(r["generalization_gap"]) for r in rows], dtype=float)
    ratios = np.array([float(r["retention_ratio"]) for r in rows], dtype=float)

    q_train_low = float(np.quantile(train_acc, 0.25))
    q_train_high = float(np.quantile(train_acc, 0.70))
    q_long_med = float(np.quantile(long_acc, 0.50))
    q_long_high = float(np.quantile(long_acc, 0.75))
    q_gap_low = float(np.quantile(gaps, 0.25))
    q_gap_med = float(np.quantile(gaps, 0.50))
    q_gap_high = float(np.quantile(gaps, 0.75))
    q_ratio_med = float(np.quantile(ratios, 0.50))

    for r in rows:
        acc_train = float(r["train_acc"])
        acc_long = float(r["long_acc"])
        gap = float(r["generalization_gap"])
        ratio = float(r["retention_ratio"])

        # Low-capability corner
        if acc_train <= q_train_low and acc_long <= q_long_med:
            code = 0
        # Train well but poor extrapolation
        elif acc_train >= q_train_high and gap >= q_gap_high:
            code = 3
        # Strong long-context + low degradation
        elif acc_long >= q_long_high and gap <= q_gap_med and ratio >= q_ratio_med:
            code = 2
        else:
            code = 1

        r["phase_code"] = int(code)
        r["phase_name"] = phase_name(int(code))


def run_scan(args: argparse.Namespace) -> List[Dict[str, float]]:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cpu")
    gen = AlgorithmicKVGenerator(vocab_size=args.vocab_size)

    betas = [float(x.strip()) for x in args.betas.split(",") if x.strip()]
    gammas = [float(x.strip()) for x in args.gammas.split(",") if x.strip()]

    results: List[Dict[str, float]] = []
    total_runs = len(betas) * len(gammas)
    run_idx = 0

    for gamma in gammas:
        for beta in betas:
            run_idx += 1
            print(f"[Run {run_idx}/{total_runs}] beta={beta:.3f}, gamma={gamma:.3f}")

            model = TinyCausalTransformer(
                vocab_size=args.vocab_size,
                d_model=args.d_model,
                nhead=args.nhead,
                ff_mult=args.ff_mult,
                num_layers=args.num_layers,
                max_ctx_tokens=max(args.train_len, args.eval_long_len),
                dropout=args.dropout,
            ).to(device)

            opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

            for step in range(args.train_steps):
                model.train()
                x, y, m = gen.generate_batch(args.train_batch_size, args.train_len, beta, gamma)
                x = x.to(device)
                y = y.to(device)
                m = m.to(device)
                logits = model(x)
                loss = masked_ce_loss(logits, y, m)

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()

                if (step + 1) % args.log_every == 0:
                    print(f"  step {step+1:4d}/{args.train_steps} | loss={loss.item():.4f}")

            train_eval = evaluate(
                model,
                gen,
                beta,
                gamma,
                seq_len=args.train_len,
                eval_batches=args.eval_batches,
                batch_size=args.eval_batch_size,
                device=device,
            )
            long_eval = evaluate(
                model,
                gen,
                beta,
                gamma,
                seq_len=args.eval_long_len,
                eval_batches=args.eval_batches,
                batch_size=args.eval_batch_size,
                device=device,
            )

            acc_train = float(train_eval["acc"])
            acc_long = float(long_eval["acc"])
            loss_train = float(train_eval["loss"])
            loss_long = float(long_eval["loss"])
            ratio = acc_long / max(acc_train, EPS)
            gap = acc_train - acc_long

            alpha_theory = gamma / (2.0 * max(beta, EPS))
            phase = classify_phase_static(acc_train, acc_long, ratio)

            row = {
                "beta": beta,
                "gamma": gamma,
                "alpha_theory": alpha_theory,
                "train_acc": acc_train,
                "long_acc": acc_long,
                "train_loss": loss_train,
                "long_loss": loss_long,
                "retention_ratio": ratio,
                "generalization_gap": gap,
                "phase_code": phase,
                "phase_name": phase_name(phase),
            }
            results.append(row)

            print(
                "  eval "
                f"train_acc={acc_train:.3f}, long_acc={acc_long:.3f}, "
                f"gap={gap:.3f}, ratio={ratio:.3f}, phase={phase_name(phase)}"
            )

    return results


def save_csv(rows: List[Dict[str, float]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def load_csv(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            row: Dict[str, float] = {}
            for k, v in r.items():
                if k == "phase_name":
                    row[k] = str(v)
                else:
                    try:
                        row[k] = float(v)
                    except (TypeError, ValueError):
                        row[k] = v
            rows.append(row)
    return rows


def plot_heatmap(
    rows: List[Dict[str, float]],
    betas: List[float],
    gammas: List[float],
    key: str,
    title: str,
    cbar_label: str,
    out_path: Path,
    cmap: str = "viridis",
) -> None:
    z = np.zeros((len(gammas), len(betas)), dtype=float)
    for r in rows:
        i = gammas.index(float(r["gamma"]))
        j = betas.index(float(r["beta"]))
        z[i, j] = float(r[key])

    bb, gg = np.meshgrid(np.array(betas), np.array(gammas))

    fig, ax = plt.subplots(figsize=(8.5, 6.5), dpi=180)
    im = ax.pcolormesh(bb, gg, z, shading="auto", cmap=cmap)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\gamma$")
    ax.set_title(title)
    ax.set_xlim(min(betas), max(betas))
    ax.set_ylim(min(gammas), max(gammas))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_phase_diagram(rows: List[Dict[str, float]], betas: List[float], gammas: List[float], out_path: Path) -> None:
    z = np.zeros((len(gammas), len(betas)), dtype=int)
    alpha = np.zeros((len(gammas), len(betas)), dtype=float)

    for r in rows:
        i = gammas.index(float(r["gamma"]))
        j = betas.index(float(r["beta"]))
        z[i, j] = int(r["phase_code"])
        alpha[i, j] = float(r["alpha_theory"])

    bb, gg = np.meshgrid(np.array(betas), np.array(gammas))

    fig, ax = plt.subplots(figsize=(9, 7), dpi=180)
    cmap = ListedColormap(["#9aa0a6", "#9ddf9b", "#f4c76b", "#ef6b6b"])
    ax.imshow(
        z,
        origin="lower",
        extent=(min(betas), max(betas), min(gammas), max(gammas)),
        aspect="auto",
        cmap=cmap,
        alpha=0.75,
    )

    if len(betas) >= 2 and len(gammas) >= 2:
        cs = ax.contour(bb, gg, alpha, levels=[0.1, 0.2, 0.4, 0.8, 1.2], colors="black", linewidths=1.0)
        ax.clabel(cs, inline=True, fontsize=8, fmt=lambda v: rf"$\alpha={v:.2f}$")

    for r in rows:
        ax.text(float(r["beta"]), float(r["gamma"]), int(r["phase_code"]), ha="center", va="center", fontsize=8)

    ax.set_title("Empirical Phase Diagram from Tiny Transformer Behavior")
    ax.set_xlabel(r"$\beta$ (locality)")
    ax.set_ylabel(r"$\gamma$ (information rate)")
    ax.set_xlim(min(betas), max(betas))
    ax.set_ylim(min(gammas), max(gammas))
    ax.grid(alpha=0.15)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def summarize(rows: List[Dict[str, float]]) -> List[str]:
    lines: List[str] = []
    if not rows:
        return lines

    arr_train = np.array([float(r["train_acc"]) for r in rows])
    arr_long = np.array([float(r["long_acc"]) for r in rows])
    arr_gap = np.array([float(r["generalization_gap"]) for r in rows])
    arr_alpha = np.array([float(r["alpha_theory"]) for r in rows])

    corr_alpha_long = float(np.corrcoef(arr_alpha, arr_long)[0, 1]) if len(rows) > 1 else 0.0
    corr_alpha_gap = float(np.corrcoef(arr_alpha, arr_gap)[0, 1]) if len(rows) > 1 else 0.0

    lines.append(f"mean train_acc={arr_train.mean():.4f}")
    lines.append(f"mean long_acc={arr_long.mean():.4f}")
    lines.append(f"mean gap={arr_gap.mean():.4f}")
    lines.append(f"corr(alpha_theory, long_acc)={corr_alpha_long:.4f}")
    lines.append(f"corr(alpha_theory, gap)={corr_alpha_gap:.4f}")

    # Phase counts
    cnt: Dict[str, int] = {}
    for r in rows:
        p = str(r["phase_name"])
        cnt[p] = cnt.get(p, 0) + 1
    for k in sorted(cnt.keys()):
        lines.append(f"phase_count[{k}]={cnt[k]}")

    # Best/worst long-context points
    best = max(rows, key=lambda r: float(r["long_acc"]))
    worst = min(rows, key=lambda r: float(r["long_acc"]))
    lines.append(
        "best_long_acc="
        f"{best['long_acc']:.4f} at beta={best['beta']:.3f}, gamma={best['gamma']:.3f}, phase={best['phase_name']}"
    )
    lines.append(
        "worst_long_acc="
        f"{worst['long_acc']:.4f} at beta={worst['beta']:.3f}, gamma={worst['gamma']:.3f}, phase={worst['phase_name']}"
    )

    return lines


def write_summary_md(rows: List[Dict[str, float]], out_path: Path) -> None:
    lines = summarize(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# Tiny Transformer Algorithmic Phase Scan Summary\n\n")
        f.write("## Key Metrics\n")
        for l in lines:
            f.write(f"- {l}\n")
        f.write("\n## Files\n")
        f.write("- transformer_phase_results.csv\n")
        f.write("- transformer_phase_diagram.png\n")
        f.write("- transformer_long_acc_heatmap.png\n")
        f.write("- transformer_gap_heatmap.png\n")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Scan beta-gamma phase diagram with tiny Transformer")
    ap.add_argument("--out-dir", type=str, default="plots/transformer_phase")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--phase-mode", type=str, default="relative", choices=["relative", "static"])
    ap.add_argument("--from-csv", type=str, default="", help="If set, skip training and load this results CSV.")

    # Data grid
    ap.add_argument("--betas", type=str, default="0.15,0.5,1.5,4.0")
    ap.add_argument("--gammas", type=str, default="0.05,0.3,0.6,0.9")

    # Task/model
    ap.add_argument("--vocab-size", type=int, default=60)
    ap.add_argument("--train-len", type=int, default=128)
    ap.add_argument("--eval-long-len", type=int, default=512)
    ap.add_argument("--d-model", type=int, default=48)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--ff-mult", type=int, default=2)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)

    # Optimization
    ap.add_argument("--train-steps", type=int, default=120)
    ap.add_argument("--train-batch-size", type=int, default=64)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--eval-batches", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=30)

    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    betas = [float(x.strip()) for x in args.betas.split(",") if x.strip()]
    gammas = [float(x.strip()) for x in args.gammas.split(",") if x.strip()]

    if args.from_csv:
        rows = load_csv(Path(args.from_csv))
        # keep only rows inside selected beta/gamma grid for plotting consistency
        rows = [
            r
            for r in rows
            if float(r["beta"]) in betas and float(r["gamma"]) in gammas
        ]
        if not rows:
            raise ValueError("No matching rows found in --from-csv for provided --betas/--gammas.")
    else:
        rows = run_scan(args)

    if args.phase_mode == "relative":
        assign_relative_phase_labels(rows)

    csv_path = out_dir / "transformer_phase_results.csv"
    phase_path = out_dir / "transformer_phase_diagram.png"
    long_acc_path = out_dir / "transformer_long_acc_heatmap.png"
    gap_path = out_dir / "transformer_gap_heatmap.png"
    summary_path = out_dir / "transformer_phase_summary.md"

    save_csv(rows, csv_path)
    plot_phase_diagram(rows, betas, gammas, phase_path)
    plot_heatmap(
        rows,
        betas,
        gammas,
        key="long_acc",
        title="Long-Context Accuracy Heatmap",
        cbar_label="accuracy @ eval_long_len",
        out_path=long_acc_path,
        cmap="viridis",
    )
    plot_heatmap(
        rows,
        betas,
        gammas,
        key="generalization_gap",
        title="Generalization Gap Heatmap (train_acc - long_acc)",
        cbar_label="gap",
        out_path=gap_path,
        cmap="magma",
    )
    write_summary_md(rows, summary_path)

    print("\n[Done] Output files:")
    print(f"  - {csv_path}")
    print(f"  - {phase_path}")
    print(f"  - {long_acc_path}")
    print(f"  - {gap_path}")
    print(f"  - {summary_path}")

    print("\n[Quick summary]")
    for line in summarize(rows):
        print(f"  - {line}")


if __name__ == "__main__":
    main()
