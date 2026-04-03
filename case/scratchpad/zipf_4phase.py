import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from pathlib import Path
from tqdm.auto import tqdm

# Ensure project root is importable when this script is run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import SSMBlock

class PhaseTransitionDataGen:
    """Core task: learn to retrieve the correct value for a queried key from noisy, long-range key-value memories."""
    def __init__(self, vocab_size=60):
        # Vocabulary split:
        # 1-19: pure noise (Noise)
        # 20-39: logical keys (Keys)
        # 40-59: logical values (Values)
        self.noise_tokens = np.arange(1, 20)
        self.keys = np.arange(20, 40)
        self.values = np.arange(40, 60)
        
    def generate_batch(self, batch_size, seq_len, beta, gamma):
        x_batch = np.zeros((batch_size, seq_len), dtype=np.int64)
        y_batch = np.zeros((batch_size, seq_len), dtype=np.int64)
        m_batch = np.zeros((batch_size, seq_len), dtype=np.float32)
        
        for b in range(batch_size):
            memory = [] # Record current (Key, Value) pairs
            t = 0
            while t < seq_len:
                # 1. Generate filler noise with probability gamma
                if np.random.rand() < gamma:
                    x_batch[b, t] = np.random.choice(self.noise_tokens)
                    t += 1
                    continue
                
                # 2. Logical operation: write or read
                if len(memory) == 0 or np.random.rand() < 0.5:
                    # Write: create a new logical mapping K -> V
                    if t + 1 < seq_len:
                        k, v = np.random.choice(self.keys), np.random.choice(self.values)
                        x_batch[b, t] = k
                        x_batch[b, t+1] = v
                        memory.append((k, v))
                        t += 2
                    else:
                        t += 1 # End of sequence, skip
                else:
                    # Read: use beta-controlled pointer jumps to query memory
                    if beta <= 0.01:
                        # 🔴 Abyss region: globally uniform random (beta=0)
                        d = np.random.randint(1, len(memory) + 1)
                    else:
                        # 🟢/🟠/🟡 regions: Zipf power law controls pointer jump distance
                        probs = 1.0 / (np.arange(1, len(memory) + 1) ** (beta + 1))
                        probs /= probs.sum()
                        d = np.random.choice(np.arange(1, len(memory) + 1), p=probs)
                    
                    target_idx = len(memory) - d
                    k, v = memory[target_idx]
                    
                    x_batch[b, t] = k  # Input is the key
                    y_batch[b, t] = v  # Model predicts the corresponding value
                    m_batch[b, t] = 1.0 # Compute loss only here (reasoning point)
                    t += 1
                    
        return torch.tensor(x_batch), torch.tensor(y_batch), torch.tensor(m_batch)

class StandardTransformer(nn.Module):
    def __init__(self, vocab_size=60, d_model=64, max_ctx_tokens=2048):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, max_ctx_tokens, d_model))
        layer = nn.TransformerEncoderLayer(d_model, nhead=4, dim_feedforward=128, batch_first=True)
        # Use a 2-layer standard Transformer to emulate scaling-law behavior
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.fc = nn.Linear(d_model, vocab_size)
        
        # Prevent the model from seeing future tokens (causal mask)
        self.register_buffer("mask", torch.nn.Transformer.generate_square_subsequent_mask(max_ctx_tokens))

    def forward(self, x):
        seq_len = x.size(1)
        emb = self.embedding(x) + self.pos_emb[:, :seq_len, :]
        causal_mask = self.mask[:seq_len, :seq_len]
        out = self.transformer(emb, mask=causal_mask, is_causal=True)
        return self.fc(out)


class StandardMamba(nn.Module):
    def __init__(self, vocab_size=60, d_model=64, d_state=16, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([SSMBlock(d_model, d_state=d_state) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        out = self.embedding(x)
        for layer in self.layers:
            out = layer(out)
        out = self.final_norm(out)
        return self.fc(out)

def run_physics_simulation(train_steps=501):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = PhaseTransitionDataGen()
    
    # Four physical anchor regimes
    regimes = {
        "🟢 Natural (Wikipedia)": {"beta": 2.0, "gamma": 0.8, "train_steps": train_steps},
        "🟠 CoT (reasoning with explanations)": {"beta": 0.5, "gamma": 0.4, "train_steps": train_steps},
        "🟡 Edge of Chaos (o1/sweet spot)": {"beta": 0.05, "gamma": 0.05, "train_steps": train_steps},
        "🔴 The Abyss (topological collapse zone)": {"beta": 0.0, "gamma": 0.0, "train_steps": train_steps}
    }
    
    train_len = 256
    test_lengths = [256, 512, 1024, 2048, 4096, 8192]
    results = {}

    def train_model(model, beta, gamma, desc, batch_size=256, train_steps=501):
        # Linear scaling: lr ∝ batch_size (4x batch → 4x lr)
        base_lr = 3e-4
        lr = base_lr * (batch_size / 256)
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=train_steps, eta_min=1e-6)
        model.train()
        pbar = tqdm(range(train_steps), desc=desc, leave=False)
        for _ in pbar:
            x, y, m = gen.generate_batch(batch_size, train_len, beta, gamma)
            x, y, m = x.to(device), y.to(device), m.to(device)

            logits = model(x)
            # Compute masked cross-entropy loss
            loss = (F.cross_entropy(logits.reshape(-1, 60), y.reshape(-1), reduction='none') * m.reshape(-1)).sum() / (m.sum() + 1e-8)

            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

    def eval_model(model, beta, gamma, seq_len, batch_size=64):
        model.eval()
        with torch.no_grad():
            x, y, m = gen.generate_batch(batch_size, seq_len, beta, gamma)
            x, y, m = x.to(device), y.to(device), m.to(device)
            logits = model(x)
            loss = (F.cross_entropy(logits.reshape(-1, 60), y.reshape(-1), reduction='none') * m.reshape(-1)).sum() / (m.sum() + 1e-8)
            preds = logits.argmax(-1)
            acc = ((preds == y) * m).sum() / (m.sum() + 1e-8)
        return {"loss": loss.item(), "acc": acc.item()}
    
    print("🚀 Starting the LLM manifold phase-transition simulator...\n")
    print(f"Train length: {train_len} | Test lengths: {test_lengths}\n")
    
    for name, params in regimes.items():
        steps = params.get("train_steps", 501)
        print(f"Transitioning to regime: {name} (β={params['beta']}, γ={params['gamma']}) [train_steps={steps}]")
        transformer = StandardTransformer(max_ctx_tokens=max(test_lengths)).to(device)
        mamba = StandardMamba().to(device)

        train_bs = 1024  # 4x default 256; lr auto-scaled in train_model
        train_model(transformer, params["beta"], params["gamma"], desc=f"{name} | Transformer train", batch_size=train_bs, train_steps=steps)
        train_model(mamba, params["beta"], params["gamma"], desc=f"{name} | Mamba train", batch_size=train_bs, train_steps=steps)

        results[name] = {"Transformer": {}, "Mamba": {}}
        for test_len in test_lengths:
            eval_bs = 64
            while True:
                try:
                    trans_res = eval_model(transformer, params['beta'], params['gamma'], test_len, batch_size=eval_bs)
                    break
                except torch.OutOfMemoryError:
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    eval_bs //= 2
                    if eval_bs == 0:
                        raise

            mamba_bs = 64
            while True:
                try:
                    mamba_res = eval_model(mamba, params['beta'], params['gamma'], test_len, batch_size=mamba_bs)
                    break
                except torch.OutOfMemoryError:
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    mamba_bs //= 2
                    if mamba_bs == 0:
                        raise

            results[name]["Transformer"][test_len] = trans_res
            results[name]["Mamba"][test_len] = mamba_res
            print(
                f"  --> L={test_len:4d} | "
                f"T Loss: {trans_res['loss']:.4f}, T Acc: {trans_res['acc']:.2%} (bs={eval_bs}) | "
                f"M Loss: {mamba_res['loss']:.4f}, M Acc: {mamba_res['acc']:.2%} (bs={mamba_bs})"
            )
        print("")

    print("=" * 60)
    print("📊 Final report of the four physical phases:")
    print("=" * 60)
    for name, res in results.items():
        for test_len in test_lengths:
            t_res = res["Transformer"][test_len]
            m_res = res["Mamba"][test_len]
            print(
                f"| {name.ljust(25)} | L={str(test_len).rjust(4)} | "
                f"T Loss: {t_res['loss']:.4f} | T Acc: {t_res['acc']:>7.2%} | "
                f"M Loss: {m_res['loss']:.4f} | M Acc: {m_res['acc']:>7.2%} |"
            )
    print("=" * 60)

if __name__ == "__main__":
    import fire
    fire.Fire(run_physics_simulation)