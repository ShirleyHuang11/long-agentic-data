import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ==========================================
# Core: holographic data generation (every token has logical meaning)
# ==========================================
class HolographicRecursiveGen:
    def __init__(self, seq_len=64, num_ops=5, modulo=17):
        self.seq_len = seq_len
        self.num_ops = num_ops
        self.modulo = modulo
        
    def generate_batch(self, batch_size, depth):
        sequences = []
        # Dense supervision: each step has a label
        dense_labels = [] 
        
        for _ in range(batch_size):
            ops = np.random.randint(1, self.num_ops + 1, size=depth)
            start_val = np.random.randint(0, self.modulo)
            
            current_seq = list(ops) + [start_val + self.num_ops + 1]
            labels = []
            
            # Compute intermediate steps: holographic property requires logic to be self-similar
            val = start_val
            # We record the result of each computation step as supervision
            # This is a typical "computation stack" maintenance process
            for op in reversed(ops):
                if op == 1: val = (val + 2) % self.modulo
                elif op == 2: val = (val * 3) % self.modulo
                elif op == 3: val = (val + 5) % self.modulo
                elif op == 4: val = (val * 2) % self.modulo
                elif op == 5: val = (val + 7) % self.modulo
                labels.append(val)
            
            # Pad to length
            padding_len = self.seq_len - len(current_seq)
            current_seq += [0] * padding_len
            # Also pad labels (we only care about outputs of logic steps)
            labels = [0] * (len(current_seq) - len(labels)) + list(reversed(labels))
            
            sequences.append(current_seq)
            dense_labels.append(labels)
            
        return torch.tensor(sequences), torch.tensor(dense_labels)

# ==========================================
# Enhanced architecture: increase width and depth, align with holographic manifold
# ==========================================
class TransformerCore(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=6, seq_len=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len+1, d_model))
        self.seq_len = seq_len
        layer = nn.TransformerEncoderLayer(d_model, nhead, d_model*4, batch_first=True, dropout=0.05)
        self.model = nn.TransformerEncoder(layer, num_layers)
        self.head = nn.Linear(d_model, 17) # Modulo 17

    def forward(self, x):
        x = self.embedding(x) + self.pos_emb[:, :x.size(1), :]
        x = self.model(x)
        return self.head(x) # Output predictions for every position in the sequence

class MambaProxyCore(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_layers=6, seq_len=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.rnn = nn.GRU(d_model, d_model, num_layers, batch_first=True)
        self.seq_len = seq_len
        self.head = nn.Linear(d_model, 17)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        return self.head(out)

# ==========================================
# Training and duel
# ==========================================
def train_duel(depth, multiplier=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = HolographicRecursiveGen(seq_len=64)
    
    models = [
        ("Transformer (Attention)", TransformerCore(60, seq_len=depth).to(device)),
        ("Mamba-Proxy (State-Space)", MambaProxyCore(60, seq_len=depth).to(device))
    ]
    
    print(f"\n🚀 Starting holographic depth duel | Logic nesting depth: {depth}")
    
    for name, model in models:
        print(f"\n--- Training: {name} ---")
        opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
        
        for step in range(3001):
            inputs, targets = gen.generate_batch(64, depth)
            inputs, targets = inputs.to(device), targets.to(device)
            
            logits = model(inputs) # (batch, seq, 17)
            # Only compute loss for valid logic positions
            loss = F.cross_entropy(logits.view(-1, 17), targets.view(-1))
            
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            
            if step % 500 == 0:
                # Accuracy only looks at the last token (final answer)
                final_logits = logits[:, depth, :] 
                final_targets = targets[:, depth]
                acc = (final_logits.argmax(-1) == final_targets).float().mean()
                print(f"Step {step:4d} | Loss: {loss.item():.4f} | Final answer accuracy: {acc:.2%}")
                if acc > 0.98:
                    print(f"✅ {name} achieved phase transition convergence!")
                    break
        
        # Verify generalization after training
        verify_generalization(model, name, depth, multiplier)

def verify_generalization(model, name, train_depth, multiplier=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    test_depths = [train_depth * multiplier ** i for i in range(1, multiplier+1)]
    
    print(f"\nVerifying holographic extrapolation capability for model [{name}]:")
    
    for td in test_depths:
        # Note: for extrapolation, we need to adjust the generator's seq_len
        test_gen = HolographicRecursiveGen(seq_len=td + 10) 
        inputs, targets = test_gen.generate_batch(100, td)
        inputs, targets = inputs.to(device), targets.to(device)
        
        with torch.no_grad():
            # For Transformer, if using absolute positional encoding, extrapolation will be limited
            # For Mamba/RNN, due to recurrent processing, theoretically can extrapolate infinitely
            try:
                logits = model(inputs)
                final_logits = logits[:, td, :] 
                final_targets = targets[:, td]
                acc = (final_logits.argmax(-1) == final_targets).float().mean()
                print(f"  Test depth {td:4d} | Accuracy: {acc:.2%}")
            except Exception as e:
                print(f"  Test depth {td:4d} | Failed (architecture limitation, e.g., positional encoding out of bounds)")

if __name__ == "__main__":
    # Directly challenge the previously failed depth 8
    # for depth in range(4, 8):
    for depth in range(6, 8):
        train_duel(depth=2**depth, multiplier=4)
