import torch
import torch.nn as nn
import numpy as np

# ==========================================
# Optimized data generation: clearer recursive signal
# ==========================================
class RecursiveFunctionDataGen:
    def __init__(self, seq_len=32, num_ops=5, modulo=17):
        self.seq_len = seq_len
        self.num_ops = num_ops
        self.modulo = modulo
        
    def generate_batch(self, batch_size, logic_depth):
        sequences = []
        labels = []
        for _ in range(batch_size):
            ops = np.random.randint(1, self.num_ops + 1, size=logic_depth)
            start_val = np.random.randint(0, self.modulo)
            
            # Computation logic: f(g(h(x))) -> operations applied from right to left
            current_val = start_val
            for op in reversed(ops):
                if op == 1: current_val = (current_val + 2) % self.modulo
                elif op == 2: current_val = (current_val * 3) % self.modulo
                elif op == 3: current_val = (current_val + 5) % self.modulo
                elif op == 4: current_val = (current_val * 2) % self.modulo
                elif op == 5: current_val = (current_val + 7) % self.modulo
            labels.append(current_val)
            
            # Sequence: [Ops..., Start_Val, Padding...]
            seq = list(ops) + [start_val + self.num_ops + 1]
            seq += [0] * (self.seq_len - len(seq))
            sequences.append(seq[:self.seq_len])
            
        return torch.tensor(sequences), torch.tensor(labels)

# ==========================================
# Architecture optimization: aligned with holographic computation physics
# ==========================================
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=128): # Increase dimension to 128
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 128, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model, 8, 256, batch_first=True, dropout=0.0)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.fc = nn.Linear(d_model, 17) # 17 here corresponds to modulo

    def forward(self, x):
        # Find Start_Val position in sequence (i.e., logic_depth) for prediction, or use the end directly
        x = self.embedding(x) + self.pos_encoder[:, :x.size(1), :]
        x = self.transformer(x)
        return self.fc(x[:, 0, :]) # Key: in holographic tasks, the first or last token is more representative

class StateSpaceProxy(nn.Module):
    def __init__(self, vocab_size, d_model=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.rnn = nn.GRU(d_model, d_model, num_layers=4, batch_first=True)
        self.fc = nn.Linear(d_model, 17)

    def forward(self, x):
        x = self.embedding(x)
        out, hidden = self.rnn(x)
        return self.fc(hidden[-1])

def test_phase_transition(depth):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = RecursiveFunctionDataGen(seq_len=64)
    
    for name, model_cls in [("Transformer", TransformerModel), ("StateSpace", StateSpaceProxy)]:
        print(f"\n>> Testing: {name} | Depth: {depth}")
        model = model_cls(50).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3) # Restore high learning rate
        crit = nn.CrossEntropyLoss()
        
        for step in range(2001):
            inputs, targets = gen.generate_batch(64, depth)
            inputs, targets = inputs.to(device), targets.to(device)
            
            logits = model(inputs)
            loss = crit(logits, targets)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            if step % 500 == 0:
                acc = (logits.argmax(-1) == targets).float().mean()
                print(f"Step {step:4d} | Loss: {loss.item():.4f} | Acc: {acc:.2%}")
                if acc > 0.95: 
                    print(f"!!! {name} achieved phase transition convergence at Step {step} !!!")
                    break

if __name__ == "__main__":
    for d in [2, 5, 8, 12]:
        test_phase_transition(d)