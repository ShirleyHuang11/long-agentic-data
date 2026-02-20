import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PhaseDyckGenerator:
    def __init__(self, k_types=4):
        self.k_types = k_types
        self.vocab_size = 2 * k_types + 3 # +1 for noise token
        self.noise_token = self.vocab_size - 1
        
    def generate_step_data(self, batch_size, phase="holographic", max_depth=16):
        x_list, y_list = [], []
        
        for _ in range(batch_size):
            depth = np.random.randint(1, max_depth + 1)
            opens = list(np.random.randint(1, self.k_types + 1, size=depth))
            t = np.random.randint(1, depth + 1)
            
            # Base clean logical chain (stack)
            base_stack = opens[:t]
            expected_close = base_stack[-1] + self.k_types
            
            # ==========================================
            # 👑 Core idea: distort the data manifold based on the selected phase
            # ==========================================
            if phase == "holographic":
                # Phase 1: perfect holographic signal, no extra noise
                x = base_stack
                
            elif phase == "noisy":
                # Phase 2: high-gamma collapse (inject 50% pure noise tokens)
                x = []
                for token in base_stack:
                    x.append(token)
                    if np.random.rand() > 0.5:
                        x.append(self.noise_token)
                        
            elif phase == "redundant":
                # Phase 3: low-beta redundancy (simulate RL loops by repeating stack-top 3 times)
                x = base_stack[:-1] + [base_stack[-1]] * 3
                
            elif phase == "lossy":
                # Phase 4: logical fault line (randomly drop 30% of intermediate reasoning steps)
                x = [token for token in base_stack if np.random.rand() > 0.3]
                if len(x) == 0: # Ensure at least one token remains
                    x = [base_stack[-1]]
            
            x_list.append(x)
            y_list.append(expected_close)
            
        max_len = max(len(seq) for seq in x_list)
        x_padded = [seq + [0]*(max_len - len(seq)) for seq in x_list]
        return torch.tensor(x_padded), torch.tensor(y_list)

class ScratchpadTransformer(nn.Module):
    def __init__(self, vocab_size, max_ctx_tokens=2048, d_model=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, max_ctx_tokens, d_model)) 
        layer = nn.TransformerEncoderLayer(d_model, 4, 128, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        seq_len = x.size(1)
        emb = self.embedding(x) + self.pos_emb[:, :seq_len, :]
        out = self.transformer(emb)
        return self.fc(out[:, -1, :])

def run_phase_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = PhaseDyckGenerator(k_types=4)
    phases = ["holographic", "noisy", "redundant", "lossy"]
    results = {}

    print("🚀 Starting the beta-gamma phase map 4-quadrant comparison experiment...")
    print("-" * 50)
    test_depth = 4096
    train_depth = 16
    
    for phase in phases:
        print(f"\n🌀 Training model in the [{phase.upper()}] quadrant...")
        model = ScratchpadTransformer(gen.vocab_size, max_ctx_tokens=test_depth).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        # Training stage (short sequences, max_depth=16)
        model.train()
        for step in range(1501):
            x, y = gen.generate_step_data(128, phase=phase, max_depth=train_depth)
            x, y = x.to(device), y.to(device)
            
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
        # Evaluation stage (extreme extrapolation)
        model.eval()
        # Build fixed-shape extreme-depth test samples directly to avoid row-size mismatch.
        eval_batch = 512
        t_x = torch.zeros((eval_batch, test_depth), dtype=torch.long)
        t_y = torch.zeros((eval_batch,), dtype=torch.long)
        for i in range(eval_batch):
            base_opens = list(np.random.randint(1, 5, size=test_depth))
            if phase == "holographic":
                seq = base_opens
            elif phase == "noisy":
                seq = []
                for token in base_opens:
                    seq.append(token)
                    if np.random.rand() > 0.5:
                        seq.append(gen.noise_token)
            elif phase == "redundant":
                seq = base_opens[:-1] + [base_opens[-1]] * 3
            else:  # lossy
                seq = [t for t in base_opens if np.random.rand() > 0.3]
                if not seq:
                    seq = [base_opens[-1]]

            seq = seq[:test_depth] + [0] * max(0, test_depth - len(seq))
            t_x[i] = torch.tensor(seq, dtype=torch.long)
            t_y[i] = base_opens[-1] + 4

        t_x = t_x.to(device)
        t_y = t_y.to(device)
        
        with torch.no_grad():
            t_logits = model(t_x)
            t_acc = (t_logits.argmax(-1) == t_y).float().mean()
            results[phase] = t_acc.item()
            print(f"  ✅ Extrapolation test (Depth={test_depth}) accuracy: {t_acc:.2%}")

    print("\n" + "=" * 50)
    print(f"📊 Final experiment results (Depth {test_depth} Extrapolation):")
    print("=" * 50)
    for p, acc in results.items():
        print(f"| {p.ljust(12)} | {acc:>7.2%} |")
    print("=" * 50)

if __name__ == "__main__":
    torch.manual_seed(42)
    run_phase_experiment()