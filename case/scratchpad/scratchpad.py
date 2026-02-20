import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ==========================================
# Holographic data setup: Chain-of-Thought data construction
# Force the model to externalize "hidden states" onto an explicit tape
# ==========================================
class CoTDyckGenerator:
    def __init__(self, k_types=4):
        self.k_types = k_types
        # Vocabulary: 0(Pad), 1~k(Open), k+1~2k(Close), 2k+1(stack-bottom marker)
        self.vocab_size = 2 * k_types + 2
        self.empty_stack_token = self.vocab_size - 1
        
    def generate_step_data(self, batch_size, max_depth=16):
        """
        Train the model to predict one-step stack operations.
        Input: currently observed symbols + historical stack state
        Output: the next expected symbol (closed logical loop)
        """
        x_list, y_list = [], []
        
        for _ in range(batch_size):
            depth = np.random.randint(1, max_depth + 1)
            # Generate the first half: random push operations
            opens = list(np.random.randint(1, self.k_types + 1, size=depth))
            
            # Randomly sample a timestep t (an RG-flow-like snapshot in holographic data)
            t = np.random.randint(1, depth + 1)
            current_stack = opens[:t] # Current stack state
            
            # Model input: all elements in the current stack -> external discrete tape (context)
            # Example: [1, 2, 1] means the stack currently contains [ { [
            x = current_stack 
            
            # The only task: predict which closing bracket should come next if matching starts now
            # It should pop the stack top, i.e., the close token for current_stack[-1]
            expected_close = current_stack[-1] + self.k_types
            
            x_list.append(x)
            y_list.append(expected_close)
            
        # Pad to equal length
        max_len = max(len(seq) for seq in x_list)
        x_padded = [seq + [0]*(max_len - len(seq)) for seq in x_list]
        
        return torch.tensor(x_padded), torch.tensor(y_list)

# ==========================================
# Minimal causal model: keep logic through external context
# ==========================================
class ScratchpadTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        # Use a recurrent encoder so memory grows linearly with context length.
        self.rnn = nn.GRU(d_model, d_model, num_layers=2, batch_first=True)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        emb = self.embedding(x)
        # Consume the full context without truncation; use the last state for prediction.
        out, _ = self.rnn(emb)
        return self.fc(out[:, -1, :]) # Use the final token representation for prediction

def run_cot_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = CoTDyckGenerator(k_types=4)
    model = ScratchpadTransformer(gen.vocab_size).to(device)
    
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    print("\n🚀 Starting [CoT Externalization] training (System 2 / Scratchpad)...")
    # Train only on very short sequences (stack depth 2~16)
    for step in range(2001):
        x, y = gen.generate_step_data(128, max_depth=16)
        x, y = x.to(device), y.to(device)
        
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        if step % 500 == 0:
            acc = (logits.argmax(-1) == y).float().mean()
            print(f"Step {step:4d} | Train Loss: {loss.item():.4f} | Acc: {acc:.2%}")

    print("\n🌍 Evaluating chain-of-thought-based 'infinite extrapolation' (Zero-Shot Extrapolation)...")
    model.eval()
    
    # Test lengths far beyond the training length (16): 256, 1024, 2048, 4096, 8192, 16384, 32768
    test_depths = [64, 256, 1024, 2048, 4096, 8192, 16384, 32768]
    
    for td in test_depths:
        # Keep total tokens roughly bounded to avoid OOM on ultra-long contexts.
        eval_batch = max(1, min(256, 131072 // td))
        # Directly construct fixed-length test samples to avoid shape mismatch from overwrite-after-padding
        t_x = torch.randint(1, gen.k_types + 1, (eval_batch, td))
        t_y = t_x[:, -1] + gen.k_types

        t_x, t_y = t_x.to(device), t_y.to(device)
        
        with torch.no_grad():
            try:
                t_logits = model(t_x)
                t_acc = (t_logits.argmax(-1) == t_y).float().mean()
                print(f"  --> Current context length (stack depth) {td:4d} | Accuracy: {t_acc:.2%}")
            except Exception as e:
                print(f"  --> Current context length (stack depth) {td:4d} | OOM or architecture limit hit")

if __name__ == "__main__":
    run_cot_experiment()