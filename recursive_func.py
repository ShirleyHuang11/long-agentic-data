import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DyckLanguageGen:
    def __init__(self, k_types=4):
        self.k_types = k_types
        self.vocab_size = 2 * k_types + 1
        
    def generate_batch(self, batch_size, depth):
        sequences, masks = [], []
        for _ in range(batch_size):
            opens = np.random.randint(1, self.k_types + 1, size=depth)
            closes = [x + self.k_types for x in reversed(opens)]
            seq = list(opens) + list(closes)
            mask = [0] * depth + [1] * depth
            sequences.append(seq)
            masks.append(mask)
        return torch.tensor(sequences), torch.tensor(masks)

class StackLSTM(nn.Module):
    def __init__(self, vocab_size, d_model=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        # Add a few more layers so the model has enough nonlinearity to approximate a step function.
        self.rnn = nn.LSTM(d_model, d_model, num_layers=3, batch_first=True)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        return self.fc(out)

class CausalTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        layer = nn.TransformerEncoderLayer(d_model, 4, d_model*4, batch_first=True, dropout=0.0)
        self.transformer = nn.TransformerEncoder(layer, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def _sinusoidal_pos_emb(self, seq_len, device, dtype):
        position = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=device, dtype=dtype) * (-np.log(10000.0) / self.d_model)
        )
        pe = torch.zeros(1, seq_len, self.d_model, device=device, dtype=dtype)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x):
        seq_len = x.size(1)
        emb = self.embedding(x)
        x = emb + self._sinusoidal_pos_emb(seq_len, emb.device, emb.dtype)
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=emb.device, dtype=emb.dtype),
            diagonal=1,
        )
        x = self.transformer(x, mask=causal_mask, is_causal=True)
        return self.fc(x)
    
def train_model_scale_invariant(model, gen, device, steps=3000, batch_size=128, min_depth=4, max_depth=32):
    model = model.to(device)
    model.train()
    # Use L2 regularization (weight decay) to push weights toward sharper regimes and form harder "gates".
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    for step in range(steps + 1):
        # Core idea: dynamic scale sampling (dynamic depth sampling).
        # Instead of always training with depth=32, each batch samples depth randomly from min_depth to max_depth.
        current_depth = np.random.randint(min_depth, max_depth + 1)

        inputs, masks = gen.generate_batch(batch_size, current_depth)
        inputs, masks = inputs.to(device), masks.to(device)

        x, y, m = inputs[:, :-1], inputs[:, 1:], masks[:, 1:].float()

        logits = model(x)
        loss_unreduced = F.cross_entropy(logits.reshape(-1, gen.vocab_size), y.reshape(-1), reduction='none')
        loss = (loss_unreduced * m.reshape(-1)).sum() / m.sum()

        opt.zero_grad()
        loss.backward()
        # Clip gradients to prevent exploding gradients.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 500 == 0:
            preds = logits.argmax(-1)
            acc = ((preds == y) * m).sum() / m.sum()
            print(f"Step {step:4d} | Depth sampled: {current_depth:2d} | Loss: {loss.item():.4f} | Stack Acc: {acc:.2%}")

    return model

def evaluate_model(model, gen, device, test_depths):
    model.eval()
    results = {}
    for td in test_depths:
        t_inputs, t_masks = gen.generate_batch(128, td)
        t_inputs, t_masks = t_inputs.to(device), t_masks.to(device)
        t_x, t_y, t_m = t_inputs[:, :-1], t_inputs[:, 1:], t_masks[:, 1:].float()

        with torch.no_grad():
            t_logits = model(t_x)
            t_preds = t_logits.argmax(-1)
            t_acc = ((t_preds == t_y) * t_m).sum() / t_m.sum()
            results[td] = t_acc.item()
    return results

def train_scale_invariant():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = DyckLanguageGen(k_types=4)
    test_depths = [32, 64, 128, 256, 512, 1024]

    model_builders = {
        "StackLSTM": lambda: StackLSTM(gen.vocab_size, d_model=128),
        "CausalTransformer": lambda: CausalTransformer(gen.vocab_size, d_model=128, num_layers=4),
    }

    all_results = {}
    for model_name, model_builder in model_builders.items():
        print(f"\n🚀 Starting [Scale-Invariant] training (RG Flow Curriculum) - {model_name}...")
        model = train_model_scale_invariant(model_builder(), gen, device)
        print(f"\n🌍 Evaluating {model_name}'s extrapolation ability after scale-invariant training...")
        all_results[model_name] = evaluate_model(model, gen, device, test_depths)

    # Test lengths far beyond the maximum training depth (32).
    print("\n📊 Extrapolation performance comparison (Accuracy):")
    header = "Depth".ljust(8) + "".join(name.rjust(20) for name in model_builders.keys())
    print(header)
    print("-" * len(header))
    for td in test_depths:
        row = str(td).ljust(8)
        for model_name in model_builders.keys():
            row += f"{all_results[model_name][td]:>19.2%} "
        print(row)

if __name__ == "__main__":
    train_scale_invariant()