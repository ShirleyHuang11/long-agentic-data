import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from fbm import FBM

# ==========================================
# 1. Data generation: long-sequence windows for Autoencoder dataset
# ==========================================
def generate_ae_dataset(h_value, total_length=50000, window_size=64, test_split=0.2):
    """
    Generate a long fBM sequence, split into windows of window_size for autoencoder reconstruction.
    This ensures the model sees the global fractal structure of the sequence, not just next-step prediction.
    """
    print(f"Generating fBM data for H = {h_value} (Total length: {total_length})...")
    # Generate the full long sequence
    f = FBM(n=total_length, hurst=h_value, length=1, method='daviesharte')
    time_series = f.fbm()
    
    # Global standardization
    time_series = (time_series - np.mean(time_series)) / np.std(time_series)
    
    # Slice into batch-sized windows
    data = []
    # Use non-overlapping or moderately overlapping windows
    step = window_size // 2 
    for i in range(0, len(time_series) - window_size, step):
        data.append(time_series[i : i + window_size])
        
    data = np.array(data)
    
    # Split into Train / Test set (important for Scaling Law: measures generalization/reconstruction)
    split_idx = int(len(data) * (1 - test_split))
    train_data = torch.tensor(data[:split_idx], dtype=torch.float32)
    test_data = torch.tensor(data[split_idx:], dtype=torch.float32)
    
    return train_data, test_data

# ==========================================
# 2. Model definition: scalable Autoencoder
# ==========================================
class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, bottleneck_size):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, bottleneck_size)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def get_param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ==========================================
# 3. Training and evaluation: focus on Test Loss
# ==========================================
def train_and_evaluate_ae(train_tensor, test_tensor, hidden_size, bottleneck_size, epochs=50, lr=0.005, batch_size=256):
    input_size = train_tensor.shape[1]
    model = Autoencoder(input_size, hidden_size, bottleneck_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_loader = DataLoader(TensorDataset(train_tensor, train_tensor), batch_size=batch_size, shuffle=True)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        for batch_x, _ in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_x)
            loss.backward()
            optimizer.step()
            
    # Evaluate Test Loss (main metric for Scaling Law)
    model.eval()
    with torch.no_grad():
        test_outputs = model(test_tensor)
        test_loss = criterion(test_outputs, test_tensor).item()
        
    return model.get_param_count(), test_loss

# ==========================================
# 4. Main experiment: test scaling behavior across different H
# ==========================================
def main():
    hurst_exponents = [0.2, 0.5, 0.8]
    window_size = 64
    
    # Control network scale (hidden and bottleneck widths scale proportionally)
    # Scale multipliers
    scales = [1, 2, 4, 8, 16]
    base_hidden = 16
    base_bottleneck = 4
    
    results = {h: {'N': [], 'Test_Loss': []} for h in hurst_exponents}
    
    for h in hurst_exponents:
        train_data, test_data = generate_ae_dataset(h_value=h, total_length=50000, window_size=window_size)
        
        print(f"\nTraining Autoencoders for H = {h}...")
        for scale in scales:
            h_size = base_hidden * scale
            b_size = base_bottleneck * scale
            
            # If bottleneck >= input size, the map becomes identity and compression is lost; cap it
            if b_size >= window_size:
                b_size = window_size // 2 
                
            params, test_loss = train_and_evaluate_ae(
                train_data, test_data, 
                hidden_size=h_size, 
                bottleneck_size=b_size,
                epochs=60  # Ensure the model converges sufficiently
            )
            
            results[h]['N'].append(params)
            results[h]['Test_Loss'].append(test_loss)
            print(f"  Params: {params:^6} | Test MSE: {test_loss:.5f}")
            
    # ==========================================
    # 5. Visualize the scaling law
    # ==========================================
    plt.figure(figsize=(10, 6))
    
    colors = {0.2: 'red', 0.5: 'green', 0.8: 'blue'}
    
    for h in hurst_exponents:
        N_vals = np.array(results[h]['N'])
        Loss_vals = np.array(results[h]['Test_Loss'])
        
        log_N = np.log10(N_vals)
        log_Loss = np.log10(Loss_vals)
        
        # Plot points and connecting lines
        plt.plot(log_N, log_Loss, marker='o', color=colors[h], label=f'Hurst (H) = {h}')
        
        # Fit scaling exponent alpha: Log(L) = -alpha * Log(N) + C
        # (Often one fits mid-to-large N to avoid small-network bias; here we fit all points)
        slope, intercept = np.polyfit(log_N, log_Loss, 1)
        alpha = -slope
        print(f"==> For H={h}, Scaling Exponent (alpha) = {alpha:.4f}")
        
        # Plot fitted dashed line
        plt.plot(log_N, slope * log_N + intercept, linestyle='--', color=colors[h], alpha=0.5)

    plt.title('Autoencoder Scaling Law vs. Intrinsic Dimension (via Hurst)', fontsize=14)
    plt.xlabel('Log10 (Parameter Count N)', fontsize=12)
    plt.ylabel('Log10 (Test MSE Loss)', fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots/mlp_hurst.png')
    plt.show()

if __name__ == "__main__":
    main()