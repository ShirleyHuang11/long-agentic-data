import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from data.pointer import HolographicPointerDataset
from model import HolographicTransformer

def train_holographic_experiment_deep(num_nodes=16, path_depths=[8, 32, 128]):
    """
    Modified experiment: significantly increase model depth, extend training time,
    to find the 'first-order phase transition' point
    """
    results = {}
    
    # Automatically select device (GPU/MPS/CPU), deep networks require hardware acceleration
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    for depth in path_depths:
        print(f"\n[{'='*40}]")
        print(f"Starting deep experiment: logical depth k = {depth}")
        
        # 1. Prepare data: keep data volume large enough to trigger phase transition
        dataset = HolographicPointerDataset(num_samples=20000, num_nodes=num_nodes, path_length=depth)
        loader = DataLoader(dataset, batch_size=128, shuffle=True)  # Increase batch_size to speed up training

        # 2. Modify network architecture: depth over width (Depth > Width)
        # According to the paper, processing data with high Epiplexity (logical depth)
        # requires deeper networks to decompress information
        model = HolographicTransformer(
            num_nodes=num_nodes,
            d_model=32,           # Reduce width (originally 64)
            nhead=4,
            dim_feedforward=64,   # Reduce feedforward network dimension
            num_layers=16,        # Significantly increase depth (originally 4) -> 
                                   # physically provides sufficient multi-hop reasoning capability
            max_seq_len=300       # Ensure it can accommodate sequence length for depth=128
        ).to(device)
        
        optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        # Introduce scheduler: reduce learning rate after plateau to help model 
        # stabilize convergence after phase transition point
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        criterion = nn.CrossEntropyLoss()

        # 3. Training loop: significantly increase epochs to observe "first-order phase transition"
        losses = []
        model.train()
        epochs = 50  # Extend training time to wait for phase transition to occur
        
        for epoch in tqdm(range(epochs), desc=f"Training (depth={depth})"):
            epoch_loss = 0
            for x, y in tqdm(loader, desc=f"Epoch {epoch+1}", leave=False):
                x, y = x.to(device), y.to(device)
                
                logits = model(x) 
                loss = criterion(logits, y)
                
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping: prevent gradient explosion in deep networks during phase transition
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(loader)
            losses.append(avg_loss)
            scheduler.step(avg_loss)
            
            # Only print key information logs to keep terminal clean
            if (epoch + 1) % 5 == 0 or epoch == 0:
                tqdm.write(f"Epoch {epoch+1:02d}/{epochs} | Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        results[depth] = losses

    # 4. Plot analysis chart
    plt.figure(figsize=(10, 6))
    for depth, loss_vals in results.items():
        plt.plot(loss_vals, label=f'Depth (k) = {depth}', linewidth=2)
    
        # Define the palette by hex codes
    ocean_serenity = [
        "#03045e",
        "#023e8a",
        "#0077b6",
        "#0096c7",
        "#00b4d8",
        "#48cae4",
        "#90e0ef",
        "#ade8f4",
        "#caf0f8"
    ][::-2]

    sns.set_palette(ocean_serenity)

    plt.xlabel('Epochs')
    plt.ylabel('Cross Entropy Loss')
    plt.title('Holographic Data: Deep Model Phase Transition')
    plt.axhline(y=2.77, color='r', linestyle='--', alpha=0.3, label='Random Guess Loss')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.savefig('plots/holographic_deep_scaling.png', dpi=300)
    plt.show()


# Run experiment
if __name__ == "__main__":
    train_holographic_experiment_deep()