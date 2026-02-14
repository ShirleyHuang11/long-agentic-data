import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from data.pointer import HolographicPointerDataset
from model import HolographicTransformer, SparseHolographicTransformer

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
    
    for depth, loss_vals in results.items():
        plt.plot(loss_vals, label=f'Depth (k) = {depth}', linewidth=2)

    plt.xlabel('Epochs')
    plt.ylabel('Cross Entropy Loss')
    plt.title('Holographic Data: Deep Model Phase Transition')
    plt.axhline(y=2.77, color='r', linestyle='--', alpha=0.3, label='Random Guess Loss')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.savefig('plots/holographic_deep_scaling.png', dpi=300)
    plt.show()


def train_single_model(model, loader, device, epochs=50, lr=2e-3, model_name="Model", depth=None):
    """Train a single model and return loss history"""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss()
    
    losses = []
    model.train()
    
    desc = f"{model_name}" + (f" (k={depth})" if depth is not None else "")
    
    for epoch in tqdm(range(epochs), desc=desc, leave=False):
        epoch_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            
            logits = model(x)
            loss = criterion(logits, y)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)
        scheduler.step(avg_loss)
        
        # Print every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            tqdm.write(f"  Epoch {epoch+1:02d}/{epochs} | Loss: {avg_loss:.4f}")
    
    return losses


def ablation_study(num_nodes=16, path_depths=[8, 32, 128], epochs=50):
    """
    Ablation study comparing multiple model variants:
    1. Standard Transformer (deep, narrow)
    2. Sparse Transformer (Top-K attention)
    3. Sparse Transformer (different top_k values)
    """
    # Automatically select device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Define model variants to test
    model_configs = [
        {
            "name": "Standard Transformer (16 layers)",
            "model_fn": lambda: HolographicTransformer(
                num_nodes=num_nodes,
                d_model=32,
                nhead=4,
                dim_feedforward=64,
                num_layers=16,
                max_seq_len=300
            ).to(device)
        },
        {
            "name": "Sparse Transformer (top_k=2)",
            "model_fn": lambda: SparseHolographicTransformer(
                num_nodes=num_nodes,
                d_model=32,
                nhead=4,
                dim_feedforward=64,
                num_layers=16,
                max_seq_len=300,
                top_k=2
            ).to(device)
        },
        {
            "name": "Sparse Transformer (top_k=4)",
            "model_fn": lambda: SparseHolographicTransformer(
                num_nodes=num_nodes,
                d_model=32,
                nhead=4,
                dim_feedforward=64,
                num_layers=16,
                max_seq_len=300,
                top_k=4
            ).to(device)
        },
        {
            "name": "Sparse Transformer (top_k=8)",
            "model_fn": lambda: SparseHolographicTransformer(
                num_nodes=num_nodes,
                d_model=32,
                nhead=4,
                dim_feedforward=64,
                num_layers=16,
                max_seq_len=300,
                top_k=8
            ).to(device)
        },
    ]
    
    all_results = {}  # {depth: {model_name: [losses]}}
    
    for depth in path_depths:
        print(f"\n[{'='*60}]")
        print(f"Testing at logical depth k = {depth}")
        print(f"[{'='*60}]")
        
        # Prepare data
        dataset = HolographicPointerDataset(
            num_samples=20000, 
            num_nodes=num_nodes, 
            path_length=depth
        )
        loader = DataLoader(dataset, batch_size=128, shuffle=True)
        
        depth_results = {}
        
        for config in model_configs:
            print(f"\n  Training: {config['name']}")
            model = config["model_fn"]()
            
            losses = train_single_model(
                model, loader, device, epochs=epochs,
                model_name=config["name"], depth=depth
            )
            
            depth_results[config["name"]] = losses
            print(f"  ✓ {config['name']}: Final loss = {losses[-1]:.4f}")
        
        all_results[depth] = depth_results
    
    # Print summary table
    print_summary_table(all_results, path_depths)
    
    # Plot results
    plot_ablation_results(all_results, path_depths)
    
    return all_results


def print_summary_table(all_results, path_depths):
    """Print a summary table of final losses for all model variants"""
    print("\n" + "="*80)
    print("ABLATION STUDY SUMMARY")
    print("="*80)
    
    # Get all model names
    model_names = list(all_results[path_depths[0]].keys())
    
    # Print header
    header = f"{'Model':<40} | " + " | ".join([f"k={d:>3}" for d in path_depths])
    print(header)
    print("-" * len(header))
    
    # Print rows
    for model_name in model_names:
        losses_str = " | ".join([
            f"{all_results[depth][model_name][-1]:>6.4f}" 
            for depth in path_depths
        ])
        print(f"{model_name:<40} | {losses_str}")
    
    print("="*80 + "\n")


def plot_ablation_results(all_results, path_depths):
    """Plot ablation study results comparing all model variants"""
    # Define color palette
    ocean_serenity = [
        "#03045e", "#023e8a", "#0077b6", "#0096c7",
        "#00b4d8", "#48cae4", "#90e0ef", "#ade8f4", "#caf0f8"
    ]
    
    # Create subplots for each depth
    fig, axes = plt.subplots(1, len(path_depths), figsize=(6*len(path_depths), 5))
    if len(path_depths) == 1:
        axes = [axes]
    
    for idx, depth in enumerate(path_depths):
        ax = axes[idx]
        depth_results = all_results[depth]
        
        for i, (model_name, losses) in enumerate(depth_results.items()):
            color = ocean_serenity[i % len(ocean_serenity)]
            ax.plot(losses, label=model_name, linewidth=2, color=color)
        
        ax.set_xlabel('Epochs', fontsize=12)
        ax.set_ylabel('Cross Entropy Loss', fontsize=12)
        ax.set_title(f'Depth (k) = {depth}', fontsize=14, fontweight='bold')
        ax.axhline(y=2.77, color='r', linestyle='--', alpha=0.3, label='Random Guess')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/ablation_study.png', dpi=300, bbox_inches='tight')
    print("\nSaved ablation plot to: plots/ablation_study.png")
    plt.show()
    
    # Also create a combined plot showing all depths for each model
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    model_names = list(all_results[path_depths[0]].keys())
    for i, model_name in enumerate(model_names):
        color = ocean_serenity[i % len(ocean_serenity)]
        for depth in path_depths:
            losses = all_results[depth][model_name]
            ax.plot(
                losses, 
                label=f'{model_name} (k={depth})',
                linewidth=2,
                color=color,
                alpha=0.7 if depth != path_depths[-1] else 1.0,
                linestyle='-' if depth == path_depths[0] else '--' if depth == path_depths[1] else ':'
            )
    
    ax.set_xlabel('Epochs', fontsize=12)
    ax.set_ylabel('Cross Entropy Loss', fontsize=12)
    ax.set_title('Ablation Study: All Model Variants Across Depths', fontsize=14, fontweight='bold')
    ax.axhline(y=2.77, color='r', linestyle='--', alpha=0.3, label='Random Guess')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/ablation_study_combined.png', dpi=300, bbox_inches='tight')
    print("Saved combined ablation plot to: plots/ablation_study_combined.png")
    plt.show()


# Run experiment
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "ablation":
        print("Running ablation study")
        ablation_study()
    else:
        train_holographic_experiment_deep()