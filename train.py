import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from data.pointer import HolographicPointerDataset
from model import (
    HolographicTransformer, 
    # SparseHolographicTransformer,  # Commented out for ablation study
    GatedHolographicNetwork,
    HolographicMamba
)
from utils import ocean_serenity, watermelon_sorbet

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
    
    # Use palette from utils (every other color for cleaner look)
    sns.set_palette(ocean_serenity[::-2])
    
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


def train_with_warmup(model, device, num_nodes=16, warmup_epochs=5, warmup_length=4):
    """
    Curriculum learning strategy:
    Step 1: Warm-up on very short logical chains (k=4) to let the model learn how to 'gate' information
    Step 2: Switch to holographic logical chains (k=32/128) for formal training
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # --- Step 1: Warm-up (simple logic alignment) ---
    print("\n>>> Stage 1: Warm-up on simple logic (k={})".format(warmup_length))
    warmup_dataset = HolographicPointerDataset(
        num_samples=5000, 
        num_nodes=num_nodes, 
        path_length=warmup_length
    )
    loader = DataLoader(warmup_dataset, batch_size=64, shuffle=True)
    
    model.train()
    for epoch in range(warmup_epochs):
        epoch_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(loader)
        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"  Warm-up Epoch {epoch+1}/{warmup_epochs} | Loss: {avg_loss:.4f}")
    
    print(">>> Stage 2: Ready for formal training on holographic depth")
    return optimizer, criterion


def train_single_model(model, loader, device, epochs=50, lr=2e-3, model_name="Model", depth=None, use_warmup=False, num_nodes=16):
    """Train a single model and return loss history"""
    # Apply warmup if requested
    if use_warmup:
        optimizer, criterion = train_with_warmup(model, device, num_nodes=num_nodes)
        # Continue with the warmup optimizer, but adjust learning rate for main training
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
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


def ablation_study(num_nodes=16, path_depths=[8, 32, 128], epochs=50, test_generalization=False, use_warmup=False):
    """
    Ablation study comparing multiple model variants:
    1. Standard Transformer (deep, narrow)
    2. LSTM-based Gated Network
    3. Mamba (SSM-based)
    
    Args:
        num_nodes: Number of nodes in the graph
        path_depths: List of logical depths to test
        epochs: Number of training epochs
        test_generalization: If True, test length generalization after training
        use_warmup: If True, use curriculum learning with warmup on short sequences
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
            "name": "Mamba (SSM, 6 layers)",
            "model_fn": lambda: HolographicMamba(
                num_nodes=num_nodes,
                d_model=64,
                d_state=128,
                num_layers=6
            ).to(device)
        },
        # {
        #     "name": "LSTM Gated Network (4 layers)",
        #     "model_fn": lambda: GatedHolographicNetwork(
        #         num_nodes=num_nodes,
        #         d_model=128,
        #         num_layers=4,
        #         pad_id=0,
        #         dropout=0.1
        #     ).to(device)
        # },
        # Sparse Transformer variants commented out for this ablation study
        # {
        #     "name": "Sparse Transformer (top_k=2)",
        #     "model_fn": lambda: SparseHolographicTransformer(
        #         num_nodes=num_nodes,
        #         d_model=32,
        #         nhead=4,
        #         dim_feedforward=64,
        #         num_layers=16,
        #         max_seq_len=300,
        #         top_k=2
        #     ).to(device)
        # },
        # {
        #     "name": "Sparse Transformer (top_k=4)",
        #     "model_fn": lambda: SparseHolographicTransformer(
        #         num_nodes=num_nodes,
        #         d_model=32,
        #         nhead=4,
        #         dim_feedforward=64,
        #         num_layers=16,
        #         max_seq_len=300,
        #         top_k=4
        #     ).to(device)
        # },
        # {
        #     "name": "Sparse Transformer (top_k=8)",
        #     "model_fn": lambda: SparseHolographicTransformer(
        #         num_nodes=num_nodes,
        #         d_model=32,
        #         nhead=4,
        #         dim_feedforward=64,
        #         num_layers=16,
        #         max_seq_len=300,
        #         top_k=8
        #     ).to(device)
        # },
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
                model_name=config["name"], depth=depth,
                use_warmup=use_warmup, num_nodes=num_nodes
            )
            
            depth_results[config["name"]] = losses
            print(f"  ✓ {config['name']}: Final loss = {losses[-1]:.4f}")
            
            # Test length generalization if requested
            if test_generalization:
                print(f"\n  Testing length generalization for {config['name']}...")
                test_length_generalization(model, device, num_nodes=num_nodes)
        
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
    # Create subplots for each depth
    fig, axes = plt.subplots(1, len(path_depths), figsize=(6*len(path_depths), 5))
    if len(path_depths) == 1:
        axes = [axes]
    
    for idx, depth in enumerate(path_depths):
        ax = axes[idx]
        depth_results = all_results[depth]
        
        for i, (model_name, losses) in enumerate(depth_results.items()):
            color = watermelon_sorbet[i % len(watermelon_sorbet)]
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
        color = watermelon_sorbet[i % len(watermelon_sorbet)]
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


def test_length_generalization(model, device, num_nodes=16, test_lengths=[32, 128, 512, 1024]):
    """
    Validate Section 2.2 of the paper: Operator Isomorphism
    Test the model's logical preservation capability on unseen ultra-long sequences
    """
    model.eval()
    print("\n" + "="*50)
    print("LENGTH GENERALIZATION TEST (Zero-shot)")
    print("="*50)
    print(f"{'Test Length (L)':<20} | {'Accuracy':<10} | {'Status':<10}")
    print("-" * 50)

    with torch.no_grad():
        for length in test_lengths:
            try:
                # Generate ultra-long test data
                test_dataset = HolographicPointerDataset(
                    num_samples=1000, 
                    num_nodes=num_nodes, 
                    path_length=length
                )
                test_loader = DataLoader(test_dataset, batch_size=32)
                
                correct = 0
                total = 0
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    outputs = model(x)
                    _, predicted = torch.max(outputs.data, 1)
                    total += y.size(0)
                    correct += (predicted == y).sum().item()
                
                accuracy = 100 * correct / total
                status = "OK"
                print(f"L = {length:<14} | {accuracy:>8.2f}% | {status:<10}")
            except Exception as e:
                # Handle models that can't process sequences longer than max_seq_len
                status = "FAILED"
                error_msg = str(e)[:30] + "..." if len(str(e)) > 30 else str(e)
                print(f"L = {length:<14} | {'N/A':>8} | {status:<10} ({error_msg})")
    print("="*50)


# Run experiment
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "ablation":
            print("Running ablation study")
            # Set test_generalization=True to test length generalization after training
            # Set use_warmup=True to use curriculum learning with warmup
            ablation_study(test_generalization=True, use_warmup=False)
    else:
        train_holographic_experiment_deep()