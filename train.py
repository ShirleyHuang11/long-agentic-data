import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
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


def train_with_curriculum(
    model, 
    device, 
    num_nodes=16, 
    stages=None,
    lr=2e-3, 
    model_name="Model",
    use_scheduler=True
):
    """
    Unified curriculum learning function supporting multiple training stages.
    
    Args:
        model: The model to train
        device: Training device
        num_nodes: Number of nodes in the graph
        stages: List of stage configs, each as dict with:
            - path_length: k value for this stage
            - epochs: number of epochs for this stage
            - num_samples: number of samples (default: 20000)
            - batch_size: batch size (default: 128)
        lr: Learning rate
        model_name: Name for logging
        use_scheduler: Whether to use learning rate scheduler
    
    Returns:
        dict: Dictionary with 'losses' and 'accuracies' lists
    """
    # Default stages for Mamba (if not provided)
    if stages is None:
        stages = [
            {"path_length": 8, "epochs": 20, 
             "num_samples": 20000, "batch_size": 128},
            {"path_length": 128, "epochs": 30,
             "num_samples": 20000, "batch_size": 128}
        ]
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5) if use_scheduler else None
    
    losses = []
    accuracies = []
    model.train()
    
    total_epochs = sum(stage["epochs"] for stage in stages)
    current_epoch = 0
    
    for stage_idx, stage in enumerate(stages):
        path_length = stage["path_length"]
        stage_epochs = stage["epochs"]
        num_samples = stage.get("num_samples", 20000)
        batch_size = stage.get("batch_size", 128)
        
        print(f"\n>>> Stage {stage_idx + 1}: Training on k={path_length} (Epochs {current_epoch + 1}-{current_epoch + stage_epochs})")
        
        stage_dataset = HolographicPointerDataset(
            num_samples=num_samples,
            num_nodes=num_nodes,
            path_length=path_length
        )
        stage_loader = DataLoader(stage_dataset, batch_size=batch_size, shuffle=True)
        
        for epoch_in_stage in tqdm(range(stage_epochs), desc=f"{model_name} Stage {stage_idx + 1} (k={path_length})", leave=False):
            epoch_loss = 0
            epoch_correct = 0
            epoch_total = 0
            
            for x, y in stage_loader:
                x, y = x.to(device), y.to(device)
                
                logits = model(x)
                loss = criterion(logits, y)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                # Calculate accuracy
                _, predicted = torch.max(logits.data, 1)
                epoch_total += y.size(0)
                epoch_correct += (predicted == y).sum().item()
            
            avg_loss = epoch_loss / len(stage_loader)
            avg_accuracy = 100.0 * epoch_correct / epoch_total
            losses.append(avg_loss)
            accuracies.append(avg_accuracy)
            
            if scheduler:
                scheduler.step(avg_loss)
            
            # Print progress (more frequent for small epoch counts)
            current_epoch += 1
            print_interval = 1 if total_epochs <= 5 else 5
            if (epoch_in_stage + 1) % print_interval == 0 or epoch_in_stage == 0:
                lr_str = f" | LR: {optimizer.param_groups[0]['lr']:.6f}" if scheduler else ""
                tqdm.write(f"  Stage {stage_idx + 1} Epoch {current_epoch:02d}/{total_epochs} | Loss: {avg_loss:.4f} | Acc: {avg_accuracy:.2f}%{lr_str}")
    
    return {'losses': losses, 'accuracies': accuracies}


# Backward compatibility: keep old function names as aliases
def train_with_warmup(model, device, num_nodes=16, warmup_epochs=5, warmup_length=4):
    """
    Legacy warmup function - now uses unified curriculum learning.
    Returns optimizer and criterion for backward compatibility.
    """
    stages = [
        {"path_length": warmup_length, "epochs": warmup_epochs, "num_samples": 5000, "batch_size": 64}
    ]
    train_with_curriculum(model, device, num_nodes=num_nodes, stages=stages, lr=1e-3, use_scheduler=False)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    print(">>> Ready for formal training on holographic depth")
    return optimizer, criterion


def train_mamba_with_curriculum(model, device, num_nodes=16, epochs=50, lr=2e-3, model_name="Mamba"):
    """
    Stage-wise curriculum training for Mamba (uses unified curriculum function):
    - Epoch 1-20: Train on k=8 data (let Mamba master basic jumps)
    - Epoch 21-50: Switch to k=128 (solve SSM's "search for logical chains" problem)
    """
    # For testing: use smaller epochs if total epochs is small
    if epochs <= 5:
        # Quick test: 2 epochs on k=8, 1 epoch on k=128
        stage1_epochs = max(2, epochs // 2)
        stage2_epochs = epochs - stage1_epochs
        stages = [
            {"path_length": 8, "epochs": stage1_epochs, 
             "num_samples": 1000, "batch_size": 64},  # Reduced data for speed
            {"path_length": 128, "epochs": stage2_epochs, 
             "num_samples": 1000, "batch_size": 64}
        ]
    else:
        # Normal training
        stage1_epochs = max(20, int(epochs * 0.4))
        stage2_epochs = epochs - stage1_epochs
        stages = [
            {"path_length": 8, "epochs": stage1_epochs, 
             "num_samples": 20000, "batch_size": 128},
            {"path_length": 128, "epochs": stage2_epochs, 
             "num_samples": 20000, "batch_size": 128}
        ]
    return train_with_curriculum(model, device, num_nodes=num_nodes, stages=stages, lr=lr, model_name=model_name)


def train_single_model(model, loader, device, epochs=50, lr=2e-3, model_name="Model", depth=None, use_warmup=False, num_nodes=16):
    """Train a single model and return loss and accuracy history"""
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
    accuracies = []
    model.train()
    
    desc = f"{model_name}" + (f" (k={depth})" if depth is not None else "")
    
    for epoch in tqdm(range(epochs), desc=desc, leave=False):
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            
            logits = model(x)
            loss = criterion(logits, y)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            # Calculate accuracy
            _, predicted = torch.max(logits.data, 1)
            epoch_total += y.size(0)
            epoch_correct += (predicted == y).sum().item()
        
        avg_loss = epoch_loss / len(loader)
        avg_accuracy = 100.0 * epoch_correct / epoch_total
        losses.append(avg_loss)
        accuracies.append(avg_accuracy)
        scheduler.step(avg_loss)
        
        # Print progress (more frequent for small epoch counts)
        print_interval = 1 if epochs <= 5 else 10
        if (epoch + 1) % print_interval == 0 or epoch == 0:
            tqdm.write(f"  Epoch {epoch+1:02d}/{epochs} | Loss: {avg_loss:.4f} | Acc: {avg_accuracy:.2f}%")
    
    return {'losses': losses, 'accuracies': accuracies}


def ablation_study(num_nodes=16, path_depths=[8, 32, 128], epochs=3, test_generalization=False, use_warmup=False, job_id="default"):
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
            "name": "Mamba (SSM, 16 layers)",
            "model_fn": lambda: HolographicMamba(
                num_nodes=num_nodes,
                d_model=64,
                d_state=256,
                num_layers=16
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
    
    all_results = {}  # {depth: {model_name: {'losses': [...], 'accuracies': [...]}}}
    
    # Separate Mamba training with curriculum learning
    mamba_config = None
    other_configs = []
    for config in model_configs:
        if "Mamba" in config["name"]:
            mamba_config = config
        else:
            other_configs.append(config)
    
    # Train Mamba with curriculum learning (once, not per depth)
    if mamba_config:
        print(f"\n[{'='*60}]")
        print(f"Training Mamba with Stage-wise Curriculum Learning")
        print(f"[{'='*60}]")
        # Calculate stage epochs for display
        if epochs <= 5:
            stage1_epochs_display = max(2, epochs // 2)
            stage2_epochs_display = epochs - stage1_epochs_display
            print(f"Stage 1: Epochs 1-{stage1_epochs_display} on k=8 (basic jumps)")
            print(f"Stage 2: Epochs {stage1_epochs_display+1}-{epochs} on k=128 (logical chains)")
        else:
            print(f"Stage 1: Epochs 1-20 on k=8 (basic jumps)")
            print(f"Stage 2: Epochs 21-{epochs} on k=128 (logical chains)")
        print(f"[{'='*60}]")
        print(f"Epochs: {epochs}")
        
        mamba_model = mamba_config["model_fn"]()
        mamba_results = train_mamba_with_curriculum(
            mamba_model, device, num_nodes=num_nodes, 
            epochs=epochs, model_name=mamba_config["name"]
        )
        mamba_losses = mamba_results['losses']
        mamba_accuracies = mamba_results['accuracies']
        print(f"  ✓ {mamba_config['name']}: Final loss = {mamba_losses[-1]:.4f} | Final acc = {mamba_accuracies[-1]:.2f}%")
        
        # Store Mamba results: split curriculum learning into stages
        # Stage 1: k=8, Stage 2: k=128
        # Determine stage1_epochs based on epochs
        if epochs <= 5:
            # Quick test: stage1 gets most epochs
            stage1_epochs = max(2, epochs // 2)
        else:
            # Normal training: fixed split at 20 epochs
            stage1_epochs = 20
        stage1_losses = mamba_losses[:stage1_epochs]
        stage1_accuracies = mamba_accuracies[:stage1_epochs]
        stage2_losses = mamba_losses[stage1_epochs:]
        stage2_accuracies = mamba_accuracies[stage1_epochs:]
        
        # Store Stage 1 results for k=8 (only if not empty)
        if 8 not in all_results:
            all_results[8] = {}
        if len(stage1_losses) > 0:
            all_results[8][mamba_config["name"]] = {
                'losses': stage1_losses,
                'accuracies': stage1_accuracies
            }
        
        # Store Stage 2 results for k=128 (only if not empty)
        if 128 not in all_results:
            all_results[128] = {}
        if len(stage2_losses) > 0:
            all_results[128][mamba_config["name"]] = {
                'losses': stage2_losses,
                'accuracies': stage2_accuracies
            }
        
        # Also store full curriculum learning curve for visualization
        if "curriculum" not in all_results:
            all_results["curriculum"] = {}
        all_results["curriculum"][mamba_config["name"]] = {
            'losses': mamba_losses,
            'accuracies': mamba_accuracies
        }
        
        # Test length generalization if requested
        if test_generalization:
            print(f"\n  Testing length generalization for {mamba_config['name']}...")
            test_length_generalization(mamba_model, device, num_nodes=num_nodes)
    
    # Train other models per depth (standard training)
    for depth in path_depths:
        print(f"\n[{'='*60}]")
        print(f"Testing at logical depth k = {depth}")
        print(f"[{'='*60}]")
        
        # Prepare data (reduce samples for quick testing)
        num_samples = 1000 if epochs <= 5 else 20000
        batch_size = 64 if epochs <= 5 else 128
        dataset = HolographicPointerDataset(
            num_samples=num_samples, 
            num_nodes=num_nodes, 
            path_length=depth
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        if depth not in all_results:
            all_results[depth] = {}
        
        # Train Mamba separately on k=32 (if depth is 32) for comparison
        if depth == 32 and mamba_config:
            print(f"\n  Training: {mamba_config['name']} (k=32, standard training)")
            mamba_model_k32 = mamba_config["model_fn"]()
            results = train_single_model(
                mamba_model_k32, loader, device, epochs=epochs,
                model_name=mamba_config["name"], depth=depth,
                use_warmup=use_warmup, num_nodes=num_nodes
            )
            all_results[depth][mamba_config["name"]] = results
            print(f"  ✓ {mamba_config['name']}: Final loss = {results['losses'][-1]:.4f} | Final acc = {results['accuracies'][-1]:.2f}%")
            
            # Test length generalization if requested
            if test_generalization:
                print(f"\n  Testing length generalization for {mamba_config['name']}...")
                test_length_generalization(mamba_model_k32, device, num_nodes=num_nodes)
        
        for config in other_configs:
            print(f"\n  Training: {config['name']}")
            model = config["model_fn"]()
            
            results = train_single_model(
                model, loader, device, epochs=epochs,
                model_name=config["name"], depth=depth,
                use_warmup=use_warmup, num_nodes=num_nodes
            )
            
            all_results[depth][config["name"]] = results
            print(f"  ✓ {config['name']}: Final loss = {results['losses'][-1]:.4f} | Final acc = {results['accuracies'][-1]:.2f}%")
            
            # Test length generalization if requested
            if test_generalization:
                print(f"\n  Testing length generalization for {config['name']}...")
                test_length_generalization(model, device, num_nodes=num_nodes)
    
    # Print summary table
    print_summary_table(all_results, path_depths)
    
    # Plot results
    plot_ablation_results(all_results, path_depths, epochs, job_id=job_id)
    
    return all_results


def print_summary_table(all_results, path_depths):
    """Print a summary table of final losses and accuracies for all model variants"""
    print("\n" + "="*100)
    print("ABLATION STUDY SUMMARY")
    print("="*100)
    
    # Get all model names from all depths (excluding curriculum key)
    all_model_names = set()
    for depth in all_results.keys():
        if depth != "curriculum":
            all_model_names.update(all_results[depth].keys())
    model_names = sorted(list(all_model_names))
    
    # Print Loss table
    print("\nLOSS:")
    header = f"{'Model':<40} | " + " | ".join([f"k={d:>3}" for d in path_depths])
    print(header)
    print("-" * len(header))
    
    for model_name in model_names:
        losses_str = " | ".join([
            f"{all_results[depth][model_name]['losses'][-1]:>6.4f}" if depth in all_results and model_name in all_results[depth] else "  N/A  "
            for depth in path_depths
        ])
        print(f"{model_name:<40} | {losses_str}")
    
    # Print Accuracy table
    print("\nACCURACY (%):")
    print(header)
    print("-" * len(header))
    
    for model_name in model_names:
        accuracies_str = " | ".join([
            f"{all_results[depth][model_name]['accuracies'][-1]:>6.2f}" if depth in all_results and model_name in all_results[depth] else "  N/A  "
            for depth in path_depths
        ])
        print(f"{model_name:<40} | {accuracies_str}")
    
    # Print curriculum learning note if applicable
    if "curriculum" in all_results:
        print("\nNote: Models with curriculum learning show Stage 1 (k=8) and Stage 2 (k=128) results separately.")
    
    print("="*100 + "\n")


def plot_ablation_results(all_results, path_depths, epochs=50, job_id="default"):
    """Plot ablation study results comparing all model variants with both loss and accuracy"""
    # Filter out "curriculum" key which is used for full curriculum visualization
    available_depths = sorted([d for d in all_results.keys() if d in path_depths and d != "curriculum"])
    
    # Create subplots for each depth: 2 rows (loss and accuracy) x N columns (depths)
    if len(available_depths) > 0:
        fig, axes = plt.subplots(2, len(available_depths), figsize=(6*len(available_depths), 10))
        if len(available_depths) == 1:
            axes = axes.reshape(2, 1)
        
        for idx, depth in enumerate(available_depths):
            depth_results = all_results[depth]
            
            # Plot Loss
            ax_loss = axes[0, idx]
            for i, (model_name, results) in enumerate(depth_results.items()):
                if isinstance(results, dict) and len(results.get('losses', [])) > 0:
                    losses = results['losses']
                    color = watermelon_sorbet[i % len(watermelon_sorbet)]
                    ax_loss.plot(range(len(losses)), losses, label=model_name, linewidth=2, color=color, marker='o', markersize=3)
            
            ax_loss.set_xlabel('Epochs', fontsize=12)
            ax_loss.set_ylabel('Cross Entropy Loss', fontsize=12)
            ax_loss.set_title(f'Loss - Depth (k) = {depth}', fontsize=14, fontweight='bold')
            ax_loss.axhline(y=2.77, color='r', linestyle='--', alpha=0.3, label='Random Guess')
            ax_loss.legend(fontsize=9)
            ax_loss.grid(True, alpha=0.3)
            
            # Plot Accuracy
            ax_acc = axes[1, idx]
            for i, (model_name, results) in enumerate(depth_results.items()):
                if isinstance(results, dict) and len(results.get('accuracies', [])) > 0:
                    accuracies = results['accuracies']
                    color = watermelon_sorbet[i % len(watermelon_sorbet)]
                    ax_acc.plot(range(len(accuracies)), accuracies, label=model_name, linewidth=2, color=color, marker='o', markersize=3)
            
            ax_acc.set_xlabel('Epochs', fontsize=12)
            ax_acc.set_ylabel('Accuracy (%)', fontsize=12)
            ax_acc.set_title(f'Accuracy - Depth (k) = {depth}', fontsize=14, fontweight='bold')
            ax_acc.legend(fontsize=9)
            ax_acc.grid(True, alpha=0.3)
            
            # Set x-axis limits and integer ticks
            if len(depth_results) > 0:
                all_losses = [results.get('losses', []) for results in depth_results.values() if isinstance(results, dict)]
                all_losses = [l for l in all_losses if len(l) > 0]
                if len(all_losses) > 0:
                    max_epochs = max(len(losses) for losses in all_losses)
                    if max_epochs > 0:
                        ax_loss.set_xlim(-0.5, max_epochs - 0.5)
                        ax_acc.set_xlim(-0.5, max_epochs - 0.5)
                        # Set integer ticks only
                        ax_loss.xaxis.set_major_locator(MaxNLocator(integer=True))
                        ax_acc.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        plt.tight_layout()
        filename = f'plots/{job_id}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\nSaved ablation plot to: {filename}")
        plt.show()
    
    # Also create a combined plot showing all depths for each model (both loss and accuracy)
    fig, (ax_loss, ax_acc) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Get all model names from all depths (excluding curriculum key)
    all_model_names = set()
    for depth in all_results.keys():
        if depth != "curriculum":
            all_model_names.update(all_results[depth].keys())
    model_names = sorted(list(all_model_names))
    
    # Plot standard models (non-curriculum) - Loss
    for i, model_name in enumerate(model_names):
        color = watermelon_sorbet[i % len(watermelon_sorbet)]
        for depth in sorted([d for d in all_results.keys() if d != "curriculum"]):
            if model_name in all_results[depth]:
                results = all_results[depth][model_name]
                if isinstance(results, dict) and len(results.get('losses', [])) > 0:
                    losses = results['losses']
                    ax_loss.plot(
                        range(len(losses)), losses,
                        label=f'{model_name} (k={depth})',
                        linewidth=2,
                        color=color,
                        alpha=0.6,
                        linestyle='-' if depth == min([d for d in all_results.keys() if d != "curriculum"]) else '--' if len([d for d in all_results.keys() if d != "curriculum"]) > 1 and depth == sorted([d for d in all_results.keys() if d != "curriculum"])[1] else ':'
                    )
    
    # Plot curriculum learning models with stage markers - Loss
    if "curriculum" in all_results:
        for model_name, full_results in all_results["curriculum"].items():
            if isinstance(full_results, dict) and len(full_results.get('losses', [])) > 0:
                full_losses = full_results['losses']
                # Find the color for this model
                model_idx = model_names.index(model_name) if model_name in model_names else 0
                color = watermelon_sorbet[model_idx % len(watermelon_sorbet)]
                
                # Plot full curriculum learning curve
                ax_loss.plot(
                    range(len(full_losses)), full_losses,
                    label=f'{model_name} (Curriculum: k=8→128)',
                    linewidth=3,
                    color=color,
                    alpha=1.0,
                    linestyle='-'
                )
                
                # Mark stage transition point
                stage1_epochs = 20
                if len(full_losses) > stage1_epochs and stage1_epochs > 0:
                    transition_x = stage1_epochs - 1  # 0-indexed
                    ax_loss.axvline(x=transition_x, color=color, linestyle=':', alpha=0.5, linewidth=1.5)
                    if transition_x < len(full_losses):
                        ax_loss.text(transition_x, full_losses[transition_x] + 0.05, 'k=8→128', 
                               fontsize=8, color=color, ha='center', 
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    ax_loss.set_xlabel('Epochs', fontsize=12)
    ax_loss.set_ylabel('Cross Entropy Loss', fontsize=12)
    ax_loss.set_title('Ablation Study: All Model Variants Across Depths (Loss)', fontsize=14, fontweight='bold')
    ax_loss.axhline(y=2.77, color='r', linestyle='--', alpha=0.3, label='Random Guess')
    ax_loss.legend(fontsize=8, ncol=2, loc='upper right')
    ax_loss.grid(True, alpha=0.3)
    # Set integer ticks only
    ax_loss.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Plot standard models (non-curriculum) - Accuracy
    for i, model_name in enumerate(model_names):
        color = watermelon_sorbet[i % len(watermelon_sorbet)]
        for depth in sorted([d for d in all_results.keys() if d != "curriculum"]):
            if model_name in all_results[depth]:
                results = all_results[depth][model_name]
                if isinstance(results, dict) and len(results.get('accuracies', [])) > 0:
                    accuracies = results['accuracies']
                    ax_acc.plot(
                        range(len(accuracies)), accuracies,
                        label=f'{model_name} (k={depth})',
                        linewidth=2,
                        color=color,
                        alpha=0.6,
                        linestyle='-' if depth == min([d for d in all_results.keys() if d != "curriculum"]) else '--' if len([d for d in all_results.keys() if d != "curriculum"]) > 1 and depth == sorted([d for d in all_results.keys() if d != "curriculum"])[1] else ':'
                    )
    
    # Plot curriculum learning models - Accuracy
    if "curriculum" in all_results:
        for model_name, full_results in all_results["curriculum"].items():
            if isinstance(full_results, dict) and len(full_results.get('accuracies', [])) > 0:
                full_accuracies = full_results['accuracies']
                model_idx = model_names.index(model_name) if model_name in model_names else 0
                color = watermelon_sorbet[model_idx % len(watermelon_sorbet)]
                
                ax_acc.plot(
                    range(len(full_accuracies)), full_accuracies,
                    label=f'{model_name} (Curriculum: k=8→128)',
                    linewidth=3,
                    color=color,
                    alpha=1.0,
                    linestyle='-'
                )
                
                # Mark stage transition point
                stage1_epochs = 20
                if len(full_accuracies) > stage1_epochs and stage1_epochs > 0:
                    transition_x = stage1_epochs - 1
                    ax_acc.axvline(x=transition_x, color=color, linestyle=':', alpha=0.5, linewidth=1.5)
                    if transition_x < len(full_accuracies):
                        ax_acc.text(transition_x, full_accuracies[transition_x] + 1, 'k=8→128', 
                               fontsize=8, color=color, ha='center', 
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    ax_acc.set_xlabel('Epochs', fontsize=12)
    ax_acc.set_ylabel('Accuracy (%)', fontsize=12)
    ax_acc.set_title('Ablation Study: All Model Variants Across Depths (Accuracy)', fontsize=14, fontweight='bold')
    ax_acc.legend(fontsize=8, ncol=2, loc='lower right')
    ax_acc.grid(True, alpha=0.3)
    # Set integer ticks only
    ax_acc.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Set x-axis limits
    max_epochs = 0
    for depth in sorted([d for d in all_results.keys() if d != "curriculum"]):
        for model_name in model_names:
            if model_name in all_results[depth]:
                results = all_results[depth][model_name]
                if isinstance(results, dict):
                    if len(results.get('losses', [])) > 0:
                        max_epochs = max(max_epochs, len(results['losses']))
                    if len(results.get('accuracies', [])) > 0:
                        max_epochs = max(max_epochs, len(results['accuracies']))
    if "curriculum" in all_results:
        for model_name, full_results in all_results["curriculum"].items():
            if isinstance(full_results, dict):
                if len(full_results.get('losses', [])) > 0:
                    max_epochs = max(max_epochs, len(full_results['losses']))
                if len(full_results.get('accuracies', [])) > 0:
                    max_epochs = max(max_epochs, len(full_results['accuracies']))
    if max_epochs > 0:
        ax_loss.set_xlim(-0.5, max_epochs - 0.5)
        ax_acc.set_xlim(-0.5, max_epochs - 0.5)
    
    plt.tight_layout()
    filename = f'plots/{job_id}_combined.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved combined ablation plot to: {filename}")
    plt.show()


def test_length_generalization(model, device, num_nodes=16, test_lengths=[32, 128, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]):
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
    # Simple argument parsing for job_id
    job_id = None
    if len(sys.argv) > 1:
        for i, arg in enumerate(sys.argv):
            if arg == '--job-id' and i + 1 < len(sys.argv):
                job_id = sys.argv[i + 1]
                break
    
    if len(sys.argv) > 1 and sys.argv[1] == "ablation":
        print("Running ablation study (quick test with 3 epochs)")
        if job_id:
            print(f"Job ID: {job_id}")
        # Quick test: 3 epochs, no generalization test, no warmup for speed
        ablation_study(epochs=3, test_generalization=True, use_warmup=True, job_id=job_id)
    else:
        train_holographic_experiment_deep()