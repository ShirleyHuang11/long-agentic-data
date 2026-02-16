import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from data.pointer import HolographicPointerDataset
from model import (
    HolographicTransformer, 
    # SparseHolographicTransformer,  # Commented out for ablation study
    GatedHolographicNetwork,
    HolographicMamba
)
from utils import ocean_serenity, watermelon_sorbet

def train_holographic_experiment_deep(sequence_length=256, path_depths=[8, 32, 128]):
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
        dataset = HolographicPointerDataset(num_samples=20000, sequence_length=256, path_length=depth)
        loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=8)  # Increase batch_size to speed up training

        # 2. Modify network architecture: depth over width (Depth > Width)
        # According to the paper, processing data with high Epiplexity (logical depth)
        # requires deeper networks to decompress information
        model = HolographicTransformer(
            sequence_length=sequence_length,
            d_model=32,           # Reduce width (originally 64)
            nhead=4,
            dim_feedforward=64,   # Reduce feedforward network dimension
            num_layers=16        # Significantly increase depth (originally 4) -> 
                                   # physically provides sufficient multi-hop reasoning capability
        ).to(device)
        model = torch.compile(model)
        
        optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
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
            for data in tqdm(loader, desc=f"Epoch {epoch+1}", leave=False):
                sequence, start_query, final_target = data
                x = sequence.to(device)
                y = final_target.to(device)
                
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
    sequence_length=256, 
    stages=None,
    lr=2e-3, 
    model_name="Model",
    use_scheduler=True,
    use_wandb=False,
    wandb_config=None,
    beta=0.0,
    gamma=0.0,
    global_batch_size=None,
    device_batch_size=None
):
    """
    Unified curriculum learning function supporting multiple training stages.
    
    Args:
        model: The model to train
        device: Training device
        sequence_length: Sequence length (vocabulary size for models)
        stages: List of stage configs, each as dict with:
            - path_length: k value for this stage
            - epochs: number of epochs for this stage
            - num_samples: number of samples (default: 20000)
            - batch_size: batch size per device (default: 128)
        lr: Learning rate
        model_name: Name for logging
        use_scheduler: Whether to use learning rate scheduler
        global_batch_size: Global batch size across all devices (if None, uses device_batch_size)
        device_batch_size: Batch size per device (if None, uses stage batch_size)
    
    Returns:
        dict: Dictionary with 'losses', 'accuracies', 'optimizer', and 'criterion'
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
    
    # Determine number of devices (for gradient accumulation calculation)
    num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
    
    losses = []
    accuracies = []
    model.train()
    
    total_epochs = sum(stage["epochs"] for stage in stages)
    current_epoch = 0
    
    for stage_idx, stage in enumerate(stages):
        path_length = stage["path_length"]
        stage_epochs = stage["epochs"]
        num_samples = stage.get("num_samples", 20000)
        stage_batch_size = stage.get("batch_size", 128)
        
        # Determine actual batch sizes and gradient accumulation steps
        if device_batch_size is not None:
            actual_device_batch_size = device_batch_size
        else:
            actual_device_batch_size = stage_batch_size
        
        if global_batch_size is not None:
            # Calculate gradient accumulation steps
            effective_batch_size = actual_device_batch_size * num_devices
            num_grad_accum_steps = max(1, global_batch_size // effective_batch_size)
            if global_batch_size % effective_batch_size != 0:
                tqdm.write(f"  Warning: global_batch_size ({global_batch_size}) not divisible by effective_batch_size ({effective_batch_size}). Using {num_grad_accum_steps} accumulation steps.")
        else:
            num_grad_accum_steps = 1
        
        print(f"\n>>> Stage {stage_idx + 1}: Training on k={path_length} (Epochs {current_epoch + 1}-{current_epoch + stage_epochs})")
        if num_grad_accum_steps > 1:
            print(f"  Device batch size: {actual_device_batch_size}, Global batch size: {global_batch_size if global_batch_size else 'N/A'}, Gradient accumulation steps: {num_grad_accum_steps}")
        
        stage_dataset = HolographicPointerDataset(
            num_samples=num_samples,
            sequence_length=256,
            path_length=path_length,
            beta=beta,
            gamma=gamma
        )
        stage_loader = DataLoader(stage_dataset, batch_size=actual_device_batch_size, shuffle=True, num_workers=8)
        
        print(f"  Dataset size: {len(stage_dataset)}, Batches per epoch: {len(stage_loader)}")
        
        for epoch_in_stage in tqdm(range(stage_epochs), desc=f"{model_name} Stage {stage_idx + 1} (k={path_length})", leave=False):
            epoch_loss = 0
            epoch_correct = 0
            epoch_total = 0
            
            optimizer.zero_grad()  # Zero gradients at the start of epoch
            
            for batch_idx, data in enumerate(stage_loader):
                sequence, start_query, final_target = data
                x = sequence.to(device)
                y = final_target.to(device)
                
                logits = model(x)
                loss = criterion(logits, y)
                
                # Scale loss by accumulation steps to maintain gradient magnitude
                loss = loss / num_grad_accum_steps
                loss.backward()
                
                # Calculate accuracy (before scaling)
                _, predicted = torch.max(logits.data, 1)
                epoch_total += y.size(0)
                epoch_correct += (predicted == y).sum().item()
                epoch_loss += loss.item() * num_grad_accum_steps  # Unscale for logging
                
                # Update weights every num_grad_accum_steps
                if (batch_idx + 1) % num_grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
            
            # Handle remaining gradients if batch count is not divisible by num_grad_accum_steps
            if len(stage_loader) % num_grad_accum_steps != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
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
            
            # Log to wandb
            if use_wandb:
                log_dict = {
                    f"{model_name}/loss": avg_loss,
                    f"{model_name}/accuracy": avg_accuracy,
                    f"{model_name}/epoch": current_epoch,
                    f"{model_name}/stage": stage_idx + 1,
                    f"{model_name}/path_length": path_length,
                    f"{model_name}/learning_rate": optimizer.param_groups[0]['lr']
                }
                if wandb_config:
                    log_dict.update({f"{model_name}/{k}": v for k, v in wandb_config.items()})
                wandb.log(log_dict, step=current_epoch)
    
    return {'losses': losses, 'accuracies': accuracies, 'optimizer': optimizer, 'criterion': criterion}


# Backward compatibility: keep old function names as aliases
def train_with_warmup(model, device, sequence_length=256, warmup_epochs=5, warmup_length=4, use_wandb=False, wandb_config=None, model_name="Model"):
    """
    Legacy warmup function - now uses unified curriculum learning.
    Returns optimizer and criterion for backward compatibility.
    
    FIX: Now properly returns the optimizer from curriculum training to preserve
    optimizer state (momentum, Adam's second moments) for true warmup continuation.
    """
    stages = [
        {"path_length": warmup_length, "epochs": warmup_epochs, "num_samples": 5000, "batch_size": 64}
    ]
    results = train_with_curriculum(model, device, sequence_length=sequence_length, stages=stages, lr=1e-3, use_scheduler=False, 
                          use_wandb=use_wandb, wandb_config=wandb_config, model_name=model_name)
    
    # FIX: Reuse the optimizer from curriculum training to preserve state
    optimizer = results['optimizer']
    criterion = results['criterion']
    
    # Optionally adjust learning rate for main training (preserves optimizer state)
    # The caller can modify lr if needed: optimizer.param_groups[0]['lr'] = new_lr
    
    print(">>> Ready for formal training on holographic depth")
    return optimizer, criterion


def train_mamba_with_curriculum(model, device, sequence_length=256, epochs=50, lr=2e-3, model_name="Mamba", use_wandb=False, wandb_config=None, beta=0.0, gamma=0.0, global_batch_size=None, device_batch_size=None):
    """
    Smooth Staircase Curriculum Learning:
    Guide the model to undergo a first-order phase transition through progressive difficulty,
    gradually extracting universal operators from rote memorization.
    """
    if epochs <= 5:
        # Quick test mode
        stages = [
            {"path_length": 8, "epochs": 2, "num_samples": 1000, "batch_size": 64},
            {"path_length": 32, "epochs": 2, "num_samples": 1000, "batch_size": 64},
            {"path_length": 128, "epochs": epochs - 4, "num_samples": 1000, "batch_size": 64}
        ]
    else:
        # Full training mode: 5-stage smooth transition
        stage_epochs = epochs // 5
        rem_epochs = epochs - (stage_epochs * 4)  # Ensure total epochs remain unchanged
        
        stages = [
            {"path_length": 8,   "epochs": stage_epochs, "num_samples": 20000, "batch_size": 128},
            {"path_length": 16,  "epochs": stage_epochs, "num_samples": 20000, "batch_size": 128},
            {"path_length": 32,  "epochs": stage_epochs, "num_samples": 20000, "batch_size": 128},
            {"path_length": 64,  "epochs": stage_epochs, "num_samples": 20000, "batch_size": 128},
            {"path_length": 128, "epochs": rem_epochs,   "num_samples": 20000, "batch_size": 128}
        ]
    return train_with_curriculum(model, device, sequence_length=sequence_length, stages=stages, lr=lr, model_name=model_name, use_wandb=use_wandb, wandb_config=wandb_config, beta=beta, gamma=gamma, global_batch_size=global_batch_size, device_batch_size=device_batch_size)


def train_single_model(model, loader, device, epochs=50, lr=2e-3, model_name="Model", depth=None, use_warmup=False, sequence_length=256, use_wandb=False, wandb_config=None, global_batch_size=None, device_batch_size=None):
    """
    Train a single model and return loss and accuracy history
    
    Args:
        model: Model to train
        loader: DataLoader for training data
        device: Training device
        epochs: Number of training epochs
        lr: Learning rate
        model_name: Name for logging
        depth: Logical depth (for logging)
        use_warmup: Whether to use warmup
        sequence_length: Sequence length
        use_wandb: Whether to use wandb
        wandb_config: Wandb config dict
        global_batch_size: Global batch size across all devices (if None, uses device_batch_size)
        device_batch_size: Batch size per device (if None, uses loader batch_size)
    """
    # Apply warmup if requested
    if use_warmup:
        optimizer, criterion = train_with_warmup(model, device, sequence_length=sequence_length, 
                                                 use_wandb=use_wandb, wandb_config=wandb_config, model_name=model_name)
        # Continue with the warmup optimizer, but adjust learning rate for main training
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Determine number of devices and gradient accumulation steps
    num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
    actual_device_batch_size = device_batch_size if device_batch_size is not None else loader.batch_size
    
    if global_batch_size is not None:
        effective_batch_size = actual_device_batch_size * num_devices
        num_grad_accum_steps = max(1, global_batch_size // effective_batch_size)
        if global_batch_size % effective_batch_size != 0:
            print(f"  Warning: global_batch_size ({global_batch_size}) not divisible by effective_batch_size ({effective_batch_size}). Using {num_grad_accum_steps} accumulation steps.")
    else:
        num_grad_accum_steps = 1
    
    if num_grad_accum_steps > 1:
        print(f"  Device batch size: {actual_device_batch_size}, Global batch size: {global_batch_size if global_batch_size else 'N/A'}, Gradient accumulation steps: {num_grad_accum_steps}")
    
    losses = []
    accuracies = []
    model.train()
    
    desc = f"{model_name}" + (f" (k={depth})" if depth is not None else "")
    
    for epoch in tqdm(range(epochs), desc=desc, leave=False):
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        
        optimizer.zero_grad()  # Zero gradients at the start of epoch
        
        for batch_idx, data in enumerate(loader):
            sequence, start_query, final_target = data
            x = sequence.to(device)
            y = final_target.to(device)
            
            logits = model(x)
            loss = criterion(logits, y)
            
            # Scale loss by accumulation steps to maintain gradient magnitude
            loss = loss / num_grad_accum_steps
            loss.backward()
            
            # Calculate accuracy (before scaling)
            _, predicted = torch.max(logits.data, 1)
            epoch_total += y.size(0)
            epoch_correct += (predicted == y).sum().item()
            epoch_loss += loss.item() * num_grad_accum_steps  # Unscale for logging
            
            # Update weights every num_grad_accum_steps
            if (batch_idx + 1) % num_grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
        
        # Handle remaining gradients if batch count is not divisible by num_grad_accum_steps
        if len(loader) % num_grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        avg_loss = epoch_loss / len(loader)
        avg_accuracy = 100.0 * epoch_correct / epoch_total
        losses.append(avg_loss)
        accuracies.append(avg_accuracy)
        scheduler.step(avg_loss)
        
        # Print progress (more frequent for small epoch counts)
        print_interval = 1 if epochs <= 5 else 10
        if (epoch + 1) % print_interval == 0 or epoch == 0:
            tqdm.write(f"  Epoch {epoch+1:02d}/{epochs} | Loss: {avg_loss:.4f} | Acc: {avg_accuracy:.2f}%")
        
        # Log to wandb
        if use_wandb:
            log_dict = {
                f"{model_name}/loss": avg_loss,
                f"{model_name}/accuracy": avg_accuracy,
                f"{model_name}/epoch": epoch + 1,
                f"{model_name}/learning_rate": optimizer.param_groups[0]['lr']
            }
            if depth is not None:
                log_dict[f"{model_name}/depth"] = depth
            if wandb_config:
                log_dict.update({f"{model_name}/{k}": v for k, v in wandb_config.items()})
            wandb.log(log_dict, step=epoch + 1)
    
    return {'losses': losses, 'accuracies': accuracies}


def ablation_study(sequence_length=256, path_depths=[8, 32, 128], epochs=3, test_generalization=False, use_warmup=False, job_id="default", use_wandb=False, wandb_project="holographic-data", wandb_entity=None, beta=0.0, gamma=0.0, global_batch_size=None, device_batch_size=None):
    """
    Ablation study comparing multiple model variants:
    1. Standard Transformer (deep, narrow)
    2. LSTM-based Gated Network
    3. Mamba (SSM-based)
    
    Args:
        sequence_length: Sequence length (vocabulary size for models)
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
                sequence_length=sequence_length,
                d_model=32,
                nhead=4,
                dim_feedforward=64,
                num_layers=16
            ).to(device)
        },
        {
            "name": f"Mamba (SSM, 8 layers)",
            "model_fn": lambda: HolographicMamba(
                sequence_length=sequence_length,
                d_model=32, # 128,
                d_state=64, # 256,
                num_layers=8, # 16
            ).to(device)
        },
        # {
        #     "name": "LSTM Gated Network (4 layers)",
        #     "model_fn": lambda: GatedHolographicNetwork(
        #         sequence_length=sequence_length,
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
        #         sequence_length=sequence_length,
        #         d_model=32,
        #         nhead=4,
        #         dim_feedforward=64,
        #         num_layers=16,
        #         top_k=2
        #     ).to(device)
        # },
        # {
        #     "name": "Sparse Transformer (top_k=4)",
        #     "model_fn": lambda: SparseHolographicTransformer(
        #         sequence_length=sequence_length,
        #         d_model=32,
        #         nhead=4,
        #         dim_feedforward=64,
        #         num_layers=16,
        #         top_k=4
        #     ).to(device)
        # },
        # {
        #     "name": "Sparse Transformer (top_k=8)",
        #     "model_fn": lambda: SparseHolographicTransformer(
        #         sequence_length=sequence_length,
        #         d_model=32,
        #         nhead=4,
        #         dim_feedforward=64,
        #         num_layers=16,
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
    
    # Initialize wandb if enabled
    if use_wandb:
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=job_id,
            config={
                "sequence_length": sequence_length,
                "path_depths": path_depths,
                "epochs": epochs,
                "test_generalization": test_generalization,
                "use_warmup": use_warmup,
                "job_id": job_id
            }
        )
    
    # Train Mamba with curriculum learning (once, not per depth)
    if mamba_config:
        print(f"\n[{'='*60}]")
        print(f"Training Mamba with Stage-wise Curriculum Learning")
        print(f"[{'='*60}]")
        # Calculate stage epochs for display
        if epochs <= 5:
            # Quick test mode: 3 stages
            print(f"Stage 1: Epochs 1-2 on k=8")
            print(f"Stage 2: Epochs 3-4 on k=32")
            print(f"Stage 3: Epochs 5-{epochs} on k=128")
        else:
            # Full training mode: 5-stage smooth transition
            stage_epochs = epochs // 5
            rem_epochs = epochs - (stage_epochs * 4)
            current_epoch = 1
            print(f"Stage 1: Epochs {current_epoch}-{current_epoch + stage_epochs - 1} on k=8")
            current_epoch += stage_epochs
            print(f"Stage 2: Epochs {current_epoch}-{current_epoch + stage_epochs - 1} on k=16")
            current_epoch += stage_epochs
            print(f"Stage 3: Epochs {current_epoch}-{current_epoch + stage_epochs - 1} on k=32")
            current_epoch += stage_epochs
            print(f"Stage 4: Epochs {current_epoch}-{current_epoch + stage_epochs - 1} on k=64")
            current_epoch += stage_epochs
            print(f"Stage 5: Epochs {current_epoch}-{epochs} on k=128")
        print(f"[{'='*60}]")
        print(f"Epochs: {epochs}")
        
        mamba_model = mamba_config["model_fn"]()
        mamba_model = torch.compile(mamba_model)
        wandb_config_mamba = {"model_type": "Mamba", "training_type": "curriculum"} if use_wandb else None
        mamba_results = train_mamba_with_curriculum(
            mamba_model, device, sequence_length=sequence_length, 
            epochs=epochs, model_name=mamba_config["name"],
            use_wandb=use_wandb, wandb_config=wandb_config_mamba,
            beta=beta, gamma=gamma,
            global_batch_size=global_batch_size, device_batch_size=device_batch_size
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
            test_length_generalization(mamba_model, device, sequence_length=sequence_length)
    
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
            sequence_length=256, 
            path_length=depth,
            beta=beta,
            gamma=gamma
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        
        if depth not in all_results:
            all_results[depth] = {}
        
        # Train Mamba separately on k=32 (if depth is 32) for comparison
        if depth == 32 and mamba_config:
            print(f"\n  Training: {mamba_config['name']} (k=32, standard training)")
            mamba_model_k32 = mamba_config["model_fn"]()
            mamba_model_k32 = torch.compile(mamba_model_k32)
            wandb_config_mamba = {"model_type": "Mamba", "training_type": "standard", "depth": depth} if use_wandb else None
            results = train_single_model(
                mamba_model_k32, loader, device, epochs=epochs,
                model_name=mamba_config["name"], depth=depth,
                use_warmup=use_warmup, sequence_length=sequence_length,
                use_wandb=use_wandb, wandb_config=wandb_config_mamba,
                global_batch_size=global_batch_size, device_batch_size=device_batch_size
            )
            all_results[depth][mamba_config["name"]] = results
            print(f"  ✓ {mamba_config['name']}: Final loss = {results['losses'][-1]:.4f} | Final acc = {results['accuracies'][-1]:.2f}%")
            
            # Test length generalization if requested
            if test_generalization:
                print(f"\n  Testing length generalization for {mamba_config['name']}...")
                test_length_generalization(mamba_model_k32, device, sequence_length=sequence_length)
        
        for config in other_configs:
            print(f"\n  Training: {config['name']}")
            model = config["model_fn"]()
            model = torch.compile(model)
            wandb_config_model = {"model_type": config["name"], "depth": depth} if use_wandb else None
            results = train_single_model(
                model, loader, device, epochs=epochs,
                model_name=config["name"], depth=depth,
                use_warmup=use_warmup, sequence_length=sequence_length,
                use_wandb=use_wandb, wandb_config=wandb_config_model,
                global_batch_size=global_batch_size, device_batch_size=device_batch_size
            )
            
            all_results[depth][config["name"]] = results
            print(f"  ✓ {config['name']}: Final loss = {results['losses'][-1]:.4f} | Final acc = {results['accuracies'][-1]:.2f}%")
            
            # Test length generalization if requested
            if test_generalization:
                print(f"\n  Testing length generalization for {config['name']}...")
                test_length_generalization(model, device, sequence_length=sequence_length)
    
    # Print summary table
    print_summary_table(all_results, path_depths)
    
    # Plot results
    plot_ablation_results(all_results, path_depths, epochs, job_id=job_id)
    
    # Log final results to wandb
    if use_wandb:
        for depth in path_depths:
            if depth in all_results:
                for model_name, results in all_results[depth].items():
                    if isinstance(results, dict):
                        wandb.log({
                            f"final/{model_name}/depth_{depth}/loss": results['losses'][-1],
                            f"final/{model_name}/depth_{depth}/accuracy": results['accuracies'][-1]
                        })
        wandb.finish()
    
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


def test_length_generalization(model, device, sequence_length=256, test_lengths=[32, 128, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072], fixed_path_length=128):
    """
    Validate Section 2.2 of the paper: Operator Isomorphism
    Test the model's logical preservation capability on unseen ultra-long sequences
    
    FIX 3: path_length should be fixed (k), while sequence_length (L) varies
    This tests length generalization: same logical depth k, but longer sequences L
    """
    model.eval()
    print("\n" + "="*50)
    print("LENGTH GENERALIZATION TEST (Zero-shot)")
    print(f"Fixed path_length (k) = {fixed_path_length}, varying sequence_length (L)")
    print("="*50)
    print(f"{'Test Length (L)':<20} | {'Accuracy':<10} | {'Status':<10}")
    print("-" * 50)

    with torch.no_grad():
        for length in test_lengths:
            try:
                # FIX 3: Use fixed path_length, not length
                # This tests if model can generalize to longer sequences with same logical depth
                test_dataset = HolographicPointerDataset(
                    num_samples=1000, 
                    sequence_length=length, 
                    path_length=fixed_path_length  # Fixed k, not length
                )
                test_loader = DataLoader(test_dataset, batch_size=32)
                
                correct = 0
                total = 0
                for data in test_loader:
                    sequence, start_query, final_target = data
                    x = sequence.to(device)
                    y = final_target.to(device)
                    outputs = model(x)
                    
                    # FIX 3: Handle variable output dimensions
                    # If model was trained on sequence_length=256, outputs have 256 classes
                    # But test data might have length > 256, so targets are in [0, length-1]
                    # We need to either:
                    # 1. Truncate targets to [0, sequence_length-1] if length > sequence_length
                    # 2. Or use a model that supports variable output dimensions
                    # For now, we'll truncate: only test if length <= sequence_length
                    if length > sequence_length:
                        # Only test samples where target < sequence_length
                        valid_mask = y < sequence_length
                        if valid_mask.sum() == 0:
                            continue  # Skip this batch if no valid samples
                        y_valid = y[valid_mask]
                        outputs_valid = outputs[valid_mask]
                        _, predicted = torch.max(outputs_valid.data, 1)
                        total += y_valid.size(0)
                        correct += (predicted == y_valid).sum().item()
                    else:
                        _, predicted = torch.max(outputs.data, 1)
                        total += y.size(0)
                        correct += (predicted == y).sum().item()
                
                if total > 0:
                    accuracy = 100 * correct / total
                    status = "OK"
                    print(f"L = {length:<14} | {accuracy:>8.2f}% | {status:<10}")
                else:
                    print(f"L = {length:<14} | {'N/A':>8} | SKIPPED (no valid samples)")
            except Exception as e:
                # Handle models that can't process sequences longer than sequence_length
                status = "FAILED"
                error_msg = str(e)[:30] + "..." if len(str(e)) > 30 else str(e)
                print(f"L = {length:<14} | {'N/A':>8} | {status:<10} ({error_msg})")
    print("="*50)


# Run experiment
if __name__ == "__main__":
    import sys
    # Simple argument parsing
    job_id = None
    use_wandb = False
    wandb_project = "holographic-data"
    wandb_entity = "hanlin-ml"
    
    global_batch_size = None
    device_batch_size = 32
    sequence_length = 128
    
    if len(sys.argv) > 1:
        for i, arg in enumerate(sys.argv):
            if arg == '--job-id' and i + 1 < len(sys.argv):
                job_id = sys.argv[i + 1]
            elif arg == '--wandb':
                use_wandb = True
            elif arg == '--wandb-project' and i + 1 < len(sys.argv):
                wandb_project = sys.argv[i + 1]
            elif arg == '--wandb-entity' and i + 1 < len(sys.argv):
                wandb_entity = sys.argv[i + 1]
            elif arg == '--global-batch-size' and i + 1 < len(sys.argv):
                global_batch_size = int(sys.argv[i + 1])
            elif arg == '--device-batch-size' and i + 1 < len(sys.argv):
                device_batch_size = int(sys.argv[i + 1])
    
    if len(sys.argv) > 1 and sys.argv[1] == "ablation":
        print("Running ablation study")
        if job_id:
            print(f"Job ID: {job_id}")
        if use_wandb:
            print(f"W&B enabled: project={wandb_project}, entity={wandb_entity}")
        ablation_study(sequence_length=sequence_length, epochs=50, test_generalization=True, use_warmup=True, job_id=job_id, 
                      use_wandb=use_wandb, wandb_project=wandb_project, wandb_entity=wandb_entity,
                      global_batch_size=global_batch_size, device_batch_size=device_batch_size)
    else:
        train_holographic_experiment_deep()