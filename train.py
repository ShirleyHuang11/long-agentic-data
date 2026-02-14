import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from data.pointer import HolographicPointerDataset
from model import HolographicTransformer

def train_holographic_experiment(num_nodes=16, path_depths=[8, 32, 128]):
    """
    Compare training performance under different logical depths (holographic concentration)
    """
    results = {}

    for depth in path_depths:
        print(f"\nStarting experiment: logical depth k = {depth}")
        
        # 1. Prepare data
        dataset = HolographicPointerDataset(num_samples=20000, num_nodes=num_nodes, path_length=depth)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        # 2. Define mini Transformer
        model = HolographicTransformer(num_nodes=num_nodes)
        
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        # 3. Training loop
        losses = []
        model.train()
        for epoch in tqdm(range(10), desc=f"Training (depth={depth})"):  # Train 10 epochs to observe trends
            epoch_loss = 0
            for x, y in tqdm(loader, desc=f"Epoch {epoch+1}", leave=False):
                logits = model(x) 
                
                loss = criterion(logits, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(loader)
            losses.append(avg_loss)
            tqdm.write(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
        
        results[depth] = losses

    # 4. Plot analysis
    plt.figure(figsize=(10, 6))
    for depth, loss_vals in results.items():
        plt.plot(loss_vals, label=f'Depth (k) = {depth}')
    
    plt.xlabel('Epochs')
    plt.ylabel('Cross Entropy Loss')
    plt.title('Holographic Data Scaling: Impact of Logical Depth')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/holographic_data_scaling.png')
    plt.show()

# Run experiment
train_holographic_experiment()