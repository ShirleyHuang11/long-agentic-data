import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class HolographicPointerDataset(Dataset):
    """
    Enhanced version: Holographic dataset for mapping β-γ phase diagrams 
    and infinite length generalization
    """
    def __init__(self, num_samples=10000, sequence_length=256, path_length=8, 
                 beta=0.0, gamma=0.0):
        self.num_samples = num_samples
        self.seq_len = sequence_length    # Physical sequence length (L) -> true length generalization metric
        self.path_length = path_length    # Logical jump depth (k) -> Epiplexity
        self.beta = beta                  # β: Controls jump locality (larger values favor nearby tokens)
        self.gamma = gamma                # γ: Noise ratio (higher values = more noise, sparser information)
        
        # Vocabulary definition (simplified)
        self.vocab_size = 1000
        self.noise_tokens = list(range(100, self.vocab_size)) 

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 1. Determine number of valid logic nodes (controlled by gamma)
        # When gamma=0, all nodes are valid; larger gamma = fewer valid nodes, more noise
        num_logic_slots = max(self.path_length + 1, int(self.seq_len * (1.0 - self.gamma)))
        
        # 2. Randomly assign positions of logic nodes in the sequence
        logic_positions = sorted(random.sample(range(self.seq_len), num_logic_slots))
        
        # 3. Build true logical chain (acyclic directed graph)
        # Select path_length positions from logic_positions to form main path, 
        # completely preventing "circular" cheating
        path_indices = random.sample(range(num_logic_slots), self.path_length + 1)
        
        # If beta > 0, enforce sequence jumps to have "locality", 
        # simulating natural language proximity correlation
        if self.beta > 0:
            path_indices = sorted(path_indices)
            # Shuffle adjacent nodes probabilistically, smaller beta = more shuffling 
            # (specific beta decay sampling logic omitted here)
        
        path_positions = [logic_positions[i] for i in path_indices]
        
        # 4. Assemble sequence (Sequence Array)
        # Fill with noise by default (background)
        sequence = [random.choice(self.noise_tokens) for _ in range(self.seq_len)]
        
        # Embed logical pointers: A -> B, B -> C
        for i in range(self.path_length):
            curr_pos = path_positions[i]
            next_pos = path_positions[i+1]
            # Encode mapping relationship with special token: e.g., token_id = next_pos (simplified)
            sequence[curr_pos] = next_pos 
            
        start_query = path_positions[0]
        final_target = path_positions[-1]
        
        return torch.tensor(sequence, dtype=torch.long), torch.tensor(start_query, dtype=torch.long), torch.tensor(final_target, dtype=torch.long)