import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class HolographicPointerDataset(Dataset):
    def __init__(self, num_samples=10000, sequence_length=256, path_length=8, 
                 beta=0.0, gamma=0.0):
        self.num_samples = num_samples
        self.seq_len = sequence_length    
        self.path_length = path_length    
        self.beta = beta                  
        self.gamma = gamma                
        
        # Vocabulary size only needs to be slightly larger than sequence_length
        # 0 is padding, 1 ~ seq_len are position pointers
        self.vocab_size = sequence_length + 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # ==========================================
        # 1. Build base noise filled with high-fidelity distractors
        # ==========================================
        # Critical fix 1: Noise cannot be special tokens, must look exactly like real pointers!
        # This prevents the model from filtering by ID range, forcing it to follow the chain.
        sequence = [random.randint(1, self.seq_len) for _ in range(self.seq_len)]
        
        # ==========================================
        # 2. Select logic chain nodes
        # ==========================================
        # Controlled by gamma: larger gamma means fewer candidate nodes available, rest are dead-end distractors
        num_logic_slots = max(self.path_length + 1, int(self.seq_len * (1.0 - self.gamma)))
        logic_positions = random.sample(range(self.seq_len), num_logic_slots)
        
        # ==========================================
        # 3. Generate real jump path (controlled by Beta)
        # ==========================================
        path_positions = []
        current_pos = random.choice(logic_positions)
        path_positions.append(current_pos)
        
        available_positions = set(logic_positions) - {current_pos}
        
        for _ in range(self.path_length):
            if self.beta == 0:
                # Beta = 0: Pure global random jump (red zone limit)
                next_pos = random.choice(list(available_positions))
            else:
                # Critical fix 2: When Beta > 0, sample with distance-based probability decay, not direct Sort!
                # Closer distances have higher selection probability, simulating natural language locality
                candidates = list(available_positions)
                distances = np.abs(np.array(candidates) - current_pos)
                # Probability inversely proportional to distance raised to (1+beta) power
                probs = 1.0 / (distances ** (1.0 + self.beta))
                probs = probs / np.sum(probs)
                next_pos = np.random.choice(candidates, p=probs)
                
            path_positions.append(next_pos)
            available_positions.remove(next_pos)
            current_pos = next_pos
            
        # ==========================================
        # 4. Embed logic pointers
        # ==========================================
        for i in range(self.path_length):
            curr_node = path_positions[i]
            next_node = path_positions[i+1]
            # Store next node position (offset +1 to avoid 0)
            sequence[curr_node] = next_node + 1 
            
        start_query = path_positions[0] + 1
        final_target = path_positions[-1]  # Target class: 0 ~ seq_len-1
        
        # ==========================================
        # 5. Assemble final input
        # ==========================================
        # Fine-tuning 1: Place Query at the end of the sequence, not the beginning
        # This allows the model to first "read" the entire graph (build internal state),
        # then directly trigger computation when it sees the Query at the end
        # Model input format: [Seq_0] [Seq_1] ... [Seq_L] [Query Token]
        sequence_tensor = torch.tensor(sequence, dtype=torch.long)
        query_tensor = torch.tensor([start_query], dtype=torch.long)
        
        input_seq = torch.cat([sequence_tensor, query_tensor])
        label = torch.tensor(final_target, dtype=torch.long)
        
        return input_seq, label

# # Test print
# dataset = HolographicPointerDataset(num_samples=10, sequence_length=64, path_length=8, beta=0.5)
# input_seq, label = dataset[0]
# print(f"Input sequence length: {len(input_seq)} (64 Graph Nodes + 1 Query)")
# print(f"Query pointer (End): {input_seq[-1]}")
# print(f"Target label (End): {label}")