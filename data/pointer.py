import random
import torch
from torch.utils.data import Dataset, DataLoader

"""
How to use this script to advance experiments?
Verify "Attention and Saturation":
Use this script to generate data to feed to a standard Transformer. You will find that as path_length increases, the Transformer's accuracy will drop dramatically.
When observing the Attention Map, you should see the heatmap become extremely blurred (uniform distribution), which confirms the paper's prediction about Attention Saturation.

Verify "Scaling Law Phase Transition":
Change num_nodes (the number of entities the model needs to remember) and path_length (logical steps).
You will find that increasing logical steps is far more difficult for reducing Loss than increasing the number of entities, which corresponds to the logarithmic scaling curve mentioned in the paper.

Verify "Holographic Generalization":
You can try training on path_length=8 but testing on path_length=64.
If your model architecture (such as the Isometric Lie Flow discussed earlier) is good enough, it should exhibit scale invariance, i.e., logical depth does not affect its accuracy.
"""

class HolographicPointerDataset(Dataset):
    """
    Generate holographic pointer chasing data.
    Goal: The model must find the final stopping point through a series of jumps.
    Sequence format: [starting point] [jump table (node:target)] [query point] -> [result]
    """
    def __init__(self, num_samples=10000, num_nodes=16, path_length=8):
        self.num_samples = num_samples
        self.num_nodes = num_nodes      # Logical breadth (n)
        self.path_length = path_length  # Logical depth (k) -> Core parameter controlling holographic concentration

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 1. Create a randomly permuted jump table (Permutation Table)
        # Each node points to another unique node, ensuring the path doesn't break easily
        nodes = list(range(self.num_nodes))
        targets = list(range(self.num_nodes))
        random.shuffle(targets)
        mapping = {n: t for n, t in zip(nodes, targets)}

        # 2. Randomly select starting point and trace path
        start_node = random.choice(nodes)
        current = start_node
        path = [start_node]
        for _ in range(self.path_length):
            current = mapping[current]
            path.append(current)
        
        final_target = path[-1]

        # 3. Construct holographic sequence (Holographic Sequence)
        # Serialize jump table: (0, targets[0]), (1, targets[1])...
        table_tokens = []
        for n in nodes:
            table_tokens.extend([n, mapping[n]])
        
        # Final input: [jump table] + [starting query point]
        # Label: [final result]
        input_seq = torch.tensor(table_tokens + [start_node], dtype=torch.long)
        label = torch.tensor(final_target, dtype=torch.long)

        return input_seq, label

# --- Experimental Configuration Suggestions ---

# 1. Simulate "holographic limit state": short window, extremely high depth
# With a window length of only 33 (16*2 + 1), require the model to handle logical chasing with depth 128
dense_config = {
    "num_nodes": 16,    # Smaller state space
    "path_length": 128  # Extremely deep logical nesting, far exceeding window length
}

# 2. Instantiate DataLoaders
dataset = HolographicPointerDataset(num_samples=50000, **dense_config)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Test print
input_sample, label_sample = dataset[0]
print(f"Input sequence length: {len(input_sample)}")
print(f"Logical jump depth: {dense_config['path_length']}")
print(f"Input example (first 10 tokens): {input_sample[:10]}")
print(f"Target label (final pointer position): {label_sample}")

# --- Advanced Experiment: Verify Length Generalization (RG Flow) ---
def evaluate_length_generalization(model, max_depth=1000):
    """
    Verification logic: Can a model trained at depth 128 directly predict successfully at depth 1000?
    """
    for d in [128, 256, 512, 1000]:
        test_dataset = HolographicPointerDataset(num_samples=1000, num_nodes=16, path_length=d)
        # Run model inference here and calculate accuracy
        print(f"Testing generalization ability at depth {d}...")