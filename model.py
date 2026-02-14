import torch
import torch.nn as nn
import torch.nn.functional as F

class HolographicTransformer(nn.Module):
    """
    Mini Transformer for processing holographic pointer data.

    Features:
    - Token embedding + learned positional embedding
    - TransformerEncoder stack (batch_first)
    - Supports variable-length sequences via padding mask
    - Optional causal (autoregressive) attention mask
    - Predicts next node (or last-step class) using the last token representation
    """

    def __init__(
        self,
        num_nodes: int = 16,
        d_model: int = 64,
        nhead: int = 4,
        dim_feedforward: int = 128,
        num_layers: int = 4,
        max_seq_len: int = 50,
        pad_id: int = 0,
        causal: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be > 0")
        if not (0 <= pad_id < num_nodes):
            raise ValueError("pad_id must be in [0, num_nodes)")

        self.num_nodes = num_nodes
        self.pad_id = pad_id
        self.causal = causal
        self.max_seq_len = max_seq_len

        self.embedding = nn.Embedding(num_nodes, d_model, padding_idx=pad_id)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.final_norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, num_nodes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: LongTensor [batch, seq_len] containing node indices (with pad_id for padding)

        Returns:
            logits: FloatTensor [batch, num_nodes]
        """
        if x.dim() != 2:
            raise ValueError(f"x must be 2D [batch, seq_len], got shape {tuple(x.shape)}")
        if x.dtype != torch.long:
            x = x.long()

        B, L = x.shape
        if L > self.max_seq_len:
            raise ValueError(
                f"seq_len ({L}) exceeds max_seq_len ({self.max_seq_len}). "
                f"Increase max_seq_len or truncate inputs."
            )

        # Positions: [batch, seq_len]
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)

        # Embeddings: [batch, seq_len, d_model]
        x_embed = self.embedding(x) + self.pos_embedding(positions)

        # Padding mask: True means "ignore"
        pad_mask = (x == self.pad_id)  # [batch, seq_len]

        # Optional causal mask (prevents attending to future tokens)
        attn_mask = None
        if self.causal:
            # [seq_len, seq_len] True = masked
            attn_mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)

        # Transformer output: [batch, seq_len, d_model]
        out = self.transformer(x_embed, mask=attn_mask, src_key_padding_mask=pad_mask)
        out = self.final_norm(out)

        # Use last token representation for prediction: [batch, num_nodes]
        logits = self.fc_out(out[:, -1, :])
        return logits

class TopKAttention(nn.Module):
    """
    Custom Top-K sparse attention mechanism
    Forces truncation of long-tail attention weights to prevent "attention saturation"
    in long-sequence holographic data
    """
    def __init__(self, d_model, nhead, top_k=4):
        super().__init__()
        self.nhead = nhead
        self.top_k = top_k
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)
        self.scale = (d_model // nhead) ** -0.5

    def forward(self, x, pad_mask=None):
        B, L, D = x.shape
        H = self.nhead
        HD = D // H

        # Linear projection and split Q, K, V
        qkv = self.qkv(x).reshape(B, L, 3, H, HD).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # [B, H, L, HD]

        # Compute raw attention scores
        scores = (q @ k.transpose(-2, -1)) * self.scale # [B, H, L, L]

        # Handle Padding Mask
        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        # === Core modification: Top-K sparsification ===
        # Prevent Softmax from becoming uniformly smooth, force attention to focus
        # on the most relevant few tokens
        if self.top_k < L:
            # Find the top K values in the last dimension (sequence length)
            topk_vals, _ = torch.topk(scores, self.top_k, dim=-1)
            # Get the K-th largest threshold
            kth_vals = topk_vals[..., -1:]
            # Set all scores below threshold to -inf, making them 0 after Softmax
            scores = torch.where(scores >= kth_vals, scores, torch.full_like(scores, float('-inf')))

        # Compute sparse attention weights
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.out(out)


class SparseTransformerLayer(nn.Module):
    """Encoder layer with integrated Top-K attention"""
    def __init__(self, d_model, nhead, dim_feedforward, top_k=4, dropout=0.1):
        super().__init__()
        self.self_attn = TopKAttention(d_model, nhead, top_k)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, pad_mask=None):
        # Pre-LN structure is more beneficial for deep network training
        src2 = self.norm1(src)
        src = src + self.dropout1(self.self_attn(src2, pad_mask=pad_mask))
        src2 = self.norm2(src)
        src = src + self.dropout2(self.linear2(self.dropout(F.relu(self.linear1(src2)))))
        return src


class SparseHolographicTransformer(nn.Module):
    """
    Replacement for original model: Holographic Transformer using sparse attention mechanism
    """
    def __init__(
        self, num_nodes=16, d_model=64, nhead=4, dim_feedforward=128, 
        num_layers=8, max_seq_len=300, pad_id=0, top_k=4
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.pad_id = pad_id
        
        self.embedding = nn.Embedding(num_nodes, d_model, padding_idx=pad_id)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Stack custom sparse attention layers
        self.layers = nn.ModuleList([
            SparseTransformerLayer(d_model, nhead, dim_feedforward, top_k)
            for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, num_nodes)

    def forward(self, x):
        B, L = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        x_embed = self.embedding(x) + self.pos_embedding(positions)
        pad_mask = (x == self.pad_id)

        out = x_embed
        for layer in self.layers:
            out = layer(out, pad_mask=pad_mask)
            
        out = self.final_norm(out)
        return self.fc_out(out[:, -1, :])

class GatedHolographicNetwork(nn.Module):
    """
    Sequence model based on dynamic gating mechanism (LSTM)
    Used to solve the "attention saturation" problem of Transformers on high-depth holographic data
    """
    def __init__(
        self, 
        num_nodes: int = 16, 
        d_model: int = 128, 
        num_layers: int = 4,   # Maintain sufficient physical depth to support multi-hop reasoning
        pad_id: int = 0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.pad_id = pad_id
        
        # Token embedding layer (RNNs naturally have sequential order, no explicit positional encoding needed)
        self.embedding = nn.Embedding(num_nodes, d_model, padding_idx=pad_id)
        
        # Core: use multi-layer LSTM to introduce dynamic gating mechanism (Dynamic Gating)
        # Forget Gate helps truncate useless information and avoid attention saturation
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.final_norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, num_nodes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: LongTensor [batch, seq_len]
        """
        # [batch, seq_len, d_model]
        x_embed = self.embedding(x)
        
        # LSTM forward pass, automatically handles explicit memory and forgetting in time series
        # out contains all timesteps' top-layer hidden states, (h_n, c_n) is the final timestep state
        out, (h_n, c_n) = self.lstm(x_embed)
        
        # Take the last timestep output for prediction
        last_hidden_state = out[:, -1, :]
        last_hidden_state = self.final_norm(last_hidden_state)
        
        logits = self.fc_out(last_hidden_state)
        return logits

class SimplifiedSelectiveSSM(nn.Module):
    """
    Minimal PyTorch implementation of Mamba (Selective State Space Model) core layer.
    Implements input-dependent sequence state tracking through dynamic gating (delta, B, C).
    """
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # State transition matrix A (in Mamba usually fixed or slowly varying)
        # For simplicity, we make it learnable here
        self.A = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
        
        # Input-dependent projections (core of Selective Mechanism)
        self.proj_B = nn.Linear(d_model, d_state)
        self.proj_C = nn.Linear(d_model, d_state)
        self.proj_delta = nn.Linear(d_model, d_model)
        
        # Output mixing projection
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        B_batch, L, D = x.shape
        
        # 1. Compute input-dependent parameters (Selective Gating)
        # Softplus ensures timestep delta is positive
        delta = F.softplus(self.proj_delta(x)) # [Batch, L, d_model]
        mat_B = self.proj_B(x)                 # [Batch, L, d_state]
        mat_C = self.proj_C(x)                 # [Batch, L, d_state]
        
        # Initialize hidden state
        h = torch.zeros(B_batch, D, self.d_state, device=x.device)
        outputs = []
        
        # 2. Discretization of continuous state space and recurrence (RNN-like unfolding)
        # Note: pure PyTorch loops are slower on long sequences, but sufficient for validating theoretical logic
        for t in range(L):
            dt = delta[:, t, :].unsqueeze(-1)          # [Batch, d_model, 1]
            
            # Discretize A and B (using simplified exponential approximation here)
            A_bar = torch.exp(-dt * torch.exp(self.A.unsqueeze(0))) # [Batch, d_model, d_state]
            B_bar = dt * mat_B[:, t, :].unsqueeze(1)                # [Batch, d_model, d_state]
            
            x_t = x[:, t, :].unsqueeze(-1)             # [Batch, d_model, 1]
            
            # State update equation: h_t = A_bar * h_{t-1} + B_bar * x_t
            h = A_bar * h + B_bar * x_t                # [Batch, d_model, d_state]
            
            # Output equation: y_t = C_t * h_t
            C_t = mat_C[:, t, :].unsqueeze(1)          # [Batch, 1, d_state]
            y_t = torch.sum(h * C_t, dim=-1)           # [Batch, d_model]
            
            outputs.append(y_t)
            
        y = torch.stack(outputs, dim=1) # [Batch, L, d_model]
        return self.out_proj(y)


class SSMBlock(nn.Module):
    """Standard SSM block with normalization and feedforward network"""
    def __init__(self, d_model, d_state=16, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.ssm = SimplifiedSelectiveSSM(d_model, d_state)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(), # Mamba prefers SiLU/Swish
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Similar to Transformer's Pre-LN structure
        x = x + self.ssm(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class HolographicMamba(nn.Module):
    """
    Holographic reasoning network based on Selective State Space Model (SSM)
    Completely abandons the Attention mechanism
    """
    def __init__(self, num_nodes=16, d_model=64, d_state=16, num_layers=6):
        super().__init__()
        self.num_nodes = num_nodes
        self.embedding = nn.Embedding(num_nodes, d_model)
        
        # Stack SSM blocks
        self.layers = nn.ModuleList([
            SSMBlock(d_model, d_state) for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, num_nodes)

    def forward(self, x):
        # Note: SSM is inherently a directional sequence model, no Positional Embedding needed
        out = self.embedding(x)
        
        for layer in self.layers:
            out = layer(out)
            
        out = self.final_norm(out)
        
        # Use the last token's hidden state for prediction
        return self.fc_out(out[:, -1, :])