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
    基于动态门控机制（LSTM）的序列模型
    用于解决 Transformer 在高深度全息数据上的“注意力饱和”问题
    """
    def __init__(
        self, 
        num_nodes: int = 16, 
        d_model: int = 128, 
        num_layers: int = 4,   # 维持一定的物理深度以支持多跳逻辑
        pad_id: int = 0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.pad_id = pad_id
        
        # 词嵌入层（循环神经网络天然具有顺序性，无需显式的位置编码）
        self.embedding = nn.Embedding(num_nodes, d_model, padding_idx=pad_id)
        
        # 核心：使用多层 LSTM 引入动态门控机制 (Dynamic Gating)
        # 遗忘门 (Forget Gate) 会帮助截断无用信息，避免注意力饱和
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
        
        # LSTM 前向传播，自动处理时间序列的显式记忆与遗忘
        # out 包含所有时间步的顶层隐藏状态，(h_n, c_n) 为最后一步的状态
        out, (h_n, c_n) = self.lstm(x_embed)
        
        # 取最后一个时间步的输出进行预测
        last_hidden_state = out[:, -1, :]
        last_hidden_state = self.final_norm(last_hidden_state)
        
        logits = self.fc_out(last_hidden_state)
        return logits
    
