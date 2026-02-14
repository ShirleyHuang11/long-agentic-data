import torch
import torch.nn as nn


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
