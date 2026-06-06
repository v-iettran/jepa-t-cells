from __future__ import annotations

import torch
from torch import nn


class JEPAPredictor(nn.Module):
    """Predict target-gene latents from context tokens and target gene queries."""

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.query_type = nn.Parameter(torch.zeros(1, 1, d_model))
        self.norm = nn.LayerNorm(d_model)
        nn.init.trunc_normal_(self.query_type, std=0.02)

    def forward(self, context_tokens: torch.Tensor, target_gene_embeddings: torch.Tensor) -> torch.Tensor:
        if target_gene_embeddings.dim() != 2:
            raise ValueError("target_gene_embeddings must have shape [n_targets, d_model]")
        queries = target_gene_embeddings.unsqueeze(0).expand(context_tokens.shape[0], -1, -1)
        queries = queries + self.query_type
        tokens = torch.cat([context_tokens, queries], dim=1)
        out = self.transformer(tokens)
        pred = out[:, -queries.shape[1] :, :]
        return self.norm(pred)
