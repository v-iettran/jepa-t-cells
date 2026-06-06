from __future__ import annotations

import torch
from torch import nn


class GeneTokenEncoder(nn.Module):
    """Transformer encoder over gene-expression tokens."""

    def __init__(
        self,
        n_genes: int,
        n_batches: int,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_genes = n_genes
        self.d_model = d_model
        self.gene_embedding = nn.Embedding(n_genes, d_model)
        self.batch_embedding = nn.Embedding(max(1, n_batches), d_model)
        self.value_mlp = nn.Sequential(
            nn.Linear(1, max(16, d_model // 4)),
            nn.GELU(),
            nn.Linear(max(16, d_model // 4), d_model),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(
        self,
        values: torch.Tensor,
        batch_id: torch.Tensor | None = None,
        gene_idx_subset: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode selected genes.

        Args:
            values: Tensor of shape [batch, n_genes].
            batch_id: Tensor of shape [batch].
            gene_idx_subset: Optional gene indices to encode. If omitted, all genes
                in vocabulary order are encoded.
        """

        if gene_idx_subset is None:
            gene_idx_subset = torch.arange(self.n_genes, device=values.device)
        gene_idx_subset = gene_idx_subset.to(values.device)
        x = values.index_select(dim=1, index=gene_idx_subset).unsqueeze(-1)
        value_tokens = self.value_mlp(x)
        gene_tokens = self.gene_embedding(gene_idx_subset).unsqueeze(0).expand(values.shape[0], -1, -1)
        tokens = value_tokens + gene_tokens
        if batch_id is not None:
            batch_tokens = self.batch_embedding(batch_id.clamp_min(0)).unsqueeze(1)
            tokens = tokens + batch_tokens
        cls = self.cls_token.expand(values.shape[0], -1, -1)
        out = torch.cat([cls, tokens], dim=1)
        out = self.encoder(out)
        return self.norm(out)
