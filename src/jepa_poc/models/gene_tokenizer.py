"""Experiment 3 gene tokenization: frozen ESM2 identity + expression.

Replaces A0's from-scratch ``nn.Embedding`` gene identity with a frozen ESM2
protein embedding projected to an identity sub-dim, concatenated with A0's
expression encoder output, then LayerNorm'd:

    t[g,c] = LayerNorm( concat[ h_id[g] , indicator[g] , h_expr[g,c] ] )

with ``d_id_proj + use_indicator + d_expr == d_model`` so the transformer width
matches A0 exactly. Genes with no canonical protein use a shared learned
``h_empty`` vector plus a binary fallback indicator. No positional encoding is
added (genes are an unordered set; ESM identity is the position signal). The A0
batch embedding is retained so batch handling is unchanged from A0.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class ESMGeneTokenEncoder(nn.Module):
    """Transformer encoder over ESM2-identity + expression gene tokens."""

    def __init__(
        self,
        esm_embeddings: np.ndarray | torch.Tensor,
        fallback_mask: np.ndarray | torch.Tensor,
        n_batches: int,
        d_model: int = 256,
        d_id_proj: int = 191,
        d_expr: int = 64,
        use_fallback_indicator: bool = True,
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
        use_grad_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.use_grad_checkpointing = use_grad_checkpointing
        esm = torch.as_tensor(np.asarray(esm_embeddings), dtype=torch.float32)
        fb = torch.as_tensor(np.asarray(fallback_mask)).bool()
        self.n_genes = int(esm.shape[0])
        self.d_esm = int(esm.shape[1])
        self.d_model = d_model
        self.d_id_proj = d_id_proj
        self.d_expr = d_expr
        self.use_fallback_indicator = use_fallback_indicator

        ind = 1 if use_fallback_indicator else 0
        if d_id_proj + ind + d_expr != d_model:
            raise ValueError(
                f"d_id_proj({d_id_proj}) + indicator({ind}) + d_expr({d_expr}) "
                f"must equal d_model({d_model})"
            )
        self.d_id = d_id_proj + ind  # full identity sub-embedding width

        # Frozen ESM2 table (not a Parameter -> never updated).
        self.register_buffer("esm_table", esm)
        self.register_buffer("fallback_mask", fb)

        # Trainable identity projection (only W_id, b_id are learned).
        self.id_proj = nn.Linear(self.d_esm, d_id_proj)
        # Shared learned embedding for genes with no canonical protein.
        self.h_empty = nn.Parameter(torch.zeros(d_id_proj))
        nn.init.trunc_normal_(self.h_empty, std=0.02)

        # A0's expression encoder: 1 -> max(16, d_model/4) -> d_expr, GELU.
        hidden = max(16, d_model // 4)
        self.value_mlp = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_expr),
        )

        self.token_norm = nn.LayerNorm(d_model)
        self.batch_embedding = nn.Embedding(max(1, n_batches), d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Per-gene identity query for the predictor (cell-agnostic), d_id -> d_model.
        self.query_proj = nn.Linear(self.d_id, d_model)

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

    def _identity(self, gene_idx: torch.Tensor) -> torch.Tensor:
        """Identity sub-embedding h_id for genes ``gene_idx`` -> [k, d_id]."""
        esm = self.esm_table.index_select(0, gene_idx)  # [k, d_esm]
        proj = self.id_proj(esm)  # [k, d_id_proj]
        fb = self.fallback_mask.index_select(0, gene_idx)  # [k]
        proj = torch.where(fb.unsqueeze(-1), self.h_empty.to(proj.dtype), proj)
        if self.use_fallback_indicator:
            indicator = fb.to(proj.dtype).unsqueeze(-1)  # [k, 1]
            return torch.cat([proj, indicator], dim=-1)  # [k, d_id]
        return proj

    def gene_query(self, gene_idx: torch.Tensor) -> torch.Tensor:
        """Predictor query token per gene -> [k, d_model]."""
        gene_idx = gene_idx.to(self.esm_table.device)
        return self.query_proj(self._identity(gene_idx))

    def forward(
        self,
        values: torch.Tensor,
        batch_id: torch.Tensor | None = None,
        gene_idx_subset: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if gene_idx_subset is None:
            gene_idx_subset = torch.arange(self.n_genes, device=values.device)
        gene_idx_subset = gene_idx_subset.to(values.device)
        b = values.shape[0]
        k = gene_idx_subset.shape[0]

        x = values.index_select(dim=1, index=gene_idx_subset).unsqueeze(-1)  # [b, k, 1]
        h_expr = self.value_mlp(x)  # [b, k, d_expr]

        h_id = self._identity(gene_idx_subset)  # [k, d_id]
        h_id = h_id.unsqueeze(0).expand(b, -1, -1)  # [b, k, d_id]

        token = torch.cat([h_id, h_expr], dim=-1)  # [b, k, d_model]
        token = self.token_norm(token)

        if batch_id is not None:
            batch_tok = self.batch_embedding(batch_id.clamp_min(0)).unsqueeze(1)  # [b,1,d]
            token = token + batch_tok

        cls = self.cls_token.expand(b, -1, -1)
        out = torch.cat([cls, token], dim=1)  # [b, 1+k, d]
        out = self._run_encoder(out)
        return self.norm(out)

    def _run_encoder(self, out: torch.Tensor) -> torch.Tensor:
        """Transformer encoder, optionally with per-layer gradient checkpointing.

        Checkpointing recomputes each layer's activations in the backward pass
        instead of storing them, which roughly halves activation memory and lets
        Arm 2's symmetric (full-gene, with-grad) target view run at batch 512 on a
        single 48 GB GPU. It is a no-op at inference (no grad) and numerically
        identical to the plain path (use_reentrant=False preserves dropout RNG)."""
        if not (self.use_grad_checkpointing and self.training and torch.is_grad_enabled()):
            return self.encoder(out)
        from torch.utils.checkpoint import checkpoint

        for layer in self.encoder.layers:
            out = checkpoint(layer, out, use_reentrant=False)
        if self.encoder.norm is not None:
            out = self.encoder.norm(out)
        return out
