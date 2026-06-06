from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from jepa_poc.data.masking import masks_to_torch, sample_masks
from jepa_poc.models.encoder import GeneTokenEncoder
from jepa_poc.models.predictor import JEPAPredictor


@dataclass
class JEPALossOutput:
    loss: torch.Tensor
    prediction_loss: torch.Tensor
    variance_loss: torch.Tensor
    target_std: torch.Tensor


class JEPA(nn.Module):
    """I-JEPA-style model with a context encoder, EMA target encoder, and predictor."""

    def __init__(
        self,
        n_genes: int,
        n_batches: int,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        predictor_layers: int = 2,
        dropout: float = 0.1,
        mask_context_frac: float = 0.30,
        n_target_blocks: int = 2,
        target_block_frac: float = 0.20,
        vicreg_weight: float = 0.01,
    ) -> None:
        super().__init__()
        self.n_genes = n_genes
        self.mask_context_frac = mask_context_frac
        self.n_target_blocks = n_target_blocks
        self.target_block_frac = target_block_frac
        self.vicreg_weight = vicreg_weight
        self.context_encoder = GeneTokenEncoder(n_genes, n_batches, d_model, n_layers, n_heads, dropout)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        self.predictor = JEPAPredictor(d_model, predictor_layers, n_heads, dropout)

    @torch.no_grad()
    def update_target_encoder(self, momentum: float) -> None:
        for target_param, context_param in zip(self.target_encoder.parameters(), self.context_encoder.parameters(), strict=True):
            target_param.data.mul_(momentum).add_(context_param.data, alpha=1.0 - momentum)

    def _make_masks(self, device: torch.device) -> tuple[torch.Tensor, list[torch.Tensor]]:
        mask = sample_masks(
            n_genes=self.n_genes,
            ctx_frac=self.mask_context_frac,
            n_target_blocks=self.n_target_blocks,
            target_frac=self.target_block_frac,
        )
        return masks_to_torch(mask, device=device)

    def forward(
        self,
        values: torch.Tensor,
        batch_id: torch.Tensor | None = None,
        context_idx: torch.Tensor | None = None,
        target_blocks: list[torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        if context_idx is None or target_blocks is None:
            context_idx, target_blocks = self._make_masks(values.device)
        context_tokens = self.context_encoder(values, batch_id, context_idx)
        with torch.no_grad():
            full_target_tokens = self.target_encoder(values, batch_id)
        predictions: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        for block in target_blocks:
            block = block.to(values.device)
            target_gene_embeddings = self.context_encoder.gene_embedding(block)
            pred = self.predictor(context_tokens, target_gene_embeddings)
            target = full_target_tokens.index_select(dim=1, index=block + 1)
            predictions.append(pred)
            targets.append(target.detach())
        return {
            "predictions": predictions,
            "targets": targets,
            "context_tokens": context_tokens,
            "target_tokens": full_target_tokens,
        }

    def compute_loss(self, outputs: dict[str, torch.Tensor | list[torch.Tensor]]) -> JEPALossOutput:
        predictions = outputs["predictions"]
        targets = outputs["targets"]
        context_tokens = outputs["context_tokens"]
        target_tokens = outputs["target_tokens"]
        assert isinstance(predictions, list)
        assert isinstance(targets, list)
        assert isinstance(context_tokens, torch.Tensor)
        assert isinstance(target_tokens, torch.Tensor)

        pred_loss = torch.stack([F.smooth_l1_loss(pred, target) for pred, target in zip(predictions, targets, strict=True)]).mean()
        cls = context_tokens[:, 0, :]
        std = torch.sqrt(cls.var(dim=0, unbiased=False) + 1e-4)
        variance_loss = torch.mean(F.relu(1.0 - std))
        loss = pred_loss + self.vicreg_weight * variance_loss
        target_std = target_tokens[:, 1:, :].std(dim=(0, 1), unbiased=False).mean()
        return JEPALossOutput(loss=loss, prediction_loss=pred_loss, variance_loss=variance_loss, target_std=target_std)

    @torch.no_grad()
    def encode(self, values: torch.Tensor, batch_id: torch.Tensor | None = None, use_target: bool = True) -> torch.Tensor:
        encoder = self.target_encoder if use_target else self.context_encoder
        return encoder(values, batch_id)[:, 0, :]
