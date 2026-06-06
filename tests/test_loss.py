from __future__ import annotations

import torch

from jepa_poc.models.jepa import JEPA


def test_loss_is_finite_and_backpropagates(synthetic_dataset):
    batch = [synthetic_dataset[i] for i in range(16)]
    values = torch.stack([x["values"] for x in batch])
    batch_id = torch.stack([x["batch_id"] for x in batch])
    model = JEPA(n_genes=values.shape[1], n_batches=3, d_model=32, n_layers=1, n_heads=4, predictor_layers=1)
    outputs = model(values, batch_id)
    losses = model.compute_loss(outputs)
    assert torch.isfinite(losses.loss)
    losses.loss.backward()
    grad_norm = sum(p.grad.abs().sum().item() for p in model.context_encoder.parameters() if p.grad is not None)
    assert grad_norm > 0
