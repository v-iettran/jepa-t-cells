from __future__ import annotations

import torch

from jepa_poc.models.jepa import JEPA


def test_jepa_overfits_tiny_synthetic_batch(synthetic_dataset):
    torch.manual_seed(0)
    batch = [synthetic_dataset[i] for i in range(64)]
    values = torch.stack([x["values"] for x in batch])
    batch_id = torch.stack([x["batch_id"] for x in batch])
    model = JEPA(
        n_genes=values.shape[1],
        n_batches=3,
        d_model=32,
        n_layers=1,
        n_heads=4,
        predictor_layers=1,
        mask_context_frac=0.5,
        n_target_blocks=1,
        target_block_frac=0.25,
        vicreg_weight=0.0,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first_loss = None
    last_loss = None
    for step in range(40):
        outputs = model(values, batch_id)
        losses = model.compute_loss(outputs)
        if first_loss is None:
            first_loss = losses.loss.item()
        last_loss = losses.loss.item()
        opt.zero_grad()
        losses.loss.backward()
        opt.step()
        model.update_target_encoder(momentum=0.9)
    assert last_loss is not None and first_loss is not None
    assert last_loss < first_loss
