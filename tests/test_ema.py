from __future__ import annotations

import torch

from jepa_poc.models.jepa import JEPA


def test_ema_update_matches_manual_computation():
    torch.manual_seed(0)
    model = JEPA(n_genes=32, n_batches=2, d_model=32, n_layers=1, n_heads=4, predictor_layers=1)
    before = [p.detach().clone() for p in model.target_encoder.parameters()]
    with torch.no_grad():
        for p in model.context_encoder.parameters():
            p.add_(torch.randn_like(p) * 0.01)
    context = [p.detach().clone() for p in model.context_encoder.parameters()]
    model.update_target_encoder(momentum=0.9)
    for target_param, old_target, context_param in zip(model.target_encoder.parameters(), before, context, strict=True):
        expected = old_target * 0.9 + context_param * 0.1
        assert torch.allclose(target_param, expected, atol=1e-6)
        assert target_param.requires_grad is False
