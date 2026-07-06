from __future__ import annotations

import torch
import torch.nn.functional as F

from jepa_poc.models.jepa import JEPA, JEPALossOutput


class JEPASIGReg(JEPA):
    """JEPA variant that replaces the VICReg variance term with SIGReg."""

    def __init__(self, *args, sigreg_num_points: int = 17, sigreg_num_slices: int = 1024, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        try:
            import lejepa
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise ImportError(
                "A1/SIGReg requires the optional 'lejepa' package. "
                "Install the pinned LeJEPA dependency before training --arm A1."
            ) from exc

        univariate_test = lejepa.univariate.EppsPulley(n_points=sigreg_num_points)
        self.sigreg_loss_fn = lejepa.multivariate.SlicingUnivariateTest(
            univariate_test=univariate_test,
            num_slices=sigreg_num_slices,
        )

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
        sigreg_loss = self.sigreg_loss_fn(cls)
        loss = pred_loss + self.vicreg_weight * sigreg_loss
        target_std = target_tokens[:, 1:, :].std(dim=(0, 1), unbiased=False).mean()
        return JEPALossOutput(loss=loss, prediction_loss=pred_loss, variance_loss=sigreg_loss, target_std=target_std)
