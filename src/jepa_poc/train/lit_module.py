from __future__ import annotations

import math
from typing import Any

import pytorch_lightning as pl
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from jepa_poc.models.jepa import JEPA


class JEPALitModule(pl.LightningModule):
    """Lightning wrapper for JEPA pretraining."""

    def __init__(
        self,
        model: JEPA,
        lr: float = 1e-4,
        weight_decay: float = 0.05,
        warmup_steps: int = 5000,
        max_steps: int = 100000,
        ema_momentum_start: float = 0.996,
        ema_momentum_end: float = 1.0,
    ) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.ema_momentum_start = ema_momentum_start
        self.ema_momentum_end = ema_momentum_end
        self.save_hyperparameters(ignore=["model"])

    def _ema_momentum(self) -> float:
        if self.max_steps <= 1:
            return self.ema_momentum_end
        progress = min(1.0, self.global_step / float(self.max_steps - 1))
        return self.ema_momentum_end - (self.ema_momentum_end - self.ema_momentum_start) * (math.cos(math.pi * progress) + 1) / 2

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self.model(batch["values"], batch.get("batch_id"))
        losses = self.model.compute_loss(outputs)
        self.log_dict(
            {
                "train/loss": losses.loss,
                "train/prediction_loss": losses.prediction_loss,
                "train/variance_loss": losses.variance_loss,
                "train/target_std": losses.target_std,
            },
            prog_bar=True,
            on_step=True,
            on_epoch=False,
        )
        return losses.loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self.model(batch["values"], batch.get("batch_id"))
        losses = self.model.compute_loss(outputs)
        self.log_dict(
            {
                "val/loss": losses.loss,
                "val/prediction_loss": losses.prediction_loss,
                "val/target_std": losses.target_std,
            },
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        return losses.loss

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        self.model.update_target_encoder(self._ema_momentum())
        self.log("train/ema_momentum", self._ema_momentum(), on_step=True, on_epoch=False)

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        def lr_lambda(step: int) -> float:
            if step < self.warmup_steps:
                return max(1e-8, step / max(1, self.warmup_steps))
            progress = min(1.0, (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps))
            return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
