"""Lightning module for Experiment 3 (Arm 1 / Arm 2 JEPA + reconstruction).

Logs every loss term separately (prediction, reconstruction, and VICReg or
SIGReg) plus CLS-embedding geometry canaries (per-dim std, effective rank,
mean pairwise cosine) identically for both arms. EMA is applied only when the
model exposes a real (non-symmetric) teacher; Arm 2's ``update_target_encoder``
is a no-op.
"""

from __future__ import annotations

import math
from typing import Any

import pytorch_lightning as pl
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from jepa_poc.models.jepa_recon import JEPAReconBase


class Exp3LitModule(pl.LightningModule):
    def __init__(
        self,
        model: JEPAReconBase,
        lr: float = 1e-4,
        weight_decay: float = 0.05,
        warmup_steps: int = 5000,
        max_steps: int = 200000,
        ema_momentum_start: float = 0.996,
        ema_momentum_end: float = 1.0,
        geometry_log_every_n_steps: int = 2000,
    ) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.ema_momentum_start = ema_momentum_start
        self.ema_momentum_end = ema_momentum_end
        self.geometry_log_every_n_steps = geometry_log_every_n_steps
        self.use_ema = not getattr(model, "is_symmetric", False)
        self.save_hyperparameters(ignore=["model"])

    def _ema_momentum(self) -> float:
        if self.max_steps <= 1:
            return self.ema_momentum_end
        progress = min(1.0, self.global_step / float(self.max_steps - 1))
        return self.ema_momentum_end - (self.ema_momentum_end - self.ema_momentum_start) * (math.cos(math.pi * progress) + 1) / 2

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self.model(batch["values"], batch.get("batch_id"))
        losses = self.model.compute_loss(outputs)
        cls = outputs["context_tokens"][:, 0, :]
        self.log_dict(
            {
                "train/loss": losses.loss,
                "train/prediction_loss": losses.prediction_loss,
                "train/recon_loss": losses.recon_loss,
                "train/stabilization_loss": losses.stabilization_loss,
                "train/target_std": losses.target_std,
            },
            prog_bar=True,
            on_step=True,
            on_epoch=False,
        )
        if self.global_step % max(1, self.geometry_log_every_n_steps) == 0:
            self._log_geometry(cls.detach(), prefix="train/geometry")
        return losses.loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self.model(batch["values"], batch.get("batch_id"))
        losses = self.model.compute_loss(outputs)
        self.log_dict(
            {
                "val/loss": losses.loss,
                "val/prediction_loss": losses.prediction_loss,
                "val/recon_loss": losses.recon_loss,
                "val/stabilization_loss": losses.stabilization_loss,
                "val/target_std": losses.target_std,
            },
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        return losses.loss

    def _log_geometry(self, z: torch.Tensor, prefix: str) -> None:
        """Log CLS-embedding geometry canaries.

        Effective rank is computed from the singular values of the centered
        embedding matrix (numerically stable), not from ``eigvalsh`` on the
        covariance, which fails to converge when the spectrum is highly
        anisotropic / near-degenerate (LinAlgError 257). The whole block is
        guarded so a logging failure never kills training.
        """
        device_type = z.device.type
        try:
            with torch.autocast(device_type=device_type, enabled=False):
                z = z.detach().to(dtype=torch.float32)
                n = z.shape[0]
                std = z.std(dim=0, unbiased=False)
                centered = z - z.mean(dim=0, keepdim=True)
                # Singular values of the centered matrix; s_i^2 / (n-1) are the
                # covariance eigenvalues but svdvals is far more robust.
                svals = torch.linalg.svdvals(centered)
                var = (svals ** 2) / max(1, n - 1)
                probs = var / var.sum().clamp_min(1e-12)
                entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum()
                effective_rank = torch.exp(entropy)
                normed = torch.nn.functional.normalize(z, dim=1)
                cosine = normed @ normed.T
                if cosine.numel() > n:
                    off_diag_mean = (cosine.sum() - cosine.diag().sum()) / (cosine.numel() - n)
                else:
                    off_diag_mean = torch.zeros((), device=z.device)
        except Exception as exc:  # pragma: no cover - logging must never crash training
            print(f"[geometry] skipped logging at step {self.global_step}: {exc}", flush=True)
            return
        self.log_dict(
            {
                f"{prefix}/std_mean": std.mean(),
                f"{prefix}/std_min": std.min(),
                f"{prefix}/effective_rank": effective_rank,
                f"{prefix}/pairwise_cosine_mean": off_diag_mean,
            },
            on_step=True,
            on_epoch=False,
        )

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        if self.use_ema:
            self.model.update_target_encoder(self._ema_momentum())
            self.log("train/ema_momentum", self._ema_momentum(), on_step=True, on_epoch=False)

    def configure_optimizers(self) -> dict[str, Any]:
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = AdamW(params, lr=self.lr, weight_decay=self.weight_decay)

        def lr_lambda(step: int) -> float:
            if step < self.warmup_steps:
                return max(1e-8, step / max(1, self.warmup_steps))
            progress = min(1.0, (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps))
            return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
