from __future__ import annotations

import pytorch_lightning as pl
import torch


class CollapseMonitor(pl.Callback):
    """Stop training if the target encoder representation collapses."""

    def __init__(self, threshold: float = 0.01, patience: int = 100) -> None:
        super().__init__()
        self.threshold = threshold
        self.patience = patience
        self.bad_steps = 0

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx: int) -> None:
        model = getattr(pl_module, "model", None)
        if model is None:
            return
        batch_id = batch.get("batch_id")
        if batch_id is not None:
            batch_id = batch_id.to(pl_module.device)
        with torch.no_grad():
            z = model.encode(batch["values"].to(pl_module.device), batch_id, use_target=True)
            std = z.std(dim=0, unbiased=False).mean()
        pl_module.log("train/collapse_std", std, on_step=True, on_epoch=False)
        if std.item() < self.threshold:
            self.bad_steps += 1
        else:
            self.bad_steps = 0
        if self.bad_steps >= self.patience:
            raise RuntimeError(
                f"Representation collapse detected: mean target std below {self.threshold} "
                f"for {self.patience} consecutive steps."
            )


class CheckpointEncoderCallback(pl.Callback):
    """Save the EMA target encoder separately at the end of training."""

    def __init__(self, output_path: str) -> None:
        super().__init__()
        self.output_path = output_path

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        torch.save(pl_module.model.target_encoder.state_dict(), self.output_path)
