"""Train the JEPA POC on the pretrain split.

Adds long-run conveniences over the original scaffold:
  * resume from ``runs/poc/last.ckpt`` automatically
  * step-keyed checkpoint via ``ModelCheckpoint`` (every N steps)
  * CSV logger -> ``runs/poc/metrics.csv``
  * persistent DataLoader workers (lower per-epoch overhead)
  * timestamped stdout logging so the tee'd log file is easy to scan
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import anndata as ad
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

from jepa_poc.config import ensure_dir, load_config
from jepa_poc.data.loader import TCellDataset, fit_encoders
from jepa_poc.models.jepa import JEPA
from jepa_poc.train.callbacks import CheckpointEncoderCallback, CollapseMonitor
from jepa_poc.train.lit_module import JEPALitModule


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _make_loader(ds: TCellDataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=shuffle,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/poc.yaml")
    parser.add_argument("--resume", default=None, help="Optional explicit checkpoint to resume from.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    processed = Path(cfg.data.output_dir) / "pretrain_processed.h5ad"
    if not processed.exists():
        raise FileNotFoundError(f"{processed} not found. Run scripts/prep_data.py first.")
    run_dir = ensure_dir(cfg.data.run_dir)
    pl.seed_everything(int(cfg.seed), workers=True)

    _log(f"Loading {processed}")
    t0 = time.time()
    adata = ad.read_h5ad(processed)
    _log(f"Loaded shape={adata.shape} in {time.time() - t0:.1f}s")

    encoders = fit_encoders(adata.obs, cfg.data.batch_key, "annotation_group", cfg.data.perturbation_key)
    _log(
        f"Encoders: {len(encoders.batch_to_id)} batches, {len(encoders.label_to_id)} labels, "
        f"{len(encoders.perturbation_to_id)} perturbations"
    )

    train_ds = TCellDataset(
        adata,
        encoders=encoders,
        split="train",
        batch_key=cfg.data.batch_key,
        label_key="annotation_group",
        perturbation_key=cfg.data.perturbation_key,
    )
    val_ds = TCellDataset(
        adata,
        encoders=encoders,
        split="val",
        batch_key=cfg.data.batch_key,
        label_key="annotation_group",
        perturbation_key=cfg.data.perturbation_key,
    )
    _log(f"train_ds={len(train_ds):,} cells | val_ds={len(val_ds):,} cells")

    train_loader = _make_loader(train_ds, cfg.train.batch_size, shuffle=True, num_workers=cfg.train.num_workers)
    val_loader = _make_loader(val_ds, cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers)

    model = JEPA(
        n_genes=adata.n_vars,
        n_batches=max(1, len(encoders.batch_to_id)),
        d_model=cfg.model.d_model,
        n_layers=cfg.model.n_layers,
        n_heads=cfg.model.n_heads,
        predictor_layers=cfg.model.predictor_layers,
        dropout=cfg.model.dropout,
        mask_context_frac=cfg.model.mask_context_frac,
        n_target_blocks=cfg.model.n_target_blocks,
        target_block_frac=cfg.model.target_block_frac,
        vicreg_weight=cfg.model.vicreg_weight,
    )
    n_params = sum(p.numel() for p in model.parameters())
    _log(f"Model built: {n_params/1e6:.2f}M parameters")

    lit = JEPALitModule(
        model,
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        warmup_steps=cfg.train.warmup_steps,
        max_steps=cfg.train.max_steps,
        ema_momentum_start=cfg.train.ema_momentum_start,
        ema_momentum_end=cfg.train.ema_momentum_end,
    )

    ckpt_dir = ensure_dir(run_dir / "checkpoints")
    ckpt_every = int(getattr(cfg.train, "ckpt_every_n_steps", 5000))
    step_ckpt = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="step={step}",
        every_n_train_steps=ckpt_every,
        save_top_k=3,
        save_last=True,
        monitor="train/loss",
        mode="min",
        auto_insert_metric_name=False,
    )
    callbacks = [
        CollapseMonitor(cfg.train.collapse_std_threshold, cfg.train.collapse_patience),
        CheckpointEncoderCallback(str(run_dir / "ema_target_encoder.pt")),
        step_ckpt,
    ]

    csv_logger = CSVLogger(save_dir=str(run_dir), name="csv_logs")

    last_ckpt = run_dir / "last.ckpt"
    auto_last = ckpt_dir / "last.ckpt"
    resume_from = args.resume
    if resume_from is None:
        if last_ckpt.exists():
            resume_from = str(last_ckpt)
        elif auto_last.exists():
            resume_from = str(auto_last)
    if resume_from:
        _log(f"Resuming from {resume_from}")
    else:
        _log("Starting fresh (no checkpoint found)")

    trainer = pl.Trainer(
        max_steps=cfg.train.max_steps,
        accelerator="auto",
        devices=cfg.train.devices,
        strategy=cfg.train.strategy,
        precision=cfg.train.precision,
        default_root_dir=str(run_dir),
        log_every_n_steps=cfg.train.log_every_n_steps,
        val_check_interval=cfg.train.val_check_interval,
        callbacks=callbacks,
        logger=csv_logger,
        enable_progress_bar=False,
    )

    _log("Starting trainer.fit(...)")
    trainer.fit(lit, train_loader, val_loader, ckpt_path=resume_from)
    trainer.save_checkpoint(run_dir / "last.ckpt")
    _log(f"Training complete. Final checkpoint -> {run_dir / 'last.ckpt'}")


if __name__ == "__main__":
    main()
