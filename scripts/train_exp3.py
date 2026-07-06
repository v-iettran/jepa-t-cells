"""Train an Experiment 3 arm (Arm 1 or Arm 2) on the pretrain split.

Mirrors scripts/train_jepa.py (resume, step checkpoints, CSV logging, persistent
workers) but builds the new ESM2-tokenized JEPA-with-reconstruction model and the
Exp3 Lightning module. EMA is applied only for Arm 1; Arm 2 is symmetric/EMA-free.

Usage:
    python scripts/train_exp3.py --config configs/exp3.yaml --arm arm1
    python scripts/train_exp3.py --config configs/exp3.yaml --arm arm2 --lambda-sig 0.1

Short probes for the lambda_sig scan:
    python scripts/train_exp3.py --arm arm2 --lambda-sig 1.0 \
        --max-steps 3000 --run-dir runs/exp3_sigscan/ls1p0
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

from jepa_poc.config import ensure_dir, load_config
from jepa_poc.data.loader import BalancedBatchSampler, TCellDataset, fit_encoders
from jepa_poc.models.jepa_recon import JEPAArm1, JEPAArm2
from jepa_poc.train.callbacks import CheckpointEncoderCallback, CollapseMonitor
from jepa_poc.train.lit_module_exp3 import Exp3LitModule


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _make_loader(ds: TCellDataset, batch_size: int, shuffle: bool, num_workers: int,
                 *, balanced: bool = False, seed: int = 0) -> DataLoader:
    if balanced:
        sampler = BalancedBatchSampler(ds.batch_ids, batch_size, seed=seed, drop_last=True)
        _log(f"BalancedBatchSampler: {sampler.n_groups} groups, per_group={sampler.per_group}, "
             f"{len(sampler)} batches/epoch")
        return DataLoader(
            ds,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=shuffle,
    )


def _configured_path(cfg, key: str, default_name: str) -> Path:
    configured = getattr(cfg.data, key, None)
    if configured:
        return Path(configured)
    return Path(cfg.data.output_dir) / default_name


def _run_dir(cfg, arm: str, override: str | None) -> Path:
    if override:
        return Path(override)
    key = f"run_dir_{arm.lower()}"
    configured = getattr(cfg.data, key, None)
    if configured:
        return Path(configured)
    base = Path(cfg.data.run_dir)
    return base.parent / f"{base.name}_{arm}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/exp3.yaml")
    parser.add_argument("--arm", choices=["arm1", "arm2"], required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--run-dir", default=None, help="Override the run/output directory.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override total steps (e.g. for short probes).")
    parser.add_argument("--lambda-sig", type=float, default=None, help="Arm 2 SIGReg weight override.")
    parser.add_argument("--lambda-rec", type=float, default=None, help="Reconstruction weight override.")
    parser.add_argument("--geometry-every", type=int, default=None, help="Override geometry-logging interval (steps).")
    parser.add_argument("--batch-size", type=int, default=None, help="Override training batch size (e.g. 512).")
    parser.add_argument("--within-batch", action="store_true",
                        help="Compute the anti-collapse regularizer per technical-batch group, then average.")
    parser.add_argument("--balanced-sampler", action="store_true",
                        help="Use a sampler guaranteeing batch_size/n_groups cells per batch group per step.")
    parser.add_argument("--grad-checkpointing", action="store_true",
                        help="Gradient-checkpoint the encoder layers (needed for batch 512 on a 48GB GPU).")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    cfg = load_config(args.config)
    max_steps = int(args.max_steps if args.max_steps is not None else cfg.train.max_steps)
    processed = _configured_path(cfg, "pretrain_processed_path", "pretrain_processed.h5ad")
    if not processed.exists():
        raise FileNotFoundError(f"{processed} not found. Run scripts/prep_data.py first.")
    run_dir = ensure_dir(_run_dir(cfg, args.arm, args.run_dir))
    pl.seed_everything(int(cfg.seed), workers=True)

    esm_path = Path(cfg.data.esm_embeddings_path)
    if not esm_path.exists():
        raise FileNotFoundError(f"{esm_path} not found. Run scripts/build_esm2_gene_embeddings.py first.")
    esm = np.load(esm_path, allow_pickle=True)
    esm_table, fallback_mask = esm["embeddings"], esm["fallback_mask"]
    _log(f"ESM table {esm_table.shape}, fallback={int(fallback_mask.sum())}/{len(fallback_mask)}")

    _log(f"Loading {processed}")
    t0 = time.time()
    adata = ad.read_h5ad(processed)
    _log(f"Loaded shape={adata.shape} in {time.time() - t0:.1f}s")
    if adata.n_vars != esm_table.shape[0]:
        raise ValueError(f"Gene count mismatch: adata {adata.n_vars} vs ESM table {esm_table.shape[0]}")

    encoders = fit_encoders(adata.obs, cfg.data.batch_key, "annotation_group", cfg.data.perturbation_key)
    _log(f"Encoders: {len(encoders.batch_to_id)} batches, {len(encoders.perturbation_to_id)} perturbations")

    train_ds = TCellDataset(adata, encoders=encoders, split="train", batch_key=cfg.data.batch_key,
                            label_key="annotation_group", perturbation_key=cfg.data.perturbation_key)
    val_ds = TCellDataset(adata, encoders=encoders, split="val", batch_key=cfg.data.batch_key,
                          label_key="annotation_group", perturbation_key=cfg.data.perturbation_key)
    _log(f"train_ds={len(train_ds):,} cells | val_ds={len(val_ds):,} cells")

    batch_size = int(args.batch_size if args.batch_size is not None else cfg.train.batch_size)
    _log(f"batch_size={batch_size} | within_batch_reg={args.within_batch} | "
         f"balanced_sampler={args.balanced_sampler} | grad_checkpointing={args.grad_checkpointing}")
    train_loader = _make_loader(train_ds, batch_size, shuffle=True, num_workers=cfg.train.num_workers,
                                balanced=args.balanced_sampler, seed=int(cfg.seed))
    # Val batch kept smaller than train: it does not feed the within-batch SIGReg
    # (which needs >=256/group at train time) and a large val forward fragments the
    # CUDA allocator right before the first training step on resume.
    val_batch = min(batch_size, 256)
    val_loader = _make_loader(val_ds, val_batch, shuffle=False, num_workers=cfg.train.num_workers)

    n_batches = max(1, len(encoders.batch_to_id))
    lambda_rec = float(args.lambda_rec if args.lambda_rec is not None else cfg.model.lambda_rec)
    common = dict(
        esm_embeddings=esm_table,
        fallback_mask=fallback_mask,
        n_batches=n_batches,
        d_model=cfg.model.d_model,
        d_id_proj=int(cfg.model.d_id_proj),
        d_expr=int(cfg.model.d_expr),
        use_fallback_indicator=bool(cfg.model.use_fallback_indicator),
        n_layers=cfg.model.n_layers,
        n_heads=cfg.model.n_heads,
        predictor_layers=cfg.model.predictor_layers,
        dropout=cfg.model.dropout,
        mask_context_frac=cfg.model.mask_context_frac,
        n_target_blocks=cfg.model.n_target_blocks,
        target_block_frac=cfg.model.target_block_frac,
        lambda_rec=lambda_rec,
        within_batch_stabilization=bool(args.within_batch),
        use_grad_checkpointing=bool(args.grad_checkpointing),
    )
    if args.arm == "arm1":
        model = JEPAArm1(stabilization_weight=float(cfg.model.vicreg_weight), **common)
    else:
        lambda_sig = float(args.lambda_sig if args.lambda_sig is not None else cfg.model.lambda_sig)
        model = JEPAArm2(
            stabilization_weight=lambda_sig,
            sigreg_num_points=int(cfg.model.sigreg_num_points),
            sigreg_num_slices=int(cfg.model.sigreg_num_slices),
            **common,
        )
        _log(f"Arm 2 lambda_sig={lambda_sig}")
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _log(f"Model built ({args.arm}): {n_trainable/1e6:.2f}M trainable params, lambda_rec={lambda_rec}")

    lit = Exp3LitModule(
        model,
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        warmup_steps=cfg.train.warmup_steps,
        max_steps=max_steps,
        ema_momentum_start=cfg.train.ema_momentum_start,
        ema_momentum_end=cfg.train.ema_momentum_end,
        geometry_log_every_n_steps=int(
            args.geometry_every if args.geometry_every is not None
            else getattr(cfg.train, "geometry_log_every_n_steps", 2000)
        ),
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
    _log(f"Resuming from {resume_from}" if resume_from else "Starting fresh (no checkpoint found)")

    trainer = pl.Trainer(
        max_steps=max_steps,
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
        num_sanity_val_steps=0,
        gradient_clip_val=1.0,
    )

    _log("Starting trainer.fit(...)")
    trainer.fit(lit, train_loader, val_loader, ckpt_path=resume_from)
    trainer.save_checkpoint(run_dir / "last.ckpt")
    _log(f"Training complete. Final checkpoint -> {run_dir / 'last.ckpt'}")


if __name__ == "__main__":
    main()
