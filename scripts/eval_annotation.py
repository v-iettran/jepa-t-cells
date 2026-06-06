"""Culture-condition annotation evaluation for the JEPA POC.

Embeds donor-held-out cells with the JEPA target encoder and scores a linear and
a kNN probe. Also runs a PCA baseline (PCA on the same log-CPM HVG matrix, then
the identical probes) so the JEPA representation is compared apples-to-apples
against a simple linear method rather than only against chance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import torch

from jepa_poc.baselines.pca import PCABaseline
from jepa_poc.config import ensure_dir, load_config
from jepa_poc.data.loader import TCellDataset, fit_encoders, make_dataloader
from jepa_poc.eval.annotation import embed_loader, run_knn_probe, run_linear_probe
from jepa_poc.models.jepa import JEPA


def _log(msg: str) -> None:
    import time

    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _load_model(cfg, adata: ad.AnnData, n_batches: int, checkpoint: Path) -> JEPA:
    model = JEPA(
        n_genes=adata.n_vars,
        n_batches=n_batches,
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
    if checkpoint.exists():
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        state = {k.removeprefix("model."): v for k, v in state.items() if k.startswith("model.")}
        model.load_state_dict(state, strict=False)
        _log(f"Loaded checkpoint {checkpoint} (step={ckpt.get('global_step', '?')})")
    else:
        _log(f"WARNING: checkpoint {checkpoint} not found; using random weights")
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/poc.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--refresh-embeddings", action="store_true", help="ignore cached JEPA embeddings")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_dir = ensure_dir(cfg.data.run_dir)
    checkpoint = Path(args.checkpoint) if args.checkpoint else Path(cfg.data.run_dir) / "last.ckpt"
    adata_path = Path(cfg.data.output_dir) / "annotation_processed.h5ad"
    if not adata_path.exists():
        raise FileNotFoundError(f"{adata_path} not found. Run scripts/prep_data.py first.")

    adata = ad.read_h5ad(adata_path)
    encoders = fit_encoders(adata.obs, cfg.data.batch_key, "annotation_group", cfg.data.perturbation_key)
    model = _load_model(cfg, adata, max(1, len(encoders.batch_to_id)), checkpoint)
    train_ds = TCellDataset(adata, encoders=encoders, split="train", batch_key=cfg.data.batch_key, label_key="annotation_group", perturbation_key=cfg.data.perturbation_key)
    test_ds = TCellDataset(adata, encoders=encoders, split="test", batch_key=cfg.data.batch_key, label_key="annotation_group", perturbation_key=cfg.data.perturbation_key)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ----- JEPA embeddings (cached) -----
    cache_dir = ensure_dir(Path(cfg.data.run_dir) / "_embed_cache")
    cache_file = cache_dir / "annotation_embeds.npz"
    if cache_file.exists() and not args.refresh_embeddings:
        _log(f"Loading cached JEPA embeddings from {cache_file}")
        c = np.load(cache_file)
        train_z, train_y, test_z, test_y = c["train_z"], c["train_y"], c["test_z"], c["test_y"]
    else:
        train_loader = make_dataloader(train_ds, cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers)
        test_loader = make_dataloader(test_ds, cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers)
        _log(f"Embedding {train_ds.adata.n_obs:,} train + {test_ds.adata.n_obs:,} test cells with JEPA target encoder")
        train_z, train_y = embed_loader(model, train_loader, device=device)
        test_z, test_y = embed_loader(model, test_loader, device=device)
        np.savez(cache_file, train_z=train_z, train_y=train_y, test_z=test_z, test_y=test_y)
        _log(f"Cached JEPA embeddings -> {cache_file}")

    _log("Scoring JEPA probes")
    results = {
        "jepa": {
            "linear_probe": run_linear_probe(train_z, train_y, test_z, test_y, cfg.eval.linear_probe_c_grid),
            "knn_probe": run_knn_probe(train_z, train_y, test_z, test_y, cfg.eval.knn_k),
        },
        # Kept at top level for backward compatibility with earlier result files.
        "linear_probe": None,
        "knn_probe": None,
    }
    results["linear_probe"] = results["jepa"]["linear_probe"]
    results["knn_probe"] = results["jepa"]["knn_probe"]

    # ----- PCA baseline (same probes, same labels) -----
    n_comp = int(cfg.eval.pca_components)
    _log(f"Fitting PCA baseline ({n_comp} components) on log-CPM HVG matrix")
    pca = PCABaseline(n_components=n_comp, seed=int(cfg.seed)).fit(train_ds.adata)
    pca_train_z = pca.embed(train_ds.adata)
    pca_test_z = pca.embed(test_ds.adata)
    # Labels align with adata row order (shuffle=False), so reuse the JEPA labels.
    _log("Scoring PCA-baseline probes")
    results["pca_baseline"] = {
        "n_components": n_comp,
        "linear_probe": run_linear_probe(pca_train_z, train_y, pca_test_z, test_y, cfg.eval.linear_probe_c_grid),
        "knn_probe": run_knn_probe(pca_train_z, train_y, pca_test_z, test_y, cfg.eval.knn_k),
    }

    out = run_dir / "annotation_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    _log(f"Wrote {out}")
    jf = results["jepa"]["linear_probe"]["macro_f1"]
    pf = results["pca_baseline"]["linear_probe"]["macro_f1"]
    _log(f"Linear macro-F1: JEPA={jf:.4f}  PCA={pf:.4f}  delta={jf - pf:+.4f}")


if __name__ == "__main__":
    main()
