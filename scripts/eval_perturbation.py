"""Perturbation-response evaluation for the JEPA POC.

Two complementary views (see runs/poc/perturbation_results.json):

* ``representation_quality`` (Part A): decode the latent perturbation direction
  (held-out cells vs condition-matched controls) and correlate with the true
  expression delta. Measures whether the JEPA embedding *encodes* the
  perturbation effect. No head, no gene-identity generalization, no leakage.

* ``head_prediction`` (Part B): predict an unseen gene's effect from a non-leaky,
  genome-wide co-expression identity feature (built from control pseudobulk).
  Plus a ``baseline_mean_train_delta`` reference (a gene-agnostic predictor) so
  we can tell real gene-specific signal from "predict the average effect".

All deltas use condition-matched controls and are reported overall, per held-out
gene, and per culture condition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import torch

from jepa_poc.config import ensure_dir, load_config
from jepa_poc.data.loader import TCellDataset, fit_encoders, make_dataloader
from jepa_poc.eval.annotation import embed_loader
from jepa_poc.eval.gene_features import (
    build_coexpression_gene_embedding,
    features_for_names,
    gene_names_from_adata,
    grn_state_features,
    jepa_gene_embedding_features,
)
from jepa_poc.eval.perturbation import (
    condition_group_means,
    decode_latent,
    delta_metrics,
    fit_linear_decoder,
    matched_control_means,
    predict_head,
    sample_matched_controls,
    train_perturbation_head,
)
from jepa_poc.models.jepa import JEPA


def _log(msg: str) -> None:
    import time

    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _load_model(cfg, n_vars: int, n_batches: int, checkpoint: Path) -> JEPA:
    model = JEPA(
        n_genes=n_vars,
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
        if any(k.startswith("model.") for k in state):
            state = {k.removeprefix("model."): v for k, v in state.items() if k.startswith("model.")}
            model.load_state_dict(state, strict=False)
        elif any(k.startswith("gene_embedding.") for k in state):
            model.target_encoder.load_state_dict(state, strict=False)
            model.context_encoder.load_state_dict(state, strict=False)
        else:
            model.load_state_dict(state, strict=False)
        _log(f"Loaded checkpoint {checkpoint} (step={ckpt.get('global_step', '?')})")
    else:
        _log(f"WARNING: checkpoint {checkpoint} not found; using random weights")
    return model


def _subsample(n: int, cap: int, rng: np.random.Generator) -> np.ndarray:
    if cap <= 0 or n <= cap:
        return np.arange(n)
    return np.sort(rng.choice(n, size=cap, replace=False))


def _stratified_subsample(cond: np.ndarray, cap: int, rng: np.random.Generator) -> np.ndarray:
    """Sample up to ``cap`` indices, balanced across condition labels."""

    if cap <= 0 or len(cond) <= cap:
        return np.arange(len(cond))
    cond = cond.astype(str)
    conds = np.unique(cond)
    per = max(1, cap // len(conds))
    keep: list[np.ndarray] = []
    for c in conds:
        pool = np.where(cond == c)[0]
        take = min(per, pool.size)
        keep.append(rng.choice(pool, size=take, replace=False))
    return np.sort(np.concatenate(keep))


def _configured_path(cfg, key: str, default_name: str) -> Path:
    configured = getattr(cfg.data, key, None)
    if configured:
        return Path(configured)
    return Path(cfg.data.output_dir) / default_name


def _build_gene_features(
    mode: str,
    *,
    names: np.ndarray,
    states: np.ndarray,
    model: JEPA,
    gene_names: list[str],
    cfg,
) -> tuple[np.ndarray, str]:
    if mode == "coexpression":
        sym_to_vec, feat_dim = build_coexpression_gene_embedding(
            cfg.data.pseudobulk_path,
            n_components=int(getattr(cfg.eval, "coexpr_components", 50)),
            guide_type_key=cfg.data.guide_type_key,
            control_guide_type=cfg.data.non_targeting_guide_type,
            seed=int(cfg.seed),
        )
        return features_for_names(names, sym_to_vec, feat_dim), f"coexpression_svd_{feat_dim}d"

    if mode == "jepa":
        return (
            jepa_gene_embedding_features(names, gene_names, model.target_encoder.gene_embedding.weight),
            f"jepa_gene_embedding_{model.target_encoder.gene_embedding.embedding_dim}d",
        )

    if mode == "grn":
        grn_dir = getattr(cfg.grn, "output_dir", "data/grn") if "grn" in cfg else "data/grn"
        feats = grn_state_features(names, states, gene_names, grn_dir)
        return feats, f"grn_state_matched_{feats.shape[1]}d"

    if mode == "grn_jepa":
        grn_dir = getattr(cfg.grn, "output_dir", "data/grn") if "grn" in cfg else "data/grn"
        grn_feats = grn_state_features(names, states, gene_names, grn_dir)
        jepa_feats = jepa_gene_embedding_features(names, gene_names, model.target_encoder.gene_embedding.weight)
        return np.concatenate([grn_feats, jepa_feats], axis=1), f"grn_plus_jepa_{grn_feats.shape[1] + jepa_feats.shape[1]}d"

    raise ValueError(f"Unknown feature mode: {mode}")


def _assert_grn_no_eval_leakage(cfg, eval_adata: ad.AnnData) -> None:
    grn_dir = Path(getattr(cfg.grn, "output_dir", "data/grn") if "grn" in cfg else "data/grn")
    barcode_files = sorted(grn_dir.glob("genie3_ntc_barcodes*.txt"))
    if not barcode_files:
        raise FileNotFoundError(f"No GENIE3 barcode files found in {grn_dir}")
    grn_barcodes: set[str] = set()
    for path in barcode_files:
        grn_barcodes.update(line.strip() for line in path.read_text().splitlines() if line.strip())
    overlap = grn_barcodes.intersection(map(str, eval_adata.obs_names))
    if overlap:
        raise RuntimeError(f"GENIE3 leakage guard failed: {len(overlap)} GRN input barcodes overlap perturbation eval cells")
    _log(f"GENIE3 leakage guard passed ({len(grn_barcodes):,} GRN barcodes checked)")


def _group_metrics(
    z: np.ndarray,
    expr: np.ndarray,
    cond: np.ndarray,
    gene: np.ndarray,
    *,
    pred_expr: np.ndarray | None,
    decoder: np.ndarray,
    ctrl_means: dict,
    top_ks,
    test_genes: list[str],
) -> dict:
    """Build overall / per-gene / per-condition delta metrics for one view.

    If ``pred_expr`` is given, this is the head-prediction view (group means of
    predicted expression). Otherwise it is the representation view (decode the
    group's mean latent and the matched control's mean latent).
    """

    def metrics_for(mask: np.ndarray) -> dict | None:
        if mask.sum() == 0:
            return None
        ctrl_expr_mean, ctrl_z_mean = matched_control_means(ctrl_means, cond[mask])
        true_mean = expr[mask].mean(axis=0)
        if pred_expr is not None:
            pred_mean = pred_expr[mask].mean(axis=0)
            control_ref = ctrl_expr_mean
        else:
            pred_mean = decode_latent(z[mask].mean(axis=0)[None, :], decoder)[0]
            control_ref = decode_latent(ctrl_z_mean[None, :], decoder)[0]
        return delta_metrics(pred_mean, true_mean, control_ref, top_ks)

    out: dict = {"overall": metrics_for(np.ones(len(cond), dtype=bool))}
    per_gene = {}
    for g in test_genes:
        m = gene.astype(str) == g
        r = metrics_for(m)
        if r is not None:
            r["n_cells"] = int(m.sum())
            per_gene[g] = r
    out["per_gene"] = per_gene
    per_cond = {}
    for c in np.unique(cond.astype(str)):
        m = cond.astype(str) == c
        r = metrics_for(m)
        if r is not None:
            r["n_cells"] = int(m.sum())
            per_cond[c] = r
    out["per_condition"] = per_cond
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/poc.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--run-dir", default=None, help="Override output/cache directory for this eval arm.")
    parser.add_argument("--n-control", type=int, default=150000, help="controls to embed (stratified by condition)")
    parser.add_argument("--n-train-head", type=int, default=200000, help="train targeting cells to embed for the head")
    parser.add_argument("--refresh-embeddings", action="store_true", help="ignore cached embeddings")
    parser.add_argument(
        "--feature-mode",
        choices=["coexpression", "jepa", "grn", "grn_jepa"],
        default="coexpression",
        help="Perturbation identity feature for the head arm.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_dir = ensure_dir(args.run_dir or cfg.data.run_dir)
    checkpoint = Path(args.checkpoint) if args.checkpoint else Path(cfg.data.run_dir) / "last.ckpt"
    eval_path = _configured_path(cfg, "perturb_eval_processed_path", "perturb_eval_processed.h5ad")
    control_path = _configured_path(cfg, "perturb_controls_processed_path", "perturb_controls_processed.h5ad")
    if not eval_path.exists() or not control_path.exists():
        raise FileNotFoundError("Processed perturbation files not found. Run scripts/prep_data.py first.")

    top_ks = list(cfg.eval.top_k_de)
    pkey = cfg.data.perturbation_key
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(int(cfg.seed))

    # Cache embeddings keyed by the exact inputs they depend on: the encoder
    # checkpoint and the subsample caps/seed. This lets different encoders (A0 vs
    # A1) keep separate caches in the shared cache dir, and lets head runs reuse
    # the selected encoder's embeddings instead of re-embedding from scratch.
    cache_dir = ensure_dir(Path(cfg.data.run_dir) / "_embed_cache")
    _cache_key = hashlib.md5(
        f"{checkpoint.resolve()}|nc={args.n_control}|nt={args.n_train_head}|seed={int(cfg.seed)}".encode()
    ).hexdigest()[:12]
    cache_file = cache_dir / f"perturb_embeds_{_cache_key}.npz"

    _log("Loading processed AnnData (eval + controls)")
    eval_adata = ad.read_h5ad(eval_path)
    control_adata = ad.read_h5ad(control_path)
    combined_obs = ad.concat([control_adata, eval_adata], join="inner").obs
    encoders = fit_encoders(combined_obs, cfg.data.batch_key, "annotation_group", pkey)
    model = _load_model(cfg, eval_adata.n_vars, max(1, len(encoders.batch_to_id)), checkpoint)
    if args.feature_mode in {"grn", "grn_jepa"}:
        _assert_grn_no_eval_leakage(cfg, eval_adata)

    if cache_file.exists() and not args.refresh_embeddings:
        _log(f"Loading cached embeddings from {cache_file}")
        c = np.load(cache_file, allow_pickle=True)
        control_z, control_expr, control_cond = c["control_z"], c["control_expr"], c["control_cond"]
        train_z, train_expr, train_cond, train_gene = c["train_z"], c["train_expr"], c["train_cond"], c["train_gene"]
        test_z, test_expr, test_cond, test_gene = c["test_z"], c["test_expr"], c["test_cond"], c["test_gene"]
    else:
        # ----- controls (stratified subsample by condition) -----
        ctrl_cond_all = control_adata.obs["culture_condition"].astype(str).to_numpy()
        ctrl_idx = _stratified_subsample(ctrl_cond_all, args.n_control, rng)
        control_sub = control_adata[ctrl_idx].copy()
        control_ds = TCellDataset(control_sub, encoders=encoders, batch_key=cfg.data.batch_key, label_key="annotation_group", perturbation_key=pkey)
        _log(f"Embedding {control_ds.adata.n_obs:,} controls")
        control_z, _ = embed_loader(model, make_dataloader(control_ds, cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers), device)
        control_expr = control_ds.values
        control_cond = control_ds.adata.obs["culture_condition"].astype(str).to_numpy()

        # ----- train targeting (random subsample for the head) -----
        train_mask = eval_adata.obs["split"].astype(str).to_numpy() == "train"
        train_pos = np.where(train_mask)[0]
        sel = _subsample(train_pos.size, args.n_train_head, rng)
        train_sub = eval_adata[train_pos[sel]].copy()
        train_ds = TCellDataset(train_sub, encoders=encoders, batch_key=cfg.data.batch_key, label_key="annotation_group", perturbation_key=pkey)
        _log(f"Embedding {train_ds.adata.n_obs:,} train targeting cells")
        train_z, _ = embed_loader(model, make_dataloader(train_ds, cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers), device)
        train_expr = train_ds.values
        train_cond = train_ds.adata.obs["culture_condition"].astype(str).to_numpy()
        train_gene = train_ds.adata.obs[pkey].astype(str).to_numpy()

        # ----- test targeting (all held-out cells) -----
        test_ds = TCellDataset(eval_adata, encoders=encoders, split="test", batch_key=cfg.data.batch_key, label_key="annotation_group", perturbation_key=pkey)
        _log(f"Embedding {test_ds.adata.n_obs:,} test (held-out gene) cells")
        test_z, _ = embed_loader(model, make_dataloader(test_ds, cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers), device)
        test_expr = test_ds.values
        test_cond = test_ds.adata.obs["culture_condition"].astype(str).to_numpy()
        test_gene = test_ds.adata.obs[pkey].astype(str).to_numpy()

        np.savez(
            cache_file,
            control_z=control_z, control_expr=control_expr, control_cond=control_cond,
            train_z=train_z, train_expr=train_expr, train_cond=train_cond, train_gene=train_gene,
            test_z=test_z, test_expr=test_expr, test_cond=test_cond, test_gene=test_gene,
        )
        _log(f"Cached embeddings -> {cache_file}")

    test_genes = sorted(set(test_gene.astype(str).tolist()))
    _log(f"Test (held-out) genes: {test_genes}")

    # Condition-matched control means (expression + latent) for fair deltas.
    ctrl_means = condition_group_means(control_expr, control_z, control_cond)

    # Shared linear decoder z -> expression, fit on controls only.
    decoder = fit_linear_decoder(control_z, control_expr)

    # ---------------- Part A: representation quality ----------------
    _log("Part A: representation-quality (decoded latent perturbation direction)")
    representation = _group_metrics(
        test_z, test_expr, test_cond, test_gene,
        pred_expr=None, decoder=decoder, ctrl_means=ctrl_means, top_ks=top_ks, test_genes=test_genes,
    )

    # ---------------- Part B: head prediction with selected gene features --------
    gene_names = gene_names_from_adata(eval_adata)
    _log(f"Part B: building '{args.feature_mode}' gene features")
    train_feat, feature_label = _build_gene_features(
        args.feature_mode,
        names=train_gene,
        states=train_cond,
        model=model,
        gene_names=gene_names,
        cfg=cfg,
    )
    test_feat, _ = _build_gene_features(
        args.feature_mode,
        names=test_gene,
        states=test_cond,
        model=model,
        gene_names=gene_names,
        cfg=cfg,
    )
    feat_dim = train_feat.shape[1]
    _log(f"Feature mode '{args.feature_mode}': feat_dim={feat_dim}")

    # Pair each perturbed cell with a same-condition control for the head input.
    train_ctrl_idx = sample_matched_controls(train_cond, control_cond, rng)
    test_ctrl_idx = sample_matched_controls(test_cond, control_cond, rng)

    _log(f"Training perturbation head on {train_z.shape[0]:,} cells, feat_dim={feat_dim}")
    head = train_perturbation_head(
        control_z[train_ctrl_idx],
        train_z,
        train_feat,
        hidden_dim=cfg.eval.perturb_head_hidden,
        epochs=cfg.eval.perturb_head_epochs,
        device=device,
    )
    pred_z = predict_head(head, control_z[test_ctrl_idx], test_feat, device=device)
    pred_expr = decode_latent(pred_z, decoder)
    head_prediction = _group_metrics(
        test_z, test_expr, test_cond, test_gene,
        pred_expr=pred_expr, decoder=decoder, ctrl_means=ctrl_means, top_ks=top_ks, test_genes=test_genes,
    )

    # ---------------- Baseline: gene-agnostic mean train delta -------------------
    _log("Baseline: mean train delta (gene-agnostic predictor)")
    train_ctrl_expr_mean, _ = matched_control_means(ctrl_means, train_cond)
    mean_train_delta = train_expr.mean(axis=0) - train_ctrl_expr_mean

    def baseline_for(mask: np.ndarray) -> dict | None:
        if mask.sum() == 0:
            return None
        ctrl_expr_mean, _ = matched_control_means(ctrl_means, test_cond[mask])
        true_delta = test_expr[mask].mean(axis=0) - ctrl_expr_mean
        from jepa_poc.eval.metrics import delta_pearson, precision_at_k

        r = {"delta_pearson": delta_pearson(mean_train_delta, true_delta)}
        for k in top_ks:
            r[f"precision_at_{k}"] = precision_at_k(mean_train_delta, true_delta, k)
        return r

    baseline = {"overall": baseline_for(np.ones(len(test_cond), dtype=bool)), "per_gene": {}}
    for g in test_genes:
        r = baseline_for(test_gene.astype(str) == g)
        if r is not None:
            baseline["per_gene"][g] = r

    results = {
        "n_test_cells": int(test_z.shape[0]),
        "n_train_head_cells": int(train_z.shape[0]),
        "n_control_cells": int(control_z.shape[0]),
        "test_perturbations": test_genes,
        "feature_mode": args.feature_mode,
        "feature_label": feature_label,
        "feature_dim": int(feat_dim),
        "representation_quality": representation,
        "head_prediction": head_prediction,
        "baseline_mean_train_delta": baseline,
    }
    out = run_dir / f"perturbation_results_{args.feature_mode}.json"
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    _log(f"Wrote {out}")


if __name__ == "__main__":
    main()
