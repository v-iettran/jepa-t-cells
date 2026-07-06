"""Build per-state GENIE3-style GRNs and graph embeddings for Experiment 2."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import OmegaConf
from scipy import sparse
from sklearn.ensemble import RandomForestRegressor

from jepa_poc.config import ensure_dir, load_config


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _configured_path(cfg, key: str, default_name: str) -> Path:
    configured = getattr(cfg.data, key, None)
    if configured:
        return Path(configured)
    return Path(cfg.data.output_dir) / default_name


def _to_dense_float32(x: Any) -> np.ndarray:
    if sparse.issparse(x):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


def _normalize_cpm(x: np.ndarray, target_sum: float = 1_000_000.0) -> np.ndarray:
    totals = x.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    return ((x / totals) * target_sum).astype(np.float32)


def _gene_names(adata: ad.AnnData) -> np.ndarray:
    if "gene_symbol" in adata.var:
        return adata.var["gene_symbol"].astype(str).to_numpy()
    if "gene_name" in adata.var:
        return adata.var["gene_name"].astype(str).to_numpy()
    return np.asarray([str(g) for g in adata.var_names])


def _qc_mask(adata: ad.AnnData, cfg) -> np.ndarray:
    obs = adata.obs
    keep = np.ones(adata.n_obs, dtype=bool)
    low_quality_key = cfg.data.low_quality_key
    if low_quality_key in obs:
        keep &= ~obs[low_quality_key].fillna(False).astype(bool).to_numpy()

    min_genes = int(getattr(cfg.grn, "min_genes_by_counts", 200))
    if "n_genes_by_counts" in obs:
        keep &= obs["n_genes_by_counts"].fillna(0).to_numpy() >= min_genes
    else:
        X = adata.X
        n_genes = np.asarray((X > 0).sum(axis=1)).reshape(-1) if sparse.issparse(X) else (np.asarray(X) > 0).sum(axis=1)
        keep &= n_genes >= min_genes

    max_mt = float(getattr(cfg.grn, "max_pct_counts_mt", 15.0))
    if "pct_counts_mt" in obs:
        keep &= obs["pct_counts_mt"].fillna(100.0).to_numpy() <= max_mt
    else:
        names = np.char.upper(_gene_names(adata).astype(str))
        mt_mask = np.char.startswith(names, "MT-")
        if mt_mask.any():
            X = adata.X
            totals = np.asarray(X.sum(axis=1)).reshape(-1) if sparse.issparse(X) else np.asarray(X).sum(axis=1)
            mt = np.asarray(X[:, mt_mask].sum(axis=1)).reshape(-1) if sparse.issparse(X) else np.asarray(X[:, mt_mask]).sum(axis=1)
            totals[totals == 0] = 1.0
            keep &= (mt / totals * 100.0) <= max_mt
        else:
            _log("WARNING: pct_counts_mt missing and no MT-* genes found in HVG matrix; mitochondrial QC cannot be recomputed.")
    return keep


def _subsample_state(adata: ad.AnnData, cfg, state: str) -> tuple[ad.AnnData, dict[str, int]]:
    rng = np.random.default_rng(int(getattr(cfg.grn, "seed", 1)))
    donor_key = cfg.data.donor_key
    cond = adata.obs["culture_condition"].astype(str).to_numpy()
    donor = adata.obs[donor_key].astype(str).to_numpy()
    state_mask = cond == state
    cap = int(getattr(cfg.grn, "per_donor_cap", 12500))
    selected: list[np.ndarray] = []
    donor_counts: dict[str, int] = {}
    for d in sorted(np.unique(donor[state_mask])):
        pool = np.where(state_mask & (donor == d))[0]
        take = min(cap, pool.size)
        chosen = pool if pool.size <= cap else np.sort(rng.choice(pool, size=take, replace=False))
        selected.append(chosen)
        donor_counts[d] = int(take)
    if not selected:
        raise RuntimeError(f"No NTC cells available for state {state!r} after QC")
    idx = np.sort(np.concatenate(selected))
    return adata[idx].copy(), donor_counts


def _fit_target_importance(
    x: np.ndarray,
    target_idx: int,
    *,
    n_trees: int,
    max_features,
    seed: int,
) -> np.ndarray:
    y = x[:, target_idx]
    rf = RandomForestRegressor(
        n_estimators=n_trees,
        max_features=max_features,
        random_state=seed + target_idx,
        n_jobs=1,
    )
    rf.fit(x, y)
    imp = rf.feature_importances_.astype(np.float32)
    imp[target_idx] = 0.0
    return imp


def _run_genie3_like(x: np.ndarray, cfg) -> np.ndarray:
    n_trees = int(getattr(cfg.grn, "n_trees", 1000))
    max_features = getattr(cfg.grn, "max_features", "sqrt")
    if isinstance(max_features, str) and max_features.isdigit():
        max_features = int(max_features)
    seed = int(getattr(cfg.grn, "seed", 1))
    n_jobs = int(getattr(cfg.grn, "n_jobs", -1))
    _log(f"Fitting GENIE3-style forests for {x.shape[1]} targets (n_trees={n_trees}, max_features={max_features})")
    rows = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
        delayed(_fit_target_importance)(x, i, n_trees=n_trees, max_features=max_features, seed=seed)
        for i in range(x.shape[1])
    )
    # Stored target x regulator. Edges are regulator -> target.
    return np.stack(rows, axis=0).astype(np.float32)


def _write_importance_edges(matrix: np.ndarray, genes: list[str], path: Path, top_k: int) -> None:
    rows = []
    for target_idx, weights in enumerate(matrix):
        if top_k > 0 and top_k < weights.size:
            regulator_idx = np.argpartition(weights, -top_k)[-top_k:]
            regulator_idx = regulator_idx[np.argsort(weights[regulator_idx])[::-1]]
        else:
            regulator_idx = np.argsort(weights)[::-1]
        for reg_idx in regulator_idx:
            weight = float(weights[reg_idx])
            if weight <= 0.0:
                continue
            rows.append(
                {
                    "regulator": genes[reg_idx],
                    "target": genes[target_idx],
                    "regulator_idx": int(reg_idx),
                    "target_idx": int(target_idx),
                    "importance": weight,
                }
            )
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except ImportError:
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        _log(f"pyarrow/fastparquet unavailable; wrote {csv_path} instead of parquet")


def _node2vec_embeddings(matrix: np.ndarray, genes: list[str], cfg) -> np.ndarray:
    top_k = int(getattr(cfg.grn, "top_k_edges_per_target", 10))
    dim = int(getattr(cfg.grn, "embedding_dim", 128))
    graph = nx.DiGraph()
    graph.add_nodes_from(range(len(genes)))
    for target_idx, weights in enumerate(matrix):
        regulator_idx = np.argpartition(weights, -top_k)[-top_k:] if top_k < weights.size else np.arange(weights.size)
        for reg_idx in regulator_idx:
            weight = float(weights[reg_idx])
            if weight > 0:
                graph.add_edge(int(reg_idx), int(target_idx), weight=weight)

    try:
        from node2vec import Node2Vec

        node2vec = Node2Vec(
            graph,
            dimensions=dim,
            walk_length=int(getattr(cfg.grn, "node2vec_walk_length", 80)),
            num_walks=int(getattr(cfg.grn, "node2vec_num_walks", 10)),
            workers=max(1, int(getattr(cfg.grn, "n_jobs", -1))),
            seed=int(getattr(cfg.grn, "seed", 1)),
            quiet=True,
        )
        model = node2vec.fit(
            window=int(getattr(cfg.grn, "node2vec_window", 10)),
            min_count=1,
            batch_words=4,
            epochs=int(getattr(cfg.grn, "node2vec_epochs", 5)),
        )
        emb = np.zeros((len(genes), dim), dtype=np.float32)
        for idx in range(len(genes)):
            key = str(idx)
            if key in model.wv:
                emb[idx] = model.wv[key]
        return emb
    except Exception as exc:
        _log(f"WARNING: node2vec unavailable or failed ({exc}); falling back to SVD of GENIE3 importance matrix.")
        u, s, _ = np.linalg.svd(matrix, full_matrices=False)
        k = min(dim, u.shape[1])
        emb = np.zeros((matrix.shape[0], dim), dtype=np.float32)
        emb[:, :k] = (u[:, :k] * s[:k]).astype(np.float32)
        return emb


def _heldout_genes(cfg) -> list[str]:
    path = getattr(cfg.data, "heldout_genes_path", None)
    if path and Path(path).exists():
        payload = json.loads(Path(path).read_text())
        if isinstance(payload, dict):
            return list(map(str, payload.get("experiment_2_heldout_genes", [])))
        return list(map(str, payload))
    return list(map(str, getattr(cfg.data, "heldout_perturbation_genes", [])))


def _write_state_divergence(matrices: dict[str, np.ndarray], genes: list[str], cfg) -> None:
    if len(matrices) < 2:
        return
    gene_to_idx = {g.upper(): i for i, g in enumerate(genes)}
    heldout = [g for g in _heldout_genes(cfg) if g.upper() in gene_to_idx]
    rows = []
    states = sorted(matrices)
    for gene in heldout:
        target_idx = gene_to_idx[gene.upper()]
        top_by_state = {}
        for state, matrix in matrices.items():
            weights = matrix[target_idx]
            top = np.argpartition(weights, -20)[-20:]
            top_by_state[state] = set(map(int, top[weights[top] > 0]))
        for i, state_a in enumerate(states):
            for state_b in states[i + 1 :]:
                a = top_by_state[state_a]
                b = top_by_state[state_b]
                union = a | b
                jaccard = len(a & b) / len(union) if union else 0.0
                rows.append(
                    {
                        "gene": gene,
                        "state_a": state_a,
                        "state_b": state_b,
                        "top20_regulator_jaccard": jaccard,
                        "n_top_a": len(a),
                        "n_top_b": len(b),
                    }
                )
    out_dir = ensure_dir("outputs/grn")
    pd.DataFrame(rows).to_csv(out_dir / "heldout_gene_state_divergence.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/exp2.yaml")
    parser.add_argument("--state", choices=["Rest", "Stim8hr", "Stim48hr"], default=None)
    parser.add_argument("--skip-genie3", action="store_true", help="Only rebuild embeddings from existing importance matrix.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = ensure_dir(getattr(cfg.grn, "output_dir", "data/grn"))
    control_path = _configured_path(cfg, "perturb_controls_processed_path", "perturb_controls_processed.h5ad")
    eval_path = _configured_path(cfg, "perturb_eval_processed_path", "perturb_eval_processed.h5ad")
    if not control_path.exists():
        raise FileNotFoundError(f"{control_path} not found. Run scripts/prep_data.py first.")

    _log(f"Loading controls from {control_path}")
    controls = ad.read_h5ad(control_path)
    genes = _gene_names(controls).astype(str).tolist()
    ntc_mask = controls.obs[cfg.data.perturbation_key].astype(str).to_numpy() == cfg.data.control_label
    if cfg.data.guide_type_key in controls.obs:
        ntc_mask &= controls.obs[cfg.data.guide_type_key].astype(str).to_numpy() == cfg.data.non_targeting_guide_type
    qc = _qc_mask(controls, cfg)
    controls = controls[ntc_mask & qc].copy()
    _log(f"NTC+QC controls: {controls.n_obs:,} cells x {controls.n_vars:,} genes")

    states = [args.state] if args.state else list(getattr(cfg.grn, "states", ["Rest", "Stim8hr", "Stim48hr"]))
    metadata: dict[str, Any] = {
        "control_path": str(control_path),
        "genes": genes,
        "states": {},
        "normalization": "CPM",
        "n_trees": int(getattr(cfg.grn, "n_trees", 1000)),
        "max_features": str(getattr(cfg.grn, "max_features", "sqrt")),
    }
    used_barcodes: list[str] = []
    matrices_for_diagnostic: dict[str, np.ndarray] = {}

    for state in states:
        state_slug = state.lower()
        t0 = time.time()
        state_adata, donor_counts = _subsample_state(controls, cfg, state)
        used_barcodes.extend(map(str, state_adata.obs_names))
        _log(f"{state}: selected {state_adata.n_obs:,} cells | donor_counts={donor_counts}")
        importance_path = out_dir / f"genie3_importance_{state_slug}.parquet"
        matrix_path = out_dir / f"genie3_importance_{state_slug}.npy"
        emb_path = out_dir / f"gene_emb_grn_{state_slug}.npy"
        if args.skip_genie3:
            matrix = np.load(matrix_path)
        else:
            x = _to_dense_float32(state_adata.X)
            x = _normalize_cpm(x, float(getattr(cfg.grn, "cpm_target_sum", 1_000_000.0)))
            matrix = _run_genie3_like(x, cfg)
            np.save(matrix_path, matrix)
            _write_importance_edges(matrix, genes, importance_path, int(getattr(cfg.grn, "top_k_edges_per_target", 10)))
        emb = _node2vec_embeddings(matrix, genes, cfg)
        np.save(emb_path, emb)
        matrices_for_diagnostic[state] = matrix
        metadata["states"][state] = {
            "n_cells": int(state_adata.n_obs),
            "donor_counts": donor_counts,
            "importance_matrix": str(matrix_path),
            "importance_edges": str(importance_path),
            "embedding": str(emb_path),
            "elapsed_sec": time.time() - t0,
        }
        _log(f"{state}: wrote {emb_path} shape={emb.shape}")

    used_barcode_set = set(used_barcodes)
    if eval_path.exists():
        eval_adata = ad.read_h5ad(eval_path, backed="r")
        overlap = used_barcode_set.intersection(map(str, eval_adata.obs_names))
        eval_adata.file.close()
        if overlap:
            raise RuntimeError(f"GENIE3 leakage guard failed: {len(overlap)} NTC barcodes overlap perturbation eval cells")

    suffix = f"_{args.state.lower()}" if args.state else ""
    (out_dir / f"genie3_ntc_barcodes{suffix}.txt").write_text("\n".join(sorted(used_barcode_set)) + "\n")
    (out_dir / "node2vec_config.json").write_text(json.dumps(OmegaConf.to_container(cfg.grn, resolve=True), indent=2))
    (out_dir / f"grn_metadata{suffix}.json").write_text(json.dumps(metadata, indent=2))
    _write_state_divergence(matrices_for_diagnostic, genes, cfg)
    _log(f"Wrote metadata -> {out_dir / f'grn_metadata{suffix}.json'}")


if __name__ == "__main__":
    main()
