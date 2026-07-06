"""Shared cell-pool loader for the consolidated benchmark (benchmarking.md).

Every model in the comparison (Experiment-1 JEPA, Arm 1, Arm 2, PCA, scVI) is
scored on the *same* cells. This module loads those cells once into a model-
agnostic form so the embedding step is the only thing that differs per model:

  * control : NTC cells (perturb_controls_processed.h5ad), condition-stratified
              subsample. Used to fit the z->expression decoders and as the
              condition-matched control reference for every delta.
  * train   : targeting cells with split=="train" (perturbations SEEN during
              pretraining). Used for retrieval signatures + effect-size strata.
  * test    : targeting cells with split=="test" (the 5 strict held-out genes).
  * fit     : a subsample of the pretrain cells the JEPA models trained on,
              used ONLY to fit the unsupervised baselines (PCA / scVI) so they
              see the same data as the arms.

Each pool exposes both the normalized log-CPM matrix (JEPA / PCA input + decode
target) and the raw counts (scVI input), plus condition / gene / batch metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse

from jepa_poc.data.loader import Encoders, fit_encoders, normalize_log_cpm


@dataclass
class CellPool:
    """A set of cells in model-agnostic form."""

    values: np.ndarray            # [n, G] normalized log-CPM (float32)
    raw: sparse.csr_matrix        # [n, G] raw counts (for scVI)
    cond: np.ndarray              # [n] culture_condition (str)
    gene: np.ndarray              # [n] perturbed gene name (str)
    batch_id: np.ndarray          # [n] JEPA batch vocab id (int64)
    batch_str: np.ndarray         # [n] raw 10xrun_id (str, for scVI)

    @property
    def n(self) -> int:
        return int(self.values.shape[0])


@dataclass
class EvalPools:
    control: CellPool
    train: CellPool
    test: CellPool
    encoders: Encoders
    n_batches: int
    n_genes: int
    gene_names: list[str]


def _as_csr(x) -> sparse.csr_matrix:
    if sparse.issparse(x):
        return x.tocsr().astype(np.float32, copy=False)
    return sparse.csr_matrix(np.asarray(x, dtype=np.float32))


def _gene_names(adata: ad.AnnData) -> list[str]:
    if "gene_symbol" in adata.var:
        return adata.var["gene_symbol"].astype(str).tolist()
    if "gene_name" in adata.var:
        return adata.var["gene_name"].astype(str).tolist()
    return list(map(str, adata.var_names))


def _stratified_subsample(cond: np.ndarray, cap: int, rng: np.random.Generator) -> np.ndarray:
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


def _subsample(n: int, cap: int, rng: np.random.Generator) -> np.ndarray:
    if cap <= 0 or n <= cap:
        return np.arange(n)
    return np.sort(rng.choice(n, size=cap, replace=False))


def _make_pool(
    adata: ad.AnnData,
    encoders: Encoders,
    *,
    batch_key: str,
    pkey: str,
) -> CellPool:
    raw = _as_csr(adata.X)
    values = normalize_log_cpm(np.asarray(raw.toarray(), dtype=np.float32))
    cond = adata.obs["culture_condition"].astype(str).to_numpy()
    gene = (
        adata.obs[pkey].astype(str).to_numpy()
        if pkey in adata.obs
        else np.full(adata.n_obs, "control", dtype=object)
    )
    if batch_key in adata.obs:
        batch_str = adata.obs[batch_key].astype(str).to_numpy()
        batch_id = np.array(
            [encoders.batch_to_id.get(b, 0) for b in batch_str], dtype=np.int64
        )
    else:
        batch_str = np.full(adata.n_obs, "unknown", dtype=object)
        batch_id = np.zeros(adata.n_obs, dtype=np.int64)
    return CellPool(values=values, raw=raw, cond=cond, gene=gene, batch_id=batch_id, batch_str=batch_str)


def load_eval_pools(
    cfg,
    *,
    n_control: int = 150_000,
    n_train: int = 200_000,
    seed: int | None = None,
) -> EvalPools:
    """Load the control / train / test pools shared by every model."""

    rng = np.random.default_rng(int(cfg.seed) if seed is None else seed)
    pkey = cfg.data.perturbation_key
    batch_key = cfg.data.batch_key

    eval_path = Path(cfg.data.perturb_eval_processed_path)
    control_path = Path(cfg.data.perturb_controls_processed_path)
    if not eval_path.exists() or not control_path.exists():
        raise FileNotFoundError("Processed perturbation files not found; run scripts/prep_data.py first.")

    eval_adata = ad.read_h5ad(eval_path)
    control_adata = ad.read_h5ad(control_path)

    # Shared batch / label / perturbation vocab (matches the existing eval scripts).
    combined_obs = ad.concat([control_adata, eval_adata], join="inner").obs
    encoders = fit_encoders(combined_obs, batch_key, "annotation_group", pkey)
    n_batches = max(1, len(encoders.batch_to_id))
    gene_names = _gene_names(eval_adata)

    # ----- control (condition-stratified) -----
    ctrl_cond_all = control_adata.obs["culture_condition"].astype(str).to_numpy()
    ctrl_idx = _stratified_subsample(ctrl_cond_all, n_control, rng)
    control = _make_pool(control_adata[ctrl_idx].copy(), encoders, batch_key=batch_key, pkey=pkey)

    # ----- train targeting (split=="train") -----
    train_mask = eval_adata.obs["split"].astype(str).to_numpy() == "train"
    train_pos = np.where(train_mask)[0]
    sel = _subsample(train_pos.size, n_train, rng)
    train = _make_pool(eval_adata[train_pos[sel]].copy(), encoders, batch_key=batch_key, pkey=pkey)

    # ----- test targeting (split=="test", held-out genes) -----
    test_mask = eval_adata.obs["split"].astype(str).to_numpy() == "test"
    test = _make_pool(eval_adata[np.where(test_mask)[0]].copy(), encoders, batch_key=batch_key, pkey=pkey)

    return EvalPools(
        control=control,
        train=train,
        test=test,
        encoders=encoders,
        n_batches=n_batches,
        n_genes=eval_adata.n_vars,
        gene_names=gene_names,
    )


def load_fit_pool(
    cfg,
    encoders: Encoders,
    *,
    n_fit: int = 500_000,
    seed: int | None = None,
) -> CellPool:
    """Subsample of the pretrain cells the JEPA models trained on (for PCA/scVI).

    Uses split=="train" of pretrain_processed.h5ad so the baselines are fit on
    exactly the data the arms saw (not the held-out validation cells).
    """

    rng = np.random.default_rng((int(cfg.seed) if seed is None else seed) + 1)
    pretrain_path = Path(cfg.data.pretrain_processed_path)
    if not pretrain_path.exists():
        raise FileNotFoundError(f"pretrain file not found: {pretrain_path}")
    pkey = cfg.data.perturbation_key
    batch_key = cfg.data.batch_key

    adata = ad.read_h5ad(pretrain_path)
    if "split" in adata.obs:
        train_pos = np.where(adata.obs["split"].astype(str).to_numpy() == "train")[0]
    else:
        train_pos = np.arange(adata.n_obs)
    sel = _subsample(train_pos.size, n_fit, rng)
    return _make_pool(adata[train_pos[sel]].copy(), encoders, batch_key=batch_key, pkey=pkey)
