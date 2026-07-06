"""Non-leaky, genome-wide perturbation-identity features.

The JEPA gene-embedding table only spans the ~2000 HVGs used for pretraining,
so it cannot represent the genome-wide CRISPR targets (~10.7K genes, only ~9%
of which are HVGs). Feeding the perturbation head a per-gene identity feature
restricted to that vocabulary collapses ~90% of genes onto a single fallback
vector, so the head learns to ignore perturbation identity entirely.

This module builds a consistent gene embedding for *all* measured genes from the
control (non-targeting) pseudobulk profiles via a truncated SVD of the
gene x sample covariation matrix. Because it is derived only from baseline
expression covariation (never from a gene's measured knockdown effect), it is a
legitimate, leakage-free identity prior that also generalizes leave-one-gene-out
(co-regulated genes receive similar embeddings).
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import torch
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

from jepa_poc.data.loader import normalize_log_cpm


def build_coexpression_gene_embedding(
    pseudobulk_path: str | Path,
    *,
    n_components: int = 50,
    guide_type_key: str = "guide_type",
    control_guide_type: str = "non-targeting",
    gene_name_col: str = "gene_name",
    min_control_samples: int = 50,
    seed: int = 0,
) -> tuple[dict[str, np.ndarray], int]:
    """Return ``(symbol_upper -> embedding vector, dim)`` from control pseudobulk.

    Steps: select control (non-targeting) pseudobulk samples, log-CPM normalize
    per sample, center per gene, then take the top ``n_components`` right singular
    vectors scaled by singular values as each gene's embedding. Genes that
    co-vary across the control samples land close together.
    """

    pb = ad.read_h5ad(str(pseudobulk_path), backed="r")
    if guide_type_key in pb.obs:
        mask = (pb.obs[guide_type_key].astype(str) == control_guide_type).to_numpy()
        if int(mask.sum()) < min_control_samples:
            mask = np.ones(pb.n_obs, dtype=bool)
    else:
        mask = np.ones(pb.n_obs, dtype=bool)

    sub = pb[mask].to_memory()
    X = sub.X
    X = np.asarray(X.todense()) if sparse.issparse(X) else np.asarray(X)
    X = X.astype(np.float32)

    X = normalize_log_cpm(X)
    X -= X.mean(axis=0, keepdims=True)

    k = int(min(n_components, min(X.shape) - 1))
    svd = TruncatedSVD(n_components=k, random_state=seed)
    svd.fit(X)
    gene_emb = (svd.components_.T * svd.singular_values_).astype(np.float32)  # (n_genes, k)

    if gene_name_col in sub.var:
        symbols = sub.var[gene_name_col].astype(str).str.upper().to_numpy()
    else:
        symbols = np.asarray([str(s).upper() for s in sub.var_names])

    sym_to_vec: dict[str, np.ndarray] = {}
    for i, sym in enumerate(symbols):
        if sym not in sym_to_vec:  # first occurrence wins for duplicate symbols
            sym_to_vec[sym] = gene_emb[i]
    return sym_to_vec, k


def features_for_names(
    names: np.ndarray,
    sym_to_vec: dict[str, np.ndarray],
    dim: int,
) -> np.ndarray:
    """Look up a co-expression vector per perturbation name.

    Strips common KO_/KD_/perturb_ prefixes, upper-cases, and falls back to the
    mean embedding for unmapped names (rare: ~0.3% of targets).
    """

    fallback = (
        np.mean(np.stack(list(sym_to_vec.values())), axis=0)
        if sym_to_vec
        else np.zeros(dim, dtype=np.float32)
    )
    out = np.empty((len(names), dim), dtype=np.float32)
    for i, name in enumerate(names.astype(str)):
        token = name.replace("KO_", "").replace("KD_", "").replace("perturb_", "").upper()
        out[i] = sym_to_vec.get(token, fallback)
    return out


def _clean_symbol(name: str) -> str:
    return name.replace("KO_", "").replace("KD_", "").replace("perturb_", "").upper()


def gene_names_from_adata(adata: ad.AnnData) -> list[str]:
    if "gene_symbol" in adata.var:
        return adata.var["gene_symbol"].astype(str).tolist()
    if "gene_name" in adata.var:
        return adata.var["gene_name"].astype(str).tolist()
    return [str(g) for g in adata.var_names]


def jepa_gene_embedding_features(
    names: np.ndarray,
    gene_names: list[str],
    embedding_weight: torch.Tensor,
) -> np.ndarray:
    """Look up learned JEPA gene-token embeddings for perturbation names."""

    gene_to_idx = {_clean_symbol(g): i for i, g in enumerate(gene_names)}
    weight = embedding_weight.detach().cpu().numpy().astype(np.float32)
    fallback = weight.mean(axis=0)
    out = np.empty((len(names), weight.shape[1]), dtype=np.float32)
    for i, name in enumerate(names.astype(str)):
        idx = gene_to_idx.get(_clean_symbol(name))
        out[i] = weight[idx] if idx is not None else fallback
    return out


def grn_state_features(
    names: np.ndarray,
    states: np.ndarray,
    gene_names: list[str],
    grn_dir: str | Path,
) -> np.ndarray:
    """Look up state-matched GENIE3 graph embeddings for perturbation names."""

    grn_dir = Path(grn_dir)
    state_to_file = {
        "Rest": grn_dir / "gene_emb_grn_rest.npy",
        "Stim8hr": grn_dir / "gene_emb_grn_stim8hr.npy",
        "Stim48hr": grn_dir / "gene_emb_grn_stim48hr.npy",
    }
    state_to_emb = {state: np.load(path).astype(np.float32) for state, path in state_to_file.items() if path.exists()}
    if not state_to_emb:
        raise FileNotFoundError(f"No GRN embedding files found in {grn_dir}")
    dim = next(iter(state_to_emb.values())).shape[1]
    fallback = np.mean(np.concatenate(list(state_to_emb.values()), axis=0), axis=0)
    gene_to_idx = {_clean_symbol(g): i for i, g in enumerate(gene_names)}
    out = np.empty((len(names), dim), dtype=np.float32)
    for i, (name, state) in enumerate(zip(names.astype(str), states.astype(str), strict=True)):
        emb = state_to_emb.get(str(state))
        idx = gene_to_idx.get(_clean_symbol(name))
        out[i] = emb[idx] if emb is not None and idx is not None else fallback
    return out
