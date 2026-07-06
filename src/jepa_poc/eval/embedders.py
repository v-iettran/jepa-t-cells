"""Pluggable embedders for the consolidated benchmark.

Every embedder turns a :class:`~jepa_poc.eval.pools.CellPool` into an ``[n, d]``
latent matrix. The downstream tasks (decode / retrieval / stratification) never
know which model produced the latents, so the comparison is apples-to-apples.

Backends
--------
* ``jepa_orig`` : Experiment-1 ``GeneTokenEncoder`` (nn.Embedding gene identity).
* ``jepa_esm``  : Experiment-3 ``ESMGeneTokenEncoder`` (Arm 1 / Arm 2).
* ``pca``       : sklearn PCA fit on the same pretrain cells as the arms.
* ``scvi``      : scVI VAE latent, trained on the same pretrain cells.

The two JEPA backends take the CLS token of the full-gene forward pass; PCA and
scVI are fit on the ``fit`` pool (pretrain subsample) before embedding.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from jepa_poc.eval.pools import CellPool
from jepa_poc.models.encoder import GeneTokenEncoder
from jepa_poc.models.gene_tokenizer import ESMGeneTokenEncoder


def _log(msg: str) -> None:
    import time

    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


@torch.no_grad()
def _torch_embed(
    encoder: torch.nn.Module,
    pool: CellPool,
    device: str,
    batch_size: int = 256,
) -> np.ndarray:
    encoder.eval().to(device)
    out: list[np.ndarray] = []
    n = pool.n
    use_cuda = device == "cuda"
    for i in range(0, n, batch_size):
        values = torch.as_tensor(pool.values[i : i + batch_size], dtype=torch.float32, device=device)
        batch_id = torch.as_tensor(pool.batch_id[i : i + batch_size], dtype=torch.long, device=device)
        with torch.autocast(device_type="cuda" if use_cuda else "cpu", dtype=torch.bfloat16, enabled=use_cuda):
            z = encoder(values, batch_id)[:, 0, :]
        out.append(z.float().cpu().numpy())
    return np.concatenate(out, axis=0)


class Embedder:
    """Base class. ``fit`` is a no-op for the pretrained JEPA encoders."""

    label: str
    latent_dim: int

    def fit(self, fit_pool: CellPool) -> None:  # noqa: D401
        return None

    def embed(self, pool: CellPool) -> np.ndarray:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Experiment 1 JEPA (GeneTokenEncoder)
# --------------------------------------------------------------------------- #
class JepaOrigEmbedder(Embedder):
    def __init__(self, cfg, n_batches: int, *, encoder_path: Path | None = None, checkpoint: Path | None = None,
                 device: str = "cuda", batch_size: int = 256, label: str = "exp1"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.label = label
        self.latent_dim = int(cfg.model.d_model)
        self.encoder = GeneTokenEncoder(
            n_genes=int(cfg.data.hvg_count),
            n_batches=n_batches,
            d_model=int(cfg.model.d_model),
            n_layers=int(cfg.model.n_layers),
            n_heads=int(cfg.model.n_heads),
            dropout=float(cfg.model.dropout),
        )
        self._load(encoder_path, checkpoint)

    def _load(self, encoder_path: Path | None, checkpoint: Path | None) -> None:
        if encoder_path and Path(encoder_path).exists():
            state = torch.load(encoder_path, map_location="cpu", weights_only=False)
            report = self.encoder.load_state_dict(state, strict=False)
            _log(f"[{self.label}] loaded encoder {encoder_path}; load_report={report}")
            return
        if checkpoint and Path(checkpoint).exists():
            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
            state = ckpt.get("state_dict", ckpt)
            sub = {k.removeprefix("model.target_encoder."): v for k, v in state.items()
                   if k.startswith("model.target_encoder.")}
            if not sub:
                sub = {k.removeprefix("model.context_encoder."): v for k, v in state.items()
                       if k.startswith("model.context_encoder.")}
            report = self.encoder.load_state_dict(sub, strict=False)
            _log(f"[{self.label}] loaded {checkpoint} (step={ckpt.get('global_step','?')}); load_report={report}")
            return
        raise FileNotFoundError("JepaOrigEmbedder needs encoder_path or checkpoint")

    def embed(self, pool: CellPool) -> np.ndarray:
        return _torch_embed(self.encoder, pool, self.device, self.batch_size)


# --------------------------------------------------------------------------- #
# Experiment 3 JEPA (ESMGeneTokenEncoder, Arm 1 / Arm 2)
# --------------------------------------------------------------------------- #
class JepaEsmEmbedder(Embedder):
    def __init__(self, cfg, n_batches: int, *, arm: str, esm_table, fallback_mask,
                 encoder_path: Path | None = None, checkpoint: Path | None = None,
                 device: str = "cuda", batch_size: int = 256, label: str | None = None):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.arm = arm
        self.label = label or arm
        self.latent_dim = int(cfg.model.d_model)
        self.encoder = ESMGeneTokenEncoder(
            esm_embeddings=esm_table,
            fallback_mask=fallback_mask,
            n_batches=n_batches,
            d_model=int(cfg.model.d_model),
            d_id_proj=int(cfg.model.d_id_proj),
            d_expr=int(cfg.model.d_expr),
            use_fallback_indicator=bool(cfg.model.use_fallback_indicator),
            n_layers=int(cfg.model.n_layers),
            n_heads=int(cfg.model.n_heads),
            dropout=float(cfg.model.dropout),
        )
        self._load(encoder_path, checkpoint)

    def _load(self, encoder_path: Path | None, checkpoint: Path | None) -> None:
        if encoder_path and Path(encoder_path).exists():
            state = torch.load(encoder_path, map_location="cpu", weights_only=False)
            report = self.encoder.load_state_dict(state, strict=False)
            _log(f"[{self.label}] loaded encoder {encoder_path}; load_report={report}")
            return
        if checkpoint and Path(checkpoint).exists():
            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
            state = ckpt.get("state_dict", ckpt)
            # Arm 1 -> EMA teacher; Arm 2 -> shared symmetric encoder.
            prefix = "model.target_encoder." if self.arm == "arm1" else "model.context_encoder."
            sub = {k.removeprefix(prefix): v for k, v in state.items() if k.startswith(prefix)}
            if not sub:
                alt = "model.context_encoder." if self.arm == "arm1" else "model.target_encoder."
                sub = {k.removeprefix(alt): v for k, v in state.items() if k.startswith(alt)}
            report = self.encoder.load_state_dict(sub, strict=False)
            _log(f"[{self.label}] loaded {checkpoint} (step={ckpt.get('global_step','?')}); load_report={report}")
            return
        raise FileNotFoundError("JepaEsmEmbedder needs encoder_path or checkpoint")

    def embed(self, pool: CellPool) -> np.ndarray:
        return _torch_embed(self.encoder, pool, self.device, self.batch_size)


# --------------------------------------------------------------------------- #
# PCA baseline
# --------------------------------------------------------------------------- #
class PCAEmbedder(Embedder):
    def __init__(self, n_components: int = 50, label: str = "pca", seed: int = 13):
        self.latent_dim = n_components
        self.label = label
        self.seed = seed
        self._pca = None

    def fit(self, fit_pool: CellPool) -> None:
        from sklearn.decomposition import PCA

        _log(f"[{self.label}] fitting PCA-{self.latent_dim} on {fit_pool.n:,} cells")
        self._pca = PCA(n_components=self.latent_dim, svd_solver="randomized", random_state=self.seed)
        self._pca.fit(fit_pool.values)
        evr = float(self._pca.explained_variance_ratio_.sum())
        _log(f"[{self.label}] PCA fit done; cumulative explained variance = {evr:.3f}")

    def embed(self, pool: CellPool) -> np.ndarray:
        if self._pca is None:
            raise RuntimeError("PCAEmbedder.fit must be called before embed")
        return self._pca.transform(pool.values).astype(np.float32)


# --------------------------------------------------------------------------- #
# scVI baseline
# --------------------------------------------------------------------------- #
class ScviEmbedder(Embedder):
    """scVI VAE latent. Trained on the pretrain ``fit`` pool with batch correction.

    The model's batch categories are set to the union of all batches across the
    fit + eval pools so donor-4 controls/test cells (which the JEPA models also
    only see at eval) get a valid batch code at inference time.
    """

    def __init__(self, cfg, n_latent: int = 10, *, batch_key: str = "batch_str",
                 all_batches: list[str] | None = None, max_epochs: int = 100,
                 label: str = "scvi", seed: int = 13):
        self.latent_dim = n_latent
        self.label = label
        self.seed = seed
        self.batch_key = batch_key
        self.all_batches = all_batches
        self.max_epochs = max_epochs
        self._model = None
        self._categories: list[str] | None = None

    def _to_anndata(self, pool: CellPool):
        import anndata as ad
        import pandas as pd

        obs = pd.DataFrame({"batch_str": pool.batch_str.astype(str)})
        if self._categories is not None:
            obs["batch_str"] = pd.Categorical(obs["batch_str"], categories=self._categories)
        a = ad.AnnData(X=pool.raw.copy(), obs=obs)
        return a

    def fit(self, fit_pool: CellPool) -> None:
        import scvi
        import scanpy as sc  # noqa: F401  (import side effects / availability check)

        scvi.settings.seed = self.seed
        cats = sorted(set(map(str, fit_pool.batch_str)) | set(map(str, self.all_batches or [])))
        self._categories = cats
        adata = self._to_anndata(fit_pool)
        scvi.model.SCVI.setup_anndata(adata, batch_key="batch_str")
        _log(f"[{self.label}] training scVI (n_latent={self.latent_dim}) on {fit_pool.n:,} cells, "
             f"{len(cats)} batch categories, max_epochs={self.max_epochs}")
        model = scvi.model.SCVI(adata, n_latent=self.latent_dim)
        model.train(max_epochs=self.max_epochs)
        self._model = model
        _log(f"[{self.label}] scVI training done")

    def embed(self, pool: CellPool) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("ScviEmbedder.fit must be called before embed")
        adata = self._to_anndata(pool)
        # Register the query adata against the trained model, then read latents.
        self._model.__class__.setup_anndata(adata, batch_key="batch_str")
        z = self._model.get_latent_representation(adata)
        return np.asarray(z, dtype=np.float32)
