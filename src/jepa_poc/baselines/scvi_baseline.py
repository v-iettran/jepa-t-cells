from __future__ import annotations

from anndata import AnnData


class SCVIBaseline:
    """Thin scvi-tools wrapper kept optional so smoke tests do not require scvi import."""

    def __init__(self, n_latent: int = 10, batch_key: str = "batch", max_epochs: int = 20) -> None:
        self.n_latent = n_latent
        self.batch_key = batch_key
        self.max_epochs = max_epochs
        self.model = None

    def fit(self, adata: AnnData) -> "SCVIBaseline":
        try:
            import scvi
        except ImportError as exc:
            raise ImportError("scvi-tools is required for SCVIBaseline. Install requirements.txt.") from exc
        scvi.model.SCVI.setup_anndata(adata, batch_key=self.batch_key if self.batch_key in adata.obs else None)
        self.model = scvi.model.SCVI(adata, n_latent=self.n_latent)
        self.model.train(max_epochs=self.max_epochs)
        return self

    def embed(self, adata: AnnData):
        if self.model is None:
            raise RuntimeError("SCVIBaseline.fit must be called before embed.")
        return self.model.get_latent_representation(adata)
