from __future__ import annotations

import pytest

from jepa_poc.data.loader import TCellDataset, fit_encoders
from jepa_poc.data.synthetic import make_synthetic_anndata


@pytest.fixture()
def synthetic_adata():
    return make_synthetic_anndata(n_cells=256, n_genes=96, n_labels=4, n_batches=3, seed=7)


@pytest.fixture()
def synthetic_dataset(synthetic_adata):
    encoders = fit_encoders(synthetic_adata.obs)
    return TCellDataset(synthetic_adata, encoders=encoders)
