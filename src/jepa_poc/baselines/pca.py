from __future__ import annotations

import numpy as np
from anndata import AnnData
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge

from jepa_poc.data.loader import normalize_log_cpm


def _matrix(adata: AnnData) -> np.ndarray:
    x = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
    return normalize_log_cpm(x.astype(np.float32))


class PCABaseline:
    def __init__(self, n_components: int = 50, seed: int = 13) -> None:
        self.pca = PCA(n_components=n_components, random_state=seed)

    def fit(self, adata: AnnData) -> "PCABaseline":
        self.pca.fit(_matrix(adata))
        return self

    def embed(self, adata: AnnData) -> np.ndarray:
        return self.pca.transform(_matrix(adata))


def fit_pca_logistic(train_z: np.ndarray, train_y: np.ndarray, c: float = 1.0) -> LogisticRegression:
    clf = LogisticRegression(C=c, max_iter=2000, class_weight="balanced", multi_class="auto")
    clf.fit(train_z, train_y)
    return clf


def fit_pca_ridge(train_z: np.ndarray, train_y: np.ndarray, alpha: float = 1.0) -> Ridge:
    reg = Ridge(alpha=alpha)
    reg.fit(train_z, train_y)
    return reg
