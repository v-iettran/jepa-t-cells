from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from jepa_poc.eval.metrics import annotation_metrics


@torch.no_grad()
def embed_loader(model, loader: DataLoader, device: str | torch.device = "cpu") -> tuple[np.ndarray, np.ndarray]:
    model.to(device)
    model.eval()
    embeddings: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for batch in loader:
        values = batch["values"].to(device)
        batch_id = batch["batch_id"].to(device)
        z = model.encode(values, batch_id, use_target=True)
        embeddings.append(z.cpu().numpy())
        labels.append(batch["label"].numpy())
    return np.concatenate(embeddings), np.concatenate(labels)


def run_linear_probe(
    train_z: np.ndarray,
    train_y: np.ndarray,
    test_z: np.ndarray,
    test_y: np.ndarray,
    c_grid: list[float] | tuple[float, ...] = (0.1, 1.0, 10.0),
) -> dict[str, object]:
    best: dict[str, object] | None = None
    for c in c_grid:
        clf = LogisticRegression(C=c, max_iter=2000, class_weight="balanced", multi_class="auto")
        clf.fit(train_z, train_y)
        pred = clf.predict(test_z)
        metrics = annotation_metrics(test_y, pred)
        metrics["C"] = c
        if best is None or metrics["macro_f1"] > best["macro_f1"]:
            best = metrics
    assert best is not None
    return best


def run_knn_probe(train_z: np.ndarray, train_y: np.ndarray, test_z: np.ndarray, test_y: np.ndarray, k: int = 15) -> dict[str, object]:
    clf = KNeighborsClassifier(n_neighbors=k, metric="cosine")
    clf.fit(train_z, train_y)
    pred = clf.predict(test_z)
    metrics = annotation_metrics(test_y, pred)
    metrics["k"] = k
    return metrics
