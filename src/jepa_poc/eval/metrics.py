from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score


def annotation_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class_f1": f1_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def pearsonr_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    denom = np.sqrt((a**2).sum(axis=1) * (b**2).sum(axis=1))
    denom[denom == 0] = np.nan
    return (a * b).sum(axis=1) / denom


def delta_pearson(pred_delta: np.ndarray, true_delta: np.ndarray) -> float:
    return float(np.nanmean(pearsonr_rows(np.atleast_2d(pred_delta), np.atleast_2d(true_delta))))


def precision_at_k(pred_delta: np.ndarray, true_delta: np.ndarray, k: int = 100) -> float:
    pred_rank = np.argsort(-np.abs(pred_delta))[:k]
    true_rank = np.argsort(-np.abs(true_delta))[:k]
    y_true = np.zeros_like(pred_delta, dtype=int)
    y_pred = np.zeros_like(pred_delta, dtype=int)
    y_true[true_rank] = 1
    y_pred[pred_rank] = 1
    return float(precision_score(y_true, y_pred, zero_division=0))
