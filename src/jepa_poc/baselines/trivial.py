from __future__ import annotations

import numpy as np


class IdentityPredictor:
    """Predict no perturbation effect."""

    def predict(self, control: np.ndarray) -> np.ndarray:
        return np.asarray(control)


class MeanPerturbedPredictor:
    """Predict the average perturbation effect seen in training perturbations."""

    def __init__(self) -> None:
        self.mean_delta: np.ndarray | None = None

    def fit(self, control: np.ndarray, perturbed: np.ndarray) -> "MeanPerturbedPredictor":
        self.mean_delta = np.asarray(perturbed).mean(axis=0) - np.asarray(control).mean(axis=0)
        return self

    def predict(self, control: np.ndarray) -> np.ndarray:
        if self.mean_delta is None:
            raise RuntimeError("MeanPerturbedPredictor.fit must be called before predict.")
        return np.asarray(control) + self.mean_delta
