from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from jepa_poc.eval.metrics import delta_pearson, precision_at_k


class PerturbationHead(nn.Module):
    def __init__(self, latent_dim: int, feature_dim: int | None = None, hidden_dim: int = 256) -> None:
        super().__init__()
        feature_dim = latent_dim if feature_dim is None else feature_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim + feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, control_z: torch.Tensor, perturbation_feature: torch.Tensor) -> torch.Tensor:
        delta = self.net(torch.cat([control_z, perturbation_feature], dim=-1))
        return control_z + delta


def fit_linear_decoder(z: np.ndarray, x: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    """Closed-form ridge decoder mapping latent vectors to expression."""

    z_aug = np.concatenate([z, np.ones((z.shape[0], 1), dtype=z.dtype)], axis=1)
    lhs = z_aug.T @ z_aug + ridge * np.eye(z_aug.shape[1])
    rhs = z_aug.T @ x
    return np.linalg.solve(lhs, rhs)


def decode_latent(z: np.ndarray, decoder: np.ndarray) -> np.ndarray:
    z_aug = np.concatenate([z, np.ones((z.shape[0], 1), dtype=z.dtype)], axis=1)
    return z_aug @ decoder


def train_perturbation_head(
    control_z: np.ndarray,
    perturbed_z: np.ndarray,
    perturbation_feature: np.ndarray,
    hidden_dim: int = 256,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str | torch.device = "cpu",
) -> PerturbationHead:
    control = torch.as_tensor(control_z, dtype=torch.float32)
    perturbed = torch.as_tensor(perturbed_z, dtype=torch.float32)
    pfeat = torch.as_tensor(perturbation_feature, dtype=torch.float32)
    loader = DataLoader(TensorDataset(control, perturbed, pfeat), batch_size=batch_size, shuffle=True)
    head = PerturbationHead(control.shape[1], pfeat.shape[1], hidden_dim).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        for batch_control, batch_perturbed, batch_pfeat in loader:
            batch_control = batch_control.to(device)
            batch_perturbed = batch_perturbed.to(device)
            batch_pfeat = batch_pfeat.to(device)
            pred = head(batch_control, batch_pfeat)
            loss = nn.functional.smooth_l1_loss(pred, batch_perturbed)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return head


@torch.no_grad()
def predict_head(head: PerturbationHead, control_z: np.ndarray, perturbation_feature: np.ndarray, device: str | torch.device = "cpu") -> np.ndarray:
    head.eval()
    control = torch.as_tensor(control_z, dtype=torch.float32, device=device)
    pfeat = torch.as_tensor(perturbation_feature, dtype=torch.float32, device=device)
    return head(control, pfeat).cpu().numpy()


def perturbation_metrics(
    pred_expr: np.ndarray,
    control_expr: np.ndarray,
    true_expr: np.ndarray,
    top_ks: list[int] | tuple[int, ...] = (20, 100),
) -> dict[str, float]:
    pred_delta = pred_expr.mean(axis=0) - control_expr.mean(axis=0)
    true_delta = true_expr.mean(axis=0) - control_expr.mean(axis=0)
    out: dict[str, float] = {"delta_pearson": delta_pearson(pred_delta, true_delta)}
    for k in top_ks:
        out[f"precision_at_{k}"] = precision_at_k(pred_delta, true_delta, k)
    return out


def condition_group_means(
    expr: np.ndarray,
    z: np.ndarray,
    cond: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per-condition mean expression and mean latent for a control pool."""

    means: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for c in np.unique(cond.astype(str)):
        m = cond.astype(str) == c
        means[str(c)] = (expr[m].mean(axis=0), z[m].mean(axis=0))
    return means


def matched_control_means(
    means_by_cond: dict[str, tuple[np.ndarray, np.ndarray]],
    cond_array: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Composition-weighted control means matching ``cond_array``'s condition mix.

    Removes the culture-condition confound: a perturbed group is compared against
    controls whose Rest/Stim8hr/Stim48hr proportions match the group's own.
    """

    conds, counts = np.unique(cond_array.astype(str), return_counts=True)
    frac = counts / counts.sum()
    any_expr, any_z = next(iter(means_by_cond.values()))
    expr_mean = np.zeros_like(any_expr)
    z_mean = np.zeros_like(any_z)
    for c, f in zip(conds, frac):
        if str(c) not in means_by_cond:
            continue
        e, zz = means_by_cond[str(c)]
        expr_mean = expr_mean + f * e
        z_mean = z_mean + f * zz
    return expr_mean, z_mean


def sample_matched_controls(
    target_cond: np.ndarray,
    control_cond: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample one same-condition control index per target cell (with replacement)."""

    target_cond = target_cond.astype(str)
    control_cond = control_cond.astype(str)
    idx = np.zeros(len(target_cond), dtype=np.int64)
    pools = {c: np.where(control_cond == c)[0] for c in np.unique(control_cond)}
    all_idx = np.arange(len(control_cond))
    for c in np.unique(target_cond):
        m = target_cond == c
        pool = pools.get(c)
        if pool is None or pool.size == 0:
            pool = all_idx
        idx[m] = rng.choice(pool, size=int(m.sum()), replace=True)
    return idx


def delta_metrics(
    pred_mean: np.ndarray,
    true_mean: np.ndarray,
    control_mean: np.ndarray,
    top_ks: list[int] | tuple[int, ...] = (20, 100),
) -> dict[str, float]:
    """Delta-Pearson + precision@k for a single group, given precomputed means."""

    pred_delta = pred_mean - control_mean
    true_delta = true_mean - control_mean
    out: dict[str, float] = {"delta_pearson": delta_pearson(pred_delta, true_delta)}
    for k in top_ks:
        out[f"precision_at_{k}"] = precision_at_k(pred_delta, true_delta, k)
    return out
