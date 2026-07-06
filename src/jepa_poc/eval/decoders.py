"""Latent -> expression decoders shared by the benchmark tasks.

Both decoders are fit ONLY on control cells (never on perturbations), so a high
decode score means the perturbation effect was encoded as a geometric shift in a
latent space defined entirely by normal-cell biology (benchmarking.md sec 5.0).

  * linear : closed-form ridge (reused from jepa_poc.eval.perturbation).
  * mlp    : 2-hidden-layer GELU MLP with early-stopping on a held-out split.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class MLPDecoder(nn.Module):
    def __init__(self, latent_dim: int, out_dim: int, hidden: int = 1024, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def train_mlp_decoder(
    z: np.ndarray,
    x: np.ndarray,
    device: torch.device,
    hidden: int = 1024,
    epochs: int = 40,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    val_frac: float = 0.05,
    seed: int = 13,
) -> tuple[MLPDecoder, float]:
    rng = np.random.default_rng(seed)
    n = z.shape[0]
    perm = rng.permutation(n)
    n_val = int(n * val_frac)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    zt = torch.as_tensor(z, dtype=torch.float32)
    xt = torch.as_tensor(x, dtype=torch.float32)
    tr = DataLoader(TensorDataset(zt[tr_idx], xt[tr_idx]), batch_size=batch_size, shuffle=True, drop_last=True)
    zv, xv = zt[val_idx].to(device), xt[val_idx].to(device)

    model = MLPDecoder(z.shape[1], x.shape[1], hidden=hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val = float("inf")
    best_state = None
    for _ in range(epochs):
        model.train()
        for bz, bx in tr:
            bz, bx = bz.to(device), bx.to(device)
            loss = nn.functional.mse_loss(model(bz), bx)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = nn.functional.mse_loss(model(zv), xv).item()
        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_val


@torch.no_grad()
def mlp_decode(model: MLPDecoder, z: np.ndarray, device: torch.device, batch: int = 4096) -> np.ndarray:
    out = []
    for i in range(0, z.shape[0], batch):
        zz = torch.as_tensor(z[i : i + batch], dtype=torch.float32, device=device)
        out.append(model(zz).cpu().numpy())
    return np.concatenate(out, axis=0)
