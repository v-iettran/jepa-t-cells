"""Experiment 2: MLP-probe representation quality for frozen A0/A1 encoders.

The standard `eval_perturbation.py` representation-quality probe fits a *linear*
ridge decoder z -> expression on controls, then scores the decoded latent
perturbation delta against the true expression delta. If an encoder stores
perturbation information *non-linearly* (plausible for A1/SIGReg, whose embedding
is forced near-isotropic), a linear decoder can miss it.

This script swaps ONLY the decoder (linear -> MLP), keeping everything else
identical: the same cached frozen embeddings, the same condition-matched control
referencing, and the same delta_pearson / precision@k metric over group means.
It reuses the existing per-encoder embedding caches (no re-embedding), so it runs
in minutes.

For each encoder it reports, on the held-out-gene test set:
  * linear decoder (reproduces the pipeline number, as a correctness check)
  * MLP decoder, readout = decode(mean latent)        [apples-to-apples swap]
  * MLP decoder, readout = mean(decode(per-cell))      [full nonlinear readout]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jepa_poc.config import ensure_dir, load_config  # noqa: E402
from jepa_poc.eval.perturbation import (  # noqa: E402
    condition_group_means,
    decode_latent,
    delta_metrics,
    fit_linear_decoder,
    matched_control_means,
)

JEPA_ROOT = Path(__file__).resolve().parents[1]


def cache_path_for(cfg, checkpoint: Path, n_control: int, n_train_head: int) -> Path:
    key = hashlib.md5(
        f"{checkpoint.resolve()}|nc={n_control}|nt={n_train_head}|seed={int(cfg.seed)}".encode()
    ).hexdigest()[:12]
    return Path(cfg.data.run_dir) / "_embed_cache" / f"perturb_embeds_{key}.npz"


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
    for ep in range(epochs):
        model.train()
        for bz, bx in tr:
            bz, bx = bz.to(device), bx.to(device)
            pred = model(bz)
            loss = nn.functional.mse_loss(pred, bx)
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


def repr_metrics(z, expr, cond, gene, decode_fn, ctrl_means, top_ks, test_genes, percell=False):
    """Mirror eval_perturbation._group_metrics for representation quality.

    decode_fn maps an array [m, d] -> [m, n_genes].
    percell=False: pred = decode(mean latent); percell=True: pred = mean(decode(cells)).
    """

    def metrics_for(mask):
        if mask.sum() == 0:
            return None
        ctrl_expr_mean, ctrl_z_mean = matched_control_means(ctrl_means, cond[mask])
        true_mean = expr[mask].mean(axis=0)
        if percell:
            pred_mean = decode_fn(z[mask]).mean(axis=0)
        else:
            pred_mean = decode_fn(z[mask].mean(axis=0)[None, :])[0]
        control_ref = decode_fn(ctrl_z_mean[None, :])[0]
        return delta_metrics(pred_mean, true_mean, control_ref, top_ks)

    out = {"overall": metrics_for(np.ones(len(cond), dtype=bool)), "per_gene": {}}
    for g in test_genes:
        m = gene.astype(str) == g
        r = metrics_for(m)
        if r is not None:
            r["n_cells"] = int(m.sum())
            out["per_gene"][g] = r
    return out


def evaluate_encoder(label, cache_file, top_ks, device, epochs, hidden):
    c = np.load(cache_file, allow_pickle=True)
    control_z, control_expr, control_cond = c["control_z"], c["control_expr"], c["control_cond"]
    test_z, test_expr, test_cond, test_gene = c["test_z"], c["test_expr"], c["test_cond"], c["test_gene"]
    test_genes = sorted(set(test_gene.astype(str).tolist()))
    ctrl_means = condition_group_means(control_expr, control_z, control_cond)

    # --- linear decoder (reproduce pipeline number) ---
    lin = fit_linear_decoder(control_z, control_expr)
    lin_res = repr_metrics(
        test_z, test_expr, test_cond, test_gene,
        lambda zz: decode_latent(zz, lin), ctrl_means, top_ks, test_genes,
    )

    # --- MLP decoder ---
    print(f"[{label}] training MLP decoder on {control_z.shape[0]:,} control cells "
          f"(z{control_z.shape[1]} -> expr{control_expr.shape[1]})")
    mlp, val_mse = train_mlp_decoder(control_z, control_expr, device, hidden=hidden, epochs=epochs)
    print(f"[{label}] MLP val MSE = {val_mse:.4f}")
    decode_fn = lambda zz: mlp_decode(mlp, zz, device)
    mlp_meanz = repr_metrics(test_z, test_expr, test_cond, test_gene, decode_fn, ctrl_means, top_ks, test_genes, percell=False)
    mlp_percell = repr_metrics(test_z, test_expr, test_cond, test_gene, decode_fn, ctrl_means, top_ks, test_genes, percell=True)

    return {
        "label": label,
        "cache_file": str(cache_file),
        "n_control": int(control_z.shape[0]),
        "n_test": int(test_z.shape[0]),
        "mlp_val_mse": val_mse,
        "linear": lin_res,
        "mlp_decode_meanz": mlp_meanz,
        "mlp_decode_percell": mlp_percell,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exp2.yaml")
    ap.add_argument("--a0-ckpt", default="runs/exp2_A0/checkpoints/last.ckpt")
    ap.add_argument("--a1-ckpt", default="runs/exp2_A1/checkpoints/last.ckpt")
    ap.add_argument("--n-control", type=int, default=150000)
    ap.add_argument("--n-train-head", type=int, default=200000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--out-dir", default="runs/exp2_mlp_probe")
    args = ap.parse_args()

    cfg = load_config(args.config)
    top_ks = list(getattr(cfg.eval, "top_k_de", [20, 100]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ensure_dir(Path(args.out_dir))

    results = {}
    for label, ckpt in [("A0", Path(args.a0_ckpt)), ("A1", Path(args.a1_ckpt))]:
        cache_file = cache_path_for(cfg, ckpt, args.n_control, args.n_train_head)
        if not cache_file.exists():
            raise FileNotFoundError(f"No embedding cache for {label} at {cache_file}. Run eval_perturbation first.")
        print(f"\n===== {label}: {ckpt} =====")
        print(f"cache: {cache_file}")
        results[label] = evaluate_encoder(label, cache_file, top_ks, device, args.epochs, args.hidden)

    (out_dir / "mlp_probe_results.json").write_text(json.dumps(results, indent=2))

    # ---- console summary ----
    def ov(d):
        o = d["overall"]
        return o["delta_pearson"], o.get(f"precision_at_{top_ks[0]}"), o.get(f"precision_at_{top_ks[-1]}")

    print("\n================ Representation quality (overall, held-out genes) ================")
    print(f"{'encoder':<6} {'decoder':<22} {'delta_pearson':>13} {'p@'+str(top_ks[0]):>7} {'p@'+str(top_ks[-1]):>7}")
    for label in results:
        for name, key in [("linear (ridge)", "linear"),
                          ("MLP decode(mean z)", "mlp_decode_meanz"),
                          ("MLP mean(decode cell)", "mlp_decode_percell")]:
            dp, p1, p2 = ov(results[label][key])
            print(f"{label:<6} {name:<22} {dp:>13.4f} {p1:>7.2f} {p2:>7.2f}")
    print(f"\nWrote {out_dir / 'mlp_probe_results.json'}")


if __name__ == "__main__":
    main()
