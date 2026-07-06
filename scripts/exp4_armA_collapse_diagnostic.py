"""Diagnose exp4 Arm A rank-2 collapse: one-sided hinge vs projector absorption.

Read-only on existing checkpoints. No retraining.

Usage:
    PYTHONPATH=src python scripts/exp4_armA_collapse_diagnostic.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from jepa_poc.config import load_config
from jepa_poc.models.gene_tokenizer import ESMGeneTokenEncoder

GAMMA = 1.0  # VICReg std floor: mean(relu(gamma - std_j))


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def effective_rank_entropy(eigvals: np.ndarray) -> float:
    """Entropy-based eff-rank (training logs + eval_exp4_armA geometry)."""
    eigvals = np.clip(np.asarray(eigvals, np.float64), 0, None)
    s = eigvals.sum()
    if s <= 0:
        return 0.0
    probs = eigvals / s
    probs = probs[probs > 0]
    return float(np.exp(-(probs * np.log(probs)).sum()))


def effective_rank_svd(z: np.ndarray) -> float:
    """SVD-based eff-rank (exp4_representation_checks Check 1)."""
    centered = z - z.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(centered, full_matrices=False)
    var = s ** 2
    p = var / var.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def participation_ratio(eigvals: np.ndarray) -> float:
    eigvals = np.clip(np.asarray(eigvals, np.float64), 0, None)
    return float(eigvals.sum() ** 2 / (np.square(eigvals).sum() + 1e-12))


def vicreg_hinge(z: torch.Tensor) -> float:
    std = torch.sqrt(z.var(dim=0, unbiased=False) + 1e-4)
    return float(torch.mean(F.relu(GAMMA - std)).item())


def vicreg_hinge_within_batch(z: torch.Tensor, batch_id: torch.Tensor) -> float:
    terms = []
    for g in batch_id.unique():
        mask = batch_id == g
        if mask.sum() < 2:
            continue
        terms.append(vicreg_hinge(z[mask]))
    return float(np.mean(terms)) if terms else float("nan")


@torch.no_grad()
def embed_cls(
    encoder: ESMGeneTokenEncoder,
    values: np.ndarray,
    device: str,
    batch_size: int = 512,
) -> np.ndarray:
    encoder.eval().to(device)
    out = []
    use_cuda = device == "cuda"
    for i in range(0, len(values), batch_size):
        v = torch.as_tensor(values[i : i + batch_size], dtype=torch.float32, device=device)
        with torch.autocast(device_type="cuda" if use_cuda else "cpu", dtype=torch.bfloat16, enabled=use_cuda):
            z = encoder(v, None)[:, 0, :]
        out.append(z.float().cpu().numpy())
    return np.concatenate(out, axis=0)


def load_encoder_weights(path: Path, encoder_key: str = "context_encoder") -> dict:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "state_dict" in obj:
        sd = obj["state_dict"]
        prefix = f"model.{encoder_key}."
        return {k.removeprefix(prefix): v for k, v in sd.items() if k.startswith(prefix)}
    return obj


def build_encoder(cfg) -> ESMGeneTokenEncoder:
    esm = np.load(cfg.data.esm_embeddings_path, allow_pickle=True)
    return ESMGeneTokenEncoder(
        esm_embeddings=esm["embeddings"],
        fallback_mask=esm["fallback_mask"],
        n_batches=2,
        d_model=int(cfg.model.d_model),
        d_id_proj=int(cfg.model.d_id_proj),
        d_expr=int(cfg.model.d_expr),
        use_fallback_indicator=bool(cfg.model.use_fallback_indicator),
        n_layers=int(cfg.model.n_layers),
        n_heads=int(cfg.model.n_heads),
        dropout=float(cfg.model.dropout),
    )


def stratified_control_pool(adata, batch_key: str, n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (expr, batch_id_int, batch_str) for n balanced control cells."""
    cond = adata.obs["culture_condition"].astype(str).to_numpy()
    batch_str = adata.obs[batch_key].astype(str).to_numpy()
    rng = np.random.default_rng(seed)
    conds = np.unique(cond)
    per = n // len(conds)
    idx_parts = []
    for c in conds:
        pool = np.where(cond == c)[0]
        idx_parts.append(rng.choice(pool, size=min(per, len(pool)), replace=False))
    idx = np.sort(np.concatenate(idx_parts))
    expr = adata.X[idx]
    if hasattr(expr, "toarray"):
        expr = expr.toarray()
    expr = np.asarray(expr, dtype=np.float32)
    batches = batch_str[idx]
    # map batch strings to 0/1
    uniq = np.unique(batches)
    bid = np.array([np.where(uniq == b)[0][0] for b in batches], dtype=np.int64)
    return expr, bid, batches


def spectrum_stats(z: np.ndarray) -> dict:
    z = np.asarray(z, np.float64)
    centered = z - z.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(1, z.shape[0] - 1)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(eigvals)[::-1]
    med = float(np.median(eigvals))
    floor_lo, floor_hi = 0.8 * GAMMA ** 2, 1.2 * GAMMA ** 2
    floor_mask = (eigvals >= floor_lo) & (eigvals <= floor_hi)
    std_j = z.std(axis=0, ddof=0)
    std_floor_mask = (std_j >= 0.8 * GAMMA) & (std_j <= 1.2 * GAMMA)
    return {
        "top20_eigenvalues": eigvals[:20].tolist(),
        "eig_median": med,
        "eig_min": float(eigvals.min()),
        "lambda1_over_median": float(eigvals[0] / med) if med > 0 else float("inf"),
        "floor_cluster_size_eig": int(floor_mask.sum()),
        "floor_cluster_frac_eig": float(floor_mask.mean()),
        "participation_ratio": participation_ratio(eigvals),
        "effective_rank_entropy_cov": effective_rank_entropy(eigvals),
        "effective_rank_entropy_svd": effective_rank_svd(z),
        "per_dim_std_mean": float(std_j.mean()),
        "per_dim_std_median": float(np.median(std_j)),
        "per_dim_std_min": float(std_j.min()),
        "per_dim_std_max": float(std_j.max()),
        "n_dims_at_std_floor": int(std_floor_mask.sum()),
        "n_dims_below_gamma": int((std_j < GAMMA).sum()),
        "n_dims_above_gamma": int((std_j >= GAMMA).sum()),
        "vicreg_hinge_global": vicreg_hinge(torch.as_tensor(z)),
        "all_eigenvalues": eigvals.tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/exp4_armA/last.ckpt")
    ap.add_argument("--ema-encoder", default="runs/exp4_armA/ema_target_encoder.pt")
    ap.add_argument("--config", default="configs/exp3.yaml")
    ap.add_argument("--n-cells", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out-dir", default="runs/benchmark/exp4_armA/checks/collapse_diagnostic")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- locate inputs ---
    _log("Loading 20K balanced control pool")
    adata = ad.read_h5ad(cfg.data.perturb_controls_processed_path)
    expr, batch_id_np, batch_str = stratified_control_pool(adata, cfg.data.batch_key, args.n_cells, args.seed)

    enc = build_encoder(cfg)

    # Online context encoder from checkpoint
    _log(f"Loading online context encoder from {args.checkpoint}")
    online_sd = load_encoder_weights(Path(args.checkpoint), "context_encoder")
    enc.load_state_dict(online_sd, strict=True)
    z_online = embed_cls(enc, expr, device)
    _log(f"Embedded online CLS: {z_online.shape}")

    # EMA target encoder (eval report used this)
    enc_ema = build_encoder(cfg)
    _log(f"Loading EMA target encoder from {args.ema_encoder}")
    ema_sd = torch.load(args.ema_encoder, map_location="cpu", weights_only=False)
    enc_ema.load_state_dict(ema_sd, strict=True)
    z_ema = embed_cls(enc_ema, expr, device)

    # Precomputed embeddings from benchmark (sanity)
    bench = np.load("runs/benchmark/exp4_armA/embeddings_none.npz", allow_pickle=True)
    z_bench = bench["control_z"][: args.n_cells]  # may not be same cells but same pool size

    # --- Check 1: CLS eigenspectrum (online encoder) ---
    chk1 = spectrum_stats(z_online)
    batch_id_t = torch.as_tensor(batch_id_np, dtype=torch.long)
    z_t = torch.as_tensor(z_online, dtype=torch.float32)
    chk1["vicreg_hinge_within_batch"] = vicreg_hinge_within_batch(z_t, batch_id_t)
    chk1["gamma_std_floor"] = GAMMA
    chk1["gamma_sq_floor_band"] = [0.8 * GAMMA ** 2, 1.2 * GAMMA ** 2]

    # --- Check 2: reg-space vs CLS ---
    # From jepa_recon: stabilization = mean(relu(1-gamma-std)) on context_tokens[:,0,:]
    # No projector/expander between encoder CLS and VICReg.
    chk2 = {
        "reg_tensor": "context_tokens[:, 0, :] (CLS embedding)",
        "projector_expander_present": False,
        "vicreg_formula": "mean(relu(gamma - std_j)) with gamma=1.0, std_j = sqrt(var_j + 1e-4), unbiased=False",
        "covariance_term_in_loss": False,
        "cls_effective_rank_entropy": chk1["effective_rank_entropy_cov"],
        "reg_space_effective_rank_entropy": chk1["effective_rank_entropy_cov"],
        "rank_delta": 0.0,
    }

    # --- Check 3: reconcile eff-rank discrepancy ---
    train_df = pd.read_csv("runs/exp4_armA/csv_logs/version_0/metrics.csv")
    train_er = train_df["train/geometry/effective_rank"].dropna()
    train_last = float(train_er.iloc[-1])
    train_mid = float(train_er.iloc[len(train_er) // 2])

    eval_json = json.loads(Path("runs/benchmark/exp4_armA/exp4_armA_eval.json").read_text())
    checks_json = json.loads(Path("runs/benchmark/exp4_armA/checks/exp4_representation_checks.json").read_text())

    chk3 = {
        "definitions": {
            "training_log": "entropy of normalized cov eigenvalues on training-batch CLS (lit_module _log_geometry); logged every 2000 steps on context encoder during training",
            "eval_exp4_armA": "entropy of normalized cov eigenvalues on 20K balanced control cells; EMA target encoder; batch_mode=none",
            "representation_checks": "entropy of normalized singular values (SVD) on 20K control embeddings_none.npz",
        },
        "exp4_armA_online_context_20k_none": chk1["effective_rank_entropy_cov"],
        "exp4_armA_ema_target_20k_none": float(spectrum_stats(z_ema)["effective_rank_entropy_cov"]),
        "exp4_armA_training_log_end": train_last,
        "exp4_armA_training_log_mid": train_mid,
        "exp4_armA_eval_report_geometry": eval_json["exp4_armA"]["geometry"]["none"]["effective_rank"],
        "exp4_armA_representation_checks_svd": checks_json["check1_svd"]["effective_rank_entropy"],
        "arm1_none_eval_report_NOT_exp4": eval_json["arm1_recomputed"]["geometry"]["none"]["effective_rank"],
        "gaps": {
            "training_vs_eval_ema_same_def": train_last - eval_json["exp4_armA"]["geometry"]["none"]["effective_rank"],
            "online_vs_ema_same_pool": chk1["effective_rank_entropy_cov"] - float(spectrum_stats(z_ema)["effective_rank_entropy_cov"]),
            "cov_vs_svd_same_embeddings": chk1["effective_rank_entropy_cov"] - effective_rank_svd(z_bench[:args.n_cells]),
        },
    }

    # --- Verdicts ---
    # Check 1: 2 huge + floor?
    top2_frac = sum(chk1["top20_eigenvalues"][:2]) / sum(chk1["all_eigenvalues"])
    n90 = int(np.searchsorted(np.cumsum(np.array(chk1["all_eigenvalues"]) / sum(chk1["all_eigenvalues"])), 0.9) + 1)
    if chk1["lambda1_over_median"] > 10 and chk1["n_dims_at_std_floor"] > 50:
        verdict1 = (
            f"One-sided hinge confirmed: λ1/λ_med={chk1['lambda1_over_median']:.1f}, "
            f"{chk1['n_dims_at_std_floor']}/256 dims at std≈γ, top-2 eigvals explain {top2_frac:.1%} variance, "
            f"90% variance in {n90} dims. VICReg satisfied (hinge={chk1['vicreg_hinge_global']:.4f}) via 2 runaway + floor."
        )
    elif chk1["lambda1_over_median"] > 5:
        verdict1 = (
            f"Partial hinge signature: dominant eigenvalue (λ1/λ_med={chk1['lambda1_over_median']:.1f}) "
            f"but floor cluster only {chk1['n_dims_at_std_floor']} dims at γ."
        )
    else:
        verdict1 = "Spectrum does not show classic 2-huge + γ-floor pattern."

    if chk2["rank_delta"] == 0 and not chk2["projector_expander_present"]:
        verdict2 = (
            "No projector absorption: VICReg operates directly on CLS (no expander head). "
            f"Reg-space eff-rank = CLS eff-rank = {chk1['effective_rank_entropy_cov']:.2f}. "
            "Collapse is IN the regularized tensor; a reg swap on the same head would target CLS."
        )
    else:
        verdict2 = "Projector absorption possible — reg-space rank differs from CLS."

    arm1_er = eval_json["arm1_recomputed"]["geometry"]["none"]["effective_rank"]
    if abs(chk3["exp4_armA_training_log_end"] - arm1_er) < 1.0 and chk3["exp4_armA_eval_report_geometry"] < 3:
        verdict3 = (
            f"Measurement mismatch clarified: arm1-none eff-rank {arm1_er:.1f} is a DIFFERENT model (global VICReg arm1), "
            f"not exp4. exp4 eval geometry {chk3['exp4_armA_eval_report_geometry']:.1f} ≈ online {chk3['exp4_armA_online_context_20k_none']:.1f}; "
            f"training log end {train_last:.1f} is same representation (online, training batches) within ~{abs(train_last-chk3['exp4_armA_eval_report_geometry']):.1f}."
        )
    else:
        verdict3 = (
            f"Training log ({train_last:.1f}) vs eval EMA ({chk3['exp4_armA_eval_report_geometry']:.1f}) vs online ({chk3['exp4_armA_online_context_20k_none']:.1f}): "
            f"online-EMA gap {chk3['gaps']['online_vs_ema_same_pool']:.2f}; training-eval gap {chk3['gaps']['training_vs_eval_ema_same_def']:.2f}."
        )

    if not chk2["projector_expander_present"] and chk1["n_dims_at_std_floor"] > 30:
        recommendation = "(A) Proceed to SIGReg swap on CLS — hinge degeneracy is the mechanism; no projector bypass."
    elif chk2["projector_expander_present"]:
        recommendation = "(B) Projector absorption — regularize/decode CLS directly instead."
    else:
        recommendation = "(C) Resolve measurement mismatch before choosing reg swap."

    results = {
        "inputs": {
            "checkpoint_online": str(args.checkpoint),
            "ema_encoder": str(args.ema_encoder),
            "vicreg_gamma": GAMMA,
            "vicreg_tensor": "CLS context_tokens[:,0,:]",
            "projector_expander": False,
            "vicreg_weight": float(cfg.model.vicreg_weight),
            "within_batch_vicreg": True,
            "n_cells": args.n_cells,
            "batch_mode": "none",
            "encoder_used_for_check1": "online context_encoder from last.ckpt",
        },
        "check1_cls_eigenspectrum": chk1,
        "check2_reg_vs_cls": chk2,
        "check3_effrank_reconciliation": chk3,
        "verdicts": {"check1": verdict1, "check2": verdict2, "check3": verdict3},
        "recommendation": recommendation,
    }

    with open(out_dir / "collapse_diagnostic.json", "w") as f:
        json.dump(results, f, indent=2)

    # --- plots ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ev = np.array(chk1["all_eigenvalues"])
    axes[0, 0].semilogy(np.arange(1, len(ev) + 1), ev, "o-", ms=3)
    axes[0, 0].axhline(GAMMA ** 2, color="green", ls="--", alpha=0.7, label=f"γ²={GAMMA**2}")
    axes[0, 0].set_xlabel("Eigenvalue index")
    axes[0, 0].set_ylabel("Eigenvalue (log)")
    axes[0, 0].set_title(f"CLS covariance spectrum (online, 20K)\nλ1/λ_med={chk1['lambda1_over_median']:.1f}, eff-rank={chk1['effective_rank_entropy_cov']:.2f}")
    axes[0, 0].legend(fontsize=8)

    std_j = z_online.std(axis=0, ddof=0)
    axes[0, 1].hist(std_j, bins=50, color="#4C78A8", alpha=0.85)
    axes[0, 1].axvline(GAMMA, color="red", ls="--", label=f"γ={GAMMA}")
    axes[0, 1].set_xlabel("Per-dim std_j")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title(f"Per-dim std (hinge floor)\n{chk1['n_dims_at_std_floor']} dims in [0.8γ,1.2γ], hinge={chk1['vicreg_hinge_global']:.4f}")
    axes[0, 1].legend(fontsize=8)

    # training eff-rank curve
    steps = train_df["step"].values
    er_train = train_df["train/geometry/effective_rank"].values
    m = ~np.isnan(er_train)
    axes[1, 0].plot(steps[m], er_train[m], alpha=0.5, lw=0.8, label="training log (online, batches)")
    axes[1, 0].axhline(chk1["effective_rank_entropy_cov"], color="blue", ls="-", label=f"eval online 20K: {chk1['effective_rank_entropy_cov']:.2f}")
    axes[1, 0].axhline(chk3["exp4_armA_ema_target_20k_none"], color="orange", ls="--", label=f"eval EMA 20K: {chk3['exp4_armA_ema_target_20k_none']:.2f}")
    axes[1, 0].axhline(arm1_er, color="gray", ls=":", label=f"arm1-none (other model): {arm1_er:.1f}")
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].set_ylabel("Effective rank")
    axes[1, 0].set_title("Eff-rank reconciliation (Check 3)")
    axes[1, 0].legend(fontsize=7)
    axes[1, 0].set_ylim(0, max(10, np.nanmax(er_train[m]) * 1.1))

    # loss terms at end of training
    last = train_df.dropna(subset=["train/loss"]).iloc[-1]
    terms = ["train/prediction_loss", "train/recon_loss", "train/stabilization_loss"]
    vals = [float(last[t]) for t in terms]
    axes[1, 1].bar(["pred", "recon", "VICReg"], vals, color=["#4C78A8", "#F58518", "#54A24B"])
    axes[1, 1].set_ylabel("Loss (final step)")
    axes[1, 1].set_title(f"Final per-term loss @ step {int(last['step'])}")

    fig.suptitle("exp4 Arm A collapse diagnostic", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "collapse_diagnostic.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    md = f"""# exp4 Arm A rank-2 collapse diagnostic

## Inputs located
| Item | Value |
|------|-------|
| Online encoder | `{args.checkpoint}` → `context_encoder` |
| EMA encoder (eval report) | `{args.ema_encoder}` |
| VICReg γ | {GAMMA} (`mean(relu(γ - std_j))`) |
| VICReg tensor | CLS `context_tokens[:,0,:]` — **no projector/expander** |
| Covariance in loss | **No** (variance hinge only) |
| Pool | {args.n_cells:,} balanced control cells, batch_mode=none |

## Check 1 — CLS eigenspectrum
| Metric | Value |
|--------|-------|
| λ₁ / λ_median | {chk1['lambda1_over_median']:.2f} |
| Floor cluster (eig ∈ [0.8,1.2]γ²) | {chk1['floor_cluster_size_eig']} / 256 |
| Dims at std floor ([0.8,1.2]γ) | {chk1['n_dims_at_std_floor']} / 256 |
| Dims below γ (hinge inactive) | {chk1['n_dims_below_gamma']} |
| Participation ratio | {chk1['participation_ratio']:.2f} |
| Eff-rank (cov entropy) | {chk1['effective_rank_entropy_cov']:.2f} |
| Eff-rank (SVD entropy) | {chk1['effective_rank_entropy_svd']:.2f} |
| VICReg hinge (global 20K) | {chk1['vicreg_hinge_global']:.4f} |
| VICReg hinge (within-batch) | {chk1['vicreg_hinge_within_batch']:.4f} |

**Verdict:** {verdict1}

## Check 2 — reg-space vs CLS
**Verdict:** {verdict2}

## Check 3 — eff-rank reconciliation
| Source | Eff-rank |
|--------|----------|
| Training log (end, online batches) | {train_last:.2f} |
| Eval geometry (EMA, 20K none) | {chk3['exp4_armA_eval_report_geometry']:.2f} |
| This diagnostic (online, 20K none) | {chk3['exp4_armA_online_context_20k_none']:.2f} |
| arm1-none eval (**different model**) | {arm1_er:.1f} |

**Verdict:** {verdict3}

## Recommendation
**{recommendation}**
"""
    (out_dir / "COLLAPSE_DIAGNOSTIC_REPORT.md").write_text(md)

    _log(verdict1)
    _log(verdict2)
    _log(verdict3)
    _log(f"Recommendation: {recommendation}")
    _log(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
