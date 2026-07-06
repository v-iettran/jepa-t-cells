"""exp4 Arm A: ridge decode on mean-pooled context-encoder gene tokens (not CLS).

Re-embeds the same eval pools as ``embeddings_none.npz`` using the online
``context_encoder`` from the training checkpoint, ``batch_mode=none``, and
z = mean(context_tokens[:, 1:, :], dim=genes). Runs the same Tasks 1-2 ridge
scoring + 20K geometry eff-rank as ``eval_exp4_armA.py``.

Usage:
    PYTHONPATH=src python scripts/exp4_armA_gene_meanpool_eval.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from jepa_poc.config import ensure_dir, load_config
from jepa_poc.eval.pools import load_eval_pools
from jepa_poc.eval.tasks import BenchmarkEvaluator
from jepa_poc.models.gene_tokenizer import ESMGeneTokenEncoder


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def effective_rank(z: np.ndarray) -> float:
    z = np.asarray(z, np.float64)
    centered = z - z.mean(0, keepdims=True)
    cov = centered.T @ centered / max(1, z.shape[0] - 1)
    eigvals = np.clip(np.linalg.eigvalsh(cov), 0, None)
    probs = eigvals / eigvals.sum()
    probs = probs[probs > 0]
    return float(np.exp(-(probs * np.log(probs)).sum()))


def content_scale(z: np.ndarray) -> float:
    z = np.asarray(z, np.float64)
    mu = z.mean(0, keepdims=True)
    return float(np.sqrt(((z - mu) ** 2).sum(1).mean()))


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


def load_context_encoder(cfg, checkpoint: Path, device: str) -> ESMGeneTokenEncoder:
    enc = build_encoder(cfg)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    prefix = "model.context_encoder."
    sub = {k.removeprefix(prefix): v for k, v in state.items() if k.startswith(prefix)}
    if not sub:
        raise KeyError(f"No {prefix} weights in {checkpoint}")
    enc.load_state_dict(sub, strict=True)
    return enc.eval().to(device)


@torch.no_grad()
def embed_mean_gene_tokens(
    encoder: ESMGeneTokenEncoder,
    values: np.ndarray,
    device: str,
    batch_size: int = 512,
) -> np.ndarray:
    """batch_mode=none; z = mean(encoder output over gene positions 1:)."""
    out: list[np.ndarray] = []
    use_cuda = device == "cuda"
    for i in range(0, values.shape[0], batch_size):
        v = torch.as_tensor(values[i : i + batch_size], dtype=torch.float32, device=device)
        with torch.autocast(device_type="cuda" if use_cuda else "cpu", dtype=torch.bfloat16, enabled=use_cuda):
            tokens = encoder(v, None)  # [B, 1+G, D]
            z = tokens[:, 1:, :].mean(dim=1)
        out.append(z.float().cpu().numpy())
    return np.concatenate(out, axis=0)


def score_cache(cache: dict, device: str, cfg, top_ks) -> dict:
    ev = BenchmarkEvaluator(cache, device=device, mlp_epochs=40, mlp_hidden=1024,
                            seed=int(cfg.seed))
    t12 = ev.tasks_1_2(top_ks)
    rows = ev._per_perturbation_decode(top_ks, min_cells=20)
    genes = list(rows)
    eff = np.array([rows[g]["effect_size"] for g in genes])
    q1, q2 = np.quantile(eff, [1 / 3, 2 / 3])
    strata = {"weak": [], "medium": [], "strong": []}
    for g in genes:
        e = rows[g]["effect_size"]
        s = "weak" if e <= q1 else ("medium" if e <= q2 else "strong")
        strata[s].append(g)
    strata_ridge = {
        s: float(np.nanmean([rows[g]["linear_delta_pearson"] for g in gl])) if gl else None
        for s, gl in strata.items()
    }
    lin_o = t12["linear_ridge"]["overall"]
    return {
        "ridge_delta_pearson": float(lin_o["delta_pearson"]),
        "mlp_decode_meanz_delta_pearson": float(t12["mlp_decode_meanz"]["overall"]["delta_pearson"]),
        "ridge_p@20": float(lin_o.get("precision_at_20", float("nan"))),
        "strata_ridge": strata_ridge,
        "content_scale_control_z": content_scale(cache["control_z"]),
        "n_test_genes": len(ev.test_genes),
        "per_gene_ridge": {g: float(t12["linear_ridge"]["per_gene"][g]["delta_pearson"])
                           for g in t12["linear_ridge"]["per_gene"]},
    }


def geometry_subset_idx(batch_str: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    groups = np.unique(batch_str)
    per = n // len(groups)
    parts = [rng.choice(np.where(batch_str == g)[0], size=per, replace=False) for g in groups]
    return np.sort(np.concatenate(parts))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/exp4_armA/last.ckpt")
    ap.add_argument("--baseline-npz", default="runs/benchmark/exp4_armA/embeddings_none.npz")
    ap.add_argument("--baseline-eval", default="runs/benchmark/exp4_armA/exp4_armA_eval.json")
    ap.add_argument("--out-dir", default="runs/benchmark/exp4_armA/checks/gene_meanpool")
    ap.add_argument("--n-geometry", type=int, default=20_000)
    ap.add_argument("--batch-size", type=int, default=512)
    args = ap.parse_args()

    cfg = load_config("configs/exp3.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    top_ks = list(cfg.eval.top_k_de)
    out_dir = ensure_dir(Path(args.out_dir))

    _log("Loading cached eval arrays from embeddings_none.npz")
    baseline = np.load(args.baseline_npz, allow_pickle=True)
    pools = load_eval_pools(cfg, n_control=150_000, n_train=200_000, seed=int(cfg.seed))
    control_batch_str = pools.control.batch_str

    _log(f"Loading context_encoder from {args.checkpoint}")
    encoder = load_context_encoder(cfg, Path(args.checkpoint), device)

    cache: dict = {
        "model": "exp4_armA_gene_meanpool",
        "latent_dim": int(cfg.model.d_model),
        "control_expr": baseline["control_expr"],
        "control_cond": baseline["control_cond"],
        "train_expr": baseline["train_expr"],
        "train_cond": baseline["train_cond"],
        "train_gene": baseline["train_gene"],
        "test_expr": baseline["test_expr"],
        "test_cond": baseline["test_cond"],
        "test_gene": baseline["test_gene"],
        "gene_names": baseline["gene_names"],
    }
    for split in ("control", "train", "test"):
        expr = cache[f"{split}_expr"]
        _log(f"Embedding {split} ({expr.shape[0]:,} cells) — mean-pooled gene tokens, batch_mode=none")
        cache[f"{split}_z"] = embed_mean_gene_tokens(encoder, expr, device, batch_size=args.batch_size)
        if device == "cuda":
            torch.cuda.empty_cache()
    np.savez(out_dir / "embeddings_gene_meanpool_none.npz", **cache)

    _log("Scoring ridge decode (Tasks 1-2, same pipeline as exp4_armA_eval)")
    scores = score_cache(cache, device, cfg, top_ks)

    geo_idx = geometry_subset_idx(control_batch_str, min(args.n_geometry, len(control_batch_str)), int(cfg.seed))
    z_geo = cache["control_z"][geo_idx]
    scores["geometry"] = {
        "n_cells": int(len(geo_idx)),
        "effective_rank": effective_rank(z_geo),
        "content_scale_global": content_scale(z_geo),
    }

    ref = {}
    ref_path = Path(args.baseline_eval)
    if ref_path.exists():
        ref = json.loads(ref_path.read_text()).get("exp4_armA", {}).get("none", {})

    scores["cls_baseline_none"] = {
        "ridge_delta_pearson": ref.get("ridge_delta_pearson"),
        "ridge_p@20": ref.get("ridge_p@20"),
        "strata_ridge": ref.get("strata_ridge"),
        "effective_rank": None,
        "note": "EMA target encoder (ema_target_encoder.pt) from prior exp4_armA_eval",
    }
    if ref_path.exists():
        full = json.loads(ref_path.read_text())
        scores["cls_baseline_none"]["effective_rank"] = (
            full.get("exp4_armA", {}).get("geometry", {}).get("none", {}).get("effective_rank")
        )

    # Same-checkpoint context_encoder CLS (apples-to-apples with gene mean-pool)
    ctx_cls_path = out_dir / "context_encoder_cls_eval.json"
    if ctx_cls_path.exists():
        scores["context_encoder_cls_same_ckpt"] = json.loads(ctx_cls_path.read_text())

    out_json = out_dir / "gene_meanpool_eval.json"
    out_json.write_text(json.dumps(scores, indent=2))
    _log(f"Wrote {out_json}")

    def f(x, n=3):
        return f"{x:.{n}f}" if isinstance(x, (int, float)) and x == x else "n/a"

    lines = [
        "# exp4 Arm A — mean-pooled gene tokens vs CLS (batch_mode=none)\n",
        f"Encoder: **context_encoder** from `{args.checkpoint}`  ",
        f"| z = mean(`context_tokens[:, 1:, :]`) | batch token **off**\n",
        "## Ridge decode (held-out 5 genes)\n",
        "| representation | ridge Δ-Pearson | p@20 | weak | medium | strong | content scale |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| **gene mean-pool** | **{f(scores['ridge_delta_pearson'])}** | {f(scores['ridge_p@20'],2)} | "
        f"{f(scores['strata_ridge']['weak'])} | {f(scores['strata_ridge']['medium'])} | "
        f"{f(scores['strata_ridge']['strong'])} | {f(scores['content_scale_control_z'],2)} |",
    ]
    if ref:
        sr = ref.get("strata_ridge", {})
        lines.append(
            f"| CLS (EMA, prior eval) | {f(ref.get('ridge_delta_pearson'))} | {f(ref.get('ridge_p@20'),2)} | "
            f"{f(sr.get('weak'))} | {f(sr.get('medium'))} | {f(sr.get('strong'))} | "
            f"{f(ref.get('content_scale_control_z'),2)} |"
        )
    lines += [
        "\n## Geometry (20K balanced control cells)\n",
        f"| representation | effective rank |",
        f"|---|---:|",
        f"| **gene mean-pool** | **{f(scores['geometry']['effective_rank'],2)}** |",
    ]
    if scores["cls_baseline_none"].get("effective_rank") is not None:
        lines.append(
            f"| CLS (EMA, prior eval) | {f(scores['cls_baseline_none']['effective_rank'],2)} |"
        )
    lines += ["\n## Per held-out gene (ridge Δ-Pearson)\n"]
    for g, v in sorted(scores.get("per_gene_ridge", {}).items()):
        lines.append(f"- **{g}**: {f(v)}")

    report = out_dir / "GENE_MEANPOOL_REPORT.md"
    report.write_text("\n".join(lines) + "\n")
    _log(f"Wrote {report}")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
