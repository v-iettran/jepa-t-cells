"""Task 5 cross-dataset transfer for exp4 Arm A (batch_mode=none).

Re-embeds the cached Arce expression matrices from the batchfix Task-5 build
through the frozen exp4 Arm A EMA encoder, scores 5a (decode transfer) and 5b
(agreement), and compares to the batch-corrupted arm1 batchfix run and PCA.

Usage:
    PYTHONPATH=src python scripts/task5_exp4_armA.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jepa_poc.config import ensure_dir, load_config  # noqa: E402
from jepa_poc.eval.metrics import delta_pearson, precision_at_k  # noqa: E402
from jepa_poc.eval.perturbation import (  # noqa: E402
    condition_group_means,
    decode_latent,
    fit_linear_decoder,
    matched_control_means,
)
from jepa_poc.models.gene_tokenizer import ESMGeneTokenEncoder  # noqa: E402

_SKIP = {"control", "non-targeting", "non_targeting", "ntc", "aavs1", "no-targeting"}


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


@torch.no_grad()
def embed_values(
    encoder: ESMGeneTokenEncoder,
    values: np.ndarray,
    device: str,
    batch_size: int = 256,
) -> np.ndarray:
    encoder.eval().to(device)
    out: list[np.ndarray] = []
    use_cuda = device == "cuda"
    for i in range(0, values.shape[0], batch_size):
        v = torch.as_tensor(values[i : i + batch_size], dtype=torch.float32, device=device)
        with torch.autocast(device_type="cuda" if use_cuda else "cpu", dtype=torch.bfloat16, enabled=use_cuda):
            z = encoder(v, None)[:, 0, :]
        out.append(z.float().cpu().numpy())
    return np.concatenate(out, axis=0)


def load_encoder(cfg, encoder_path: Path, device: str) -> ESMGeneTokenEncoder:
    esm = np.load(cfg.data.esm_embeddings_path, allow_pickle=True)
    enc = ESMGeneTokenEncoder(
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
    state = torch.load(encoder_path, map_location="cpu", weights_only=False)
    enc.load_state_dict(state, strict=True)
    return enc.to(device)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def score_agreement(
    czi_genes: np.ndarray,
    czi_z: np.ndarray,
    czi_cond: np.ndarray,
    czi_ctrl_z: np.ndarray,
    czi_ctrl_cond: np.ndarray,
    arce_genes: np.ndarray,
    arce_z: np.ndarray,
    arce_cond: np.ndarray,
    arce_ctrl_z: np.ndarray,
    arce_ctrl_cond: np.ndarray,
    overlap: list[str],
) -> dict:
    ctrl_z_czi = condition_group_means(np.zeros((len(czi_ctrl_cond), 1)), czi_ctrl_z, czi_ctrl_cond)
    ctrl_z_arce = condition_group_means(np.zeros((len(arce_ctrl_cond), 1)), arce_ctrl_z, arce_ctrl_cond)

    matched_cos: dict[str, float] = {}
    czi_norms, arce_norms = [], []
    for g in overlap:
        m_czi = czi_genes.astype(str) == g
        m_arce = arce_genes.astype(str) == g
        if m_czi.sum() < 5 or m_arce.sum() < 5:
            continue
        ref_czi, _ = matched_control_means(ctrl_z_czi, czi_cond[m_czi])
        ref_arce, _ = matched_control_means(ctrl_z_arce, arce_cond[m_arce])
        s_czi = czi_z[m_czi].mean(0) - ref_czi
        s_arce = arce_z[m_arce].mean(0) - ref_arce
        matched_cos[g] = _cosine(s_czi, s_arce)
        czi_norms.append(float(np.linalg.norm(s_czi)))
        arce_norms.append(float(np.linalg.norm(s_arce)))

    genes = list(matched_cos.keys())
    mismatched = []
    for g in genes:
        for h in genes:
            if g == h:
                continue
            m_czi = czi_genes.astype(str) == g
            m_arce = arce_genes.astype(str) == h
            ref_czi, _ = matched_control_means(ctrl_z_czi, czi_cond[m_czi])
            ref_arce, _ = matched_control_means(ctrl_z_arce, arce_cond[m_arce])
            s_czi = czi_z[m_czi].mean(0) - ref_czi
            s_arce = arce_z[m_arce].mean(0) - ref_arce
            mismatched.append(_cosine(s_czi, s_arce))

    matched_vals = list(matched_cos.values())
    labels = [1] * len(matched_vals) + [0] * len(mismatched)
    scores = matched_vals + mismatched
    auroc = float(roc_auc_score(labels, scores)) if len(set(labels)) > 1 else 0.5
    czi_mean_norm = float(np.mean(czi_norms)) if czi_norms else 0.0
    arce_mean_norm = float(np.mean(arce_norms)) if arce_norms else 0.0
    return {
        "n_genes": len(genes),
        "genes": genes,
        "matched_cosine_per_gene": matched_cos,
        "matched_mean": float(np.mean(matched_vals)) if matched_vals else None,
        "matched_median": float(np.median(matched_vals)) if matched_vals else None,
        "mismatched_mean": float(np.mean(mismatched)) if mismatched else None,
        "separation": float(np.mean(matched_vals) - np.mean(mismatched)) if matched_vals and mismatched else None,
        "auroc_matched_vs_mismatched": auroc,
        "czi_signature_norm_mean": czi_mean_norm,
        "arce_signature_norm_mean": arce_mean_norm,
        "signal_retention": arce_mean_norm / (czi_mean_norm + 1e-12),
    }


def _fit_mlp(z: np.ndarray, x: np.ndarray, seed: int = 13) -> MLPRegressor:
    mlp = MLPRegressor(
        hidden_layer_sizes=(256, 256),
        activation="relu",
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=seed,
    )
    mlp.fit(z, x)
    return mlp


def score_decode_transfer(
    arce_pert_gene: np.ndarray,
    arce_pert_expr: np.ndarray,
    arce_pert_z: np.ndarray,
    arce_pert_cond: np.ndarray,
    arce_ctrl_expr: np.ndarray,
    arce_ctrl_z: np.ndarray,
    arce_ctrl_cond: np.ndarray,
    *,
    min_cells: int = 20,
    seed: int = 13,
) -> dict:
    ridge_dec = fit_linear_decoder(arce_ctrl_z, arce_ctrl_expr, ridge=1e-3)
    mlp = _fit_mlp(arce_ctrl_z, arce_ctrl_expr, seed=seed)
    ctrl_means = condition_group_means(arce_ctrl_expr, arce_ctrl_z, arce_ctrl_cond)

    rows: dict[str, dict] = {}
    for g in np.unique(arce_pert_gene.astype(str)):
        if g.lower() in _SKIP or g.upper().startswith("NON-TARGET"):
            continue
        m = arce_pert_gene.astype(str) == g
        if int(m.sum()) < min_cells:
            continue
        expr_m, z_m, cond_m = arce_pert_expr[m], arce_pert_z[m], arce_pert_cond[m]
        ctrl_expr_ref, ctrl_z_ref = matched_control_means(ctrl_means, cond_m)
        true_delta = expr_m.mean(0) - ctrl_expr_ref
        eff = float(np.linalg.norm(true_delta))

        pred_ridge = decode_latent(z_m.mean(0, keepdims=True), ridge_dec)[0] - decode_latent(
            ctrl_z_ref.reshape(1, -1), ridge_dec
        )[0]
        ridge_dp = delta_pearson(pred_ridge, true_delta)
        ridge_p20 = precision_at_k(pred_ridge, true_delta, 20)

        pred_mlp_cells = mlp.predict(z_m) - mlp.predict(ctrl_z_ref.reshape(1, -1))
        pred_mlp_delta = pred_mlp_cells.mean(0)
        mlp_dp = delta_pearson(pred_mlp_delta, true_delta)
        mlp_p20 = precision_at_k(pred_mlp_delta, true_delta, 20)

        rows[g] = {
            "effect_size": eff,
            "n_cells": int(m.sum()),
            "ridge_delta_pearson": ridge_dp,
            "mlp_percell_delta_pearson": mlp_dp,
            "ridge_precision_at_20": ridge_p20,
            "mlp_percell_precision_at_20": mlp_p20,
        }

    if not rows:
        return {"error": "no perturbations", "n_perturbations": 0}

    genes = list(rows.keys())
    eff = np.array([rows[g]["effect_size"] for g in genes])
    q1, q2 = float(np.quantile(eff, 1 / 3)), float(np.quantile(eff, 2 / 3))

    def stratum(e: float) -> str:
        return "weak" if e <= q1 else ("medium" if e <= q2 else "strong")

    strata: dict[str, list[str]] = {"weak": [], "medium": [], "strong": []}
    for g in genes:
        strata[stratum(rows[g]["effect_size"])].append(g)

    def _pool(gl: list[str]) -> dict:
        if not gl:
            return {"n": 0}
        return {
            "n": len(gl),
            "effect_size_mean": float(np.mean([rows[g]["effect_size"] for g in gl])),
            "ridge_delta_pearson": float(np.mean([rows[g]["ridge_delta_pearson"] for g in gl])),
            "mlp_percell_delta_pearson": float(np.mean([rows[g]["mlp_percell_delta_pearson"] for g in gl])),
            "ridge_p@20": float(np.mean([rows[g]["ridge_precision_at_20"] for g in gl])),
        }

    val_pred = mlp.predict(arce_ctrl_z[: min(5000, len(arce_ctrl_z))])
    val_mse = float(np.mean((val_pred - arce_ctrl_expr[: len(val_pred)]) ** 2))

    return {
        "n_perturbations": len(genes),
        "val_mse": val_mse,
        "pooled": _pool(genes),
        "by_stratum": {s: _pool(gl) for s, gl in strata.items()},
        "effect_quantiles": {"q33": q1, "q66": q2},
        "per_perturbation": rows,
    }


def batch_diagnostic(
    czi_ctrl_z: np.ndarray,
    czi_ctrl_cond: np.ndarray,
    arce_ctrl_z: np.ndarray,
    arce_ctrl_cond: np.ndarray,
    *,
    n_per: int = 5000,
    k: int = 30,
    seed: int = 13,
) -> dict:
    rng = np.random.default_rng(seed)
    # Rest-only subset where possible (matches batchfix task5)
    czi_rest = np.where(czi_ctrl_cond.astype(str) == "Rest")[0]
    arce_rest = np.where(np.char.find(arce_ctrl_cond.astype(str), "rest") >= 0)[0]
    if len(czi_rest) == 0:
        czi_rest = np.arange(len(czi_ctrl_z))
    if len(arce_rest) == 0:
        arce_rest = np.arange(len(arce_ctrl_z))
    ic = rng.choice(czi_rest, min(n_per, len(czi_rest)), replace=False)
    ia = rng.choice(arce_rest, min(n_per, len(arce_rest)), replace=False)
    z = np.vstack([czi_ctrl_z[ic], arce_ctrl_z[ia]])
    labels = np.array([0] * len(ic) + [1] * len(ia))
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(z)
    _, idx = nn.kneighbors(z)
    purity = []
    for i, neigh in enumerate(idx):
        ds = labels[neigh[1:]]
        purity.append(float((ds == labels[i]).mean()))
    return {
        "knn_dataset_purity": float(np.mean(purity)),
        "k": k,
        "n_per_dataset": int(min(n_per, len(ic), len(ia))),
        "rest_only": True,
        "interpretation": "0.5=mixed, 1.0=separated",
    }


def load_czi_pert_from_benchmark(bench: np.lib.npyio.NpzFile, overlap: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gene = np.concatenate([bench["train_gene"].astype(str), bench["test_gene"].astype(str)])
    expr = np.concatenate([bench["train_expr"], bench["test_expr"]], axis=0)
    cond = np.concatenate([bench["train_cond"].astype(str), bench["test_cond"].astype(str)])
    mask = np.isin(gene, overlap)
    return expr[mask], gene[mask], cond[mask]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="runs/exp4_armA/ema_target_encoder.pt")
    ap.add_argument("--ref-npz", default="runs/task5_batchfix/arm1/task5.npz")
    ap.add_argument("--ref-results", default="runs/task5_batchfix/task5_results.json")
    ap.add_argument("--benchmark", default="runs/benchmark/exp4_armA/embeddings_none.npz")
    ap.add_argument("--out-dir", default="runs/benchmark/exp4_armA/task5")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    cfg = load_config("configs/exp3.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = ensure_dir(Path(args.out_dir))

    ref = np.load(args.ref_npz, allow_pickle=True)
    overlap = ref["overlap_genes"].astype(str).tolist()
    _log(f"Overlap genes (n={len(overlap)}): {overlap}")

    _log("Loading exp4 Arm A encoder")
    encoder = load_encoder(cfg, Path(args.encoder), device)

    _log(f"Embedding Arce controls ({ref['arce_ctrl_expr'].shape[0]:,} cells)")
    arce_ctrl_z = embed_values(encoder, ref["arce_ctrl_expr"], device, args.batch_size)
    _log(f"Embedding Arce perturbed ({ref['arce_pert_expr'].shape[0]:,} cells)")
    arce_pert_z = embed_values(encoder, ref["arce_pert_expr"], device, args.batch_size)

    bench = np.load(args.benchmark, allow_pickle=True)
    czi_ctrl_z = bench["control_z"]
    czi_ctrl_cond = bench["control_cond"].astype(str)
    czi_pert_expr, czi_pert_gene, czi_pert_cond = load_czi_pert_from_benchmark(bench, overlap)
    _log(f"Embedding CZI overlap pert cells ({len(czi_pert_gene):,})")
    czi_pert_z = embed_values(encoder, czi_pert_expr, device, args.batch_size)

    np.savez_compressed(
        out_dir / "task5.npz",
        model="exp4_armA",
        latent_dim=int(cfg.model.d_model),
        batch_mode="none",
        overlap_genes=np.array(overlap, dtype=object),
        gene_names=ref["gene_names"],
        arce_ctrl_z=arce_ctrl_z,
        arce_ctrl_expr=ref["arce_ctrl_expr"],
        arce_ctrl_cond=ref["arce_ctrl_cond"],
        arce_pert_z=arce_pert_z,
        arce_pert_expr=ref["arce_pert_expr"],
        arce_pert_cond=ref["arce_pert_cond"],
        arce_pert_gene=ref["arce_pert_gene"],
        czi_ctrl_z=czi_ctrl_z,
        czi_ctrl_cond=czi_ctrl_cond,
        czi_pert_z=czi_pert_z,
        czi_pert_gene=czi_pert_gene,
        czi_pert_cond=czi_pert_cond,
    )
    _log(f"Wrote {out_dir / 'task5.npz'}")

    _log("Scoring 5b agreement")
    agreement = score_agreement(
        czi_pert_gene, czi_pert_z, czi_pert_cond, czi_ctrl_z, czi_ctrl_cond,
        ref["arce_pert_gene"], arce_pert_z, ref["arce_pert_cond"].astype(str),
        arce_ctrl_z, ref["arce_ctrl_cond"].astype(str),
        overlap,
    )

    _log("Scoring 5a decode transfer")
    decode = score_decode_transfer(
        ref["arce_pert_gene"], ref["arce_pert_expr"], arce_pert_z,
        ref["arce_pert_cond"].astype(str),
        ref["arce_ctrl_expr"], arce_ctrl_z, ref["arce_ctrl_cond"].astype(str),
        seed=args.seed,
    )

    _log("Batch diagnostic (Step 6)")
    batch_diag = batch_diagnostic(
        czi_ctrl_z, czi_ctrl_cond, arce_ctrl_z, ref["arce_ctrl_cond"].astype(str),
    )

    eval_json = json.loads(Path("runs/benchmark/exp4_armA/exp4_armA_eval.json").read_text())
    in_dist = {
        "ridge": eval_json["exp4_armA"]["none"]["ridge_delta_pearson"],
        "mlp_percell": eval_json["exp4_armA"]["none"]["mlp_percell_delta_pearson"],
    }

    results = {
        "exp4_armA": {
            "latent_dim": int(cfg.model.d_model),
            "batch_mode": "none",
            "agreement": agreement,
            "decode_transfer": decode,
            "batch_diagnostic": batch_diag,
            "in_distribution": in_dist,
        }
    }

    ref_results = json.loads(Path(args.ref_results).read_text())
    for key in ("arm1", "pca"):
        if key in ref_results:
            results[key] = ref_results[key]

    out_json = out_dir / "task5_exp4_armA_results.json"
    out_json.write_text(json.dumps(results, indent=2))
    _log(f"Wrote {out_json}")

    # Markdown report
    e4 = results["exp4_armA"]
    arm1 = results.get("arm1", {})
    pca = results.get("pca", {})
    lines = [
        "# Task 5 — exp4 Arm A cross-dataset transfer (batch_mode=none)\n",
        "External: Arce et al. 2024. Encoder: `runs/exp4_armA/ema_target_encoder.pt`.",
        "Same Arce expression / overlap genes as `runs/task5_batchfix/`.\n",
        "## 5a — Decode transfer (fresh ridge on Arce controls)\n",
        "| model | dim | in-dist ridge | transfer ridge | Δ (transfer−in-dist) | strong stratum | p@20 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, r in [("exp4_armA", e4), ("arm1 (batchfix)", arm1), ("pca", pca)]:
        if not r:
            continue
        dt = r.get("decode_transfer", {})
        pool = dt.get("pooled", {})
        strong = dt.get("by_stratum", {}).get("strong", {})
        ind = r.get("in_distribution", {})
        lines.append(
            f"| {label} | {r.get('latent_dim', '?')} | "
            f"{ind.get('ridge', float('nan')):.3f} | "
            f"{pool.get('ridge_delta_pearson', float('nan')):.3f} | "
            f"{pool.get('ridge_delta_pearson', 0) - ind.get('ridge', 0):+.3f} | "
            f"{strong.get('ridge_delta_pearson', float('nan')):.3f} | "
            f"{pool.get('ridge_p@20', float('nan')):.2f} |"
        )

    lines += [
        "\n## 5b — Cross-dataset agreement (HEADLINE)\n",
        "| model | matched cos | mismatched | separation | AUROC | |sCZI| | |sArce| | retention |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, r in [("exp4_armA", e4), ("arm1 (batchfix)", arm1), ("pca", pca)]:
        if not r:
            continue
        a = r.get("agreement", {})
        lines.append(
            f"| {label} | {a.get('matched_mean', float('nan')):.3f} | "
            f"{a.get('mismatched_mean', float('nan')):.3f} | "
            f"{a.get('separation', float('nan')):.3f} | "
            f"{a.get('auroc_matched_vs_mismatched', float('nan')):.3f} | "
            f"{a.get('czi_signature_norm_mean', float('nan')):.3f} | "
            f"{a.get('arce_signature_norm_mean', float('nan')):.3f} | "
            f"{a.get('signal_retention', float('nan')):.3f} |"
        )

    lines += [
        "\n### Per overlapping gene (matched cosine)\n",
        "| gene | exp4_armA | arm1 | pca |",
        "|---|---:|---:|---:|",
    ]
    for g in overlap:
        e4c = e4["agreement"]["matched_cosine_per_gene"].get(g, float("nan"))
        a1c = arm1.get("agreement", {}).get("matched_cosine_per_gene", {}).get(g, float("nan"))
        pcac = pca.get("agreement", {}).get("matched_cosine_per_gene", {}).get(g, float("nan"))
        lines.append(f"| {g} | {e4c:.3f} | {a1c:.3f} | {pcac:.3f} |")

    lines += [
        "\n## Step 6 — Batch integration (kNN dataset purity, Rest controls)\n",
        "| model | purity |",
        "|---|---:|",
    ]
    for label, r in [("exp4_armA", e4), ("arm1 (batchfix)", arm1), ("pca", pca)]:
        if not r:
            continue
        bd = r.get("batch_diagnostic", {})
        lines.append(f"| {label} | {bd.get('knn_dataset_purity', float('nan')):.3f} |")

    e4_dec = pool.get("ridge_delta_pearson", 0) if (pool := e4["decode_transfer"].get("pooled")) else 0
    pca_dec = pca.get("decode_transfer", {}).get("pooled", {}).get("ridge_delta_pearson", 0)
    arm1_dec = arm1.get("decode_transfer", {}).get("pooled", {}).get("ridge_delta_pearson", 0)
    e4_auc = e4["agreement"]["auroc_matched_vs_mismatched"]
    pca_auc = pca.get("agreement", {}).get("auroc_matched_vs_mismatched", 0)

    lines += [
        "\n## Verdict\n",
        f"- **5a decode:** exp4 Arm A transfer ridge **{e4_dec:.3f}** vs PCA **{pca_dec:.3f}** vs arm1 **{arm1_dec:.3f}**.",
        f"- **5b agreement:** exp4 AUROC **{e4_auc:.3f}** vs PCA **{pca_auc:.3f}** "
        f"(matched cos {e4['agreement']['matched_mean']:.3f} vs {pca.get('agreement',{}).get('matched_mean',0):.3f}).",
        f"- **Retention |sArce|/|sCZI|:** {e4['agreement']['signal_retention']:.3f} "
        f"(arm1 {arm1.get('agreement',{}).get('signal_retention',0):.3f}, pca {pca.get('agreement',{}).get('signal_retention',0):.3f}).",
    ]

    report = out_dir / "TASK5_EXP4_ARM_A_REPORT.md"
    report.write_text("\n".join(lines))
    _log(f"Wrote {report}")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
