"""Representation diagnostics for exp4 Arm A (Checks 1–3).

Check 1: SVD spectrum of exp4-none control embeddings (is eff-rank=1.8 real?)
Check 2: Batch residual presence (balanced acc, AUROC, batch vs biology subspace)
Check 3: Task 5 cross-dataset transfer vs batch-fixed arm1 reference

Usage:
    PYTHONPATH=src python scripts/exp4_representation_checks.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jepa_poc.config import ensure_dir, load_config  # noqa: E402
from jepa_poc.eval.metrics import delta_pearson  # noqa: E402
from jepa_poc.eval.perturbation import (  # noqa: E402
    condition_group_means,
    fit_linear_decoder,
    decode_latent,
    matched_control_means,
)
from jepa_poc.models.gene_tokenizer import ESMGeneTokenEncoder  # noqa: E402


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


@torch.no_grad()
def embed_values(
    encoder: ESMGeneTokenEncoder,
    values: np.ndarray,
    batch_id: np.ndarray | None,
    device: str,
    batch_size: int = 512,
    batch_mode: str = "none",
) -> np.ndarray:
    encoder.eval().to(device)
    n_batches = int(encoder.batch_embedding.num_embeddings)
    out = []
    use_cuda = device == "cuda"
    for i in range(0, values.shape[0], batch_size):
        v = torch.as_tensor(values[i : i + batch_size], dtype=torch.float32, device=device)
        with torch.autocast(device_type="cuda" if use_cuda else "cpu", dtype=torch.bfloat16, enabled=use_cuda):
            if batch_mode == "none":
                z = encoder(v, None)[:, 0, :]
            else:
                bid = torch.as_tensor(batch_id[i : i + batch_size], dtype=torch.long, device=device)
                z = encoder(v, bid)[:, 0, :]
        out.append(z.float().cpu().numpy())
    return np.concatenate(out, axis=0)


def load_encoder(cfg, encoder_path: Path, device: str) -> ESMGeneTokenEncoder:
    esm = np.load(cfg.data.esm_embeddings_path, allow_pickle=True)
    enc = ESMGeneTokenEncoder(
        esm_embeddings=esm["embeddings"], fallback_mask=esm["fallback_mask"], n_batches=2,
        d_model=int(cfg.model.d_model), d_id_proj=int(cfg.model.d_id_proj),
        d_expr=int(cfg.model.d_expr), use_fallback_indicator=bool(cfg.model.use_fallback_indicator),
        n_layers=int(cfg.model.n_layers), n_heads=int(cfg.model.n_heads),
        dropout=float(cfg.model.dropout),
    )
    state = torch.load(encoder_path, map_location="cpu", weights_only=False)
    enc.load_state_dict(state, strict=True)
    return enc.to(device)


def stratified_batch_sample(batch_str: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    groups = np.unique(batch_str)
    per = n // len(groups)
    keep = [rng.choice(np.where(batch_str == g)[0], size=per, replace=False) for g in groups]
    return np.sort(np.concatenate(keep))


def check1_svd(z: np.ndarray, out_dir: Path) -> dict:
    z = np.asarray(z, np.float64)
    centered = z - z.mean(axis=0, keepdims=True)
    # economy SVD
    _, s, _ = np.linalg.svd(centered, full_matrices=False)
    var = s ** 2
    var_norm = var / var.sum()
    cum = np.cumsum(var_norm)
    n90 = int(np.searchsorted(cum, 0.90) + 1)
    n95 = int(np.searchsorted(cum, 0.95) + 1)
    # eff-rank (entropy of normalized spectrum)
    p = var_norm[var_norm > 0]
    eff_rank = float(np.exp(-(p * np.log(p)).sum()))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(np.arange(1, len(s) + 1), var_norm, "o-", ms=3)
    axes[0].axvline(n90, color="red", ls="--", label=f"90% var @ {n90} dims")
    axes[0].set_xlabel("Singular value index")
    axes[0].set_ylabel("Normalized variance")
    axes[0].set_title("Singular value spectrum (exp4-none, 20K control)")
    axes[0].legend()
    axes[0].set_xlim(0, min(100, len(s)))
    axes[1].semilogy(np.arange(1, min(51, len(s) + 1)), s[:50], "o-", ms=4)
    axes[1].set_xlabel("Index")
    axes[1].set_ylabel("Singular value (log)")
    axes[1].set_title(f"Top-50 singular values | eff-rank={eff_rank:.2f}")
    fig.tight_layout()
    fig.savefig(out_dir / "check1_svd_spectrum.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    top5_share = float(var_norm[:5].sum())
    top2_share = float(var_norm[:2].sum())
    return {
        "n_cells": int(z.shape[0]), "n_dims": int(z.shape[1]),
        "effective_rank_entropy": eff_rank,
        "n_singular_values_for_90pct_variance": n90,
        "n_singular_values_for_95pct_variance": n95,
        "top1_variance_share": float(var_norm[0]),
        "top2_variance_share": top2_share,
        "top5_variance_share": top5_share,
        "verdict": "2D_manifold" if n90 <= 3 else ("smooth_decay_many_dims" if n90 >= 20 else "intermediate"),
    }


def _fit_direction(z: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, dict]:
    """Logistic regression weights as a direction; report balanced acc + AUROC."""
    scaler = StandardScaler()
    x = scaler.fit_transform(z)
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.25, random_state=13, stratify=y)
    clf = LogisticRegression(max_iter=3000, random_state=13)
    clf.fit(x_tr, y_tr)
    pred = clf.predict(x_te)
    proba = clf.predict_proba(x_te)
    labels = np.unique(y)
    counts = np.bincount(np.array([{v: i for i, v in enumerate(labels)}[yy] for yy in y]))
    metrics = {
        "raw_accuracy": float(accuracy_score(y_te, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_te, pred)),
        "majority_class_rate": float(counts.max() / len(y)),
        "class_counts": {str(l): int((y == l).sum()) for l in labels},
    }
    if len(labels) == 2:
        metrics["auroc"] = float(roc_auc_score(y_te, proba[:, 1]))
    else:
        metrics["auroc"] = float(roc_auc_score(y_te, proba, multi_class="ovr", average="macro"))
    # direction: mean absolute coef across classes
    w = clf.coef_.mean(axis=0)
    w = w / (np.linalg.norm(w) + 1e-12)
    return w.astype(np.float64), metrics


def check2_batch_vs_biology(z: np.ndarray, batch_str: np.ndarray, cond: np.ndarray, out_dir: Path) -> dict:
    w_batch, batch_m = _fit_direction(z, batch_str.astype(str))
    w_bio, bio_m = _fit_direction(z, cond.astype(str))
    cos_align = float(np.abs(np.dot(w_batch, w_bio)))
    return {
        "batch_classifier": batch_m,
        "biology_classifier_condition": bio_m,
        "batch_bio_direction_cosine": cos_align,
        "interpretation_batch": (
            "batch_linearly_present" if batch_m["balanced_accuracy"] > 0.6 else "batch_mostly_gone"
        ),
        "interpretation_alignment": (
            "aligned_harmful" if cos_align > 0.5 else "orthogonal_harmless"
        ),
    }


def load_czi_pert_from_benchmark(bench: np.lib.npyio.NpzFile, overlap: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CZI pert cells for overlap genes from benchmark train+test pools (matches task5 build)."""
    gene = np.concatenate([bench["train_gene"].astype(str), bench["test_gene"].astype(str)])
    expr = np.concatenate([bench["train_expr"], bench["test_expr"]], axis=0)
    cond = np.concatenate([bench["train_cond"].astype(str), bench["test_cond"].astype(str)])
    ov = np.asarray(overlap, dtype=str)
    mask = np.isin(gene, ov)
    return expr[mask], gene[mask], cond[mask]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def task5_agreement(
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
    ctrl_z_by_cond_czi = condition_group_means(
        np.zeros((len(czi_ctrl_cond), 1)), czi_ctrl_z, czi_ctrl_cond
    )
    ctrl_z_by_cond_arce = condition_group_means(
        np.zeros((len(arce_ctrl_cond), 1)), arce_ctrl_z, arce_ctrl_cond
    )

    matched_cos = {}
    czi_norms, arce_norms = [], []
    for g in overlap:
        m_czi = czi_genes.astype(str) == g
        m_arce = arce_genes.astype(str) == g
        if m_czi.sum() < 5 or m_arce.sum() < 5:
            continue
        zczi = czi_z[m_czi]
        zarce = arce_z[m_arce]
        ref_czi, _ = matched_control_means(ctrl_z_by_cond_czi, czi_cond[m_czi])
        ref_arce, _ = matched_control_means(ctrl_z_by_cond_arce, arce_cond[m_arce])
        s_czi = zczi.mean(0) - ref_czi
        s_arce = zarce.mean(0) - ref_arce
        matched_cos[g] = _cosine(s_czi, s_arce)
        czi_norms.append(float(np.linalg.norm(s_czi)))
        arce_norms.append(float(np.linalg.norm(s_arce)))

    genes = list(matched_cos.keys())
    mismatched = []
    for i, g in enumerate(genes):
        for h in genes:
            if g == h:
                continue
            m_czi = czi_genes.astype(str) == g
            m_arce = arce_genes.astype(str) == h
            ref_czi, _ = matched_control_means(ctrl_z_by_cond_czi, czi_cond[m_czi])
            ref_arce, _ = matched_control_means(ctrl_z_by_cond_arce, arce_cond[m_arce])
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
        "mismatched_mean": float(np.mean(mismatched)) if mismatched else None,
        "separation": float(np.mean(matched_vals) - np.mean(mismatched)) if matched_vals and mismatched else None,
        "auroc_matched_vs_mismatched": auroc,
        "czi_signature_norm_mean": czi_mean_norm,
        "arce_signature_norm_mean": arce_mean_norm,
        "signal_retention": arce_mean_norm / (czi_mean_norm + 1e-12),
    }


def task5_decode_transfer(
    arce_pert_gene: np.ndarray,
    arce_pert_expr: np.ndarray,
    arce_pert_z: np.ndarray,
    arce_pert_cond: np.ndarray,
    arce_ctrl_expr: np.ndarray,
    arce_ctrl_z: np.ndarray,
    arce_ctrl_cond: np.ndarray,
) -> dict:
    decoder = fit_linear_decoder(arce_ctrl_z, arce_ctrl_expr, ridge=1e-3)
    ctrl_means = condition_group_means(arce_ctrl_expr, arce_ctrl_z, arce_ctrl_cond)
    genes = np.unique(arce_pert_gene.astype(str))
    scores = []
    for g in genes:
        if g.lower() in {"control", "non-targeting", "aavs1", "ntc"}:
            continue
        m = arce_pert_gene.astype(str) == g
        if m.sum() < 20:
            continue
        expr_m, z_m, cond_m = arce_pert_expr[m], arce_pert_z[m], arce_pert_cond[m]
        ctrl_expr_ref, ctrl_z_ref = matched_control_means(ctrl_means, cond_m)
        true_delta = expr_m.mean(0) - ctrl_expr_ref
        pred_delta = decode_latent(z_m.mean(0, keepdims=True), decoder)[0] - decode_latent(ctrl_z_ref.reshape(1, -1), decoder)[0]
        scores.append(delta_pearson(pred_delta, true_delta))
    return {"ridge_delta_pearson": float(np.mean(scores)), "n_perturbations": len(scores)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="runs/exp4_armA/ema_target_encoder.pt")
    ap.add_argument("--embeddings-none", default="runs/benchmark/exp4_armA/embeddings_none.npz")
    ap.add_argument("--task5-arm1", default="runs/task5_batchfix/arm1/task5.npz")
    ap.add_argument("--task5-ref", default="runs/task5_batchfix/task5_results.json")
    ap.add_argument("--benchmark-train", default="runs/benchmark/exp4_armA/embeddings_none.npz")
    ap.add_argument("--out-dir", default="runs/benchmark/exp4_armA/checks")
    ap.add_argument("--n-svd", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    cfg = load_config("configs/exp3.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = ensure_dir(Path(args.out_dir))
    results: dict = {}

    # --- load control cells for checks 1 & 2 ---
    _log("Loading control embeddings + metadata")
    emb = np.load(args.embeddings_none, allow_pickle=True)
    z_all = emb["control_z"]
    cond_all = emb["control_cond"].astype(str)

    controls_path = Path(cfg.data.perturb_controls_processed_path)
    adata = ad.read_h5ad(controls_path)
    ctrl_cond_full = adata.obs["culture_condition"].astype(str).to_numpy()
    batch_full = adata.obs[cfg.data.batch_key].astype(str).to_numpy()
    rng = np.random.default_rng(args.seed)
    conds = np.unique(ctrl_cond_full)
    per = 150_000 // len(conds)
    idx_parts = []
    for c in conds:
        pool = np.where(ctrl_cond_full == c)[0]
        idx_parts.append(rng.choice(pool, size=min(per, len(pool)), replace=False))
    pool_idx = np.sort(np.concatenate(idx_parts))
    batch_all = batch_full[pool_idx]
    cond_all = ctrl_cond_full[pool_idx]
    z_all = emb["control_z"]
    if len(pool_idx) != len(z_all):
        _log(f"WARN: pool_idx {len(pool_idx)} != control_z {len(z_all)}; using first {len(z_all)}")
        batch_all = batch_all[: len(z_all)]
        cond_all = cond_all[: len(z_all)]

    geo_idx = stratified_batch_sample(batch_all, args.n_svd, args.seed)
    z20 = z_all[geo_idx]
    batch20 = batch_all[geo_idx]
    cond20 = cond_all[geo_idx]

    _log("Check 1: SVD spectrum")
    results["check1_svd"] = check1_svd(z20, out_dir)

    _log("Check 2: batch vs biology subspace")
    results["check2_batch_biology"] = check2_batch_vs_biology(z20, batch20, cond20, out_dir)

    # --- Check 3: Task 5 transfer ---
    _log("Check 3: building exp4 task5 embeddings (batch_mode=none)")
    encoder = load_encoder(cfg, Path(args.encoder), device)
    t5 = np.load(args.task5_arm1, allow_pickle=True)
    overlap = t5["overlap_genes"].astype(str).tolist()

    arce_ctrl_z = embed_values(encoder, t5["arce_ctrl_expr"], None, device, batch_mode="none")
    arce_pert_z = embed_values(encoder, t5["arce_pert_expr"], None, device, batch_mode="none")

    # CZI pert: benchmark train+test overlap genes, re-embed with exp4-none
    bench = np.load(args.benchmark_train, allow_pickle=True)
    czi_pert_expr, czi_pert_gene, czi_pert_cond = load_czi_pert_from_benchmark(bench, overlap)
    _log(f"CZI pert cells for overlap genes: {len(czi_pert_gene):,}")
    czi_pert_z = embed_values(encoder, czi_pert_expr, None, device, batch_mode="none")
    czi_ctrl_z = bench["control_z"]
    czi_ctrl_cond = bench["control_cond"].astype(str)

    _log("Check 3: scoring agreement + decode transfer")
    agreement = task5_agreement(
        czi_pert_gene, czi_pert_z, czi_pert_cond, czi_ctrl_z, czi_ctrl_cond,
        t5["arce_pert_gene"], arce_pert_z, t5["arce_pert_cond"].astype(str),
        arce_ctrl_z, t5["arce_ctrl_cond"].astype(str),
        overlap,
    )
    decode = task5_decode_transfer(
        t5["arce_pert_gene"], t5["arce_pert_expr"], arce_pert_z, t5["arce_pert_cond"].astype(str),
        t5["arce_ctrl_expr"], arce_ctrl_z, t5["arce_ctrl_cond"].astype(str),
    )
    results["check3_task5"] = {"agreement": agreement, "decode_transfer": decode}

    ref = json.loads(Path(args.task5_ref).read_text()).get("arm1", {})
    results["arm1_reference_task5"] = {
        "agreement": ref.get("agreement", {}),
        "decode_transfer_ridge": ref.get("decode_transfer", {}).get("pooled", {}).get("ridge_delta_pearson"),
    }

    out_json = out_dir / "exp4_representation_checks.json"
    out_json.write_text(json.dumps(results, indent=2))
    _log(f"Wrote {out_json}")

    # markdown report
    c1 = results["check1_svd"]
    c2 = results["check2_batch_biology"]
    c3a = agreement
    arm1a = results["arm1_reference_task5"]["agreement"]
    lines = [
        "# exp4 Arm A — Representation Checks 1–3\n",
        "## Check 1 — SVD spectrum (exp4-none, 20K control)\n",
        f"- Effective rank (entropy): **{c1['effective_rank_entropy']:.2f}**",
        f"- Singular values for **90%** variance: **{c1['n_singular_values_for_90pct_variance']}**",
        f"- Top-2 variance share: **{c1['top2_variance_share']:.1%}** | Top-5: **{c1['top5_variance_share']:.1%}**",
        f"- Verdict: **{c1['verdict']}**",
        f"- Plot: `check1_svd_spectrum.png`\n",
        "## Check 2 — Batch residual vs biology\n",
        f"| classifier | raw acc | balanced acc | AUROC |",
        f"|---|---|---|---|",
        f"| batch (10xrun) | {c2['batch_classifier']['raw_accuracy']:.1%} | "
        f"{c2['batch_classifier']['balanced_accuracy']:.1%} | {c2['batch_classifier']['auroc']:.3f} |",
        f"| culture condition | {c2['biology_classifier_condition']['raw_accuracy']:.1%} | "
        f"{c2['biology_classifier_condition']['balanced_accuracy']:.1%} | "
        f"{c2['biology_classifier_condition']['auroc']:.3f} |",
        f"- Batch–biology direction cosine: **{c2['batch_bio_direction_cosine']:.3f}** "
        f"({c2['interpretation_alignment']})",
        f"- Batch classifier: {c2['interpretation_batch']}\n",
        "## Check 3 — Task 5 transfer (batch_mode=none) vs arm1 batchfix\n",
        "| metric | exp4_armA | arm1 (ref) |",
        "|---|---|---|",
        f"| 5a decode transfer ridge Δ-Pearson | {decode['ridge_delta_pearson']:.3f} | "
        f"{results['arm1_reference_task5']['decode_transfer_ridge']:.3f} |",
        f"| 5b matched cosine (mean) | {c3a['matched_mean']:.3f} | {arm1a.get('matched_mean', 0):.3f} |",
        f"| 5b AUROC matched vs mismatched | {c3a['auroc_matched_vs_mismatched']:.3f} | "
        f"{arm1a.get('auroc_matched_vs_mismatched', 0):.3f} |",
        f"| 5b signal retention (|sArce|/|sCZI|) | {c3a['signal_retention']:.3f} | "
        f"{arm1a.get('signal_retention', 0):.3f} |",
        f"| |sCZI| mean | {c3a['czi_signature_norm_mean']:.3f} | {arm1a.get('czi_signature_norm_mean', 0):.3f} |",
        f"| |sArce| mean | {c3a['arce_signature_norm_mean']:.3f} | {arm1a.get('arce_signature_norm_mean', 0):.3f} |",
    ]
    arm1_dec = results["arm1_reference_task5"]["decode_transfer_ridge"]
    exp4_dec = decode["ridge_delta_pearson"]
    exp4_better_dec = exp4_dec > arm1_dec
    exp4_better_agree = (
        c3a["auroc_matched_vs_mismatched"] > arm1a.get("auroc_matched_vs_mismatched", 0)
        and c3a["separation"] > arm1a.get("separation", 0)
    )
    lines += [
        "\n## Check 3 verdict\n",
        (
            f"**Decode transfer (decisive): exp4 {exp4_dec:.3f} vs arm1 {arm1_dec:.3f}** — "
            + ("exp4 wins." if exp4_better_dec else "arm1 wins; low eff-rank / 2D geometry may be limiting transfer.")
        ),
        (
            f"Agreement (5b): matched cos {c3a['matched_mean']:.3f} vs {arm1a.get('matched_mean', 0):.3f}, "
            f"AUROC {c3a['auroc_matched_vs_mismatched']:.3f} vs {arm1a.get('auroc_matched_vs_mismatched', 0):.3f} "
            + ("(exp4 better)" if exp4_better_agree else "(arm1 better or degenerate — see Check 1 if all cosines ~1)")
        ),
    ]
    report = out_dir / "EXP4_REPRESENTATION_CHECKS_REPORT.md"
    report.write_text("\n".join(lines))
    _log(f"Wrote {report}")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
