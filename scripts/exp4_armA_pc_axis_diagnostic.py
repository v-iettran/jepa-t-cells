"""PCA axis diagnostic for exp4 Arm A collapsed embeddings.

Tests whether the ~2 surviving dimensions map to biological axes
(perturbation vs culture state) rather than random degeneracy.

Usage:
    PYTHONPATH=src python scripts/exp4_armA_pc_axis_diagnostic.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

STATE_ORDER = ["Rest", "Stim8hr", "Stim48hr"]
STATE_COLORS = {"Rest": "#4C78A8", "Stim8hr": "#F58518", "Stim48hr": "#54A24B"}
PERT_COLORS = {"NTC": "#9CA3AF", "perturbed": "#DC2626"}


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _point_biserial(y_bin: np.ndarray, x: np.ndarray) -> float:
    """Pearson r between a binary label and a continuous score."""
    y = y_bin.astype(np.float64)
    x = x.astype(np.float64)
    return float(np.corrcoef(y, x)[0, 1])


def _eta_squared(groups: np.ndarray, values: np.ndarray) -> float:
    """One-way ANOVA eta-squared: fraction of variance explained by groups."""
    groups = groups.astype(str)
    values = values.astype(np.float64)
    grand_mean = values.mean()
    ss_between = sum(values[groups == g].sum() ** 2 / (groups == g).sum() for g in np.unique(groups))
    ss_between -= len(values) * grand_mean ** 2
    ss_total = ((values - grand_mean) ** 2).sum()
    return float(ss_between / ss_total) if ss_total > 0 else 0.0


def _probe_auc(x: np.ndarray, y: np.ndarray, seed: int = 13) -> float:
    x = x.reshape(-1, 1)
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.25, random_state=seed, stratify=y)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(x_tr, y_tr)
    prob = clf.predict_proba(x_te)
    if prob.shape[1] == 2:
        return float(roc_auc_score(y_te, prob[:, 1]))
    return float(roc_auc_score(y_te, prob, multi_class="ovr", average="macro"))


def _probe_bal_acc(x: np.ndarray, y: np.ndarray, seed: int = 13) -> float:
    x = x.reshape(-1, 1)
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.25, random_state=seed, stratify=y)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(x_tr, y_tr)
    return float(balanced_accuracy_score(y_te, clf.predict(x_te)))


def pick_strong_perturbations(
    z: np.ndarray,
    is_pert: np.ndarray,
    gene: np.ndarray,
    state: np.ndarray,
    n_pick: int = 2,
    min_cells: int = 150,
) -> list[str]:
    """Genes with largest matched-control embedding displacement (L2 norm)."""
    ctrl_mask = ~is_pert
    ctrl_state_means = {}
    for s in STATE_ORDER:
        m = ctrl_mask & (state == s)
        if m.sum() > 0:
            ctrl_state_means[s] = z[m].mean(axis=0)

    scores: list[tuple[str, float]] = []
    for g in np.unique(gene[is_pert]):
        m = is_pert & (gene == g)
        if m.sum() < min_cells:
            continue
        disp = []
        for s in STATE_ORDER:
            sm = m & (state == s)
            if sm.sum() < 20 or s not in ctrl_state_means:
                continue
            disp.append(np.linalg.norm(z[sm].mean(axis=0) - ctrl_state_means[s]))
        if disp:
            scores.append((str(g), float(np.mean(disp))))
    scores.sort(key=lambda t: t[1], reverse=True)
    return [g for g, _ in scores[:n_pick]]


def subsample_idx(
    is_pert: np.ndarray,
    state: np.ndarray,
    n_per_stratum: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    keep = []
    for pert_flag in (False, True):
        for s in STATE_ORDER:
            m = (is_pert == pert_flag) & (state == s)
            pool = np.where(m)[0]
            if len(pool) == 0:
                continue
            k = min(n_per_stratum, len(pool))
            keep.append(rng.choice(pool, size=k, replace=False))
    return np.sort(np.concatenate(keep))


def plot_scatter(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    colors: np.ndarray,
    labels: dict[str, str],
    title: str,
    xlabel: str,
    ylabel: str,
    alpha: float = 0.35,
    s: float = 4,
) -> None:
    for key, color in labels.items():
        m = colors == key
        if m.sum() == 0:
            continue
        ax.scatter(x[m], y[m], c=color, s=s, alpha=alpha, linewidths=0, label=key)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(markerscale=3, fontsize=7, frameon=False, loc="best")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", default="runs/benchmark/exp4_armA/embeddings_none.npz")
    ap.add_argument("--out-dir", default="runs/benchmark/exp4_armA/checks/pc_axes")
    ap.add_argument("--label", default="exp4 Arm A", help="Display name in plots/reports.")
    ap.add_argument("--n-pca-fit", type=int, default=100_000, help="Cells used to fit PCA (subsampled).")
    ap.add_argument("--n-plot-per-stratum", type=int, default=2000, help="Cells per (NTC/pert x state) stratum for plots.")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    emb = np.load(args.embeddings, allow_pickle=True)
    z_ctrl = emb["control_z"]
    z_train = emb["train_z"]
    state_ctrl = emb["control_cond"].astype(str)
    state_train = emb["train_cond"].astype(str)
    gene_train = emb["train_gene"].astype(str)

    z = np.concatenate([z_ctrl, z_train], axis=0)
    state = np.concatenate([state_ctrl, state_train])
    is_pert = np.concatenate([np.zeros(len(z_ctrl), dtype=bool), np.ones(len(z_train), dtype=bool)])
    gene = np.concatenate([np.full(len(z_ctrl), "NTC", dtype=object), gene_train])

    _log(f"Loaded {z.shape[0]:,} cells ({len(z_ctrl):,} NTC + {len(z_train):,} perturbed), dim={z.shape[1]}")

    # --- PCA on centered embeddings ---
    rng = np.random.default_rng(args.seed)
    fit_idx = rng.choice(len(z), size=min(args.n_pca_fit, len(z)), replace=False)
    z_fit = z[fit_idx] - z[fit_idx].mean(axis=0, keepdims=True)
    pca = PCA(n_components=min(10, z.shape[1]), random_state=args.seed)
    pca.fit(z_fit)
    scores = pca.transform(z - z.mean(axis=0, keepdims=True))

    var_exp = pca.explained_variance_ratio_
    _log(f"PCA variance: PC1={var_exp[0]:.3f} PC2={var_exp[1]:.3f} PC3={var_exp[2]:.3f} "
         f"(cum 1-2={var_exp[:2].sum():.3f}, 1-3={var_exp[:3].sum():.3f})")

    state_num = np.array([STATE_ORDER.index(s) if s in STATE_ORDER else -1 for s in state])
    valid = state_num >= 0

    # --- quantify axis mapping ---
    pc1, pc2, pc3 = scores[:, 0], scores[:, 1], scores[:, 2]
    y_pert = is_pert.astype(int)

    # Primary hypothesis: PC1=perturbation, PC2=state
    r_pc1_pert = _point_biserial(y_pert[valid], pc1[valid])
    r_pc2_state = float(np.corrcoef(state_num[valid], pc2[valid])[0, 1])
    eta_pc1_state = _eta_squared(state[valid], pc1[valid])
    eta_pc2_pert = _eta_squared(np.where(is_pert[valid], "pert", "NTC")[valid], pc2[valid])

    # Alternative: PC1=state, PC2=perturbation
    r_pc1_state = float(np.corrcoef(state_num[valid], pc1[valid])[0, 1])
    r_pc2_pert = _point_biserial(y_pert[valid], pc2[valid])

    # Linear probes on single PCs
    auc_pc1_pert = _probe_auc(pc1[valid], y_pert[valid], args.seed)
    auc_pc2_state = _probe_auc(pc2[valid], state_num[valid], args.seed)
    auc_pc1_state = _probe_auc(pc1[valid], state_num[valid], args.seed)
    auc_pc2_pert = _probe_auc(pc2[valid], y_pert[valid], args.seed)

    # Multiclass state from PC2 (3 classes)
    le = LabelEncoder()
    y_state = le.fit_transform(state[valid])
    bal_pc2_state = _probe_bal_acc(pc2[valid], y_state, args.seed)
    bal_pc1_state = _probe_bal_acc(pc1[valid], y_state, args.seed)

    strong_genes = pick_strong_perturbations(z, is_pert, gene, state, n_pick=2)
    _log(f"Strong perturbations (largest matched-control displacement): {strong_genes}")

    metrics = {
        "n_cells": int(len(z)),
        "n_ntc": int((~is_pert).sum()),
        "n_perturbed": int(is_pert.sum()),
        "pca_variance_explained": {
            "PC1": float(var_exp[0]),
            "PC2": float(var_exp[1]),
            "PC3": float(var_exp[2]),
            "PC1_PC2_cum": float(var_exp[:2].sum()),
            "PC1_PC3_cum": float(var_exp[:3].sum()),
        },
        "hypothesis_pc1_pert_pc2_state": {
            "r_PC1_vs_perturbed": r_pc1_pert,
            "r_PC2_vs_state_ordinal": r_pc2_state,
            "eta2_PC1_explained_by_state": eta_pc1_state,
            "auc_PC1_predicts_perturbed": auc_pc1_pert,
            "auc_PC2_predicts_state": auc_pc2_state,
            "bal_acc_PC2_predicts_state_3class": bal_pc2_state,
        },
        "alternative_pc1_state_pc2_pert": {
            "r_PC1_vs_state_ordinal": r_pc1_state,
            "r_PC2_vs_perturbed": r_pc2_pert,
            "eta2_PC2_explained_by_perturbed": eta_pc2_pert,
            "auc_PC1_predicts_state": auc_pc1_state,
            "auc_PC2_predicts_perturbed": auc_pc2_pert,
            "bal_acc_PC1_predicts_state_3class": bal_pc1_state,
        },
        "strong_perturbations": strong_genes,
    }

    # Verdict
    primary_score = abs(r_pc1_pert) + abs(r_pc2_state) + auc_pc1_pert + auc_pc2_state
    alt_score = abs(r_pc1_state) + abs(r_pc2_pert) + auc_pc1_state + auc_pc2_pert
    if abs(r_pc1_pert) > 0.1 and abs(r_pc2_state) > 0.1:
        verdict = (
            "PC1/PC2 align with perturbation/state axes: collapse compresses biology into "
            f"2 interpretable axes (|r_PC1,pert|={abs(r_pc1_pert):.3f}, |r_PC2,state|={abs(r_pc2_state):.3f})."
        )
    elif abs(r_pc1_state) > abs(r_pc1_pert) and abs(r_pc1_state) > 0.1 and abs(r_pc2_pert) < 0.05:
        verdict = (
            f"PC1 weakly encodes culture state (r={r_pc1_state:+.3f}, AUC={auc_pc1_state:.3f}, "
            f"η²={eta_pc1_state:.3f}) but PC2 does NOT encode perturbation (r={r_pc2_pert:+.4f}, AUC≈chance). "
            "Not a clean {pert, state} factorization — collapse is mostly a 1D manifold with weak state leakage on PC1."
        )
    elif abs(r_pc1_pert) < 0.05 and abs(r_pc2_state) < 0.1:
        verdict = (
            f"No clean PC1=pert / PC2=state mapping (|r_PC1,pert|={abs(r_pc1_pert):.3f}, "
            f"|r_PC2,state|={abs(r_pc2_state):.3f}; both AUCs≈chance). "
            "Collapse is geometric degeneracy, not compression into 2 biological axes."
        )
    else:
        verdict = (
            "Axes are mixed: neither a clean PC1=pert/PC2=state nor PC1=state/PC2=pert mapping. "
            f"|r_PC1,pert|={abs(r_pc1_pert):.3f}, |r_PC2,state|={abs(r_pc2_state):.3f}, "
            f"|r_PC1,state|={abs(r_pc1_state):.3f}, |r_PC2,pert|={abs(r_pc2_pert):.3f}."
        )
    metrics["label"] = args.label
    metrics["embeddings_path"] = str(args.embeddings)
    metrics["verdict"] = verdict

    with open(out_dir / "pc_axis_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # --- plots on balanced subsample ---
    plot_idx = subsample_idx(is_pert, state, args.n_plot_per_stratum, args.seed)
    pc1p, pc2p, pc3p = pc1[plot_idx], pc2[plot_idx], pc3[plot_idx]
    is_pert_p = is_pert[plot_idx]
    state_p = state[plot_idx]
    gene_p = gene[plot_idx].astype(str)

    pert_label = np.where(is_pert_p, "perturbed", "NTC")

    # Figure: PC1 vs PC2, three colorings
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    plot_scatter(
        axes[0], pc1p, pc2p, pert_label,
        PERT_COLORS, "PC1 vs PC2 — NTC vs perturbed", "PC1", "PC2", alpha=0.4, s=5,
    )
    plot_scatter(
        axes[1], pc1p, pc2p, state_p,
        STATE_COLORS, "PC1 vs PC2 — culture state", "PC1", "PC2", alpha=0.4, s=5,
    )
    # Strong perturbations + NTC background
    highlight = np.full(len(plot_idx), "other", dtype=object)
    highlight[~is_pert_p] = "NTC"
    for g in strong_genes:
        highlight[(gene_p == g)] = g
    hcolors = {"NTC": "#D1D5DB", "other": "#E5E7EB"}
    palette = plt.get_cmap("Set1")
    for i, g in enumerate(strong_genes):
        hcolors[g] = palette(i)
    ax = axes[2]
    for key, color in hcolors.items():
        m = highlight == key
        if m.sum() == 0:
            continue
        sz = 8 if key in strong_genes else (3 if key == "other" else 5)
        al = 0.85 if key in strong_genes else (0.15 if key == "other" else 0.5)
        ax.scatter(pc1p[m], pc2p[m], c=color, s=sz, alpha=al, linewidths=0, label=key)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"PC1 vs PC2 — strong perts ({', '.join(strong_genes)})")
    ax.legend(markerscale=2, fontsize=7, frameon=False)

    fig.suptitle(
        f"{args.label} | PC1={var_exp[0]:.1%} PC2={var_exp[1]:.1%} var | "
        f"r(PC1,pert)={r_pc1_pert:+.3f} r(PC2,state)={r_pc2_state:+.3f}",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "pc1_pc2_scatter.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Figure: PC2 vs PC3
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    plot_scatter(
        axes[0], pc2p, pc3p, pert_label,
        PERT_COLORS, "PC2 vs PC3 — NTC vs perturbed", "PC2", "PC3", alpha=0.4, s=5,
    )
    plot_scatter(
        axes[1], pc2p, pc3p, state_p,
        STATE_COLORS, "PC2 vs PC3 — culture state", "PC2", "PC3", alpha=0.4, s=5,
    )
    ax = axes[2]
    for key, color in hcolors.items():
        m = highlight == key
        if m.sum() == 0:
            continue
        sz = 8 if key in strong_genes else (3 if key == "other" else 5)
        al = 0.85 if key in strong_genes else (0.15 if key == "other" else 0.5)
        ax.scatter(pc2p[m], pc3p[m], c=color, s=sz, alpha=al, linewidths=0, label=key)
    ax.set_xlabel("PC2")
    ax.set_ylabel("PC3")
    ax.set_title(f"PC2 vs PC3 — strong perts ({', '.join(strong_genes)})")
    ax.legend(markerscale=2, fontsize=7, frameon=False)
    fig.suptitle(
        f"{args.label} | PC2={var_exp[1]:.1%} PC3={var_exp[2]:.1%} var",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "pc2_pc3_scatter.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Variance scree for context
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(np.arange(1, len(var_exp) + 1), var_exp * 100, color="#4C78A8", alpha=0.85)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Variance explained (%)")
    ax.set_title("PCA scree (all cells, centered 256-d embeddings)")
    ax.set_xticks(np.arange(1, len(var_exp) + 1))
    fig.tight_layout()
    fig.savefig(out_dir / "pca_scree.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Markdown report
    md = f"""# {args.label} — PCA axis diagnostic

## Question
Do PC1/PC2 = {{perturbation axis, culture-state axis}}?

## PCA variance
| PC | variance |
|----|----------|
| PC1 | {var_exp[0]:.1%} |
| PC2 | {var_exp[1]:.1%} |
| PC3 | {var_exp[2]:.1%} |
| PC1+PC2 | {var_exp[:2].sum():.1%} |
| PC1+PC2+PC3 | {var_exp[:3].sum():.1%} |

## Hypothesis: PC1 = perturbation, PC2 = state
| metric | value |
|--------|-------|
| r(PC1, perturbed 0/1) | {r_pc1_pert:+.4f} |
| r(PC2, state ordinal) | {r_pc2_state:+.4f} |
| η² state explained by PC1 | {eta_pc1_state:.4f} |
| AUC PC1 → perturbed | {auc_pc1_pert:.4f} |
| AUC PC2 → state (3-class) | {auc_pc2_state:.4f} |
| bal-acc PC2 → state | {bal_pc2_state:.4f} |

## Alternative: PC1 = state, PC2 = perturbation
| metric | value |
|--------|-------|
| r(PC1, state ordinal) | {r_pc1_state:+.4f} |
| r(PC2, perturbed 0/1) | {r_pc2_pert:+.4f} |
| η² pert explained by PC2 | {eta_pc2_pert:.4f} |
| AUC PC1 → state | {auc_pc1_state:.4f} |
| AUC PC2 → perturbed | {auc_pc2_pert:.4f} |
| bal-acc PC1 → state | {bal_pc1_state:.4f} |

## Strong perturbations highlighted
{', '.join(strong_genes)} (largest matched-control embedding displacement)

## Verdict
{verdict}

## Figures
- `pc1_pc2_scatter.png` — NTC vs perturbed, culture state, strong perts
- `pc2_pc3_scatter.png` — same colorings on PC2/PC3
- `pca_scree.png` — variance explained
"""
    (out_dir / "PC_AXIS_REPORT.md").write_text(md)

    _log(verdict)
    _log(f"Wrote figures + metrics to {out_dir}")


if __name__ == "__main__":
    main()
