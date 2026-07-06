"""Experiment 2: UMAP of JEPA gene embeddings for A0 (VICReg) and A1 (SIGReg).

For each model we take the learned gene_embedding.weight (4000 x 256) from the
EMA target encoder, L2-normalise (cosine geometry, matching the TF enrichment),
build a shared cosine kNN graph, then run Leiden clustering and a UMAP layout on
that same graph (seed=1) so the projection and the clusters are consistent.

Colour encodes the Leiden cluster; marker shape encodes the gene category against
two reference sets restricted to the 4000 HVGs:
  * BTLA-TCR-4h DEGs  -> stars
  * Lambert v1.01 TFs -> dots (with a dark edge)
  * genes that are both -> stars with a heavier edge (DEG shape takes priority)
  * all other genes    -> small faint dots
The top enriched TFs (from the Stage-1 enrichment CSVs) are text-labelled.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from matplotlib.lines import Line2D

VIET_ROOT = Path("/mnt/R0/Projects/POIAZ/Viet")
JEPA_ROOT = VIET_ROOT / "JEPA-for-t-cells"
DEFAULT_TF = VIET_ROOT / "models_TFs_analysis/outputs/tf_embedding_neighbor_outputs/TF_names_v_1.01.txt"
DEFAULT_DEG = VIET_ROOT / "BulkFormer/data/original_processed_data/BTLAvsTCR_4h_DEGs.csv"
DEFAULT_VOCAB = JEPA_ROOT / "configs/gene_vocab_4000.tsv"
SEED = 1


def load_gene_embedding(ckpt: Path, encoder: str = "target") -> np.ndarray:
    obj = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    for key in (
        f"model.{encoder}_encoder.gene_embedding.weight",
        f"{encoder}_encoder.gene_embedding.weight",
        "gene_embedding.weight",
    ):
        if key in state:
            return state[key].float().numpy()
    raise KeyError(f"No gene_embedding weight found in {ckpt}.")


def compute_umap_leiden(emb: np.ndarray, resolution: float) -> tuple[np.ndarray, np.ndarray]:
    """Shared cosine kNN graph -> Leiden clusters + UMAP layout (consistent geometry)."""
    norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    adata = ad.AnnData(norm.astype(np.float32))
    sc.pp.neighbors(adata, n_neighbors=30, use_rep="X", metric="cosine", random_state=SEED)
    sc.tl.leiden(adata, resolution=resolution, random_state=SEED, flavor="igraph", n_iterations=2, directed=False)
    sc.tl.umap(adata, min_dist=0.3, random_state=SEED)
    xy = np.asarray(adata.obsm["X_umap"])
    labels = adata.obs["leiden"].to_numpy().astype(int)
    return xy, labels


def plot_panel(ax, xy, labels, genes_u, deg_set, tf_set, top_tfs, title) -> None:
    is_deg = np.array([g in deg_set for g in genes_u])
    is_tf = np.array([g in tf_set for g in genes_u])
    is_both = is_deg & is_tf
    is_deg_only = is_deg & ~is_tf
    is_tf_only = is_tf & ~is_deg
    is_other = ~is_deg & ~is_tf

    n_clusters = int(labels.max()) + 1
    palette = plt.get_cmap("tab20", max(n_clusters, 1))
    colors = palette(labels % 20) if n_clusters > 20 else palette(labels)

    ax.scatter(xy[is_other, 0], xy[is_other, 1], s=6, c=colors[is_other],
               linewidths=0, alpha=0.45, zorder=1)
    ax.scatter(xy[is_tf_only, 0], xy[is_tf_only, 1], s=34, c=colors[is_tf_only],
               linewidths=0.5, edgecolors="#1f2937", alpha=0.95, zorder=3)
    ax.scatter(xy[is_deg_only, 0], xy[is_deg_only, 1], s=130, c=colors[is_deg_only], marker="*",
               edgecolors="black", linewidths=0.6, zorder=4)
    ax.scatter(xy[is_both, 0], xy[is_both, 1], s=170, c=colors[is_both], marker="*",
               edgecolors="black", linewidths=1.3, zorder=5)

    # Leiden cluster id at each cluster centroid.
    for cl in range(n_clusters):
        m = labels == cl
        if m.sum() == 0:
            continue
        cx, cy = xy[m, 0].mean(), xy[m, 1].mean()
        ax.text(cx, cy, str(cl), fontsize=11, fontweight="bold", color="black",
                ha="center", va="center", zorder=7,
                bbox=dict(boxstyle="circle,pad=0.15", fc="white", ec="black", lw=0.6, alpha=0.8))

    label_rows = {g.upper(): i for i, g in enumerate(genes_u)}
    for tf in top_tfs:
        i = label_rows.get(tf.upper())
        if i is not None:
            ax.annotate(tf, (xy[i, 0], xy[i, 1]), fontsize=7, fontweight="bold",
                        color="#0b3d91", xytext=(3, 3), textcoords="offset points", zorder=8)

    shape_handles = [
        Line2D([], [], marker="o", color="w", markerfacecolor="#9ca3af", markersize=5,
               label=f"other genes ({is_other.sum()})"),
        Line2D([], [], marker="o", color="w", markerfacecolor="#9ca3af", markeredgecolor="#1f2937",
               markersize=8, label=f"TF ({is_tf.sum()})"),
        Line2D([], [], marker="*", color="w", markerfacecolor="#9ca3af", markeredgecolor="black",
               markersize=14, label=f"BTLA DEG ({is_deg.sum()})"),
        Line2D([], [], marker="*", color="w", markerfacecolor="#9ca3af", markeredgecolor="black",
               markeredgewidth=1.6, markersize=16, label=f"TF + DEG ({is_both.sum()})"),
    ]
    ax.legend(handles=shape_handles, loc="best", fontsize=8, framealpha=0.9,
              title=f"{n_clusters} Leiden clusters (colour)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a0", type=Path, default=JEPA_ROOT / "runs/exp2_A0/earlystop_step135000.ckpt")
    ap.add_argument("--a1", type=Path, default=JEPA_ROOT / "runs/exp2_A1/earlystop_step135000.ckpt")
    ap.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    ap.add_argument("--tf", type=Path, default=DEFAULT_TF)
    ap.add_argument("--deg", type=Path, default=DEFAULT_DEG)
    ap.add_argument("--out-dir", type=Path, default=JEPA_ROOT / "runs/exp2_tf_analysis")
    ap.add_argument("--n-labels", type=int, default=10, help="Top enriched TFs to label per panel.")
    ap.add_argument("--resolution", type=float, default=1.0, help="Leiden resolution.")
    args = ap.parse_args()
    sc.settings.verbosity = 0

    vocab = pd.read_csv(args.vocab, sep="\t").sort_values("idx").reset_index(drop=True)
    genes = vocab["gene"].astype(str).to_numpy()
    genes_u = np.array([g.upper() for g in genes])

    deg_df = pd.read_csv(args.deg)
    deg_set = {str(g).upper() for g in deg_df["gene"].dropna()} & set(genes_u)
    tf_set = {ln.strip().upper() for ln in args.tf.read_text().splitlines() if ln.strip()} & set(genes_u)
    print(f"DEGs in vocab: {len(deg_set)} | TFs in vocab: {len(tf_set)} | both: {len(deg_set & tf_set)}")

    def top_tfs(label: str) -> list[str]:
        f = args.out_dir / f"tf_deg_enrichment_{label}.csv"
        if not f.is_file():
            return []
        return pd.read_csv(f).head(args.n_labels)["TF"].astype(str).tolist()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(18, 8.5))
    for ax, ckpt, label, title in (
        (axes[0], args.a0, "A0", "A0: VICReg — gene-embedding UMAP (Leiden)"),
        (axes[1], args.a1, "A1", "A1: SIGReg — gene-embedding UMAP (Leiden)"),
    ):
        print(f"[{label}] computing Leiden + UMAP from {ckpt}")
        emb = load_gene_embedding(ckpt)
        xy, labels = compute_umap_leiden(emb, args.resolution)
        np.save(args.out_dir / f"gene_umap_{label}.npy", xy)
        cat = np.where(
            [g in deg_set for g in genes_u],
            np.where([g in tf_set for g in genes_u], "TF+DEG", "DEG"),
            np.where([g in tf_set for g in genes_u], "TF", "other"),
        )
        pd.DataFrame({"gene": genes, "leiden": labels, "category": cat,
                      "umap1": xy[:, 0], "umap2": xy[:, 1]}).to_csv(
            args.out_dir / f"gene_umap_leiden_{label}.csv", index=False)
        print(f"[{label}] {int(labels.max()) + 1} Leiden clusters")
        plot_panel(ax, xy, labels, genes_u, deg_set, tf_set, top_tfs(label), title)

    fig.suptitle(
        "Exp 2 gene embeddings — Leiden clusters (colour); BTLA-TCR-4h DEGs (stars), TFs (dots)",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = args.out_dir / "gene_umap_A0_A1.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
