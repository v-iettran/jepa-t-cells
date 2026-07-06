"""Experiment 2: TF -> BTLA-TCR-DEG target enrichment from JEPA gene embeddings.

Faithful re-implementation of the Experiment 1 "Route B" pipeline
(models_TFs_analysis/scripts/jepa_tf_deg_enrichment.py), generalized for:
  * the 4000-HVG Experiment 2 vocabulary, and
  * the early-stopped A0/A1 PyTorch-Lightning checkpoints (full state_dict),
    from which we read the EMA *target* encoder's gene-embedding table to match
    Experiment 1 (which used ema_target_encoder.pt).

Methodology (identical to Exp 1)
--------------------------------
1. Load the learned gene_embedding.weight (n_genes x d), row-aligned to the HVG
   vocabulary (gene_vocab_4000.tsv).
2. Cosine similarity between every gene pair (rows L2-normalised).
3. For each TF present in the HVGs, its "targets" are the genes whose cosine
   similarity exceeds that TF's own 95th-percentile similarity (self excluded).
4. JEPA_BTLA_TCR_4h_list = BTLAvsTCR_4h DEGs that fall inside the HVGs.
5. Per TF, count targets in that DEG list; compute normalized rate, odds ratio
   vs background, a one-sided Fisher exact test, and BH-FDR across TFs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import fisher_exact

VIET_ROOT = Path("/mnt/R0/Projects/POIAZ/Viet")
JEPA_ROOT = VIET_ROOT / "JEPA-for-t-cells"
DEFAULT_TF = VIET_ROOT / "models_TFs_analysis/outputs/tf_embedding_neighbor_outputs/TF_names_v_1.01.txt"
DEFAULT_DEG = VIET_ROOT / "BulkFormer/data/original_processed_data/BTLAvsTCR_4h_DEGs.csv"
DEFAULT_VOCAB = JEPA_ROOT / "configs/gene_vocab_4000.tsv"


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """Return BH-FDR adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n, dtype=float)
    out[order] = np.clip(ranked, 0.0, 1.0)
    return out


def load_gene_embedding(ckpt: Path, encoder: str) -> np.ndarray:
    """Extract gene_embedding.weight from a Lightning checkpoint or flat state dict."""
    obj = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    candidates = [
        f"model.{encoder}_encoder.gene_embedding.weight",
        f"{encoder}_encoder.gene_embedding.weight",
        "gene_embedding.weight",
    ]
    for key in candidates:
        if key in state:
            return state[key].float().numpy()
    raise KeyError(f"No gene_embedding weight found in {ckpt}. Tried {candidates}.")


def run_enrichment(
    emb: np.ndarray,
    genes: np.ndarray,
    tf_path: Path,
    deg_path: Path,
    percentile: float,
) -> tuple[pd.DataFrame, int, float]:
    genes_u = np.array([g.upper() for g in genes])
    n_genes = len(genes)

    norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    cos = norm @ norm.T

    tf_all = {ln.strip().upper() for ln in tf_path.read_text().splitlines() if ln.strip()}
    upper_to_row = {g: i for i, g in enumerate(genes_u)}
    tf_rows = {g: upper_to_row[g] for g in tf_all if g in upper_to_row}

    deg_df = pd.read_csv(deg_path)
    deg_all = {str(g).upper() for g in deg_df["gene"].dropna()}
    deg_mask = np.array([g in deg_all for g in genes_u])
    n_deg = int(deg_mask.sum())
    background_deg_rate = n_deg / n_genes

    rows = []
    for tf, r in tf_rows.items():
        sims = cos[r].copy()
        sims[r] = -np.inf
        thr = np.percentile(sims[np.isfinite(sims)], percentile)
        target_mask = sims > thr
        total = int(target_mask.sum())
        if total == 0:
            continue
        a = int((target_mask & deg_mask).sum())
        normalized = a / total
        odds = normalized / background_deg_rate if background_deg_rate > 0 else np.nan

        universe = n_genes - 1
        deg_in_universe = n_deg - (1 if deg_mask[r] else 0)
        b = total - a
        c = deg_in_universe - a
        d = universe - total - c
        _, p = fisher_exact([[a, b], [c, d]], alternative="greater")

        rows.append(
            {
                "TF": genes[r],
                "blta_deg_target_count": a,
                "total_tf_targets": total,
                "normalized_deg_rate": normalized,
                "background_deg_rate": background_deg_rate,
                "odds_ratio": odds,
                "fisher_p": p,
            }
        )

    df = pd.DataFrame(rows)
    df["fisher_fdr"] = benjamini_hochberg(df["fisher_p"].to_numpy())
    df = df.sort_values(["odds_ratio", "fisher_p"], ascending=[False, True]).reset_index(drop=True)
    return df, n_deg, background_deg_rate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--label", type=str, required=True, help="Model label, e.g. A0 or A1.")
    ap.add_argument("--encoder", choices=["target", "context"], default="target")
    ap.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    ap.add_argument("--tf", type=Path, default=DEFAULT_TF)
    ap.add_argument("--deg", type=Path, default=DEFAULT_DEG)
    ap.add_argument("--percentile", type=float, default=95.0)
    ap.add_argument("--out-dir", type=Path, default=JEPA_ROOT / "runs/exp2_tf_analysis")
    args = ap.parse_args()

    vocab = pd.read_csv(args.vocab, sep="\t").sort_values("idx").reset_index(drop=True)
    genes = vocab["gene"].astype(str).to_numpy()

    emb = load_gene_embedding(args.ckpt, args.encoder)
    if emb.shape[0] != len(genes):
        raise AssertionError(f"embedding rows {emb.shape[0]} != vocab size {len(genes)}")
    print(f"[{args.label}] gene embedding {emb.shape} from {args.encoder} encoder")

    df, n_deg, bg = run_enrichment(emb, genes, args.tf, args.deg, args.percentile)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"tf_deg_enrichment_{args.label}.csv"
    df.to_csv(out, index=False)
    n_sig = int((df["fisher_fdr"] < 0.05).sum())
    n_or = int((df["odds_ratio"] >= 1.5).sum())
    print(f"[{args.label}] DEGs in HVG = {n_deg} | background_rate = {bg:.4f}")
    print(f"[{args.label}] TFs tested = {len(df)} | odds_ratio>=1.5 = {n_or} | FDR<0.05 = {n_sig}")
    print(f"[{args.label}] wrote {out}")
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
