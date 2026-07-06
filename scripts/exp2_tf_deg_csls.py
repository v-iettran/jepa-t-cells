"""Experiment 2: TF -> BTLA-TCR-DEG association scored with CSLS (instead of odds-ratio).

Motivation
----------
The Stage-1 pipeline ranked TFs by an *odds ratio*: fraction of a TF's top-cosine
neighbours that are BTLA DEGs, over the background DEG rate. Plain cosine
neighbourhoods suffer from hubness -- some genes are spuriously close to many
others -- which inflates/deflates these counts.

CSLS (Cross-domain Similarity Local Scaling; Conneau et al. 2018,
"Word Translation Without Parallel Data") corrects for this:

    CSLS(a, b) = 2 * cos(a, b) - r_K(a) - r_K(b)
    r_K(x)     = mean cosine of x to its K nearest neighbours (self excluded)

Here we replace the odds-ratio with a CSLS-based effect size: for each TF we
measure its mean hubness-corrected proximity to the BTLA_TCR_4h DEG set and
compare it to its mean proximity to the whole HVG background.

Per TF we report:
    csls_deg_mean : mean CSLS from the TF to the DEG set (self excluded)
    csls_bg_mean  : mean CSLS from the TF to all other HVGs (background)
    csls_delta    : csls_deg_mean - csls_bg_mean  (the effect size; replaces odds_ratio)
    csls_z, csls_p, csls_fdr : analytic one-sided significance that the TF sits
        closer to DEGs than to a random gene set of equal size (finite-population
        corrected sampling test), BH-FDR across TFs.

Rows are sorted by csls_delta descending (most DEG-proximal TF first).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp2_tf_deg_enrichment import (  # noqa: E402
    DEFAULT_DEG,
    DEFAULT_TF,
    DEFAULT_VOCAB,
    JEPA_ROOT,
    benjamini_hochberg,
    load_gene_embedding,
)


def csls_matrix(emb: np.ndarray, k: int) -> np.ndarray:
    """Full CSLS matrix: 2*cos(a,b) - rK(a) - rK(b)."""
    norm_emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    cos = norm_emb @ norm_emb.T
    cos_no_self = cos.copy()
    np.fill_diagonal(cos_no_self, -np.inf)
    # mean cosine to the K nearest neighbours (per row), self excluded
    part = np.partition(cos_no_self, -k, axis=1)[:, -k:]
    r_k = part.mean(axis=1)
    return 2.0 * cos - r_k[:, None] - r_k[None, :]


def run_csls(
    emb: np.ndarray,
    genes: np.ndarray,
    tf_path: Path,
    deg_path: Path,
    k: int,
) -> pd.DataFrame:
    genes_u = np.array([g.upper() for g in genes])
    n_genes = len(genes)

    csls = csls_matrix(emb, k)

    tf_all = {ln.strip().upper() for ln in tf_path.read_text().splitlines() if ln.strip()}
    upper_to_row = {g: i for i, g in enumerate(genes_u)}
    tf_rows = {g: upper_to_row[g] for g in tf_all if g in upper_to_row}

    deg_df = pd.read_csv(deg_path)
    deg_all = {str(g).upper() for g in deg_df["gene"].dropna()}
    deg_idx = np.array([i for i, g in enumerate(genes_u) if g in deg_all])
    deg_idx_set = set(deg_idx.tolist())

    n_pop = n_genes - 1  # non-self universe
    rows = []
    for tf, r in tf_rows.items():
        deg_used = deg_idx[deg_idx != r]  # exclude self if the TF is itself a DEG
        m = deg_used.size
        if m == 0:
            continue
        row_vals = csls[r]
        deg_mean = float(row_vals[deg_used].mean())

        bg = np.delete(row_vals, r)  # all non-self genes
        bg_mean = float(bg.mean())
        bg_std = float(bg.std(ddof=1))
        delta = deg_mean - bg_mean

        # finite-population sampling test: is the DEG-set mean higher than a random
        # size-m subset of the background?
        fpc = np.sqrt((n_pop - m) / (n_pop - 1)) if n_pop > 1 else 1.0
        se = (bg_std / np.sqrt(m)) * fpc if m > 0 and bg_std > 0 else np.nan
        z = delta / se if se and not np.isnan(se) and se > 0 else 0.0
        p = float(norm.sf(z))

        rows.append(
            {
                "TF": genes[r],
                "is_deg": r in deg_idx_set,
                "n_deg_used": int(m),
                "csls_deg_mean": deg_mean,
                "csls_bg_mean": bg_mean,
                "csls_delta": delta,
                "csls_z": float(z),
                "csls_p": p,
            }
        )

    df = pd.DataFrame(rows)
    df["csls_fdr"] = benjamini_hochberg(df["csls_p"].to_numpy())
    df = df.sort_values(["csls_delta", "csls_p"], ascending=[False, True]).reset_index(drop=True)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--label", type=str, required=True, help="Model label, e.g. A0 or A1.")
    ap.add_argument("--encoder", choices=["target", "context"], default="target")
    ap.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    ap.add_argument("--tf", type=Path, default=DEFAULT_TF)
    ap.add_argument("--deg", type=Path, default=DEFAULT_DEG)
    ap.add_argument("--k", type=int, default=10, help="K nearest neighbours for CSLS local scaling.")
    ap.add_argument("--out-dir", type=Path, default=JEPA_ROOT / "runs/exp2_tf_analysis")
    args = ap.parse_args()

    vocab = pd.read_csv(args.vocab, sep="\t").sort_values("idx").reset_index(drop=True)
    genes = vocab["gene"].astype(str).to_numpy()

    emb = load_gene_embedding(args.ckpt, args.encoder)
    if emb.shape[0] != len(genes):
        raise AssertionError(f"embedding rows {emb.shape[0]} != vocab size {len(genes)}")
    print(f"[{args.label}] gene embedding {emb.shape} from {args.encoder} encoder | CSLS K={args.k}")

    df = run_csls(emb, genes, args.tf, args.deg, args.k)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"tf_deg_csls_{args.label}.csv"
    df.to_csv(out, index=False)
    n_sig = int((df["csls_fdr"] < 0.05).sum())
    print(f"[{args.label}] TFs scored = {len(df)} | csls_delta>0 = {int((df['csls_delta'] > 0).sum())} "
          f"| FDR<0.05 = {n_sig}")
    print(f"[{args.label}] wrote {out}")
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
