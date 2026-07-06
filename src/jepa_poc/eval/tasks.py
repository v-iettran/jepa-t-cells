"""Benchmark Tasks 1-4 (benchmarking.md), model-agnostic.

Consumes a standardized embedding cache (see scripts/build_embeddings.py) and
produces every decision-driving metric. Task 5 (cross-dataset transfer) is
deferred.

  Task 1  Decode perturbation effects (MLP decode(mean z)) -> delta-Pearson, p@k.
  Task 2  Linear (ridge) vs non-linear (MLP) decode + the gap.
  Task 3  Perturbation retrieval recall@k + mAP (STRING / CORUM / Reactome).
  Task 4  Effect-size stratification: bin perturbations by ||delta_true|| and
          re-report Tasks 1-3 within strong / medium / weak strata.

All deltas use condition-matched controls. Decoders are fit on control cells
only and reused across the held-out evaluation (Tasks 1-2) and the per-stratum
evaluation (Task 4) so the numbers are directly comparable.
"""

from __future__ import annotations

import numpy as np
import torch

from jepa_poc.eval.decoders import mlp_decode, train_mlp_decoder
from jepa_poc.eval.metrics import delta_pearson, precision_at_k
from jepa_poc.eval.perturbation import (
    condition_group_means,
    decode_latent,
    delta_metrics,
    fit_linear_decoder,
    matched_control_means,
)
from jepa_poc.eval.retrieval import (
    build_relatedness,
    perturbation_signatures,
    retrieval_metrics,
)

_SKIP_GENES = {"control", "non-targeting", "unknown", "nan"}


def _decode_group_metrics(z, expr, cond, gene, decode_fn, ctrl_means, top_ks, genes):
    """decode(mean z) delta metrics, overall + per held-out gene."""

    def metrics_for(mask):
        if mask.sum() == 0:
            return None
        _, ctrl_z_mean = matched_control_means(ctrl_means, cond[mask])
        true_mean = expr[mask].mean(axis=0)
        pred_mean = decode_fn(z[mask].mean(axis=0)[None, :])[0]
        control_ref = decode_fn(ctrl_z_mean[None, :])[0]
        return delta_metrics(pred_mean, true_mean, control_ref, top_ks)

    out = {"overall": metrics_for(np.ones(len(cond), dtype=bool)), "per_gene": {}}
    for g in genes:
        m = gene.astype(str) == g
        r = metrics_for(m)
        if r is not None:
            r["n_cells"] = int(m.sum())
            out["per_gene"][g] = r
    return out


class BenchmarkEvaluator:
    """Runs Tasks 1-4 on one model's embedding cache."""

    def __init__(self, cache: dict, device: str = "cuda", *, mlp_epochs: int = 40,
                 mlp_hidden: int = 1024, seed: int = 13):
        self.device = torch.device("cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu")
        self.seed = seed
        self.model = str(cache["model"]) if "model" in cache else "?"
        self.latent_dim = int(cache["latent_dim"]) if "latent_dim" in cache else int(cache["control_z"].shape[1])

        self.control_z = cache["control_z"]
        self.control_expr = cache["control_expr"]
        self.control_cond = cache["control_cond"]
        self.train_z = cache["train_z"]
        self.train_expr = cache["train_expr"]
        self.train_cond = cache["train_cond"]
        self.train_gene = cache["train_gene"].astype(str)
        self.test_z = cache["test_z"]
        self.test_expr = cache["test_expr"]
        self.test_cond = cache["test_cond"]
        self.test_gene = cache["test_gene"].astype(str)
        self.test_genes = sorted(set(self.test_gene.tolist()))

        self.ctrl_means = condition_group_means(self.control_expr, self.control_z, self.control_cond)

        # Decoders fit on controls only (reused by Tasks 1, 2, 4).
        self.lin = fit_linear_decoder(self.control_z, self.control_expr)
        self.mlp, self.mlp_val_mse = train_mlp_decoder(
            self.control_z, self.control_expr, self.device,
            hidden=mlp_hidden, epochs=mlp_epochs, seed=seed,
        )

    def _lin_fn(self, zz):
        return decode_latent(zz, self.lin)

    def _mlp_fn(self, zz):
        return mlp_decode(self.mlp, zz, self.device)

    # ----------------- Tasks 1 + 2 (held-out genes) ----------------- #
    def tasks_1_2(self, top_ks) -> dict:
        linear = _decode_group_metrics(self.test_z, self.test_expr, self.test_cond, self.test_gene,
                                       self._lin_fn, self.ctrl_means, top_ks, self.test_genes)
        mlp = _decode_group_metrics(self.test_z, self.test_expr, self.test_cond, self.test_gene,
                                    self._mlp_fn, self.ctrl_means, top_ks, self.test_genes)
        lin_p = linear["overall"]["delta_pearson"]
        mlp_p = mlp["overall"]["delta_pearson"]
        return {
            "linear_ridge": linear,
            "mlp_decode_meanz": mlp,
            "mlp_val_mse": float(self.mlp_val_mse),
            # Task 2 deliverable: non-linearity advantage (MLP - ridge). The doc
            # writes the gap as ridge - MLP; we store both so the sign is explicit.
            "gap_mlp_minus_ridge": float(mlp_p - lin_p),
            "gap_ridge_minus_mlp": float(lin_p - mlp_p),
        }

    # ----------------- Signatures (shared by Task 3 + 4) ----------------- #
    def signatures(self, min_cells: int = 20) -> dict[str, np.ndarray]:
        sigs = perturbation_signatures(self.train_z, self.train_gene, self.train_cond, self.ctrl_means, min_cells)
        sigs.update(perturbation_signatures(self.test_z, self.test_gene, self.test_cond, self.ctrl_means, min_cells))
        return sigs

    # ----------------- Task 3 (retrieval) ----------------- #
    def task_3(self, sources: list[str], ks: list[int], *, min_cells: int = 20,
               string_threshold: int = 700, max_set_size: int = 200) -> dict:
        sigs = self.signatures(min_cells=min_cells)
        all_genes = sorted(sigs.keys())
        out: dict[str, dict] = {"n_signatures": len(all_genes)}
        for src in sources:
            try:
                related = build_relatedness(src, all_genes, string_threshold=string_threshold,
                                            max_set_size=max_set_size)
            except FileNotFoundError as e:
                out[src] = {"error": str(e)}
                continue
            all_q = [g for g in all_genes if related.get(g)]
            held_q = [g for g in self.test_genes if related.get(g)]
            out[src] = {
                "all_queries": retrieval_metrics(sigs, all_q, related, ks)["summary"],
                "held_out_queries": retrieval_metrics(sigs, held_q, related, ks)["summary"],
            }
        return out

    # ----------------- Task 4 (effect-size stratification) ----------------- #
    def _per_perturbation_decode(self, top_ks, min_cells: int) -> dict[str, dict]:
        """Per-perturbation true-effect magnitude + linear/MLP decode metrics.

        Covers all SEEN train perturbations (>= min_cells) plus the held-out genes.
        """

        rows: dict[str, dict] = {}

        def add(z, expr, cond, gene_arr, seen: bool):
            for g in np.unique(gene_arr):
                if g in _SKIP_GENES:
                    continue
                m = gene_arr == g
                if int(m.sum()) < min_cells:
                    continue
                ctrl_expr_mean, ctrl_z_mean = matched_control_means(self.ctrl_means, cond[m])
                true_delta = expr[m].mean(axis=0) - ctrl_expr_mean
                eff = float(np.linalg.norm(true_delta))
                n_de = int((np.abs(true_delta) > 0.1).sum())
                row = {"effect_size": eff, "n_de_0.1": n_de, "n_cells": int(m.sum()), "seen": seen}
                for tag, fn in (("linear", self._lin_fn), ("mlp", self._mlp_fn)):
                    pred = fn(z[m].mean(axis=0)[None, :])[0]
                    ctrl_ref = fn(ctrl_z_mean[None, :])[0]
                    pred_delta = pred - ctrl_ref
                    row[f"{tag}_delta_pearson"] = delta_pearson(pred_delta, true_delta)
                    for k in top_ks:
                        row[f"{tag}_precision_at_{k}"] = precision_at_k(pred_delta, true_delta, k)
                rows[g] = row

        add(self.train_z, self.train_expr, self.train_cond, self.train_gene, seen=True)
        add(self.test_z, self.test_expr, self.test_cond, self.test_gene, seen=False)
        return rows

    def task_4(self, top_ks, ks: list[int], sources: list[str], *, min_cells: int = 20,
               string_threshold: int = 700, max_set_size: int = 200) -> dict:
        rows = self._per_perturbation_decode(top_ks, min_cells)
        if not rows:
            return {"error": "no perturbations passed min_cells"}

        genes = list(rows.keys())
        eff = np.array([rows[g]["effect_size"] for g in genes])
        q1, q2 = np.quantile(eff, [1 / 3, 2 / 3])

        def stratum_of(e):
            return "weak" if e <= q1 else ("medium" if e <= q2 else "strong")

        strata_genes = {"weak": [], "medium": [], "strong": []}
        for g in genes:
            strata_genes[stratum_of(rows[g]["effect_size"])].append(g)

        # ---- decode (Tasks 1-2) averaged within each stratum ----
        metric_keys = (["linear_delta_pearson", "mlp_delta_pearson"]
                       + [f"linear_precision_at_{k}" for k in top_ks]
                       + [f"mlp_precision_at_{k}" for k in top_ks])
        decode_strata: dict[str, dict] = {}
        for s, gl in strata_genes.items():
            if not gl:
                decode_strata[s] = {"n_perturbations": 0}
                continue
            agg = {"n_perturbations": len(gl),
                   "effect_size_mean": float(np.mean([rows[g]["effect_size"] for g in gl]))}
            for mk in metric_keys:
                agg[mk] = float(np.nanmean([rows[g][mk] for g in gl]))
            agg["gap_mlp_minus_ridge"] = agg["mlp_delta_pearson"] - agg["linear_delta_pearson"]
            decode_strata[s] = agg

        # ---- retrieval (Task 3) within each stratum (queries limited to bin) ----
        sigs = self.signatures(min_cells=min_cells)
        retr_strata: dict[str, dict] = {}
        for src in sources:
            try:
                related = build_relatedness(src, sorted(sigs.keys()),
                                            string_threshold=string_threshold, max_set_size=max_set_size)
            except FileNotFoundError as e:
                retr_strata[src] = {"error": str(e)}
                continue
            per_src = {}
            for s, gl in strata_genes.items():
                q = [g for g in gl if g in sigs and related.get(g)]
                per_src[s] = retrieval_metrics(sigs, q, related, ks)["summary"]
            retr_strata[src] = per_src

        return {
            "effect_size_quantiles": {"q33": float(q1), "q66": float(q2)},
            "n_perturbations_total": len(genes),
            "decode_by_stratum": decode_strata,
            "retrieval_by_stratum": retr_strata,
            "per_perturbation": rows,
        }
