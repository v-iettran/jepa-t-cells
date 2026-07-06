"""Perturbation retrieval (benchmarking.md Task 3) + relatedness ground truths.

A perturbation's *effect signature* is the mean shift it causes in embedding
space (mean perturbed - condition-matched mean control), kept as a latent-dim
vector. For each query knockdown we rank all others by signature cosine and
score how many of its pre-defined related partners land in the top-k
(recall@k + mean average precision).

Three relatedness ground truths are supported (the model must beat PCA/scVI on
all of them to count as "biologically organized"):

  * ``string``   : high-confidence STRING PPI edges (combined_score >= 700).
  * ``corum``    : two genes share a CORUM protein complex.
  * ``reactome`` : two genes share a Reactome pathway (size-capped so giant
                   generic pathways don't make everything trivially related).

CORUM/Reactome are co-membership sets; STRING is an explicit edge list. All are
reduced to the same ``symbol -> set[related symbols]`` form, restricted to the
candidate gene pool, so the scorer is identical across ground truths.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np

from jepa_poc.eval.perturbation import matched_control_means


# --------------------------------------------------------------------------- #
# Signatures
# --------------------------------------------------------------------------- #
def perturbation_signatures(
    z: np.ndarray,
    gene: np.ndarray,
    cond: np.ndarray,
    ctrl_means: dict,
    min_cells: int = 20,
) -> dict[str, np.ndarray]:
    """s_g = mean(perturbed z for gene g) - condition-matched mean(control z)."""

    sigs: dict[str, np.ndarray] = {}
    gene = gene.astype(str)
    for g in np.unique(gene):
        if g in {"control", "non-targeting", "unknown", "nan"}:
            continue
        m = gene == g
        if int(m.sum()) < min_cells:
            continue
        _, ctrl_z_mean = matched_control_means(ctrl_means, cond[m])
        sigs[g] = z[m].mean(axis=0) - ctrl_z_mean
    return sigs


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def retrieval_metrics(
    signatures: dict[str, np.ndarray],
    query_genes: list[str],
    related: dict[str, set[str]],
    ks: list[int],
) -> dict:
    """Rank candidates by signature cosine; score recovery of related genes."""

    names = list(signatures.keys())
    if not names:
        return {"summary": {"n_queries_scored": 0, "n_candidates": 0, "mAP": None}, "per_query": {}}
    mat = np.stack([signatures[n] for n in names], axis=0)
    mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
    idx = {n: i for i, n in enumerate(names)}

    per_query: dict[str, dict] = {}
    recall_acc = {k: [] for k in ks}
    ap_list: list[float] = []
    for q in query_genes:
        if q not in idx:
            continue
        rel = {r for r in related.get(q, set()) if r in idx and r != q}
        if not rel:
            continue
        sims = mat @ mat[idx[q]]
        order = [names[i] for i in np.argsort(-sims) if names[i] != q]
        hits = np.array([1.0 if n in rel else 0.0 for n in order])
        n_rel = len(rel)
        rec = {k: float(hits[:k].sum() / n_rel) for k in ks}
        if hits.sum() > 0:
            csum = np.cumsum(hits)
            ranks = np.arange(1, len(hits) + 1)
            ap = float((csum / ranks * hits).sum() / n_rel)
        else:
            ap = 0.0
        per_query[q] = {"n_related_in_pool": n_rel, "average_precision": ap,
                        **{f"recall_at_{k}": rec[k] for k in ks}}
        for k in ks:
            recall_acc[k].append(rec[k])
        ap_list.append(ap)

    summary = {
        "n_queries_scored": len(ap_list),
        "n_candidates": len(names),
        "mAP": float(np.mean(ap_list)) if ap_list else None,
        **{f"recall_at_{k}": (float(np.mean(recall_acc[k])) if recall_acc[k] else None) for k in ks},
    }
    return {"summary": summary, "per_query": per_query}


# --------------------------------------------------------------------------- #
# Ground truths
# --------------------------------------------------------------------------- #
def _open(path: Path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


def _relatedness_from_genesets(
    symbols: list[str],
    gene_sets: list[set[str]],
    max_set_size: int = 200,
) -> dict[str, set[str]]:
    """Two symbols are related if they co-occur in any (size-capped) gene set."""

    want = set(symbols)
    related: dict[str, set[str]] = {s: set() for s in symbols}
    for members in gene_sets:
        members = {m for m in members if m in want}
        if len(members) < 2 or len(members) > max_set_size:
            continue
        members = list(members)
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                related[a].add(b)
                related[b].add(a)
    return related


def build_relatedness_string(symbols: list[str], string_dir="data/string", score_threshold=700):
    from jepa_poc.eval.string_ppi import build_relatedness

    related, _ = build_relatedness(symbols, string_dir=string_dir, score_threshold=score_threshold)
    return related


def build_relatedness_corum(
    symbols: list[str],
    corum_path: str | Path = "data/corum/coreComplexes.txt",
    max_set_size: int = 200,
) -> dict[str, set[str]]:
    """CORUM core complexes. Co-membership in a complex => related."""

    corum_path = Path(corum_path)
    if not corum_path.exists():
        raise FileNotFoundError(f"CORUM file not found: {corum_path} (run scripts/download_genesets.py)")
    gene_sets: list[set[str]] = []
    with _open(corum_path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        # CORUM column is typically "subunits(Gene name)".
        col = None
        for cand in ("subunits(Gene name)", "subunits(Gene name syn)", "Subunits gene name"):
            if cand in header:
                col = header.index(cand)
                break
        if col is None:  # best-effort: find a column mentioning "Gene name"
            for i, h in enumerate(header):
                if "gene name" in h.lower():
                    col = i
                    break
        if col is None:
            raise ValueError(f"Could not find a gene-name column in CORUM header: {header}")
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= col:
                continue
            members = {g.strip() for g in parts[col].replace(",", ";").split(";") if g.strip()}
            if members:
                gene_sets.append(members)
    return _relatedness_from_genesets(symbols, gene_sets, max_set_size=max_set_size)


def build_relatedness_reactome(
    symbols: list[str],
    gmt_path: str | Path = "data/reactome/ReactomePathways.gmt",
    max_set_size: int = 200,
) -> dict[str, set[str]]:
    """Reactome pathways from a symbol-keyed GMT. Co-membership => related."""

    gmt_path = Path(gmt_path)
    if not gmt_path.exists():
        raise FileNotFoundError(f"Reactome GMT not found: {gmt_path} (run scripts/download_genesets.py)")
    gene_sets: list[set[str]] = []
    with _open(gmt_path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            members = {g.strip() for g in parts[2:] if g.strip()}
            if members:
                gene_sets.append(members)
    return _relatedness_from_genesets(symbols, gene_sets, max_set_size=max_set_size)


def build_relatedness(
    source: str,
    symbols: list[str],
    *,
    string_threshold: int = 700,
    max_set_size: int = 200,
) -> dict[str, set[str]]:
    """Dispatch to one of the supported ground-truth sources."""

    if source == "string":
        return build_relatedness_string(symbols, score_threshold=string_threshold)
    if source == "corum":
        return build_relatedness_corum(symbols, max_set_size=max_set_size)
    if source == "reactome":
        return build_relatedness_reactome(symbols, max_set_size=max_set_size)
    raise ValueError(f"unknown relatedness source: {source}")
