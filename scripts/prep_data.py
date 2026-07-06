"""Prepare CD4+ Perturb-seq data for JEPA POC training and evaluation.

Streams the CZI multi-shard release one file at a time (backed-mode), injects
``culture_condition`` / ``donor_id`` / ``10xrun_id`` from the filename and the
supplementary sample-metadata table, applies low-quality and KD-efficiency
filters, caps per-shard targeting cells to stay within host RAM, computes HVGs
on non-targeting controls only, and writes processed AnnData files for
pretraining, annotation, and perturbation evaluation.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from jepa_poc.config import ensure_dir, load_config
from jepa_poc.data.hvg import select_hvgs
from jepa_poc.data.synthetic import make_synthetic_anndata
from jepa_poc.data.vocab import GeneVocab


CONDITION_TOKENS = {"Rest", "Stim8hr", "Stim48hr"}


def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _subsample(adata: ad.AnnData, n: int, seed: int) -> ad.AnnData:
    if n is None or adata.n_obs <= n:
        return adata.copy()
    rng = np.random.default_rng(seed)
    idx = rng.choice(adata.n_obs, size=n, replace=False)
    return adata[idx].copy()


def _standardize_var_names(adata: ad.AnnData) -> ad.AnnData:
    adata = adata.copy()
    if "gene_symbol" in adata.var:
        adata.var_names = adata.var["gene_symbol"].astype(str)
    elif "gene_name" in adata.var:
        adata.var_names = adata.var["gene_name"].astype(str)
    adata.var_names_make_unique()
    return adata


def _assign_donor_split(
    adata: ad.AnnData,
    donor_key: str,
    seed: int,
    split_path: str | Path | None = None,
) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    adata = adata.copy()
    if donor_key not in adata.obs:
        adata.obs["split"] = rng.choice(["train", "val", "test"], size=adata.n_obs, p=[0.8, 0.1, 0.1])
        return adata
    donors = np.array(sorted(adata.obs[donor_key].astype(str).unique()))

    if split_path is not None and Path(split_path).exists():
        split_cfg = json.loads(Path(split_path).read_text())
        donor_to_split: dict[str, str] = {}
        for split_name in ["train", "val", "test"]:
            for donor in split_cfg.get(split_name, []):
                donor_to_split[str(donor)] = split_name
        missing = sorted(set(donors) - set(donor_to_split))
        if missing:
            raise RuntimeError(f"Donor split file {split_path} is missing donor(s): {missing}")
        adata.obs["split"] = [donor_to_split[str(d)] for d in adata.obs[donor_key].astype(str)]
        return adata

    rng.shuffle(donors)
    if len(donors) == 1:
        adata.obs["split"] = rng.choice(["train", "val", "test"], size=adata.n_obs, p=[0.8, 0.1, 0.1])
        return adata
    n_train = max(1, int(round(0.7 * len(donors))))
    n_val = max(1, int(round(0.15 * len(donors))))
    n_train = min(n_train, len(donors) - 2)
    train = set(donors[:n_train])
    val = set(donors[n_train : n_train + n_val])
    split = ["train" if d in train else "val" if d in val else "test" for d in adata.obs[donor_key].astype(str)]
    adata.obs["split"] = split
    if split_path is not None:
        split_summary = {
            "train": sorted(train),
            "val": sorted(val),
            "test": sorted(set(donors) - train - val),
        }
        path = Path(split_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(split_summary, indent=2))
    return adata


def _collapse_labels(adata: ad.AnnData, label_key: str, groups: dict[str, list[str]]) -> ad.AnnData:
    adata = adata.copy()
    if label_key not in adata.obs:
        adata.obs["annotation_group"] = "Other"
        return adata
    labels = adata.obs[label_key].astype(str)
    out = []
    for label in labels:
        assigned = "Other"
        for group, patterns in groups.items():
            if any(pattern.lower() in label.lower() for pattern in patterns):
                assigned = group
                break
        out.append(assigned)
    adata.obs["annotation_group"] = out
    return adata


def _parse_filename(stem: str) -> tuple[str, str]:
    """Parse ``D1_Rest.assigned_guide`` into ``("D1", "Rest")``."""

    base = stem.split(".")[0]
    parts = base.split("_")
    donor = parts[0] if parts and parts[0].startswith("D") else "Dunknown"
    cond = next((p for p in parts[1:] if p in CONDITION_TOKENS), "Unknown")
    return donor, cond


def _build_sample_lookup(sample_metadata: pd.DataFrame | None) -> dict[tuple[str, str], tuple[str, str]]:
    """Map (donor_tag, condition_tag) -> (10xrun_id, donor_id) from sample metadata."""

    if sample_metadata is None:
        return {}
    lookup: dict[tuple[str, str], tuple[str, str]] = {}
    for _, row in sample_metadata.iterrows():
        sid = str(row.get("cell_sample_id", ""))
        cond = next((p for p in sid.split("_") if p in CONDITION_TOKENS), None)
        donor = next((p for p in sid.split("_") if p.startswith("D") and p[1:].isdigit()), None)
        if donor is None or cond is None:
            continue
        lookup[(donor, cond)] = (str(row["10xrun_id"]), str(row["donor_id"]))
    return lookup


def _normalize_kd_eff(kd_eff: pd.DataFrame | None) -> pd.DataFrame | None:
    """Normalize the CSV's first unnamed column to ``index`` per the data-sharing README."""

    if kd_eff is None:
        return None
    cols = list(kd_eff.columns)
    if "index" not in cols and cols and (cols[0].startswith("Unnamed") or cols[0] == ""):
        kd_eff = kd_eff.rename(columns={cols[0]: "index"})
    return kd_eff


def _apply_quality_filter(adata: ad.AnnData, low_quality_key: str) -> ad.AnnData:
    if low_quality_key not in adata.obs:
        return adata
    keep = ~adata.obs[low_quality_key].fillna(False).astype(bool)
    return adata[keep].copy()


def _curate_targets(adata: ad.AnnData, sgrna_lib: pd.DataFrame | None, perturbation_key: str) -> ad.AnnData:
    if sgrna_lib is None or "guide_id" not in adata.obs:
        return adata
    if "sgRNA" not in sgrna_lib or "target_gene_name" not in sgrna_lib:
        return adata
    guide_to_target = dict(zip(sgrna_lib["sgRNA"].astype(str), sgrna_lib["target_gene_name"].astype(str)))
    adata = adata.copy()
    curated = adata.obs["guide_id"].astype(str).map(guide_to_target)
    if perturbation_key in adata.obs:
        curated = curated.fillna(adata.obs[perturbation_key].astype(str))
    adata.obs[perturbation_key] = curated.astype(str)
    return adata


def _apply_effective_guides(
    adata: ad.AnnData,
    kd_eff: pd.DataFrame | None,
    guide_type_key: str,
    non_targeting_label: str,
) -> ad.AnnData:
    if kd_eff is None or "guide_id" not in adata.obs:
        return adata
    if "index" not in kd_eff or "signif_knockdown" not in kd_eff:
        return adata
    eff_set = set(kd_eff.loc[kd_eff["signif_knockdown"].fillna(False).astype(bool), "index"].astype(str))
    is_control = adata.obs.get(guide_type_key, pd.Series([""] * adata.n_obs, index=adata.obs.index)).astype(str) == non_targeting_label
    is_effective = adata.obs["guide_id"].astype(str).isin(eff_set)
    keep = (is_control | is_effective).to_numpy()
    return adata[keep].copy()


def _normalize_control_label(
    adata: ad.AnnData,
    guide_type_key: str,
    non_targeting_label: str,
    perturbation_key: str,
    control_label: str,
) -> ad.AnnData:
    if guide_type_key not in adata.obs or perturbation_key not in adata.obs:
        return adata
    adata = adata.copy()
    is_control = adata.obs[guide_type_key].astype(str) == non_targeting_label
    perturbation = adata.obs[perturbation_key].astype(str).copy()
    perturbation[is_control.to_numpy()] = control_label
    adata.obs[perturbation_key] = perturbation
    return adata


def _resident_gb() -> float:
    try:
        with open(f"/proc/{__import__('os').getpid()}/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / (1024 * 1024)
    except OSError:
        pass
    return float("nan")


def _configured_path(cfg, key: str, default_name: str) -> Path:
    """Return an explicit data path when configured, else output_dir/default."""

    configured = getattr(cfg.data, key, None)
    if configured:
        return Path(configured)
    return Path(cfg.data.output_dir) / default_name


def _stream_load_filtered(
    paths: list[Path],
    cfg,
    sgrna_lib: pd.DataFrame | None,
    kd_eff: pd.DataFrame | None,
    sample_lookup: dict[tuple[str, str], tuple[str, str]],
    per_shard_targeting_cap: int,
    spill_dir: Path,
) -> ad.AnnData:
    """Open each shard backed, filter on ``.obs`` only, write a slim per-shard
    h5ad to ``spill_dir``, then concat all slims at the end.

    Filter order: low-quality drop, optional effective-guides filter (keeps all
    non-targeting controls). All NTC cells are kept; targeting cells are
    randomly capped per shard so each slim stays bounded. Spilling to disk
    avoids holding all 12 slim shards in RAM at once (~150 GB at 20% density).
    """

    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing perturb-seq shard(s):\n  - " + "\n  - ".join(missing)
        )

    spill_dir.mkdir(parents=True, exist_ok=True)

    # Resume-from-spill: if every expected slim file already exists, skip the
    # expensive streaming pass and reuse the cached slims on disk. This check
    # MUST run BEFORE any cleanup of the spill directory.
    expected = [spill_dir / f"slim_{i:02d}_{Path(p).stem}.h5ad" for i, p in enumerate(paths, start=1)]
    if all(p.exists() and p.stat().st_size > 0 for p in expected):
        _log(f"Reusing {len(expected)} cached slim shards from {spill_dir} (skipping streaming pass)")
        return expected, [Path(p).stem for p in paths], None

    # Only clean stale slim files when we're going to re-stream from scratch.
    for stale in spill_dir.glob("slim_*.h5ad"):
        stale.unlink()

    eff_set: set[str] | None = None
    if (
        bool(cfg.data.effective_guides_only)
        and kd_eff is not None
        and "index" in kd_eff.columns
        and "signif_knockdown" in kd_eff.columns
    ):
        eff_set = set(
            kd_eff.loc[
                kd_eff["signif_knockdown"].fillna(False).astype(bool), "index"
            ].astype(str)
        )
        _log(f"Effective-guides filter active: {len(eff_set)} guides pass signif_knockdown")
    else:
        _log("Effective-guides filter inactive (flag off, file missing, or columns missing)")

    rng = np.random.default_rng(int(cfg.data.split_seed))
    slim_paths: list[Path] = []
    keys: list[str] = []
    n_seen_total = 0
    n_kept_total = 0
    n_ntc_total = 0
    n_tgt_total = 0

    for i, path in enumerate(paths, start=1):
        donor_tag, cond_tag = _parse_filename(Path(path).stem)
        t0 = time.time()
        a = ad.read_h5ad(str(path), backed="r")
        n_seen = int(a.n_obs)
        obs = a.obs.copy()

        obs["culture_condition"] = cond_tag
        runid, real_donor = sample_lookup.get((donor_tag, cond_tag), ("unknown", donor_tag))
        obs["10xrun_id"] = runid
        obs["donor_id"] = real_donor

        keep = pd.Series(True, index=obs.index)
        if cfg.data.low_quality_key in obs:
            keep &= ~obs[cfg.data.low_quality_key].fillna(False).astype(bool)

        gt_key = cfg.data.guide_type_key
        ntc_label = cfg.data.non_targeting_guide_type
        if gt_key in obs:
            guide_types = obs[gt_key].astype(str)
        else:
            guide_types = pd.Series("", index=obs.index)
        is_ntc = guide_types == ntc_label

        if eff_set is not None and "guide_id" in obs:
            is_effective = obs["guide_id"].astype(str).isin(eff_set)
            keep &= (is_ntc | is_effective)

        kept_mask = keep.to_numpy()
        is_ntc_arr = is_ntc.to_numpy()
        kept_idx = np.where(kept_mask)[0]
        ntc_kept = kept_idx[is_ntc_arr[kept_idx]]
        tgt_kept = kept_idx[~is_ntc_arr[kept_idx]]

        if per_shard_targeting_cap and len(tgt_kept) > per_shard_targeting_cap:
            tgt_kept = np.sort(rng.choice(tgt_kept, size=per_shard_targeting_cap, replace=False))

        final_idx = np.sort(np.concatenate([ntc_kept, tgt_kept]))
        if final_idx.size == 0:
            _log(f"  [{i}/{len(paths)}] {Path(path).name}: 0 cells kept, skipping")
            a.file.close()
            continue

        X_sub = a.X[final_idx]
        if not sparse.issparse(X_sub):
            X_sub = sparse.csr_matrix(np.asarray(X_sub))
        else:
            X_sub = X_sub.tocsr()
        if X_sub.dtype != np.float32:
            X_sub = X_sub.astype(np.float32)
        sub_obs = obs.iloc[final_idx].copy()
        sub_var = a.var.copy()
        sub = ad.AnnData(X=X_sub, obs=sub_obs, var=sub_var)

        a.file.close()
        del a, X_sub, obs, sub_obs, sub_var, kept_mask, is_ntc_arr, kept_idx
        gc.collect()

        slim_path = spill_dir / f"slim_{i:02d}_{Path(path).stem}.h5ad"
        sub.write_h5ad(slim_path, compression="gzip", compression_opts=4)
        nnz = int(sub.X.nnz) if sparse.issparse(sub.X) else int(np.prod(sub.X.shape))
        density = nnz / float(sub.shape[0] * sub.shape[1]) if sub.shape[0] * sub.shape[1] else 0.0
        slim_paths.append(slim_path)
        keys.append(Path(path).stem)
        n_kept_total += int(final_idx.size)
        n_seen_total += n_seen
        n_ntc_total += int(ntc_kept.size)
        n_tgt_total += int(tgt_kept.size)
        del sub
        gc.collect()
        dt = time.time() - t0
        _log(
            f"  [{i}/{len(paths)}] {Path(path).name}: "
            f"kept {final_idx.size:,}/{n_seen:,} "
            f"({ntc_kept.size:,} NTC + {tgt_kept.size:,} targeting) "
            f"density={density:.3f} | RSS={_resident_gb():.1f}GB | {dt:.1f}s -> {slim_path.name}"
        )

    summary = (
        f"Total kept across {len(slim_paths)} shards: {n_kept_total:,}/{n_seen_total:,} cells "
        f"({n_ntc_total:,} NTC + {n_tgt_total:,} targeting)"
    )
    _log(summary)
    return slim_paths, keys, summary


def _hvg_concat(
    slim_paths: list[Path],
    keys: list[str],
    *,
    n_top_genes: int,
    batch_key: str,
    guide_type_key: str,
    non_targeting_label: str,
) -> tuple[ad.AnnData, list[str]]:
    """Concat 12 slim shards in two passes to stay under 100 GB peak RAM.

    Pass 1: load each slim, keep NTC rows only, concat -> compute HVGs on the
    NTC-only matrix (~900 K x 18 K sparse, ~30 GB).
    Pass 2: load each slim again, subset to HVG columns, concat -> the final
    combined matrix is ~4 M x 2 K sparse (~30 GB peak vs >200 GB without the
    HVG subset).
    """

    _log("HVG pass 1/2: building NTC-only concat for HVG selection")
    t0 = time.time()
    ntc_pieces: list[ad.AnnData] = []
    var_ref: pd.DataFrame | None = None
    for i, p in enumerate(slim_paths, start=1):
        s = ad.read_h5ad(p)
        s = _standardize_var_names(s)
        if var_ref is None:
            var_ref = s.var.copy()
        if guide_type_key in s.obs:
            mask = s.obs[guide_type_key].astype(str).to_numpy() == non_targeting_label
            ntc_pieces.append(s[mask].copy())
        del s
        gc.collect()
        _log(f"  pass1 [{i}/{len(slim_paths)}] {p.name}: cumulative NTC pieces RSS={_resident_gb():.1f}GB")
    ntc_concat = ad.concat(ntc_pieces, join="outer", label="shard_id", keys=keys)
    del ntc_pieces
    gc.collect()
    _log(f"NTC concat shape={ntc_concat.shape}, took {time.time() - t0:.1f}s | RSS={_resident_gb():.1f}GB")

    _log(f"Selecting top {n_top_genes} HVGs (seurat_v3 on raw counts)")
    t0 = time.time()
    hvg_genes = select_hvgs(ntc_concat, n_top_genes=n_top_genes, batch_key=batch_key)
    _log(f"HVG selection done in {time.time() - t0:.1f}s ({len(hvg_genes)} genes)")
    del ntc_concat
    gc.collect()

    _log("HVG pass 2/2: subsetting each slim to HVGs and concatenating")
    t0 = time.time()
    pieces: list[ad.AnnData] = []
    for i, (p, k) in enumerate(zip(slim_paths, keys), start=1):
        s = ad.read_h5ad(p)
        s = _standardize_var_names(s)
        common = [g for g in hvg_genes if g in s.var_names]
        s = s[:, common].copy()
        pieces.append(s)
        _log(f"  pass2 [{i}/{len(slim_paths)}] {p.name}: shape={s.shape} RSS={_resident_gb():.1f}GB")
    combined = ad.concat(pieces, join="outer", label="shard_id", keys=keys)
    del pieces
    gc.collect()
    _log(f"HVG-subset concat shape={combined.shape}, took {time.time() - t0:.1f}s | RSS={_resident_gb():.1f}GB")
    return combined, hvg_genes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/poc.yaml")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic processed files for local smoke runs.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = ensure_dir(cfg.data.output_dir)
    ensure_dir(Path(cfg.data.gene_vocab_path).parent)
    _log(f"prep_data starting (synthetic={args.synthetic}) — output_dir={out_dir}")

    if args.synthetic:
        perturb = make_synthetic_anndata(n_cells=4096, n_genes=500, n_perturbations=8, seed=int(cfg.seed))
        sgrna_lib = None
        kd_eff = None
    else:
        shard_paths = [Path(p) for p in cfg.data.perturb_seq_shards]
        _log(f"Loading {len(shard_paths)} shards via streaming filter")

        sgrna_lib = pd.read_csv(cfg.data.sgrna_library_path) if Path(cfg.data.sgrna_library_path).exists() else None
        _log(f"sgrna_library: {'loaded' if sgrna_lib is not None else 'missing'}")

        kd_eff = pd.read_csv(cfg.data.guide_kd_path) if Path(cfg.data.guide_kd_path).exists() else None
        kd_eff = _normalize_kd_eff(kd_eff)
        _log(f"guide_kd_efficiency: {'loaded' if kd_eff is not None else 'missing'}")

        sample_meta = pd.read_csv(cfg.data.sample_metadata_path) if Path(cfg.data.sample_metadata_path).exists() else None
        sample_lookup = _build_sample_lookup(sample_meta)
        _log(f"sample_metadata: {'loaded' if sample_meta is not None else 'missing'} ({len(sample_lookup)} mappings)")

        per_shard_cap = max(1, int(cfg.data.perturb_eval_cells) // max(1, len(shard_paths)))
        spill_dir = Path(cfg.data.output_dir) / "_slim_shards"
        _log(f"per-shard targeting cap = {per_shard_cap:,} cells | spill_dir = {spill_dir}")

        slim_paths, keys, _ = _stream_load_filtered(
            shard_paths,
            cfg,
            sgrna_lib,
            kd_eff,
            sample_lookup,
            per_shard_targeting_cap=per_shard_cap,
            spill_dir=spill_dir,
        )

        perturb, hvg_genes = _hvg_concat(
            slim_paths,
            keys,
            n_top_genes=int(cfg.data.hvg_count),
            batch_key=cfg.data.batch_key,
            guide_type_key=cfg.data.guide_type_key,
            non_targeting_label=cfg.data.non_targeting_guide_type,
        )

    if args.synthetic:
        _log(f"Standardizing var names; current shape={perturb.shape}")
        perturb = _standardize_var_names(perturb)
        hvg_genes = None  # synthetic path will run select_hvgs below
    perturb = _apply_quality_filter(perturb, cfg.data.low_quality_key)
    perturb = _curate_targets(perturb, sgrna_lib, cfg.data.perturbation_key)
    if cfg.data.effective_guides_only:
        perturb = _apply_effective_guides(perturb, kd_eff, cfg.data.guide_type_key, cfg.data.non_targeting_guide_type)
    perturb = _normalize_control_label(
        perturb,
        cfg.data.guide_type_key,
        cfg.data.non_targeting_guide_type,
        cfg.data.perturbation_key,
        cfg.data.control_label,
    )
    _log(f"After post-filters: shape={perturb.shape} | RSS={_resident_gb():.1f}GB")

    perturbation_series = perturb.obs[cfg.data.perturbation_key].astype(str)
    controls = perturb[perturbation_series == cfg.data.control_label].copy()
    targeting = perturb[perturbation_series != cfg.data.control_label].copy()
    _log(f"Split: {controls.n_obs:,} controls, {targeting.n_obs:,} targeting")
    if controls.n_obs == 0:
        raise RuntimeError(f"No control cells found (perturbation_key='{cfg.data.perturbation_key}' == '{cfg.data.control_label}').")
    del perturb
    gc.collect()

    controls = _subsample(controls, int(cfg.data.perturb_control_cells), int(cfg.data.split_seed) + 1)
    targeting = _subsample(targeting, int(cfg.data.perturb_eval_cells), int(cfg.data.split_seed) + 2)
    _log(f"After subsample caps: controls={controls.n_obs:,}, targeting={targeting.n_obs:,}")

    if hvg_genes is None:
        _log("Selecting HVGs on controls (seurat_v3 on raw counts)...")
        t0 = time.time()
        hvg_genes = select_hvgs(controls, int(cfg.data.hvg_count), cfg.data.batch_key)
        _log(f"HVG selection done in {time.time() - t0:.1f}s ({len(hvg_genes)} genes)")
        common = [g for g in hvg_genes if g in controls.var_names and g in targeting.var_names]
        if not common:
            raise RuntimeError("No HVGs survived intersection between controls and targeting cells.")
        controls = controls[:, common].copy()
        targeting = targeting[:, common].copy()
    else:
        # HVGs already applied during _hvg_concat; controls/targeting share the
        # same HVG column space already.
        common = [g for g in hvg_genes if g in controls.var_names and g in targeting.var_names]
    vocab = GeneVocab(genes=common)
    vocab.to_tsv(cfg.data.gene_vocab_path)
    _log(f"Wrote gene vocab ({len(common)} genes) to {cfg.data.gene_vocab_path}")

    controls = _collapse_labels(controls, cfg.data.label_key, dict(cfg.data.condition_groups))
    targeting = _collapse_labels(targeting, cfg.data.label_key, dict(cfg.data.condition_groups))

    controls_ann = _assign_donor_split(
        controls,
        cfg.data.donor_key,
        int(cfg.data.split_seed),
        getattr(cfg.data, "donor_split_path", None),
    )

    heldout_path = getattr(cfg.data, "heldout_genes_path", None)
    if heldout_path and Path(heldout_path).exists():
        heldout_cfg = json.loads(Path(heldout_path).read_text())
        heldout = set(map(str, heldout_cfg.get("experiment_2_heldout_genes", heldout_cfg)))
    else:
        heldout = set(map(str, cfg.data.heldout_perturbation_genes))
    targeting.obs["split"] = [
        "test" if str(p) in heldout else "train" for p in targeting.obs[cfg.data.perturbation_key].astype(str)
    ]
    _log(f"Heldout perturbation genes: {sorted(heldout)}")

    pretrain_parts = [controls_ann[controls_ann.obs["split"] != "test"].copy()]
    targeting_train = targeting[targeting.obs["split"] == "train"].copy()
    if targeting_train.n_obs:
        pretrain_parts.append(targeting_train)
    pretrain = ad.concat(pretrain_parts, join="inner", label="source", keys=["control"] + (["targeting_train"] if targeting_train.n_obs else []))
    rng = np.random.default_rng(int(cfg.data.split_seed))
    pretrain.obs["split"] = rng.choice(["train", "val"], size=pretrain.n_obs, p=[0.9, 0.1])

    annotation_path = _configured_path(cfg, "annotation_processed_path", "annotation_processed.h5ad")
    perturb_controls_path = _configured_path(cfg, "perturb_controls_processed_path", "perturb_controls_processed.h5ad")
    perturb_eval_path = _configured_path(cfg, "perturb_eval_processed_path", "perturb_eval_processed.h5ad")
    pretrain_path = _configured_path(cfg, "pretrain_processed_path", "pretrain_processed.h5ad")
    for path in [annotation_path, perturb_controls_path, perturb_eval_path, pretrain_path]:
        ensure_dir(path.parent)

    _log(f"Writing processed h5ad files (pretrain={pretrain.n_obs:,}, annotation={controls_ann.n_obs:,}, "
         f"perturb_eval={targeting.n_obs:,}, perturb_controls={controls.n_obs:,})...")
    t0 = time.time()
    controls_ann.write_h5ad(annotation_path)
    controls.write_h5ad(perturb_controls_path)
    targeting.write_h5ad(perturb_eval_path)
    pretrain.write_h5ad(pretrain_path)
    _log(f"Wrote 4 h5ad files in {time.time() - t0:.1f}s")

    split_summary = {
        "pretrain": {name: int(count) for name, count in pretrain.obs["split"].value_counts().items()},
        "annotation": {name: int(count) for name, count in controls_ann.obs["split"].value_counts().items()},
        "perturbation_eval": {
            "train_perturbations": int(np.sum(targeting.obs["split"] == "train")),
            "test_perturbations": int(np.sum(targeting.obs["split"] == "test")),
            "test_genes": sorted(set(targeting.obs.loc[targeting.obs["split"] == "test", cfg.data.perturbation_key].astype(str))),
        },
        "n_hvgs": len(common),
    }
    split_summary_path = Path(getattr(cfg.data, "split_summary_path", out_dir / "split_summary.json"))
    ensure_dir(split_summary_path.parent)
    split_summary_path.write_text(json.dumps(split_summary, indent=2))
    _log(f"Wrote split_summary.json to {split_summary_path}")
    print(json.dumps(split_summary, indent=2), flush=True)
    _log("prep_data done.")


if __name__ == "__main__":
    main()
