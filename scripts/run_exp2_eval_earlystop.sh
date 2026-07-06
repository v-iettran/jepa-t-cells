#!/usr/bin/env bash
# Experiment 2 evaluation pipeline for the EARLY-STOPPED encoders (step 135000).
#
# Scope for this round:
#   * Encoder selection: A0 (VICReg) vs A1 (SIGReg)
#   * Perturbation heads: H0 (co-expression), H2 (JEPA gene-token), H3 (GENIE3 GRN)
#   * H3+H2 (grn_jepa concat) is intentionally DEFERRED and not run here.
#
# Notes:
#   * Uses the resumable checkpoints under runs/exp2_A*/checkpoints/last.ckpt
#     (training can still resume from these later).
#   * eval_perturbation.py caches cell embeddings keyed by (checkpoint, subsample
#     caps, seed) at runs/exp2/_embed_cache/perturb_embeds_<key>.npz. So each
#     encoder is embedded at most once: A0 and A1 keep separate caches, and the
#     head runs (H0/H2/H3) reuse the SELECTED encoder's existing cache. No
#     --refresh-embeddings is needed; correctness is guaranteed by the cache key.
set -euo pipefail

# Force single-threaded native BLAS/OpenMP. On this 96-core host the OpenBLAS
# threadpool deadlocks (all threads parked in futex_wait) when sklearn's
# TruncatedSVD runs after the GPU embedding step has forked DataLoader workers
# in a process that loaded torch's OpenMP runtime. Capping to 8 was not enough;
# single-threaded BLAS has no threadpool to deadlock on. This is cheap here
# because all embeddings are cached (the only CPU compute left is a small SVD
# and metrics).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

export PYTHONPATH="${PYTHONPATH:-src}"
CONFIG="${1:-configs/exp2.yaml}"

CKPT_A0="runs/exp2_A0/checkpoints/last.ckpt"
CKPT_A1="runs/exp2_A1/checkpoints/last.ckpt"

log() { printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"; }

for f in "${CKPT_A0}" "${CKPT_A1}"; do
  [[ -s "${f}" ]] || { echo "Missing checkpoint: ${f}" >&2; exit 1; }
done

# ----------------------------- Stage 1: encoder selection -----------------------------
log "Stage 1: A0 (VICReg) representation quality"
python scripts/eval_perturbation.py \
  --config "${CONFIG}" \
  --checkpoint "${CKPT_A0}" \
  --run-dir runs/exp2_eval_A0 \
  --feature-mode coexpression

log "Stage 1: A1 (SIGReg) representation quality"
python scripts/eval_perturbation.py \
  --config "${CONFIG}" \
  --checkpoint "${CKPT_A1}" \
  --run-dir runs/exp2_eval_A1 \
  --feature-mode coexpression

BEST_CKPT="$(python - <<'PY'
import json
from pathlib import Path

def score(path):
    return float(json.loads(Path(path).read_text())["representation_quality"]["overall"]["delta_pearson"])

a0 = score("runs/exp2_eval_A0/perturbation_results_coexpression.json")
a1 = score("runs/exp2_eval_A1/perturbation_results_coexpression.json")
print("runs/exp2_A1/checkpoints/last.ckpt" if a1 > a0 else "runs/exp2_A0/checkpoints/last.ckpt")
PY
)"
log "Selected best encoder checkpoint: ${BEST_CKPT}"

# ----------------------------- Stage 2: perturbation heads -----------------------------
# All head runs reuse the SELECTED encoder's embedding cache (already computed
# during Stage 1 selection), so no re-embedding happens here.
log "H0: co-expression head on selected encoder"
python scripts/eval_perturbation.py \
  --config "${CONFIG}" \
  --checkpoint "${BEST_CKPT}" \
  --run-dir runs/exp2_head_H0 \
  --feature-mode coexpression

log "H2: JEPA gene-token head on selected encoder"
python scripts/eval_perturbation.py \
  --config "${CONFIG}" \
  --checkpoint "${BEST_CKPT}" \
  --run-dir runs/exp2_head_H2 \
  --feature-mode jepa

log "H3: GENIE3 state-matched GRN head on selected encoder"
python scripts/eval_perturbation.py \
  --config "${CONFIG}" \
  --checkpoint "${BEST_CKPT}" \
  --run-dir runs/exp2_head_H3 \
  --feature-mode grn

# ----------------------------- Stage 3: report -----------------------------
# H3+H2 (grn_jepa) is deferred; its row will render as "pending" in the report.
log "Writing Experiment 2 report"
python scripts/write_exp2_report.py --output EXPERIMENT_2_REPORT.md

log "Experiment 2 early-stop evaluation pipeline complete"
