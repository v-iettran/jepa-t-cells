#!/bin/bash
set -uo pipefail

PROJ=/mnt/R0/Projects/POIAZ/Viet/JEPA-for-t-cells
ENV_PY=/home/qtran/miniconda3/envs/bulkformer/bin/python
CKPT=$PROJ/runs/poc/checkpoints/last.ckpt
CONFIG=$PROJ/configs/poc.yaml
LOG=$PROJ/runs/poc/eval_perturb_v2.log

cd "$PROJ"
export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=0

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "=== PERTURBATION EVAL v2 START ==="
"$ENV_PY" -u scripts/eval_perturbation.py --config "$CONFIG" --checkpoint "$CKPT" 2>&1 | tee -a "$LOG"
PE=${PIPESTATUS[0]}
log "eval_perturbation exited $PE"
if [ "$PE" -ne 0 ]; then
    log "ABORT: perturbation eval failed"
    exit 1
fi

log "--- make_figures ---"
"$ENV_PY" -u scripts/make_figures.py --config "$CONFIG" 2>&1 | tee -a "$LOG"
log "make_figures exited ${PIPESTATUS[0]}"
log "=== PERTURBATION EVAL v2 COMPLETE ==="
