#!/bin/bash
set -euo pipefail

PROJ=/mnt/R0/Projects/POIAZ/Viet/JEPA-for-t-cells
ENV_PY=/home/qtran/miniconda3/envs/bulkformer/bin/python
CKPT=$PROJ/runs/poc/checkpoints/last.ckpt
CONFIG=$PROJ/configs/poc.yaml
LOG=$PROJ/runs/poc/eval.log

cd "$PROJ"
export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=0

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "=== EVAL CHAIN START ==="
log "Checkpoint: $CKPT"

log "--- Step 1/3: Annotation eval ---"
"$ENV_PY" -u scripts/eval_annotation.py --config "$CONFIG" --checkpoint "$CKPT" 2>&1 | tee -a "$LOG"
ANNOT_EXIT=$?
log "Annotation eval exited $ANNOT_EXIT"

if [ $ANNOT_EXIT -ne 0 ]; then
    log "ABORT: annotation eval failed"
    exit 1
fi

log "--- Step 2/3: Perturbation eval ---"
"$ENV_PY" -u scripts/eval_perturbation.py --config "$CONFIG" --checkpoint "$CKPT" 2>&1 | tee -a "$LOG"
PERTURB_EXIT=$?
log "Perturbation eval exited $PERTURB_EXIT"

if [ $PERTURB_EXIT -ne 0 ]; then
    log "ABORT: perturbation eval failed"
    exit 1
fi

log "--- Step 3/3: Make figures ---"
"$ENV_PY" -u scripts/make_figures.py --config "$CONFIG" 2>&1 | tee -a "$LOG"
FIG_EXIT=$?
log "Make figures exited $FIG_EXIT"

log "=== EVAL CHAIN COMPLETE ==="
log "Results in $PROJ/runs/poc/"
