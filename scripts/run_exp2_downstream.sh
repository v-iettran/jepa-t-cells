#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-src}"

CONFIG="${1:-configs/exp2.yaml}"
SLEEP_SECONDS="${EXP2_POLL_SECONDS:-600}"

log() {
  printf '[%(%H:%M:%S)T] %s\n' -1 "$*"
}

wait_for_file() {
  local path="$1"
  log "Waiting for ${path}"
  until [[ -s "${path}" ]]; do
    sleep "${SLEEP_SECONDS}"
  done
  log "Found ${path}"
}

wait_for_file "runs/exp2_A0/last.ckpt"
wait_for_file "runs/exp2_A1/last.ckpt"

log "Evaluating A0 representation quality"
python scripts/eval_perturbation.py \
  --config "${CONFIG}" \
  --checkpoint runs/exp2_A0/last.ckpt \
  --run-dir runs/exp2_eval_A0 \
  --feature-mode coexpression

log "Evaluating A1 representation quality"
python scripts/eval_perturbation.py \
  --config "${CONFIG}" \
  --checkpoint runs/exp2_A1/last.ckpt \
  --run-dir runs/exp2_eval_A1 \
  --feature-mode coexpression

BEST_CKPT="$(python - <<'PY'
import json
from pathlib import Path

def score(path):
    payload = json.loads(Path(path).read_text())
    return float(payload["representation_quality"]["overall"]["delta_pearson"])

a0 = score("runs/exp2_eval_A0/perturbation_results_coexpression.json")
a1 = score("runs/exp2_eval_A1/perturbation_results_coexpression.json")
print("runs/exp2_A1/last.ckpt" if a1 > a0 else "runs/exp2_A0/last.ckpt")
PY
)"
log "Selected best encoder checkpoint: ${BEST_CKPT}"

log "Running H0 co-expression head on E*"
python scripts/eval_perturbation.py \
  --config "${CONFIG}" \
  --checkpoint "${BEST_CKPT}" \
  --run-dir runs/exp2_head_H0 \
  --feature-mode coexpression

log "Running H2 JEPA gene-token head on E*"
python scripts/eval_perturbation.py \
  --config "${CONFIG}" \
  --checkpoint "${BEST_CKPT}" \
  --run-dir runs/exp2_head_H2 \
  --feature-mode jepa

wait_for_file "data/grn/gene_emb_grn_rest.npy"
wait_for_file "data/grn/gene_emb_grn_stim8hr.npy"
wait_for_file "data/grn/gene_emb_grn_stim48hr.npy"

log "Running H3 GRN head on E*"
python scripts/eval_perturbation.py \
  --config "${CONFIG}" \
  --checkpoint "${BEST_CKPT}" \
  --run-dir runs/exp2_head_H3 \
  --feature-mode grn

log "Running H3+H2 concat head on E*"
python scripts/eval_perturbation.py \
  --config "${CONFIG}" \
  --checkpoint "${BEST_CKPT}" \
  --run-dir runs/exp2_head_H3_concat \
  --feature-mode grn_jepa

log "Writing Experiment 2 report"
python scripts/write_exp2_report.py --output EXPERIMENT_2_REPORT.md
log "Experiment 2 downstream pipeline complete"
