# JEPA T-cells POC

This repo is a one-and-a-half-week proof-of-concept scaffold for a JEPA-style model on T-cell single-cell RNA-seq. It implements an I-JEPA-inspired objective with a context encoder, EMA target encoder, target-gene predictor, and latent-space loss. The code is designed to run before the real datasets are available by using a synthetic AnnData fixture for tests and smoke runs.

## What This POC Tests

The goal is not to prove a publishable benchmark result. The goal is to answer whether the JEPA objective trains stably on T-cell scRNA-seq-style data and produces embeddings that are usable for:

- 12-way T-cell annotation on held-out donors.
- Perturbation response prediction on held-out strong-effect perturbations.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

For CUDA, install the PyTorch wheel matching your local driver before installing the remaining requirements.

## Expected Data Layout

Set these paths in `configs/poc.yaml` after you place the real files:

```yaml
data:
  tcellatlas_path: data/raw/tcellatlas.h5ad
  perturb_seq_path: data/raw/perturb_seq.h5ad
```

Both loaders expect `.h5ad` files. Required `obs` columns are configurable:

- `batch`
- `donor`
- `cell_type`
- `perturbation`

If the incoming datasets use different column names, update the matching keys in `configs/poc.yaml`.

## Smoke Test Without Real Data

```bash
pytest tests/
```

To create synthetic processed files and run the full script workflow without real data:

```bash
python scripts/prep_data.py --synthetic
python scripts/train_jepa.py
python scripts/eval_annotation.py
python scripts/eval_perturbation.py
python scripts/make_figures.py
```

For quick local debugging on CPU, reduce `train.max_steps`, `train.devices`, and `model.d_model` in `configs/poc.yaml`.

## Real POC Workflow

```bash
python scripts/prep_data.py
python scripts/train_jepa.py
python scripts/eval_annotation.py
python scripts/eval_perturbation.py
python scripts/make_figures.py
```

Outputs go to `runs/poc/` by default:

- `last.ckpt`
- `ema_target_encoder.pt`
- `annotation_results.json`
- `perturbation_results.json`
- `results_table.md`
- `results_summary.png`

## Model

The POC model uses:

- Gene token embedding plus expression-value MLP.
- Batch embedding added to every gene token.
- 4-layer transformer context encoder.
- EMA target encoder with stop-gradient.
- 2-layer transformer predictor.
- Smooth-L1 latent prediction loss plus a small VICReg variance term.

The default config targets 2 RTX A6000 GPUs using Lightning DDP.

## POC Success Criteria

The POC is feasibility-positive if:

- JEPA linear-probe annotation macro-F1 is at least competitive with PCA+logistic regression.
- Perturbation delta-expression Pearson and Precision@k beat trivial identity / mean-effect baselines and are not worse than PCA+ridge.
- Training curves show decreasing validation loss and the collapse monitor does not fire.

## Next, If Greenlit

Deferred from the POC:

- Contrastive baseline.
- scGPT zero-shot baseline.
- GEARS / CPA / scGen baselines.
- v1 scaling run.
- Full 68-class annotation.
- Leave-one-donor perturbation robustness split.
- Full ablation matrix.
- Interpretability beyond UMAP and basic attribution hooks.
