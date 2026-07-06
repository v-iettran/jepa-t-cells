# Experiment 2 Report

Auto-generated summary. Fill in interpretation and figures after reviewing the full JSON outputs.

## Encoder Comparison

| Encoder | Representation-quality delta Pearson |
|---|---:|
| A0: VICReg, 4000 HVGs | 0.4174 |
| A1: SIGReg, 4000 HVGs | 0.2675 |

Selected encoder: **A0**

## Head Comparison

| Head feature | Head delta Pearson |
|---|---:|
| H0: co-expression SVD | -0.1480 |
| H2: JEPA gene-token embedding | -0.1217 |
| H3: GENIE3 state-matched GRN | -0.0637 |
| H3+H2: GRN + JEPA concat | pending |

## Artifact Paths

- `configs/exp2.yaml`
- `configs/gene_vocab_4000.tsv`
- `data/processed/*_4000.h5ad`
- `data/grn/`
- `runs/exp2_A0/`, `runs/exp2_A1/`
- `runs/exp2_head_H0/`, `runs/exp2_head_H2/`, `runs/exp2_head_H3/`, `runs/exp2_head_H3_concat/`
