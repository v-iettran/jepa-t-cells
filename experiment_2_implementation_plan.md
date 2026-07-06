# Experiment 2 Implementation Plan — JEPA for T-cells

## Context for the executing agent

This continues the `jepa-t-cells` project (Viet, MSc AI, UCD). Experiment 1 trained an I-JEPA-style model (transformer + EMA target encoder + block masking + VICReg variance term) on the CZI Primary Human CD4+ T-Cell Perturb-seq dataset (2000 HVGs, ~3.37M cells, 6.8 days on one A6000). Results:

- Culture-condition annotation: JEPA (0.993 macro-F1) ≈ PCA (0.994). **Saturated — demoted to sanity check only.**
- Perturbation representation quality (Part A): overall delta-Pearson **0.591** vs gene-agnostic baseline 0.286. **This works — the latent space encodes perturbation biology.**
- Unseen-gene head prediction (Part B): overall delta-Pearson **0.023**. **This FAILED — the 50-d co-expression gene feature cannot generalize to unseen perturbed genes. Fixing this is Experiment 2's primary scientific goal.**
- Gene-embedding TF-neighborhood analysis recovered EGR2/EGR3/FOXP3/FOXP1 as TCR-inhibition TFs plus novel candidates. **Strong interpretability asset — preserve the gene-token architecture that produced it.**

Experiment 2 tests three hypotheses. Do NOT wholesale-replace the architecture. The transformer + gene-token + EMA design is validated and produces the interpretable embeddings; we make controlled single-variable changes.

---

## The three hypotheses

| ID | Hypothesis | What changes | Type |
|---|---|---|---|
| **H1 (A1)** | SIGReg is a better collapse regularizer than the VICReg variance term, yielding cleaner embedding geometry and equal-or-better representation quality. | Encoder loss: VICReg variance term → SIGReg (official `lejepa` package). Everything else identical to A0. | Encoder arm |
| **H2 (H_JEPA)** | The model's OWN learned gene-token embeddings are a strong enough functional descriptor of the perturbed gene to fix unseen-gene prediction — no external knowledge needed. | Perturbation head: 50-d co-expression feature → perturbed gene's JEPA gene-token embedding. Frozen encoder. | Head arm |
| **H3 (H_GRN)** | An external GENIE3-inferred, state-matched GRN embedding of the perturbed gene improves unseen-gene prediction, and may beat the model's own embeddings. | Perturbation head: gene identity feature → state-matched GENIE3 node2vec embedding (Option B). Frozen encoder. GENIE3 fit on NTC cells only. | Head arm |

**Baselines retained in every comparison:**
- A0: Experiment 1 model re-run on 4000 HVGs (the new floor encoder).
- H0: Experiment 1's 50-d co-expression head (reproduces the 0.023 failure as the head baseline).
- Gene-agnostic mean-delta baseline (from Exp 1).

---

## Execution sequencing (as specified)

```
STAGE 0: Preprocess 4000 HVGs  ──┐
                                 │
STAGE 1: Fit GENIE3 (NTC only) ──┤  (GENIE3 runs once preprocessing done;
                                 │   its output is only needed for H3)
                                 │
STAGE 2: ┌─ A0 encoder train ────┤  PARALLEL
         └─ A1 encoder train ────┤  (the two encoder arms; H1 test = A0 vs A1)
                                 │
STAGE 3: Pick best encoder E*    │
         Head sweep on E*:       │
         ┌─ H0 (baseline) ───────┤  PARALLEL with each other
         ├─ H2 / H_JEPA ─────────┤  (H2 test = H0 vs H_JEPA)
                                 │
STAGE 4: H3 / H_GRN on E* ───────┘  (needs GENIE3 output from Stage 1)
                                     (H3 test = H_GRN vs H_JEPA vs H0)
```

Concretely: Stage 0 first. Then GENIE3 (Stage 1) and the two encoder trainings (Stage 2) can all run concurrently — GENIE3 does not depend on the encoders and the encoders do not depend on GENIE3. Stage 3 (H0, H_JEPA heads) runs once an encoder is ready and does NOT need GENIE3. Stage 4 (H_GRN head) runs last because it needs both E* and the GENIE3 output.

---

## STAGE 0 — Preprocess 4000 HVGs

Reuse the Experiment 1 pipeline (`scripts/prep_data.py`) with one change: HVG count 2000 → 4000.

### Steps

1. Same two-pass memory-efficient strategy as Exp 1 (backed-mode shard filtering → slim shards → HVG-aware concat).
2. **Pass 1 (HVG selection):** `seurat_v3` on NTC cells, top **4000** HVGs. Write `gene_vocab_4000.tsv`.
3. **Pass 2:** subset all slim shards to the 4000 HVGs, concatenate.
4. Same filtering as Exp 1: drop `low_quality=True`; keep only `signif_knockdown=True` targeting cells; NTC cap 500k; targeting cap 3M.
5. Same normalization: log-CPM on the fly in the dataset `__getitem__`.

### Critical: preserve splits exactly

The donor split and the held-out gene split MUST be identical across all arms and consistent with Exp 1's logic. Commit them as JSON.

- Donor split: Donors 1–3 train, Donor 4 test (same as Exp 1).
- **Held-out perturbation genes: EXPAND from 5 to ~15–20.** Exp 1's 5 genes (CD28, CTLA4, FOXP3, IL2RA, STAT3) showed high per-gene variance (0.384–0.747), making the overall number unstable. A 15–20 gene held-out set gives a credible generalization estimate. Select held-out genes stratified by effect size (include strong, medium, weak effectors) and by whether they are TFs vs surface receptors vs signaling. Record the list in `configs/heldout_genes.json`. Keep the original 5 as a clearly-labeled subset so Exp 1 ↔ Exp 2 remains comparable.

### Outputs

```
data/processed/pretrain_processed_4000.h5ad
data/processed/annotation_processed_4000.h5ad
data/processed/perturb_controls_processed_4000.h5ad
data/processed/perturb_eval_processed_4000.h5ad
configs/gene_vocab_4000.tsv
configs/donor_split.json
configs/heldout_genes.json
```

### Sanity checks before proceeding

- 4000 genes selected, no NaNs, sparse format preserved.
- Held-out genes are NOT present in any training perturbation cells (grep the perturbation labels).
- Report cell counts per split, per state, per donor.

---

## STAGE 1 — GENIE3 (NTC cells only, three state-specific networks, Option B)

Runs concurrently with Stage 2. Output only consumed in Stage 4.

### Design decision: Option B (state-matched embeddings)

We fit GENIE3 **separately per state** (Rest, Stim8hr, Stim48hr) on NTC cells only, producing three gene embeddings per gene. The perturbation head will look up the embedding matching the prediction's state. Rationale: the held-out genes (CTLA4, IL2RA, FOXP3, etc.) are activation-induced with state-dependent regulatory roles; a state-matched descriptor is sharper than an averaged one for exactly these genes.

### LEAKAGE GUARD (non-negotiable)

GENIE3 is fit on **NTC (non-targeting control) cells ONLY**. No targeting-perturbation cells, and absolutely no held-out-gene perturbation cells, may enter GENIE3 fitting. If any perturbation cells leak in, the unseen-gene benchmark is silently contaminated. Assert this in code: filter to `guide_type == NTC` before fitting, and assert zero overlap between GENIE3 input cell barcodes and the perturbation-eval cells.

### Steps

1. **Split NTC cells by state:** Rest, Stim8hr, Stim48hr.
2. **Decide regulator set:** GENIE3 conventionally uses a candidate-TF list as regulators. Use the Lambert et al. v1.01 human TF catalog intersected with the 4000 HVGs (this is the same TF list used in Exp 1's literature validation). All 4000 genes are targets; TFs are candidate regulators. This keeps GENIE3 tractable and biologically principled.
3. **Compute / subsample for tractability:** GENIE3 on millions of cells × 4000 genes is infeasible directly. Use one of:
   - Pseudobulk: aggregate NTC cells into pseudobulk replicates (e.g., by donor × state × random subsampling into N pseudobulk profiles per state). GENIE3 on pseudobulk is standard and fast.
   - Cell subsample: randomly subsample ~20–50k NTC cells per state and run GENIE3 on those.
   - Prefer pseudobulk if replicate count is sufficient (aim ≥ 30 pseudobulk profiles per state); else cell subsample.
4. **Run GENIE3 three times**, once per state → three importance matrices `G_rest`, `G_stim8`, `G_stim48`, each shape (4000 targets × n_TF regulators).
5. **Diagnostic (do this and report):** For the held-out genes specifically, measure how different their top-k regulators are across the three states (Jaccard overlap of top-20 regulators per gene per state-pair). This both validates the Option B choice and produces a "regulatory rewiring across activation states" figure. Report a small table: held-out gene × pairwise-state-Jaccard.
6. **Build per-state gene embeddings:** run node2vec on each importance matrix (treat as weighted directed graph; TF→target edges weighted by GENIE3 importance). Produce `gene_emb_grn_rest`, `gene_emb_grn_stim8`, `gene_emb_grn_stim48`, each (4000 × d_grn). Use a modest d_grn (e.g., 64 or 128). Pin node2vec hyperparameters (walk length, num walks, p, q) and record them.
   - Note: genes that are targets-only (not regulators) still get node embeddings as long as they have incoming edges. Genes with no edges in a given state get a zero/learned-default embedding — flag how many this affects per state, especially among held-out genes.

### Outputs

```
data/grn/genie3_importance_rest.parquet
data/grn/genie3_importance_stim8.parquet
data/grn/genie3_importance_stim48.parquet
data/grn/gene_emb_grn_rest.npy        # (4000, d_grn)
data/grn/gene_emb_grn_stim8.npy
data/grn/gene_emb_grn_stim48.npy
data/grn/node2vec_config.json
outputs/grn/heldout_gene_state_divergence.csv   # the diagnostic
data/grn/genie3_ntc_barcodes.txt      # for the leakage assertion
```

### Sanity checks

- Assert leakage guard (zero overlap with perturbation-eval cells).
- All three embedding matrices have 4000 rows aligned to `gene_vocab_4000.tsv` order.
- Report how many held-out genes have non-trivial (non-default) embeddings in each state.

---

## STAGE 2 — Encoder arms A0 and A1 (parallel; this is the H1 test)

Two encoder trainings, run concurrently if two GPUs/slots available, else sequentially.

### A0 — Experiment 1 model on 4000 HVGs (the floor)

Identical to Exp 1 architecture and hyperparameters, only the gene vocabulary changes (2000 → 4000):

- GeneTokenEncoder: gene embedding (4000×256) + value MLP (1→64→256) + batch embedding + CLS + 4-layer transformer (d=256, 4 heads).
- EMA target encoder, momentum 0.996→1.0 cosine.
- Predictor: 2-layer transformer.
- Masking: 30% context, two 20% target blocks.
- Loss: Smooth-L1 latent prediction + VICReg variance term (weight 0.01) on CLS.
- AdamW lr=1e-4 wd=0.05, bf16, batch 512, warmup 5000 + cosine, max 200k steps.

Expect ~9–11 days at 4000 HVGs on one A6000 (Exp 1 was 6.8 days at 2000). Budget accordingly.

### A1 — A0 with VICReg → SIGReg

Single change: replace the VICReg variance term with SIGReg from the official package.

```python
import lejepa
univariate_test = lejepa.univariate.EppsPulley(num_points=17)
sigreg_loss_fn = lejepa.multivariate.SlicingUnivariateTest(
    univariate_test=univariate_test,
    num_slices=1024,
)
# In the training step, on the CLS embeddings of the batch:
loss_sigreg = sigreg_loss_fn(cls_embeddings)   # cls_embeddings: [batch, 256]
total_loss = smooth_l1_pred_loss + lambda_sig * loss_sigreg
```

Implementation notes:
- **Install the official repo first** (`github.com/rbalestr-lab/lejepa`), pin the commit, and run their minimal example on random tensors to confirm gradients flow before integrating.
- **SIGReg needs a reasonably large batch** to estimate the characteristic function via the random slices. Keep batch ≥ 256 (512 is fine). Do NOT shrink batch for A1.
- `lambda_sig` is the single trade-off hyperparameter. Start at a value that makes the SIGReg term magnitude comparable to the old VICReg term's contribution; sweep [0.01, 0.1, 1.0] if time allows, else pick one and document.
- Apply SIGReg to the same embeddings VICReg was applied to (the CLS token). Optionally also to target-block embeddings — but for a clean single-variable H1 test, match Exp 1's application point (CLS only) in the primary run.

### Collapse / geometry monitoring (BOTH arms, every N steps)

Log identically for A0 and A1 so H1 is a fair comparison:
- Per-dim std of CLS embeddings across batch.
- Effective rank of CLS embedding covariance.
- Mean pairwise cosine similarity of CLS embeddings.
- KNN annotation accuracy on a small held-out probe set (the canary).

H1 success looks like: A1 trains stably (no collapse), achieves effective rank ≥ A0, and matches-or-beats A0 on representation quality (Stage 3 Part A) with a more isotropic embedding distribution.

### Outputs (per arm)

```
runs/exp2_A0/ema_target_encoder.pt
runs/exp2_A0/training_log.csv          # incl. geometry metrics
runs/exp2_A1/ema_target_encoder.pt
runs/exp2_A1/training_log.csv
outputs/exp2/H1_encoder_geometry_comparison.csv
```

---

## STAGE 3 — Head sweep H0 and H_JEPA on best encoder (parallel; this is the H2 test)

### Pick E*

After A0 and A1 finish, evaluate both on **representation quality (Part A)** — the perturbation delta-Pearson with a linear decoder, exactly as Exp 1 Part A. Pick the encoder with higher overall Part A delta-Pearson as E*. Record both numbers. (Also report the H1 geometry comparison regardless of which wins.)

Freeze E*. All head arms use the SAME frozen E* embeddings, so head differences are cleanly attributable.

### Shared head setup

The perturbation head predicts a perturbed cell's embedding (or delta) from: a control cell's E* embedding + a **gene-identity feature** for the perturbed gene. The head is a small MLP. The ONLY thing that differs across H0/H2/H3 is the gene-identity feature.

- **H0 (baseline):** 50-d co-expression feature (SVD of control pseudobulk), exactly as Exp 1. Reproduces the 0.023 failure.
- **H2 / H_JEPA:** the perturbed gene's JEPA gene-token embedding from E* (256-d), looked up from E*'s gene embedding table. No external data. Tests whether the model's own embeddings carry enough functional info.

Run H0 and H2 in parallel (both cheap — frozen encoder, only a small MLP trains).

### Evaluation (identical to Exp 1 Part B + retrieval)

For each held-out gene, on condition-matched controls:
- delta-Pearson, precision@20, precision@100.
- Overall (pooled across held-out genes).
- Compare against H0 and the gene-agnostic mean-delta baseline.

H2 success: H_JEPA overall delta-Pearson substantially exceeds H0's 0.023 and the gene-agnostic baseline's 0.286.

### Outputs

```
runs/exp2_head_H0/perturbation_results.json
runs/exp2_head_H2/perturbation_results.json
outputs/exp2/H2_head_comparison.csv
```

---

## STAGE 4 — H_GRN head on E* (this is the H3 test)

Runs last; needs both E* (Stage 3) and the GENIE3 embeddings (Stage 1).

### H3 / H_GRN setup

Same frozen E*, same MLP head structure, same eval. Gene-identity feature = **state-matched GENIE3 node2vec embedding (Option B)**.

State-matched lookup logic:
```python
# For a perturbation-eval cell with known state s and perturbed gene g:
if state == "Rest":      grn_feat = gene_emb_grn_rest[gene_idx[g]]
elif state == "Stim8hr": grn_feat = gene_emb_grn_stim8[gene_idx[g]]
elif state == "Stim48hr":grn_feat = gene_emb_grn_stim48[gene_idx[g]]
# State is known at both train and inference time because the perturbation
# eval is condition-matched (perturbed cell compared to condition-matched controls).
```

Optionally also test H_GRN+JEPA (concatenate GENIE3 state-matched embedding with the JEPA gene-token embedding) as a fourth feature variant — nearly free and tests complementarity. Label it clearly.

### LEAKAGE re-assertion

Before running, re-assert that the GENIE3 embeddings being loaded were fit on NTC cells with zero overlap with the perturbation-eval cells. Load `genie3_ntc_barcodes.txt` and assert disjoint from eval barcodes.

### Evaluation

Identical metrics to Stage 3. The headline three-way (four-way) comparison table:

| Head feature | Overall delta-Pearson | precision@20 | precision@100 |
|---|---|---|---|
| gene-agnostic mean (Exp 1 baseline) | 0.286 | — | — |
| H0: co-expression (Exp 1) | ~0.023 | — | — |
| H2: JEPA gene-token embedding | ? | ? | ? |
| H3: GENIE3 state-matched | ? | ? | ? |
| (optional) H3+H2 concat | ? | ? | ? |

H3 interpretation:
- If H3 > H2: external regulatory knowledge adds value beyond the model's own embeddings.
- If H2 ≥ H3: the model's learned embeddings already capture the functional info — a stronger, cooler result (the model gets GRN-level knowledge "for free").
- Either outcome is publishable and informative.

### Outputs

```
runs/exp2_head_H3/perturbation_results.json
runs/exp2_head_H3_concat/perturbation_results.json   # if run
outputs/exp2/H3_head_comparison.csv
outputs/exp2/EXPERIMENT_2_RESULTS_TABLE.md            # the master comparison
```

---

## Secondary benchmarks (run on E*, supporting not headline)

After the head sweep, run these on E* for completeness. These are Tier 2/3 from the benchmark discussion — they support the "good encoder" claim but are not the headline.

- **Perturbation retrieval:** embed each perturbation's mean effect (perturbed − matched control, in E* space); for each held-out perturbation, retrieve nearest neighbors among all perturbations; measure whether same-gene / same-pathway perturbations rank high. Report recall@k.
- **TF-neighborhood / GRN recovery:** repeat Exp 1's TF-neighborhood analysis on E* (now with 4000 HVGs → more TFs in vocab). Optionally formalize against a ground-truth GRN (DoRothEA) as AUROC rather than only literature counts.
- **Annotation (sanity floor only):** 3-class culture-condition linear probe. Expect ~0.99 again; report once to confirm E* is a sane encoder, do NOT feature as a result.
- **Donor batch integration (optional):** scIB metrics using donor as batch.

---

## Reporting

Produce `EXPERIMENT_2_REPORT.md` mirroring the Exp 1 report structure:

1. Dates, training time, hardware per arm.
2. Architecture (A0 vs A1 diff; head variants).
3. Preprocessing (4000 HVG changes, expanded held-out gene set).
4. GENIE3 protocol (per-state, NTC-only, leakage guard, the rewiring diagnostic figure).
5. **H1 result:** A0 vs A1 — representation quality + embedding geometry comparison.
6. **H2 result:** H0 vs H_JEPA — the unseen-gene fix.
7. **H3 result:** the master three/four-way head comparison table.
8. Secondary benchmarks.
9. Verdicts per hypothesis + recommendations for Experiment 3.

---

## Behavioral guidance for the agent

- **Verify before building.** Confirm all Stage 0 outputs and the leakage guards before training anything expensive.
- **Identical splits across all arms.** Any drift in donor split or held-out gene set invalidates the comparisons. Load from the committed JSON every time; never regenerate.
- **Frozen encoder for all head arms.** H0/H2/H3 differ ONLY in the gene-identity feature. Same E*, same head architecture, same eval harness, same random seed for the head MLP.
- **No invented results.** If GENIE3 fails to converge, node2vec produces degenerate embeddings, or SIGReg won't integrate, report and stop.
- **SIGReg gate:** run the official lejepa minimal example on random tensors before integrating into A1. If gradients don't flow or the loss is NaN, do not proceed with A1 — report and fall back to A0-only for this round.
- **Pin all versions and configs.** lejepa commit, node2vec params, GENIE3 params, all in committed config files.
- **Save intermediate artifacts** (GENIE3 matrices, all three GRN embeddings, E* embeddings, attribution outputs) so re-analysis doesn't require retraining.
- **Watch compute.** Encoder arms are ~9–11 days each. If only one long slot is available, prefer A0 (the floor + the encoder the head sweep needs) and defer A1/SIGReg to Experiment 3 — the head sweep (H0/H2/H3) is where the primary scientific result lives and it only needs ONE trained encoder.

## References

- LeJEPA / SIGReg code: https://github.com/rbalestr-lab/lejepa
- LeJEPA paper: Balestriero & LeCun, arXiv:2511.08544 (2025)
- scJEPA (dual-space, source of the SIGReg-for-single-cell idea): Zhai et al., LMRL Workshop ICLR 2026, OpenReview KadCjvcLOz
- GENIE3: Huynh-Thu et al., PLoS ONE 2010
- GEARS (perturbation prediction with gene graphs, conceptual precedent for H3): Roohani et al., Nat Biotechnol 2024
- Dataset: CZI Primary Human CD4+ T-Cell Perturb-seq, virtualcellmodels.cziscience.com
- Exp 1 report: EXPERIMENT_1_REPORT.md in the repo
