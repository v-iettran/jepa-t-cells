# JEPA for T-cells — Full Project Narrative

*A running, narrative account of everything attempted so far: what each experiment tested, what worked, what didn't, and why each next step was taken. Compiled from the repo's reports, plans, diagnostics, run logs, and result JSONs.*

Last compiled: 6 Jul 2026.

---

## 0. The question and the yardstick

**Goal.** Learn a self-supervised representation of CD4⁺ T-cells from single-cell RNA-seq that captures **perturbation biology** — i.e., what a gene knockdown does to the transcriptome — *without ever training on perturbation labels*. The model is a **JEPA** (Joint-Embedding Predictive Architecture): mask a fraction of a cell's genes, encode the visible genes, and have a predictor reproduce the *latent* representations of the masked genes (not their raw counts). This is the single-cell analogue of I-JEPA.

**Dataset.** The CZI **Primary Human CD4⁺ T Cell Perturb-seq** dataset: 12 shards = 4 donors × 3 culture conditions (Rest / Stim8hr / Stim48hr), ~22M raw cells × ~18K genes, CRISPR knockdowns with verified guides. Non-targeting-control (NTC) cells are used for pretraining and as the control reference; targeting (perturbed) cells are used only for evaluation.

**The bar for success (from `benchmarking.md`).** Two recent benchmarks (Ahlmann-Eltze et al., Nature Methods 2025; Bendidi et al. 2024) found deep foundation models routinely *fail to beat* simple baselines on perturbation tasks. So every metric is compared against **PCA** and **scVI** on the same cells. Beating PCA/scVI is the definition of success. The evaluation suite has five tasks:

1. **Task 1 — Decode.** Fit a decoder (`z → 2000-gene expression`) on **control cells only**; check whether feeding a held-out knockdown's mean embedding reproduces its true expression delta (Δ-Pearson, precision@k). All effects are **deltas** vs *state-matched* controls.
2. **Task 2 — Linear vs non-linear.** Same decode with ridge vs MLP; the *gap* tells you whether the signal is linearly readable, non-linearly encoded, or absent.
3. **Task 3 — Retrieval.** Do functionally related perturbations land near each other in embedding space? (recall@k, mAP vs STRING / CORUM / Reactome.)
4. **Task 4 — Effect-size stratification.** Most knockdowns barely perturb; re-report every metric in weak/medium/**strong** strata. The strong stratum is the honest "is the model good" number.
5. **Task 5 — Cross-dataset transfer.** Freeze the encoder, transfer to Arce et al. 2024 (a different-lab CD4 Perturb-seq). The strongest test that the biology is real, not dataset memorization.

Everything below is the story of moving through Experiments 1 → 2 → 3 → 4 against this yardstick.

---

## 1. Experiment 1 — The proof-of-concept baseline (A0)

**What we built.** An I-JEPA-style model on **2000 HVGs**:

- **Context encoder** `GeneTokenEncoder`: learned `nn.Embedding(2000, 256)` gene identity + value MLP (`1→64→256`) + batch embedding + `[CLS]` token + 4-layer / 4-head / d=256 pre-norm Transformer. No positional encoding (genes are an unordered set).
- **Target encoder**: EMA copy of the context encoder, momentum 0.996 → 1.0 cosine, stop-gradient.
- **Predictor**: 2-layer Transformer with a learnable query-type token.
- **Objective**: mask 30% context genes, sample two 20%-of-genes target blocks; predict target-block latents from context; **Smooth-L1** prediction loss + a small **VICReg variance floor** (weight 0.01) on the CLS token to prevent collapse.
- **Training**: batch 512, AdamW (lr 1e-4, wd 0.05), 5k warmup + cosine, bf16, ~184k steps, **~6.8 days on one A6000**.

**Data prep.** Two-pass memory-efficient pipeline: backed-mode shard filtering (drop low-quality cells; keep only verified-knockdown targeting cells; NTC cap 500K, targeting cap 3M) → slim shards → `seurat_v3` HVG selection on NTC cells (batch_key `10xrun_id`) → subset + concat. log-CPM normalization on the fly. Final: 3.37M pretrain cells, plus separate annotation/control/eval pools. Five genes were held out **entirely from pretraining** — **CD28, CTLA4, FOXP3, IL2RA, STAT3** — the strict unseen-perturbation anchors.

### What worked

- **Training was stable.** Prediction loss rose from ~0.05 to ~0.26 as `target_std` grew 0.76 → 1.34 — the *healthy* JEPA signature (the EMA target gets richer/harder to predict), while the VICReg term fell to near-zero (no collapse).
- **Annotation (sanity check).** 3-class culture-condition linear probe: **JEPA macro-F1 0.993 ≈ PCA 0.994.** Essentially saturated — stimulation state is trivially separable — so this was demoted to a sanity check from here on.
- **Perturbation representation quality (Part A).** With a control-fit linear decoder, held-out knockdowns decoded at **overall Δ-Pearson 0.591**, well above the gene-agnostic mean-delta baseline (0.286). FOXP3 reached 0.747. **The latent space encodes gene-specific perturbation biology it was never trained on** — the core POC win.
- **Interpretability (literature validation).** Cosine-neighborhoods of the learned gene embeddings, cross-checked against 9,657 papers via the Paperclip CLI over 19 TFs, recovered **canonical TCR-inhibition TFs (EGR2, EGR3, FOXP3, FOXP1 — all strongly supported)** plus novel candidates. The embedding geometry is biologically meaningful.

### What didn't work

- **Unseen-gene head prediction (Part B) FAILED: overall Δ-Pearson 0.023.** A small MLP given a control cell's embedding + a 50-d co-expression gene-identity feature could not generalize to held-out genes. **The 50-d co-expression descriptor was too coarse.** Fixing this became Experiment 2's primary goal.

### Why Experiment 2

Two open threads: (1) the gene vocabulary was small (2000) and HVGs were selected *within* states, possibly under-representing stimulation-program biology; (2) the head arm failed, so we needed a better *gene-identity descriptor*. And separately, we wanted to test whether **SIGReg** (a newer anti-collapse regularizer) beats VICReg.

---

## 2. Experiment 2 — Bigger vocab, GRN priors, and the first SIGReg test

**Setup.** Re-preprocessed at **4000 HVGs**. Three hypotheses, controlled single-variable changes on the validated architecture:

| ID | Hypothesis | Change |
|----|-----------|--------|
| **H1** | SIGReg is a better collapse regularizer than VICReg | Encoder loss: VICReg → SIGReg (`lejepa` Epps-Pulley slicing test). **A0** (VICReg) vs **A1** (SIGReg). |
| **H2** | The model's *own* learned gene-token embedding is a strong enough gene descriptor to fix Part B | Head feature: 50-d co-expression → 256-d JEPA gene-token embedding (H2), vs baseline (H0). |
| **H3** | An external GENIE3 state-matched GRN embedding fixes Part B | Head feature: state-matched GENIE3 node2vec embedding (H3), fit on NTC cells only with a strict leakage guard. |

**GENIE3 GRN.** Three per-state networks (Rest/Stim8hr/Stim48hr) on NTC cells only, TF regulators (Lambert catalog), node2vec embeddings, with a hard leakage assertion (no perturbation-eval barcodes in the GRN fit).

### What worked

- **A0 remained the reliable encoder.** Representation-quality Δ-Pearson **0.417 (linear)** and, with the higher-capacity MLP `decode(mean z)` probe, **0.917** — the trusted diagnostic that A0's latent carries strong perturbation signal.

### What didn't work

- **H1 (SIGReg) FAILED.** A1 (SIGReg) scored **0.137 linear / 0.285 MLP(mean z)** vs A0's **0.417 / 0.917**. SIGReg produced a much weaker representation. (First SIGReg red flag.)
- **H2 and H3 (the head fixes) FAILED.** Every head feature — co-expression (H0 −0.148), JEPA gene-token (H2 −0.122), GENIE3 GRN (H3 −0.064) — gave **negative** overall Δ-Pearson. Neither the model's own embeddings nor an external GRN rescued unseen-gene *head prediction*. The Part B generalization problem was **not** a gene-descriptor problem.

### Why Experiment 3

Two takeaways reframed the project:

1. **The head-prediction framing was a dead end.** The value of the model is in the **representation** (Part A / decode), not in a supervised head predicting unseen genes. Evaluation should center on decode + retrieval + transfer (this crystallized into the Tasks 1–5 `benchmarking.md` framework).
2. **A richer gene tokenization might help the representation directly.** Instead of a from-scratch learned gene ID, inject a **frozen ESM2 protein embedding** per gene — biological prior knowledge about what each gene *is*. And add a **reconstruction** term so the per-gene outputs stay grounded in expression. SIGReg would get a second, cleaner test.

---

## 3. The consolidated benchmark — establishing the real baselines

Before Experiment 3's verdicts could mean anything, we built the **model-agnostic benchmark harness** (`src/jepa_poc/eval/`: `pools.py`, `embedders.py`, `tasks.py`) that scores every model — exp1, arm1, arm2, **PCA, scVI** — on identical cells through Tasks 1–4. This is the reference table everything is measured against.

**Tasks 1–2, held-out decode (5 strict genes):**

| model | dim | ridge Δ-Pearson | MLP(mean z) | MLP(per-cell) | ridge p@20 |
|-------|-----|----------------:|------------:|--------------:|-----------:|
| exp1 | 256 | 0.591 | −0.678 | 0.581 | 0.25 |
| arm1 (exp3, ESM2+VICReg+EMA) | 256 | 0.716 | 0.433 | 0.430 | 0.35 |
| arm2 (exp3, ESM2+SIGReg) | 256 | 0.700 | 0.031 | 0.527 | 0.40 |
| **pca** | 50 | **0.832** | 0.483 | 0.804 | 0.55 |
| scvi | 10 | 0.502 | 0.095 | 0.715 | 0.25 |

**Task 4, strong-effect stratum (ridge Δ-Pearson):** pca **0.740** > arm1 0.635 > arm2 0.597 > scvi 0.536 > exp1 0.511.

**Task 3, retrieval:** all models near the floor (mAP ≈ 0.02 STRING) — none of the encoders, nor PCA/scVI, organize perturbations by pathway meaningfully at this scale. A shared negative.

**The uncomfortable headline: plain PCA-50 beat every JEPA variant on decode (0.832).** This set up the central tension for the rest of the project — *why can't a 5M-param learned encoder beat a linear projection, and what is holding it back?*

---

## 4. Experiment 3 — ESM2 tokenization + reconstruction (arm1 / arm2)

**What changed vs A0** (`experiment_3_implementation_plan.md`), both arms:

- **Gene tokenization:** frozen **ESM2-650M** per-gene protein embedding (mean-pooled over residues) → learned linear projection to the identity sub-dim (d_id=191 + fallback indicator), concatenated with the expression MLP output (d_expr=64), LayerNorm → 256-d token. Only 13/2000 genes (0.65%) needed the non-coding fallback — well under the 5% gate.
- **Reconstruction loss:** a per-gene decoder (`256→128→1`) predicts the masked gene's scalar expression from the predictor output (MSE), added to the prediction + stabilization losses.

The two arms differ **only** in stabilization:

- **arm1** = VICReg variance floor + EMA teacher-student + stop-gradient (like A0).
- **arm2** = **SIGReg** (Epps-Pulley slicing, λ_sig tuned to 0.1 via a scan, *not* the mis-scaled 0.01) + **symmetric, EMA-free** encoder (gradients through both views).

### What worked

- **ESM2 + reconstruction improved decode over exp1.** arm1 ridge **0.716** and arm2 **0.700** both beat exp1's 0.591 — the richer tokenization + reconstruction helped the representation.
- **SIGReg trained stably this time** (arm2 no collapse during training, effective rank healthy), a step up from Exp 2's A1.

### What didn't work

- **Still short of PCA (0.832).** Neither arm closed the gap to the linear baseline.
- **arm2 MLP(mean z) = 0.031** — SIGReg's isotropic geometry reads poorly under the mean-z decode; consistent with SIGReg spreading signal thinly.

### The diagnostics that reframed everything

Chasing *why* the learned encoders under-perform produced three major findings:

**(a) OOD compression (`ood_compression_diagnostic.md`).** On Task 5 transfer to Arce, the **ESM2-tokenized** encoders squashed the entire external dataset into a tiny latent region (cell-to-cell scale: arm1 **0.65**, arm2 **0.60**) — ~20× more compression than exp1 (13.5) and far worse than PCA (16.4). The frozen ESM2 tokenization overfits the CZI input manifold; off-manifold inputs collapse to a corner. A genuine OOD-robustness liability of ESM2 tokens.

**(b) The batch-token hijack — the pivotal discovery (`within_batch_regularizer_handoff.md`).** The CZI data has 2 technical batches, and the model gets a batch token to factor out technical noise. But the anti-collapse regularizer was computed **globally over the whole minibatch** — which is *trivially satisfied by pushing the 2 batch anchors far apart*, leaving almost no variance budget for per-cell biology. Evidence:
  - Dropping the batch token at inference (`batch_mode=none`) collapsed the apparent "content scale" (arm1 16.3 → 2.2, arm2 14.7 → 0.8) — most of the spread *was the batch axis, not biology*.
  - Yet decode **improved** with the batch token dropped (**arm1 ridge 0.716 → 0.830, matching PCA's 0.832**) — the batch axis was *masking* decodable biology.
  - arm2 (SIGReg) was worst: content scale 0.80, decode flat after unmasking — its anti-collapse objective was satisfied *almost entirely by batch separation*.

  **This explained the PCA gap:** the learned encoders were spending their representational budget on a technical nuisance axis. It also gave a concrete fix (Experiment 4).

**(c) Task 5b reconciliation (`task5b_followup_offset_centering.md`).** Task 5 had a contradiction: decode-transfer (5a) showed arm1 transfers competitively with PCA (0.75 vs 0.77), but raw-cosine agreement (5b) showed the JEPA signatures "collapsed" (matched cosine ≈ 0). The follow-up showed 5b was **confounded by an un-cancelled nonlinear domain offset and dimension-mismatched cosine** (256-d JEPA vs 50-d PCA vs 10-d scVI). The decisive logic: a fresh ridge readout can rescale a small-but-intact signal but cannot resurrect genuinely-collapsed directions — so 5a's success means the perturbation info *is* preserved. Verdict leaned toward **(b) offset/dimension artifact**, not representational failure — softening the original "JEPA fails to transfer" headline.

### Why Experiment 4

The batch-token hijack (b) was the actionable one. If the global regularizer is satisfied by batch separation, then **computing the anti-collapse term *within each batch group separately, then averaging*** removes the shortcut and forces the model to spread cells *within* each batch — i.e., spread the biology. This single change (plus batch 512 for SIGReg's sample needs) defined Experiment 4, and it would also give SIGReg its **definitive fair test**.

---

## 5. Experiment 4 — Within-batch regularizer (Arm A / Arm B)

**The one change** (`within_batch_regularizer_handoff.md`): global anti-collapse → **within-batch** anti-collapse (`_within_batch_average`: compute the term per batch group on that group's cells only, no global centering, average across groups). Everything else identical to exp3 arm1/arm2.

| Arm | Tokenization | Regularizer | Stabilization |
|-----|-------------|-------------|---------------|
| **A** | ESM2 + expression | **within-batch VICReg** | EMA + teacher-student |
| **B** | ESM2 + expression | **within-batch SIGReg** | EMA-free, symmetric |

Supporting infrastructure added: a **balanced batch sampler** guaranteeing ≥256 cells/group per step (SIGReg needs it), **gradient checkpointing** (to fit batch 512 for the symmetric Arm B), gradient clipping, and geometry logging via SVD.

### Arm A — completed. What worked

- **The batch corruption is fixed.** Batch-variance share **63.6% (old arm1-trained) → 0.1% (Arm A trained)**; within-batch content scales equalized (arm1 R1/R2 1.16/13.56 → Arm A 16.37/16.37); batch centroid distance 25.4 → 0.84. **The regularizer no longer spends its budget separating batch anchors** — exactly the intended effect.
- **Batch and biology are cleanly separated.** Batch is still *linearly present* (89% balanced accuracy) but its direction is **orthogonal to biology** (cosine 0.004) — harmless, not entangled.
- **Decode is competitive and stable across inference modes.** Arm A ridge **0.798 (none) / 0.780 (trained)** — unlike old arm1, which needed the batch-token-dropped trick to reach 0.83. Arm A is *better-by-construction*: no inference-time fiddling.

### Arm A — what didn't work

- **It did not beat arm1-none (0.830) or PCA (0.832) on decode** — Arm A lands at 0.798. The fix cleaned the geometry but didn't unlock new signal.
- **A severe rank-2 collapse appeared.** Effective rank fell to **~1.8–2.4** (vs old arm1-none's 6.2). PCA of the CLS embeddings shows ~95% of variance in 2 dimensions. The representation is nearly **2-dimensional**.

### The rank-2 collapse investigation (read-only diagnostics)

A dedicated diagnostic (`exp4_armA_collapse_diagnostic.py`) tested two hypotheses for the collapse:

- **Projector absorption?** No — VICReg acts **directly on the CLS token**, there is no projector/expander head. Reg-space eff-rank = CLS eff-rank. A regularizer swap on the same head would target the collapsed tensor.
- **One-sided-hinge degeneracy?** Partly yes. On training batches the marginal std sits right at the γ=1 floor (std_min ≈ 0.98, hinge ≈ 0.002 — "satisfied") **while the joint spectrum stays ~2D**. The one-sided hinge `max(0, γ − std_j)` pins per-dimension variance but does nothing to prevent a low-rank *joint* covariance. Classic hinge failure.
- **An eff-rank measurement mismatch** was also surfaced and reconciled: training-log eff-rank (2.4, online encoder on training minibatches with batch token) vs eval-geometry (1.8, EMA on control pool, batch_mode=none) vs a fresh diagnostic embed (3.8) differ mainly by *pool, encoder (online vs EMA), and cell sampling* — not by formula. The much-cited "arm1-none 6.2" is a **different model** and not comparable.

**Gene mean-pool control (`exp4_armA_gene_meanpool_eval.py`).** We tested whether the missing rank/biology is hiding in the per-gene tokens by mean-pooling `context_tokens[:, 1:, :]` instead of using the CLS. Result: mean-pooling **raises eff-rank (~5.1 vs ~1.75)** but **destroys decode (ridge 0.50 vs 0.80)** and shrinks content scale ~10×. So the perturbation signal is concentrated specifically in the **CLS readout**; the low CLS rank is not "biology hiding in the gene tokens."

**Where this points.** No projector bypass → a two-sided / shape-aware regularizer (like SIGReg, which is what Arm B tests) applied to the same CLS is the correctly-targeted next lever; the one-sided VICReg hinge is the mechanistic cause of the 2D collapse.

### Arm B — in progress

Arm B (within-batch SIGReg) is the **definitive SIGReg test**: within-batch (can't cheat via batch separation) at adequate batch size. Training has been bumpy — an OOM at batch 512 (fixed with expandable-segments + gradient checkpointing), a corrupt NaN checkpoint at step 65k (quarantined; resumed cleanly from step 60k with gradient clipping). **Currently at ~step 77k / 200k**, healthy: finite losses, **effective rank ~198** (vs Arm A's ~2.4 — SIGReg is genuinely keeping the space high-rank), std_mean ≈ 0.99. The decision gate: SIGReg is *rescued* if Arm B's content scale and decode become competitive with Arm A; *dropped* if it still collapses/decodes flat despite this fair shot.

---

## 6. Scoreboard — held-out decode (5 strict genes, Δ-Pearson)

| Model | Tokenization | Regularizer | ridge (trained) | ridge (none) | MLP(mean z) | eff-rank (none) |
|-------|-------------|-------------|----------------:|-------------:|------------:|----------------:|
| **PCA-50** | — | — | — | **0.832** | 0.483 | — |
| exp1 (A0) | learned ID | global VICReg+EMA | 0.591 | 0.652 | −0.678 | — |
| exp2 A0 (4000 HVG) | learned ID | global VICReg+EMA | 0.417 | — | 0.917 | — |
| exp2 A1 (4000 HVG) | learned ID | global SIGReg | 0.137 | — | 0.285 | — |
| exp3 arm1 | ESM2 | global VICReg+EMA | 0.716 | 0.830 | 0.433 | 6.2 |
| exp3 arm2 | ESM2 | global SIGReg | 0.700 | 0.711 | 0.031 | — |
| **exp4 Arm A** | ESM2 | **within-batch VICReg** | 0.780 | 0.798 | 0.588 | **1.8** |
| exp4 Arm B | ESM2 | within-batch SIGReg | *training (~77k/200k)* | — | — | ~198 (train) |
| scVI-10 | — | — | — | 0.502 | 0.095 | — |

*(Decode numbers come from different eval snapshots — exp1/exp2 from their own eval runs, arm1/arm2 from the consolidated benchmark and batch-token test, Arm A from `exp4_armA_eval.json`. Ridge Δ-Pearson is the decision metric throughout; MLP-per-cell is high-variance and not used to adjudicate.)*

---

## 7. What we've learned (the throughline)

1. **The JEPA representation encodes real, gene-specific perturbation biology** it never trained on (Exp 1's 0.591; interpretable TF neighborhoods; competitive transfer in Task 5a). The core scientific premise holds.
2. **Supervised unseen-gene *head prediction* is the wrong framing** — every gene-descriptor variant failed (Exp 2). Value lives in the representation (decode / retrieval / transfer), not a predictive head.
3. **The hardest competitor is PCA-50 (0.832 decode).** No learned encoder has cleanly beaten it. Understanding *why* drove the whole diagnostic arc.
4. **A large chunk of the gap was a technical artifact, not a modeling ceiling:** the global anti-collapse regularizer was hijacked by the batch token. Dropping the batch token at inference recovered arm1 to 0.830 (≈PCA); computing the regularizer within-batch (Exp 4 Arm A) fixed it *by construction* (batch-variance share 63.6% → 0.1%).
5. **ESM2 tokenization is a double-edged sword:** it improved in-distribution decode over learned IDs, but it **overfits the training manifold and compresses OOD data ~20×** — a transfer liability.
6. **VICReg's one-sided variance hinge is prone to low-rank collapse.** Fixing the batch hijack exposed it: Arm A satisfies the per-dimension floor while collapsing to a ~2D joint manifold. The signal survives (decode 0.80) but the geometry is degenerate — motivating a **shape-aware regularizer** (Arm B / SIGReg) on the CLS.
7. **SIGReg has failed twice under earlier conditions** (Exp 2 A1; Exp 3 arm2 batch-hijack starvation). Arm B is its fair, decisive test — and early on it *is* holding the space high-rank (~198), which is the property VICReg lacks.

---

## 8. Open items / next steps

- **Finish Arm B (within-batch SIGReg)** to 200k, then run the full benchmark + geometry + transfer suite against Arm A. This settles the SIGReg question (rescued vs dropped) and tells us whether a two-sided regularizer fixes Arm A's rank-2 collapse while keeping decode.
- **Decide the regularizer** for the deferred **graph-loss round** (GRN-informed loss), layered on whichever of Arm A / Arm B wins.
- **Retrieval is a shared floor** (Task 3 mAP ≈ 0.02 for everyone) — revisit whether any representation choice can organize perturbations by pathway, or accept decode/transfer as the primary axes.
- **OOD robustness of ESM2 tokens** remains unaddressed — a candidate for a future inference-time alignment or an OOD-robustness regularizer if transfer becomes a headline goal.

---

## 9. Where things live (pointers)

**Narrative source docs:** `EXPERIMENT_1_REPORT.md`, `EXPERIMENT_2_REPORT.md`, `experiment_2_preparation.md`, `experiment_2_implementation_plan.md`, `experiment_3_implementation_plan.md`, `benchmarking.md`, `ood_compression_diagnostic.md`, `task5_agent_guidance.md`, `task5b_followup_offset_centering.md`, `within_batch_regularizer_handoff.md`.

**Key results:**
- Consolidated benchmark: `runs/benchmark/benchmark_table.md`, `runs/benchmark/benchmark_results.json`
- Batch-token test: `runs/benchmark/batch_token_indist.json`
- Task 5 transfer: `runs/task5/task5_results.json`, `runs/task5_batchfix/task5_results.json`
- Exp 4 Arm A: `runs/benchmark/exp4_armA/EXP4_ARM_A_EVAL_REPORT.md` and `checks/` (representation checks, PC-axis, collapse diagnostic, gene mean-pool)
- Training curves: `runs/benchmark/training_curves/`

**Models:** `runs/exp3_arm1`, `runs/exp3_arm2`, `runs/exp4_armA`, `runs/exp4_armB` (training). Code: `src/jepa_poc/` (models, eval harness), `scripts/`.
