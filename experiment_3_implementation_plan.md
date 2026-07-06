# Agent Handoff: Launch Arm 1 and Arm 2 — Self-Contained Spec

## 0. What you are building (read first — you have no prior context on this project)

You are training **two transformer encoders** on a single-cell RNA-seq perturbation dataset (CD4⁺ T-cells, CRISPR knockdowns). Each model is a **JEPA** (Joint-Embedding Predictive Architecture): it masks some of a cell's genes, encodes the cell, and a predictor tries to reproduce the *latent representations* of the masked genes. The goal is a representation useful for predicting perturbation effects.

The two arms differ in exactly two ways from a known baseline ("A0"), and are otherwise identical to it:

- **Arm 1** = A0 + new gene tokenization + a reconstruction loss. Keeps A0's stabilization (VICReg + EMA teacher-student).
- **Arm 2** = Arm 1's tokenization and reconstruction, but replaces the stabilization with SIGReg and removes EMA/teacher-student (symmetric encoder).

A0 already exists (trained, 2000 HVGs, VICReg+EMA, no reconstruction, old tokenization; representation-quality Δ-Pearson 0.417). These two new arms let us compare: does the new tokenization help (Arm 1 vs A0), and does SIGReg-without-EMA work vs VICReg-with-EMA (Arm 1 vs Arm 2).

Everything below is specified concretely. Do not infer from outside knowledge; follow the shapes and formulas given.

---

## 1. Data and fixed configuration (identical for both arms)

- **Dataset:** CZI CD4⁺ T-cell Perturb-seq. Use the **same preprocessed data, same 2000 HVGs, same splits as A0/Experiment 1.** Load the committed gene vocab and split JSONs; do NOT recompute HVGs or regenerate splits.
- **Gene count:** G = 2000 (HVGs).
- **Splits:** donor-held-out (Donors 1–3 train, Donor 4 test) + held-out perturbation gene set — load from committed JSON, identical to A0.
- **Normalization:** log1p-CPM (as A0).
- **Architecture (match A0 exactly):** transformer encoder, **4 layers, 4 heads, d_model = 256**, pre-norm. (Do NOT use 8 layers — keep A0's size so the comparison isolates tokenization/stabilization, not depth.)
- **Batch size:** match A0, but **must be ≥ 256** (required for SIGReg in Arm 2 — its distribution estimate needs an adequate batch).
- **Optimizer / lr schedule / total steps:** match A0 exactly (same AdamW config, same warmup+cosine, same step count for matched-step comparison).
- **Masking:** match A0's masking scheme (same context/target block masking and ratios A0 used).

---

## 2. Gene tokenization (NEW — used by BOTH arms)

This replaces A0's tokenization. In A0, the gene-identity signal was a from-scratch learned `nn.Embedding(G, d_model)`. We replace the identity part with a **frozen ESM2 protein embedding**, concatenated with an expression embedding.

For each gene `g` in cell `c`, build token `t[g,c] ∈ R^{d_model}`:

### 2a. Identity sub-embedding (frozen ESM2)
- Precompute, once, an ESM2 embedding per gene: take the gene's **canonical protein**, run it through a **frozen ESM2 model**, **mean-pool over residues** → vector `e_esm[g] ∈ R^{d_esm}` (d_esm depends on the ESM2 variant chosen; record it).
- Store these as a fixed `[G, d_esm]` table (frozen, not updated in training).
- A **learned linear projection** maps it to the identity sub-dim: `h_id[g] = W_id @ e_esm[g] + b_id`, where `W_id ∈ R^{d_id × d_esm}`. Only `W_id, b_id` are trainable; `e_esm` is frozen.

**Fallback for genes with no canonical protein** (some HVGs may be non-coding): use a single shared learned vector `h_empty ∈ R^{d_id}` for all such genes, plus append a binary indicator feature (1 if fallback, 0 otherwise). **Before training, report how many of the 2000 HVGs use the fallback.** If > 5% (>100 genes), pause and report to Viet.

### 2b. Expression sub-embedding
- Encode the scalar log1p-CPM expression `x[g,c] ∈ R` with **the same expression encoder A0 used** (A0's value MLP: `1 → max(16, d/4) → d_expr`, GELU). Use A0's exact expression encoder so it is not a changed variable. → `h_expr[g,c] ∈ R^{d_expr}`.

### 2c. Assemble token
```
t[g,c] = LayerNorm( concat[ h_id[g] , h_expr[g,c] ] )      # shape [d_model]
# d_model = d_id + d_expr (+1 if you append the fallback indicator)
```
Pick `d_id` and `d_expr` so they sum to 256 (match A0's d_model). Suggested split d_id=192, d_expr=64, but you may match whatever proportion is natural; record the choice.

### 2d. Sequence assembly (both arms)
- Prepend a learnable `[CLS]` token. Sequence = `[CLS, t[1,c], …, t[G,c]]`.
- **No positional encoding** — genes are an unordered set; the identity embedding is the position signal. (Do not add sinusoidal/learned positional encodings.)

---

## 3. Encoder outputs (both arms)

Run the token sequence through the transformer. Keep **both** output granularities:
- `z[c] ∈ R^{256}` = the `[CLS]` output → the **cell embedding**.
- `u[g,c] ∈ R^{256}` = the per-gene outputs (one per gene token) → **per-gene contextualized embeddings**.

You need both: `z` is used by the prediction loss and the stabilization regularizer; `u` is used by the reconstruction loss.

---

## 4. The two views (both arms)

Following A0's JEPA masking:
- **Context view:** encoder applied to the *unmasked* subset of gene tokens.
- **Target view:** encoder applied to the *full* (or complementary) set.
- A **predictor** (small MLP) maps context-view representations to predicted target representations, at the **per-gene** level for masked genes.

The difference between arms is ONLY how the target view is produced and how collapse is prevented (see §5, §6).

---

## 5. Arm 1 — loss specification (VICReg + EMA + teacher-student + reconstruction)

Arm 1 keeps A0's stabilization (EMA teacher, VICReg) and **adds a reconstruction term**.

### 5a. Encoders
- **Student encoder** `f_θ` (trainable) processes the context view.
- **Teacher encoder** `f_θ'` = **EMA copy** of the student (momentum schedule identical to A0, e.g. 0.996 → 1.0 cosine). Teacher processes the target view. **Stop-gradient on the teacher output** (target is detached), exactly as A0.

### 5b. Loss terms
```
# (i) Predictive similarity loss — per-gene, masked genes M
#     pred = predictor(student context output) for masked genes
#     target = teacher target-view per-gene output (DETACHED)
L_pred = mean over g in M of  [ 1 - cosine( predictor(ctx)[g] , stopgrad(u_teacher[g]) ) ]
#   (or smooth-L1 if A0 used that — match A0's choice)

# (ii) Reconstruction loss — NEW — per-gene decode of masked genes from student per-gene output u[g]
#     decoder_rec: small MLP  R^256 -> R^1  applied per gene
L_rec = mean over g in M of  ( decoder_rec(u_student[g]) - x[g,c] )^2

# (iii) VICReg variance term — on the CLS cell embedding z (match A0)
#     std over the batch, per dimension; hinge at 1
std_z = sqrt( var(z over batch, per dim) + eps )       # shape [256]
L_vic = mean over dims of  relu( 1 - std_z )            # A0's variance-floor form (no covariance term)
```

### 5c. Total loss (Arm 1)
```
L_arm1 = L_pred + λ_rec * L_rec + λ_vic * L_vic
```
- `λ_vic = 0.01` (same as A0).
- `λ_rec`: start at a value that makes L_rec's magnitude comparable to L_pred early in training; if unsure start `λ_rec = 1.0` and confirm no single term dominates (see §8 monitoring). Record the value.

---

## 6. Arm 2 — loss specification (SIGReg, EMA-free, symmetric + reconstruction)

Arm 2 removes EMA/teacher-student entirely and uses SIGReg as the sole collapse-prevention mechanism.

### 6a. Encoder
- **Single shared encoder** `f_θ` (trainable). **Both** the context view and the target view pass through this same encoder.
- **No EMA, no teacher, no stop-gradient.** Gradients flow through both views.

### 6b. Loss terms
```
# (i) Predictive similarity loss — per-gene, masked genes M
#     BOTH views through the same encoder; NO stopgrad on the target
L_pred = mean over g in M of  [ 1 - cosine( predictor(ctx)[g] , u_target[g] ) ]   # NO stopgrad

# (ii) Reconstruction loss — identical to Arm 1
L_rec = mean over g in M of  ( decoder_rec(u[g]) - x[g,c] )^2

# (iii) SIGReg — on the CLS cell embedding z, anti-collapse
#     Use the official lejepa package. SIGReg projects z onto random directions
#     and applies the Epps-Pulley normality test toward N(0, I).
```
SIGReg implementation:
```python
import lejepa
# build once:
sigreg = lejepa.multivariate.SlicingUnivariateTest(
    univariate_test=lejepa.univariate.EppsPulley(num_points=17),
    num_slices=1024,
)
# per step, on the batch of CLS embeddings z (shape [batch, 256]):
L_sig = sigreg(z)
```
- **Pin the lejepa commit.** Before integrating, run lejepa's minimal example on random tensors to confirm gradients flow and the loss is finite.

### 6c. Total loss (Arm 2)
```
L_arm2 = L_pred + λ_rec * L_rec + λ_sig * L_sig
```
- `λ_rec`: same value as Arm 1 (hold constant across arms so the only difference is stabilization).
- `λ_sig`: **MUST be tuned — do NOT use 0.01.** See §7. (The 0.01 inherited from VICReg is mis-scaled for the Epps-Pulley statistic and caused a prior run's SIGReg term to sit flat and never satisfy.)

---

## 7. SIGReg weight scan (Arm 2 only — run BEFORE the full Arm 2 training)

1. Run short probes (~2–3k steps each) at `λ_sig ∈ {0.01, 0.1, 1.0}` (add a lower/higher value if behavior is odd).
2. For each, log over steps: `L_sig` value, per-dim std of z, effective rank of z's covariance, and `L_pred` + `L_rec`.
3. **Pick the λ_sig where ALL hold:** (a) `L_sig` *decreases and plateaus* (does NOT sit flat/unsatisfied), (b) `L_pred` and `L_rec` still decrease healthily (SIGReg not dominating/stalling them), (c) effective rank of z rises but training does not stall.
4. If no value satisfies all three, **report** — that itself is an informative result about SIGReg+reconstruction interaction. Document the scan either way.

---

## 8. Launch both arms in parallel

- Launch Arm 1 and Arm 2 **concurrently**, independent runs.
- Train both to the **same step count as A0** (matched-step comparison).
- **Log every N steps, identically for both arms:**
  - per-dim std of z (collapse check),
  - effective rank of z covariance (`exp(entropy(eigvals/sum))`),
  - mean pairwise cosine of z across the batch,
  - each loss term separately (`L_pred`, `L_rec`, and `L_vic` or `L_sig`),
  - KNN-probe accuracy on a small held-out labeled set (canary).
- **Arm 2 watch:** confirm `L_sig` is *decreasing*, not flat. If it goes flat (the prior failure mode), the weight scan failed — pause and report.
- Save the **EMA target encoder** (Arm 1) and the **encoder** (Arm 2) at the matched step and at plateau.

---

## 9. Evaluation (run on Arm 1, Arm 2, and re-run on A0/A1 for the full table)

Current harness is anisotropy-favoring only (linear decode, precision@k), which structurally disadvantages Arm 2 (SIGReg → isotropic). Add a higher-capacity probe and a retrieval metric.

1. **Linear (ridge) perturbation decode** — freeze encoder, fit ridge from cell embedding to perturbation delta; report Δ-Pearson, p@20, p@100. (Continuity with Exp 1 / A0.)
2. **MLP perturbation decode, `decode(mean z)` variant** — same as linear but with an MLP probe; report Δ-Pearson, p@20, p@100. (Higher-capacity; A0 scored 0.917 vs A1 0.285 on this — it is the trusted diagnostic.) **Use the `decode(mean z)` variant only.** Do NOT use any `mean(decode cell)` variant.
3. **Precision@k** — companion to linear decode (same geometric preference).
4. **Perturbation retrieval recall@k — BUILD THIS** (not in current harness; isotropy-favoring, the one metric that can favor Arm 2):
   - For each perturbation p, signature `s_p = mean(perturbed cell embeddings) − mean(matched control embeddings)`.
   - For each held-out perturbation, rank all other perturbations by `cosine(s_p, s_q)`.
   - Score whether biologically related perturbations rank highest. **Relatedness ground truth: ASK VIET** for the definition to use (e.g., same gene family / same pathway via STRING or GENIE3 / same complex) before building — do not invent it.
   - Report recall@k and mAP.

Held-out integrity: all decode/retrieval evaluation uses the held-out perturbation genes and held-out donor only; assert these were never in training.

---

## 10. Reporting

1. **Configs:** confirm 2000 HVGs + A0 architecture/splits/steps; token spec; d_id/d_expr split; ESM2 variant + d_esm; ESM2 fallback count; expression-encoder (A0's); λ_rec value; Arm 2 λ_sig scan + chosen value.
2. **Training:** curves for both arms, all loss terms separately, geometry metrics; explicit confirmation Arm 2's `L_sig` decreased (not flat).
3. **Results table** (Δ-Pearson / p@20 / p@100 / retrieval recall@k for each):
   | Encoder | linear | MLP decode(mean z) | retrieval |
   |---|---|---|---|
   | A0 (old token, VICReg+EMA) | 0.417 | 0.917 | ? |
   | A1 (old token, SIGReg+EMA) | 0.137 | 0.285 | ? |
   | Arm 1 (new token, VICReg+EMA+recon) | ? | ? | ? |
   | Arm 2 (new token, SIGReg, EMA-free+recon) | ? | ? | ? |
4. **Verdicts:**
   - **Tokenization:** Arm 1 vs A0 (note Arm 1 also adds reconstruction, so this reflects token+recon, not pure token).
   - **Stabilization:** Arm 1 vs Arm 2 (clean — identical except VICReg+EMA vs SIGReg+EMA-free; both have token+recon).
   - Recommendation for which config proceeds to the graph-loss experiment and to architecture/full-genome scaling.

---

## Behavioral guidance

- **Match A0 exactly** on architecture, gene set (2000 HVGs), splits, steps, batch (≥256), optimizer, masking, expression encoder. The ONLY intended changes: tokenization (both arms), +reconstruction (both arms), and stabilization (Arm 2). Anything else changed silently invalidates the comparison.
- **Retune λ_sig** (§7); never inherit 0.01.
- **No stopgrad in Arm 2**; **stopgrad on teacher in Arm 1**. Do not mix these up — it is the core difference.
- **Hold λ_rec identical across both arms.**
- **Report, don't paper over:** large ESM2 fallback count, SIGReg that won't satisfy, any term dominating — pause and report.
- **Build retrieval** but get the relatedness definition from Viet first.
- **No `mean(decode cell)` metric** — use `decode(mean z)` only.
- Pin all versions (lejepa commit, ESM2 variant/weights); save all checkpoints and eval artifacts.

## References
- SIGReg / lejepa: github.com/rbalestr-lab/lejepa ; arXiv:2511.08544
- A0 / Experiment 1 baseline config: EXPERIMENT_1_REPORT.md (match its settings)
