# Agent Handoff: Within-Batch Regularizer Retraining (Arm A + Arm B)

## Why this round exists (context — read fully; no prior project context assumed)

We train JEPA-style transformer encoders on the CZI CD4⁺ T-cell Perturb-seq dataset. Prior diagnostic work uncovered a specific failure mode that this round fixes:

**The anti-collapse regularizer was being satisfied by separating technical batch anchors instead of by spreading biology.** The CZI training data contains 2 technical batches (experimental runs), and the model is given a "batch token" so it can factor out technical noise. But the current models compute their anti-collapse regularizer (VICReg or SIGReg) **globally over the whole minibatch** — which is trivially satisfied by pushing the 2 batch anchors far apart, leaving almost no variance budget for per-cell biological content. Evidence:
- Dropping the batch token at inference collapsed the latent "content scale" (arm1 16.3→2.2, arm2 14.7→0.8), showing most of the apparent spread was the batch axis, not biology.
- Yet in-distribution decode *improved* with the batch token dropped (arm1 ridge 0.716→0.830, matching PCA's 0.832), proving the batch axis was *masking* decodable biology.
- arm2 (SIGReg) was the worst: content scale 0.80, decode flat after unmasking — SIGReg's anti-collapse objective was satisfied almost entirely by batch separation, so little biology was ever learned.

**The fix:** compute the anti-collapse regularizer **within each batch group separately, then average** — so separating batch anchors no longer satisfies it, forcing the model to spread cells *within* each batch (the biology). This round retrains with that fix and tests whether it (a) improves the learned representation and (b) finally lets SIGReg work.

**No graph loss this round.** The graph loss is deferred to the next round, layered on whichever arm wins here. This round changes ONE thing vs the current models: global regularizer → within-batch regularizer. Keep attribution clean.

## The two arms

| Arm | Tokenization | Regularizer | Stabilization | Graph loss |
|---|---|---|---|---|
| **A** | ESM2 + expression | **within-batch VICReg** | EMA + teacher-student (as current arm1) | none |
| **B** | ESM2 + expression | **within-batch SIGReg** | EMA-free, symmetric (as current arm2) | none |

Everything else — ESM2 tokenization, architecture (layers/heads/d_model), data, splits, held-out genes, optimizer, lr schedule, total steps, reconstruction + prediction losses — is **identical to the current arm1 (for Arm A) and arm2 (for Arm B)**. The ONLY change is the regularizer going from global to within-batch, plus the batch-size change below.

## Batch size: 512 (required, with a constraint)

Current models use batch 256. **This round uses batch 512.** Reason: within-batch SIGReg estimates its distribution (Epps-Pulley test over random projections) *per batch group*; with 2 groups it needs ≥256 cells per group, so ≥512 total. VICReg (Arm A) tolerates smaller groups fine (it uses simple second moments), but use 512 for both arms so they are comparable.

**Per-group sample-adequacy constraint (critical for Arm B):** the 2 batch groups will NOT always split 256/256 in a random minibatch. If a step draws 300/212, the SIGReg estimate on the 212-group is below floor. **Use a batch sampler that guarantees ≥256 cells from each batch group per step** (stratified/balanced sampling by batch ID), OR confirm the data loader already balances batches per minibatch. Report which. For Arm A (VICReg) this is not critical, but use the same sampler for both so they match.

## The within-batch regularizer — implementation (get this EXACTLY right)

This is the crux. The regularizer must be a genuine **per-group computation, averaged** — not a global computation, and not a global term with a within-batch term added.

**Pseudocode (applies to both VICReg and SIGReg; only the inner term differs):**
```
def within_batch_regularizer(cls_embeddings, batch_ids, reg_fn):
    # cls_embeddings: [B, D] CLS cell embeddings for the minibatch
    # batch_ids:      [B]    technical batch group per cell (2 groups)
    # reg_fn:         the per-group anti-collapse term (VICReg var+cov, OR SIGReg Epps-Pulley)
    total = 0.0
    groups = unique(batch_ids)
    for g in groups:
        z_g = cls_embeddings[batch_ids == g]      # cells in this batch group ONLY
        assert z_g.shape[0] >= 256                # Arm B: SIGReg needs adequate samples
        total += reg_fn(z_g)                       # compute the term on this group ALONE
    return total / len(groups)                     # average across groups
```

**Hard requirements:**
- `reg_fn` operates on `z_g` (one group's embeddings) ONLY. It never sees the other group, and there is NO global statistic computed across groups.
- **Do NOT mean-center or normalize across the full batch before grouping.** Any global centering reintroduces the batch-separation shortcut. Center/normalize *within* `reg_fn` on `z_g` if the term requires it.
- **Do NOT add a global regularizer term alongside the within-batch one.** The within-batch term fully replaces the global one.
- For **Arm A**, `reg_fn` = VICReg variance+covariance terms (same form/weights as current arm1, just computed on `z_g`).
- For **Arm B**, `reg_fn` = SIGReg via `lejepa` (`SlicingUnivariateTest(EppsPulley(num_points=17), num_slices=1024)`) computed on `z_g`. Same λ_sig as current arm2 unless the scan below indicates otherwise.

**The leak to guard against:** if implemented wrong (global centering, or global+local), the model can STILL satisfy the regularizer by separating batch anchors, and the whole round is wasted. The validation in the geometry diagnostics (below) will catch this — if content scale is still batch-dominated after training, the within-batch computation leaked.

**SIGReg weight (Arm B):** the current arm2 λ_sig was tuned for the global SIGReg. Within-batch changes the term's scale (averaged over 2 groups of half size). Run a short λ_sig sanity check (a few hundred steps at the current value ±) confirming the SIGReg term decreases and plateaus per-group and doesn't dominate prediction/reconstruction. Adjust if needed; document.

## Prediction and reconstruction losses: UNCHANGED, global

Only the anti-collapse regularizer goes within-batch. The prediction loss (per-gene latent prediction) and reconstruction loss (per-gene decode of masked genes) stay exactly as in the current arm1/arm2 — computed normally over the minibatch. Do not within-batch these.

## Evaluation (run on both new arms, plus current arm1/arm2 and PCA/scVI as reference)

**Use `batch_mode=none` for ALL JEPA embeddings** — this is the established-correct inference mode (drops the batch token, which prior work showed is the fair comparison). Report `batch_mode=trained` too for the new arms as a secondary check, but `none` is primary.

**1. In-distribution decode (the headline metric):**
- Held-out perturbation decode, same pipeline as the validated Tasks 1–2: 150K-NTC-fit ridge decoder, average→decode→subtract with state-matched controls, 5 held-out genes.
- Report **ridge Δ-Pearson** (deterministic, the decision metric), p@20, and per-stratum (weak/medium/strong) ridge.
- Reference line: PCA 0.832 ridge, arm1-none 0.830, arm2-none 0.711.
- (MLP per-cell may be reported but is known to be high-variance run-to-run; do not let it drive conclusions — ridge decides.)

**2. Geometry diagnostics (the mechanism check — THIS verifies the fix worked):**
For both new arms, report:
- **Content scale per batch group** (cell-to-cell RMS spread, computed within each batch group separately) AND globally. The fix WORKED if within-group content scale is now substantial (not collapsed to ~0.8) and the global scale is no longer dominated by the between-batch gap.
- **Effective rank** of the CLS embedding covariance (within-group and global).
- **Batch-axis check:** how much of the total embedding variance is explained by the batch label (e.g., variance of per-batch means / total variance, or a linear batch-classification accuracy from the embedding). The fix WORKED if this is much lower than the current arm1/arm2 (where the batch axis dominated).
- Compare all of the above to the current (global-regularizer) arm1/arm2 to show the change.

## Decision gates (write the verdicts explicitly)

**Arm A (within-batch VICReg):**
- Success = content scale no longer batch-dominated (geometry diagnostics) AND decode ≥ current arm1-none (~0.83), ideally exceeding it (recovering signal that the global regularizer never let the model learn).
- If decode improves beyond 0.83 → the within-batch fix recovered never-learned signal; strong result, this becomes the baseline for the graph-loss round.
- If decode ≈ 0.83 (matches the inference-fix number) but geometry is cleaner → the representation is better-by-construction (no batch-token fiddling needed, cleaner for transfer/retrieval) even if decode is saturated; still the preferred baseline.

**Arm B (within-batch SIGReg) — THE DEFINITIVE SIGReg TEST:**
- This is SIGReg's fair shot: within-batch (so it can't cheat via batch separation) at adequate batch size (≥256/group).
- **SIGReg is RESCUED** if Arm B's content scale is comparable to Arm A's AND it decodes competitively (≈ Arm A, or at least ≈ 0.80+). Then SIGReg stays in the project.
- **SIGReg is DROPPED** if Arm B STILL shows collapsed within-group content scale and/or flat decode despite this fair test. Given SIGReg's prior failures (A1 collapse, arm2 variance-starvation), a third failure under fair conditions is conclusive: consolidate on VICReg, drop SIGReg.
- State the verdict explicitly and which condition was met. This round is designed to settle the SIGReg question definitively — do not hedge.

## Reporting

1. Configs: confirm everything matches current arm1/arm2 except within-batch regularizer + batch 512; the sampler used (per-group balance); Arm B λ_sig check.
2. Training curves: all loss terms separately (prediction, reconstruction, regularizer) per arm; confirm the within-batch regularizer term behaves (Arm B SIGReg decreases per-group, doesn't stall prediction/recon).
3. Geometry diagnostics table: content scale (within-group + global), effective rank, batch-axis variance share — new arms vs current arm1/arm2. **This verifies the fix.**
4. Decode table: ridge Δ-Pearson + per-stratum, batch_mode=none, new arms vs current arms vs PCA/scVI.
5. Explicit verdicts: Arm A success/not; Arm B SIGReg rescued/dropped (with the gate condition met).
6. Recommendation: which arm proceeds to the graph-loss round.

## Behavioral guidance
- Change ONLY the regularizer (global→within-batch) and batch size (256→512) vs current arm1/arm2. Everything else identical — any other drift breaks attribution.
- The within-batch regularizer must be a genuine per-group computation; no global centering, no global+local. The geometry diagnostics are the check that it didn't leak.
- Use a batch sampler guaranteeing ≥256 cells per batch group per step (critical for Arm B).
- Evaluate with batch_mode=none (primary); ridge Δ-Pearson is the decision metric (MLP-per-cell is too high-variance to adjudicate).
- State the Arm B SIGReg verdict explicitly per the decision gate — this round settles SIGReg's fate.
- No graph loss this round.
- No invented results; if SIGReg won't satisfy, if the sampler can't guarantee group sizes, if geometry shows the fix leaked — report and stop.
- Pin versions (lejepa commit, ESM2 weights); save all checkpoints and eval artifacts.

## References
- lejepa / SIGReg: github.com/rbalestr-lab/lejepa
- Prior diagnostic evidence: OOD compression report + in-distribution batch-token test (batch token hijacks the global regularizer; batch_mode=none recovers decode).
- Current baselines: arm1-none ridge 0.830, arm2-none 0.711, PCA 0.832.
