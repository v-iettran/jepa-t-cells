# Agent Task: Task 5b Follow-up — Resolve "Representational Failure vs Domain-Offset Artifact"

## Why this exists (context — read first)

Task 5 (cross-dataset transfer, CZI → Arce et al. external T-cell Perturb-seq) produced a **contradiction between its two sub-tests** that the original report did not resolve, and the report's headline ("learned encoders fail to transfer; PCA/scVI transfer") over-reads one side of it. This follow-up runs the controlled analyses that decide which interpretation is correct. Run it on the same frozen models and cached embeddings from Task 5; no retraining.

**The contradiction:**
- **5a (decode transfer, fresh ridge readout):** Arm1 transfers *competitively with PCA* — transfer ridge Δ-Pearson 0.750 (Arm1) vs 0.774 (PCA), and Arm1 even improves over its own in-distribution 0.716. Strong stratum: Arm1 0.816 vs PCA 0.844. So the perturbation information appears to survive transfer in linearly-decodable form.
- **5b (raw cosine agreement):** Arm1/Arm2 signatures appear to *collapse* — ‖s‖Arce ≈ 0.08 vs ‖s‖CZI ≈ 2.6–3.1 (≈3% retention), matched cosine ≈ 0, far below PCA (0.38) and scVI (0.32).

**Why they may not actually conflict (the hypothesis to test):** 5b is confounded by two things that structurally penalize the 256-d nonlinear JEPA encoders independent of whether they captured the biology:
1. **Un-cancelled nonlinear domain offset.** Step 6 showed every model separates CZI from Arce almost completely (kNN purity 0.93–0.99). For a (near-)linear map (PCA), the constant dataset offset cancels in the perturbed−control subtraction. For a nonlinear encoder it does not cancel cleanly, so the perturbation component can be present but swamped/distorted by the large between-dataset offset.
2. **Dimension-mismatched cosine.** PCA is 50-d, scVI 10-d, JEPA 256-d. Cosine, signature norm, and the noise floor are all dimension-dependent (noise floor ≈0.06 at 256-d, ≈0.14 at 50-d, ≈0.32 at 10-d). A 256-d space dilutes a perturbation signal across more dimensions, so raw cosine is not an apples-to-apples comparison across these dims.

**The decisive logic:** a fresh ridge readout (5a) can *rescale* a small-but-intact signal, but it **cannot reconstruct directional information that has genuinely collapsed into noise.** Arm1 scoring 0.750 in 5a is therefore strong evidence the perturbation information IS preserved in the latent — implying 5b's "collapse" is a measurement artifact of offset + dimension, not a representational failure. This follow-up tests that directly.

**Two competing conclusions this task decides between:**
- **(a) Representational failure:** the JEPA genuinely fails to encode the Arce perturbation. (Original report's headline.)
- **(b) Offset/dimension artifact:** the JEPA encodes it fine, but the raw-cosine 5b view is masked by un-cancelled domain offset and dimension dilution.

Run the analyses; report which conclusion the evidence supports. Do not assume (b) — it is the hypothesis, not the desired answer.

---

## Analyses to run (all on cached Task 5 embeddings; nothing retrained)

### Analysis 1 — Domain-offset centering before 5b (THE decisive test)

The single most important analysis. Remove the estimated CZI→Arce domain offset in latent space, then recompute 5b.

**Steps:**
1. For each model, estimate the domain offset as the difference of control-cell means across datasets:
   `offset = mean(Arce control latents) − mean(CZI control latents)`  (in that model's latent space, computed on control/NTC cells only).
   - Compute this **per state×cell-type group** where possible (Rest/Stim × Teff/Treg), not just globally — the offset may be condition-dependent. Use a per-group offset matched to each signature's composition, falling back to global if a group is too small.
2. Subtract the (composition-matched) offset from all Arce latents: `z_Arce_centered = z_Arce − offset`.
3. Rebuild the Arce signatures from centered latents: `s_g^Arce_centered = mean(centered perturbed-g) − matched centered control ref`. (Note: if the offset is estimated from controls, the control reference itself centers to ≈0, so this is effectively re-expressing the perturbation relative to the de-shifted control — which is the point.)
4. Recompute the full 5b table with centered Arce signatures: matched cosine, mismatched cosine, separation, AUROC, ‖s‖Arce, signal retention, per-gene matched cosines.

**Interpretation (this decides a vs b):**
- If centering **restores** the JEPA signatures' magnitude and matched-cosine (‖s‖Arce recovers toward ‖s‖CZI, matched cosine rises above noise floor, AUROC improves) → **conclusion (b)**: the signal was present, masked by un-cancelled offset. The original headline is wrong/over-stated.
- If centering leaves them **collapsed** (‖s‖Arce stays ≈0.08, cosine stays at noise floor) → **conclusion (a)**: genuine representational failure. The original headline stands.
- Apply the *same* centering to PCA/scVI for fairness and report — for linear PCA, centering should change little (offset already cancels), which is itself a useful consistency check that the centering is implemented correctly.

### Analysis 2 — Dimension-matched comparison

Make the cross-model comparison apples-to-apples on dimension.
1. Fit a PCA at **256 components** on the CZI fit pool (matching the JEPA latent dim) and run it through the entire 5b pipeline (and centered 5b from Analysis 1). Compare PCA-256 vs PCA-50 vs the JEPA-256 models on equal dimensional footing.
2. Additionally report a **dimension-normalized agreement metric** alongside raw cosine — e.g., the matched-cosine expressed in units of that dimension's noise floor (matched_cos / noise_floor_at_that_dim), or AUROC (which is already dimension-robust since it's rank-based). AUROC is the cleaner cross-dimension comparison; emphasize it over raw cosine.

**Interpretation:** if PCA-50's apparent 5b advantage shrinks substantially at PCA-256, part of the original 5b gap was dimension concentration, not representational superiority.

### Analysis 3 — Reconcile 5a and 5b in a common space

Check whether the agreement appears once you view it through the readout (decoded-effect space) rather than raw latent.
1. Using the fresh Arce-control-fit ridge decoder from 5a, decode both the CZI and Arce signatures into 2000-HVG expression-delta space.
2. Compute the cross-dataset agreement (matched vs mismatched cosine, AUROC) in this **decoded expression-delta space** for the 10 overlapping genes, per model.

**Interpretation:** if the JEPA models show cross-dataset agreement in decoded-effect space (even though they didn't in raw-latent 5b), that confirms the perturbation information transferred and only the raw-latent-cosine lens missed it — directly reconciling 5a (positive) with 5b (negative).

### Analysis 4 — Anatomy of the ‖s‖Arce ≈ 0.08 collapse

Interrogate the collapse number directly rather than trusting the summary statistic.
1. For Arm1/Arm2 on Arce: compute ‖domain offset‖ (CZI→Arce control-mean distance) and compare to ‖perturbation signature‖ on each side. Report the ratio ‖offset‖ / ‖s_perturbation‖ per model.
2. For a few overlapping genes, plot (or report summary stats of) the Arce perturbed-cell latents vs Arce control-cell latents: is it that perturbed and control truly coincide (genuine collapse), or that both sit far from CZI (large offset) with a small but real perturbed−control separation that's tiny *relative to the offset* but nonzero in absolute terms?
3. Report whether the perturbed−control separation in Arce, while small, is *consistent in direction* with the CZI separation (sign/cosine of the raw, un-normalized delta) — a small-but-aligned delta supports (b); a random-direction delta supports (a).

**Interpretation:** distinguishes "perturbation maps to the control point (a)" from "perturbation is real but dwarfed by domain offset (b)."

---

## Reporting

Produce a follow-up report with:
1. **Verdict up front:** does the evidence support (a) representational failure or (b) offset/dimension artifact — or a mix? State it plainly, with the deciding analysis (#1 centering) called out.
2. **Analysis 1:** centered 5b table vs original 5b table, side by side, all 5 models. The ‖s‖Arce recovery (or not) for Arm1/Arm2 is the headline number.
3. **Analysis 2:** PCA-256 vs PCA-50 vs JEPA on 5b (raw + centered); AUROC emphasized as the dimension-robust metric.
4. **Analysis 3:** cross-dataset agreement in decoded-effect space; does it reconcile 5a and 5b?
5. **Analysis 4:** offset-to-signal ratio and the perturbed-vs-control anatomy for the JEPA models.
6. **Revised interpretation of Task 5 overall:** restate what the corrected evidence supports about whether the learned encoders transfer, replacing the original report's headline if the analyses warrant it.

## Behavioral guidance
- Run on cached Task 5 embeddings and frozen models; retrain nothing (except fitting PCA-256, which is a cheap projection on the CZI fit pool, and is fit on CZI only — never on Arce).
- Apply every analysis identically to all 5 models (PCA, scVI, exp1, arm1, arm2) for fair comparison; the centering and dimension-matching must be applied to baselines too.
- The centering offset is estimated from **control cells only** and never uses held-out perturbation structure.
- Do not assume conclusion (b); report what the centering shows even if it confirms (a).
- Keep the same control-matching (state×cell-type composition) as Task 5 throughout.
- AUROC and the offset-centered ‖s‖ recovery are the two numbers that matter most; lead with them.

## Inputs available
- Cached Task 5 embeddings: `runs/task5/<model>/task5.npz` (CZI-side and Arce-side latents per model).
- Frozen models and CZI fit pool as used in Task 5.
- The 10 overlapping genes: BACH2, BATF, IL2RA, IRF1, IRF4, KLF2, LEF1, MYB, MYC, SOCS3.
- Original Task 5 report and `runs/task5/task5_results.json`.
