# Agent Guidance: Task 5 — Cross-Dataset Transfer (External Validation)

## Context (you have no prior project context — read this first)

We have trained several models on the CZI genome-scale CD4⁺ T-cell Perturb-seq dataset and evaluated them in-distribution (Tasks 1–4). The models are: **Exp1** (older model), **Arm1** (VICReg+EMA, new ESM2 tokenization), **Arm2** (SIGReg, EMA-free, new tokenization). Baselines: **PCA** (50 components) and **scVI**, fit on the same data.

In-distribution finding: **PCA beats all learned models on perturbation-effect decode** (PCA ridge Δ-Pearson ≈ 0.83 vs best learned ≈ 0.72). This is expected from the literature — linear methods win in-distribution. The hypothesis Task 5 tests is the flip side: **PCA, being a within-dataset variance maximizer, is expected to transfer POORLY to a new dataset, whereas a learned representation that captured generalizable T-cell biology may transfer better.** Task 5 is the experiment that could reveal the learned models' actual advantage. It may also show they have no advantage — both outcomes are informative; do not optimize toward a desired result.

**Frozen evaluation. No fine-tuning, no retraining.** All models (learned and baseline) are applied to the external data exactly as trained on CZI. We test whether the already-learned representation transfers.

## External dataset

**Arce et al. 2024, "Central control of dynamic gene circuits governs T cell rest and activation"** (Marson lab, Nature 637:930–939).
- Perturb-CITE-seq (CRISPRi knockdown) of primary human CD4⁺ **Teff and Treg** cells, in **resting and stimulated (48h)** states.
- ~28 knocked-down regulators (TFs/chromatin modifiers): MED12, KLF2, MYC, BATF, IRF4, STAT5B, FOXO1, GATA3, SOCS3, ZNF217, PRDM1, CBFB, and others.
- **Access:** processed data + tables on Zenodo `10.5281/zenodo.13924126` (preferred — already processed); raw on GEO SuperSeries `GSE271090`.

This is domain-matched (same cell type, same knockdown modality family, rest/stimulated axis) but genuinely out-of-distribution (different lab, donors, protocol, and a different/smaller perturbation panel). It also contains Treg cells, which add a cell-subtype shift.

## What to run

Run the full pipeline below for **all five models**: Exp1, Arm1, Arm2, PCA, scVI. Every model is evaluated identically. The comparison of interest is **how much each model degrades from in-distribution to transfer**, and especially **whether the learned models degrade less than PCA**.

---

### Step 1 — Acquire and QC the external data
- Download processed Perturb-CITE-seq from Zenodo (preferred) or raw from GSE271090.
- Apply the same QC as CZI training: drop low-quality cells; keep cells with confirmed knockdown; identify the non-targeting/safe-harbour control cells (the paper uses AAVS1 knockouts as controls).
- Record cell counts per (cell type [Teff/Treg] × state [rest/stim] × perturbation).
- Identify the control (NTC/AAVS1) cells — these are needed for the decoder and the matched-control references.

### Step 2 — Gene-space alignment (and the gate)
- Map the external dataset's genes onto the CZI 2000-HVG vocabulary. Harmonize gene IDs to a common system (Ensembl recommended); handle symbol↔ID mismatches.
- For CZI HVGs present in external data: use measured values. For CZI HVGs absent: zero-fill, and record the fraction zero-filled.
- **GATE: report the overlap fraction before proceeding.** If overlap < ~70% of the 2000 HVGs, transfer scores will be depressed for trivial input-mismatch reasons; flag this prominently and interpret all downstream numbers with that caveat. Do not silently proceed past a low overlap.

### Step 3 — Embed external cells through each FROZEN model
- For Exp1/Arm1/Arm2: run external cells through the frozen encoder → external embeddings (256-d).
- For PCA: apply the CZI-fit PCA projection (the same 50-component transform fit on CZI train cells) to the external cells → 50-d external embeddings. **Do NOT refit PCA on external data** — that would defeat the transfer test. Use the CZI-fit projection.
- For scVI: apply the CZI-trained scVI encoder to external cells (frozen), per scVI's standard transfer/query procedure → external embeddings.
- All five now have external-cell embeddings in their respective (CZI-fit) latent spaces.

### Step 4 — Fit a fresh decoder on external CONTROL cells (per model)
For the decode-transfer test (Step 5a), fit a fresh decoder on the external dataset's **control cells only**, per model:
- Input = external control cell embedding (from Step 3, that model's space). Output = external control cell expression (2000 HVGs aligned in Step 2).
- Use ridge (primary) and MLP (secondary). For MLP, use **per-cell decoding** (decode each cell, then aggregate) — NOT average-the-embedding-then-decode, which is invalid for non-linear decoders (it produced negative correlations in prior runs). For ridge, average-first and per-cell are equivalent (linear), so either is fine; use the same aggregation as the true-delta construction.
- Rationale: fitting the decoder fresh on external controls isolates the **encoder's** transferability — we test whether the frozen representation is good enough that a freshly-fit readout recovers effects in the new dataset, not whether the CZI decoder also transfers.
- Requires enough external control cells; report the count. If too few to fit a stable decoder, report and fall back to reusing the CZI decoder (note the change).

### Step 5 — The transfer evaluations

**5a. Decode transfer (reuses the Task 1 pipeline on external data).**
- For each external perturbation, build the true delta: mean(perturbed expr) − state-matched mean(control expr), where the control reference is the per-state-mean of external controls weighted by the perturbation's state composition (same construction as Task 1).
- Build the predicted delta: average perturbed embeddings → decode → minus decode(state-matched control embedding ref). (Per-cell for MLP.)
- Score Δ-Pearson and precision@k per external perturbation, then pooled.
- **Stratify by effect size** (Task 4 style): the external set is small and mixed (MED12 strong, others weak), so report strong/medium/weak separately if enough perturbations; at minimum flag each perturbation's true effect magnitude.

**5b. Cross-dataset agreement — THE HEADLINE TEST.**
This is the most robust and most important result. For perturbation genes present in **both** CZI and Arce:
- First, compute and report the **overlap set**: which of CZI's 964 perturbed genes are also knocked down in Arce. (Likely MED12, MYC, BATF, IRF4, STAT5B, FOXO1, GATA3, etc. — even 10–20 genes is enough.)
- For each overlapping gene g, build its effect signature *independently* in each dataset, both expressed in the model's latent space:
  - `s_g^CZI` = mean(CZI perturbed-g embeddings) − matched CZI control ref  [model's latent dim]
  - `s_g^Arce` = mean(Arce perturbed-g embeddings) − matched Arce control ref  [same dim]
- Compute `cos(s_g^CZI, s_g^Arce)` per overlapping gene.
- **Null/control:** also compute cosines for *mismatched* pairs (`s_g^CZI` vs `s_h^Arce`, g≠h). Matched-gene cosines should be substantially higher than mismatched if transfer is real. Report both distributions.
- Where state is comparable, match it (CZI-Stim48h-g vs Arce-stimulated-g).
- Report per-model: distribution of matched cosines, distribution of mismatched cosines, and the separation between them. **This is where PCA is expected to do worst** (its CZI-variance directions may not align across datasets) and a learned model could win.

**(Retrieval transfer is dropped** — the in-distribution retrieval benchmark (Task 3) was non-discriminative across all models including baselines, so it is not run on external data.)

### Step 6 — Batch-integration diagnostic (context for interpretation)
On the combined CZI + Arce cells (embedded through each frozen model), measure whether cells of the same type/state from the two datasets mix or separate by dataset. Use standard integration metrics (e.g., iLISI/kBET-style, or simple kNN-dataset-purity). This contextualizes the transfer scores: if a model separates the two datasets hard by batch, its transfer degradation is partly domain shift, not biology. Report per model.

---

## Interpretation guide (state these framings explicitly in the report)

- **Expect transfer scores below in-distribution scores for all models.** The result is not absolute scores but **relative degradation**: does a learned model degrade *less* than PCA from CZI→Arce?
- **The headline comparison:** in-distribution, PCA beats learned models on decode (~0.83 vs ~0.72). If on transfer the gap *narrows or reverses* — PCA drops more than Arm1/Arm2 — that supports "learned representations generalize better." If PCA still leads on transfer, the learned models have no clear advantage even here, which is a hard but important finding.
- **5b (agreement) is the cleanest signal** because it's relative (matched vs mismatched within each dataset's offsets partially cancels). Lead with it. "Matched-gene cross-dataset cosine ≫ mismatched, and the matched cosine is higher for [model] than for PCA" is the strongest possible transfer result.
- **The domain-shift confound is unavoidable.** Frame as "transfer under domain shift." A positive result is strong *because* it survived the shift; a negative result is ambiguous (could be domain, could be biology) — say so.
- **Do not optimize toward a desired outcome.** Report what the numbers show. A finding that learned models do NOT beat PCA on transfer is publishable and important.

## Reporting

Produce a report with:
1. Gene-space overlap fraction (the gate) and external cell counts per (type × state × perturbation); external control count used for decoder.
2. The CZI∩Arce overlapping perturbation-gene list (sizes the agreement test).
3. **5a:** decode-transfer Δ-Pearson / p@k per model, effect-size stratified, vs in-distribution scores (show the degradation) and vs PCA/scVI.
4. **5b (headline):** per-model matched vs mismatched cross-dataset cosine distributions, and the matched-cosine ranking across models.
5. **Step 6:** batch-integration diagnostic per model.
6. A short verdict: does any learned model degrade less than PCA / show higher cross-dataset agreement than PCA? State plainly, including if the answer is no.

## Behavioral guidance
- Evaluate all five models identically; never advantage one.
- Frozen everything; do not refit PCA or any encoder on external data (Step 3).
- MLP decode = per-cell aggregation; never average-embedding-then-decode for MLP.
- Report the overlap gate (Step 2) before interpreting any score.
- No invented results; if the external data lacks something (too few controls, poor gene overlap), report and stop rather than work around silently.
- Lead the writeup with 5b (agreement) and the relative-degradation framing, not absolute scores.

## References
- Arce et al. 2024, Nature 637:930–939. Data: GEO GSE271090; Zenodo 10.5281/zenodo.13924126.
- In-distribution results (Tasks 1–4): prior project report. PCA ridge ≈0.83, Arm1 ≈0.716, Arm2 ≈0.700, Exp1 ≈0.591, scVI ≈0.502.
