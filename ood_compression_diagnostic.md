# Agent Task: Diagnose OOD Compression — Why the JEPA Encoders Squash External Data

## Context (read first — no prior project context assumed)

We trained transformer encoders ("JEPA") on the CZI CD4⁺ T-cell Perturb-seq dataset and evaluated cross-dataset transfer to an external dataset (Arce et al., a different-lab CD4 T-cell Perturb-seq). A follow-up analysis surfaced a striking failure mode that we now want to understand:

**When the frozen JEPA encoders are applied to the external (Arce) cells, they compress the ENTIRE external dataset into a tiny degenerate region of latent space.** Measured as cell-to-cell scale (typical pairwise spread of cell embeddings):

| model | Arce cell-to-cell scale | notes |
|---|---|---|
| arm1 (ESM2 tokens, VICReg+EMA) | **0.65** | severe compression |
| arm2 (ESM2 tokens, SIGReg) | **0.60** | severe compression |
| exp1 (random-init `nn.Embedding` tokens, VICReg+EMA) | 13.49 | mild |
| pca-50 | 16.37 | none |
| pca-256 | 24.20 | none |
| scvi-10 | 2.29 | mild |

So the two **ESM2-tokenized** arms squash Arce ~20× more than Exp1 (which differs mainly by using random-init gene-ID embeddings instead of ESM2) and far more than the linear baselines. On *in-distribution* CZI data these same encoders spread cells normally — the compression is specific to out-of-distribution input. This looks like the encoder overfitting the CZI input manifold: anything off that manifold collapses to a corner.

**Goal:** identify *why* the ESM2-tokenized arms compress OOD data, and *where* in the network it happens, so we know whether it's fixable (and how). This is diagnostic, not a fix — characterize the mechanism. Run on frozen models and cached/available data; retrain nothing.

## Leading hypotheses to test (ranked by suspicion)

- **H1 — ESM2 tokenization is the cause.** Exp1 (no ESM2) compresses far less. The frozen ESM2 protein embeddings may anchor the representation tightly to CZI-specific expression-value combinations, so OOD inputs fall outside the learned token-interaction regime and collapse. *This is the prime suspect — it's the main axis distinguishing the compressing arms from the non-compressing Exp1.*
- **H2 — Zero-fill OOD input drives saturation.** Arce had 24.7% of the 2000 HVGs zero-filled (genes absent/renamed in Arce). Feeding many artificial zeros may push the value-embedding or attention into a saturated regime that collapses outputs. Confound with H1 because all models got the same zero-fill, but the *effect* of zero-fill may interact with ESM2 tokens specifically.
- **H3 — Normalization/scale mismatch at inference.** Arce's expression distribution (depth, library size, log-normalization constants) may differ from CZI, pushing the value encoder out of its trained input range.
- **H4 — A specific layer collapses.** The compression might originate at a specific point (value embedding, first attention layer, final LayerNorm, CLS pooling) rather than being distributed. LayerNorm in particular can map diverse inputs to similar outputs if upstream activations saturate.

## Analyses

### Analysis 1 — Localize WHERE the collapse happens (layer-by-layer)
For arm1 (and arm2), run a sample of Arce cells AND a matched sample of CZI cells through the frozen encoder, capturing intermediate activations at each stage:
- input tokens (after ESM2-identity + value embedding + sum)
- after each transformer layer
- the final CLS embedding

At each stage, compute the **cell-to-cell scale** (mean pairwise distance, or total variance / effective rank of the activation covariance) for the Arce batch and the CZI batch separately. Report the **ratio Arce-scale / CZI-scale per layer.**

**Interpretation:** find the layer where the Arce/CZI scale ratio first crashes. If it crashes at the input tokens → the problem is tokenization (H1/H2/H3). If tokens are fine but it crashes after a transformer layer or the final LayerNorm → the collapse is in the encoder dynamics, not the input. This single analysis tells you whether to look at the tokenizer or the transformer.

### Analysis 2 — Decompose the input token (isolate ESM2 vs value vs zero-fill)
The token = ESM2-identity-projection ⊕ value-embedding (and the arms sum/concat these). For an Arce batch and a CZI batch, separately compute the cell-to-cell scale of:
- the ESM2-identity component alone (this is *gene*-dependent, not cell-dependent — should be ~identical CZI vs Arce since it's the same gene vocab; if it differs, the zero-fill is changing which gene tokens are present)
- the value-embedding component alone (this is the cell-specific part — the expression values)
- the summed/assembled token

**Interpretation:** if the value-embedding component collapses for Arce but the ESM2 component is fine, the problem is the expression-value pathway on OOD inputs (H3 — scale/normalization), not ESM2 per se. If the assembled token collapses only when ESM2 is included (compare to a value-only forward pass), ESM2 is implicated (H1).

### Analysis 3 — The decisive ESM2 ablation (H1)
Test H1 directly by swapping the tokenization at inference (no retraining):
- **3a (ESM2-off proxy):** if feasible, run arm1's forward pass but replace the ESM2-identity tokens with Exp1-style identity tokens (or zero them), keeping everything else. Does Arce compression relax toward Exp1's level? This isolates whether ESM2 *tokens* cause the compression. (If the architectures differ too much to swap cleanly, skip to 3b.)
- **3b (the natural ablation):** Exp1 IS the no-ESM2 model and compresses far less. Strengthen this comparison: run Exp1 and arm1 on the *same* Arce cells and the *same* CZI cells, confirm the compression-ratio gap, and check whether it tracks specifically with ESM2 presence vs other arm1/exp1 differences (tokenization is the main one, but confirm training config, depth, dim are otherwise comparable so the comparison is clean). Report the confounds.

**Interpretation:** if ESM2-off relaxes the compression (3a) or the Exp1-vs-arm1 gap is cleanly attributable to ESM2 (3b), H1 is supported.

### Analysis 4 — Zero-fill stress test (H2)
Isolate the zero-fill effect on a controlled input:
- Take a batch of **CZI** cells (in-distribution, normally spread by the encoder). Artificially zero-fill the same 494 HVGs that were zero-filled for Arce. Re-embed.
- Measure: does artificially zero-filling CZI cells (otherwise in-distribution) cause them to compress the way Arce did?

**Interpretation:** if zero-filled-CZI compresses like Arce, the zero-fill (H2) is a major driver, and the OOD-transfer conclusion is partly entangled with the input-alignment artifact (a fixable preprocessing issue). If zero-filled-CZI stays spread, zero-fill is not the cause and the compression is a genuine OOD-manifold property (H1/H3).

### Analysis 5 — Input-distribution mismatch (H3)
Compare the *input* (post-normalization, pre-encoder) expression distributions of CZI vs Arce on the overlapping genes:
- per-gene mean/std, library-size/depth distribution, fraction zeros, dynamic range.
- Quantify how far Arce's input distribution sits from CZI's (e.g., per-gene z-shift, or a simple 2-sample distance).

**Interpretation:** if Arce's input distribution is far outside CZI's trained range (e.g., systematically different depth or scale), the value encoder is being asked to extrapolate, which can saturate — supporting H3 and pointing to an inference-time normalization fix (e.g., quantile-align Arce to CZI before encoding).

## Reporting

1. **Verdict up front:** which hypothesis (or combination) explains the compression — ESM2 tokenization (H1), zero-fill (H2), normalization mismatch (H3) — and *where* it happens (H4 / Analysis 1).
2. **Analysis 1:** the layer-by-layer Arce/CZI scale-ratio curve; name the layer where collapse onsets.
3. **Analysis 2:** ESM2-component vs value-component scale, CZI vs Arce.
4. **Analysis 3:** ESM2 ablation / Exp1-vs-arm1 attribution result.
5. **Analysis 4:** does zero-filling CZI reproduce the compression? (the cleanest single control)
6. **Analysis 5:** CZI-vs-Arce input-distribution distance.
7. **Implications for a fix:** based on the mechanism, what would plausibly help — e.g. if H2/H3: better gene-alignment / quantile normalization at inference; if H1: less CZI-specific tokenization or an OOD-robustness regularizer; if H4 names a layer: targeted intervention there. State which are inference-time fixes (cheap) vs require retraining.

## Behavioral guidance
- Frozen models throughout; retrain nothing.
- Run the same probe batches (same Arce cells, same CZI cells) across analyses so numbers are comparable.
- Analyses 1 and 4 are the highest-value (where it collapses, and whether zero-fill causes it) — prioritize them.
- Keep CZI as the in-distribution reference in every comparison so "compression" is always Arce-relative-to-CZI, not absolute.
- Report confounds honestly, especially the Exp1-vs-arm1 comparison (confirm they differ mainly in ESM2 tokenization and not in other config).
- No invented results; if an intermediate activation can't be captured cleanly, report the limitation.

## Inputs available
- Frozen models: exp1, arm1, arm2 (+ pca/scvi for reference, though the layer analyses apply only to the JEPA encoders).
- Cached Task-5 embeddings and the aligned/zero-filled Arce input matrix: `runs/task5/<model>/task5.npz`, `data/external/arce/`.
- CZI fit-pool cells and the 2000-HVG vocab + the 494 zero-filled gene indices.
