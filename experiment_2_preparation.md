# Experiment 2 — Preparation Notes

Planning document for the second JEPA training run. Nothing here is finalized — options are listed so we can revisit and decide before implementation.

**Baseline:** [EXPERIMENT_1_REPORT.md](EXPERIMENT_1_REPORT.md)  
**Date drafted:** 6 Jun 2026

---

## Overview of proposed changes

| Area | Experiment 1 | Experiment 2 (proposed) |
|---|---|---|
| HVG count | 2,000 | **4,000** (decided direction) |
| HVG selection strategy | `seurat_v3`, `batch_key=10xrun_id` on NTC cells | **TBD** — see §1 |
| Predictor context | Context tokens + target gene-embedding queries only | **TBD** — add GRN prior via GENIE3; see §2 |

---

## 1. HVG selection — 2,000 → 4,000

### What Experiment 1 actually did

HVG selection runs in `scripts/prep_data.py` during the two-pass concat:

1. **Pass 1:** Concatenate NTC cells only → call `select_hvgs()` (`src/jepa_poc/data/hvg.py`).
2. **Pass 2:** Subset all slim shards to those genes → write processed datasets.

Relevant code path:

```python
# prep_data.py — _hvg_concat()
hvg_genes = select_hvgs(ntc_concat, n_top_genes=n_top_genes, batch_key=batch_key)
```

Config (`configs/poc.yaml`):

```yaml
hvg_count: 2000
batch_key: 10xrun_id
```

`select_hvgs()` uses `seurat_v3` on **raw counts** with an optional `batch_key`. When set, Scanpy computes variance-stabilized normalized variance **per batch**, ranks genes within each batch, then aggregates across batches (genes that are HV in more batches and with better median rank win).

### Clarification: within-state vs across-state variance

`10xrun_id` is injected per shard from `(donor, culture_condition)` via `sample_metadata.suppl_table.csv`. There are **12 batches = 4 donors × 3 conditions** (Rest, Stim8hr, Stim48hr).

So Experiment 1 HVGs are genes that are highly variable **within each donor×condition group**, aggregated across groups. The selection explicitly **controls out**:

- The rest → 8h → 48h stimulation axis (treated as batch)
- Donor-to-donor variation (also conflated with batch, since each donor×condition is its own run)

**Answer to the open question:** We did **not** select genes by across-state variance. We selected genes variable **within** single states (per donor), then pooled rankings.

This is worth noting because the annotation eval already hit ~99% F1 — stimulation is such a strong signal that even within-state-variable genes still encode condition well.

### Proposed change: 4,000 HVGs

**Config change (minimal):**

```yaml
hvg_count: 4000
```

**Expected effects:**

| Aspect | Notes |
|---|---|
| Gene vocabulary | `gene_vocab.tsv` grows 2×; gene embedding table becomes 4000×256 |
| Compute | Target encoder runs a full forward over all genes every step; transformer attention is O(n²) in token count. Rough estimate: **~2–4× longer training** vs Experiment 1 (~6.8 days → potentially 2+ weeks at same batch size) |
| VRAM | A6000 (48 GB) may need batch-size reduction or gradient checkpointing at batch 512 |
| Masking | No change needed — `mask_context_frac` and `target_block_frac` are fractional (30% context, 20% per target block) |
| Signal vs noise | Genes ranked 2000–4000 are lower-expressed and noisier; usually mild benefit for representation learning |
| GRN (§2) | More genes → better regulatory coverage for GENIE3, especially if we restrict regulators to TFs |

### HVG batch-key options (not decided)

The batch key choice is arguably more important than the count change. Options to revisit:

#### Option A — Keep `batch_key: 10xrun_id` (status quo)

- **Pros:** Reproducible comparison to Experiment 1 except for count; controls both donor and condition effects.
- **Cons:** May under-represent stimulation-program genes (the main biological axis in this dataset); same confound as Experiment 1.

#### Option B — `batch_key: donor_id` (recommended to discuss)

- **Pros:** Controls donor/technical variation while **preserving across-state variance within each donor**. Stimulation-responsive genes can rank fairly. Best match for a model meant to capture activation dynamics and for per-state GRNs (§2).
- **Cons:** Not directly comparable to Experiment 1 HVG set; donor batch effects may still influence ranking (less than with no batch key).

#### Option C — No batch key (global HVG on pooled NTC)

- **Pros:** Simplest; naturally prioritizes the largest variance sources (likely rest→stim transitions).
- **Cons:** HVG list may be dominated by condition-separation genes; subtle perturbation-relevant genes and donor-specific noise both compete for slots.

#### Option D — Per-condition HVG, then union or intersect

- e.g. top 1500 HVGs per state (Rest / Stim8hr / Stim48hr), take union → cap at 4000.
- **Pros:** Guarantees representation of state-specific biology in the vocabulary.
- **Cons:** More complex pipeline; union can exceed 4000 (need cap rule); intersect may be too small.

#### Option E — Separate HVG pools for different purposes

- e.g. 4000 for training, but evaluate literature/TF enrichment on a different gene set.
- **Pros:** Decouples eval from training vocab.
- **Cons:** Extra complexity; usually unnecessary if Option B or C is chosen well.

### Open questions (HVG)

- [ ] Confirm `hvg_count: 4000` for Experiment 2.
- [ ] Choose `batch_key`: `10xrun_id` | `donor_id` | `None` | per-condition union.
- [ ] Re-run `prep_data.py` and verify peak RAM still fits 251 GB budget (4000 genes × ~3.4M cells sparse should be fine).
- [ ] Smoke-test one training step at batch 512 before committing to a multi-week run.
- [ ] Document chosen strategy in `EXPERIMENT_2_REPORT.md` when we write it.

---

## 2. GRN as additional predictor context (GENIE3)

### Motivation

Experiment 1 predictor (`src/jepa_poc/models/predictor.py`) takes:

1. Context encoder tokens (masked genes)
2. Target gene-embedding queries (`gene_embedding[target_idx] + query_type`)

…and predicts target encoder tokens via a 2-layer transformer. No explicit regulatory structure.

**Hypothesis for Experiment 2:** Providing a gene regulatory network (GRN) as structural context helps the predictor reconstruct the target latent space — especially for perturbation-relevant genes and held-out knockout generalization (Experiment 1 Part B was weak: overall delta_pearson 0.023).

**Tool:** [GENIE3](https://github.com/vahuynh/GENIE3) — tree-based GRN inference from expression data. Python implementation uses scikit-learn; fits one regressor per target gene.

### Data for GRN construction

- **Source cells:** NTC only (same pool as HVG selection — avoids perturbation-induced edges confounding the network).
- **States:** Rest, Stim8hr, Stim48hr (`culture_condition` in `.obs`).
- **Expression:** Log-CPM normalized (match training loader) or raw counts (GENIE3 examples often use log-transformed expression). **Decide and keep consistent.**

### Per-state (“3-layer”) GRN — idea and tradeoffs

**Idea:** Build **three separate GRNs**, one per culture condition, each representing regulatory wiring at that activation state. Biologically plausible — T cell programs rewire substantially across rest → early stim → late stim.

**Architecture implication:** Training batches are shuffled and **mix cells from all states**. Any per-state structure must be **conditioned per cell**, not assumed global.

| Approach | Per-state GRN? | Batch-mixed training? | Complexity |
|---|---|---|---|
| Single global GRN | No | Yes | Low |
| 3 GRNs + per-cell conditioning | Yes | Yes | Medium |
| 3 GRNs + state-stratified batches | Yes | No (one state per batch) | Low–medium |
| 3 GRNs + ensemble / learned mixture | Yes | Yes | Higher |

**Recommendation to revisit:** Stage the experiment — **v1 = one global GRN** (does any GRN prior help?), then **v2 = 3 per-state GRNs** (does state-specific wiring add value on top?).

### GENIE3 practical notes

**Compute:**

- GENIE3 fits one random forest per **target** gene, regressing on candidate **regulators**.
- Naive: 4000 targets × 4000 regulators × 3 states is expensive.
- **Mitigation:** Restrict `regulators` to **transcription factors** (Lambert et al. catalog — ~181 TFs already overlap Experiment 1 HVG vocab). GENIE3 supports a `regulators=` argument → ~20× cost reduction and biologically appropriate TF→target edges.
- **Subsample** NTC cells per state to ~20k–50k for GRN fitting; GENIE3 typically saturates before full 500k NTC.

**Output format:**

- Weight matrix `W[regulator, target]` (unsigned importance scores, not signed interaction).
- Post-process: top-k edges per target, row-normalize, or threshold → sparse adjacency for the model.

**Storage (proposed):**

```
data/processed/grn/
  genie3_global.npz          # or .tsv edge list
  genie3_rest.npz
  genie3_stim8hr.npz
  genie3_stim48hr.npz
  grn_metadata.json          # params, gene order, TF list, cell counts
```

### How to inject GRN into the predictor (not decided)

Current forward (`predictor.py`):

```python
queries = target_gene_embeddings + query_type
tokens = cat([context_tokens, queries])
out = transformer(tokens)
pred = out[:, -n_targets:, :]
```

Options to revisit:

#### Option 1 — Graph-biased attention (recommended starting point)

Add bias `B[i,j]` to attention logits from GRN edge weight between genes for tokens `i` and `j`. CLS token gets zero or learnable bias.

- **Pros:** Directly encodes “to predict target gene *g*, attend to its regulators in context.” Parameter-efficient.
- **Cons:** `nn.TransformerEncoderLayer` does not accept per-sample attention bias out of the box → likely need custom attention or `F.scaled_dot_product_attention` with `attn_mask`. With batch-mixed states, bias matrix may differ per cell unless we use state-stratified batches or a single global GRN.

**Sub-options:**

- 1a. Global GRN → one bias matrix per batch (same for all cells in batch).
- 1b. Per-state GRN → bias matrix indexed by `culture_condition` per cell (custom per-sample attention).
- 1c. Per-state GRN + state-stratified DataLoader → one bias per batch, simpler implementation.

#### Option 2 — GNN refinement of gene embeddings

Run GCN/GAT message passing on the full gene-embedding table using GRN adjacency **before** the transformer. For per-state GRNs: three refined embedding tables, gather by cell state.

- **Pros:** Clean per-cell conditioning without custom attention; gene embeddings become structure-aware.
- **Cons:** Extra module and hyperparameters (layers, dropout); GRN affects encoder path if applied globally — scope carefully (predictor-only vs shared gene embeddings).

#### Option 3 — Structural embeddings as additive prior

Precompute node embeddings from GRN (node2vec, Laplacian eigenmaps, or GENIE3 importance features). Add to `gene_embedding` or only to predictor queries.

- **Pros:** Simple gather-by-state; no change to attention kernel.
- **Cons:** Static features may be redundant with learnable embeddings; less expressive than Options 1–2.

#### Option 4 — GRN edges as extra tokens

Add regulator tokens (with edge-weight channel) to the predictor sequence.

- **Pros:** Very explicit.
- **Cons:** Sequence length blow-up; likely slower than bias or GNN.

### Suggested ablation ladder (when we implement)

1. **Baseline:** Experiment 2 without GRN (HVG changes only) — isolates HVG effect.
2. **+ Global GRN** (Option 1a or 3) — does any structural prior help?
3. **+ Per-state GRN** (Option 1b/1c or 2) — does state-specific wiring help?
4. **+ Shuffled GRN control** — same degree/weight distribution, random edges. **Critical:** separates biological signal from “extra inductive bias” alone.

### Evaluation criteria (unchanged suite, new comparisons)

Run the same eval as Experiment 1 and compare:

| Metric | Experiment 1 (reference) | Experiment 2 target |
|---|---|---|
| Annotation linear macro-F1 | 0.993 | Likely saturated; not primary success metric |
| Perturbation repr-quality delta_pearson (overall) | **0.591** | Primary — latent space quality |
| Perturbation head delta_pearson (overall) | 0.023 | Secondary — generalization to held-out genes |
| Gene-agnostic baseline | 0.286 | Should remain well below repr-quality |
| TF literature validation | 4/19 strong | Exploratory — embedding neighborhoods |

**Honest expectation:** Experiment 1 already recovered real TF biology from attention alone (EGR2/3, FOXP3, FOXP1). GRN gains may be **modest** and show up most in Part B (held-out gene prediction), not annotation.

### Open questions (GRN)

- [ ] v1 global only, or go straight to 3 per-state networks?
- [ ] TF-only regulators vs all 4000 genes as regulators?
- [ ] Raw counts vs log-CPM for GENIE3 input?
- [ ] Cells per state for GRN fitting (subsample size)?
- [ ] Integration option: attention bias vs GNN vs structural embeddings?
- [ ] Predictor-only vs also conditioning context encoder?
- [ ] Top-k per target vs full weighted adjacency?
- [ ] Shuffled-GRN control included in same experiment or follow-up?

---

## 3. Implementation checklist (when decisions are made)

### Data / prep

- [ ] Update `configs/poc.yaml` (`hvg_count`, possibly `batch_key`)
- [ ] Re-run `scripts/prep_data.py` → new processed h5ad + `gene_vocab.tsv`
- [ ] New script: `scripts/build_grn.py` (GENIE3 on NTC, per-state + global)
- [ ] Save GRN artifacts under `data/processed/grn/`

### Model / training

- [ ] Extend `JEPAPredictor` (or sibling class) with chosen GRN mechanism
- [ ] Plumb `culture_condition` (or state id) from `TCellDataset` if per-state conditioning needed
- [ ] New config section e.g. `model.grn_path`, `model.grn_mode: none | global | per_state`
- [ ] Update `n_genes` everywhere from vocab length
- [ ] VRAM / throughput smoke test before long run

### Reporting

- [ ] `EXPERIMENT_2_REPORT.md` — document every decision locked above
- [ ] Side-by-side table vs Experiment 1 on all eval metrics

---

## 4. Decision log

Record final choices here when we lock them in.

| Decision | Choice | Date | Notes |
|---|---|---|---|
| HVG count | 4000 (leaning) | — | |
| HVG batch_key | TBD | — | |
| GRN scope | TBD | — | global vs per-state |
| GRN integration | TBD | — | attention bias vs GNN vs embeddings |
| GENIE3 regulators | TBD | — | TF-only recommended |
| Training budget | TBD | — | expect longer than Exp 1 |

---

## References

- Experiment 1 report: [EXPERIMENT_1_REPORT.md](EXPERIMENT_1_REPORT.md)
- GENIE3: [github.com/vahuynh/GENIE3](https://github.com/vahuynh/GENIE3)
- HVG code: `src/jepa_poc/data/hvg.py`, `scripts/prep_data.py`
- Predictor: `src/jepa_poc/models/predictor.py`
- JEPA forward: `src/jepa_poc/models/jepa.py`
