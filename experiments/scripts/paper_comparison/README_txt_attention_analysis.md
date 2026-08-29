# TxT post-training attention analysis

This workflow extracts post-softmax self-attention from a trained TxT checkpoint, keeps the query/key axes explicit, and produces gene-level rankings, class contrasts, full class-mean matrices and publication-ready plots without storing every subject's 2000 x 2000 matrix.

## Metrics

For dense attention `A[sample, head, query_gene, key_gene]`:

- **Key incoming enrichment:** `G * mean_over_queries(A[..., key])`. A value of 1 is uniform attention. This is the primary ranking of genes that receive attention.
- **Query specificity:** `1 - entropy(A[..., query, :]) / log(G)`. Query row sums are not used because the softmax makes every row sum equal to 1.
- **Q/K norms:** exported separately as geometric diagnostics; they are not attention weights.
- **Class contrasts:** OLS class coefficient adjusted for `dataset_gse`, with class-by-cohort interaction and BH correction within each metric/contrast family.

For sparse PPI-VMA attention, incoming source/key mass is divided by its opportunity under uniform attention within each target neighbourhood. Target/query specificity and the top-edge export exclude degree-1 targets, whose sole edge would otherwise have a structurally forced softmax weight of 1.

## Single-run extraction

```bash
MPLCONFIGDIR=/tmp/mplconfig python3 \
  experiments/scripts/paper_comparison/analyze_txt_attention.py \
  --run-dir results/paper_comparison/<run>/seed_<seed>/<model> \
  --split test \
  --device cpu \
  --batch-size 2 \
  --bootstrap-iterations 1000 \
  --top-genes 25 \
  --matrix-top-genes 48
```

The run directory must contain the exact `best_model.pt`, `args.json`, `selected_genes.csv`, `gene_embedding.csv` and split artifacts used during training. The script verifies the gene-token ordering before extraction.

Main outputs under `<run-dir>/attention_analysis/<split>/`:

- `gene_attention_ranking.csv`: global dense key and query ranks.
- `gene_attention_summary.csv`: class/head/gene summaries and bootstrap intervals.
- `gene_attention_contrasts.csv`: cohort-adjusted contrasts, effect sizes, p values, q values and cohort interactions.
- `class_mean_attention.npz`: complete class x head x query x key mean matrices and gene order.
- `sample_gene_attention_metrics.npz`: compact per-subject gene metrics, not full matrices.
- `dense_top_attention_edges.csv`: strongest dense query-to-key edges.
- `attention_qc.csv` and `attention_analysis_manifest.json`: numerical QC and provenance, including SHA-256 hashes for the checkpoint and analysis script.
- `vma_*`: degree-corrected sparse-attention outputs when a VMA branch is present.
- `plots/`: PNG and PDF figures.

## Multi-run consensus

After analysing all checkpoints, combine the exports with:

```bash
MPLCONFIGDIR=/tmp/mplconfig python3 \
  experiments/scripts/paper_comparison/summarize_txt_attention_runs.py \
  --analysis-dirs <analysis-dir-1> <analysis-dir-2> ... \
  --labels <label-1> <label-2> ... \
  --output-dir <consensus-output-dir>
```

The summariser uses percentile ranks, inclusion-adjusted consensus scores, top-k frequency, Spearman concordance and Jaccard top-k overlap. If different seeds select different top-2000 gene sets, the consensus score treats absence as zero percentile support while retaining the raw inclusion frequency and within-run percentile columns for inspection.

## Current local result

The only complete non-smoke post-training checkpoint currently available in this workspace is the exploratory seed-101 run:

```text
results/paper_comparison/txt_volumetric_mps_seed101_beta1p5_lossstop/
```

Both its baseline and beta-1.5 VMA checkpoint have been analysed on the held-out test split. The final 10-seed experiment directories currently contain only `args.json`, so this local output must not be described as a final multi-seed attention signature.

## Interpretation boundary

Attention-derived ranks are descriptive properties of this trained model. They are not, by themselves, causal feature importance, differential expression, biomarker validity or evidence of a disease mechanism. Confirm top genes with multi-seed stability and held-out perturbation/occlusion, and control blood analyses for expression distribution, age, sex, cohort and leukocyte composition.
