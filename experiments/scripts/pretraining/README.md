# TxT pretraining scripts

This folder contains only the pretraining entry points and registries.

Main files:

```text
pretrain_txt_gexbert.py                     masked transcriptomic pretraining
build_ppi_embedding.py                      HIPPIE PPI node2vec gene embedding init
run_pretraining.ps1                         main parametrized pretraining launcher
run_ppi_init.ps1                            HIPPIE PPI node2vec launcher
check_pretraining_matrix.py                 matrix sanity checks
aggregate_pretraining_experiment_results.py aggregate pretraining outputs
pretraining_checkpoint_manifest.csv         checkpoint registry
ppi_embedding_manifest.csv                  PPI embedding registry
txt_2l2h_pretraining_grid.csv               planned P0-P5 experiment grid
```

Fine-tuning scripts live in:

```text
experiments/scripts/finetuning/
```

Post-hoc plotting and summary scripts live in:

```text
experiments/scripts/summary/
```

## HIPPIE PPI embedding init

This path is separate from masked transcriptomic restoration pretraining. It follows the TxT paper's embedding idea: initialize gene embeddings from node2vec on a protein-protein interaction network, then train/fine-tune TxT with that embedding file.

Default run:

```powershell
.\experiments\scripts\pretraining\run_ppi_init.ps1
```

By default the launcher downloads HIPPIE current MITAB, filters human interactions at confidence score `>= 0.73`, maps the embedding to all genes in `task_dataset/processed/txt_pairwise_multitask/shared_ad_mci_ctl/X.csv`, and writes the deduplicated undirected network to:

```text
pretraining_dataset/ppi_networks/hippie_highconf_edges.csv
```

and writes TxT-ready embeddings to:

```text
results/pretraining/ppi_init/hippie_highconf_dim128_score0p73_seed42/gene_embedding.csv
```

For the final multitask pipeline, prefer passing `ppi_node_embedding.csv` with `--embedding-gene-policy mapped_only` so feature selection is restricted to genes present in the PPI graph, matching the TxT supplementary setup:

```powershell
python experiments\scripts\paper_comparison\run_txt_multitask_cv_and_seeds.py `
  --x-file task_dataset\processed\txt_pairwise_multitask\shared_ad_mci_ctl\X.csv `
  --y-file task_dataset\processed\txt_pairwise_multitask\shared_ad_mci_ctl\y.csv `
  --embed-file results\pretraining\ppi_init\hippie_highconf_dim128_score0p73_seed42\ppi_node_embedding.csv `
  --embedding-gene-policy mapped_only `
  --gene-selection mad `
  --max-genes 1000 `
  --d-model 128 --d-ff 512 `
  --augmentation borderline_smote
```

The run writes `ppi_embedding_report.json`, including graph size, split size, link-prediction sanity metrics, target-gene coverage, and genes initialized randomly because they were missing from HIPPIE.
