# Lee & Lee 2020 reproduction/adaptation

Full-grid adaptation of Lee & Lee 2020, *Prediction of Alzheimer's disease using blood gene expression data*, for any local binary TxT task. The paper trained classifiers for AD vs cognitively normal; AD vs MCI and MCI vs CTL must be reported as benchmark extensions.

## Paper-oriented defaults

- shared test only, 10 repeats;
- `train_inner / val_inner / outer_test`, with `inner_val_ratio=0.125` and `shared_test_size=0.20` (total 70/10/20);
- train-only imputation, scaling, limma DEG and VAE fitting;
- limma FDR `<0.01`, no nominal-p override and no top-k fallback;
- all five feature families: DEG, VAE, DEG∩TF, DEG∩CFG and DEG∩HPRD hubs;
- all five classifiers: LR, L1-LR, SVM RBF, RF and DNN;
- no implicit class weighting (`--class-weighting balanced` is an explicit ablation);
- fixed 0.5 classification threshold.

`--strict-v2` is retained for compatibility and now keeps the complete 5×5 grid. Existing `--split-manifest-dir` support remains the preferred way to reuse the exact TxT shared-test membership.

## Feature-set provenance

- `deg`: limma FDR `<0.01` on `train_inner`.
- `tf_genes`: train-only DEG intersected with the 608-gene TRANSFAC list in `MOESM2`.
- `vae`: genuine VAE with mean/log-variance, reparameterisation and KL loss; ELU encoder, softplus scale, tanh decoder, Adagrad at 0.001, followed by supervised train-only fine-tuning. Full-batch training makes each configured epoch one optimizer update. The default 100/200/300 latent-size rule is explicitly stored as a reconstruction because the paper only says the dimension was chosen similarly to the CFG count. Use `--vae-latent-policy fixed --vae-latent-dim N` for a declared exact size.
- `cfg_genes`: disabled unless `--allow-cfg-supplement-proxy` is supplied. `MOESM3-5` list only outcome-derived DEG rows from the complete original datasets, so this proxy is not independent of holdouts drawn from those GSE cohorts.
- `hub_genes`: requires `--hprd-network-file PATH`; nothing is downloaded automatically. The common HPRD Release 9 flat-file layout (gene columns 0 and 3) and simple two-column edge lists are supported. Hubs use unique undirected degree `>10`; path, SHA-256, columns, edge/node counts and threshold are persisted.

## Run

```powershell
python SOTA\source\nature-2020\run_experiments.py `
  --x-file <task-X.csv> `
  --y-file <task-y.csv> `
  --split-manifest-dir <task-split-manifests> `
  --class-names <label-0> <label-1> `
  --result-root <output-directory> `
  --strict-v2
```

To opt in to every conditional reconstruction, add:

```powershell
--allow-cfg-supplement-proxy --hprd-network-file <local-HPRD-edge-file>
```

Smoke run:

```powershell
python SOTA\source\nature-2020\run_experiments.py --smoke --result-root results\SOTA\_smoke_nature-2020
```

Targeted tests:

```powershell
python -m unittest discover -s SOTA\source\nature-2020\tests -v
```
