# Five-paper shared-test benchmark

This runner is a paper-aligned **local adaptation**, not a claim that the
published numerical results can be regenerated on a different cohort. It runs
the paper-aligned model/feature branches selected for this comparison on the
three TxT pairwise tasks, deliberately excluding direct all-gene inputs, while
keeping the test subjects fixed across methods.

## Common protocol

- tasks: `ad_vs_mci`, `ad_vs_ctl`, `mci_vs_ctl`;
- shared test only;
- 10 repeats, seeds 101 through 110;
- 70/10/20 train/validation/test membership generated once on the shared
  three-class cohort with the exact TxT splitting code, then filtered by
  `sample_id` for each binary task;
- preprocessing, learned representations, feature selection, resampling and
  augmentation are fitted on training data only;
- direct all-gene/no-feature-selection inputs are excluded from this benchmark;
- validation is available for tuning/checkpointing; the held-out test is not
  used to pick a model/feature configuration;
- all configurations are retained.  Do not select the displayed “best SOTA”
  by comparing shared-test means.

The orchestrator refuses a pre-existing split manifest whose membership does
not exactly match the canonical TxT seed.  It also records a hashed command
plan and refuses to mix a different configuration into the same paper output
directory. Wrapper-only metadata changes are accepted when commands, paper
code, data, splits and dependencies are unchanged. Completed runs are
resumable through each runner's `--skip-existing` behavior.

## Matrices

| Paper | Requested paper-aligned matrix per task/repeat |
|---|---|
| Lee & Lee 2020 | DEG, VAE, TF, Hub and CFG × LR, L1-LR, RBF-SVM, RF and DNN (25) |
| Kelly et al. 2023 | knowledge, VSSRFE+LR, LASSO and VAE latent × LR, RBF-SVM, XGBoost, RF and MLP (20); direct all-gene feature sets and standalone all-gene CNN/VAE classifier are excluded |
| One2MFusion 2023 | LASSO-image CNN and LASSO two-branch One2MFusion (2); standalone all-gene FNN is excluded |
| Diagnostics 2025 | XGBoost top-300 → fixed-95 SFBS × DL, SVM, GBM and RF × no-SMOTE/Borderline-SMOTE (8) |
| Hariharan 2026 | unaugmented Chi2/ANOVA/RFE/ElasticNet × six classifiers; augmented Chi2/ANOVA/RFE/ElasticNet × six classifiers; plus the unaugmented six-selector DNN comparison at k=500; no-FS/all-genes is excluded |

“Requested” is deliberate: the safe default records Lee Hub and CFG, and
Kelly knowledge genes, as skipped when their required independent external
inputs do not exist.  They are not silently replaced and are not included in
rankings. With an HPRD edge file and Kelly curated list supplied, the complete
selected non-all-gene matrices execute. CFG remains excluded from the primary safe matrix
because the available supplement proxy was derived using outcomes from the
same GSE cohorts.

Kelly uses the fixed optimized hyperparameters reported by the paper/public
implementation, including the 159-gene VSSRFE panel, instead of repeating the
100/200-iteration Bayesian searches inside every one of the 30 local splits.
For AD-vs-MCI and MCI-vs-control this is explicitly recorded as a transferred
AD profile rather than task-specific tuning.

Kelly's 169-million-parameter VAE is executed by an isolated PyTorch CUDA
worker on this Windows setup. Its architecture, objective, optimizer, learning
rate, batch size and early stopping match the selected paper profile; manifests
record the framework adaptation explicitly. Isolation also avoids native DLL
conflicts caused by loading TensorFlow/scikit-learn before PyTorch.

## Reproduction commands

Run from the repository root:

```powershell
python SOTA\source\run_shared_paper_benchmark.py --paper lee-2020
python SOTA\source\run_shared_paper_benchmark.py --paper kelly-2023
python SOTA\source\run_shared_paper_benchmark.py --paper one2mfusion-2023
python SOTA\source\run_shared_paper_benchmark.py --paper diagnostics-2025
python SOTA\source\run_shared_paper_benchmark.py --paper hariharan-2026
```

Use `--dry-run` to inspect every subprocess, `--smoke` for a one-seed reduced
integration check, and `--overwrite-paper-root` only when intentionally
restarting that paper.  `--n-jobs` controls classical tuning.  XGBoost's device
can be selected with `--xgboost-device cpu|cuda|auto`.

Optional external inputs:

- Lee Hub genes require a versioned HPRD interaction edge file via
  `--hprd-network-file`.  Without it, Hub is recorded as skipped rather than
  silently approximated.
- Lee CFG can only be run as a separate, explicitly leaky sensitivity analysis
  with `--allow-cfg-supplement-proxy`; it is off in the five primary commands.
- Kelly's exact curated knowledge-gene union was not included in the public
  repository.  A recovered list can be supplied with
  `--kelly-knowledge-genes-file`; otherwise the reproducible train-only top-3000
  MAD component is recorded but skipped.  `--allow-incomplete-kelly-knowledge`
  runs it only as the separately labelled `mad_top3000_only` ablation.

Hariharan CTGAN uses PyTorch CUDA automatically when it is available.  Override
with `--ctgan-device cpu` or require it with `--ctgan-device cuda`.  The full
command is intentionally compute-heavy (120 CTGAN fits across 3 tasks × 10
seeds × 4 feature selectors), but is resumable at individual run level.

## Fidelity limits that remain explicit

- Lee & Lee was originally an AD-vs-control study.  Its VAE details permit a
  faithful reconstruction, but its latent-size rule is not deterministic for
  a new split.  CFG supplement tables are an outcome-derived proxy and require
  an explicit opt-in in the underlying command.  The processed HPRD network is
  not supplied by the paper.
- Kelly did not study AD-vs-MCI or MCI-vs-control.  The public code conflicts
  with the text on VAE learning rate and contains no exact curated knowledge
  list; both are recorded in manifests.
- One2MFusion originally included GSE140829 as a third cohort.  This benchmark
  intentionally uses the same two-batch TxT cohort.
- Diagnostics publishes the XGBoost→SFBS95 outline but not the SFBS estimator,
  scorer, complete gene list, gene-expression DL architecture, or model
  hyperparameters.  The implementation therefore labels these choices as
  assumptions and disables silent fallback.
- Hariharan does not publish code, its chi-square bin count and post-CTGAN noise
  filter are unspecified, and its protocol descriptions conflict.  Real CTGAN
  is used train-only; omitted filtering is recorded rather than invented.

These limitations are why result captions should use wording such as
“paper-aligned leakage-safe adaptation on the TxT cohort” rather than “exact
numeric reproduction”.
