# Results policy and provenance

`results/` is the local output root for training runs, checkpoints, metrics, Optuna studies, attention exports, and reproduced baselines. Generated artifacts are intentionally ignored by Git because they are large, numerous, and often contain machine-specific paths.

The headline TxT values reported in the main README are descriptive repeated-split summaries for the selected three-layer model:

| Task | Accuracy | Macro-F1 | ROC-AUC |
|---|---:|---:|---:|
| AD vs MCI | 0.632 | 0.599 | 0.661 |
| AD vs control | 0.740 | 0.736 | 0.843 |
| MCI vs control | 0.740 | 0.734 | 0.803 |

Interpretation constraints:

- validation selects checkpoints; test reports the final split metrics;
- repeated test partitions overlap, so variation across seeds is descriptive;
- the reproduced literature configurations were screened by mean shared-test ROC-AUC and are therefore exploratory upper envelopes, not untouched confirmatory estimates;
- attention-derived gene priorities are hypotheses, not validated biomarkers.

To regenerate outputs, follow the commands in the root README. Before publishing a release, export a compact machine-readable summary together with its exact command, Git commit, environment, split seeds, and selection rule.
