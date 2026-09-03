# Contributing

This repository is research software under active development. Focused bug reports, reproducibility findings, and small, well-tested improvements are welcome.

## Before opening a change

1. Create an isolated Python 3.10–3.12 environment.
2. Install `requirements.txt`.
3. Keep downloaded data and generated experiment artifacts out of Git.
4. Add or update a data-free test for behavioural changes.
5. Run the verification commands below.

```bash
python -m compileall -q source experiments/scripts SOTA/source
python -m pytest -q tests
python experiments/scripts/paper_comparison/run_txt_multitask_final_pipeline.py \
  --preset balanced --dry-run
```

## Reproducibility expectations

Changes to experiments should document the exact command, split seed, data-preparation mode, feature-selection scope, scaling scope, checkpoint rule, and output schema. Feature selection, scaling, and augmentation must never learn from validation or test data in the main protocol.

## Data and licensing

Do not commit raw/processed participant-level matrices, trained checkpoints, downloaded papers, or third-party repositories. Before contributing code, review [NOTICE.md](NOTICE.md); no open-source licence has yet been granted for the repository.
