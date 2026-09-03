# Quick start

This guide gives three progressively more expensive ways to evaluate the project. Start with the data-free path: it proves that the repository is wired correctly without downloading biomedical data or launching training.

![Three-level quick-start](images/quickstart.svg)

## Level 1 — data-free validation

### 1. Create the environment

```bash
git clone https://github.com/tomgiorgini/transcriptome-transformer-ad-mci.git
cd transcriptome-transformer-ad-mci

python3 -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Python 3.10–3.12 is recommended. Training uses PyTorch 2.x.

### 2. Run the automated checks

```bash
python -m compileall -q source experiments/scripts SOTA/source
python -m pytest -q tests
```

The core test suite uses synthetic fixtures and does not require GEO matrices. Tests colocated under `SOTA/source/` require `requirements-optional.txt`.

### 3. Inspect the final experiment plan

```bash
python experiments/scripts/paper_comparison/run_txt_multitask_final_pipeline.py \
  --preset balanced --dry-run
```

The command writes a manifest under `results/` and prints the five planned stages: dataset construction, three PPI embedding dimensions, and the final grid. It does not execute those stages.

Expected summary:

```text
Dry run complete. Planned steps: 5.
[build_dataset] ...
[build_ppi_dim64] ...
[build_ppi_dim128] ...
[build_ppi_dim256] ...
[run_grid] ...
```

## Level 2 — prepare data and run a CPU smoke test

The expression matrices are rebuilt from public GEO accessions and remain ignored by Git.

```bash
Rscript pretraining_dataset/scripts/prepare_addneuromed_geo.R \
  --output-dir=task_dataset --install

python experiments/scripts/baseline/build_alzheimer_dataset.py
python experiments/scripts/paper_comparison/build_txt_pairwise_datasets.py
```

Check that the aligned cohort contains 711 samples and 19,460 genes. A change in those counts should trigger an annotation audit before training.

Run one deliberately small CPU experiment:

```bash
python experiments/scripts/paper_comparison/run_txt_multitask_cv_and_seeds.py \
  --run-mode seeds \
  --architectures 1l2h \
  --device cpu \
  --smoke \
  --result-root results/smoke/txt_multitask
```

Each completed run produces metrics, a training log, model metadata, and the exact configuration under its result directory.

## Level 3 — reproduce the research configuration

Full experiments require a CUDA-capable system and take substantially longer. Build the PPI initialization and launch the maintained grid through the orchestrator:

```bash
python experiments/scripts/paper_comparison/run_txt_multitask_final_pipeline.py \
  --preset balanced \
  --device cuda \
  --ppi-device cuda
```

Before a long run, inspect all available controls:

```bash
python experiments/scripts/paper_comparison/run_txt_multitask_final_pipeline.py --help
python experiments/scripts/paper_comparison/run_txt_multitask_cv_and_seeds.py --help
```

See [reproducibility.md](reproducibility.md) for leakage controls, software environments, computational cost tiers, and the result-provenance checklist.

## Common problems

| Symptom | What to check |
|---|---|
| `FileNotFoundError` below `task_dataset/processed/` | Complete the three data-preparation commands in Level 2. |
| CUDA is unavailable | Use `--device cpu` for a smoke test; full grids are intended for CUDA. |
| GEO counts differ from 711 × 19,460 | Stop and inspect platform annotations or label filtering. |
| PPI embedding file is missing | Run the final pipeline without `--skip-ppi-build`. |
| A generated file appears in Git status | Keep datasets, checkpoints and logs under their documented ignored directories. |
