# TxT

Documentazione del modello `TxT` implementato in questa repo.

## Input

- `x`: tensore `batch_size x n_genes`
- `mask`: opzionale, maschera di attenzione

## Architettura

Il modello e' composto da:

1. `Embedder`
2. `TUPE_A`
3. stack di `EncoderLayer`
4. aggregazione finale `Flatten` oppure `Avgpool`
5. una `TaskSpecificLayer`

## Output

`forward()` restituisce una lista con i logits delle task head.

## Relazione con `train_txt.py`

Il trainer espone:

- embedding random oppure esterni
- `d_model`, `d_ff`, `n_heads`, `n_layers`
- `norm_first`
- `aggfunc`
- dimensioni della head finale

Output standard:

- `best_model.pt`
- `gene_embedding.csv`
- `training_log.csv`
- `metrics_summary.csv`
- `selected_genes.csv`
- `test_predictions.csv`
- `test_confusion_matrix.csv`
- `test_classification_report.csv`
- `model_summary.json`
