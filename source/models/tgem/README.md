# T-GEM

Documentazione del modello `TGemClassifier` implementato in questa repo.

## Input

- `x`: tensore `batch_size x n_genes`

## Architettura

`TGemClassifier` e' composto da:

1. `n_layers` blocchi `MultiAttentionLayer`
2. `n_layers` blocchi `ResidualLayer`
3. una attivazione globale sul vettore risultante
4. un classificatore lineare finale

## Output

- logits `batch_size x n_classes`

## Relazione con `train_tgem.py`

Il trainer espone:

- split `official`, `custom`, `stratified`
- scelta di `n_heads`, `n_layers`, `dropout`, `activation`
- preprocessing comune tramite `prepare_dataset()`

Output standard di training:

- `best_model.pt`
- `training_log.csv`
- `metrics_summary.csv`
- `selected_genes.csv`
- `test_predictions.csv`
- `test_confusion_matrix.csv`
- `test_classification_report.csv`
- `model_summary.json`
