# Paper-faithful VMA

`PaperFaithfulVMA` implements Equations 7–10 of GRAMformer on projected,
position-aligned modality streams:

1. `Z = [q, k_1, ..., k_M]`;
2. `volume = sqrt(max(det(Z^T Z), 0) + eps)`;
3. `score = (-beta * volume + sum_m <q, k_m>) / sqrt(d_head)`;
4. softmax over the aligned key positions;
5. one sigmoid value gate per modality, followed by the `1/M` average.

It intentionally has no L2 normalization, RMS normalization, sparse PPI edge
substitution, expression-contrast message, detached gradient path, scalar or
per-head outer `gamma`, or parallel dense-attention branch.

## TUPE

- `tupe_mode="off"` is the paper-faithful default. Passing a TUPE tensor is an
  error, which prevents an accidental departure from the original equation.
- `tupe_mode="on"` adds TUPE exactly once, after the paper score and before its
  softmax. This is an explicit extension, not part of the original equation.

## Inputs

The low-level operator accepts:

- query: `[batch, heads, n_queries, d_head]`;
- each key/value modality: `[batch, heads, n_keys, d_head]`;
- optional TUPE: broadcastable to `[batch, heads, n_queries, n_keys]`.

All modalities must share the same aligned `n_keys`, and the paper constraint
`num_modalities + 1 <= d_head` is enforced. `PaperFaithfulMultiHeadVMA` adds
independent key/value projections for each modality and an output projection.

```python
from source.models.txt_volumetric import PaperFaithfulMultiHeadVMA

layer = PaperFaithfulMultiHeadVMA(
    n_heads=2,
    d_model=64,
    d_embed=64,
    num_modalities=2,
    beta=1.0,
    tupe_mode="off",  # exact paper score
)
output = layer(query_stream, [modality_1, modality_2])
```

This layer is deliberately not registered as a current `model_variant`. The
existing TxT pipeline exposes one expression sequence plus a PPI graph, not two
predefined aligned modality streams. Wiring it in without first defining those
streams would silently introduce a new biological assumption and would no
longer be a literal reproduction of the paper's multimodal setup.
