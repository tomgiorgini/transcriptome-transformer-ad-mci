# Experiment entry points

This directory intentionally contains only the maintained public interface:

| File | Purpose |
|---|---|
| <code>prepare_data.py</code> | Build the canonical AD/MCI/control matrix, labels, and split from local GEO-derived files. |
| <code>build_ppi_embedding.py</code> | Download/filter HIPPIE and train the node2vec initialization. |
| <code>train.py</code> | Train and evaluate the non-VMA multi-task TxT model with the full supported option surface. |

Run any file with <code>--help</code> before an expensive experiment. Supporting reusable code lives under <code>source/</code>; generated outputs belong under <code>results/</code>.
