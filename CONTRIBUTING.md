# Contributing

Keep the public interface focused: new experiment behaviour should normally be exposed through <code>experiments/train.py</code>, not through another launcher.

Before opening a change:

~~~bash
python -m compileall -q source experiments
python experiments/prepare_data.py --help
python experiments/build_ppi_embedding.py --help
python experiments/train.py --help
~~~

Do not commit participant-level matrices, downloaded PPI files, checkpoints, logs, or third-party source trees. Changes to evaluation must preserve train-only feature selection/scaling and validation-only checkpoint selection.

Review [NOTICE.md](NOTICE.md) before contributing; no open-source licence has yet been granted.
