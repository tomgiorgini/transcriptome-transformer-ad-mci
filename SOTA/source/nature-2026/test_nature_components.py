from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) in sys.path:
    sys.path.remove(str(MODULE_DIR))
sys.path.insert(0, str(MODULE_DIR))
for _module_name in ("augmentation", "balancing", "metrics"):
    sys.modules.pop(_module_name, None)

import augmentation as augmentation_module
from augmentation import augment_training_data
from balancing import balance_training_data


_FS_SPEC = importlib.util.spec_from_file_location(
    "nature_2026_feature_selection",
    MODULE_DIR / "feature_selection.py",
)
if _FS_SPEC is None or _FS_SPEC.loader is None:
    raise RuntimeError("Could not load Nature feature_selection.py")
fs = importlib.util.module_from_spec(_FS_SPEC)
sys.modules[_FS_SPEC.name] = fs
_FS_SPEC.loader.exec_module(fs)

_MODELS_SPEC = importlib.util.spec_from_file_location(
    "nature_2026_models",
    MODULE_DIR / "models.py",
)
if _MODELS_SPEC is None or _MODELS_SPEC.loader is None:
    raise RuntimeError("Could not load Nature models.py")
nature_models = importlib.util.module_from_spec(_MODELS_SPEC)
sys.modules[_MODELS_SPEC.name] = nature_models
_MODELS_SPEC.loader.exec_module(nature_models)


class NatureFeatureSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(17)
        y = np.asarray([0] * 10 + [1] * 10, dtype=np.int32)
        signal = y + rng.normal(0.0, 0.05, size=len(y))
        self.x = pd.DataFrame(
            {
                "signal": signal,
                "g1": rng.uniform(size=len(y)),
                "g2": rng.uniform(size=len(y)),
                "g3": rng.uniform(size=len(y)),
            }
        )
        self.y = y

    def test_paper_comparison_k_values(self) -> None:
        self.assertEqual(fs.PAPER_INTEGRATED_K["lasso"], 500)
        self.assertEqual(fs.PAPER_INTEGRATED_K["rf_importance"], 500)

    def test_all_genes_is_no_selection_baseline(self) -> None:
        result = fs.select_features("all_genes", self.x, self.y, seed=2)
        self.assertEqual(result.selected_genes, self.x.columns.tolist())
        self.assertEqual(result.metadata["paper_method"], "No feature-selection baseline")

    def test_chi2_uses_train_fitted_discretization(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = fs.select_features(
                "chi2",
                self.x,
                self.y,
                seed=2,
                k_overrides={"chi2": 2},
                chi2_bins=4,
            )
        self.assertEqual(len(result.selected_genes), 2)
        self.assertEqual(result.metadata["chi2_bins_requested"], 4)
        self.assertEqual(result.metadata["chi2_discretizer_fit_scope"], "feature_selection_fit_data_only")
        self.assertFalse(result.metadata["paper_bin_count_disclosed"])


class NatureTrainingAndAugmentationTests(unittest.TestCase):
    def test_training_undersampling_is_deterministic_and_balanced(self) -> None:
        x = pd.DataFrame({"g": np.arange(10, dtype=float)}, index=[f"s{i}" for i in range(10)])
        y = np.asarray([0] * 7 + [1] * 3, dtype=np.int32)
        first = balance_training_data(x, y, "undersample", seed=42)
        second = balance_training_data(x, y, "undersample", seed=42)
        self.assertEqual(first.x_train.index.tolist(), second.x_train.index.tolist())
        self.assertEqual(first.manifest["class_counts_after"], {"0": 3, "1": 3})
        self.assertFalse(first.manifest["validation_and_test_modified"])

    def test_ctgan_mode_uses_external_ctgan_interface(self) -> None:
        class FakeCTGAN:
            kwargs: dict[str, object] = {}

            def __init__(self, **kwargs):
                FakeCTGAN.kwargs = kwargs
                self.table: pd.DataFrame | None = None

            def set_random_state(self, seed):
                self.seed = seed

            def fit(self, table, discrete_columns):
                self.table = table.copy()
                self.discrete_columns = discrete_columns

            def sample(self, count, condition_column=None, condition_value=None):
                assert self.table is not None
                sampled = pd.concat([self.table.iloc[[0]].copy()] * count, ignore_index=True)
                sampled[condition_column] = condition_value
                return sampled

        fake_module = types.ModuleType("ctgan")
        fake_module.__version__ = "test"
        fake_module.CTGAN = FakeCTGAN
        x = pd.DataFrame(
            {"g1": np.linspace(0.0, 1.0, 8), "g2": np.linspace(1.0, 0.0, 8)},
            index=[f"s{i}" for i in range(8)],
        )
        y = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32)
        with patch.dict(sys.modules, {"ctgan": fake_module}), patch.object(
            augmentation_module,
            "_train_and_sample_ctgan",
            wraps=augmentation_module._train_and_sample_ctgan_inprocess,
        ):
            result = augment_training_data(
                x,
                y,
                mode="ctgan",
                seed=5,
                target_size=12,
                latent_dim=128,
                epochs=2,
                batch_size=4,
                learning_rate=0.001,
                ctgan_pac=1,
            )
        self.assertEqual(len(result.y_train), 12)
        self.assertEqual(result.manifest["augmentation_display_name"], "external_ctgan")
        self.assertEqual(result.manifest["reproduction_status"], "paper_aligned_best_effort_external_ctgan")
        self.assertEqual(FakeCTGAN.kwargs["generator_dim"], (256, 256))
        self.assertEqual(FakeCTGAN.kwargs["pac"], 1)


class NatureModelTests(unittest.TestCase):
    def test_paper_reported_classical_estimator_counts(self) -> None:
        rng = np.random.default_rng(3)
        x = pd.DataFrame(rng.normal(size=(30, 8)), columns=[f"g{i}" for i in range(8)])
        y = np.asarray([0, 1] * 15, dtype=np.int32)
        for model_name, expected_estimators in (("rf", 100), ("adaboost", 200), ("xgboost", 100)):
            with self.subTest(model=model_name):
                _, _, _, metadata = nature_models.fit_predict_sklearn(
                    model_name,
                    x.iloc[:20],
                    y[:20],
                    x.iloc[20:],
                    y[20:],
                    seed=9,
                    n_jobs=1,
                )
                self.assertEqual(metadata["n_estimators"], expected_estimators)

    def test_cnn_has_two_way_softmax_output(self) -> None:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        from tensorflow import keras

        model = nature_models._paper_cnn_model(keras, n_features=64, dropout=0.2)
        self.assertEqual(model.output_shape[-1], 2)
        self.assertEqual(model.layers[-1].activation.__name__, "softmax")


if __name__ == "__main__":
    unittest.main()
