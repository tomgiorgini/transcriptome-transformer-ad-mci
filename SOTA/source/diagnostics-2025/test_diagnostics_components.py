from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) in sys.path:
    sys.path.remove(str(MODULE_DIR))
sys.path.insert(0, str(MODULE_DIR))
sys.modules.pop("metrics", None)

from metrics import confusion_frame, predictions_frame


_FS_SPEC = importlib.util.spec_from_file_location(
    "diagnostics_2025_feature_selection",
    MODULE_DIR / "feature_selection.py",
)
if _FS_SPEC is None or _FS_SPEC.loader is None:
    raise RuntimeError("Could not load Diagnostics feature_selection.py")
fs = importlib.util.module_from_spec(_FS_SPEC)
sys.modules[_FS_SPEC.name] = fs
_FS_SPEC.loader.exec_module(fs)


class DiagnosticsFeatureSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(7)
        self.x = pd.DataFrame(
            rng.normal(size=(24, 6)),
            columns=[f"g{i}" for i in range(6)],
        )
        self.y = np.asarray([0, 1] * 12, dtype=np.int32)

    def test_true_sfbs_uses_one_fixed_target(self) -> None:
        selected, trace, metadata = fs._true_sfbs_select(
            self.x,
            self.y,
            target_genes=3,
            cv_folds=2,
            seed=11,
            n_jobs=1,
        )
        self.assertEqual(len(selected), 3)
        self.assertEqual(metadata["sfbs_k_features"], 3)
        self.assertEqual(metadata["sfbs_fixed_target_effective"], 3)
        self.assertEqual(metadata["reproduction_status"], "paper_aligned_best_effort")
        self.assertFalse(trace.empty)

    def test_sfbs_failure_aborts_unless_fallback_is_explicit(self) -> None:
        ranking = pd.DataFrame(
            {
                "gene": self.x.columns,
                "xgboost_importance": np.arange(self.x.shape[1], 0, -1, dtype=float),
                "xgboost_rank": np.arange(1, self.x.shape[1] + 1),
            }
        )
        with (
            patch.object(fs, "_xgboost_importance_ranking", return_value=ranking),
            patch.object(fs, "_true_sfbs_select", side_effect=ValueError("forced failure")),
        ):
            with self.assertRaisesRegex(RuntimeError, "fallback is disabled"):
                fs.select_xgboost_sfbs_genes(
                    self.x,
                    self.y,
                    seed=3,
                    top_k=6,
                    target_genes=3,
                )
            result = fs.select_xgboost_sfbs_genes(
                self.x,
                self.y,
                seed=3,
                top_k=6,
                target_genes=3,
                allow_sfbs_fallback=True,
            )
        self.assertEqual(len(result.selected_genes), 3)
        self.assertTrue(result.metadata["fallback_used"])
        self.assertEqual(result.metadata["reproduction_status"], "fallback_not_sfbs")

    def test_binary_artifacts_use_generic_labels(self) -> None:
        predictions = predictions_frame(
            pd.Index(["a", "b"]),
            np.asarray([0, 1]),
            np.asarray([0.2, 0.8]),
            np.asarray([0, 1]),
        )
        self.assertIn("score_positive_class", predictions.columns)
        self.assertNotIn("score_ad", predictions.columns)
        confusion = confusion_frame(np.asarray([0, 1]), np.asarray([0, 1]))
        self.assertEqual(confusion.index.tolist(), ["true_class_0", "true_class_1"])


if __name__ == "__main__":
    unittest.main()
