from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))
for _module_name in ("data", "feature_selection", "image_transformer", "metrics", "models", "run_experiments", "summarize_results"):
    sys.modules.pop(_module_name, None)

import metrics
import models
import run_experiments
from feature_selection import SelectedFeatureSet


class ProfileTests(unittest.TestCase):
    @staticmethod
    def _args(profile: str) -> SimpleNamespace:
        return SimpleNamespace(
            implementation_profile=profile,
            paper_lasso_alpha=None,
            image_gene_order=None,
            epochs=None,
            batch_size_fusion=None,
            batch_size_single=None,
            early_stopping_monitor=None,
            start_from_epoch=None,
            learning_rate=None,
        )

    def test_paper_profile_resolves_documented_defaults(self) -> None:
        args = self._args("paper")
        run_experiments.apply_implementation_profile(args)
        self.assertEqual(args.paper_lasso_alpha, 1e-6)
        self.assertEqual(args.image_gene_order, "fisher")
        self.assertEqual((args.batch_size_single, args.batch_size_fusion), (30, 30))
        self.assertEqual(run_experiments.model_learning_rate(args, "fnn"), 1e-4)
        self.assertEqual(args.start_from_epoch, 250)

    def test_public_code_profile_records_notebook_conflicts(self) -> None:
        args = self._args("public_code")
        run_experiments.apply_implementation_profile(args)
        self.assertEqual(args.paper_lasso_alpha, 2e-4)
        self.assertEqual(args.image_gene_order, "input")
        self.assertEqual((args.batch_size_single, args.batch_size_fusion), (32, 64))
        self.assertEqual(run_experiments.model_learning_rate(args, "fnn"), 1e-3)
        self.assertEqual(run_experiments.model_learning_rate(args, "cnn"), 1e-4)


class RoutingTests(unittest.TestCase):
    def test_standalone_fnn_gets_all_genes_and_fusion_gets_lasso_genes(self) -> None:
        columns = ["g1", "g2", "g3", "g4"]
        train = pd.DataFrame(np.arange(24).reshape(6, 4), columns=columns)
        val = train.iloc[:2].copy()
        test = train.iloc[2:4].copy()
        selected = SelectedFeatureSet(
            x_train=train[["g2", "g4"]],
            x_val=val[["g2", "g4"]],
            x_test=test[["g2", "g4"]],
            selected_genes=["g2", "g4"],
            metadata={},
        )
        all_frames = (train, val, test)
        fnn_frames = run_experiments.gene_frames_for_model("fnn", all_frames, selected)
        fusion_frames = run_experiments.gene_frames_for_model("one2mfusion", all_frames, selected)
        self.assertEqual(fnn_frames[0].shape[1], 4)
        self.assertEqual(fusion_frames[0].columns.tolist(), ["g2", "g4"])


class ArchitectureTests(unittest.TestCase):
    def test_fusion_has_two_post_concatenation_dense_layers(self) -> None:
        model = models.build_fusion(8, (32, 32, 3), seed=1, learning_rate=1e-4)
        self.assertIsNotNone(model.get_layer("fusion_dense_1"))
        self.assertIsNotNone(model.get_layer("fusion_dense_2"))
        lr = float(model.optimizer.learning_rate.numpy())
        self.assertAlmostEqual(lr, 1e-4, places=7)

    def test_branch_dense_layers_include_kernel_bias_and_activity_regularization(self) -> None:
        model = models.build_fnn(8, seed=1, learning_rate=1e-4)
        regularized = [
            layer
            for layer in model.layers
            if getattr(layer, "units", None) == 32 and getattr(layer, "kernel_regularizer", None) is not None
        ]
        self.assertEqual(len(regularized), 2)
        for layer in regularized:
            self.assertIsNotNone(layer.bias_regularizer)
            self.assertIsNotNone(layer.activity_regularizer)


class TaskAgnosticArtifactTests(unittest.TestCase):
    def test_prediction_score_is_named_for_the_positive_class(self) -> None:
        frame = metrics.predictions_frame(pd.Index(["s1", "s2"]), np.array([0, 1]), np.array([0.1, 0.9]), np.array([0, 1]))
        self.assertIn("score_positive_class", frame.columns)
        self.assertNotIn("score_ad", frame.columns)


if __name__ == "__main__":
    unittest.main()
