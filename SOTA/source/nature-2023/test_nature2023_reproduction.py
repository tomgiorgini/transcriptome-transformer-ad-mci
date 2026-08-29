from __future__ import annotations

import tempfile
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))
for _module_name in ("data", "deep_models", "feature_selection", "metrics", "models", "run_experiments", "summarize_results"):
    sys.modules.pop(_module_name, None)

import feature_selection
import metrics
import models
import run_experiments


class KnowledgeGenesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.train = pd.DataFrame(
            {
                "g1": [0.0, 0.0, 0.0, 0.0],
                "g2": [0.0, 1.0, 2.0, 3.0],
                "g3": [0.0, 10.0, 0.0, 10.0],
                "g4": [1.0, 1.0, 2.0, 2.0],
            }
        )

    def test_missing_curated_list_still_runs_train_only_top_mad(self) -> None:
        result = feature_selection.knowledge_genes_features(
            self.train,
            self.train.iloc[:2],
            self.train.iloc[2:],
            None,
            mad_top_k=2,
        )
        self.assertEqual(result.selected_genes, ["g2", "g3"])
        self.assertEqual(result.metadata["status"], "incomplete_missing_curated")
        self.assertEqual(result.metadata["mad_genes_selected"], 2)

    def test_curated_genes_are_unioned_with_top_mad(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            curated = Path(temp_dir) / "curated.txt"
            curated.write_text("g1\ng4\nnot_in_matrix\n", encoding="utf-8")
            result = feature_selection.knowledge_genes_features(
                self.train,
                self.train.iloc[:2],
                self.train.iloc[2:],
                curated,
                mad_top_k=1,
            )
        self.assertEqual(result.selected_genes, ["g1", "g3", "g4"])
        self.assertEqual(result.metadata["status"], "ok")
        self.assertEqual(result.metadata["curated_overlap_genes"], 2)

    def test_missing_curated_list_is_skipped_unless_ablation_is_explicit(self) -> None:
        base_args = dict(
            knowledge_genes_file=None,
            knowledge_mad_top_k=2,
        )
        skipped = run_experiments.build_feature_set(
            "knowledge_genes",
            self.train,
            np.asarray([0, 1, 0, 1]),
            self.train.iloc[:2],
            self.train.iloc[2:],
            SimpleNamespace(**base_args, allow_incomplete_knowledge=False),
            seed=1,
        )
        self.assertTrue(skipped.metadata["status"].startswith("skipped"))

        ablation = run_experiments.build_feature_set(
            "knowledge_genes",
            self.train,
            np.asarray([0, 1, 0, 1]),
            self.train.iloc[:2],
            self.train.iloc[2:],
            SimpleNamespace(**base_args, allow_incomplete_knowledge=True),
            seed=1,
        )
        self.assertEqual(ablation.name, "mad_top3000_only")
        self.assertEqual(ablation.metadata["status"], "ok_incomplete_ablation")


class VssrfeTests(unittest.TestCase):
    def test_feature_selection_is_refit_inside_each_cv_fold(self) -> None:
        rng = np.random.default_rng(7)
        x_train = pd.DataFrame(rng.normal(size=(30, 8)), columns=[f"g{i}" for i in range(8)])
        y_train = np.tile([0, 1], 15)
        with mock.patch.object(
            feature_selection,
            "_vssrfe_select_path",
            wraps=feature_selection._vssrfe_select_path,
        ) as select_path:
            result = feature_selection.vssrfe_lr_features(
                x_train,
                y_train,
                x_train.iloc[:6],
                x_train.iloc[6:12],
                seed=3,
                n_iter=1,
                cv_folds=3,
                n_jobs=1,
                min_genes=2,
                max_genes=3,
                step_genes=1,
                extra_gene_counts=[],
                fixed_c=1.0,
                fixed_n_genes=None,
            )
        self.assertEqual(select_path.call_count, 4)  # three validation folds + final train refit
        self.assertEqual(result.metadata["score"], "average_precision")
        self.assertTrue(all("average_precision" in row for row in result.metadata["candidate_scores"]))


class TorchVaeTests(unittest.TestCase):
    def test_torch_backend_trains_and_returns_finite_latents(self) -> None:
        rng = np.random.default_rng(19)
        train = rng.uniform(0.0, 1.0, size=(12, 8)).astype(np.float32)
        val = rng.uniform(0.0, 1.0, size=(4, 8)).astype(np.float32)
        test = rng.uniform(0.0, 1.0, size=(5, 8)).astype(np.float32)
        z_train, z_val, z_test, metadata = feature_selection._torch_vae_latents_subprocess(
            train,
            val,
            test,
            seed=3,
            epochs=2,
            batch_size=4,
            patience=2,
            architecture="basic",
            learning_rate=1e-3,
            reconstruction_loss="categorical_crossentropy",
            device_name="cpu",
            hidden_dims=(16, 8, 4),
            latent_dim=3,
        )
        self.assertEqual(z_train.shape, (12, 3))
        self.assertEqual(z_val.shape, (4, 3))
        self.assertEqual(z_test.shape, (5, 3))
        self.assertTrue(np.isfinite(z_train).all())
        self.assertEqual(metadata["backend"], "torch")
        self.assertEqual(metadata["device"], "cpu")


class ModelTuningTests(unittest.TestCase):
    def test_classifier_bayes_search_optimizes_average_precision(self) -> None:
        captured: dict[str, object] = {}

        class FakeSearch:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self.estimator = kwargs["estimator"]

            def fit(self, x, y):
                self.best_estimator_ = self.estimator.fit(x, y)
                self.best_params_ = {}
                self.best_score_ = 0.5
                return self

        x = np.arange(48, dtype=np.float32).reshape(12, 4)
        y = np.tile([0, 1], 6)
        with mock.patch.object(models, "BayesSearchCV", FakeSearch):
            models.fit_tuned_model("lr", x, y, seed=1, n_iter=2, cv_folds=2, n_jobs=1)
        self.assertEqual(captured["scoring"], "average_precision")


class ConfigurationTests(unittest.TestCase):
    def test_full_paper_matrix_has_25_classical_plus_2_standalone(self) -> None:
        args = SimpleNamespace(
            feature_sets=list(run_experiments.PAPER_FEATURE_SETS),
            models=["lr", "svm", "xgboost", "rf", "mlp"],
            include_deep=True,
        )
        matrix = run_experiments.requested_configuration_matrix(args)
        self.assertEqual(len(matrix), 27)
        deep = [item for item in matrix if item[1] in run_experiments.PAPER_STANDALONE_MODELS]
        self.assertEqual(deep, [("all_genes", "vae_classifier"), ("all_genes", "cnn")])

    def test_vae_profile_resolves_paper_and_public_code_conflict(self) -> None:
        paper = SimpleNamespace(implementation_profile="paper", vae_epochs=None, vae_learning_rate=None)
        public = SimpleNamespace(implementation_profile="public_code", vae_epochs=None, vae_learning_rate=None)
        run_experiments.apply_implementation_profile(paper)
        run_experiments.apply_implementation_profile(public)
        self.assertEqual(paper.vae_epochs, 1000)
        self.assertEqual(paper.vae_learning_rate, 1e-5)
        self.assertEqual(public.vae_learning_rate, 1e-4)


class TaskAgnosticArtifactTests(unittest.TestCase):
    def test_predictions_and_confusion_labels_are_not_hardcoded_to_ad_mci(self) -> None:
        frame = metrics.predictions_frame(pd.Index(["s1", "s2"]), np.array([0, 1]), np.array([0.2, 0.8]), np.array([0, 1]))
        self.assertIn("score_positive_class", frame.columns)
        self.assertNotIn("score_ad", frame.columns)
        confusion = metrics.confusion_frame(np.array([0, 1]), np.array([0, 1]))
        self.assertEqual(confusion.index.tolist(), ["true_class_0", "true_class_1"])


if __name__ == "__main__":
    unittest.main()
