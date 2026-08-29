from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from feature_selection import (  # noqa: E402
    FeatureSet,
    fit_vae_feature_set,
    load_hprd_hub_genes,
    resolve_vae_latent_dim,
)
from metrics import classification_report_frame, confusion_frame, predictions_frame  # noqa: E402
from models import build_paper_estimator  # noqa: E402
from run_experiments import parse_args  # noqa: E402
from summarize_results import summarize  # noqa: E402


class PaperEstimatorTests(unittest.TestCase):
    def test_cli_defaults_are_full_grid_paper_profile_and_keep_manifest_support(self) -> None:
        with patch.object(sys, "argv", ["run_experiments.py"]):
            args = parse_args()
        self.assertEqual(args.repeats, 10)
        self.assertEqual(args.inner_val_ratio, 0.125)
        self.assertEqual(args.fdr_threshold, 0.01)
        self.assertEqual(args.deg_fallback_top_k, 0)
        self.assertEqual(args.feature_sets, ["deg", "vae", "tf_genes", "cfg_genes", "hub_genes"])
        self.assertEqual(args.models, ["lr", "l1_lr", "svm", "rf", "dnn"])
        self.assertEqual(args.class_weighting, "off")
        self.assertFalse(args.allow_cfg_supplement_proxy)
        self.assertIsNone(args.split_manifest_dir)

    def test_paper_defaults_do_not_apply_implicit_class_weights(self) -> None:
        lr, lr_meta = build_paper_estimator("lr", n_features=9, seed=3)
        self.assertIsNone(lr.penalty)
        self.assertIsNone(lr.class_weight)
        self.assertIsNone(lr_meta["class_weight"])

        l1, l1_meta = build_paper_estimator("l1_lr", n_features=9, seed=3)
        self.assertEqual(l1.penalty, "l1")
        self.assertEqual(l1.C, 10000.0)
        self.assertEqual(l1_meta["lambda"], 0.0001)
        self.assertIsNone(l1.class_weight)

        svm, svm_meta = build_paper_estimator("svm", n_features=9, seed=3)
        self.assertIn("standardscaler", svm.named_steps)
        self.assertAlmostEqual(svm.named_steps["svc"].gamma, 1.0 / 9.0)
        self.assertIsNone(svm.named_steps["svc"].class_weight)
        self.assertIn("e1071", str(svm_meta["feature_scaling"]))

        rf, rf_meta = build_paper_estimator("rf", n_features=9, seed=3)
        self.assertEqual(rf.n_estimators, 500)
        self.assertEqual(rf.max_features, "sqrt")
        self.assertIsNone(rf.class_weight)
        self.assertEqual(rf_meta["max_features"], "sqrt")

    def test_balanced_weighting_is_explicit(self) -> None:
        rf, metadata = build_paper_estimator("rf", n_features=4, seed=1, class_weight="balanced")
        self.assertEqual(rf.class_weight, "balanced")
        self.assertEqual(metadata["class_weight"], "balanced")


class FeatureSelectionTests(unittest.TestCase):
    def test_latent_dimension_policy_is_explicit(self) -> None:
        self.assertEqual(resolve_vae_latent_dim(1)[0], 1)
        self.assertEqual(resolve_vae_latent_dim(334)[0], 100)
        self.assertEqual(resolve_vae_latent_dim(697)[0], 200)
        self.assertEqual(resolve_vae_latent_dim(1604)[0], 300)
        dimension, metadata = resolve_vae_latent_dim(80, policy="fixed", fixed_dim=40)
        self.assertEqual(dimension, 40)
        self.assertEqual(metadata["latent_policy_status"], "user_declared")
        with self.assertRaises(ValueError):
            resolve_vae_latent_dim(80, policy="fixed", fixed_dim=None)

    def test_hprd_loader_uses_unique_undirected_degree_and_records_hash(self) -> None:
        rows = [
            ["A", "id_a", "ref_a", "B"],
            ["A", "id_a", "ref_a", "C"],
            ["A", "id_a", "ref_a", "D"],
            ["B", "id_b", "ref_b", "A"],  # duplicate reversed edge
            ["A", "id_a", "ref_a", "A"],  # self edge
            ["B", "id_b", "ref_b", "C"],
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "BINARY_PROTEIN_PROTEIN_INTERACTIONS.txt"
            pd.DataFrame(rows).to_csv(path, sep="\t", header=False, index=False)
            hubs, provenance = load_hprd_hub_genes(path, degree_threshold=2)
        self.assertEqual(hubs, {"A"})
        self.assertEqual(provenance["n_unique_edges"], 4)
        self.assertEqual(provenance["gene_columns_zero_based"], [0, 3])
        self.assertEqual(len(str(provenance["network_sha256"])), 64)
        self.assertEqual(provenance["source_policy"], "user-supplied local file; no automatic download or substitution")

    @unittest.skipUnless(importlib.util.find_spec("tensorflow"), "TensorFlow not installed")
    def test_vae_is_variational_supervised_and_train_only(self) -> None:
        rng = np.random.default_rng(7)
        base = FeatureSet(
            "deg",
            rng.random((12, 1), dtype=np.float32),
            rng.random((4, 1), dtype=np.float32),
            rng.random((5, 1), dtype=np.float32),
            ["gene_0"],
            "completed",
            {},
        )
        result = fit_vae_feature_set(
            base,
            y_train=np.asarray([0, 1] * 6, dtype=np.int64),
            y_val=np.asarray([0, 1, 0, 1], dtype=np.int64),
            seed=7,
            epochs=1,
            batch_size=None,
            latent_policy="fixed",
            fixed_latent_dim=1,
            fine_tune_epochs=1,
        )
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.x_train.shape, (12, 1))
        self.assertEqual(result.x_test.shape, (5, 1))
        self.assertEqual(result.metadata["architecture"], "variational_autoencoder")
        self.assertIn("KL", str(result.metadata["regularization"]))
        self.assertEqual(result.metadata["optimizer"], "adagrad")
        self.assertTrue(result.metadata["supervised_fine_tuning"])
        self.assertEqual(result.metadata["fit_scope"], "train_inner")
        self.assertEqual(result.metadata["outer_test_usage"], "transform_only")


class ArtifactTests(unittest.TestCase):
    def test_label_artifacts_are_task_generic(self) -> None:
        y_true = np.asarray([0, 1, 0, 1])
        y_pred = np.asarray([0, 1, 1, 1])
        scores = np.asarray([0.1, 0.8, 0.7, 0.9])
        names = ("CTL", "MCI")
        predictions = predictions_frame(pd.Index(["s1", "s2", "s3", "s4"]), y_true, scores, y_pred, names)
        self.assertIn("score_positive", predictions.columns)
        self.assertNotIn("score_ad", predictions.columns)
        self.assertEqual(set(predictions["positive_class"]), {"MCI"})
        confusion = confusion_frame(y_true, y_pred, names)
        self.assertEqual(confusion.index.tolist(), ["true_CTL", "true_MCI"])
        report = classification_report_frame(y_true, y_pred, names)
        self.assertIn("CTL", report.index)
        self.assertIn("MCI", report.index)

    def test_pr_auc_ranking_is_sorted_by_pr_auc(self) -> None:
        metric_columns = [
            "pr_auc",
            "roc_auc",
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "macro_f1",
            "weighted_f1",
            "log_loss",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                {"protocol": "batch_holdout", "scenario": "shared_test", "feature_set": "deg", "model": "lr", "pr_auc": 0.9, "roc_auc": 0.6},
                {"protocol": "batch_holdout", "scenario": "shared_test", "feature_set": "deg", "model": "rf", "pr_auc": 0.7, "roc_auc": 0.95},
            ]
            for index, row in enumerate(rows):
                for column in metric_columns:
                    row.setdefault(column, 0.5)
                run_dir = root / f"run_{index}"
                run_dir.mkdir()
                pd.DataFrame([row]).to_csv(run_dir / "metrics.csv", index=False)
            summarize(root)
            ranking = pd.read_csv(root / "ranking_by_pr_auc.csv")
        self.assertEqual(ranking.iloc[0]["model"], "lr")
        self.assertGreater(ranking.iloc[0]["pr_auc_mean"], ranking.iloc[1]["pr_auc_mean"])


if __name__ == "__main__":
    unittest.main()
