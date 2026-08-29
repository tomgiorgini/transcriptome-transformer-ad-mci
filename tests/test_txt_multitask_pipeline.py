from __future__ import annotations

import json
import sys
import tempfile
import unittest
from argparse import Namespace
from io import BytesIO
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch

from experiments.scripts.paper_comparison import train_txt_multitask as multitask
from experiments.scripts.paper_comparison import run_txt_multitask_final_grid as final_grid
from experiments.scripts.paper_comparison import run_txt_multitask_final_pipeline as final_pipeline
from experiments.scripts.paper_comparison import run_txt_multitask_seed_followup as seed_followup
from experiments.scripts.pretraining import build_ppi_embedding as ppi_embedding
from source.pipeline.dataset import PreparedDataset
from source.models.txt.model import TxT


class MultitaskEmbeddingTests(unittest.TestCase):
    def test_external_embedding_is_aligned_to_selected_genes_with_random_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            embed_file = Path(tmpdir) / "ppi_node_embedding.csv"
            pd.DataFrame(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                index=pd.Index(["TP53", "APOE"], name="Gene"),
            ).to_csv(embed_file)

            embedding, manifest = multitask.build_embedding_for_selected_genes(
                gene_names=["APOE", "MISSING"],
                embed_dim=3,
                seed=7,
                embed_file=embed_file,
            )

        self.assertEqual(list(embedding.index), ["APOE", "MISSING"])
        np.testing.assert_allclose(embedding.loc["APOE"].to_numpy(dtype=np.float32), np.array([4.0, 5.0, 6.0]))
        self.assertEqual(manifest["embedding_source"], "external_file")
        self.assertEqual(manifest["target_genes"], 2)
        self.assertEqual(manifest["matched_genes"], 1)
        self.assertEqual(manifest["missing_genes"], 1)
        self.assertEqual(manifest["missing_gene_names"], ["MISSING"])
        self.assertAlmostEqual(manifest["coverage"], 0.5)

    def test_global_std_rescaling_matches_random_scale_and_preserves_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            embed_file = Path(tmpdir) / "ppi.csv"
            raw = np.array([[1.0, 5.0], [3.0, 9.0], [8.0, 2.0]], dtype=np.float64)
            pd.DataFrame(raw, index=pd.Index(["G1", "G2", "G3"], name="Gene")).to_csv(embed_file)

            embedding, manifest = multitask.build_embedding_for_selected_genes(
                ["G1", "G2", "G3"],
                embed_dim=2,
                seed=7,
                embed_file=embed_file,
                init_scale=0.02,
                rescale="global_std",
            )

        transformed = embedding.to_numpy()
        self.assertAlmostEqual(float(transformed.std()), 0.02, places=6)
        raw_distances = [np.linalg.norm(raw[i] - raw[j]) for i, j in [(0, 1), (0, 2), (1, 2)]]
        transformed_distances = [np.linalg.norm(transformed[i] - transformed[j]) for i, j in [(0, 1), (0, 2), (1, 2)]]
        ratios = np.asarray(transformed_distances) / np.asarray(raw_distances)
        self.assertTrue(np.allclose(ratios, ratios[0]))
        self.assertEqual(manifest["rescale"], "global_std")

    def test_candidate_gene_file_can_filter_random_initialization_independently(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            x_file = Path(tmpdir) / "X.csv"
            pd.DataFrame({"sample_id": ["S1"], "G1": [1.0], "G2": [2.0], "G3": [3.0]}).to_csv(x_file, index=False)
            candidate_file = Path(tmpdir) / "ppi_genes.csv"
            pd.DataFrame({"Gene": ["G1", "G3", "NOT_IN_EXPRESSION"]}).to_csv(candidate_file, index=False)
            args = Namespace(
                x_file=x_file,
                embed_file=None,
                embedding_gene_policy="all",
                candidate_gene_file=candidate_file,
            )

            candidates, manifest = multitask.resolve_candidate_genes(args)

        self.assertEqual(candidates, ["G1", "G3"])
        self.assertTrue(manifest["candidate_gene_filter_applied"])
        self.assertEqual(manifest["expression_genes_after_filter"], 2)


class MultitaskArchitectureTests(unittest.TestCase):
    def _embedding_file(self, directory: str) -> Path:
        path = Path(directory) / "embedding.csv"
        pd.DataFrame(
            np.random.default_rng(1).normal(0.0, 0.02, size=(6, 4)),
            index=pd.Index([f"G{i}" for i in range(6)], name="Gene"),
        ).to_csv(path)
        return path

    def test_separate_encoders_and_layer_norm_forward(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model = TxT(
                embed_file=str(self._embedding_file(tmpdir)),
                gene_list=[f"G{i}" for i in range(6)],
                n_heads=2,
                d_model=4,
                d_ff=8,
                n_layers=1,
                aggfunc="Avgpool",
                d_hidden1=8,
                d_hidden2=4,
                d_output_dict={"AD_vs_MCI": 2, "AD_vs_CTL": 2, "MCI_vs_CTL": 2},
                head_norm="layer",
                encoder_sharing="separate",
            )
            outputs = model(torch.rand(4, 6))

        self.assertEqual(len(model.encoder_modules()), 3)
        self.assertEqual([tuple(output.shape) for output in outputs], [(4, 2), (4, 2), (4, 2)])
        self.assertFalse(any(isinstance(module, torch.nn.BatchNorm1d) for module in model.task_specific_layers.modules()))
        first_parameters = [next(transformer.parameters()) for transformer in model.encoder_modules()]
        self.assertEqual(len({id(parameter) for parameter in first_parameters}), 3)

    def test_task_attention_adapter_and_expression_residual_are_zero_safe(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model = TxT(
                embed_file=str(self._embedding_file(tmpdir)),
                gene_list=[f"G{i}" for i in range(6)],
                n_heads=2,
                d_model=4,
                d_ff=8,
                n_layers=1,
                aggfunc="Avgpool",
                d_hidden1=8,
                d_hidden2=4,
                d_output_dict={"AD_vs_MCI": 2, "AD_vs_CTL": 2, "MCI_vs_CTL": 2},
                pooling_mode="task_attention",
                attention_pooling_hidden_dim=3,
                primary_adapter_dim=2,
                expression_residual="additive_zero_init",
            )
            outputs = model(torch.rand(4, 6))

        self.assertEqual([tuple(output.shape) for output in outputs], [(4, 2), (4, 2), (4, 2)])
        self.assertEqual(len(model.task_attention_pooling), 3)
        self.assertTrue(torch.equal(model.primary_adapter.up.weight, torch.zeros_like(model.primary_adapter.up.weight)))
        self.assertEqual(float(model.encoder_modules()[0].encoder.expression_scale.detach()), 0.0)

    def test_tupe_default_is_exactly_explicit_on_and_off_removes_patient_signal(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            kwargs = dict(
                embed_file=str(self._embedding_file(tmpdir)),
                gene_list=[f"G{i}" for i in range(6)],
                n_heads=2,
                d_model=4,
                dropout=0.0,
                d_ff=8,
                n_layers=1,
                aggfunc="Avgpool",
                d_hidden1=8,
                d_hidden2=4,
                d_output_dict={"AD_vs_MCI": 2},
                head_norm="none",
            )
            torch.manual_seed(31)
            implicit = TxT(**kwargs).eval()
            next_after_implicit = torch.rand(4)
            torch.manual_seed(31)
            explicit = TxT(**kwargs, tupe_mode="on").eval()
            next_after_explicit = torch.rand(4)
            torch.manual_seed(31)
            no_tupe = TxT(**kwargs, tupe_mode="off").eval()
            x_left = torch.tensor(
                [[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]], dtype=torch.float32
            )
            x_right = torch.tensor(
                [[1.0, 0.1, 0.8, 0.3, 0.6, 0.0]], dtype=torch.float32
            )
            with torch.no_grad():
                implicit_output = implicit(x_left)[0]
                explicit_output = explicit(x_left)[0]
                no_tupe_left = no_tupe(x_left)[0]
                no_tupe_right = no_tupe(x_right)[0]

        self.assertEqual(list(implicit.state_dict()), list(explicit.state_dict()))
        torch.testing.assert_close(next_after_implicit, next_after_explicit, rtol=0, atol=0)
        torch.testing.assert_close(implicit_output, explicit_output, rtol=0, atol=0)
        torch.testing.assert_close(no_tupe_left, no_tupe_right, rtol=0, atol=0)

    def test_weighted_multitask_loss_matches_manual_weighted_average(self) -> None:
        logits = [
            torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            torch.tensor([[0.5, 0.0], [0.0, 0.5]]),
        ]
        labels = torch.tensor([[0, 0, 0], [1, 1, 1]])
        masks = torch.ones((2, 3), dtype=torch.bool)
        criteria = [torch.nn.CrossEntropyLoss() for _ in range(3)]
        weights = [0.5, 0.25, 0.25]

        loss, _ = multitask.multitask_loss(logits, labels, masks, criteria, weights)
        manual = sum(weight * criterion(task_logits, labels[:, idx]) for idx, (weight, criterion, task_logits) in enumerate(zip(weights, criteria, logits))) / sum(weights)

        self.assertTrue(torch.allclose(loss, manual))

    def test_mask_aware_heads_only_normalize_valid_task_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model = TxT(
                embed_file=str(self._embedding_file(tmpdir)),
                gene_list=[f"G{i}" for i in range(6)],
                n_heads=2,
                d_model=4,
                d_ff=8,
                n_layers=1,
                aggfunc="Avgpool",
                d_hidden1=8,
                d_hidden2=4,
                d_output_dict={"AD_vs_MCI": 2, "AD_vs_CTL": 2, "MCI_vs_CTL": 2},
                head_norm="batch",
            )
            observed_batch_sizes: list[int] = []
            hooks = []
            for head in model.task_specific_layers:
                hooks.append(head.register_forward_pre_hook(lambda _module, inputs: observed_batch_sizes.append(inputs[0].shape[0])))
            mask = torch.tensor(
                [[True, False, True], [True, True, False], [False, False, True], [False, False, False]],
                dtype=torch.bool,
            )
            model.train()
            outputs = model(torch.rand(4, 6), task_sample_mask=mask)
            for hook in hooks:
                hook.remove()

        self.assertEqual(observed_batch_sizes, [2, 1, 2])
        self.assertEqual([tuple(output.shape) for output in outputs], [(4, 2), (4, 2), (4, 2)])
        for task_idx, output in enumerate(outputs):
            self.assertTrue(torch.equal(output[~mask[:, task_idx]], torch.zeros_like(output[~mask[:, task_idx]])))

    def test_zero_gated_ppi_residual_is_an_exact_random_initialization_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            embed_file = self._embedding_file(tmpdir)
            prior_file = Path(tmpdir) / "ppi_prior.csv"
            pd.DataFrame(
                np.random.default_rng(2).normal(size=(6, 4)),
                index=pd.Index([f"G{i}" for i in range(6)], name="Gene"),
            ).to_csv(prior_file)
            kwargs = dict(
                embed_file=str(embed_file), gene_list=[f"G{i}" for i in range(6)], n_heads=2,
                d_model=4, d_ff=8, n_layers=1, aggfunc="Avgpool", d_hidden1=8, d_hidden2=4,
                d_output_dict={"AD_vs_MCI": 2, "AD_vs_CTL": 2, "MCI_vs_CTL": 2},
            )
            torch.manual_seed(19)
            baseline = TxT(**kwargs)
            torch.manual_seed(19)
            gated = TxT(**kwargs, ppi_prior_file=str(prior_file), ppi_gate_init=0.0)
            baseline.eval()
            gated.eval()
            x = torch.rand(3, 6)
            with torch.no_grad():
                baseline_outputs = baseline(x)
                gated_outputs = gated(x)

        self.assertEqual(gated.ppi_gate_values(), {"shared": 0.0})
        for baseline_output, gated_output in zip(baseline_outputs, gated_outputs):
            self.assertTrue(torch.allclose(baseline_output, gated_output))

    def test_primary_protected_projection_removes_auxiliary_conflict(self) -> None:
        parameter = torch.nn.Parameter(torch.zeros(2))
        gradients = [
            [torch.tensor([1.0, 0.0])],
            [torch.tensor([-1.0, 1.0])],
            [torch.tensor([0.0, 1.0])],
        ]

        projected = multitask.overwrite_shared_gradients_primary_protected(
            [parameter], gradients, [0.5, 0.25, 0.25]
        )

        self.assertEqual(projected, 1)
        self.assertTrue(torch.allclose(parameter.grad, torch.tensor([0.5, 0.5])))
        self.assertGreaterEqual(float(torch.dot(parameter.grad, gradients[0][0])), 0.0)

    def test_checkpoint_ensemble_respects_rank_and_epoch_gap(self) -> None:
        entries: list[dict[str, object]] = []
        state = {"weight": torch.tensor([1.0])}
        for epoch, value in [(1, 0.4), (2, 0.5), (5, 0.45), (9, 0.6)]:
            entries = multitask.update_checkpoint_ensemble(
                entries, value, epoch, state, "val_multitask_auc_mean", ensemble_size=3, min_gap=3
            )

        self.assertEqual([entry["epoch"] for entry in entries], [9, 2, 5])

    def test_evaluation_checkpoint_cli_supports_plural_and_rejects_both_forms(self) -> None:
        with mock.patch.object(
            sys,
            "argv",
            ["train_txt_multitask.py", "--evaluation-only-checkpoint", "single.pt"],
        ):
            single_args = multitask.parse_args()

        self.assertEqual(single_args.evaluation_only_checkpoint, Path("single.pt"))
        self.assertIsNone(single_args.evaluation_only_checkpoints)

        with mock.patch.object(
            sys,
            "argv",
            ["train_txt_multitask.py", "--evaluation-only-checkpoints", "a.pt", "b.pt"],
        ):
            args = multitask.parse_args()

        self.assertIsNone(args.evaluation_only_checkpoint)
        self.assertEqual(args.evaluation_only_checkpoints, [Path("a.pt"), Path("b.pt")])

        with mock.patch.object(
            sys,
            "argv",
            [
                "train_txt_multitask.py",
                "--evaluation-only-checkpoint",
                "single.pt",
                "--evaluation-only-checkpoints",
                "a.pt",
                "b.pt",
            ],
        ):
            with self.assertRaises(SystemExit):
                multitask.parse_args()

    def test_ensemble_artifacts_use_supplied_rank_one_as_best_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            source_paths = [directory / "source_a.pt", directory / "source_b.pt"]
            states = [
                {"weight": torch.tensor([1.0])},
                {"weight": torch.tensor([2.0])},
            ]
            for path, state in zip(source_paths, states):
                torch.save(state, path)

            copied_paths = multitask.save_ensemble_checkpoint_artifacts(
                directory / "output", states, source_paths
            )

            best_state = multitask.load_checkpoint_state(
                directory / "output" / "best_model.pt", torch.device("cpu")
            )
            rank_two_state = multitask.load_checkpoint_state(
                directory / "output" / "ensemble_checkpoint_rank2.pt",
                torch.device("cpu"),
            )

        self.assertEqual([path.name for path in copied_paths], [
            "ensemble_checkpoint_rank1.pt",
            "ensemble_checkpoint_rank2.pt",
        ])
        self.assertTrue(torch.equal(best_state["weight"], states[0]["weight"]))
        self.assertTrue(torch.equal(rank_two_state["weight"], states[1]["weight"]))

    def test_source_member_metadata_reads_ranked_epoch_and_score(self) -> None:
        summary = {
            "training_summary": {
                "checkpoint_metric": "val_multitask_auc_mean",
                "checkpoint_ensemble": [
                    {"rank": 1, "epoch": 3, "checkpoint_value": 0.7},
                    {"rank": 2, "epoch": 8, "checkpoint_value": 0.65},
                ],
            }
        }

        metadata = multitask.source_checkpoint_member_metadata(
            Path("ensemble_checkpoint_rank2.pt"), summary
        )

        self.assertEqual(metadata["epoch"], 8)
        self.assertEqual(metadata["score"], 0.65)
        self.assertEqual(metadata["score_metric"], "val_multitask_auc_mean")

    def test_probability_ensemble_and_diagnostics_cover_every_validation_batch(self) -> None:
        class DiagnosticToyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.logit = torch.nn.Parameter(torch.tensor(0.0))
                self.latest_diagnostics: dict[str, object] = {}

            def forward(
                self,
                x: torch.Tensor,
                task_sample_mask: torch.Tensor | None = None,
            ) -> list[torch.Tensor]:
                del task_sample_mask
                logits = torch.stack(
                    [torch.zeros(x.size(0), device=x.device), self.logit.expand(x.size(0))],
                    dim=1,
                )
                self.latest_diagnostics = {
                    "shared.layer_0": {
                        "signal": float(x.mean().detach().cpu() + self.logit.detach().cpu())
                    }
                }
                return [logits.clone() for _ in multitask.TASK_NAMES]

            def volumetric_diagnostics(self) -> dict[str, object]:
                return self.latest_diagnostics

        gene_x = np.arange(5, dtype=np.float32).reshape(-1, 1)
        source_y = np.array([0, 1, 2, 0, 1], dtype=np.int64)
        sample_ids = np.array([f"s{idx}" for idx in range(5)])
        dataset = PreparedDataset(
            class_names=["Control", "MCI", "AD"],
            gene_names=["G1"],
            train_ids=sample_ids,
            val_ids=sample_ids,
            test_ids=sample_ids,
            train_gene_x=gene_x,
            val_gene_x=gene_x,
            test_gene_x=gene_x,
            train_y=source_y,
            val_y=source_y,
            test_y=source_y,
        )
        model = DiagnosticToyModel()
        state_dicts = [
            {"logit": torch.tensor(0.0)},
            {"logit": torch.tensor(2.0)},
        ]
        diagnostics: dict[str, object] = {}

        results, _ = multitask.evaluate_multitask(
            model,  # type: ignore[arg-type]
            dataset,
            "val",
            gene_x,
            source_y,
            sample_ids,
            multitask.build_task_specs(dataset.class_names),
            batch_size=2,
            device=torch.device("cpu"),
            criteria=[torch.nn.CrossEntropyLoss() for _ in multitask.TASK_NAMES],
            task_loss_weights=[1.0, 1.0, 1.0],
            state_dicts=state_dicts,
            volumetric_diagnostics_collector=diagnostics,
        )

        expected_positive_probability = (
            torch.softmax(torch.tensor([0.0, 0.0]), dim=0)[1]
            + torch.softmax(torch.tensor([0.0, 2.0]), dim=0)[1]
        ) / 2.0
        for result in results.values():
            np.testing.assert_allclose(
                result["y_prob"][:, 1],
                float(expected_positive_probability),
                rtol=1e-6,
            )
        self.assertAlmostEqual(
            diagnostics["diagnostics"]["shared.layer_0"]["signal"],  # type: ignore[index]
            3.0,
        )
        self.assertEqual(diagnostics["scope"]["samples_evaluated"], 5)  # type: ignore[index]
        self.assertEqual(diagnostics["scope"]["forward_batches"], 6)  # type: ignore[index]
        self.assertEqual(diagnostics["scope"]["forwarded_samples"], 10)  # type: ignore[index]
        self.assertTrue(diagnostics["scope"]["full_split"])  # type: ignore[index]

    def test_external_embedding_dimension_mismatch_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            embed_file = Path(tmpdir) / "gene_embedding.csv"
            pd.DataFrame([[1.0, 2.0]], index=pd.Index(["APOE"], name="Gene")).to_csv(embed_file)

            with self.assertRaisesRegex(ValueError, "Embedding dimension mismatch"):
                multitask.build_embedding_for_selected_genes(
                    gene_names=["APOE"],
                    embed_dim=3,
                    seed=7,
                    embed_file=embed_file,
                )

    def test_mapped_only_candidate_genes_match_embedding_keys_case_insensitively(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            x_file = Path(tmpdir) / "X.csv"
            embed_file = Path(tmpdir) / "ppi_node_embedding.csv"
            pd.DataFrame(columns=["sample_id", "apoe", "TP53", "unmapped"]).to_csv(x_file, index=False)
            pd.DataFrame(
                [[1.0, 2.0], [3.0, 4.0]],
                index=pd.Index(["APOE", "tp53"], name="Gene"),
            ).to_csv(embed_file)
            args = Namespace(x_file=x_file, embed_file=embed_file, embedding_gene_policy="mapped_only")

            candidates, manifest = multitask.resolve_candidate_genes(args)

        self.assertEqual(candidates, ["apoe", "TP53"])
        self.assertTrue(manifest["candidate_gene_filter_applied"])
        self.assertEqual(manifest["expression_genes_before_filter"], 3)
        self.assertEqual(manifest["expression_genes_after_filter"], 2)


class MultitaskFeatureSelectionTests(unittest.TestCase):
    def test_mad_selection_uses_train_split_only(self) -> None:
        dataset = PreparedDataset(
            class_names=["Control", "MCI", "AD"],
            gene_names=["train_variable", "val_variable", "flat"],
            train_ids=np.array(["s1", "s2", "s3"]),
            val_ids=np.array(["s4", "s5"]),
            test_ids=np.array(["s6"]),
            train_gene_x=np.array(
                [
                    [0.0, 1.0, 0.5],
                    [10.0, 1.0, 0.5],
                    [20.0, 1.0, 0.5],
                ],
                dtype=np.float32,
            ),
            val_gene_x=np.array([[1.0, 100.0, 0.5], [1.0, -100.0, 0.5]], dtype=np.float32),
            test_gene_x=np.array([[1.0, 50.0, 0.5]], dtype=np.float32),
            train_y=np.array([0, 1, 2], dtype=np.int64),
            val_y=np.array([0, 1], dtype=np.int64),
            test_y=np.array([2], dtype=np.int64),
        )

        selected, details = multitask.select_simple_gene_ranking(dataset, max_genes=1, method="mad")

        self.assertEqual(selected.gene_names, ["train_variable"])
        self.assertEqual(details.loc[0, "gene"], "train_variable")
        self.assertEqual(details.loc[0, "selection_method"], "mad")
        self.assertGreater(details.loc[0, "selection_score"], 0.0)

    def test_ad_mci_priority_anova_union_records_primary_quota(self) -> None:
        train_x = np.array(
            [
                [0.0, 0.0, 0.0, 0.1],
                [0.0, 0.0, 0.0, 0.2],
                [4.0, 0.0, 4.0, 0.3],
                [4.0, 0.0, 4.0, 0.4],
                [0.0, 4.0, 4.0, 0.5],
                [0.0, 4.0, 4.0, 0.6],
            ],
            dtype=np.float32,
        )
        dataset = PreparedDataset(
            class_names=["Control", "MCI", "AD"],
            gene_names=["ad_mci_signal", "mci_ctl_signal", "ad_ctl_signal", "noise"],
            train_ids=np.array(["c1", "c2", "m1", "m2", "a1", "a2"]),
            val_ids=np.array(["v1"]),
            test_ids=np.array(["t1"]),
            train_gene_x=train_x,
            val_gene_x=np.zeros((1, 4), dtype=np.float32),
            test_gene_x=np.zeros((1, 4), dtype=np.float32),
            train_y=np.array([0, 0, 1, 1, 2, 2], dtype=np.int64),
            val_y=np.array([0], dtype=np.int64),
            test_y=np.array([1], dtype=np.int64),
        )
        task_specs = multitask.build_task_specs(dataset.class_names)

        selected, details = multitask.select_weighted_pairwise_anova_union(
            dataset,
            task_specs,
            max_genes=3,
            ad_mci_gene_fraction=0.5,
        )

        self.assertEqual(len(selected.gene_names), 3)
        self.assertEqual(set(details["selection_method"]), {"ad_mci_priority_anova_union"})
        self.assertEqual(int(details.loc[0, "AD_vs_MCI_quota"]), 2)


class MultitaskAugmentationTests(unittest.TestCase):
    def test_balanced_class_batch_sampler_has_equal_counts_without_synthetic_rows(self) -> None:
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2], dtype=np.int64)
        sampler = multitask.BalancedClassBatchSampler(labels, batch_size=6, seed=11)

        batches = list(sampler)

        self.assertEqual(len(batches), len(labels) // 6)
        for batch in batches:
            self.assertEqual(np.bincount(labels[batch], minlength=3).tolist(), [2, 2, 2])
            self.assertTrue(all(0 <= index < len(labels) for index in batch))

    def test_balanced_class_batch_sampler_requires_divisible_batch_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible by 3"):
            multitask.BalancedClassBatchSampler(np.array([0, 0, 1, 1, 2, 2]), batch_size=8, seed=1)

    def test_sanitize_augmented_features_clips_only_minmax_scaled_values(self) -> None:
        raw = np.array([[-2.0, 0.5, 2.0, np.nan, np.inf, -np.inf]], dtype=np.float32)

        minmax = multitask.sanitize_augmented_features(raw, scaler="minmax")
        standard = multitask.sanitize_augmented_features(raw, scaler="standard")

        np.testing.assert_allclose(minmax, np.array([[0.0, 0.5, 1.0, 0.0, 1.0, 0.0]], dtype=np.float32))
        np.testing.assert_allclose(standard, np.array([[-2.0, 0.5, 2.0, 0.0, 0.0, 0.0]], dtype=np.float32))

    def test_ctgan_requires_minmax_scaler(self) -> None:
        dataset = PreparedDataset(
            class_names=["Control", "MCI", "AD"],
            gene_names=["g1", "g2"],
            train_ids=np.array(["s1", "s2", "s3"]),
            val_ids=np.array(["s4"]),
            test_ids=np.array(["s5"]),
            train_gene_x=np.array([[0.0, -1.0], [0.5, 0.0], [1.0, 1.0]], dtype=np.float32),
            val_gene_x=np.array([[0.1, 0.2]], dtype=np.float32),
            test_gene_x=np.array([[0.3, 0.4]], dtype=np.float32),
            train_y=np.array([0, 1, 2], dtype=np.int64),
            val_y=np.array([0], dtype=np.int64),
            test_y=np.array([1], dtype=np.int64),
        )
        args = Namespace(augmentation="ctgan", scaler="standard")

        with self.assertRaisesRegex(ValueError, "requires --scaler minmax"):
            multitask.apply_train_augmentation(dataset, args)

    def test_borderline_smote_skips_without_import_when_classes_are_balanced(self) -> None:
        dataset = PreparedDataset(
            class_names=["Control", "MCI", "AD"],
            gene_names=["g1", "g2"],
            train_ids=np.array(["s1", "s2", "s3"]),
            val_ids=np.array(["s4"]),
            test_ids=np.array(["s5"]),
            train_gene_x=np.array([[0.0, 0.1], [0.5, 0.6], [1.0, 0.9]], dtype=np.float32),
            val_gene_x=np.array([[0.1, 0.2]], dtype=np.float32),
            test_gene_x=np.array([[0.3, 0.4]], dtype=np.float32),
            train_y=np.array([0, 1, 2], dtype=np.int64),
            val_y=np.array([0], dtype=np.int64),
            test_y=np.array([1], dtype=np.int64),
        )
        args = Namespace(
            augmentation="borderline_smote",
            seed=11,
            smote_k_neighbors=5,
            smote_m_neighbors=10,
            smote_kind="borderline-1",
        )

        augmented, manifest = multitask.apply_train_augmentation(dataset, args)

        self.assertIs(augmented, dataset)
        self.assertEqual(manifest["augmentation"], "borderline_smote")
        self.assertEqual(manifest["status"], "skipped_not_enough_minority_or_already_balanced")
        self.assertEqual(manifest["validation_and_test_scope"], "not_augmented")

    def test_pca_neighbor_augmentation_is_mci_ctl_task_specific(self) -> None:
        dataset = PreparedDataset(
            class_names=["Control", "MCI", "AD"],
            gene_names=["g1", "g2", "g3"],
            train_ids=np.array(["c1", "c2", "c3", "c4", "m1", "m2", "a1", "a2"]),
            val_ids=np.array(["v1"]), test_ids=np.array(["t1"]),
            train_gene_x=np.array([
                [0.0, 0.1, 0.2], [0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5],
                [0.5, 0.6, 0.7], [0.7, 0.8, 0.9], [0.9, 0.2, 0.1], [0.8, 0.1, 0.2],
            ], dtype=np.float32),
            val_gene_x=np.zeros((1, 3), dtype=np.float32), test_gene_x=np.zeros((1, 3), dtype=np.float32),
            train_y=np.array([0, 0, 0, 0, 1, 1, 2, 2], dtype=np.int64),
            val_y=np.array([0], dtype=np.int64), test_y=np.array([1], dtype=np.int64),
        )
        args = Namespace(seed=7, pca_neighbor_components=2, pca_neighbor_k=1, pca_neighbor_gap_fraction=0.5)

        augmented, manifest = multitask.apply_pca_neighbor_mci_ctl_augmentation(dataset, args)
        synthetic_id = str(augmented.train_ids[-1])
        _, masks = multitask.make_task_targets_for_samples(
            augmented.train_y[-1:], multitask.build_task_specs(dataset.class_names), np.array([synthetic_id])
        )

        self.assertEqual(len(augmented.train_ids), len(dataset.train_ids) + 1)
        self.assertEqual(manifest["n_synthetic"], 1)
        self.assertTrue(synthetic_id.startswith(multitask.PCA_MCI_CTL_SYNTHETIC_PREFIX))
        self.assertEqual(masks.tolist(), [[False, False, True]])

    def test_row_normalized_ppi_prior_has_controlled_scale_and_zero_missing_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "ppi.csv"
            pd.DataFrame(
                [[1.0, 2.0, 3.0, 4.0], [4.0, 1.0, 0.0, 2.0]],
                index=pd.Index(["G1", "g2"], name="Gene"),
            ).to_csv(source)
            prior, manifest = multitask.build_row_normalized_ppi_prior(
                ["G1", "G2", "missing"], embed_dim=4, embed_file=source, init_scale=0.02
            )

        values = prior.to_numpy(dtype=np.float64)
        np.testing.assert_allclose(np.linalg.norm(values[:2], axis=1), np.array([0.04, 0.04]), atol=1e-6)
        np.testing.assert_allclose(values[2], np.zeros(4))
        self.assertEqual(manifest["matched_genes"], 2)

    def test_pca_neighbor_all_tasks_uses_normal_biological_task_masks(self) -> None:
        dataset = PreparedDataset(
            class_names=["Control", "MCI", "AD"],
            gene_names=["g1", "g2", "g3"],
            train_ids=np.array(["c1", "c2", "c3", "m1", "m2", "a1", "a2", "a3", "a4"]),
            val_ids=np.array(["v1"]), test_ids=np.array(["t1"]),
            train_gene_x=np.array([
                [0.0, 0.1, 0.2], [0.1, 0.2, 0.3], [0.2, 0.3, 0.4],
                [0.5, 0.6, 0.7], [0.7, 0.8, 0.9],
                [0.9, 0.2, 0.1], [0.8, 0.1, 0.2], [0.85, 0.15, 0.25], [0.75, 0.25, 0.15],
            ], dtype=np.float32),
            val_gene_x=np.zeros((1, 3), dtype=np.float32), test_gene_x=np.zeros((1, 3), dtype=np.float32),
            train_y=np.array([0, 0, 0, 1, 1, 2, 2, 2, 2], dtype=np.int64),
            val_y=np.array([0], dtype=np.int64), test_y=np.array([1], dtype=np.int64),
        )
        args = Namespace(
            seed=7, pca_neighbor_components=2, pca_neighbor_k=1,
            pca_neighbor_gap_fraction=1.0, pca_neighbor_target_count=6,
        )

        augmented, manifest = multitask.apply_pca_neighbor_all_tasks_augmentation(dataset, args)
        synthetic_y = augmented.train_y[len(dataset.train_y):]
        synthetic_ids = augmented.train_ids[len(dataset.train_ids):]
        _, masks = multitask.make_task_targets_for_samples(
            synthetic_y, multitask.build_task_specs(dataset.class_names), synthetic_ids
        )

        self.assertEqual(manifest["generated_synthetic_by_class"], {"Control": 3, "MCI": 4, "AD": 2})
        self.assertEqual(manifest["n_synthetic"], 9)
        self.assertEqual(manifest["target_class_count"], 6)
        self.assertTrue(masks.any(axis=0).all())


class MultitaskCheckpointTests(unittest.TestCase):
    def test_primary_auc_checkpoint_prioritizes_ad_mci_with_auxiliary_support(self) -> None:
        val_results = {
            "AD_vs_MCI": {"roc_auc": 0.70, "macro_f1": 0.55},
            "AD_vs_CTL": {"roc_auc": 0.90, "macro_f1": 0.80},
            "MCI_vs_CTL": {"roc_auc": 0.60, "macro_f1": 0.52},
        }

        value, details = multitask.compute_checkpoint_value(
            checkpoint_metric="val_primary_auc_minus_025_loss",
            val_results=val_results,
            val_loss=0.4,
            primary_task="AD_vs_MCI",
        )

        expected_aux = (0.90 + 0.60) / 2.0
        expected = 0.75 * 0.70 + 0.25 * expected_aux - 0.25 * 0.4
        self.assertAlmostEqual(value, expected)
        self.assertAlmostEqual(details["val_primary_roc_auc"], 0.70)
        self.assertAlmostEqual(details["val_auxiliary_auc_mean"], expected_aux)

    def test_primary_auc_checkpoint_falls_back_when_primary_auc_is_nan(self) -> None:
        val_results = {
            "AD_vs_MCI": {"roc_auc": np.nan, "macro_f1": 0.60},
            "AD_vs_CTL": {"roc_auc": 0.80, "macro_f1": 0.70},
            "MCI_vs_CTL": {"roc_auc": 0.70, "macro_f1": 0.65},
        }

        value, details = multitask.compute_checkpoint_value(
            checkpoint_metric="val_primary_auc_minus_025_loss",
            val_results=val_results,
            val_loss=0.2,
            primary_task="AD_vs_MCI",
        )

        expected_aux = (0.80 + 0.70) / 2.0
        expected = 0.75 * 0.60 + 0.25 * expected_aux - 0.25 * 0.2
        self.assertAlmostEqual(value, expected)
        self.assertEqual(details["val_primary_checkpoint_signal_source"], "macro_f1_fallback")
        self.assertAlmostEqual(details["val_primary_checkpoint_signal"], 0.60)


class MultitaskGridSummaryTests(unittest.TestCase):
    def test_balanced_grid_preset_fills_defaults_without_overriding_explicit_values(self) -> None:
        args = Namespace(
            preset="balanced",
            embedding_sources=None,
            d_models=[128],
            dropouts=None,
            max_genes_list=None,
            gene_selections=None,
            ad_mci_gene_fractions=None,
            augmentations=None,
        )

        final_grid.apply_grid_preset(args)

        self.assertEqual(args.embedding_sources, ["ppi"])
        self.assertEqual(args.d_models, [128])
        self.assertEqual(args.dropouts, [0.3, 0.4, 0.5])
        self.assertEqual(args.max_genes_list, [512, 1000, 1500, 2000])
        self.assertEqual(args.gene_selections, ["mad", "ad_mci_priority_anova_union", "pairwise_anova_union"])
        self.assertEqual(args.ad_mci_gene_fractions, [0.5, 0.6])
        self.assertEqual(args.augmentations, ["none", "borderline_smote"])

    def test_custom_grid_preset_requires_all_grid_lists(self) -> None:
        args = Namespace(
            preset="custom",
            embedding_sources=["ppi"],
            d_models=None,
            dropouts=[0.4],
            max_genes_list=[1000],
            gene_selections=["mad"],
            ad_mci_gene_fractions=[0.5],
            augmentations=["none"],
        )

        with self.assertRaisesRegex(ValueError, "d_models"):
            final_grid.apply_grid_preset(args)

    def test_grid_jobs_do_not_duplicate_fraction_insensitive_selectors(self) -> None:
        args = Namespace(
            result_root=Path("/tmp/grid"),
            embedding_sources=["random"],
            gene_selections=["mad", "ad_mci_priority_anova_union"],
            max_genes_list=[1000],
            augmentations=["none"],
            d_models=[128],
            dropouts=[0.4],
            ad_mci_gene_fractions=[0.5, 0.6],
            d_ff_multiplier=4,
            ppi_embedding_template=Path("/tmp/ppi_dim{dim}.csv"),
            ppi_gene_policy="mapped_only",
            skip_missing_ppi=False,
            limit=None,
        )

        jobs, skipped = final_grid.build_jobs(args)

        self.assertEqual(skipped, [])
        self.assertEqual(len(jobs), 3)
        self.assertEqual(
            [job.job_name for job in jobs],
            [
                "random_mad_k1000_none_d128_ff512_do0p4_admci0p5",
                "random_ad-mci-priority-anova-union_k1000_none_d128_ff512_do0p4_admci0p5",
                "random_ad-mci-priority-anova-union_k1000_none_d128_ff512_do0p4_admci0p6",
            ],
        )

    def test_grid_input_validation_fails_when_no_runnable_jobs(self) -> None:
        args = Namespace(dry_run=False, x_file=Path("/tmp/X.csv"), y_file=Path("/tmp/y.csv"))

        with self.assertRaisesRegex(ValueError, "zero runnable jobs"):
            final_grid.validate_grid_inputs(args, jobs=[])

    def test_grid_input_validation_fails_with_actionable_missing_dataset_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = Namespace(
                dry_run=False,
                x_file=Path(tmpdir) / "missing_X.csv",
                y_file=Path(tmpdir) / "missing_y.csv",
            )
            jobs = [
                final_grid.GridJob(
                    job_name="job",
                    result_root=Path(tmpdir) / "job",
                    embedding_source="random",
                    embed_file=None,
                    embedding_gene_policy="all",
                    gene_selection="mad",
                    max_genes=1000,
                    augmentation="none",
                    d_model=128,
                    d_ff=512,
                    dropout=0.4,
                    ad_mci_gene_fraction=0.5,
                )
            ]

            with self.assertRaisesRegex(FileNotFoundError, "build_txt_pairwise_datasets.py"):
                final_grid.validate_grid_inputs(args, jobs)

    def test_ranked_summary_keeps_each_architecture_for_primary_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result_root = Path(tmpdir)
            cv_dir = result_root / "job_a" / "cv5"
            cv_dir.mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "evaluation": "cv5",
                        "arch": "arch_low",
                        "split": "val",
                        "task": "AD_vs_MCI",
                        "roc_auc_mean": 0.60,
                        "macro_f1_mean": 0.50,
                        "balanced_accuracy_mean": 0.51,
                    },
                    {
                        "evaluation": "cv5",
                        "arch": "arch_high",
                        "split": "val",
                        "task": "AD_vs_MCI",
                        "roc_auc_mean": 0.70,
                        "macro_f1_mean": 0.55,
                        "balanced_accuracy_mean": 0.56,
                    },
                    {
                        "evaluation": "cv5",
                        "arch": "arch_low",
                        "split": "val",
                        "task": "AD_vs_CTL",
                        "roc_auc_mean": 0.80,
                        "macro_f1_mean": 0.75,
                    },
                    {
                        "evaluation": "cv5",
                        "arch": "arch_high",
                        "split": "val",
                        "task": "AD_vs_CTL",
                        "roc_auc_mean": 0.82,
                        "macro_f1_mean": 0.76,
                    },
                ]
            ).to_csv(cv_dir / "cv5_summary.csv", index=False)

            ranked = final_grid.collect_ranked_summary(result_root)

        self.assertEqual(ranked["arch"].tolist(), ["arch_high", "arch_low"])
        self.assertEqual(ranked["primary_roc_auc_mean"].tolist(), [0.70, 0.60])

    def test_write_ranked_summary_collects_existing_grid_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result_root = Path(tmpdir)
            cv_dir = result_root / "job_a" / "cv5"
            cv_dir.mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "evaluation": "cv5",
                        "arch": "arch_a",
                        "split": "val",
                        "task": "AD_vs_MCI",
                        "roc_auc_mean": 0.68,
                        "macro_f1_mean": 0.57,
                        "balanced_accuracy_mean": 0.58,
                    },
                    {
                        "evaluation": "cv5",
                        "arch": "arch_a",
                        "split": "val",
                        "task": "AD_vs_CTL",
                        "roc_auc_mean": 0.84,
                        "macro_f1_mean": 0.74,
                    },
                ]
            ).to_csv(cv_dir / "cv5_summary.csv", index=False)

            ranked_path = final_grid.write_ranked_summary(result_root)

            self.assertEqual(ranked_path, result_root / "grid_ranked_summary.csv")
            ranked = pd.read_csv(ranked_path)
            self.assertEqual(ranked.loc[0, "job_name"], "job_a")
            self.assertAlmostEqual(ranked.loc[0, "primary_roc_auc_mean"], 0.68)

    def test_write_ranked_summary_writes_header_when_no_runs_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result_root = Path(tmpdir)

            ranked_path = final_grid.write_ranked_summary(result_root)

            ranked = pd.read_csv(ranked_path)
            self.assertIn("primary_roc_auc_mean", ranked.columns)
            self.assertTrue(ranked.empty)


class MultitaskFinalPipelineTests(unittest.TestCase):
    def _pipeline_args(self, tmpdir: str) -> Namespace:
        root = Path(tmpdir)
        return Namespace(
            python_exe="python",
            source_x_file=root / "source" / "X.csv",
            source_y_file=root / "source" / "y.csv",
            source_split_file=root / "source" / "splits" / "official_seed42.csv",
            dataset_output_dir=root / "processed" / "txt_pairwise_multitask",
            shared_dataset_dir=None,
            skip_dataset_build=False,
            skip_ppi_build=False,
            skip_existing_ppi=False,
            hippie_file=root / "HIPPIE-current.mitab.txt",
            ppi_result_root=root / "results" / "pretraining" / "ppi_init",
            ppi_score_threshold=0.73,
            ppi_seed=42,
            ppi_component_policy="largest",
            ppi_device="cpu",
            ppi_epochs=3,
            ppi_batch_size=256,
            ppi_lr=0.01,
            ppi_walk_length=20,
            ppi_context_size=10,
            ppi_walks_per_node=10,
            ppi_negative_samples=5,
            ppi_max_pairs_per_epoch=1000,
            ppi_skip_node2vec=False,
            ppi_dims=None,
            grid_result_root=root / "results" / "paper_comparison" / "txt_multitask_final_grid",
            preset="pilot",
            run_mode="cv",
            device="cpu",
            architectures=["1l2h"],
            embedding_sources=None,
            d_models=[64, 128],
            dropouts=None,
            max_genes_list=None,
            gene_selections=None,
            ad_mci_gene_fractions=None,
            augmentations=None,
            ppi_gene_policy="mapped_only",
            grid_skip_existing=True,
            grid_skip_missing_ppi=False,
            smoke=False,
            dry_run=True,
        )

    def test_final_pipeline_builds_dataset_ppi_and_grid_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = self._pipeline_args(tmpdir)

            steps = final_pipeline.build_pipeline_steps(args, grid_passthrough=["--batch-size", "8"])

        self.assertEqual([step.name for step in steps], ["build_dataset", "build_ppi_dim64", "build_ppi_dim128", "run_grid"])
        dataset_cmd = steps[0].command
        self.assertIn("build_txt_pairwise_datasets.py", dataset_cmd[2])
        self.assertIn("--output-dir", dataset_cmd)

        ppi_dim64 = steps[1].command
        self.assertIn("build_ppi_embedding.py", ppi_dim64[2])
        self.assertIn("--embedding-dim", ppi_dim64)
        self.assertEqual(ppi_dim64[ppi_dim64.index("--embedding-dim") + 1], "64")
        self.assertIn("hippie_highconf_dim64_score0p73_seed42", ppi_dim64[ppi_dim64.index("--result-dir") + 1])

        grid_cmd = steps[-1].command
        self.assertIn("run_txt_multitask_final_grid.py", grid_cmd[2])
        self.assertIn("--ppi-embedding-template", grid_cmd)
        self.assertIn("hippie_highconf_dim{dim}_score0p73_seed42", grid_cmd[grid_cmd.index("--ppi-embedding-template") + 1])
        self.assertEqual(grid_cmd[grid_cmd.index("--d-models") + 1 : grid_cmd.index("--d-models") + 3], ["64", "128"])
        self.assertIn("--skip-existing", grid_cmd)
        self.assertEqual(grid_cmd[-2:], ["--batch-size", "8"])

    def test_final_pipeline_custom_preset_requires_ppi_or_model_dims(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = self._pipeline_args(tmpdir)
            args.preset = "custom"
            args.d_models = None
            args.ppi_dims = None

            with self.assertRaisesRegex(ValueError, "--ppi-dims or --d-models"):
                final_pipeline.infer_ppi_dims(args)


class MultitaskSeedFollowupTests(unittest.TestCase):
    def test_seed_followup_selects_top_val_cv_rows_and_builds_seed_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            grid_root = Path(tmpdir) / "grid"
            grid_root.mkdir()
            x_file = Path(tmpdir) / "X.csv"
            y_file = Path(tmpdir) / "y.csv"
            x_file.write_text("sample_id,g1\ns1,0.1\n", encoding="utf-8")
            y_file.write_text("sample_id,label,label_name\ns1,0,Control\n", encoding="utf-8")
            pd.DataFrame(
                [
                    {
                        "job_name": "job_low",
                        "evaluation": "cv5",
                        "arch": "txt_multitask_1l2h_d128_ff512_do0p4_b16_sharedpool_none_random",
                        "split": "val",
                        "primary_roc_auc_mean": 0.60,
                        "primary_macro_f1_mean": 0.50,
                        "auxiliary_roc_auc_mean": 0.70,
                    },
                    {
                        "job_name": "job_high",
                        "evaluation": "cv5",
                        "arch": "txt_multitask_1l2h_d128_ff512_do0p4_b16_sharedpool_none_random",
                        "split": "val",
                        "primary_roc_auc_mean": 0.75,
                        "primary_macro_f1_mean": 0.55,
                        "auxiliary_roc_auc_mean": 0.72,
                    },
                    {
                        "job_name": "job_test_only",
                        "evaluation": "cv5",
                        "arch": "txt_multitask_1l2h_d128_ff512_do0p4_b16_sharedpool_none_random",
                        "split": "test",
                        "primary_roc_auc_mean": 0.99,
                        "primary_macro_f1_mean": 0.99,
                        "auxiliary_roc_auc_mean": 0.99,
                    },
                ]
            ).to_csv(grid_root / "grid_ranked_summary.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "job_name": "job_low",
                        "result_root": str(grid_root / "job_low"),
                        "embedding_source": "random",
                        "embed_file": "",
                        "embedding_gene_policy": "all",
                        "gene_selection": "mad",
                        "max_genes": 1000,
                        "augmentation": "none",
                        "d_model": 128,
                        "d_ff": 512,
                        "dropout": 0.4,
                        "ad_mci_gene_fraction": 0.5,
                        "status": "planned",
                    },
                    {
                        "job_name": "job_high",
                        "result_root": str(grid_root / "job_high"),
                        "embedding_source": "random",
                        "embed_file": "",
                        "embedding_gene_policy": "all",
                        "gene_selection": "mad",
                        "max_genes": 1000,
                        "augmentation": "none",
                        "d_model": 128,
                        "d_ff": 512,
                        "dropout": 0.4,
                        "ad_mci_gene_fraction": 0.5,
                        "status": "planned",
                    },
                ]
            ).to_csv(grid_root / "grid_jobs.csv", index=False)
            (grid_root / "grid_config.json").write_text(
                json.dumps(
                    {
                        "architectures": ["1l2h"],
                        "batch_size": 16,
                        "checkpoint_metric": "val_primary_auc_minus_025_loss",
                        "class_weighting": "off",
                        "device": "cpu",
                        "early_stopping_patience": 30,
                        "epochs": 100,
                        "lr": 0.0001,
                        "python_exe": "python",
                        "run_mode": "cv",
                        "seeds": [101, 102],
                        "skip_existing": True,
                        "smoke": False,
                        "smote_k_neighbors": 5,
                        "smote_m_neighbors": 10,
                        "smote_kind": "borderline-1",
                        "weight_decay": 0.0001,
                        "x_file": str(x_file),
                        "y_file": str(y_file),
                    }
                ),
                encoding="utf-8",
            )
            args = Namespace(
                grid_root=grid_root,
                ranked_summary=None,
                result_root=Path(tmpdir) / "seed_followup",
                top_n=1,
                split="val",
                evaluation="cv5",
                python_exe=None,
                device=None,
                seeds=None,
                epochs=None,
                early_stopping_patience=None,
                skip_existing=True,
                smoke=False,
                dry_run=True,
            )

            jobs = seed_followup.build_followup_jobs(args)

        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].source_job_name, "job_high")
        command = jobs[0].command
        self.assertIn("--run-mode", command)
        self.assertEqual(command[command.index("--run-mode") + 1], "seeds")
        self.assertEqual(command[command.index("--architectures") + 1], "1l2h")
        self.assertIn("--skip-existing", command)
        self.assertEqual(command[command.index("--seeds") + 1 : command.index("--seeds") + 3], ["101", "102"])


class PpiDownloadTests(unittest.TestCase):
    def test_download_hippie_uses_explicit_ssl_context_when_downloading(self) -> None:
        class FakeResponse(BytesIO):
            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *args: object) -> None:
                return None

        with tempfile.TemporaryDirectory() as tmpdir:
            destination = Path(tmpdir) / "HIPPIE-current.mitab.txt"
            sentinel_context = object()
            seen: dict[str, object] = {}
            original_urlopen = ppi_embedding.urllib.request.urlopen
            original_context_builder = ppi_embedding.build_download_ssl_context

            def fake_urlopen(url: str, **kwargs: object) -> FakeResponse:
                seen["url"] = url
                seen["context"] = kwargs.get("context")
                return FakeResponse(b"hippie")

            try:
                ppi_embedding.urllib.request.urlopen = fake_urlopen  # type: ignore[assignment]
                ppi_embedding.build_download_ssl_context = lambda: sentinel_context  # type: ignore[assignment]

                output = ppi_embedding.download_hippie("https://example.test/hippie.txt", destination)
                output_bytes = destination.read_bytes()
            finally:
                ppi_embedding.urllib.request.urlopen = original_urlopen  # type: ignore[assignment]
                ppi_embedding.build_download_ssl_context = original_context_builder  # type: ignore[assignment]

        self.assertEqual(output, destination)
        self.assertEqual(output_bytes, b"hippie")
        self.assertEqual(seen["url"], "https://example.test/hippie.txt")
        self.assertIs(seen["context"], sentinel_context)

    def test_compact_ppi_console_report_keeps_full_report_out_of_terminal(self) -> None:
        report = {
            "target_gene_coverage": {
                "missing_gene_names": ["g1", "g2", "g3"],
            }
        }

        compact = ppi_embedding.compact_report_for_console(report, missing_gene_preview=2)

        self.assertEqual(report["target_gene_coverage"]["missing_gene_names"], ["g1", "g2", "g3"])
        self.assertEqual(compact["target_gene_coverage"]["missing_gene_names"], "<see ppi_embedding_report.json>")
        self.assertEqual(compact["target_gene_coverage"]["missing_gene_names_preview"], ["g1", "g2"])
        self.assertEqual(compact["target_gene_coverage"]["missing_gene_names_omitted_from_console"], 1)


if __name__ == "__main__":
    unittest.main()
