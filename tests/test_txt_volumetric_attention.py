from __future__ import annotations

import json
import math
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.scripts.paper_comparison.txt_volumetric import common as protocol_common
from experiments.scripts.paper_comparison.txt_volumetric import run_paired_protocol
from source.models.txt.model import TxT
from source.models.txt_volumetric import (
    TxTVolumetric,
    VolumetricAttentionAugmentation,
    load_induced_ppi_graph,
    segment_softmax,
    volumetric_volume,
)


TASK_OUTPUTS = {"AD_vs_MCI": 2, "AD_vs_CTL": 2, "MCI_vs_CTL": 2}


class PpiGraphTests(unittest.TestCase):
    def test_hippie_parsing_threshold_deduplication_alignment_and_isolates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            edge_file = Path(tmpdir) / "hippie.csv"
            pd.DataFrame(
                [
                    (" g1 ", "G2", 0.73),
                    ("g2", "G1", 0.80),  # duplicate: retain maximum confidence
                    ("G1", "g1", 0.99),  # self-loop
                    ("G2", "G3", 0.729),  # below inclusive threshold
                    ("g2", "g3", 0.90),
                    ("G3", "NOT_SELECTED", 0.95),
                ],
                columns=["protein1", "protein2", "score"],
            ).to_csv(edge_file, index=False)

            graph = load_induced_ppi_graph(
                edge_file,
                ["G3", "g1", "G2", "isolated"],
                score_threshold=0.73,
            )

        self.assertEqual(graph.genes, ("G3", "G1", "G2", "ISOLATED"))
        self.assertEqual(graph.n_undirected_edges, 2)
        self.assertEqual(graph.n_directed_edges, 4)
        self.assertEqual(graph.degrees.tolist(), [1, 1, 2, 0])
        self.assertEqual(graph.isolated_genes, ["ISOLATED"])
        self.assertEqual(graph.self_loops_removed, 1)
        self.assertEqual(graph.duplicate_rows_removed, 1)
        self.assertEqual(len(graph.source_sha256), 64)
        directed = set(map(tuple, graph.edge_index.transpose(0, 1).tolist()))
        self.assertEqual(directed, {(1, 2), (2, 1), (2, 0), (0, 2)})
        scores = graph.to_edge_frame().set_index(["target_index", "source_index"])["score"]
        self.assertAlmostEqual(float(scores.loc[(1, 2)]), 0.80)
        self.assertAlmostEqual(float(scores.loc[(2, 0)]), 0.90)


class VolumetricPrimitiveTests(unittest.TestCase):
    def test_volume_matches_direct_gram_determinant(self) -> None:
        generator = torch.Generator().manual_seed(17)
        q = torch.randn(2, 3, 5, generator=generator, dtype=torch.float64)
        anchor = torch.randn(2, 3, 5, generator=generator, dtype=torch.float64)
        neighbor = torch.randn(2, 3, 5, generator=generator, dtype=torch.float64)

        observed = volumetric_volume(q, anchor, neighbor, eps=1e-8)
        z = torch.stack((q, anchor, neighbor), dim=-1).float()
        expected = torch.sqrt(torch.linalg.det(z.transpose(-2, -1) @ z).clamp_min(0) + 1e-8)

        self.assertEqual(observed.dtype, torch.float32)
        self.assertTrue(torch.allclose(observed, expected, atol=1e-6, rtol=1e-5))

    def test_volume_has_finite_gradients_for_degenerate_triplets(self) -> None:
        q = torch.tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
        anchor = (2.0 * q).detach().requires_grad_(True)
        neighbor = (3.0 * q).detach().requires_grad_(True)

        value = volumetric_volume(q, anchor, neighbor, eps=1e-8).sum()
        value.backward()

        self.assertTrue(torch.isfinite(value))
        for tensor in (q, anchor, neighbor):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())

    def test_segment_softmax_normalizes_each_destination_and_honors_mask(self) -> None:
        logits = torch.tensor(
            [
                [[1.0, 2.0, -1.0, 0.0, 3.0]],
                [[-2.0, 1.0, 4.0, 4.0, 4.0]],
            ]
        )
        destinations = torch.tensor([0, 0, 1, 1, 1])
        mask = torch.tensor([True, False, True, True, False])

        weights = segment_softmax(logits, destinations, num_segments=3, mask=mask)

        self.assertTrue(torch.equal(weights[..., ~mask], torch.zeros_like(weights[..., ~mask])))
        for batch_idx in range(weights.size(0)):
            self.assertAlmostEqual(float(weights[batch_idx, 0, destinations == 0].sum()), 1.0)
            self.assertAlmostEqual(float(weights[batch_idx, 0, destinations == 1].sum()), 1.0)


class VolumetricOperatorTests(unittest.TestCase):
    def _operator(self, edge_index: torch.Tensor) -> VolumetricAttentionAugmentation:
        return VolumetricAttentionAugmentation(
            n_heads=2,
            d_head=4,
            edge_index=edge_index,
            n_nodes=4,
            beta=1.5,
            eps=1e-8,
            dropout=0.0,
            gate_init=0.0,
        )

    def test_tupe_changes_patient_weights_isolates_stay_zero_and_gradients_are_finite(self) -> None:
        # Node 0 has two neighbors, nodes 1/2 one neighbor each, node 3 is isolated.
        edge_index = torch.tensor([[0, 0, 1, 2], [1, 2, 0, 0]], dtype=torch.long)
        operator = self._operator(edge_index)
        generator = torch.Generator().manual_seed(31)
        base_q = torch.randn(1, 2, 4, 4, generator=generator)
        base_k = torch.randn(1, 2, 4, 4, generator=generator)
        base_v = torch.randn(1, 2, 4, 4, generator=generator)
        q = base_q.repeat(2, 1, 1, 1).requires_grad_(True)
        k = base_k.repeat(2, 1, 1, 1).requires_grad_(True)
        v = base_v.repeat(2, 1, 1, 1).requires_grad_(True)
        tupe = torch.zeros(2, 2, 4, 4)
        tupe[1, :, 0, 1] = 4.0
        tupe[1, :, 0, 2] = -4.0

        output, weights, volumes, logits = operator.compute_vma_output(q, k, v, tupe)
        destination_zero = edge_index[0] == 0

        self.assertFalse(torch.allclose(weights[0, :, destination_zero], weights[1, :, destination_zero]))
        self.assertTrue(torch.equal(output[:, :, 3], torch.zeros_like(output[:, :, 3])))
        self.assertEqual(tuple(weights.shape), (2, 2, 4))
        self.assertEqual(tuple(volumes.shape), (2, 2, 4))
        self.assertEqual(tuple(logits.shape), (2, 2, 4))
        output.square().mean().backward()
        for tensor in (q, k, v):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())

    def test_empty_graph_returns_exact_zero_branch(self) -> None:
        operator = self._operator(torch.empty((2, 0), dtype=torch.long))
        q = torch.randn(2, 2, 4, 4)
        output, weights, volumes, logits = operator.compute_vma_output(
            q,
            torch.randn_like(q),
            torch.randn_like(q),
            torch.zeros(2, 2, 4, 4),
        )

        self.assertTrue(torch.equal(output, torch.zeros_like(output)))
        self.assertEqual(tuple(weights.shape), (2, 2, 0))
        self.assertEqual(tuple(volumes.shape), (2, 2, 0))
        self.assertEqual(tuple(logits.shape), (2, 2, 0))


class TxTVolumetricModelTests(unittest.TestCase):
    @staticmethod
    def _embedding_file(directory: str, n_genes: int = 6, d_embed: int = 4) -> Path:
        path = Path(directory) / "embedding.csv"
        pd.DataFrame(
            np.random.default_rng(5).normal(0.0, 0.02, size=(n_genes, d_embed)),
            index=pd.Index([f"G{index}" for index in range(n_genes)], name="Gene"),
        ).to_csv(path)
        return path

    @staticmethod
    def _kwargs(embed_file: Path) -> dict[str, object]:
        return {
            "embed_file": str(embed_file),
            "gene_list": [f"G{index}" for index in range(6)],
            "n_heads": 2,
            "d_model": 4,
            "dropout": 0.2,
            "d_ff": 8,
            "n_layers": 1,
            "aggfunc": "Avgpool",
            "d_hidden1": 8,
            "d_hidden2": 4,
            "d_output_dict": TASK_OUTPUTS,
            "head_norm": "batch",
            "encoder_sharing": "shared",
        }

    @staticmethod
    def _edge_index() -> torch.Tensor:
        return torch.tensor([[0, 0, 1, 2, 4, 5], [1, 2, 0, 0, 5, 4]], dtype=torch.long)

    def test_zero_gate_preserves_baseline_parameters_and_logits_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            kwargs = self._kwargs(self._embedding_file(tmpdir))
            torch.manual_seed(101)
            baseline = TxT(**kwargs)
            torch.manual_seed(101)
            volumetric = TxTVolumetric(
                **kwargs,
                edge_index=self._edge_index(),
                volumetric_gate_init=0.0,
            )
            baseline_state = baseline.state_dict()
            volumetric_state = volumetric.state_dict()
            for name, value in baseline_state.items():
                self.assertIn(name, volumetric_state)
                self.assertTrue(torch.equal(value, volumetric_state[name]), name)
            baseline.eval()
            volumetric.eval()
            x = torch.rand(9, 6)
            task_mask = torch.tensor(
                [[False, True, True]] * 3
                + [[True, False, True]] * 3
                + [[True, True, False]] * 3,
                dtype=torch.bool,
            )
            with torch.no_grad():
                expected = baseline(x, task_sample_mask=task_mask)
                observed = volumetric(x, task_sample_mask=task_mask)

        self.assertEqual(volumetric.volumetric_gate_values(), {"shared.layer_0": 0.0})
        for expected_logits, observed_logits in zip(expected, observed):
            self.assertTrue(torch.equal(expected_logits, observed_logits))
            self.assertEqual(tuple(observed_logits.shape), (9, 2))

    def test_class_only_heads_still_receive_six_rows_while_encoder_receives_nine(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model = TxTVolumetric(
                **self._kwargs(self._embedding_file(tmpdir)),
                edge_index=self._edge_index(),
            )
            task_mask = torch.tensor(
                [[False, True, True]] * 3
                + [[True, False, True]] * 3
                + [[True, True, False]] * 3,
                dtype=torch.bool,
            )
            encoder_batches: list[int] = []
            head_batches: list[int] = []
            encoder_hook = model.transformer.encoder.register_forward_pre_hook(
                lambda _module, inputs: encoder_batches.append(int(inputs[0].size(0)))
            )
            head_hooks = [
                head.register_forward_pre_hook(
                    lambda _module, inputs: head_batches.append(int(inputs[0].size(0)))
                )
                for head in model.task_specific_layers
            ]
            model.train()
            model(torch.rand(9, 6), task_sample_mask=task_mask)
            encoder_hook.remove()
            for hook in head_hooks:
                hook.remove()

        self.assertEqual(encoder_batches, [9])
        self.assertEqual(head_batches, [6, 6, 6])

    def test_checkpoint_round_trip_includes_graph_and_learned_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            kwargs = self._kwargs(self._embedding_file(tmpdir))
            original = TxTVolumetric(**kwargs, edge_index=self._edge_index())
            augmentation = next(original.iter_volumetric_augmentations())[1]
            augmentation.gamma.data.fill_(0.25)
            original.eval()
            x = torch.rand(3, 6)
            with torch.no_grad():
                expected = original(x)
            checkpoint = Path(tmpdir) / "model.pt"
            torch.save(original.state_dict(), checkpoint)

            restored = TxTVolumetric(**kwargs, edge_index=self._edge_index())
            restored.load_state_dict(torch.load(checkpoint, weights_only=True))
            restored.eval()
            with torch.no_grad():
                observed = restored(x)

        self.assertTrue(torch.equal(restored.ppi_edge_index, original.ppi_edge_index))
        self.assertAlmostEqual(restored.volumetric_raw_gate_values()["shared.layer_0"], 0.25)
        for expected_logits, observed_logits in zip(expected, observed):
            self.assertTrue(torch.equal(expected_logits, observed_logits))


class PairedProtocolTests(unittest.TestCase):
    def test_beta_tie_selects_lower_candidate_without_test_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result_root = Path(tmpdir)
            for beta, score in ((0.0, 0.5), (0.5, 0.7), (1.0, 0.7), (1.5, 0.6)):
                run_dir = protocol_common.candidate_dir(result_root, 101, beta)
                run_dir.mkdir(parents=True)
                (run_dir / "best_model.pt").write_bytes(b"placeholder")
                (run_dir / "model_summary.json").write_text(
                    json.dumps(
                        {
                            "training_summary": {
                                "checkpoint_metric": protocol_common.CHECKPOINT_METRIC,
                                "best_checkpoint_value": score,
                            }
                        }
                    ),
                    encoding="utf-8",
                )

            selection = protocol_common.select_beta_for_seed(
                result_root,
                101,
                (0.0, 0.5, 1.0, 1.5),
            )

        self.assertEqual(selection.selected_beta, 0.5)
        self.assertEqual(selection.tie_break, "lower_beta_on_exact_score_tie")

    def test_fixed_worker_command_preserves_multitask_baseline_and_hides_candidate_test(self) -> None:
        args = Namespace(
            python_exe="python3",
            worker=protocol_common.DEFAULT_WORKER,
            x_file=protocol_common.DEFAULT_SHARED_DATASET / "X.csv",
            y_file=protocol_common.DEFAULT_SHARED_DATASET / "y.csv",
            ppi_edge_file=protocol_common.DEFAULT_PPI_EDGE_FILE,
            ppi_score_threshold=0.73,
            volumetric_eps=1e-8,
            volumetric_gate_init=0.0,
            early_stopping_patience=80,
            lr=1e-4,
            weight_decay=1e-4,
        )
        protocol = {
            "max_genes": 2000,
            "epochs": 200,
            "device": "cpu",
            "max_train_batches": None,
            "max_val_batches": None,
        }
        command = run_paired_protocol.build_worker_command(
            args,
            protocol,
            seed=101,
            split_file=Path("split.csv"),
            result_dir=Path("run"),
            model_variant="ppi_volumetric",
            beta=1.5,
            evaluate_test="off",
        )
        command_text = " ".join(command)

        for expected in (
            "--max-genes 2000",
            "--gene-selection variance",
            "--batch-size 9",
            "--train-sampling balanced_classes",
            "--task-loss-weights 0.5 0.25 0.25",
            "--mask-aware-heads on",
            "--n-heads 2",
            "--n-layers 1",
            "--d-model 64",
            "--d-ff 256",
            "--dropout 0.2",
            "--volumetric-volume-mode raw",
            "--checkpoint-metric val_weighted_70_15_15_auc_minus_025_loss",
            "--evaluate-test off",
        ):
            self.assertIn(expected, command_text)

    def test_selected_evaluation_removes_validation_only_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            candidate = Path(tmpdir) / "candidate"
            candidate.mkdir()
            protocol_common.write_json(
                candidate / "worker_command.json",
                {
                    "argv": [
                        "python3",
                        "worker.py",
                        "--result-dir",
                        str(candidate),
                        "--evaluate-test",
                        "off",
                        "--skip-final-test",
                    ]
                },
            )
            command = protocol_common.evaluation_command_from_selection(
                {
                    "selected_run_dir": str(candidate),
                    "selected_checkpoint": str(candidate / "best_model.pt"),
                },
                Path(tmpdir) / "selected_test",
            )

        self.assertNotIn("--skip-final-test", command)
        self.assertEqual(command[command.index("--evaluate-test") + 1], "on")
        self.assertIn("--evaluation-only-checkpoint", command)

    def test_bootstrap_interval_is_deterministic(self) -> None:
        values = np.array([-0.1, 0.0, 0.1, 0.2], dtype=float)
        first = protocol_common.bootstrap_mean_ci(values, replicates=500, seed=19)
        second = protocol_common.bootstrap_mean_ci(values, replicates=500, seed=19)
        self.assertEqual(first, second)
        self.assertLessEqual(first[0], float(values.mean()))
        self.assertGreaterEqual(first[1], float(values.mean()))


if __name__ == "__main__":
    unittest.main()
