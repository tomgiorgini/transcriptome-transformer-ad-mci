from __future__ import annotations

import math
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from experiments.scripts.paper_comparison.train_txt_multitask import (
    clip_model_gradients,
    partition_volumetric_parameters,
)
from experiments.scripts.paper_comparison.txt_volumetric import (
    common as protocol_common,
)
from experiments.scripts.paper_comparison.txt_volumetric import run_paired_protocol
from source.models.txt import TxT
from source.models.txt_volumetric import TxTVolumetric
from source.models.txt_volumetric.layers import VolumetricAttentionAugmentation


TASK_OUTPUTS = {
    "AD_vs_MCI": 2,
    "AD_vs_CTL": 2,
    "MCI_vs_CTL": 2,
}


def _legacy_raw_volume(
    q_i: torch.Tensor,
    k_i: torch.Tensor,
    k_j: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    q = q_i.float()
    anchor = k_i.float()
    neighbor = k_j.float()
    q_q = (q * q).sum(dim=-1)
    q_anchor = (q * anchor).sum(dim=-1)
    q_neighbor = (q * neighbor).sum(dim=-1)
    anchor_anchor = (anchor * anchor).sum(dim=-1)
    anchor_neighbor = (anchor * neighbor).sum(dim=-1)
    neighbor_neighbor = (neighbor * neighbor).sum(dim=-1)
    determinant = (
        q_q * (anchor_anchor * neighbor_neighbor - anchor_neighbor.square())
        - q_anchor * (q_anchor * neighbor_neighbor - anchor_neighbor * q_neighbor)
        + q_neighbor * (q_anchor * anchor_neighbor - anchor_anchor * q_neighbor)
    )
    return torch.sqrt(torch.clamp_min(determinant, 0.0) + eps)


class VolumetricScalingTests(unittest.TestCase):
    @staticmethod
    def _operator(mode: str) -> VolumetricAttentionAugmentation:
        return VolumetricAttentionAugmentation(
            n_heads=2,
            d_head=4,
            edge_index=torch.tensor(
                [[0, 0, 1, 2, 3, 3], [1, 2, 0, 0, 1, 2]],
                dtype=torch.long,
            ),
            n_nodes=4,
            beta=1.0,
            eps=1e-8,
            dropout=0.0,
            gate_init=0.0,
            volumetric_volume_mode=mode,
        )

    @staticmethod
    def _projected_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        generator = torch.Generator().manual_seed(701)
        q = torch.randn(3, 2, 4, 4, generator=generator)
        k = torch.randn(3, 2, 4, 4, generator=generator)
        v = torch.randn(3, 2, 4, 4, generator=generator)
        tupe = torch.randn(3, 2, 4, 4, generator=generator)
        return q, k, v, tupe

    def test_raw_mode_preserves_legacy_gram_determinant_formula(self) -> None:
        operator = self._operator("raw")
        q, k, v, tupe = self._projected_inputs()

        _, _, observed, observed_logits = operator.compute_vma_output(q, k, v, tupe)
        destination, source = operator.edge_index
        q_i = q.index_select(2, destination)
        k_i = k.index_select(2, destination)
        k_j = k.index_select(2, source)
        expected = _legacy_raw_volume(
            q_i,
            k_i,
            k_j,
            eps=1e-8,
        )
        expected_logits = (
            -expected + (q_i * k_i).sum(dim=-1) + (q_i * k_j).sum(dim=-1)
        ) / math.sqrt(operator.d_head) + tupe[:, :, destination, source]

        torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)
        torch.testing.assert_close(observed_logits, expected_logits, rtol=0.0, atol=0.0)

    def test_l2_mode_volume_is_invariant_to_positive_q_k_scaling(self) -> None:
        operator = self._operator("l2")
        q, k, v, tupe = self._projected_inputs()

        _, _, reference, _ = operator.compute_vma_output(q, k, v, tupe)
        _, _, scaled, _ = operator.compute_vma_output(
            q * 7.25,
            k * 0.125,
            v,
            tupe,
        )

        torch.testing.assert_close(scaled, reference, rtol=1e-5, atol=1e-6)


class ZeroGateTrainingFairnessTests(unittest.TestCase):
    @staticmethod
    def _embedding_file(directory: str, n_genes: int = 6, d_embed: int = 4) -> Path:
        path = Path(directory) / "embedding.csv"
        pd.DataFrame(
            np.random.default_rng(17).normal(0.0, 0.02, size=(n_genes, d_embed)),
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
        return torch.tensor(
            [[0, 0, 1, 2, 3, 4, 5, 5], [1, 2, 0, 0, 4, 3, 3, 4]],
            dtype=torch.long,
        )

    def _paired_models(self, embed_file: Path) -> tuple[TxT, TxTVolumetric]:
        kwargs = self._kwargs(embed_file)
        torch.manual_seed(101)
        baseline = TxT(**kwargs)
        torch.manual_seed(101)
        volumetric = TxTVolumetric(
            **kwargs,
            edge_index=self._edge_index(),
            volumetric_gate_init=0.0,
            volumetric_dropout=0.0,
            volumetric_volume_mode="raw",
        )
        return baseline, volumetric

    @staticmethod
    def _batch() -> torch.Tensor:
        return torch.rand(9, 6, generator=torch.Generator().manual_seed(313))

    def test_train_mode_zero_gate_matches_baseline_when_vma_dropout_is_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            baseline, volumetric = self._paired_models(self._embedding_file(tmpdir))
            x = self._batch()
            baseline.train()
            volumetric.train()

            torch.manual_seed(919)
            expected = baseline(x)
            torch.manual_seed(919)
            observed = volumetric(x)

        for baseline_logits, volumetric_logits in zip(expected, observed):
            self.assertTrue(torch.equal(baseline_logits, volumetric_logits))

    def test_l2_extra_state_rejects_raw_constructor_on_strict_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            embed_file = self._embedding_file(tmpdir)
            kwargs = self._kwargs(embed_file)
            torch.manual_seed(101)
            source = TxTVolumetric(
                **kwargs,
                edge_index=self._edge_index(),
                volumetric_dropout=0.0,
                volumetric_volume_mode="l2",
            )
            torch.manual_seed(307)
            restored = TxTVolumetric(
                **kwargs,
                edge_index=self._edge_index(),
                volumetric_dropout=0.0,
                volumetric_volume_mode="raw",
            )

            with self.assertRaisesRegex(
                RuntimeError,
                "Checkpoint volumetric_volume_mode='l2'.*constructed model value 'raw'",
            ):
                restored.load_state_dict(source.state_dict(), strict=True)

    def test_first_optimizer_step_keeps_shared_parameters_paired_and_opens_gamma(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            baseline, volumetric = self._paired_models(self._embedding_file(tmpdir))
            x = self._batch()
            target_generator = torch.Generator().manual_seed(411)
            targets = [torch.randn((9, 2), generator=target_generator) for _ in TASK_OUTPUTS]
            baseline.train()
            volumetric.train()
            baseline_optimizer = torch.optim.AdamW(
                baseline.parameters(), lr=1e-3, weight_decay=1e-4
            )
            volumetric_optimizer = torch.optim.AdamW(
                volumetric.parameters(), lr=1e-3, weight_decay=1e-4
            )

            torch.manual_seed(929)
            baseline_outputs = baseline(x)
            torch.manual_seed(929)
            volumetric_outputs = volumetric(x)
            baseline_loss = sum(
                F.mse_loss(output, target)
                for output, target in zip(baseline_outputs, targets)
            )
            volumetric_loss = sum(
                F.mse_loss(output, target)
                for output, target in zip(volumetric_outputs, targets)
            )
            baseline_loss.backward()
            volumetric_loss.backward()

            baseline_parameters = dict(baseline.named_parameters())
            volumetric_parameters = dict(volumetric.named_parameters())
            for name, parameter in baseline_parameters.items():
                self.assertIn(name, volumetric_parameters)
                self.assertTrue(
                    torch.equal(parameter.grad, volumetric_parameters[name].grad),
                    name,
                )

            gamma = next(volumetric.iter_volumetric_augmentations())[1].gamma
            self.assertIsNotNone(gamma.grad)
            self.assertNotEqual(float(gamma.grad), 0.0)

            baseline_optimizer.step()
            volumetric_optimizer.step()

            for name, parameter in baseline.named_parameters():
                self.assertTrue(
                    torch.equal(parameter, dict(volumetric.named_parameters())[name]),
                    name,
                )
            self.assertNotEqual(float(gamma.detach()), 0.0)


    def test_l2_fair_candidate_flags_survive_selected_evaluation_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            temporary_root = Path(tmpdir)
            candidate_dir = temporary_root / "candidate"
            evaluation_dir = temporary_root / "selected_test"
            args = Namespace(
                python_exe="python3",
                worker=protocol_common.DEFAULT_WORKER,
                x_file=protocol_common.DEFAULT_SHARED_DATASET / "X.csv",
                y_file=protocol_common.DEFAULT_SHARED_DATASET / "y.csv",
                ppi_edge_file=protocol_common.DEFAULT_PPI_EDGE_FILE,
                ppi_score_threshold=0.73,
                volumetric_eps=1e-8,
                volumetric_gate_init=0.0,
                volumetric_volume_mode="l2",
                volumetric_dropout=0.0,
                post_model_construction_reseed="on",
                grad_clip_scope="separate_volumetric",
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
            candidate_command = run_paired_protocol.build_worker_command(
                args,
                protocol,
                seed=101,
                split_file=Path("split.csv"),
                result_dir=candidate_dir,
                model_variant="ppi_volumetric",
                beta=1.0,
                evaluate_test="off",
            )

            def option_value(command: list[str], option: str) -> str:
                index = command.index(option)
                return command[index + 1]

            expected_options = {
                "--volumetric-volume-mode": "l2",
                "--volumetric-dropout": "0.0",
                "--post-model-construction-reseed": "on",
                "--grad-clip-scope": "separate_volumetric",
            }
            for option, value in expected_options.items():
                self.assertEqual(option_value(candidate_command, option), value)

            protocol_common.save_worker_command(
                candidate_dir,
                candidate_command,
                kind="beta_candidate_train_no_test",
                seed=101,
                beta=1.0,
            )
            checkpoint = candidate_dir / "best_model.pt"
            evaluation_command = protocol_common.evaluation_command_from_selection(
                {
                    "selected_run_dir": str(candidate_dir),
                    "selected_checkpoint": str(checkpoint),
                },
                evaluation_dir,
            )

            for option, value in expected_options.items():
                self.assertEqual(option_value(evaluation_command, option), value)
            self.assertEqual(option_value(evaluation_command, "--evaluate-test"), "on")
            self.assertEqual(
                option_value(evaluation_command, "--evaluation-only-checkpoint"),
                str(checkpoint),
            )

    def test_separate_vma_clipping_preserves_paired_shared_first_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            baseline, volumetric = self._paired_models(self._embedding_file(tmpdir))
            x = self._batch()
            target_generator = torch.Generator().manual_seed(419)
            targets = [torch.randn((9, 2), generator=target_generator) for _ in TASK_OUTPUTS]
            baseline.train()
            volumetric.train()
            baseline_optimizer = torch.optim.AdamW(
                baseline.parameters(), lr=1e-3, weight_decay=1e-4
            )
            volumetric_optimizer = torch.optim.AdamW(
                volumetric.parameters(), lr=1e-3, weight_decay=1e-4
            )

            torch.manual_seed(937)
            baseline_outputs = baseline(x)
            torch.manual_seed(937)
            volumetric_outputs = volumetric(x)
            baseline_loss = sum(
                F.mse_loss(output, target)
                for output, target in zip(baseline_outputs, targets)
            )
            volumetric_loss = sum(
                F.mse_loss(output, target)
                for output, target in zip(volumetric_outputs, targets)
            )
            baseline_loss.backward()
            volumetric_loss.backward()

            baseline_non_vma, baseline_vma = partition_volumetric_parameters(baseline)
            volumetric_non_vma, volumetric_vma = partition_volumetric_parameters(volumetric)
            self.assertFalse(baseline_vma)
            self.assertTrue(volumetric_vma)

            def gradient_norm(parameters: list[torch.nn.Parameter]) -> float:
                squared_norm = sum(
                    float(parameter.grad.detach().float().square().sum())
                    for parameter in parameters
                    if parameter.grad is not None
                )
                return math.sqrt(squared_norm)

            max_norm = 1e-4
            self.assertGreater(gradient_norm(baseline_non_vma), max_norm)
            self.assertGreater(gradient_norm(volumetric_non_vma), max_norm)
            self.assertGreater(gradient_norm(volumetric_vma), max_norm)

            clip_model_gradients(
                baseline,
                max_norm,
                scope="separate_volumetric",
                model_variant="baseline",
            )
            clip_model_gradients(
                volumetric,
                max_norm,
                scope="separate_volumetric",
                model_variant="ppi_volumetric",
            )

            self.assertLessEqual(gradient_norm(baseline_non_vma), max_norm * 1.001)
            self.assertLessEqual(gradient_norm(volumetric_non_vma), max_norm * 1.001)
            self.assertLessEqual(gradient_norm(volumetric_vma), max_norm * 1.001)
            baseline_parameters = dict(baseline.named_parameters())
            volumetric_parameters = dict(volumetric.named_parameters())
            for name, parameter in baseline_parameters.items():
                self.assertTrue(
                    torch.equal(parameter.grad, volumetric_parameters[name].grad),
                    name,
                )

            gamma = next(volumetric.iter_volumetric_augmentations())[1].gamma
            self.assertIsNotNone(gamma.grad)
            self.assertNotEqual(float(gamma.grad), 0.0)
            baseline_optimizer.step()
            volumetric_optimizer.step()

            for name, parameter in baseline.named_parameters():
                self.assertTrue(
                    torch.equal(parameter, dict(volumetric.named_parameters())[name]),
                    name,
                )
            self.assertNotEqual(float(gamma.detach()), 0.0)


if __name__ == "__main__":
    unittest.main()
