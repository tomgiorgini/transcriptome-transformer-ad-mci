from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import torch

from experiments.scripts.paper_comparison.analyze_txt_attention import (
    _finalize_class_attention_sums,
    _scatter_sum,
    adjusted_class_contrast,
    benjamini_hochberg,
    bootstrap_mean_interval,
    build_vma_edges,
    select_matrix_genes,
)
from experiments.scripts.paper_comparison.summarize_txt_attention_runs import (
    TOP_K_VALUES,
    build_consensus,
)


class AttentionStatisticsTests(unittest.TestCase):
    def test_benjamini_hochberg_is_monotone_in_sorted_p_values(self) -> None:
        p_values = np.asarray([0.01, 0.04, 0.03, np.nan, 0.002])
        adjusted = benjamini_hochberg(p_values)

        self.assertTrue(np.isnan(adjusted[3]))
        finite = np.isfinite(p_values)
        order = np.argsort(p_values[finite])
        self.assertTrue(np.all(np.diff(adjusted[finite][order]) >= -1e-12))
        self.assertTrue(np.all(adjusted[finite] >= p_values[finite]))
        self.assertTrue(np.all(adjusted[finite] <= 1.0))

    def test_adjusted_contrast_recovers_class_effect_with_cohort_covariate(self) -> None:
        labels = np.asarray([0, 1, 0, 1, 0, 1, 0, 1])
        cohorts = np.asarray(["A", "A", "B", "B", "A", "A", "B", "B"])
        cohort_shift = (cohorts == "B").astype(float) * 5.0
        values = np.column_stack(
            [
                2.0 * labels + cohort_shift,
                -3.0 * labels + 0.5 * cohort_shift,
            ]
        )

        result = adjusted_class_contrast(values, labels, cohorts, 1, 0)

        np.testing.assert_allclose(result["adjusted_beta"], [2.0, -3.0], atol=1e-10)
        self.assertEqual(result["n_positive"], 4)
        self.assertEqual(result["n_negative"], 4)
        self.assertTrue(result["design_full_rank"])

    def test_adjusted_contrast_rejects_rank_deficient_design(self) -> None:
        labels = np.asarray([0, 0, 1, 1])
        cohorts = np.asarray(["A", "A", "B", "B"])
        values = np.column_stack([labels.astype(float), 2.0 * labels])

        result = adjusted_class_contrast(values, labels, cohorts, 1, 0)

        self.assertFalse(result["design_full_rank"])
        self.assertTrue(np.isnan(result["adjusted_beta"]).all())
        self.assertTrue(np.isnan(result["p_value"]).all())

    def test_bootstrap_transform_is_applied_after_sample_mean(self) -> None:
        values = np.asarray([[1.0], [4.0]])

        lower, upper = bootstrap_mean_interval(
            values,
            iterations=0,
            seed=1,
            transform=np.log2,
        )

        expected = np.log2(np.asarray([2.5]))
        np.testing.assert_allclose(lower, expected)
        np.testing.assert_allclose(upper, expected)


class AttentionAlignmentTests(unittest.TestCase):
    def test_class_matrix_finalization_preserves_sums_and_weighted_mean(self) -> None:
        sums = torch.tensor([[[[2.0, 0.0]]], [[[0.0, 6.0]]]])
        original = sums.clone()

        class_means, overall = _finalize_class_attention_sums(
            sums,
            np.asarray([2, 3]),
        )

        self.assertTrue(torch.equal(sums, original))
        np.testing.assert_allclose(class_means[:, 0, 0], [[1.0, 0.0], [0.0, 2.0]])
        np.testing.assert_allclose(overall[0, 0], [0.4, 1.2])
        weighted = np.tensordot(np.asarray([2, 3]) / 5.0, class_means, axes=(0, 0))
        np.testing.assert_allclose(weighted, overall)

    def test_scatter_sum_uses_gene_indices_on_last_axis(self) -> None:
        values = torch.tensor([[[0.2, 0.3, 0.5]]])
        indices = torch.tensor([1, 1, 2])

        observed = _scatter_sum(values, indices, n_genes=4)

        expected = torch.tensor([[[0.0, 0.5, 0.5, 0.0]]])
        self.assertTrue(torch.allclose(observed, expected))

    def test_matrix_selection_includes_both_key_and_query_rankings(self) -> None:
        ranking = pd.DataFrame(
            {
                "gene_index": np.arange(8),
                "key_rank": [1, 2, 3, 4, 5, 6, 7, 8],
                "query_rank": [8, 7, 6, 5, 1, 2, 3, 4],
            }
        )

        selected = select_matrix_genes(ranking, 4)

        self.assertEqual(set(selected.tolist()), {0, 1, 4, 5})

    def test_vma_all_edges_are_weighted_by_class_sample_counts(self) -> None:
        class_edge_means = {
            "attention_weights": np.asarray(
                [[[0.9, 0.1, 0.2, 0.8]], [[0.2, 0.8, 0.6, 0.4]]]
            ),
            "volumes": np.asarray(
                [[[9.0, 1.0, 2.0, 8.0]], [[2.0, 8.0, 6.0, 4.0]]]
            ),
            "logits": np.asarray(
                [[[4.0, 1.0, 2.0, 3.0]], [[2.0, 3.0, 4.0, 1.0]]]
            ),
        }
        edges = build_vma_edges(
            class_edge_means,
            np.asarray([[0, 0, 1, 1], [1, 2, 0, 2]]),
            ["Control", "AD"],
            np.asarray([1, 3]),
            ["G0", "G1", "G2"],
            top_edges=4,
        )

        all_head = edges[(edges["class_name"] == "ALL") & (edges["head"] == 0)]
        by_offset = all_head.set_index("edge_offset")
        self.assertAlmostEqual(by_offset.loc[0, "attention_mean"], 0.375)
        self.assertAlmostEqual(by_offset.loc[1, "attention_mean"], 0.625)
        self.assertAlmostEqual(
            by_offset.loc[1, "degree_corrected_attention_enrichment"], 1.25
        )
        self.assertTrue((all_head["target_ppi_degree"] >= 2).all())


class AttentionConsensusTests(unittest.TestCase):
    @staticmethod
    def _ranking(label: str, rows: list[tuple[str, int, float]]) -> pd.DataFrame:
        frame = pd.DataFrame(rows, columns=["gene", "key_rank", "key_percentile_score"])
        frame["query_rank"] = frame["key_rank"]
        frame["query_percentile_score"] = frame["key_percentile_score"]
        frame["incoming_enrichment_mean"] = 1.0
        frame["query_specificity_mean"] = 0.0
        frame["run_label"] = label
        for top_k in TOP_K_VALUES:
            frame[f"key_top{top_k}"] = frame["key_rank"] <= top_k
            frame[f"query_top{top_k}"] = frame["query_rank"] <= top_k
        return frame

    def test_consensus_penalizes_genes_missing_from_runs(self) -> None:
        first = self._ranking("seed1", [("rare_top", 1, 1.0), ("stable", 2, 0.9)])
        second = self._ranking("seed2", [("stable", 1, 1.0), ("other", 2, 0.9)])

        consensus = build_consensus([first, second]).set_index("gene")

        self.assertGreater(
            consensus.loc["stable", "key_consensus_score"],
            consensus.loc["rare_top", "key_consensus_score"],
        )
        self.assertLess(
            consensus.loc["stable", "consensus_key_rank"],
            consensus.loc["rare_top", "consensus_key_rank"],
        )


if __name__ == "__main__":
    unittest.main()
