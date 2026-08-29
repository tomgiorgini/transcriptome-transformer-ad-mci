from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


@dataclass
class ImageMapping:
    gene_names: list[str]
    fisher_scores: dict[str, float]
    group_labels: list[int]
    pixel_x: list[int]
    pixel_y: list[int]
    metadata: dict[str, Any]


class LDAImageTransformer:
    def __init__(self, pixels: int = 90, n_groups: int = 15, order_by: str = "fisher"):
        self.pixels = pixels
        self.n_groups = n_groups
        self.order_by = order_by
        self.mapping: ImageMapping | None = None

    def fit(self, x_train: pd.DataFrame, y_train: np.ndarray) -> "LDAImageTransformer":
        fisher = _fisher_scores(x_train, y_train)
        if self.order_by == "fisher":
            ordered_genes = fisher.sort_values(ascending=False).index.astype(str).tolist()
        elif self.order_by == "input":
            ordered_genes = x_train.columns.astype(str).tolist()
        else:
            raise ValueError(f"Unsupported order_by: {self.order_by}")
        ordered_scores = fisher.loc[ordered_genes]
        effective_groups = _effective_group_count(len(ordered_genes), self.n_groups)
        group_labels = _equal_frequency_groups(len(ordered_genes), effective_groups)

        # Genes are samples for LDA. Each gene is described by its train_inner expression profile.
        gene_vectors = x_train.loc[:, ordered_genes].to_numpy(dtype=np.float32).T
        unique_groups = np.unique(group_labels)
        if len(ordered_genes) <= len(unique_groups) or len(unique_groups) < 3:
            coords = _fallback_line_coords(len(ordered_genes))
            method = "fallback_line"
        else:
            lda = LinearDiscriminantAnalysis(n_components=2)
            coords = lda.fit_transform(gene_vectors, group_labels)
            if coords.shape[1] == 1:
                coords = np.column_stack([coords[:, 0], np.zeros(coords.shape[0], dtype=coords.dtype)])
            if not np.isfinite(coords).all():
                coords = _fallback_line_coords(len(ordered_genes))
                method = "fallback_nonfinite_lda"
            else:
                method = "lda"

        rotated_coords, coord_method = _rotate_with_minimum_bounding_rectangle(coords)
        pixel_y, pixel_x = _digitize_coords(rotated_coords, self.pixels)
        self.mapping = ImageMapping(
            gene_names=ordered_genes,
            fisher_scores={gene: float(score) for gene, score in ordered_scores.items()},
            group_labels=[int(value) for value in group_labels],
            pixel_x=[int(value) for value in pixel_x],
            pixel_y=[int(value) for value in pixel_y],
            metadata={
                "method": method,
                "scope": "fisher groups and LDA fit on train_inner only",
                "pixels": self.pixels,
                "requested_groups": self.n_groups,
                "effective_groups": int(len(unique_groups)),
                "gene_order": self.order_by,
                "coordinate_transform": coord_method,
                "n_genes": len(ordered_genes),
                "collisions": int(len(ordered_genes) - len(set(zip(pixel_x.tolist(), pixel_y.tolist())))),
            },
        )
        return self

    def transform(self, x: pd.DataFrame) -> np.ndarray:
        if self.mapping is None:
            raise RuntimeError("LDAImageTransformer must be fit before transform.")
        values = x.loc[:, self.mapping.gene_names].to_numpy(dtype=np.float32)
        values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)
        images = np.zeros((len(x), self.pixels, self.pixels, 3), dtype=np.float32)
        counts = np.zeros((self.pixels, self.pixels), dtype=np.float32)
        for gene_idx, (row, col) in enumerate(zip(self.mapping.pixel_y, self.mapping.pixel_x)):
            images[:, row, col, 0] += values[:, gene_idx]
            images[:, row, col, 1] += values[:, gene_idx]
            images[:, row, col, 2] += values[:, gene_idx]
            counts[row, col] += 1.0
        nonzero = counts > 0
        images[:, nonzero, :] /= counts[nonzero][None, :, None]
        return np.nan_to_num(images, nan=0.0, posinf=1.0, neginf=0.0)

    def manifest(self) -> dict[str, Any]:
        if self.mapping is None:
            raise RuntimeError("LDAImageTransformer must be fit before manifest.")
        return {
            **self.mapping.metadata,
            "genes": self.mapping.gene_names,
            "group_labels": self.mapping.group_labels,
            "pixel_x": self.mapping.pixel_x,
            "pixel_y": self.mapping.pixel_y,
            "fisher_scores": self.mapping.fisher_scores,
        }


def _fisher_scores(x_train: pd.DataFrame, y_train: np.ndarray) -> pd.Series:
    x0 = x_train.loc[y_train == 0]
    x1 = x_train.loc[y_train == 1]
    mean0 = x0.mean(axis=0)
    mean1 = x1.mean(axis=0)
    var0 = x0.var(axis=0).replace(0.0, np.nan)
    var1 = x1.var(axis=0).replace(0.0, np.nan)
    scores = ((mean1 - mean0) ** 2) / (var0 + var1)
    return scores.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _effective_group_count(n_items: int, requested_groups: int) -> int:
    if n_items <= 2:
        return 1
    return max(1, min(requested_groups, n_items - 1))


def _equal_frequency_groups(n_items: int, n_groups: int) -> np.ndarray:
    groups = np.floor(np.arange(n_items) * n_groups / n_items).astype(int)
    return groups.clip(0, n_groups - 1)


def _fallback_line_coords(n_items: int) -> np.ndarray:
    if n_items == 1:
        return np.array([[0.0, 0.0]], dtype=np.float32)
    x = np.linspace(0.0, 1.0, n_items)
    y = np.zeros(n_items)
    return np.column_stack([x, y]).astype(np.float32)


def _rotate_with_minimum_bounding_rectangle(coords: np.ndarray) -> tuple[np.ndarray, str]:
    if len(coords) < 3:
        return coords, "fallback_no_hull"
    try:
        hull_points = coords[ConvexHull(coords).vertices]
        _, rotation = _minimum_bounding_rectangle(hull_points)
        return np.dot(rotation, coords.T).T, "minimum_bounding_rectangle"
    except QhullError:
        return coords, "fallback_qhull_error"


def _digitize_coords(coords: np.ndarray, pixels: int) -> tuple[np.ndarray, np.ndarray]:
    coord_0 = _digitize_axis(coords[:, 0], pixels)
    coord_1 = _digitize_axis(coords[:, 1], pixels)
    return coord_0, coord_1


def _digitize_axis(values: np.ndarray, pixels: int) -> np.ndarray:
    axis_min = float(np.min(values))
    axis_max = float(np.max(values))
    if axis_min == axis_max:
        return np.zeros(len(values), dtype=int)
    bins = np.linspace(axis_min, axis_max, pixels)
    return (np.digitize(values, bins) - 1).astype(int).clip(0, pixels - 1)


def _minimum_bounding_rectangle(hull_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pi2 = np.pi / 2
    edges = hull_points[1:] - hull_points[:-1]
    angles = np.arctan2(edges[:, 1], edges[:, 0])
    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)
    rotations = np.vstack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)]).T
    rotations = rotations.reshape((-1, 2, 2))
    rotated = np.dot(rotations, hull_points.T)
    min_x = np.nanmin(rotated[:, 0], axis=1)
    max_x = np.nanmax(rotated[:, 0], axis=1)
    min_y = np.nanmin(rotated[:, 1], axis=1)
    max_y = np.nanmax(rotated[:, 1], axis=1)
    best_idx = np.argmin((max_x - min_x) * (max_y - min_y))
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    rotation = rotations[best_idx]
    corners = np.zeros((4, 2))
    corners[0] = np.dot([x1, y2], rotation)
    corners[1] = np.dot([x2, y2], rotation)
    corners[2] = np.dot([x2, y1], rotation)
    corners[3] = np.dot([x1, y1], rotation)
    return corners, rotation
