from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .utils import save_json


def softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    for y_true_value, y_pred_value in zip(y_true, y_pred):
        matrix[int(y_true_value), int(y_pred_value)] += 1
    return matrix


def classification_stats(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int):
    confusion = confusion_matrix_np(y_true, y_pred, n_classes)
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    true_positive = np.diag(confusion)

    precision = np.divide(
        true_positive,
        predicted,
        out=np.zeros_like(true_positive, dtype=float),
        where=predicted != 0,
    )
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros_like(true_positive, dtype=float),
        where=support != 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(true_positive, dtype=float),
        where=(precision + recall) != 0,
    )

    accuracy = float((y_true == y_pred).mean())
    macro_f1 = float(f1.mean())
    weighted_f1 = float(np.average(f1, weights=support))
    return accuracy, macro_f1, weighted_f1, confusion, precision, recall, f1, support


def binary_roc_auc_np(y_true_binary: np.ndarray, y_score: np.ndarray) -> float:
    y_true_binary = np.asarray(y_true_binary, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    positives = int(y_true_binary.sum())
    negatives = int(len(y_true_binary) - positives)
    if positives == 0 or negatives == 0:
        return float("nan")

    order = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[order]
    ranks = np.empty(len(y_score), dtype=np.float64)
    start = 0
    while start < len(y_score):
        end = start + 1
        while end < len(y_score) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end

    positive_rank_sum = float(ranks[y_true_binary == 1].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def one_vs_rest_roc_auc(y_true: np.ndarray, y_prob: np.ndarray, class_names: list[str]) -> dict[str, float]:
    aucs: dict[str, float] = {}
    for class_idx, class_name in enumerate(class_names):
        safe_name = str(class_name).replace(" ", "_")
        aucs[f"roc_auc_{safe_name}"] = binary_roc_auc_np((y_true == class_idx).astype(np.int64), y_prob[:, class_idx])
    valid = [value for value in aucs.values() if not np.isnan(value)]
    aucs["roc_auc_ovr_macro"] = float(np.mean(valid)) if valid else float("nan")
    return aucs


def build_report_dataframe(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> pd.DataFrame:
    accuracy, macro_f1, weighted_f1, _, precision, recall, f1, support = classification_stats(
        y_true, y_pred, len(class_names)
    )

    rows = []
    for idx, class_name in enumerate(class_names):
        rows.append(
            {
                "class": class_name,
                "precision": precision[idx],
                "recall": recall[idx],
                "f1_score": f1[idx],
                "support": int(support[idx]),
            }
        )

    total_support = int(np.sum(support))
    rows.append(
        {
            "class": "macro avg",
            "precision": float(np.mean(precision)),
            "recall": float(np.mean(recall)),
            "f1_score": macro_f1,
            "support": total_support,
        }
    )
    rows.append(
        {
            "class": "weighted avg",
            "precision": float(np.average(precision, weights=support)),
            "recall": float(np.average(recall, weights=support)),
            "f1_score": weighted_f1,
            "support": total_support,
        }
    )
    rows.append(
        {
            "class": "accuracy",
            "precision": accuracy,
            "recall": accuracy,
            "f1_score": accuracy,
            "support": total_support,
        }
    )
    return pd.DataFrame(rows)


def report_to_text(report_df: pd.DataFrame) -> str:
    lines = [f"{'class':<14}{'precision':>12}{'recall':>12}{'f1-score':>12}{'support':>10}"]
    for row in report_df.itertuples(index=False):
        lines.append(
            f"{row[0]:<14}{row[1]:>12.4f}{row[2]:>12.4f}{row[3]:>12.4f}{int(row[4]):>10d}"
        )
    return "\n".join(lines)


def print_report(name: str, report_df: pd.DataFrame) -> None:
    accuracy_row = report_df.loc[report_df["class"] == "accuracy"].iloc[0]
    macro_row = report_df.loc[report_df["class"] == "macro avg"].iloc[0]
    weighted_row = report_df.loc[report_df["class"] == "weighted avg"].iloc[0]
    print(
        f"\n[{name}] accuracy={accuracy_row['f1_score']:.4f} "
        f"macro_f1={macro_row['f1_score']:.4f} weighted_f1={weighted_row['f1_score']:.4f}"
    )
    print(report_to_text(report_df))


def save_selected_genes(result_dir: Path, gene_names: list[str]) -> None:
    pd.DataFrame({"gene": gene_names}).to_csv(result_dir / "selected_genes.csv", index=False)


def save_split_assignments(result_dir: Path, train_ids: np.ndarray, val_ids: np.ndarray, test_ids: np.ndarray) -> None:
    rows = (
        [{"sample_id": sample_id, "split": "train"} for sample_id in train_ids]
        + [{"sample_id": sample_id, "split": "val"} for sample_id in val_ids]
        + [{"sample_id": sample_id, "split": "test"} for sample_id in test_ids]
    )
    pd.DataFrame(rows).to_csv(result_dir / "split_assignments.csv", index=False)


def save_args(result_dir: Path, args_dict: dict[str, object]) -> None:
    save_json(result_dir / "args.json", args_dict)


def save_training_history(result_dir: Path, history_rows: list[dict[str, object]]) -> None:
    pd.DataFrame(history_rows).to_csv(result_dir / "training_log.csv", index=False)


def save_metrics_summary(result_dir: Path, split_results: dict[str, dict[str, object]]) -> None:
    rows = []
    for split_name, split_result in split_results.items():
        row = {
            "split": split_name,
            "samples": len(split_result["sample_ids"]),
            "loss": split_result.get("loss"),
            "accuracy": split_result["accuracy"],
            "macro_f1": split_result["macro_f1"],
            "weighted_f1": split_result["weighted_f1"],
            "balanced_accuracy": split_result.get("balanced_accuracy"),
        }
        row.update({key: value for key, value in split_result.items() if key.startswith("roc_auc_")})
        rows.append(row)
    pd.DataFrame(rows).to_csv(result_dir / "metrics_summary.csv", index=False)


def save_test_artifacts(result_dir: Path, split_result: dict[str, object], class_names: list[str]) -> None:
    predictions = pd.DataFrame(
        {
            "sample_id": split_result["sample_ids"],
            "y_true": split_result["y_true"],
            "y_true_name": [class_names[idx] for idx in split_result["y_true"]],
            "y_pred": split_result["y_pred"],
            "y_pred_name": [class_names[idx] for idx in split_result["y_pred"]],
        }
    )

    probabilities = split_result["y_prob"]
    for class_idx in range(probabilities.shape[1]):
        predictions[f"prob_{class_idx}"] = probabilities[:, class_idx]
    predictions.to_csv(result_dir / "test_predictions.csv", index=False)

    pd.DataFrame(
        split_result["confusion_matrix"],
        index=class_names,
        columns=class_names,
    ).to_csv(result_dir / "test_confusion_matrix.csv")

    split_result["report_df"].to_csv(result_dir / "test_classification_report.csv", index=False)
    with (result_dir / "test_classification_report.txt").open("w", encoding="utf-8") as handle:
        handle.write(report_to_text(split_result["report_df"]))
        handle.write("\n")
