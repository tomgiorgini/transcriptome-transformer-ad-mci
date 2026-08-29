#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_OUTPUT_DIR = ROOT / "task_dataset" / "processed" / "txt_pairwise_multitask"
DEFAULT_SOURCE_DIR = ROOT / "task_dataset" / "processed" / "alzheimer_multiclass"
DEFAULT_X_FILE = DEFAULT_SOURCE_DIR / "X.csv"
DEFAULT_Y_FILE = DEFAULT_SOURCE_DIR / "y.csv"
DEFAULT_OFFICIAL_SPLIT_FILE = DEFAULT_SOURCE_DIR / "splits" / "official_seed42.csv"


@dataclass(frozen=True)
class PairwiseTask:
    name: str
    dirname: str
    negative_class: str
    positive_class: str


TASKS = [
    PairwiseTask("AD_vs_MCI", "ad_vs_mci", "MCI", "AD"),
    PairwiseTask("AD_vs_CTL", "ad_vs_ctl", "Control", "AD"),
    PairwiseTask("MCI_vs_CTL", "mci_vs_ctl", "Control", "MCI"),
]


def portable_path(path: Path) -> str:
    """Serialize repository paths without leaking a developer's home directory."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the shared AD/MCI/Control dataset for TxT multitask training and "
            "three physical pairwise datasets: AD_vs_MCI, AD_vs_CTL, MCI_vs_CTL."
        )
    )
    parser.add_argument("--x-file", type=Path, default=DEFAULT_X_FILE)
    parser.add_argument("--y-file", type=Path, default=DEFAULT_Y_FILE)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_OFFICIAL_SPLIT_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def normalize_class_name(value: str) -> str:
    return str(value).strip().lower().replace(" ", "").replace("_", "")


def canonical_label_name(value: str) -> str:
    normalized = normalize_class_name(value)
    if normalized in {"control", "ctl", "cn", "nc"}:
        return "Control"
    if normalized == "mci":
        return "MCI"
    if normalized in {"ad", "dementia", "alzheimer"}:
        return "AD"
    raise ValueError(f"Unsupported label_name: {value!r}")


def load_inputs(x_file: Path, y_file: Path, split_file: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x_df = pd.read_csv(x_file)
    y_df = pd.read_csv(y_file)
    split_df = pd.read_csv(split_file)

    for frame_name, frame in [("X", x_df), ("y", y_df), ("split", split_df)]:
        if "sample_id" not in frame.columns:
            raise ValueError(f"{frame_name} file must contain sample_id.")
        frame["sample_id"] = frame["sample_id"].astype(str).str.strip()

    if "label_name" not in y_df.columns:
        raise ValueError("y file must contain label_name so pairwise datasets can be built unambiguously.")
    if "split" not in split_df.columns:
        raise ValueError("split file must contain split.")

    y_df["label_name"] = y_df["label_name"].map(canonical_label_name)
    split_df["split"] = split_df["split"].astype(str).str.strip().str.lower()

    invalid_splits = sorted(set(split_df["split"]) - {"train", "val", "test"})
    if invalid_splits:
        raise ValueError(f"Invalid split values: {invalid_splits}")

    expected = set(y_df["sample_id"])
    missing_x = expected - set(x_df["sample_id"])
    missing_split = expected - set(split_df["sample_id"])
    if missing_x or missing_split:
        raise ValueError(
            "Input files do not align: "
            f"missing in X={len(missing_x)}, missing in split={len(missing_split)}"
        )
    return x_df, y_df, split_df


def write_dataset(
    output_dir: Path,
    x_df: pd.DataFrame,
    y_df: pd.DataFrame,
    split_df: pd.DataFrame,
    summary_rows: list[dict[str, object]],
    dataset_name: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "splits").mkdir(parents=True, exist_ok=True)
    x_df.to_csv(output_dir / "X.csv", index=False)
    y_df.to_csv(output_dir / "y.csv", index=False)
    split_df.to_csv(output_dir / "splits" / "official_seed42.csv", index=False)

    for split_name, split_part in split_df.groupby("split"):
        split_labels = y_df[y_df["sample_id"].isin(split_part["sample_id"])]
        counts = split_labels["label_name"].value_counts().to_dict()
        summary_rows.append(
            {
                "dataset": dataset_name,
                "split": split_name,
                "samples": int(len(split_labels)),
                **{f"n_{key}": int(value) for key, value in sorted(counts.items())},
            }
        )


def build_pairwise_y(y_df: pd.DataFrame, task: PairwiseTask) -> pd.DataFrame:
    pair_y = y_df[y_df["label_name"].isin({task.negative_class, task.positive_class})].copy()
    label_map = {task.negative_class: 0, task.positive_class: 1}
    pair_y["label"] = pair_y["label_name"].map(label_map).astype(int)
    pair_y["label_name"] = pair_y["label_name"].map(
        {
            task.negative_class: task.negative_class,
            task.positive_class: task.positive_class,
        }
    )
    return pair_y[["sample_id", "label", "label_name"]]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    x_df, y_df, split_df = load_inputs(args.x_file, args.y_file, args.split_file)
    summary_rows: list[dict[str, object]] = []

    shared_dir = args.output_dir / "shared_ad_mci_ctl"
    write_dataset(
        shared_dir,
        x_df,
        y_df[["sample_id", "label", "label_name"]],
        split_df[["sample_id", "split"]],
        summary_rows,
        "shared_ad_mci_ctl",
    )

    manifest = {
        "source_x": portable_path(args.x_file),
        "source_y": portable_path(args.y_file),
        "source_split": portable_path(args.split_file),
        "shared_dataset": portable_path(shared_dir),
        "pairwise_tasks": {},
    }

    for task in TASKS:
        pair_y = build_pairwise_y(y_df, task)
        keep_ids = set(pair_y["sample_id"])
        pair_x = x_df[x_df["sample_id"].isin(keep_ids)].copy()
        pair_split = split_df[split_df["sample_id"].isin(keep_ids)][["sample_id", "split"]].copy()
        task_dir = args.output_dir / task.dirname
        write_dataset(task_dir, pair_x, pair_y, pair_split, summary_rows, task.name)
        manifest["pairwise_tasks"][task.name] = {
            "directory": portable_path(task_dir),
            "negative_class": task.negative_class,
            "positive_class": task.positive_class,
            "label_encoding": {task.negative_class: 0, task.positive_class: 1},
        }

    summary = pd.DataFrame(summary_rows).fillna(0)
    summary.to_csv(args.output_dir / "dataset_summary.csv", index=False)
    with (args.output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Pairwise TxT datasets written to: {args.output_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
