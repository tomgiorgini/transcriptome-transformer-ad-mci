#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.pipeline.utils import DEFAULT_GROUPS_DIR, DEFAULT_PROCESSED_DATA_DIR, DEFAULT_RAW_DATA_DIR


MATRIX_FILENAME = "matrix.txt"
AD_FILENAME = "AD.txt"
MCI_FILENAME = "MCI.txt"
CONTROL_FILENAME = "CTL.txt"
LABEL_ENCODING = {"Control": 0, "MCI": 1, "AD": 2}


def clean_text(value: object) -> str:
    text = str(value).replace("\ufeff", "").strip()
    if text.startswith('"') and text.endswith('"') and len(text) >= 2:
        text = text[1:-1].strip()
    return text


def read_label_file(path: Path) -> list[str]:
    sample_ids: list[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig", errors="ignore") as handle:
        for raw_line in handle:
            sample_id = clean_text(raw_line)
            if sample_id and sample_id not in seen:
                sample_ids.append(sample_id)
                seen.add(sample_id)
    return sample_ids


def load_expression_matrix(path: Path) -> pd.DataFrame:
    matrix = pd.read_csv(
        path,
        sep="\t",
        engine="python",
        comment="#",
        skip_blank_lines=True,
    )
    matrix.columns = [clean_text(col) for col in matrix.columns]
    matrix = matrix.dropna(axis=0, how="all").dropna(axis=1, how="all")

    first_col = matrix.columns[0]
    first_series = matrix.iloc[:, 0].map(clean_text)
    numeric_fraction = pd.to_numeric(first_series, errors="coerce").notna().mean()
    unique_fraction = first_series.nunique(dropna=True) / len(first_series)

    if first_col == "" or first_col.lower().startswith("unnamed") or (numeric_fraction < 0.5 and unique_fraction > 0.5):
        matrix = matrix.set_index(matrix.columns[0])
        matrix.index = [clean_text(idx) for idx in matrix.index]

    matrix.columns = [clean_text(col) for col in matrix.columns]
    matrix = matrix.loc[~pd.Index(matrix.index).duplicated(keep="first")]
    matrix = matrix.loc[:, ~pd.Index(matrix.columns).duplicated(keep="first")]
    return matrix


def detect_orientation(matrix: pd.DataFrame, labeled_samples: set[str]) -> str:
    column_ids = {clean_text(col) for col in matrix.columns if clean_text(col)}
    index_ids = {clean_text(idx) for idx in matrix.index if clean_text(idx)}

    column_overlap = len(column_ids & labeled_samples)
    index_overlap = len(index_ids & labeled_samples)
    return "rows=genes, columns=samples" if column_overlap >= index_overlap else "rows=samples, columns=genes"


def assemble_dataset(matrix: pd.DataFrame, label_map: dict[str, str], orientation: str) -> tuple[pd.DataFrame, pd.Series, dict[str, list[str]]]:
    missing_by_label = {"AD": [], "MCI": [], "Control": []}

    if orientation == "rows=genes, columns=samples":
        present_samples = [col for col in matrix.columns if clean_text(col) in label_map]
        x_df = matrix.loc[:, present_samples].transpose()
    else:
        present_samples = [idx for idx in matrix.index if clean_text(idx) in label_map]
        x_df = matrix.loc[present_samples, :].copy()

    x_df.index = [clean_text(idx) for idx in x_df.index]
    x_df = x_df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    x_df = x_df.loc[[sample_id for sample_id in x_df.index if sample_id in label_map]]

    y_series = pd.Series(
        [LABEL_ENCODING[label_map[sample_id]] for sample_id in x_df.index],
        index=x_df.index,
        name="label",
    )

    for sample_id, label_name in label_map.items():
        if sample_id not in x_df.index:
            missing_by_label[label_name].append(sample_id)

    return x_df, y_series, missing_by_label


def validate_disjoint_groups(group_members: dict[str, list[str]]) -> None:
    seen_by_sample: dict[str, str] = {}
    overlaps: list[str] = []
    for label_name, sample_ids in group_members.items():
        for sample_id in sample_ids:
            previous = seen_by_sample.get(sample_id)
            if previous is not None and previous != label_name:
                overlaps.append(f"{sample_id} ({previous}, {label_name})")
            else:
                seen_by_sample[sample_id] = label_name
    if overlaps:
        preview = ", ".join(overlaps[:10])
        raise ValueError(f"Group files contain overlapping sample ids across classes. Example: {preview}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the canonical Alzheimer dataset from the raw matrix and label files.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_RAW_DATA_DIR)
    parser.add_argument("--groups-dir", type=Path, default=DEFAULT_GROUPS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PROCESSED_DATA_DIR)
    parser.add_argument("--matrix", default=MATRIX_FILENAME)
    parser.add_argument("--ad", default=AD_FILENAME)
    parser.add_argument("--mci", default=MCI_FILENAME)
    parser.add_argument("--control", default=CONTROL_FILENAME)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    return parser.parse_args()


def build_stratified_split(
    y: pd.Series,
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> pd.DataFrame:
    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1 or train_ratio + val_ratio >= 1:
        raise ValueError("Expected 0 < train_ratio, val_ratio and train_ratio + val_ratio < 1.")

    train_ids, remainder_ids = train_test_split(
        y.index.to_numpy(),
        train_size=train_ratio,
        random_state=seed,
        stratify=y.to_numpy(),
    )
    remainder_y = y.loc[remainder_ids]
    relative_val_ratio = val_ratio / (1.0 - train_ratio)
    val_ids, test_ids = train_test_split(
        remainder_ids,
        train_size=relative_val_ratio,
        random_state=seed,
        stratify=remainder_y.to_numpy(),
    )

    split_by_id = {
        **{sample_id: "train" for sample_id in train_ids},
        **{sample_id: "val" for sample_id in val_ids},
        **{sample_id: "test" for sample_id in test_ids},
    }
    return pd.DataFrame(
        {"sample_id": y.index, "split": [split_by_id[sample_id] for sample_id in y.index]}
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    matrix = load_expression_matrix(args.input_dir / args.matrix)
    ad_ids = read_label_file(args.groups_dir / args.ad)
    mci_ids = read_label_file(args.groups_dir / args.mci)
    control_ids = read_label_file(args.groups_dir / args.control)

    validate_disjoint_groups({"AD": ad_ids, "MCI": mci_ids, "Control": control_ids})

    label_map: dict[str, str] = {}
    for label_name, sample_ids in (("AD", ad_ids), ("MCI", mci_ids), ("Control", control_ids)):
        for sample_id in sample_ids:
            label_map[sample_id] = label_name

    orientation = detect_orientation(matrix, set(label_map))
    x_df, y_series, missing_by_label = assemble_dataset(matrix, label_map, orientation)

    x_df.to_csv(args.output_dir / "X.csv", index=True, index_label="sample_id")
    y_series.to_frame().assign(label_name=y_series.map({0: "Control", 1: "MCI", 2: "AD"})).to_csv(
        args.output_dir / "y.csv",
        index=True,
        index_label="sample_id",
    )
    split_df = build_stratified_split(
        y_series,
        seed=args.split_seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    split_dir = args.output_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(split_dir / f"official_seed{args.split_seed}.csv", index=False)

    print(f"Orientation detected: {orientation}")
    print(f"Samples: {x_df.shape[0]}")
    print(f"Genes: {x_df.shape[1]}")
    print("Class distribution:")
    print(y_series.value_counts().sort_index().rename(index={0: 'Control', 1: 'MCI', 2: 'AD'}).to_string())
    for label_name in ("AD", "MCI", "Control"):
        print(f"Missing {label_name} IDs: {len(missing_by_label[label_name])}")
    print("Split distribution:")
    print(split_df["split"].value_counts().reindex(["train", "val", "test"]).to_string())


if __name__ == "__main__":
    main()
