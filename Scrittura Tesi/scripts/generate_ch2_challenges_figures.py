from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "task_dataset" / "processed" / "txt_pairwise_multitask" / "shared_ad_mci_ctl"
METADATA_DIR = ROOT / "pretraining_dataset" / "geo_downloads"
FIGURE_DIR = ROOT / "Scrittura Tesi" / "figures"

LABEL_NAMES = {0: "CTL", 1: "MCI", 2: "AD"}
DIAGNOSIS_COLORS = {"AD": "#C44E52", "MCI": "#4C72B0", "CTL": "#55A868"}
BATCH_COLORS = {"GSE63060": "#8172B2", "GSE63061": "#DD8452"}


def load_data() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    x = pd.read_csv(DATA_DIR / "X.csv", index_col=0)
    y_raw = pd.read_csv(DATA_DIR / "y.csv", index_col=0).iloc[:, 0]
    diagnosis = y_raw.map(LABEL_NAMES)

    metadata_60 = pd.read_csv(
        METADATA_DIR
        / "GSE63060"
        / "eset_1_GPL6947"
        / "GSE63060_sample_metadata.tsv.gz",
        sep="\t",
    )
    ids_60 = set(metadata_60["sample_id"].astype(str))
    gsm_ids = x.index.to_series().astype(str).str.extract(r"(GSM\d+)", expand=False)
    batch = pd.Series(
        np.where(gsm_ids.isin(ids_60), "GSE63060", "GSE63061"),
        index=x.index,
        name="Batch",
    )
    diagnosis.index = x.index
    diagnosis.name = "Diagnosis"
    return x, diagnosis, batch


def top_variance_columns(x: pd.DataFrame, n_features: int) -> pd.Index:
    variance = x.var(axis=0)
    return variance.nlargest(min(n_features, x.shape[1])).index


def gene_wise_zscore(x: pd.DataFrame) -> pd.DataFrame:
    std = x.std(axis=0).replace(0, np.nan)
    return ((x - x.mean(axis=0)) / std).fillna(0.0)


def generate_separated_heatmap(
    x: pd.DataFrame, diagnosis: pd.Series, batch: pd.Series
) -> Path:
    selected_samples = diagnosis.isin(["AD", "MCI"])
    x_pair = x.loc[selected_samples]
    diagnosis_pair = diagnosis.loc[selected_samples]
    batch_pair = batch.loc[selected_samples]

    genes = top_variance_columns(x_pair, 500)
    standardized = gene_wise_zscore(x_pair.loc[:, genes]).clip(-2.5, 2.5)

    # Genes are ordered once using the pooled AD/MCI data. Samples are never
    # clustered across diagnoses: AD and MCI remain in separate panels.
    gene_distance = pdist(standardized.T.to_numpy(), metric="correlation")
    gene_order = leaves_list(linkage(gene_distance, method="average"))
    ordered_genes = standardized.columns[gene_order]

    def ordered_samples(label: str) -> pd.Index:
        sample_ids = diagnosis_pair.index[diagnosis_pair == label]
        metadata = pd.DataFrame(
            {
                "batch": batch_pair.loc[sample_ids],
                "sample": sample_ids.astype(str),
            },
            index=sample_ids,
        )
        return metadata.sort_values(["batch", "sample"]).index

    ad_samples = ordered_samples("AD")
    mci_samples = ordered_samples("MCI")
    panels = [("AD", ad_samples), ("MCI", mci_samples)]

    sns.set_theme(style="white", context="paper", font_scale=1.08)
    width_ratios = [len(ad_samples), len(mci_samples)]
    fig = plt.figure(figsize=(12.0, 7.4))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[0.035, 1.0],
        width_ratios=width_ratios,
        hspace=0.025,
        wspace=0.035,
    )
    batch_cmap = ListedColormap(
        [BATCH_COLORS["GSE63060"], BATCH_COLORS["GSE63061"]]
    )
    heatmap_image = None

    for column, (label, samples) in enumerate(panels):
        batch_axis = fig.add_subplot(grid[0, column])
        batch_values = (batch_pair.loc[samples] == "GSE63061").astype(int).to_numpy()[None, :]
        batch_axis.imshow(batch_values, aspect="auto", cmap=batch_cmap, vmin=0, vmax=1)
        batch_axis.set_xticks([])
        batch_axis.set_yticks([])
        batch_axis.set_title(f"{label} (n={len(samples)})", fontweight="bold", pad=8)
        for spine in batch_axis.spines.values():
            spine.set_visible(False)

        heatmap_axis = fig.add_subplot(grid[1, column])
        panel_values = standardized.loc[samples, ordered_genes].T.to_numpy()
        heatmap_image = heatmap_axis.imshow(
            panel_values,
            aspect="auto",
            interpolation="nearest",
            cmap="vlag",
            vmin=-2.5,
            vmax=2.5,
        )
        heatmap_axis.set_xticks([])
        heatmap_axis.set_yticks([])
        heatmap_axis.set_xlabel(f"{label} samples")
        if column == 0:
            heatmap_axis.set_ylabel("Top 500 genes by variance\n(shared gene order)")
        for spine in heatmap_axis.spines.values():
            spine.set_color("#777777")
            spine.set_linewidth(0.6)

    colorbar = fig.colorbar(
        heatmap_image,
        ax=fig.axes,
        location="right",
        fraction=0.025,
        pad=0.025,
        shrink=0.72,
    )
    colorbar.set_label("Expression z-score")
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            color=color,
            label=label,
            markersize=7,
        )
        for label, color in BATCH_COLORS.items()
    ]
    fig.legend(
        handles=legend_handles,
        title="Batch annotation",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=2,
        frameon=False,
    )
    fig.subplots_adjust(left=0.075, right=0.90, top=0.93, bottom=0.10)

    output = FIGURE_DIR / "2.4_ad_mci_separated_heatmap.png"
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def generate_pca(
    x: pd.DataFrame, diagnosis: pd.Series, batch: pd.Series
) -> Path:
    genes = top_variance_columns(x, 2000)
    standardized = gene_wise_zscore(x.loc[:, genes])
    pca = PCA(n_components=2, random_state=0)
    coordinates = pca.fit_transform(standardized.to_numpy(dtype=np.float32))
    explained = pca.explained_variance_ratio_ * 100

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharex=True, sharey=True)

    for label in ("AD", "MCI", "CTL"):
        mask = diagnosis.to_numpy() == label
        axes[0].scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=22,
            alpha=0.72,
            color=DIAGNOSIS_COLORS[label],
            edgecolors="none",
            label=label,
        )

    for label in ("GSE63060", "GSE63061"):
        mask = batch.to_numpy() == label
        axes[1].scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=22,
            alpha=0.72,
            color=BATCH_COLORS[label],
            edgecolors="none",
            label=label,
        )

    axes[0].set_title("A   Clinical labels", loc="left", fontweight="bold")
    axes[1].set_title("B   Experimental batches", loc="left", fontweight="bold")

    for axis in axes:
        axis.set_xlabel(f"PC1 ({explained[0]:.1f}% variance)")
        axis.axhline(0, color="#B8B8B8", linewidth=0.7, zorder=0)
        axis.axvline(0, color="#B8B8B8", linewidth=0.7, zorder=0)
        axis.legend(frameon=False, loc="upper right")
        axis.grid(color="#E4E4E4", linewidth=0.6)

    axes[0].set_ylabel(f"PC2 ({explained[1]:.1f}% variance)")
    fig.tight_layout()

    output = FIGURE_DIR / "2.4_pca_diagnosis_batch.png"
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    x, diagnosis, batch = load_data()
    heatmap = generate_separated_heatmap(x, diagnosis, batch)
    pca = generate_pca(x, diagnosis, batch)
    print(heatmap)
    print(pca)


if __name__ == "__main__":
    main()
