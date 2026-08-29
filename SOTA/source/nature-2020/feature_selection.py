from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler


@dataclass
class FeatureSet:
    name: str
    x_train: np.ndarray
    x_val: np.ndarray
    x_test: np.ndarray
    feature_names: list[str]
    status: str
    metadata: dict[str, object]


def clean_gene_name(value: object) -> str | None:
    if pd.isna(value):
        return None
    gene = str(value).strip()
    if not gene or gene.startswith("#") or gene.lower() in {"nan", "na", "n/a", "null", "none", "-"}:
        return None
    return gene


def load_tf_genes(supplement_dir: Path) -> set[str]:
    path = supplement_dir / "41598_2020_60595_MOESM2_ESM.xlsx"
    df = pd.read_excel(path, sheet_name="Table S1")
    return {g for g in (clean_gene_name(v) for v in df.iloc[:, 0].tolist()) if g}


def load_cfg_genes(supplement_dir: Path, min_score: int = 3) -> set[str]:
    genes: set[str] = set()
    for filename in ("41598_2020_60595_MOESM3_ESM.xlsx", "41598_2020_60595_MOESM4_ESM.xlsx", "41598_2020_60595_MOESM5_ESM.xlsx"):
        path = supplement_dir / filename
        df = pd.read_excel(path, sheet_name=1)
        if "Gene" not in df.columns or "Final_CFG" not in df.columns:
            continue
        scores = pd.to_numeric(df["Final_CFG"], errors="coerce")
        for gene in df.loc[scores >= min_score, "Gene"].tolist():
            cleaned = clean_gene_name(gene)
            if cleaned:
                genes.add(cleaned)
    return genes


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_hprd_hub_genes(
    network_file: Path,
    degree_threshold: int = 10,
    gene_columns: tuple[int, int] | None = None,
) -> tuple[set[str], dict[str, object]]:
    """Load a local HPRD edge file and return genes with degree > threshold.

    HPRD Release 9's ``BINARY_PROTEIN_PROTEIN_INTERACTIONS.txt`` stores gene
    symbols in zero-based columns 0 and 3. A plain two-column edge list is also
    accepted. The function never downloads or silently substitutes a network.
    """

    network_file = network_file.resolve()
    if not network_file.is_file():
        raise FileNotFoundError(f"HPRD network file does not exist: {network_file}")
    if degree_threshold < 0:
        raise ValueError("degree_threshold must be non-negative.")
    frame = pd.read_csv(
        network_file,
        sep=None,
        engine="python",
        header=None,
        comment="#",
        dtype=str,
        keep_default_na=False,
    )
    if frame.shape[1] < 2:
        raise ValueError(f"HPRD network must contain at least two columns: {network_file}")
    columns = gene_columns or ((0, 3) if frame.shape[1] >= 4 else (0, 1))
    if min(columns) < 0 or max(columns) >= frame.shape[1] or columns[0] == columns[1]:
        raise ValueError(f"Invalid HPRD gene columns {columns} for a {frame.shape[1]}-column file.")

    adjacency: dict[str, set[str]] = defaultdict(set)
    unique_edges: set[tuple[str, str]] = set()
    for left_raw, right_raw in frame.iloc[:, list(columns)].itertuples(index=False, name=None):
        left = clean_gene_name(left_raw)
        right = clean_gene_name(right_raw)
        if not left or not right or left == right:
            continue
        edge = tuple(sorted((left, right)))
        if edge in unique_edges:
            continue
        unique_edges.add(edge)
        adjacency[left].add(right)
        adjacency[right].add(left)
    hubs = {gene for gene, neighbours in adjacency.items() if len(neighbours) > degree_threshold}
    provenance: dict[str, object] = {
        "network_file": str(network_file),
        "network_sha256": sha256_file(network_file),
        "network_size_bytes": network_file.stat().st_size,
        "gene_columns_zero_based": list(columns),
        "degree_rule": f"degree > {degree_threshold}",
        "degree_threshold": degree_threshold,
        "n_raw_rows": int(len(frame)),
        "n_unique_edges": len(unique_edges),
        "n_nodes": len(adjacency),
        "n_hubs": len(hubs),
        "source_policy": "user-supplied local file; no automatic download or substitution",
    }
    return hubs, provenance


def fit_imputer_scaler(
    x_df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    scaler_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    x_train_raw = x_df.iloc[train_idx]
    x_val_raw = x_df.iloc[val_idx]
    x_test_raw = x_df.iloc[test_idx]
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(x_train_raw)
    x_val = imputer.transform(x_val_raw)
    x_test = imputer.transform(x_test_raw)
    meta: dict[str, object] = {"imputer": "SimpleImputer(strategy=median)", "scaler": scaler_name, "fit_scope": "train_inner"}
    if scaler_name == "minmax":
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)
        x_test = scaler.transform(x_test)
    elif scaler_name != "none":
        raise ValueError(f"Unsupported scaler: {scaler_name}")
    return (
        np.nan_to_num(x_train.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0),
        np.nan_to_num(x_val.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0),
        np.nan_to_num(x_test.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0),
        meta,
    )


def run_limma_deg(
    x_train_df: pd.DataFrame,
    y_train: np.ndarray,
    script_path: Path,
    fdr_threshold: float,
    work_dir: Path,
    p_value_threshold: float | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nature2020_limma_") as tmp:
        tmp_path = Path(tmp)
        x_path = tmp_path / "x_train.csv"
        y_path = tmp_path / "y_train.csv"
        out_path = tmp_path / "deg_table.csv"
        x_train_df.to_csv(x_path)
        pd.DataFrame({"label": y_train}, index=x_train_df.index).to_csv(y_path)
        subprocess.run(["Rscript", str(script_path), str(x_path), str(y_path), str(out_path)], check=True)
        deg_table = pd.read_csv(out_path)
    deg_table = deg_table.rename(columns={"adj.P.Val": "adjusted_p", "P.Value": "p_value", "logFC": "log_fc"})
    deg_table.to_csv(work_dir / "deg_table.csv", index=False)
    if p_value_threshold is not None:
        p_values = pd.to_numeric(deg_table["p_value"], errors="coerce")
        selected = deg_table.loc[p_values < p_value_threshold, "gene"].astype(str).tolist()
    else:
        adjusted = pd.to_numeric(deg_table["adjusted_p"], errors="coerce")
        selected = deg_table.loc[adjusted < fdr_threshold, "gene"].astype(str).tolist()
    return deg_table, [g for g in selected if g in x_train_df.columns]


def subset_feature_set(
    name: str,
    x_train_all: np.ndarray,
    x_val_all: np.ndarray,
    x_test_all: np.ndarray,
    all_gene_names: list[str],
    selected_genes: list[str],
    metadata: dict[str, object],
) -> FeatureSet:
    position_by_gene = {gene: position for position, gene in enumerate(all_gene_names)}
    positions = [position_by_gene[g] for g in dict.fromkeys(selected_genes) if g in position_by_gene]
    feature_names = [all_gene_names[i] for i in positions]
    if not positions:
        return empty_feature_set(
            name,
            x_train_all.shape[0],
            x_val_all.shape[0],
            x_test_all.shape[0],
            "skipped_empty_feature_set",
            metadata,
        )
    return FeatureSet(name, x_train_all[:, positions], x_val_all[:, positions], x_test_all[:, positions], feature_names, "completed", {**metadata, "n_features": len(feature_names)})


def empty_feature_set(
    name: str,
    n_train: int,
    n_val: int,
    n_test: int,
    status: str,
    metadata: dict[str, object],
) -> FeatureSet:
    return FeatureSet(
        name,
        np.empty((n_train, 0), dtype=np.float32),
        np.empty((n_val, 0), dtype=np.float32),
        np.empty((n_test, 0), dtype=np.float32),
        [],
        status,
        metadata,
    )


def resolve_vae_latent_dim(
    n_features: int,
    policy: str = "paper_reconstruction",
    fixed_dim: int | None = None,
) -> tuple[int, dict[str, object]]:
    """Resolve a VAE dimension without presenting an unstated rule as exact.

    Lee & Lee state that the latent dimension was chosen to be similar to the
    number of CFG-selected genes, but do not give a deterministic rule for a new
    train split. ``paper_reconstruction`` preserves the repository's explicit
    100/200/300 bins and records that they are a reconstruction. ``fixed`` lets
    callers reproduce a declared dimension exactly.
    """

    if n_features < 1:
        raise ValueError("The VAE requires at least one input feature.")
    if policy == "fixed":
        if fixed_dim is None:
            raise ValueError("fixed_dim is required when latent policy is 'fixed'.")
        if fixed_dim < 1 or fixed_dim > n_features:
            raise ValueError(f"fixed_dim must be between 1 and n_features={n_features}.")
        return fixed_dim, {
            "latent_policy": "fixed",
            "latent_policy_status": "user_declared",
            "latent_policy_rule": f"fixed_dim={fixed_dim}",
        }
    if policy != "paper_reconstruction":
        raise ValueError(f"Unsupported VAE latent policy: {policy}")
    if n_features <= 500:
        latent_dim = min(100, n_features)
        rule = "min(100, n_features) for n_features <= 500"
    elif n_features <= 1200:
        latent_dim = 200
        rule = "200 for 500 < n_features <= 1200"
    else:
        latent_dim = 300
        rule = "300 for n_features > 1200"
    return latent_dim, {
        "latent_policy": "paper_reconstruction",
        "latent_policy_status": "disclosed_reconstruction",
        "latent_policy_rule": rule,
        "paper_statement": "latent dimension similar to the number of CFG-selected genes",
        "under_specification": "The paper gives no deterministic latent-dimension rule for new train splits.",
    }


def vae_latent_dim(n_features: int) -> int:
    """Backwards-compatible shorthand for the disclosed reconstruction policy."""

    return resolve_vae_latent_dim(n_features)[0]


def build_variational_autoencoder(
    n_features: int,
    latent_dim: int,
    seed: int,
):
    """Build the paper-oriented VAE with reparameterisation and KL loss."""

    import tensorflow as tf
    from tensorflow import keras

    tf.keras.utils.set_random_seed(seed)
    hidden_dim = max(n_features // 2, latent_dim * 2, 10)

    class Sampling(keras.layers.Layer):
        def call(self, inputs):
            z_mean, z_log_var = inputs
            epsilon = tf.random.normal(shape=tf.shape(z_mean), seed=seed)
            sample = z_mean + tf.exp(0.5 * z_log_var) * epsilon
            kl_per_sample = -0.5 * tf.reduce_sum(
                1.0 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                axis=1,
            )
            self.add_loss(tf.reduce_mean(kl_per_sample))
            return sample

    inputs = keras.Input(shape=(n_features,), name="expression")
    encoded = keras.layers.Dense(hidden_dim, activation="elu", name="encoder_elu")(inputs)
    z_mean = keras.layers.Dense(latent_dim, name="z_mean")(encoded)
    z_scale = keras.layers.Dense(latent_dim, activation="softplus", name="z_scale_softplus")(encoded)
    z_log_var = keras.layers.Lambda(
        lambda scale: 2.0 * tf.math.log(scale + keras.backend.epsilon()),
        name="z_log_var",
    )(z_scale)
    z_sample = Sampling(name="reparameterized_sample")([z_mean, z_log_var])
    encoder = keras.Model(inputs, [z_mean, z_log_var, z_sample], name="vae_encoder")

    latent_inputs = keras.Input(shape=(latent_dim,), name="latent_input")
    decoded = keras.layers.Dense(hidden_dim, activation="tanh", name="decoder_tanh")(latent_inputs)
    reconstruction = keras.layers.Dense(n_features, activation="tanh", name="reconstruction_tanh")(decoded)
    decoder = keras.Model(latent_inputs, reconstruction, name="vae_decoder")
    vae = keras.Model(inputs, decoder(encoder(inputs)[2]), name="variational_autoencoder")

    def reconstruction_loss(y_true, y_pred):
        return tf.reduce_mean(tf.reduce_sum(tf.square(y_true - y_pred), axis=1))

    vae.compile(optimizer=keras.optimizers.Adagrad(learning_rate=0.001), loss=reconstruction_loss)
    architecture = {
        "architecture": "variational_autoencoder",
        "encoder_activation": "elu",
        "scale_activation": "softplus",
        "decoder_activation": "tanh",
        "sampling": "z_mean + exp(0.5 * z_log_var) * epsilon",
        "regularization": "analytic KL(q(z|x) || N(0,I))",
        "reconstruction_loss": "sum_squared_error",
        "optimizer": "adagrad",
        "learning_rate": 0.001,
        "hidden_dim": hidden_dim,
        "hidden_dim_policy": "max(floor(n_features/2), 2*latent_dim, 10); disclosed reconstruction where the supplement is not explicit",
    }
    return vae, encoder, architecture


def fit_vae_feature_set(
    base: FeatureSet,
    y_train: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    epochs: int,
    batch_size: int | None,
    latent_policy: str = "paper_reconstruction",
    fixed_latent_dim: int | None = None,
    fine_tune_epochs: int | None = None,
) -> FeatureSet:
    if base.status != "completed" or base.x_train.shape[1] < 1:
        return empty_feature_set(
            "vae",
            base.x_train.shape[0],
            base.x_val.shape[0],
            base.x_test.shape[0],
            "skipped_empty_feature_set",
            {"base_status": base.status},
        )
    from tensorflow import keras

    if len(y_train) != len(base.x_train) or len(y_val) != len(base.x_val):
        raise ValueError("VAE labels must align with train_inner and val_inner matrices.")
    if epochs <= 0:
        raise ValueError("VAE epochs/iterations must be positive.")
    n_features = base.x_train.shape[1]
    latent_dim, latent_metadata = resolve_vae_latent_dim(n_features, latent_policy, fixed_latent_dim)
    vae, encoder, architecture = build_variational_autoencoder(n_features, latent_dim, seed)
    if batch_size is not None and batch_size <= 0:
        raise ValueError("batch_size must be positive when specified.")
    effective_batch_size = len(base.x_train) if batch_size is None else min(batch_size, len(base.x_train))
    validation_data = (base.x_val, base.x_val) if len(base.x_val) else None
    history = vae.fit(
        base.x_train,
        base.x_train,
        validation_data=validation_data,
        epochs=epochs,
        batch_size=effective_batch_size,
        shuffle=True,
        verbose=0,
    )

    supervised_epochs = epochs if fine_tune_epochs is None else fine_tune_epochs
    if supervised_epochs < 0:
        raise ValueError("fine_tune_epochs must be non-negative.")
    fine_tune_history = None
    if supervised_epochs:
        classifier_output = keras.layers.Dense(1, activation="sigmoid", name="supervised_output")(encoder.outputs[2])
        fine_tune_model = keras.Model(encoder.input, classifier_output, name="vae_supervised_fine_tuning")
        fine_tune_model.compile(
            optimizer=keras.optimizers.Adagrad(learning_rate=0.001),
            loss="binary_crossentropy",
        )
        labelled_validation = (base.x_val, y_val) if len(base.x_val) else None
        fine_tune_history = fine_tune_model.fit(
            base.x_train,
            y_train,
            validation_data=labelled_validation,
            epochs=supervised_epochs,
            batch_size=effective_batch_size,
            shuffle=True,
            verbose=0,
        )

    def encode_mean(values: np.ndarray) -> np.ndarray:
        if not len(values):
            return np.empty((0, latent_dim), dtype=np.float32)
        return encoder.predict(values, verbose=0)[0].astype(np.float32)

    train_mean = encode_mean(base.x_train)
    val_mean = encode_mean(base.x_val)
    test_mean = encode_mean(base.x_test)
    feature_names = [f"vae_{i:03d}" for i in range(latent_dim)]
    return FeatureSet(
        "vae",
        train_mean,
        val_mean,
        test_mean,
        feature_names,
        "completed",
        {
            "base_feature_set": base.name,
            "base_n_features": n_features,
            "latent_dim": latent_dim,
            **latent_metadata,
            **architecture,
            "pretraining_iterations_configured": epochs,
            "pretraining_iterations_ran": len(history.history.get("loss", [])),
            "supervised_fine_tuning": bool(supervised_epochs),
            "fine_tuning_iterations_configured": supervised_epochs,
            "fine_tuning_iterations_ran": len(fine_tune_history.history.get("loss", [])) if fine_tune_history else 0,
            "batch_size": effective_batch_size,
            "batch_size_policy": "full_train_inner" if batch_size is None else "user_declared",
            "representation": "deterministic z_mean after supervised fine-tuning",
            "fit_scope": "train_inner",
            "outer_test_usage": "transform_only",
        },
    )


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
