from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class AugmentationResult:
    x_train: pd.DataFrame
    y_train: np.ndarray
    manifest: dict[str, Any]


def _build_generator(tf, keras, latent_dim: int, n_features: int):
    noise = keras.layers.Input(shape=(latent_dim,))
    label = keras.layers.Input(shape=(1,))
    label_embed = keras.layers.Embedding(2, latent_dim)(label)
    label_flat = keras.layers.Flatten()(label_embed)
    x = keras.layers.Concatenate()([noise, label_flat])
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    out = keras.layers.Dense(n_features, activation="sigmoid")(x)
    return keras.Model([noise, label], out, name="conditional_tabular_generator")


def _build_discriminator(keras, n_features: int):
    features = keras.layers.Input(shape=(n_features,))
    label = keras.layers.Input(shape=(1,))
    label_embed = keras.layers.Embedding(2, n_features)(label)
    label_flat = keras.layers.Flatten()(label_embed)
    x = keras.layers.Concatenate()([features, label_flat])
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    out = keras.layers.Dense(1, activation="sigmoid")(x)
    return keras.Model([features, label], out, name="conditional_tabular_discriminator")


def _train_conditional_gan(
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    latent_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> tuple[Any, dict[str, Any]]:
    import tensorflow as tf
    from tensorflow import keras

    tf.keras.utils.set_random_seed(seed)
    generator = _build_generator(tf, keras, latent_dim, x_train.shape[1])
    discriminator = _build_discriminator(keras, x_train.shape[1])
    d_optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    g_optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = keras.losses.BinaryCrossentropy()
    rng = np.random.default_rng(seed)
    losses: list[dict[str, float]] = []

    real_x = x_train.astype(np.float32)
    real_y = y_train.astype(np.int32).reshape(-1, 1)
    for epoch in range(epochs):
        order = rng.permutation(len(real_x))
        epoch_d: list[float] = []
        epoch_g: list[float] = []
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            if len(idx) == 0:
                continue
            batch_x = tf.convert_to_tensor(real_x[idx], dtype=tf.float32)
            batch_y = tf.convert_to_tensor(real_y[idx], dtype=tf.int32)
            current_batch = len(idx)
            noise = tf.convert_to_tensor(rng.normal(size=(current_batch, latent_dim)).astype(np.float32), dtype=tf.float32)

            with tf.GradientTape() as d_tape:
                fake_x = generator([noise, batch_y], training=True)
                real_logits = discriminator([batch_x, batch_y], training=True)
                fake_logits = discriminator([fake_x, batch_y], training=True)
                d_loss = loss_fn(tf.ones_like(real_logits), real_logits) + loss_fn(tf.zeros_like(fake_logits), fake_logits)
            d_grads = d_tape.gradient(d_loss, discriminator.trainable_variables)
            d_optimizer.apply_gradients(zip(d_grads, discriminator.trainable_variables))

            sampled_labels = tf.convert_to_tensor(rng.integers(0, 2, size=(current_batch, 1), dtype=np.int32), dtype=tf.int32)
            noise = tf.convert_to_tensor(rng.normal(size=(current_batch, latent_dim)).astype(np.float32), dtype=tf.float32)
            with tf.GradientTape() as g_tape:
                fake_x = generator([noise, sampled_labels], training=True)
                fake_logits = discriminator([fake_x, sampled_labels], training=True)
                g_loss = loss_fn(tf.ones_like(fake_logits), fake_logits)
            g_grads = g_tape.gradient(g_loss, generator.trainable_variables)
            g_optimizer.apply_gradients(zip(g_grads, generator.trainable_variables))
            epoch_d.append(float(d_loss.numpy()))
            epoch_g.append(float(g_loss.numpy()))
        if epoch == epochs - 1 or epoch % max(1, epochs // 10) == 0:
            losses.append({"epoch": float(epoch), "d_loss": float(np.mean(epoch_d)), "g_loss": float(np.mean(epoch_g))})
    metadata = {
        "tensorflow_version": tf.__version__,
        "latent_dim": int(latent_dim),
        "generator_hidden_layers": [256, 256],
        "discriminator_hidden_layers": [256, 256],
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "loss_trace": losses,
    }
    return generator, metadata


def _synthetic_label_plan(
    y_train: np.ndarray,
    n_to_generate: int,
    strategy: str,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    class_counts = pd.Series(y_train).value_counts().reindex([0, 1], fill_value=0).astype(int)
    rng_labels: list[int] = []
    if strategy == "proportional":
        class_probs = (class_counts / class_counts.sum()).to_numpy(dtype=float)
        rng = rng or np.random.default_rng()
        return rng.choice(np.array([0, 1], dtype=np.int32), size=n_to_generate, p=class_probs).astype(np.int32)

    if strategy == "minority":
        minority_label = int(class_counts.idxmin())
        return np.full(n_to_generate, minority_label, dtype=np.int32)

    if strategy != "balanced":
        raise ValueError(f"Unsupported GAN sampling strategy: {strategy}")

    final_target_per_class = int(np.ceil((len(y_train) + n_to_generate) / 2.0))
    desired = {
        int(label): max(0, final_target_per_class - int(count))
        for label, count in class_counts.items()
    }
    for label in [0, 1]:
        rng_labels.extend([label] * min(desired[label], n_to_generate - len(rng_labels)))
    while len(rng_labels) < n_to_generate:
        current_counts = class_counts.copy()
        for label in rng_labels:
            current_counts.loc[label] += 1
        next_label = int(current_counts.idxmin())
        rng_labels.append(next_label)
    return np.asarray(rng_labels[:n_to_generate], dtype=np.int32)


def _realism_summary(real_x: np.ndarray, synthetic_x: np.ndarray) -> dict[str, float]:
    if synthetic_x.size == 0:
        return {
            "realism_mean_abs_feature_mean_diff": float("nan"),
            "realism_mean_abs_feature_std_diff": float("nan"),
            "realism_feature_mean_correlation": float("nan"),
        }
    real_mean = np.nanmean(real_x, axis=0)
    synthetic_mean = np.nanmean(synthetic_x, axis=0)
    real_std = np.nanstd(real_x, axis=0)
    synthetic_std = np.nanstd(synthetic_x, axis=0)
    mean_abs_mean_diff = float(np.nanmean(np.abs(real_mean - synthetic_mean)))
    mean_abs_std_diff = float(np.nanmean(np.abs(real_std - synthetic_std)))
    if np.nanstd(real_mean) > 0 and np.nanstd(synthetic_mean) > 0:
        mean_correlation = float(np.corrcoef(real_mean, synthetic_mean)[0, 1])
    else:
        mean_correlation = float("nan")
    return {
        "realism_mean_abs_feature_mean_diff": mean_abs_mean_diff,
        "realism_mean_abs_feature_std_diff": mean_abs_std_diff,
        "realism_feature_mean_correlation": mean_correlation,
    }


def _class_counts(values: np.ndarray) -> dict[str, int]:
    return {str(k): int(v) for k, v in pd.Series(values).value_counts().reindex([0, 1], fill_value=0).astype(int).to_dict().items()}


def _train_and_sample_ctgan_inprocess(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    n_to_generate: int,
    seed: int,
    latent_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    sampling_strategy: str,
    pac: int,
    enable_gpu: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if pac <= 0:
        raise ValueError("ctgan_pac must be positive.")
    if batch_size <= 0 or batch_size % 2 != 0:
        raise ValueError("CTGAN batch_size must be a positive even number.")
    if batch_size % pac != 0:
        raise ValueError("CTGAN batch_size must be divisible by ctgan_pac.")
    try:
        import ctgan
        from ctgan import CTGAN
    except ImportError as exc:
        raise RuntimeError(
            "augmentation='ctgan' requires the external ctgan package. "
            "Install requirements-nature-2026.txt; use augmentation='gan' only for the explicitly labelled local approximation."
        ) from exc

    label_column = "__ctgan_class_label__"
    while label_column in x_train.columns:
        label_column = f"_{label_column}"
    training_table = x_train.copy()
    training_table[label_column] = y_train.astype(np.int32)
    synthesizer = CTGAN(
        embedding_dim=int(latent_dim),
        generator_dim=(256, 256),
        discriminator_dim=(256, 256),
        generator_lr=float(learning_rate),
        discriminator_lr=float(learning_rate),
        batch_size=int(batch_size),
        epochs=int(epochs),
        pac=int(pac),
        log_frequency=False,
        verbose=False,
        enable_gpu=bool(enable_gpu),
    )
    synthesizer.set_random_state(seed)
    synthesizer.fit(training_table, discrete_columns=[label_column])

    rng = np.random.default_rng(seed + 991)
    label_plan = _synthetic_label_plan(
        y_train.astype(np.int32),
        n_to_generate,
        sampling_strategy,
        rng=rng,
    )
    sampled_frames: list[pd.DataFrame] = []
    for label in (0, 1):
        count = int((label_plan == label).sum())
        if count == 0:
            continue
        sampled = synthesizer.sample(count, condition_column=label_column, condition_value=int(label))
        sampled[label_column] = int(label)
        sampled_frames.append(sampled)
    if not sampled_frames:
        return (
            np.empty((0, x_train.shape[1]), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
            {},
        )
    synthetic_table = pd.concat(sampled_frames, ignore_index=True)
    order = rng.permutation(len(synthetic_table))
    synthetic_table = synthetic_table.iloc[order].reset_index(drop=True)
    synthetic_y = synthetic_table[label_column].to_numpy(dtype=np.int32)
    synthetic_features = synthetic_table.loc[:, x_train.columns].apply(pd.to_numeric, errors="coerce")
    synthetic_x = np.clip(
        np.nan_to_num(synthetic_features.to_numpy(dtype=np.float32), nan=0.5, posinf=1.0, neginf=0.0),
        0.0,
        1.0,
    )
    metadata = {
        "ctgan_package_version": getattr(ctgan, "__version__", "unknown"),
        "ctgan_class": "ctgan.CTGAN",
        "embedding_dim": int(latent_dim),
        "generator_hidden_layers": [256, 256],
        "discriminator_hidden_layers": [256, 256],
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "generator_learning_rate": float(learning_rate),
        "discriminator_learning_rate": float(learning_rate),
        "pac": int(pac),
        "enable_gpu": bool(enable_gpu),
    }
    return synthetic_x, synthetic_y, metadata


def _train_and_sample_ctgan(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    n_to_generate: int,
    seed: int,
    latent_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    sampling_strategy: str,
    pac: int,
    enable_gpu: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Run CTGAN in a clean interpreter where PyTorch is imported first.

    On native Windows, importing PyTorch after scikit-learn/TensorFlow can fail
    while loading c10.dll (WinError 1114). The experiment runner necessarily
    imports those libraries first, so CTGAN is isolated in a worker process.
    """

    worker = Path(__file__).with_name("ctgan_worker.py")
    if not worker.is_file():
        raise FileNotFoundError(f"Missing CTGAN worker: {worker}")
    with tempfile.TemporaryDirectory(prefix="hariharan_ctgan_") as temp_dir:
        temp_root = Path(temp_dir)
        inputs_path = temp_root / "inputs.npz"
        outputs_path = temp_root / "outputs.npz"
        metadata_path = temp_root / "metadata.json"
        np.savez_compressed(
            inputs_path,
            x=x_train.to_numpy(dtype=np.float32),
            y=y_train.astype(np.int32),
        )
        command = [
            sys.executable,
            "-u",
            str(worker),
            "--inputs",
            str(inputs_path),
            "--outputs",
            str(outputs_path),
            "--metadata",
            str(metadata_path),
            "--n-to-generate",
            str(n_to_generate),
            "--seed",
            str(seed),
            "--latent-dim",
            str(latent_dim),
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
            "--learning-rate",
            str(learning_rate),
            "--sampling-strategy",
            sampling_strategy,
            "--pac",
            str(pac),
            "--device",
            "cuda" if enable_gpu else "cpu",
        ]
        print(
            f"Starting isolated CTGAN device={'cuda' if enable_gpu else 'cpu'} "
            f"samples={len(x_train)} features={x_train.shape[1]} "
            f"synthetic={n_to_generate} epochs={epochs}",
            flush=True,
        )
        started = time.perf_counter()
        completed = subprocess.run(command)
        if completed.returncode != 0:
            raise RuntimeError(f"Isolated CTGAN worker failed with exit code {completed.returncode}.")
        elapsed = time.perf_counter() - started
        print(f"Finished isolated CTGAN elapsed_s={elapsed:.1f}", flush=True)
        with np.load(outputs_path, allow_pickle=False) as payload:
            synthetic_x = payload["synthetic_x"].astype(np.float32)
            synthetic_y = payload["synthetic_y"].astype(np.int32)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["execution_isolation"] = "fresh_python_process_torch_imported_first"
        metadata["worker_elapsed_seconds"] = float(elapsed)
        return synthetic_x, synthetic_y, metadata


def _augment_borderline_smote(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    seed: int,
    k_neighbors: int,
    m_neighbors: int,
    kind: str,
) -> AugmentationResult:
    from imblearn.over_sampling import BorderlineSMOTE

    y_values = y_train.astype(np.int32)
    class_counts = pd.Series(y_values).value_counts().reindex([0, 1], fill_value=0).astype(int)
    minority_count = int(class_counts.min())
    majority_count = int(class_counts.max())
    if minority_count < 2 or minority_count == majority_count:
        return AugmentationResult(
            x_train=x_train.copy(),
            y_train=y_values.copy(),
            manifest={
                "augmentation": "borderline_smote",
                "augmentation_scope": "train_inner",
                "status": "skipped_not_enough_minority_or_already_balanced",
                "kind": kind,
                "k_neighbors": int(k_neighbors),
                "m_neighbors": int(m_neighbors),
                "n_original_train": int(len(y_values)),
                "n_synthetic": 0,
                "n_augmented_train": int(len(y_values)),
                "real_class_counts": _class_counts(y_values),
                "synthetic_class_counts": {"0": 0, "1": 0},
                "augmented_class_counts": _class_counts(y_values),
            },
        )

    effective_k = max(1, min(int(k_neighbors), minority_count - 1))
    effective_m = max(1, min(int(m_neighbors), len(y_values) - 1))
    sampler = BorderlineSMOTE(
        sampling_strategy="auto",
        random_state=seed,
        k_neighbors=effective_k,
        m_neighbors=effective_m,
        kind=kind,
    )
    x_resampled, y_resampled = sampler.fit_resample(x_train.to_numpy(dtype=np.float32), y_values)
    x_resampled = np.clip(np.nan_to_num(x_resampled.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    y_resampled = y_resampled.astype(np.int32)
    n_synthetic = int(len(y_resampled) - len(y_values))
    synthetic_x = x_resampled[len(y_values) :] if n_synthetic > 0 else np.empty((0, x_train.shape[1]), dtype=np.float32)
    synthetic_y = y_resampled[len(y_values) :] if n_synthetic > 0 else np.empty((0,), dtype=np.int32)
    original_index = x_train.index.astype(str).tolist()
    synthetic_index = [f"synthetic_borderline_smote_{i:05d}" for i in range(n_synthetic)]
    augmented_x = pd.DataFrame(x_resampled, index=[*original_index, *synthetic_index], columns=x_train.columns)
    manifest = {
        "augmentation": "borderline_smote",
        "augmentation_scope": "train_inner",
        "status": "completed",
        "paper_reference": "Diagnostics 2025 Borderline SMOTE train-fold oversampling; validation and test folds remain original.",
        "implementation_note": "BorderlineSMOTE from imbalanced-learn is applied only after train-only feature selection and preprocessing.",
        "kind": kind,
        "k_neighbors": int(k_neighbors),
        "m_neighbors": int(m_neighbors),
        "effective_k_neighbors": int(effective_k),
        "effective_m_neighbors": int(effective_m),
        "n_original_train": int(len(y_values)),
        "n_synthetic": n_synthetic,
        "n_augmented_train": int(len(y_resampled)),
        "real_class_counts": _class_counts(y_values),
        "synthetic_class_counts": _class_counts(synthetic_y),
        "augmented_class_counts": _class_counts(y_resampled),
        **_realism_summary(x_train.to_numpy(dtype=np.float32), synthetic_x),
    }
    return AugmentationResult(x_train=augmented_x, y_train=y_resampled, manifest=manifest)


def augment_training_data(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    mode: str,
    seed: int,
    target_size: int = 2000,
    latent_dim: int = 128,
    epochs: int = 200,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    sampling_strategy: str = "balanced",
    smote_k_neighbors: int = 5,
    smote_m_neighbors: int = 10,
    smote_kind: str = "borderline-1",
    ctgan_pac: int = 1,
    ctgan_cuda: bool = False,
) -> AugmentationResult:
    if mode == "none":
        return AugmentationResult(
            x_train=x_train.copy(),
            y_train=y_train.copy(),
            manifest={
                "augmentation": "none",
                "augmentation_scope": "not_applied",
                "n_original_train": int(len(y_train)),
                "n_augmented_train": int(len(y_train)),
            },
        )
    if mode == "borderline_smote":
        return _augment_borderline_smote(
            x_train=x_train,
            y_train=y_train,
            seed=seed,
            k_neighbors=smote_k_neighbors,
            m_neighbors=smote_m_neighbors,
            kind=smote_kind,
        )

    if mode not in {"gan", "ctgan"}:
        raise ValueError(f"Unsupported augmentation mode: {mode}")

    n_to_generate = max(0, int(target_size) - len(y_train))
    if n_to_generate == 0:
        return AugmentationResult(
            x_train=x_train.copy(),
            y_train=y_train.copy(),
            manifest={
                "augmentation": mode,
                "augmentation_display_name": (
                    "external_ctgan" if mode == "ctgan" else "local_conditional_gan_approximation"
                ),
                "augmentation_scope": "train_inner",
                "status": "skipped_target_size_already_met",
                "reproduction_status": (
                    "paper_aligned_best_effort_external_ctgan_not_fitted"
                    if mode == "ctgan"
                    else "approximation_not_ctgan_not_fitted"
                ),
                "target_size": int(target_size),
                "n_original_train": int(len(y_train)),
                "n_augmented_train": int(len(y_train)),
                "sampling_strategy": sampling_strategy,
            },
        )

    rng = np.random.default_rng(seed + 991)
    if mode == "ctgan":
        synthetic_x, synthetic_y_flat, gan_metadata = _train_and_sample_ctgan(
            x_train=x_train,
            y_train=y_train.astype(np.int32),
            n_to_generate=n_to_generate,
            seed=seed,
            latent_dim=latent_dim,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            sampling_strategy=sampling_strategy,
            pac=ctgan_pac,
            enable_gpu=ctgan_cuda,
        )
    else:
        generator, gan_metadata = _train_conditional_gan(
            x_train.to_numpy(dtype=np.float32),
            y_train.astype(np.int32),
            seed,
            latent_dim,
            epochs,
            batch_size,
            learning_rate,
        )
        synthetic_y_flat = _synthetic_label_plan(
            y_train.astype(np.int32),
            n_to_generate,
            sampling_strategy,
            rng=rng,
        )
        synthetic_y_flat = rng.permutation(synthetic_y_flat).astype(np.int32)
        synthetic_y = synthetic_y_flat.reshape(-1, 1)
        noise = rng.normal(size=(n_to_generate, latent_dim)).astype(np.float32)
        synthetic_x = generator.predict([noise, synthetic_y], verbose=0).astype(np.float32)
        synthetic_x = np.clip(np.nan_to_num(synthetic_x, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

    synthetic_index = [f"synthetic_{mode}_{i:05d}" for i in range(n_to_generate)]
    synthetic_df = pd.DataFrame(synthetic_x, index=synthetic_index, columns=x_train.columns)
    augmented_x = pd.concat([x_train, synthetic_df], axis=0)
    augmented_y = np.concatenate([y_train.astype(np.int32), synthetic_y_flat])
    manifest = {
        "augmentation": mode,
        "augmentation_display_name": (
            "external_ctgan" if mode == "ctgan" else "local_conditional_gan_approximation"
        ),
        "augmentation_scope": "train_inner",
        "status": "completed",
        "paper_reference": "Nature 2026 train-fold CTGAN/TGAN augmentation.",
        "implementation_note": (
            "External ctgan.CTGAN fit and sampled only on train_inner after feature selection and train-only preprocessing."
            if mode == "ctgan"
            else "Local Keras conditional GAN approximation; this is not CTGAN/TGAN and must not be reported as an exact paper reproduction."
        ),
        "reproduction_status": (
            "paper_aligned_best_effort_external_ctgan"
            if mode == "ctgan"
            else "approximation_not_ctgan"
        ),
        "exact_reproduction_possible_from_paper": False,
        "paper_unreported_details": [
            "post-generation filtering/rejection criterion",
            "PAC value",
            "conditional sampling/class-allocation strategy",
            "random seeds",
        ],
        "post_generation_filtering": "not_reproduced; the paper does not disclose the filtering criterion",
        "target_size": int(target_size),
        "sampling_strategy": sampling_strategy,
        "n_original_train": int(len(y_train)),
        "n_synthetic": int(n_to_generate),
        "n_augmented_train": int(len(augmented_y)),
        "real_class_counts": _class_counts(y_train),
        "synthetic_class_counts": _class_counts(synthetic_y_flat),
        "augmented_class_counts": _class_counts(augmented_y),
        **_realism_summary(x_train.to_numpy(dtype=np.float32), synthetic_x),
        **gan_metadata,
    }
    return AugmentationResult(x_train=augmented_x, y_train=augmented_y, manifest=manifest)
