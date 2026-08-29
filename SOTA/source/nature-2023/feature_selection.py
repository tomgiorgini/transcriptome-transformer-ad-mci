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
from skopt import BayesSearchCV
from skopt.space import Real
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.metrics import average_precision_score, make_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


PR_AUC_SCORER = make_scorer(average_precision_score, response_method="predict")


@dataclass
class FeatureSet:
    name: str
    x_train: np.ndarray
    x_val: np.ndarray
    x_test: np.ndarray
    feature_names: list[str]
    metadata: dict[str, Any]
    selected_genes: list[str] | None = None


def all_genes(x_train: pd.DataFrame, x_val: pd.DataFrame, x_test: pd.DataFrame) -> FeatureSet:
    return FeatureSet(
        name="all_genes",
        x_train=x_train.to_numpy(dtype=np.float32),
        x_val=x_val.to_numpy(dtype=np.float32),
        x_test=x_test.to_numpy(dtype=np.float32),
        feature_names=x_train.columns.astype(str).tolist(),
        selected_genes=None,
        metadata={"status": "ok", "method": "none"},
    )


def lasso_features(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    x_test: pd.DataFrame,
    seed: int,
    n_iter: int,
    cv_folds: int,
    n_jobs: int,
    fixed_alpha: float | None = None,
) -> FeatureSet:
    if fixed_alpha is None:
        search = BayesSearchCV(
            estimator=Lasso(random_state=seed, tol=1e-4, max_iter=3000),
            search_spaces={"alpha": Real(1e-3, 1e2, prior="log-uniform")},
            scoring=PR_AUC_SCORER,
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed + 25),
            n_jobs=n_jobs,
            n_iter=n_iter,
            refit=True,
            random_state=seed,
            verbose=0,
        )
        search.fit(x_train, y_train)
        alpha = float(search.best_params_["alpha"])
        selection_meta = {
            "best_params": search.best_params_,
            "best_score": float(search.best_score_),
            "score": "average_precision",
            "tuning": "BayesSearchCV",
        }
    else:
        alpha = float(fixed_alpha)
        selection_meta = {
            "best_params": {"alpha": alpha},
            "best_score": None,
            "score": "average_precision",
            "tuning": "fixed_paper_optimized_ad",
        }

    selector = SelectFromModel(Lasso(random_state=seed, tol=1e-4, max_iter=3000, alpha=alpha))
    selector.fit(x_train, y_train)
    mask = selector.get_support()
    if not mask.any():
        coef = np.abs(selector.estimator_.coef_)
        mask[np.argmax(coef)] = True
    selected = x_train.columns[mask].astype(str).tolist()
    return FeatureSet(
        name="lasso",
        x_train=x_train.loc[:, selected].to_numpy(dtype=np.float32),
        x_val=x_val.loc[:, selected].to_numpy(dtype=np.float32),
        x_test=x_test.loc[:, selected].to_numpy(dtype=np.float32),
        feature_names=selected,
        selected_genes=selected,
        metadata={
            "status": "ok",
            "method": "Lasso+SelectFromModel",
            **selection_meta,
            "n_selected": len(selected),
        },
    )


def _vssrfe_select(data: pd.DataFrame, y: np.ndarray, n_genes: int, c_value: float, seed: int) -> list[str]:
    x = data.copy()
    n_selected = min(n_genes, x.shape[1])
    step = 100
    previous_n = x.shape[1]
    current_n = x.shape[1]
    clf = LogisticRegression(C=c_value, random_state=seed, class_weight="balanced", penalty="l2", solver="liblinear")

    while current_n > n_selected:
        current_n = max(n_selected, current_n - step)
        if previous_n / max(current_n, 1) > 2 and step > 1:
            previous_n = current_n
            step = max(1, round(step / 2))
        clf.fit(x, y)
        coef_order = np.argsort(np.abs(clf.coef_[0]))[::-1]
        keep = coef_order[:current_n]
        x = x.iloc[:, keep]
    return x.columns.astype(str).tolist()


def _vssrfe_select_path(data: pd.DataFrame, y: np.ndarray, candidates: list[int], c_value: float, seed: int) -> dict[int, list[str]]:
    remaining_targets = sorted(set(candidates), reverse=True)
    x = data.copy()
    step = 100
    previous_n = x.shape[1]
    current_n = x.shape[1]
    selected_by_count: dict[int, list[str]] = {}
    fit_count = 0
    clf = LogisticRegression(C=c_value, random_state=seed, class_weight="balanced", penalty="l2", solver="liblinear")

    for target in remaining_targets:
        target = min(target, x.shape[1])
        while current_n > target:
            current_n = max(target, current_n - step)
            if previous_n / max(current_n, 1) > 2 and step > 1:
                previous_n = current_n
                step = max(1, round(step / 2))
            clf.fit(x, y)
            fit_count += 1
            coef_order = np.argsort(np.abs(clf.coef_[0]))[::-1]
            keep = coef_order[:current_n]
            x = x.iloc[:, keep]
            if fit_count % 25 == 0 or current_n == target:
                print(
                    f"VSSRFE+LR path progress fit={fit_count} remaining_genes={current_n} target={target}",
                    flush=True,
                )
        selected_by_count[target] = x.columns.astype(str).tolist()
        print(f"VSSRFE+LR selected path checkpoint n_genes={target}", flush=True)
    return selected_by_count


def vssrfe_lr_features(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    x_test: pd.DataFrame,
    seed: int,
    n_iter: int,
    cv_folds: int,
    n_jobs: int,
    min_genes: int,
    max_genes: int,
    step_genes: int,
    extra_gene_counts: list[int],
    fixed_c: float | None = None,
    fixed_n_genes: int | None = None,
) -> FeatureSet:
    if fixed_c is None:
        print("VSSRFE+LR tuning LogisticRegression C", flush=True)
        search = BayesSearchCV(
            estimator=LogisticRegression(random_state=seed, class_weight="balanced", penalty="l2", solver="liblinear"),
            search_spaces={"C": Real(1e-7, 1e1, prior="log-uniform")},
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed + 25),
            n_jobs=n_jobs,
            n_iter=n_iter,
            refit=True,
            random_state=seed,
            scoring="average_precision",
            verbose=0,
        )
        search.fit(x_train, y_train)
        c_value = float(search.best_params_["C"])
        selection_meta = {"best_params": search.best_params_, "tuning": "BayesSearchCV", "score": "average_precision"}
        print(f"VSSRFE+LR best C={c_value}", flush=True)
    else:
        c_value = float(fixed_c)
        selection_meta = {
            "best_params": {"C": c_value},
            "tuning": "fixed_paper_optimized_ad",
            "score": "average_precision",
        }
        print(f"VSSRFE+LR fixed paper C={c_value}", flush=True)

    if fixed_n_genes is not None:
        n_genes = max(1, min(int(fixed_n_genes), x_train.shape[1]))
        best_genes = _vssrfe_select_path(x_train, y_train, [n_genes], c_value, seed)[n_genes]
        return FeatureSet(
            name="vssrfe_lr",
            x_train=x_train.loc[:, best_genes].to_numpy(dtype=np.float32),
            x_val=x_val.loc[:, best_genes].to_numpy(dtype=np.float32),
            x_test=x_test.loc[:, best_genes].to_numpy(dtype=np.float32),
            feature_names=best_genes,
            selected_genes=best_genes,
            metadata={
                "status": "ok",
                "method": "VSSRFE+LogisticRegression",
                **selection_meta,
                "best_score": None,
                "candidate_scores": [{"n_genes": int(n_genes), "average_precision": None}],
                "candidate_gene_counts": [n_genes],
                "n_selected": len(best_genes),
            },
        )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed + 11)
    candidate_scores: list[dict[str, float]] = []
    best_genes: list[str] | None = None
    best_score = float("-inf")
    lower = max(1, min(min_genes, x_train.shape[1]))
    upper = max(lower, min(max_genes, x_train.shape[1]))
    step = max(1, step_genes)
    candidates = list(range(lower, upper + 1, step))
    if candidates[-1] != upper:
        candidates.append(upper)
    max_available = x_train.shape[1]
    extra_candidates = [value for value in extra_gene_counts if lower <= value <= max_available]
    candidates = sorted({value for value in candidates + extra_candidates if lower <= value <= max_available})
    candidate_max = max(candidates) if candidates else upper
    # The public script first selected features on the complete training set and
    # then cross-validated the classifier on that already-selected matrix.  That
    # leaks every validation fold into its feature ranking.  Refit VSSRFE inside
    # each fold, use PR-AUC as described in the paper, and only then refit the
    # selected gene count on all train_inner samples.
    scores_by_count: dict[int, list[float]] = {n_genes: [] for n_genes in candidates}
    for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(x_train, y_train), start=1):
        fold_train = x_train.iloc[tr_idx]
        fold_y = y_train[tr_idx]
        fold_path = _vssrfe_select_path(fold_train, fold_y, candidates, c_value, seed + fold_idx)
        for n_genes in candidates:
            selected = fold_path[n_genes]
            clf = LogisticRegression(
                C=c_value,
                random_state=seed + fold_idx,
                class_weight="balanced",
                penalty="l2",
                solver="liblinear",
            )
            clf.fit(fold_train.loc[:, selected], fold_y)
            score = clf.predict_proba(x_train.iloc[va_idx].loc[:, selected])[:, 1]
            scores_by_count[n_genes].append(float(average_precision_score(y_train[va_idx], score)))

    best_n_genes: int | None = None
    for n_genes in candidates:
        print(f"VSSRFE+LR evaluating n_genes={n_genes}/{candidate_max}", flush=True)
        mean_score = float(np.mean(scores_by_count[n_genes]))
        candidate_scores.append({"n_genes": int(n_genes), "average_precision": mean_score})
        if mean_score > best_score:
            best_score = mean_score
            best_n_genes = n_genes

    assert best_n_genes is not None
    best_genes = _vssrfe_select_path(x_train, y_train, [best_n_genes], c_value, seed)[best_n_genes]

    assert best_genes is not None
    return FeatureSet(
        name="vssrfe_lr",
        x_train=x_train.loc[:, best_genes].to_numpy(dtype=np.float32),
        x_val=x_val.loc[:, best_genes].to_numpy(dtype=np.float32),
        x_test=x_test.loc[:, best_genes].to_numpy(dtype=np.float32),
        feature_names=best_genes,
        selected_genes=best_genes,
        metadata={
            "status": "ok",
            "method": "VSSRFE+LogisticRegression",
            **selection_meta,
            "best_score": best_score,
            "candidate_scores": candidate_scores,
            "candidate_gene_counts": candidates,
            "candidate_selection_scope": "VSSRFE refit independently inside every CV fold",
            "n_selected": len(best_genes),
        },
    )


def knowledge_genes_features(
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    x_test: pd.DataFrame,
    gene_file: Path | None,
    mad_top_k: int = 3000,
) -> FeatureSet:
    if mad_top_k < 1:
        raise ValueError("mad_top_k must be positive.")

    medians = x_train.median(axis=0)
    mad = x_train.sub(medians, axis=1).abs().median(axis=0)
    # Stable descending order is intentional.  The public script used
    # argsort()[:3000], accidentally selecting the *lowest* MAD values.
    mad_genes = mad.sort_values(ascending=False, kind="mergesort").head(min(mad_top_k, len(mad))).index.astype(str).tolist()

    curated_input: list[str] = []
    curated_overlap: list[str] = []
    if gene_file is not None and gene_file.exists():
        curated_input = list(dict.fromkeys(line.strip() for line in gene_file.read_text(encoding="utf-8").splitlines() if line.strip()))
        curated_overlap = [gene for gene in curated_input if gene in x_train.columns]

    union = set(mad_genes).union(curated_overlap)
    selected = [str(gene) for gene in x_train.columns if str(gene) in union]
    if gene_file is None or not gene_file.exists():
        status = "incomplete_missing_curated"
    elif not curated_overlap:
        status = "incomplete_no_curated_overlap"
    else:
        status = "ok"
    return FeatureSet(
        name="knowledge_genes",
        x_train=x_train.loc[:, selected].to_numpy(dtype=np.float32),
        x_val=x_val.loc[:, selected].to_numpy(dtype=np.float32),
        x_test=x_test.loc[:, selected].to_numpy(dtype=np.float32),
        feature_names=selected,
        selected_genes=selected,
        metadata={
            "status": status,
            "method": "union(train_inner top-MAD, curated knowledge genes)",
            "gene_file": str(gene_file) if gene_file else None,
            "mad_top_k_requested": int(mad_top_k),
            "mad_genes_selected": len(mad_genes),
            "curated_input_genes": len(curated_input),
            "curated_overlap_genes": len(curated_overlap),
            "n_selected": len(selected),
            "selection_scope": "MAD and curated-list intersection computed on train_inner columns only",
            "reproducibility_note": (
                "Exact curated prev_features.txt is absent from the authors' public repository; "
                "this run is incomplete until --knowledge-genes-file is supplied."
                if status != "ok"
                else "Curated list supplied; union follows the method described in the paper."
            ),
        },
    )


def _torch_vae_latents(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    seed: int,
    epochs: int,
    batch_size: int,
    patience: int,
    architecture: str,
    learning_rate: float,
    reconstruction_loss: str,
    device_name: str,
    *,
    hidden_dims: tuple[int, int, int] = (4096, 1024, 512),
    latent_dim: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Kelly VAE requested CUDA, but torch.cuda.is_available() is false.")
    device = torch.device("cuda" if device_name == "cuda" else "cpu")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False

    h1, h2, h3 = hidden_dims

    def block(in_features: int, out_features: int, *, dropout: bool = False) -> nn.Sequential:
        layers: list[nn.Module] = [nn.Linear(in_features, out_features), nn.ReLU()]
        if architecture in {"batchnorm", "batchnorm_dropout"}:
            layers.append(nn.BatchNorm1d(out_features))
        if dropout and architecture == "batchnorm_dropout":
            layers.append(nn.Dropout(0.2))
        return nn.Sequential(*layers)

    class TorchVAE(nn.Module):
        def __init__(self, feature_num: int) -> None:
            super().__init__()
            self.enc1 = block(feature_num, h1)
            self.enc2 = block(h1, h2, dropout=True)
            self.enc3 = block(h2, h3, dropout=True)
            self.mu = nn.Linear(h3, latent_dim)
            self.sigma = nn.Linear(h3, latent_dim)
            self.mu_norm = nn.BatchNorm1d(latent_dim) if architecture in {"batchnorm", "batchnorm_dropout"} else nn.Identity()
            self.sigma_norm = nn.BatchNorm1d(latent_dim) if architecture in {"batchnorm", "batchnorm_dropout"} else nn.Identity()
            self.dec1 = block(latent_dim, h3)
            self.dec2 = block(h3, h2, dropout=True)
            self.dec3 = block(h2, h1, dropout=True)
            output_layers: list[nn.Module] = [nn.Linear(h1, feature_num), nn.Sigmoid()]
            if architecture in {"batchnorm", "batchnorm_dropout"}:
                output_layers.append(nn.BatchNorm1d(feature_num))
            self.output = nn.Sequential(*output_layers)
            self.apply(self._initialize)

        @staticmethod
        def _initialize(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        def encode(self, values: torch.Tensor) -> torch.Tensor:
            encoded = self.enc3(self.enc2(self.enc1(values)))
            mu = self.mu_norm(self.mu(encoded))
            sigma = torch.clamp(self.sigma_norm(self.sigma(encoded)), -10.0, 10.0)
            return mu + torch.exp(sigma / 2.0) * torch.randn_like(mu)

        def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            latent = self.encode(values)
            decoded = self.output(self.dec3(self.dec2(self.dec1(latent))))
            return decoded, latent

    model = TorchVAE(x_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, foreach=False)
    train_tensor = torch.from_numpy(np.asarray(x_train, dtype=np.float32))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=device.type == "cuda",
    )

    def reconstruction(target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        eps = torch.finfo(prediction.dtype).eps
        if reconstruction_loss == "categorical_crossentropy":
            normalized = prediction / prediction.sum(dim=1, keepdim=True).clamp_min(eps)
            return -(target * normalized.clamp_min(eps).log()).sum(dim=1).mean()
        if reconstruction_loss == "binary_crossentropy":
            clipped = prediction.clamp(eps, 1.0 - eps)
            return -(target * clipped.log() + (1.0 - target) * (1.0 - clipped).log()).mean()
        raise ValueError(f"Unsupported reconstruction loss: {reconstruction_loss}")

    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    wait = 0
    epochs_trained = 0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        sample_count = 0
        for (cpu_batch,) in loader:
            batch = cpu_batch.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            prediction, _ = model(batch)
            loss = reconstruction(batch, prediction)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite Kelly VAE loss at epoch {epoch}: {float(loss.detach().cpu())}")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(batch)
            sample_count += len(batch)
        epoch_loss = total_loss / max(sample_count, 1)
        epochs_trained = epoch
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if epoch == 1 or epoch % 10 == 0 or wait >= patience or epoch == epochs:
            print(
                f"Kelly VAE backend=torch device={device.type} epoch={epoch}/{epochs} "
                f"loss={epoch_loss:.8g} best={best_loss:.8g} wait={wait}/{patience} "
                f"elapsed_s={time.perf_counter() - started:.1f}",
                flush=True,
            )
        if wait >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)

    def encode_array(values: np.ndarray) -> np.ndarray:
        model.eval()
        value_loader = DataLoader(
            TensorDataset(torch.from_numpy(np.asarray(values, dtype=np.float32))),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=device.type == "cuda",
        )
        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for (cpu_batch,) in value_loader:
                latent = model.encode(cpu_batch.to(device, non_blocking=device.type == "cuda"))
                chunks.append(latent.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(chunks, axis=0)

    z_train = encode_array(x_train)
    z_val = encode_array(x_val)
    z_test = encode_array(x_test)
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    elapsed_seconds = float(time.perf_counter() - started)
    del best_state, optimizer, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return z_train, z_val, z_test, {
        "backend": "torch",
        "device": str(device),
        "torch_version": torch.__version__,
        "epochs_trained": epochs_trained,
        "best_training_loss": best_loss,
        "training_elapsed_seconds": elapsed_seconds,
        "parameter_count": parameter_count,
    }


def _torch_vae_latents_subprocess(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    seed: int,
    epochs: int,
    batch_size: int,
    patience: int,
    architecture: str,
    learning_rate: float,
    reconstruction_loss: str,
    device_name: str,
    *,
    hidden_dims: tuple[int, int, int] = (4096, 1024, 512),
    latent_dim: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Train in an isolated interpreter so torch loads before sklearn/TF DLLs."""

    worker = Path(__file__).with_name("torch_vae_worker.py")
    if not worker.is_file():
        raise FileNotFoundError(f"Missing Kelly PyTorch VAE worker: {worker}")
    with tempfile.TemporaryDirectory(prefix="kelly_torch_vae_") as temp_dir:
        temp = Path(temp_dir)
        inputs_path = temp / "inputs.npz"
        outputs_path = temp / "outputs.npz"
        metadata_path = temp / "metadata.json"
        np.savez(
            inputs_path,
            x_train=np.asarray(x_train, dtype=np.float32),
            x_val=np.asarray(x_val, dtype=np.float32),
            x_test=np.asarray(x_test, dtype=np.float32),
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
            "--seed",
            str(seed),
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
            "--patience",
            str(patience),
            "--architecture",
            architecture,
            "--learning-rate",
            str(learning_rate),
            "--reconstruction-loss",
            reconstruction_loss,
            "--device",
            device_name,
            "--hidden-dims",
            *(str(value) for value in hidden_dims),
            "--latent-dim",
            str(latent_dim),
        ]
        subprocess.run(command, check=True)
        if not outputs_path.is_file() or not metadata_path.is_file():
            raise RuntimeError("Kelly PyTorch VAE worker completed without its expected outputs.")
        with np.load(outputs_path) as outputs:
            z_train = outputs["z_train"].astype(np.float32)
            z_val = outputs["z_val"].astype(np.float32)
            z_test = outputs["z_test"].astype(np.float32)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return z_train, z_val, z_test, metadata


def vae_latent_features(
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    x_test: pd.DataFrame,
    seed: int,
    epochs: int,
    batch_size: int,
    patience: int,
    architecture: str,
    learning_rate: float,
    reconstruction_loss: str = "categorical_crossentropy",
    implementation_profile: str = "paper",
    backend: str = "tensorflow",
    device: str = "cpu",
) -> FeatureSet:
    train_values = x_train.to_numpy(dtype=np.float32)
    val_values = x_val.to_numpy(dtype=np.float32)
    test_values = x_test.to_numpy(dtype=np.float32)
    training_meta: dict[str, Any]
    if backend == "torch":
        z_train, z_val, z_test, training_meta = _torch_vae_latents_subprocess(
            train_values,
            val_values,
            test_values,
            seed,
            epochs,
            batch_size,
            patience,
            architecture,
            learning_rate,
            reconstruction_loss,
            device,
        )
    elif backend == "tensorflow":
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
        feature_num = x_train.shape[1]
        inputs = tf.keras.Input(shape=(feature_num,))

        def dense_block(x, units: int, dropout: bool = False):
            x = tf.keras.layers.Dense(units, activation="relu")(x)
            if architecture in {"batchnorm", "batchnorm_dropout"}:
                x = tf.keras.layers.BatchNormalization()(x)
            if dropout and architecture == "batchnorm_dropout":
                x = tf.keras.layers.Dropout(0.2)(x)
            return x

        encoded = dense_block(inputs, 4096)
        encoded = dense_block(encoded, 1024, dropout=True)
        encoded = dense_block(encoded, 512, dropout=True)
        mu = tf.keras.layers.Dense(128, name="latent_mu")(encoded)
        if architecture in {"batchnorm", "batchnorm_dropout"}:
            mu = tf.keras.layers.BatchNormalization()(mu)
        sigma = tf.keras.layers.Dense(128, name="latent_sigma")(encoded)
        if architecture in {"batchnorm", "batchnorm_dropout"}:
            sigma = tf.keras.layers.BatchNormalization()(sigma)

        def sample_z(args):
            z_mu, z_sigma = args
            eps = tf.random.normal(shape=tf.shape(z_mu))
            z_sigma = tf.clip_by_value(z_sigma, -10.0, 10.0)
            return z_mu + tf.exp(z_sigma / 2.0) * eps

        z = tf.keras.layers.Lambda(sample_z, name="z")([mu, sigma])
        decoded = dense_block(z, 512)
        decoded = dense_block(decoded, 1024, dropout=True)
        decoded = dense_block(decoded, 4096, dropout=True)
        outputs = tf.keras.layers.Dense(feature_num, activation="sigmoid")(decoded)
        if architecture in {"batchnorm", "batchnorm_dropout"}:
            outputs = tf.keras.layers.BatchNormalization()(outputs)

        autoencoder = tf.keras.Model(inputs, outputs)
        autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=reconstruction_loss)
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor="loss", patience=patience, restore_best_weights=True)]
        history = autoencoder.fit(
            train_values,
            train_values,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            callbacks=callbacks,
            verbose=0,
        )
        encoder = tf.keras.Model(inputs, z, name="encoder")
        z_train = encoder.predict(train_values, batch_size=batch_size, verbose=0).astype(np.float32)
        z_val = encoder.predict(val_values, batch_size=batch_size, verbose=0).astype(np.float32)
        z_test = encoder.predict(test_values, batch_size=batch_size, verbose=0).astype(np.float32)
        training_meta = {
            "backend": "tensorflow",
            "device": "tensorflow_default",
            "epochs_trained": len(history.history.get("loss", [])),
            "best_training_loss": float(np.min(history.history.get("loss", [np.nan]))),
            "parameter_count": int(autoencoder.count_params()),
        }
    else:
        raise ValueError(f"Unsupported VAE backend: {backend}")
    had_nonfinite = bool((~np.isfinite(z_train)).any() or (~np.isfinite(z_val)).any() or (~np.isfinite(z_test)).any())
    z_train = np.nan_to_num(z_train, nan=0.0, posinf=10.0, neginf=-10.0)
    z_val = np.nan_to_num(z_val, nan=0.0, posinf=10.0, neginf=-10.0)
    z_test = np.nan_to_num(z_test, nan=0.0, posinf=10.0, neginf=-10.0)
    scaler = StandardScaler()
    z_train = scaler.fit_transform(z_train).astype(np.float32)
    z_val = scaler.transform(z_val).astype(np.float32)
    z_test = scaler.transform(z_test).astype(np.float32)
    z_train = np.clip(z_train, -10.0, 10.0)
    z_val = np.clip(z_val, -10.0, 10.0)
    z_test = np.clip(z_test, -10.0, 10.0)

    return FeatureSet(
        name="vae_latent",
        x_train=z_train,
        x_val=z_val,
        x_test=z_test,
        feature_names=[f"vae_z_{idx:03d}" for idx in range(128)],
        selected_genes=None,
        metadata={
            "status": "ok",
            "method": "VAE latent encoder",
            "latent_dim": 128,
            "epochs": epochs,
            "architecture": architecture,
            "learning_rate": learning_rate,
            "reconstruction_loss": reconstruction_loss,
            "implementation_profile": implementation_profile,
            **training_meta,
            "variational_objective": "No KL term: this intentionally follows the paper supplement/public code despite the VAE name.",
            "sampling_logvar_clip": [-10.0, 10.0],
            "had_nonfinite_latent_values": had_nonfinite,
            "latent_postprocessing": "StandardScaler fit on train_inner, then clip [-10, 10]",
        },
    )
