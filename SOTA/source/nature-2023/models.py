from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer, Real
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: Any
    search_space: dict[str, Any] | None


class PaperMLPClassifier(MLPClassifier):
    def set_params(self, **params: Any) -> "PaperMLPClassifier":
        hidden = params.get("hidden_layer_sizes")
        if isinstance(hidden, str):
            params["hidden_layer_sizes"] = tuple(int(part) for part in hidden.split("_"))
        return super().set_params(**params)


def _feature_key(feature_set_name: str | None) -> str:
    aliases = {
        "all_genes": "all_genes",
        "knowledge_genes": "knowledge_genes",
        "lasso": "lasso",
        "vssrfe_lr": "vssrfe_lr",
        "vae_latent": "vae",
    }
    return aliases.get(feature_set_name or "all_genes", feature_set_name or "all_genes")


def _resolve_xgboost_device(xgboost_device: str) -> str:
    if xgboost_device == "auto":
        # XGBoost does not expose a reliable, cheap CUDA-capability probe across
        # all versions supported by this repository.  CPU is the safe automatic
        # choice; callers can still explicitly request --xgboost-device cuda.
        return "cpu"
    return xgboost_device


def paper_optimized_estimator(
    model_name: str,
    feature_set_name: str | None,
    seed: int,
    xgboost_device: str = "auto",
) -> tuple[Any, dict[str, Any]]:
    feature = _feature_key(feature_set_name)
    fixed: dict[str, dict[str, dict[str, Any]]] = {
        "lr": {
            "all_genes": {"C": 0.00045526282090663776},
            "knowledge_genes": {"C": 0.035943068595763565},
            "lasso": {"C": 0.10533408514169781},
            "vae": {"C": 0.0016669407132495569},
            "vssrfe_lr": {"C": 0.2539120306026563},
        },
        "svm": {
            "all_genes": {"C": 29.187403716727133, "gamma": 1.2659524334338536e-05},
            "knowledge_genes": {"C": 6.410836753079676, "gamma": 2.8594305992584455e-05},
            "lasso": {"C": 34.310224317710926, "gamma": 0.00010753983888006124},
            "vae": {"C": 3.4028684317983213, "gamma": 0.021770708275534352},
            "vssrfe_lr": {"C": 1.1877734194503269, "gamma": 0.0006219073574834002},
        },
        "rf": {
            "all_genes": {"max_depth": 7, "min_samples_leaf": 11, "n_estimators": 3730, "min_samples_split": 15, "max_features": "sqrt"},
            "knowledge_genes": {"max_depth": 2, "min_samples_leaf": 13, "n_estimators": 451, "min_samples_split": 15, "max_features": "sqrt"},
            "lasso": {"max_depth": 2, "min_samples_leaf": 4, "n_estimators": 1157, "min_samples_split": 11, "max_features": 1},
            "vae": {"max_depth": 13, "min_samples_leaf": 1, "n_estimators": 1605, "min_samples_split": 133, "max_features": 1},
            "vssrfe_lr": {"max_depth": 9, "min_samples_leaf": 1, "n_estimators": 4178, "min_samples_split": 2, "max_features": 15},
        },
        "mlp": {
            "all_genes": {"alpha": 1e-6, "activation": "relu", "hidden_layer_sizes": (100,), "solver": "sgd", "max_iter": 1000},
            "knowledge_genes": {"alpha": 5, "activation": "relu", "hidden_layer_sizes": (15, 15), "solver": "lbfgs", "max_iter": 1000},
            "lasso": {"alpha": 0.1, "activation": "relu", "hidden_layer_sizes": (10,), "solver": "adam", "max_iter": 1000},
            "vae": {"alpha": 0.2, "activation": "relu", "hidden_layer_sizes": (100,), "solver": "sgd", "max_iter": 10000, "tol": 1e-7},
            "vssrfe_lr": {"alpha": 0.0001, "activation": "relu", "hidden_layer_sizes": (100,), "solver": "lbfgs", "max_iter": 1000},
        },
        "xgboost": {
            "all_genes": {
                "scale_pos_weight": 0.9562043795620438,
                "learning_rate": 1e-1,
                "n_estimators": 81,
                "max_depth": 7,
                "min_child_weight": 4,
                "gamma": 0,
                "colsample_bytree": 0.9990205842114916,
                "subsample": 0.998300612263123,
                "reg_alpha": 0.0002456245246915084,
                "reg_lambda": 0,
            },
            "knowledge_genes": {
                "scale_pos_weight": 0.9562043795620438,
                "learning_rate": 8e-2,
                "n_estimators": 61,
                "max_depth": 7,
                "min_child_weight": 4,
                "gamma": 0.7528211159540363,
                "colsample_bytree": 1,
                "subsample": 0.8796038044197849,
                "reg_alpha": 0.003631221742334404,
                "reg_lambda": 0,
            },
            "lasso": {
                "scale_pos_weight": 0.9562043795620438,
                "learning_rate": 2e-6,
                "n_estimators": 219,
                "max_depth": 2,
                "min_child_weight": 2,
                "gamma": 0.09701422432113668,
                "colsample_bytree": 0.9995941818118101,
                "subsample": 0.4506822043870306,
                "reg_alpha": 0.31189385846388495,
                "reg_lambda": 0,
            },
            "vae": {
                "scale_pos_weight": 0.9562043795620438,
                "learning_rate": 4e-3,
                "n_estimators": 21,
                "max_depth": 7,
                "min_child_weight": 4,
                "gamma": 0.5840810014249553,
                "colsample_bytree": 0.5195812889570743,
                "subsample": 0.6816009688535446,
                "reg_alpha": 0,
                "reg_lambda": 9.991851172872426,
            },
            "vssrfe_lr": {
                "scale_pos_weight": 0.9562043795620438,
                "learning_rate": 2e-1,
                "n_estimators": 202,
                "max_depth": 13,
                "min_child_weight": 2,
                "gamma": 0.16713401469455924,
                "colsample_bytree": 0.6067127231029593,
                "subsample": 0.600327569498154,
                "reg_alpha": 0.0043058784736429035,
                "reg_lambda": 0,
            },
        },
    }
    if model_name not in fixed or feature not in fixed[model_name]:
        raise ValueError(f"No fixed paper hyperparameters for model={model_name}, feature_set={feature_set_name}.")
    params = fixed[model_name][feature]
    if model_name == "lr":
        estimator = LogisticRegression(random_state=2, class_weight="balanced", penalty="l2", solver="liblinear", **params)
    elif model_name == "svm":
        estimator = SVC(random_state=142, kernel="rbf", class_weight="balanced", probability=True, **params)
    elif model_name == "rf":
        estimator = RandomForestClassifier(random_state=10, class_weight="balanced", n_jobs=-1, **params)
    elif model_name == "mlp":
        estimator = MLPClassifier(random_state=10, verbose=False, **params)
    elif model_name == "xgboost":
        if XGBClassifier is None:
            raise ImportError("xgboost is not installed.")
        estimator = XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            random_state=42,
            n_jobs=2,
            tree_method="hist",
            device=_resolve_xgboost_device(xgboost_device),
            **params,
        )
    else:
        raise ValueError(model_name)
    return estimator, {"best_params": params, "best_score": None, "tuning": "fixed_paper_optimized_ad", "feature_set": feature}


def model_specs(seed: int, xgboost_device: str = "auto") -> dict[str, ModelSpec]:
    specs = {
        "lr": ModelSpec(
            "lr",
            LogisticRegression(random_state=seed, class_weight="balanced", penalty="l2", solver="liblinear", max_iter=5000),
            {"C": Real(1e-7, 1e1, prior="log-uniform")},
        ),
        "svm": ModelSpec(
            "svm",
            SVC(random_state=seed, kernel="rbf", class_weight="balanced", probability=True),
            {"C": Real(1e-4, 1e3, prior="log-uniform"), "gamma": Real(1e-6, 1e1, prior="log-uniform")},
        ),
        "rf": ModelSpec(
            "rf",
            RandomForestClassifier(random_state=seed, class_weight="balanced", n_jobs=-1),
            {
                "n_estimators": Integer(100, 800),
                "max_depth": Integer(2, 20),
                "min_samples_leaf": Integer(1, 20),
                "min_samples_split": Integer(2, 30),
                "max_features": Categorical(["sqrt", "log2"]),
            },
        ),
        "mlp": ModelSpec(
            "mlp",
            PaperMLPClassifier(random_state=seed, early_stopping=False, max_iter=10000, tol=0.00001),
            {
                "activation": Categorical(["identity", "logistic", "tanh", "relu"]),
                "hidden_layer_sizes": Categorical(["50", "100", "128", "100_50", "128_64", "256_128"]),
                "solver": Categorical(["lbfgs", "sgd", "adam"]),
            },
        ),
    }
    if XGBClassifier is not None:
        specs["xgboost"] = ModelSpec(
            "xgboost",
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="aucpr",
                random_state=seed,
                n_jobs=2,
                tree_method="hist",
                device=_resolve_xgboost_device(xgboost_device),
            ),
            {
                "n_estimators": Integer(50, 500),
                "learning_rate": Real(1e-3, 2e-1, prior="log-uniform"),
                "scale_pos_weight": Real(0.25, 4.0, prior="log-uniform"),
                "max_depth": Integer(2, 10),
                "min_child_weight": Integer(1, 10),
                "gamma": Real(1e-12, 1.0, prior="log-uniform"),
                "subsample": Real(0.6, 1.0),
                "colsample_bytree": Real(0.6, 1.0),
                "reg_alpha": Real(1e-12, 1.0, prior="log-uniform"),
                "reg_lambda": Real(1e-12, 5.0, prior="log-uniform"),
            },
        )
    return specs


def fit_tuned_model(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    n_iter: int,
    cv_folds: int,
    n_jobs: int,
    feature_set_name: str | None = None,
    hyperparameter_mode: str = "bayes",
    xgboost_device: str = "auto",
) -> tuple[Any, dict[str, Any]]:
    x_train = np.nan_to_num(np.asarray(x_train, dtype=np.float32), nan=0.0, posinf=10.0, neginf=-10.0)
    x_train = np.clip(x_train, -10.0, 10.0)
    if hyperparameter_mode == "fixed_paper":
        estimator, meta = paper_optimized_estimator(model_name, feature_set_name, seed, xgboost_device)
        estimator.fit(x_train, y_train)
        return estimator, meta

    spec = model_specs(seed, xgboost_device)[model_name]
    if spec.search_space is None or n_iter <= 0:
        estimator = spec.estimator.fit(x_train, y_train)
        return estimator, {"best_params": {}, "best_score": None, "tuning": "none"}

    search = BayesSearchCV(
        estimator=spec.estimator,
        search_spaces=spec.search_space,
        scoring="average_precision",
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed),
        n_iter=n_iter,
        n_jobs=n_jobs,
        random_state=seed,
        refit=True,
        verbose=0,
    )
    search.fit(x_train, y_train)
    return search.best_estimator_, {
        "best_params": search.best_params_,
        "best_score": float(search.best_score_),
        "score": "average_precision",
        "tuning": "BayesSearchCV",
        "n_iter": n_iter,
        "cv_folds": cv_folds,
    }


def available_model_names() -> list[str]:
    return sorted(model_specs(1).keys())
