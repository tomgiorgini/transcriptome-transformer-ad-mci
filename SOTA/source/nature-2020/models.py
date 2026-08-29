from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def build_paper_estimator(
    model_name: str,
    n_features: int,
    seed: int,
    class_weight: str | None = None,
) -> tuple[Any, dict[str, object]]:
    if n_features <= 0:
        raise ValueError("n_features must be positive.")
    if class_weight not in {None, "balanced"}:
        raise ValueError("class_weight must be None or 'balanced'.")
    if model_name == "lr":
        estimator = LogisticRegression(
            penalty=None,
            solver="lbfgs",
            class_weight=class_weight,
            max_iter=5000,
            random_state=seed,
        )
        params = {
            "penalty": None,
            "solver": "lbfgs",
            "class_weight": class_weight,
            "paper_note": "The paper reports standard logistic regression and does not report class weighting.",
        }
    elif model_name == "l1_lr":
        estimator = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=10000.0,
            class_weight=class_weight,
            max_iter=5000,
            random_state=seed,
        )
        params = {
            "penalty": "l1",
            "lambda": 0.0001,
            "C": 10000.0,
            "solver": "liblinear",
            "class_weight": class_weight,
            "implementation_note": "C=1/lambda is a disclosed scikit-learn approximation of the reported R L1-GLM objective.",
        }
    elif model_name == "svm":
        gamma = 1.0 / float(n_features)
        estimator = make_pipeline(
            StandardScaler(),
            SVC(
                kernel="rbf",
                C=1.0,
                gamma=gamma,
                probability=True,
                class_weight=class_weight,
                random_state=seed,
            ),
        )
        params = {
            "kernel": "rbf",
            "C": 1.0,
            "gamma": gamma,
            "feature_scaling": "StandardScaler fit on train_inner (e1071 scale=TRUE equivalent)",
            "probability": True,
            "class_weight": class_weight,
        }
    elif model_name == "rf":
        estimator = RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            class_weight=class_weight,
            random_state=seed,
            n_jobs=-1,
        )
        params = {"n_estimators": 500, "max_features": "sqrt", "class_weight": class_weight}
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return estimator, params


def positive_scores(estimator: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(x)[:, 1]
    if hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(x)
        return 1.0 / (1.0 + np.exp(-scores))
    return estimator.predict(x).astype(float)


def available_classical_models() -> list[str]:
    return ["lr", "l1_lr", "svm", "rf"]
