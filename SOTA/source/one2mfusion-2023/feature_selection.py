from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from sklearn.metrics import average_precision_score, make_scorer
from sklearn.model_selection import StratifiedKFold
from skopt import BayesSearchCV
from skopt.space import Real

@dataclass
class SelectedFeatureSet:
    x_train: pd.DataFrame
    x_val: pd.DataFrame
    x_test: pd.DataFrame
    selected_genes: list[str]
    metadata: dict[str, Any]


def use_fixed_input_genes(x_train: pd.DataFrame, x_val: pd.DataFrame, x_test: pd.DataFrame) -> SelectedFeatureSet:
    selected = x_train.columns.astype(str).tolist()
    return SelectedFeatureSet(
        x_train=x_train.copy(),
        x_val=x_val.loc[:, selected].copy(),
        x_test=x_test.loc[:, selected].copy(),
        selected_genes=selected,
        metadata={
            "method": "fixed_input_genes",
            "scope": "genes were selected upstream before this reproduction run",
            "selection_mode": "fixed_input",
            "n_selected": len(selected),
        },
    )


def select_lasso_genes(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    x_test: pd.DataFrame,
    seed: int,
    n_iter: int,
    cv_folds: int,
    n_jobs: int,
    selection_mode: str = "nonzero",
    target_gene_count: int = 492,
    fixed_alpha: float | None = None,
) -> SelectedFeatureSet:
    if fixed_alpha is None:
        scorer = make_scorer(average_precision_score, response_method="predict")
        search = BayesSearchCV(
            estimator=Lasso(random_state=seed, tol=1e-4, max_iter=20000),
            search_spaces={"alpha": Real(1e-7, 1e-1, prior="log-uniform")},
            scoring=scorer,
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed + 17),
            n_jobs=n_jobs,
            n_iter=n_iter,
            refit=True,
            random_state=seed,
            verbose=0,
        )
        search.fit(x_train, y_train)
        alpha = float(search.best_params_["alpha"])
        best_score = float(search.best_score_)
        selection_meta = {
            "best_params": {"alpha": alpha},
            "bayes_best_params": search.best_params_,
            "best_score": best_score,
            "score": "average_precision",
            "tuning": "BayesSearchCV",
            "n_iter": n_iter,
            "cv_folds": cv_folds,
        }
    else:
        alpha = float(fixed_alpha)
        selection_meta = {
            "best_params": {"alpha": alpha},
            "best_score": None,
            "tuning": "fixed_paper",
            "n_iter": 0,
            "cv_folds": 0,
        }

    selector = SelectFromModel(Lasso(random_state=seed, tol=1e-4, max_iter=20000, alpha=alpha))
    selector.fit(x_train, y_train)
    coef = np.abs(selector.estimator_.coef_)
    nonzero_mask = coef > 1e-10
    if selection_mode == "nonzero":
        mask = nonzero_mask
        if not mask.any():
            fallback = int(np.argmax(coef))
            mask[fallback] = True
        selected = x_train.columns[mask].astype(str).tolist()
    elif selection_mode == "fixed_k":
        selected = _fixed_k_genes(x_train, y_train, coef, target_gene_count)
    else:
        raise ValueError(f"Unsupported selection_mode: {selection_mode}")
    return SelectedFeatureSet(
        x_train=x_train.loc[:, selected].copy(),
        x_val=x_val.loc[:, selected].copy(),
        x_test=x_test.loc[:, selected].copy(),
        selected_genes=selected,
        metadata={
            "method": "Lasso+SelectFromModel",
            "scope": "fit on train_inner only",
            "selection_mode": selection_mode,
            "target_gene_count": target_gene_count if selection_mode == "fixed_k" else None,
            **selection_meta,
            "n_nonzero": int(nonzero_mask.sum()),
            "n_selected": len(selected),
        },
    )


def _fixed_k_genes(x_train: pd.DataFrame, y_train: np.ndarray, coef: np.ndarray, target_gene_count: int) -> list[str]:
    k = min(max(1, target_gene_count), x_train.shape[1])
    coef_order = np.argsort(coef)[::-1]
    selected_idx: list[int] = [int(idx) for idx in coef_order if coef[idx] > 0][:k]
    if len(selected_idx) < k:
        selected = set(selected_idx)
        fisher = _fisher_scores(x_train, y_train).to_numpy()
        fisher_order = np.argsort(fisher)[::-1]
        for idx in fisher_order:
            int_idx = int(idx)
            if int_idx not in selected:
                selected_idx.append(int_idx)
                selected.add(int_idx)
            if len(selected_idx) >= k:
                break
    return x_train.columns[selected_idx[:k]].astype(str).tolist()


def _fisher_scores(x_train: pd.DataFrame, y_train: np.ndarray) -> pd.Series:
    x0 = x_train.loc[y_train == 0]
    x1 = x_train.loc[y_train == 1]
    mean0 = x0.mean(axis=0)
    mean1 = x1.mean(axis=0)
    var0 = x0.var(axis=0).replace(0.0, np.nan)
    var1 = x1.var(axis=0).replace(0.0, np.nan)
    scores = ((mean1 - mean0) ** 2) / (var0 + var1)
    return scores.replace([np.inf, -np.inf], np.nan).fillna(0.0)
