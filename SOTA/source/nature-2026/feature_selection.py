from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, chi2, f_classif
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.tree import DecisionTreeClassifier

COMMON_DIR = Path(__file__).resolve().parents[1]
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from strict_v2_utils import cv_score_estimator


PAPER_INTEGRATED_K = {
    "chi2": 10814,
    "anova": 514,
    "rfe": 1258,
    "elasticnet": 1000,
    "lasso": 500,
    "rf_importance": 500,
}


@dataclass
class FeatureSelectionResult:
    selected_genes: list[str]
    ranking: pd.DataFrame
    metadata: dict[str, Any]


def _sanitize_k(k: int, n_features: int) -> int:
    if k <= 0:
        raise ValueError("k must be positive.")
    return min(int(k), int(n_features))


def _select_all_genes(x_train: pd.DataFrame) -> FeatureSelectionResult:
    genes = x_train.columns.astype(str).tolist()
    ranking = pd.DataFrame({"gene": genes, "rank": np.arange(1, len(genes) + 1)})
    return FeatureSelectionResult(
        selected_genes=genes,
        ranking=ranking,
        metadata={
            "feature_selector": "all_genes",
            "selection_scope": "not_applicable",
            "paper_method": "No feature-selection baseline",
            "k": int(len(genes)),
            "n_selected_genes": int(len(genes)),
            "reproduction_status": "paper_reported_baseline",
        },
    )


def _select_kbest(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    selector_name: str,
    score_func,
    k: int,
) -> FeatureSelectionResult:
    k = _sanitize_k(k, x_train.shape[1])
    selector = SelectKBest(score_func=score_func, k=k)
    selector.fit(x_train, y_train)
    scores = np.nan_to_num(selector.scores_.astype(float), nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    ranking = pd.DataFrame({"gene": x_train.columns.astype(str), "score": scores})
    ranking = ranking.sort_values("score", ascending=False).reset_index(drop=True)
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    selected = ranking.head(k)["gene"].astype(str).tolist()
    return FeatureSelectionResult(
        selected_genes=selected,
        ranking=ranking,
        metadata={
            "feature_selector": selector_name,
            "selection_scope": "train_inner",
            "paper_method": selector_name,
            "k": int(k),
            "n_selected_genes": int(len(selected)),
            "score_func": getattr(score_func, "__name__", str(score_func)),
        },
    )


def _select_chi2_discretized(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    k: int,
    n_bins: int,
) -> FeatureSelectionResult:
    if n_bins < 2:
        raise ValueError("chi2_bins must be at least 2.")
    discretizer = KBinsDiscretizer(
        n_bins=int(n_bins),
        encode="ordinal",
        strategy="quantile",
        subsample=None,
    )
    discrete_values = discretizer.fit_transform(x_train)
    x_discrete = pd.DataFrame(discrete_values, index=x_train.index, columns=x_train.columns)
    result = _select_kbest(x_discrete, y_train, "chi2", chi2, k)
    effective_bins = np.asarray(discretizer.n_bins_, dtype=int)
    result.metadata = {
        **result.metadata,
        "chi2_input": "train-fitted ordinal discretization of train-only MinMax values",
        "chi2_discretizer": "sklearn.preprocessing.KBinsDiscretizer",
        "chi2_bin_strategy": "quantile",
        "chi2_bins_requested": int(n_bins),
        "chi2_effective_bins_min": int(effective_bins.min()),
        "chi2_effective_bins_max": int(effective_bins.max()),
        "chi2_discretizer_fit_scope": "feature_selection_fit_data_only",
        "paper_bin_count_disclosed": False,
        "implementation_assumptions": [
            "The paper requires binning before chi-square but does not disclose the number of bins or binning strategy; quantile bins are used here."
        ],
        "exact_reproduction_possible_from_paper": False,
        "reproduction_status": "paper_aligned_best_effort",
    }
    return result


def _select_rfe_decision_tree(x_train: pd.DataFrame, y_train: np.ndarray, seed: int, k: int, step: float) -> FeatureSelectionResult:
    k = _sanitize_k(k, x_train.shape[1])
    estimator = DecisionTreeClassifier(random_state=seed, max_depth=None)
    selector = RFE(estimator=estimator, n_features_to_select=k, step=step)
    selector.fit(x_train, y_train)
    ranks = selector.ranking_.astype(int)
    selected_mask = selector.support_
    ranking = pd.DataFrame({"gene": x_train.columns.astype(str), "rfe_rank": ranks, "selected": selected_mask})
    ranking = ranking.sort_values(["rfe_rank", "gene"], ascending=[True, True]).reset_index(drop=True)
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    selected = ranking[ranking["selected"]]["gene"].astype(str).tolist()
    return FeatureSelectionResult(
        selected_genes=selected,
        ranking=ranking,
        metadata={
            "feature_selector": "rfe",
            "selection_scope": "train_inner",
            "paper_method": "Recursive Feature Elimination",
            "paper_estimator": "DecisionTreeClassifier",
            "sklearn_step": step,
            "k": int(k),
            "n_selected_genes": int(len(selected)),
        },
    )


def _select_elasticnet(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    seed: int,
    k: int,
    natural: bool = False,
    cv_folds: int = 5,
) -> FeatureSelectionResult:
    k = _sanitize_k(k, x_train.shape[1]) if not natural else x_train.shape[1]
    trace: list[dict[str, Any]] = []
    best_params = {"alpha": 1e-4, "l1_ratio": 0.5}
    sparsity_cap = 2000
    final_safety_cap_applied = False
    if natural:
        best_score = -np.inf
        best_n_nonzero = x_train.shape[1] + 1
        best_valid_score = -np.inf
        best_valid_n_nonzero = x_train.shape[1] + 1
        best_valid_params: dict[str, float] | None = None
        candidate_grid = [
            (1.0, 0.95),
            (3e-1, 0.95),
            (1e-1, 0.9),
            (3e-2, 0.85),
            (1e-2, 0.75),
            (3e-3, 0.75),
            (1e-3, 0.75),
        ]
        for alpha, l1_ratio in candidate_grid:
            print(f"ElasticNet SGD CV evaluating alpha={alpha} l1_ratio={l1_ratio}", flush=True)
            candidate = SGDClassifier(
                loss="log_loss",
                penalty="elasticnet",
                l1_ratio=l1_ratio,
                alpha=alpha,
                max_iter=1500,
                tol=1e-3,
                random_state=seed,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=10,
            )
            mean, fold_scores = cv_score_estimator(candidate, x_train, y_train, folds=cv_folds, seed=seed + 29)
            candidate.fit(x_train, y_train)
            candidate_n_nonzero = int((np.abs(candidate.coef_).ravel() > 0).sum())
            within_sparsity_cap = 0 < candidate_n_nonzero <= sparsity_cap
            trace.append(
                {
                    "alpha": float(alpha),
                    "l1_ratio": float(l1_ratio),
                    "n_nonzero_coefficients": candidate_n_nonzero,
                    "within_sparsity_cap": within_sparsity_cap,
                    "strict_v2_cv_mean": mean,
                    "strict_v2_cv_folds": fold_scores,
                }
            )
            if within_sparsity_cap:
                valid_similar_score = mean >= best_valid_score - 0.01
                if mean > best_valid_score + 0.01 or (valid_similar_score and candidate_n_nonzero < best_valid_n_nonzero):
                    best_valid_score = mean
                    best_valid_n_nonzero = candidate_n_nonzero
                    best_valid_params = {"alpha": alpha, "l1_ratio": l1_ratio}
            similar_score = mean >= best_score - 0.01
            if mean > best_score + 0.01 or (similar_score and 0 < candidate_n_nonzero < best_n_nonzero):
                best_score = mean
                best_n_nonzero = candidate_n_nonzero
                best_params = {"alpha": alpha, "l1_ratio": l1_ratio}
        if best_valid_params is not None:
            best_params = best_valid_params
            best_n_nonzero = best_valid_n_nonzero
        print(f"ElasticNet SGD CV selected alpha={best_params['alpha']} l1_ratio={best_params['l1_ratio']}", flush=True)
    model = SGDClassifier(
        loss="log_loss",
        penalty="elasticnet",
        l1_ratio=float(best_params["l1_ratio"]),
        alpha=float(best_params["alpha"]),
        max_iter=1500,
        tol=1e-3,
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=10,
    )
    model.fit(x_train, y_train)
    coef = np.abs(model.coef_).ravel().astype(float)
    ranking = pd.DataFrame({"gene": x_train.columns.astype(str), "abs_coef": coef})
    ranking = ranking.sort_values("abs_coef", ascending=False).reset_index(drop=True)
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    nonzero = int((coef > 0).sum())
    if natural:
        selected = ranking.loc[ranking["abs_coef"] > 0, "gene"].astype(str).tolist()
        if not selected:
            selected = ranking.head(1)["gene"].astype(str).tolist()
        if len(selected) > sparsity_cap:
            selected = ranking.head(sparsity_cap)["gene"].astype(str).tolist()
            final_safety_cap_applied = True
    else:
        selected = ranking.head(k)["gene"].astype(str).tolist()
    return FeatureSelectionResult(
        selected_genes=selected,
        ranking=ranking,
        metadata={
            "feature_selector": "elasticnet",
            "selection_scope": "train_inner",
            "paper_method": "ElasticNet",
            "implementation": "SGDClassifier(log_loss, penalty=elasticnet)",
            "l1_ratio": float(best_params["l1_ratio"]),
            "alpha": float(best_params["alpha"]),
            "max_iter": 1500,
            "tol": 1e-3,
            "k": None if natural else int(k),
            "sparsity_cap": int(sparsity_cap) if natural else None,
            "final_safety_cap_applied": final_safety_cap_applied,
            "strict_v2_cv_trace": trace,
            "n_nonzero_coefficients": nonzero,
            "n_selected_genes": int(len(selected)),
            "selection_note": (
                "Strict-v2 natural mode: choose ElasticNet regularization by train_inner 5-fold CV, prefer candidates with <=2000 non-zero coefficients, and keep non-zero coefficients with a declared 2000-gene safety cap."
                if natural
                else "Paper uses ElasticNet as embedded selector; this implementation ranks absolute logistic ElasticNet coefficients and keeps k genes."
            ),
        },
    )


def _select_lasso(x_train: pd.DataFrame, y_train: np.ndarray, seed: int, k: int) -> FeatureSelectionResult:
    k = _sanitize_k(k, x_train.shape[1])
    model = LogisticRegression(
        penalty="l1",
        solver="saga",
        C=1.0,
        max_iter=5000,
        random_state=seed,
        n_jobs=1,
    )
    model.fit(x_train, y_train)
    coef = np.abs(model.coef_).ravel().astype(float)
    ranking = pd.DataFrame({"gene": x_train.columns.astype(str), "abs_coef": coef})
    ranking = ranking.sort_values("abs_coef", ascending=False).reset_index(drop=True)
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    selected = ranking.head(k)["gene"].astype(str).tolist()
    return FeatureSelectionResult(
        selected_genes=selected,
        ranking=ranking,
        metadata={
            "feature_selector": "lasso",
            "selection_scope": "train_inner",
            "paper_method": "LASSO comparison method",
            "C": 1.0,
            "max_iter": 5000,
            "k": int(k),
            "n_selected_genes": int(len(selected)),
        },
    )


def _select_rf_importance(x_train: pd.DataFrame, y_train: np.ndarray, seed: int, k: int, n_jobs: int) -> FeatureSelectionResult:
    k = _sanitize_k(k, x_train.shape[1])
    model = RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=n_jobs)
    model.fit(x_train, y_train)
    ranking = pd.DataFrame({"gene": x_train.columns.astype(str), "importance": model.feature_importances_.astype(float)})
    ranking = ranking.sort_values("importance", ascending=False).reset_index(drop=True)
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    selected = ranking.head(k)["gene"].astype(str).tolist()
    return FeatureSelectionResult(
        selected_genes=selected,
        ranking=ranking,
        metadata={
            "feature_selector": "rf_importance",
            "selection_scope": "train_inner",
            "paper_method": "Random Forest importance comparison method",
            "n_estimators": 500,
            "k": int(k),
            "n_selected_genes": int(len(selected)),
        },
    )


def select_features(
    selector_name: str,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    seed: int,
    k_overrides: dict[str, int] | None = None,
    rfe_step: float = 0.2,
    n_jobs: int = 2,
    natural_elasticnet: bool = False,
    cv_folds: int = 5,
    chi2_bins: int = 10,
) -> FeatureSelectionResult:
    if selector_name == "all_genes":
        return _select_all_genes(x_train)
    k_overrides = k_overrides or {}
    k = k_overrides.get(selector_name, PAPER_INTEGRATED_K[selector_name])
    if selector_name == "chi2":
        return _select_chi2_discretized(x_train, y_train, k, chi2_bins)
    if selector_name == "anova":
        return _select_kbest(x_train, y_train, "anova", f_classif, k)
    if selector_name == "rfe":
        return _select_rfe_decision_tree(x_train, y_train, seed, k, rfe_step)
    if selector_name == "elasticnet":
        return _select_elasticnet(x_train, y_train, seed, k, natural=natural_elasticnet, cv_folds=cv_folds)
    if selector_name == "lasso":
        return _select_lasso(x_train, y_train, seed, k)
    if selector_name == "rf_importance":
        return _select_rf_importance(x_train, y_train, seed, k, n_jobs)
    raise ValueError(f"Unsupported feature selector: {selector_name}")
