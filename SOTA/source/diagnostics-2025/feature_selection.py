from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

COMMON_DIR = Path(__file__).resolve().parents[1]
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from strict_v2_utils import cv_score_estimator, strict_v2_fs_score


PAPER_XGBOOST_TOP_K = 300
PAPER_SFBS_TARGET_GENES = 95
PAPER_UNREPORTED_FS_DETAILS = [
    "XGBoost hyperparameters",
    "SFBS estimator",
    "SFBS scoring function",
    "SFBS stopping/tie-breaking rules",
    "feature-selection random seeds",
]


@dataclass
class FeatureSelectionResult:
    selected_genes: list[str]
    ranking: pd.DataFrame
    trace: pd.DataFrame
    metadata: dict[str, Any]


def _xgboost_importance_ranking(x_train: pd.DataFrame, y_train: np.ndarray, seed: int, n_jobs: int) -> pd.DataFrame:
    model = XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=seed,
        n_jobs=n_jobs,
    )
    model.fit(x_train, y_train)
    importances = np.asarray(model.feature_importances_, dtype=float)
    ranking = pd.DataFrame({"gene": x_train.columns.astype(str), "xgboost_importance": importances})
    ranking = ranking.sort_values("xgboost_importance", ascending=False).reset_index(drop=True)
    ranking["xgboost_rank"] = np.arange(1, len(ranking) + 1)
    return ranking


def _lr_rank_within_top_genes(x_top: pd.DataFrame, y_train: np.ndarray, seed: int) -> list[str]:
    estimator = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="liblinear",
        max_iter=5000,
        random_state=seed,
    )
    estimator.fit(x_top, y_train)
    coef = np.abs(estimator.coef_).ravel()
    order = np.argsort(coef)[::-1]
    return x_top.columns[order].astype(str).tolist()


def _evaluate_candidate_sizes(
    x_ranked: pd.DataFrame,
    y_train: np.ndarray,
    ranked_genes: list[str],
    min_genes: int,
    max_genes: int,
    step: int,
    cv_folds: int,
    seed: int,
) -> tuple[list[str], pd.DataFrame]:
    max_genes = min(max_genes, len(ranked_genes))
    min_genes = min(min_genes, max_genes)
    candidate_sizes = sorted(set([*range(min_genes, max_genes + 1, step), max_genes]))
    rows: list[dict[str, Any]] = []
    estimator = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="liblinear",
        max_iter=5000,
        random_state=seed,
    )
    best_size = candidate_sizes[0]
    best_score = -np.inf
    for size in candidate_sizes:
        genes = ranked_genes[:size]
        mean, scores = cv_score_estimator(estimator, x_ranked.loc[:, genes], y_train, folds=cv_folds, seed=seed)
        std = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
        rows.append({"n_genes": int(size), "strict_v2_cv_mean": mean, "strict_v2_cv_std": std})
        if mean > best_score:
            best_score = mean
            best_size = size
    trace = pd.DataFrame(rows)
    return ranked_genes[:best_size], trace


def _strict_v2_probability_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_score = np.asarray(y_score)
    if y_score.ndim == 2:
        y_score = y_score[:, 1] if y_score.shape[1] > 1 else y_score[:, 0]
    return strict_v2_fs_score(np.asarray(y_true, dtype=int), y_score)


def _true_sfbs_select(
    x_top: pd.DataFrame,
    y_train: np.ndarray,
    target_genes: int,
    cv_folds: int,
    seed: int,
    n_jobs: int,
) -> tuple[list[str], pd.DataFrame, dict[str, Any]]:
    if target_genes <= 0:
        raise ValueError("target_genes must be positive.")
    effective_target = min(int(target_genes), int(x_top.shape[1]))
    estimator = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="liblinear",
        max_iter=5000,
        random_state=seed,
    )
    scorer = make_scorer(_strict_v2_probability_score, response_method="predict_proba")
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    selector = SequentialFeatureSelector(
        estimator,
        k_features=effective_target,
        forward=False,
        floating=True,
        scoring=scorer,
        cv=cv,
        n_jobs=n_jobs,
        clone_estimator=True,
        verbose=0,
    )
    selector.fit(x_top, y_train)
    selected = list(selector.k_feature_names_)
    rows: list[dict[str, Any]] = []
    for n_features, subset in selector.subsets_.items():
        rows.append(
            {
                "n_genes": int(n_features),
                "avg_score": float(subset.get("avg_score", np.nan)),
                "cv_scores": list(map(float, subset.get("cv_scores", []))),
                "selected_feature_names": "|".join(map(str, subset.get("feature_names", []))),
            }
        )
    trace = pd.DataFrame(rows).sort_values("n_genes")
    metadata = {
        "sfbs_implementation": "mlxtend.SequentialFeatureSelector",
        "sfbs_forward": False,
        "sfbs_floating": True,
        "sfbs_estimator": "LogisticRegression(L2, liblinear, max_iter=5000)",
        "sfbs_score": "strict_v2_fs_score",
        "sfbs_k_features": int(effective_target),
        "sfbs_fixed_target_requested": int(target_genes),
        "sfbs_fixed_target_effective": int(effective_target),
        "sfbs_target_was_capped_to_available_features": bool(effective_target != target_genes),
        "cv_folds": int(cv_folds),
        "selected_genes": int(len(selected)),
        "best_score": float(selector.k_score_),
        "reproduction_status": "paper_aligned_best_effort",
        "exact_reproduction_possible_from_paper": False,
        "implementation_assumptions": [
            "L2 logistic regression is used as the SFBS estimator.",
            "strict_v2_fs_score is used because the paper does not disclose the SFBS scorer.",
            "mlxtend floating backward selection supplies stopping and tie-breaking behavior.",
        ],
        "paper_unreported_details": PAPER_UNREPORTED_FS_DETAILS,
    }
    return selected, trace, metadata


def select_xgboost_sfbs_genes(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    seed: int,
    top_k: int = 300,
    min_genes: int = 20,
    max_genes: int = 95,
    step: int = 5,
    cv_folds: int = 5,
    n_jobs: int = 2,
    sfbs_mode: str = "true",
    target_genes: int = PAPER_SFBS_TARGET_GENES,
    allow_sfbs_fallback: bool = False,
) -> FeatureSelectionResult:
    ranking = _xgboost_importance_ranking(x_train, y_train, seed, n_jobs)
    top_genes = ranking.head(min(top_k, len(ranking)))["gene"].astype(str).tolist()
    try:
        if sfbs_mode == "true":
            selected, trace, sfbs_metadata = _true_sfbs_select(
                x_train.loc[:, top_genes],
                y_train,
                target_genes=target_genes,
                cv_folds=cv_folds,
                seed=seed,
                n_jobs=n_jobs,
            )
        elif sfbs_mode == "approximate_lr":
            lr_ranked = _lr_rank_within_top_genes(x_train.loc[:, top_genes], y_train, seed)
            selected, trace = _evaluate_candidate_sizes(
                x_train.loc[:, lr_ranked],
                y_train,
                lr_ranked,
                min_genes=min_genes,
                max_genes=max_genes,
                step=step,
                cv_folds=cv_folds,
                seed=seed,
            )
            sfbs_metadata = {
                "sfbs_implementation": "approximate_lr_ranking",
                "sfbs_estimator": "LogisticRegression(L2, liblinear, max_iter=5000)",
                "sfbs_approximation": "LR coefficient ranking inside XGBoost top-k plus CV over subset sizes",
                "sfbs_score": "strict_v2_fs_score",
                "min_genes": int(min_genes),
                "max_genes": int(max_genes),
                "step": int(step),
                "cv_folds": int(cv_folds),
                "selected_genes": int(len(selected)),
                "reproduction_status": "approximation_not_sfbs",
                "exact_reproduction_possible_from_paper": False,
                "implementation_assumptions": [
                    "L2 logistic-regression coefficient ranking substitutes for sequential floating backward selection.",
                    "Candidate subset sizes are scored with strict_v2_fs_score.",
                ],
                "paper_unreported_details": PAPER_UNREPORTED_FS_DETAILS,
            }
        else:
            raise ValueError(f"Unsupported sfbs_mode: {sfbs_mode}")
        metadata = {
            "feature_selector": (
                "xgboost_topk_sfbs_fixed_k"
                if sfbs_mode == "true"
                else "xgboost_topk_approximate_lr_not_sfbs"
            ),
            "sfbs_mode": sfbs_mode,
            "xgboost_scope": "train_inner",
            "sfbs_scope": "train_inner",
            "top_k": int(top_k),
            "paper_xgboost_top_k": PAPER_XGBOOST_TOP_K,
            "paper_sfbs_target_genes": PAPER_SFBS_TARGET_GENES,
            "sfbs_target_genes": int(target_genes) if sfbs_mode == "true" else None,
            "min_genes": int(min_genes),
            "max_genes": int(max_genes),
            "step": int(step),
            "cv_folds": int(cv_folds),
            "selected_genes": int(len(selected)),
            **sfbs_metadata,
            "fallback_used": False,
            "fallback_allowed": bool(allow_sfbs_fallback),
            "xgboost_parameter_source": "implementation assumption; the paper does not disclose XGBoost hyperparameters",
        }
    except Exception as exc:
        if not allow_sfbs_fallback:
            raise RuntimeError(
                "SFBS failed and fallback is disabled. Re-run with allow_sfbs_fallback=True "
                "(CLI: --allow-sfbs-fallback) only if an explicitly labelled XGBoost-only "
                "fallback is acceptable."
            ) from exc
        fallback_target = target_genes if sfbs_mode == "true" else PAPER_SFBS_TARGET_GENES
        selected = top_genes[: min(int(fallback_target), len(top_genes))]
        trace = pd.DataFrame(
            [
                {
                    "n_genes": int(len(selected)),
                    "macro_f1_cv_mean": np.nan,
                    "macro_f1_cv_std": np.nan,
                    "strict_v2_cv_mean": np.nan,
                    "strict_v2_cv_std": np.nan,
                    "fallback_reason": repr(exc),
                }
            ]
        )
        metadata = {
            "feature_selector": "xgboost_ranked_fallback_not_sfbs",
            "sfbs_mode_requested": sfbs_mode,
            "xgboost_scope": "train_inner",
            "sfbs_scope": "failed_before_completion",
            "top_k": int(top_k),
            "paper_xgboost_top_k": PAPER_XGBOOST_TOP_K,
            "paper_sfbs_target_genes": PAPER_SFBS_TARGET_GENES,
            "selected_genes": int(len(selected)),
            "fallback_used": True,
            "fallback_allowed": True,
            "fallback_reason": repr(exc),
            "reproduction_status": "fallback_not_sfbs",
            "exact_reproduction_possible_from_paper": False,
            "implementation_assumptions": [
                "SFBS failed; the retained genes are the highest-ranked XGBoost features only."
            ],
            "paper_unreported_details": PAPER_UNREPORTED_FS_DETAILS,
            "xgboost_parameter_source": "implementation assumption; the paper does not disclose XGBoost hyperparameters",
        }
    return FeatureSelectionResult(selected_genes=selected, ranking=ranking, trace=trace, metadata=metadata)
