"""
Fairness evaluation using Fairlearn metrics for classification and
custom group-fairness proxies for regression tasks.
"""
import numpy as np
import pandas as pd
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from app.core.logging_config import get_logger

logger = get_logger(__name__)


def evaluate_fairness(
    model, X_test: pd.DataFrame, y_test: pd.Series, sensitive_col: str, task_type: str
) -> dict:
    """Compute fairness metrics using the sensitive attribute column."""
    if sensitive_col not in X_test.columns:
        return {"error": f"Sensitive attribute '{sensitive_col}' not found in dataset."}

    y_pred = model.predict(X_test)
    sensitive_features = X_test[sensitive_col]

    metrics: dict = {}

    try:
        if task_type.lower() == "classification":
            dpd = demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive_features)
            eod = equalized_odds_difference(y_test, y_pred, sensitive_features=sensitive_features)

            # Per-group positive prediction rates
            group_rates = _per_group_positive_rates(y_pred, sensitive_features)

            # Heuristic composite fairness score
            fairness_score = max(0.0, 100 - (abs(dpd) * 100) - (abs(eod) * 50))

            metrics = {
                "demographic_parity_diff": round(float(dpd), 4),
                "equalized_odds_diff": round(float(eod), 4),
                "group_positive_rates": group_rates,
                "fairness_score": round(float(fairness_score), 2),
            }
        else:
            # Regression: compute group-wise mean prediction disparity
            metrics = _regression_fairness(y_test, y_pred, sensitive_features)

    except Exception as exc:
        logger.error("Fairness evaluation failed: %s", exc, exc_info=True)
        metrics = {"error": str(exc), "fairness_score": None}

    logger.info("Fairness evaluation complete — score=%s", metrics.get("fairness_score"))
    return metrics


def _per_group_positive_rates(y_pred, sensitive_features: pd.Series) -> dict:
    """Calculate the positive-prediction rate for each group in the sensitive attribute."""
    groups = sensitive_features.unique()
    rates = {}
    for g in groups:
        mask = sensitive_features == g
        group_preds = y_pred[mask]
        rate = float(np.mean(group_preds)) if len(group_preds) > 0 else 0.0
        rates[str(g)] = round(rate, 4)
    return rates


def _regression_fairness(y_true, y_pred, sensitive_features: pd.Series) -> dict:
    """
    For regression tasks, measure fairness as the disparity in mean prediction
    across groups of the sensitive attribute.
    """
    groups = sensitive_features.unique()
    group_means = {}
    for g in groups:
        mask = sensitive_features == g
        group_means[str(g)] = round(float(np.mean(y_pred[mask])), 4)

    if len(group_means) < 2:
        return {"message": "Need at least 2 groups for fairness comparison.", "fairness_score": 100.0}

    vals = list(group_means.values())
    max_disparity = max(vals) - min(vals)
    overall_mean = float(np.mean(y_pred))
    relative_disparity = (max_disparity / abs(overall_mean)) if overall_mean != 0 else 0.0

    fairness_score = max(0.0, 100 - relative_disparity * 100)

    return {
        "group_mean_predictions": group_means,
        "max_disparity": round(float(max_disparity), 4),
        "relative_disparity": round(float(relative_disparity), 4),
        "fairness_score": round(float(fairness_score), 2),
    }
