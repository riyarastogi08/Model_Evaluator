"""
Data drift detection using Population Stability Index (PSI) and the
Kolmogorov-Smirnov (KS) test for more robust drift assessment.
"""
import numpy as np
import pandas as pd
from scipy import stats
from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)
BUCKET_COUNT = 10


def _calculate_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = BUCKET_COUNT) -> float:
    breakpoints = np.linspace(0, 100, buckets + 1)
    bins = np.unique(np.percentile(expected, breakpoints))
    expected_counts = np.histogram(expected, bins=bins)[0].astype(float)
    actual_counts = np.histogram(actual, bins=bins)[0].astype(float)
    eps = 1e-4
    ep = (expected_counts + eps) / (expected_counts.sum() + eps * len(expected_counts))
    ap = (actual_counts + eps) / (actual_counts.sum() + eps * len(actual_counts))
    return float(np.sum((ap - ep) * np.log(ap / ep)))


def _calculate_ks(expected: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    stat, p_value = stats.ks_2samp(expected, actual)
    return float(stat), float(p_value)


def detect_drift(X: pd.DataFrame) -> dict:
    """
    Split dataset in half (reference vs current) and compute per-feature
    PSI + KS test. A feature is flagged only when BOTH signals agree,
    reducing false positives on small datasets.
    """
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return {"message": "No numeric features to evaluate for drift.", "severity": "info"}

    mid = len(X) // 2
    reference, current = X.iloc[:mid], X.iloc[mid:]

    feature_psi: dict[str, float] = {}
    feature_ks: dict[str, dict] = {}
    high_drift_features: list[str] = []

    for col in numeric_cols:
        ref_vals = reference[col].dropna().values
        cur_vals = current[col].dropna().values
        if len(ref_vals) < 5 or len(cur_vals) < 5:
            continue

        psi = _calculate_psi(ref_vals, cur_vals)
        feature_psi[col] = round(psi, 4)

        ks_stat, ks_pval = _calculate_ks(ref_vals, cur_vals)
        feature_ks[col] = {
            "statistic": round(ks_stat, 4),
            "p_value": round(ks_pval, 4),
            "drifted": bool(ks_pval < settings.KS_DRIFT_THRESHOLD),
        }

        # Only flag when both PSI and KS agree — fewer false positives
        if psi > settings.PSI_DRIFT_THRESHOLD and ks_pval < settings.KS_DRIFT_THRESHOLD:
            high_drift_features.append(col)

    # Severity based on % of features drifted, not raw count
    if not feature_psi:
        severity = "info"
    else:
        drift_pct = len(high_drift_features) / len(feature_psi)
        severity = "high" if drift_pct >= 0.5 else ("moderate" if drift_pct >= 0.2 else "low")

    logger.info(
        "Drift detection complete — severity=%s, drifted=%d/%d features",
        severity, len(high_drift_features), len(feature_psi),
    )
    return {
        "feature_psi": feature_psi,
        "feature_ks": feature_ks,
        "high_drift_features": high_drift_features,
        "severity": severity,
        "threshold": settings.PSI_DRIFT_THRESHOLD,
        "ks_threshold": settings.KS_DRIFT_THRESHOLD,
    }
