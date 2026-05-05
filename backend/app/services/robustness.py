"""
Robustness evaluation — noise injection, adversarial shift, cross-validation
stability, feature ablation, boundary stress testing, and confidence analysis.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import cross_val_score
from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)


def add_gaussian_noise(df: pd.DataFrame, noise_level: float | None = None) -> pd.DataFrame:
    noise_level = noise_level or settings.NOISE_LEVEL
    noisy_df = df.copy()
    for col in noisy_df.select_dtypes(include=[np.number]).columns:
        std = noisy_df[col].std()
        if std > 0:
            noise = np.random.normal(0, std * noise_level, size=len(noisy_df))
            noisy_df[col] += noise
    return noisy_df


def simulate_adversarial_feature_shift(df: pd.DataFrame) -> pd.DataFrame:
    """Shift numeric features across their distribution means to stress decision boundaries."""
    adv_df = df.copy()
    for col in adv_df.select_dtypes(include=[np.number]).columns:
        std = adv_df[col].std()
        mean = adv_df[col].mean()
        if std > 0:
            adv_df[col] = adv_df[col].apply(lambda x: x - (1.5 * std) if x >= mean else x + (1.5 * std))
    return adv_df


def _clip_to_boundaries(df: pd.DataFrame) -> pd.DataFrame:
    """Clip every numeric feature to its min or max (boundary stress)."""
    boundary_df = df.copy()
    for col in boundary_df.select_dtypes(include=[np.number]).columns:
        median = boundary_df[col].median()
        col_min, col_max = boundary_df[col].min(), boundary_df[col].max()
        boundary_df[col] = boundary_df[col].apply(lambda x: col_min if x < median else col_max)
    return boundary_df


def _score(y_true, y_pred, task_type: str) -> float:
    if task_type.lower() == "classification":
        return accuracy_score(y_true, y_pred)
    return r2_score(y_true, y_pred)


def _confidence_analysis(model, X: pd.DataFrame) -> dict:
    """
    For classifiers with predict_proba: compute mean confidence and the
    fraction of low-confidence predictions (< 0.6 max class probability).
    Helps flag models that are uncertain even on unperturbed data.
    """
    if not hasattr(model, "predict_proba"):
        return {"available": False}
    try:
        proba = model.predict_proba(X)
        max_proba = proba.max(axis=1)
        low_conf_threshold = 0.6
        low_conf_fraction = float(np.mean(max_proba < low_conf_threshold))
        return {
            "available": True,
            "mean_confidence": round(float(max_proba.mean()), 4),
            "min_confidence": round(float(max_proba.min()), 4),
            "low_confidence_fraction": round(low_conf_fraction, 4),
            "low_confidence_threshold": low_conf_threshold,
        }
    except Exception as exc:
        logger.warning("Confidence analysis failed: %s", exc)
        return {"available": False, "error": str(exc)}


def evaluate_robustness(model, X_test: pd.DataFrame, y_test: pd.Series, task_type: str) -> dict:
    logger.info("Starting robustness evaluation (task=%s, samples=%d)", task_type, len(X_test))

    # 1. Baseline
    y_pred_base = model.predict(X_test)
    base_score = _score(y_test, y_pred_base, task_type)

    # 2. Noise robustness
    X_noisy = add_gaussian_noise(X_test)
    noise_score = _score(y_test, model.predict(X_noisy), task_type)
    accuracy_drop = base_score - noise_score

    # 3. Adversarial shift
    X_adv = simulate_adversarial_feature_shift(X_test)
    adv_score = _score(y_test, model.predict(X_adv), task_type)
    adv_drop = base_score - adv_score

    # 4. Boundary stress
    X_boundary = _clip_to_boundaries(X_test)
    boundary_score = _score(y_test, model.predict(X_boundary), task_type)
    boundary_drop = base_score - boundary_score

    # 5. Cross-validation stability
    cv_details = _cross_validation_stability(model, X_test, y_test, task_type)

    # 6. Feature ablation
    ablation = _feature_ablation(model, X_test, y_test, task_type, base_score)

    # 7. Confidence analysis (classifiers only)
    confidence = {}
    if task_type.lower() == "classification":
        confidence = _confidence_analysis(model, X_test)

    def pct(perturbed, baseline):
        if baseline > 0 and task_type.lower() == "classification":
            return (perturbed / baseline) * 100
        if baseline != 0:
            return max(0, 100 - (abs(baseline - perturbed) / abs(baseline)) * 100)
        return 0.0

    stability = pct(noise_score, base_score)
    adv_resilience = pct(adv_score, base_score)
    boundary_resilience = pct(boundary_score, base_score)

    result = {
        "baseline_score": round(float(base_score), 4),
        "noisy_score": round(float(noise_score), 4),
        "adversarial_score": round(float(adv_score), 4),
        "boundary_score": round(float(boundary_score), 4),
        "accuracy_drop": round(float(accuracy_drop), 4),
        "adversarial_drop": round(float(adv_drop), 4),
        "boundary_drop": round(float(boundary_drop), 4),
        "stability_score": round(float(stability), 2),
        "adversarial_resilience": round(float(adv_resilience), 2),
        "boundary_resilience": round(float(boundary_resilience), 2),
        "cv_stability": cv_details,
        "feature_ablation": ablation,
        "confidence_analysis": confidence,
    }

    logger.info(
        "Robustness results — stability=%.1f%%, adv_resilience=%.1f%%, boundary_resilience=%.1f%%",
        stability, adv_resilience, boundary_resilience,
    )
    return result


def _cross_validation_stability(model, X, y, task_type: str) -> dict:
    """Run k-fold CV and report mean, std, and per-fold scores."""
    try:
        scoring = "accuracy" if task_type.lower() == "classification" else "r2"
        scores = cross_val_score(model, X, y, cv=settings.CV_FOLDS, scoring=scoring)
        return {
            "mean": round(float(scores.mean()), 4),
            "std": round(float(scores.std()), 4),
            "fold_scores": [round(float(s), 4) for s in scores],
            "folds": settings.CV_FOLDS,
        }
    except Exception as exc:
        logger.warning("Cross-validation failed: %s", exc)
        return {"error": str(exc)}


def _feature_ablation(model, X, y, task_type: str, baseline: float) -> dict:
    """Drop each feature one-by-one and measure the performance impact."""
    try:
        impacts: dict[str, float] = {}
        for col in X.columns:
            X_dropped = X.drop(columns=[col])
            try:
                pred = model.predict(X_dropped)
                score = _score(y, pred, task_type)
            except Exception:
                # Model requires all original features — use mean-imputation fallback
                X_imputed = X.copy()
                X_imputed[col] = X[col].mean()
                pred = model.predict(X_imputed)
                score = _score(y, pred, task_type)
            impacts[col] = round(float(baseline - score), 4)

        sorted_impacts = dict(sorted(impacts.items(), key=lambda kv: abs(kv[1]), reverse=True))
        return {"feature_impact": sorted_impacts}
    except Exception as exc:
        logger.warning("Feature ablation failed: %s", exc)
        return {"error": str(exc)}
