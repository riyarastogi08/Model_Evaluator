"""
RICE Priority Engine — evaluates robustness, fairness, drift, explainability,
and confidence findings to produce actionable, severity-graded priority items.

RICE Score = (Reach × Impact × Confidence) / Effort
"""
from app.core.logging_config import get_logger

logger = get_logger(__name__)


def calculate_rice_priority(
    robustness_metrics: dict,
    fairness_metrics: dict,
    drift_details: dict | None = None,
    shap_importance: dict | None = None,
    sensitive_attr: str | None = None,
) -> list[dict]:
    """Return a sorted list of RICE-scored issues with severity and remediation."""
    issues: list[dict] = []

    # ── 1. Noise Stability ──
    stability = robustness_metrics.get("stability_score", 100)
    if stability < 90:
        reach, impact = 10, (9 if stability < 80 else 6)
        confidence, effort = 8, 7
        issues.append(_issue(
            title="Model Highly Sensitive to Noise",
            description=f"Accuracy drops significantly on noisy inputs (Stability: {stability:.1f}%).",
            severity="critical" if stability < 75 else "warning",
            remediation="Apply data augmentation with Gaussian noise during training; consider regularization (L2, dropout).",
            reach=reach, impact=impact, confidence=confidence, effort=effort,
        ))

    # ── 2. Adversarial Resilience ──
    adv_resilience = robustness_metrics.get("adversarial_resilience", 100)
    if adv_resilience < 85:
        reach, impact = 8, (10 if adv_resilience < 70 else 7)
        confidence, effort = 8, 8
        issues.append(_issue(
            title="Low Adversarial Resilience",
            description=f"Model performance degrades heavily under adversarial feature shift (Resilience: {adv_resilience:.1f}%).",
            severity="critical" if adv_resilience < 70 else "warning",
            remediation="Implement adversarial training (PGD); consider ensemble methods to flatten decision boundaries.",
            reach=reach, impact=impact, confidence=confidence, effort=effort,
        ))

    # ── 3. Boundary Stress ──
    boundary_resilience = robustness_metrics.get("boundary_resilience", 100)
    if boundary_resilience < 85:
        reach, impact = 7, 6
        confidence, effort = 7, 5
        issues.append(_issue(
            title="Boundary Stress Vulnerability",
            description=f"Model struggles when features are pushed to extreme values (Resilience: {boundary_resilience:.1f}%).",
            severity="warning",
            remediation="Add boundary-condition samples to training; clip feature values during preprocessing.",
            reach=reach, impact=impact, confidence=confidence, effort=effort,
        ))

    # ── 4. Cross-Validation Instability ──
    cv = robustness_metrics.get("cv_stability", {})
    cv_std = cv.get("std", 0)
    if cv_std > 0.05:
        reach, impact = 8, 7
        confidence, effort = 9, 6
        issues.append(_issue(
            title="Cross-Validation Instability",
            description=f"High variance across CV folds (std={cv_std:.4f}). Model may overfit specific data slices.",
            severity="warning" if cv_std < 0.1 else "critical",
            remediation="Increase training data diversity; try stratified sampling; consider simpler model architectures.",
            reach=reach, impact=impact, confidence=confidence, effort=effort,
        ))

    # ── 5. Low Prediction Confidence ──
    conf = robustness_metrics.get("confidence_analysis", {})
    if conf.get("available") and conf.get("low_confidence_fraction", 0) > 0.3:
        low_frac = conf["low_confidence_fraction"]
        issues.append(_issue(
            title="Low Prediction Confidence",
            description=(
                f"{low_frac*100:.1f}% of predictions have confidence below "
                f"{conf.get('low_confidence_threshold', 0.6)*100:.0f}%. "
                f"Mean confidence: {conf.get('mean_confidence', 0)*100:.1f}%."
            ),
            severity="critical" if low_frac > 0.5 else "warning",
            remediation=(
                "Calibrate model probabilities using Platt scaling or isotonic regression; "
                "review class imbalance in training data; consider collecting more training samples."
            ),
            reach=7, impact=8, confidence=8, effort=5,
        ))

    # ── 6. Demographic Parity Bias ──
    dpd = fairness_metrics.get("demographic_parity_diff", 0)
    if abs(dpd) > 0.1:
        reach, impact = 5, 10
        confidence, effort = 9, 6
        issues.append(_issue(
            title="Demographic Parity Bias Detected",
            description=f"Model predictions depend heavily on the sensitive attribute (Disparity: {abs(dpd):.3f}).",
            severity="critical" if abs(dpd) > 0.2 else "warning",
            remediation="Use Fairlearn's ExponentiatedGradient or ThresholdOptimizer; re-sample training data for demographic balance.",
            reach=reach, impact=impact, confidence=confidence, effort=effort,
        ))

    # ── 7. Equalized Odds Violation ──
    eod = fairness_metrics.get("equalized_odds_diff", 0)
    if abs(eod) > 0.1:
        reach, impact = 5, 8
        confidence, effort = 8, 7
        issues.append(_issue(
            title="Equalized Odds Violation",
            description=f"Error rates differ significantly across sensitive groups (EOD: {abs(eod):.3f}).",
            severity="warning",
            remediation="Apply equalized odds post-processing; review feature engineering for proxy variables.",
            reach=reach, impact=impact, confidence=confidence, effort=effort,
        ))

    # ── 8. Sensitive Attribute as Top SHAP Feature ──
    if shap_importance and sensitive_attr:
        top_feature = next(iter(shap_importance), None)
        if top_feature and top_feature.lower() == sensitive_attr.lower():
            issues.append(_issue(
                title="Sensitive Attribute Is Top Predictive Feature",
                description=(
                    f"'{sensitive_attr}' is the highest-importance feature by SHAP value. "
                    "The model may be making decisions primarily based on the protected attribute."
                ),
                severity="critical",
                remediation=(
                    "Remove or constrain the sensitive attribute; apply fairness-aware training; "
                    "investigate whether proxy features carry similar information."
                ),
                reach=6, impact=10, confidence=9, effort=7,
            ))

    # ── 9. Data Drift ──
    if drift_details and drift_details.get("severity") in ("moderate", "high"):
        drifted = drift_details.get("high_drift_features", [])
        severity_level = drift_details["severity"]
        reach, impact = 9, (9 if severity_level == "high" else 6)
        confidence, effort = 7, 5
        issues.append(_issue(
            title="Data Distribution Drift Detected",
            description=f"Significant drift in {len(drifted)} feature(s): {', '.join(drifted[:5])}.",
            severity="critical" if severity_level == "high" else "warning",
            remediation="Retrain the model on recent data; investigate root cause of distribution shift; add monitoring alerts.",
            reach=reach, impact=impact, confidence=confidence, effort=effort,
        ))

    # ── Fallback ──
    if not issues:
        issues.append({
            "title": "No Critical Issues Detected",
            "description": "Model passes baseline robustness and fairness thresholds.",
            "severity": "info",
            "remediation": "Continue monitoring; schedule periodic re-evaluations.",
            "reach": 0, "impact": 0, "confidence": 0, "effort": 1, "score": 0,
        })

    result = sorted(issues, key=lambda i: i["score"], reverse=True)
    logger.info("RICE engine produced %d issue(s)", len(result))
    return result


def _issue(*, title, description, severity, remediation, reach, impact, confidence, effort) -> dict:
    score = round((reach * impact * confidence) / effort, 1)
    return {
        "title": title,
        "description": description,
        "severity": severity,
        "remediation": remediation,
        "reach": reach,
        "impact": impact,
        "confidence": confidence,
        "effort": effort,
        "score": score,
    }

