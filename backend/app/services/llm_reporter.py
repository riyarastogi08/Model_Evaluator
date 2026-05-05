"""
AI Audit Report Generator — produces a comprehensive, dynamic plain-English
report with a composite health grade (A–F) and per-finding mitigations.
"""
from app.core.logging_config import get_logger

logger = get_logger(__name__)


def _compute_health_grade(robustness_details: dict, fairness_details: dict, drift_details: dict | None) -> tuple[str, float]:
    """
    Compute a composite score (0–100) and letter grade based on all evaluation pillars.
    Weights: Robustness 40%, Fairness 30%, Stability 20%, Drift 10%
    """
    scores = []
    weights = []

    # Robustness
    stability = robustness_details.get("stability_score", 0)
    adv = robustness_details.get("adversarial_resilience", 0)
    robustness_composite = (stability * 0.6 + adv * 0.4)
    scores.append(robustness_composite)
    weights.append(0.40)

    # Fairness
    fairness_score = fairness_details.get("fairness_score")
    if fairness_score is not None:
        scores.append(fairness_score)
        weights.append(0.30)

    # CV Stability
    cv = robustness_details.get("cv_stability", {})
    cv_mean = cv.get("mean", 0)
    cv_std = cv.get("std", 0)
    if cv_mean > 0:
        cv_score = min(100, cv_mean * 100 * (1 - cv_std))
        scores.append(cv_score)
        weights.append(0.20)

    # Drift
    if drift_details:
        severity = drift_details.get("severity", "low")
        drift_score = {"low": 95, "moderate": 60, "high": 25, "info": 100}.get(severity, 80)
        scores.append(drift_score)
        weights.append(0.10)

    # Normalize weights
    total_w = sum(weights)
    composite = sum(s * w for s, w in zip(scores, weights)) / total_w if total_w > 0 else 0

    if composite >= 90:
        grade = "A"
    elif composite >= 80:
        grade = "B"
    elif composite >= 65:
        grade = "C"
    elif composite >= 50:
        grade = "D"
    else:
        grade = "F"

    return grade, round(composite, 1)


def generate_audit_report(
    task_type: str,
    robustness_details: dict,
    fairness_details: dict,
    rice_issues: list[dict],
    drift_details: dict | None = None,
) -> str:
    """Generate a comprehensive plain-English AI audit report."""
    grade, composite = _compute_health_grade(robustness_details, fairness_details, drift_details)

    lines = [
        "### Automated AI Audit Report",
        "",
        f"**Model Task Type:** {task_type}",
        f"**Overall Health Grade: {grade}** (Composite Score: {composite}/100)",
        "",
    ]

    # ── Executive Summary ──
    lines.append("---")
    lines.append("#### Executive Summary")
    critical_issues = [i for i in rice_issues if i.get("severity") == "critical"]
    warning_issues = [i for i in rice_issues if i.get("severity") == "warning"]

    if critical_issues:
        lines.append(
            f"The evaluation identified **{len(critical_issues)} critical** and "
            f"**{len(warning_issues)} warning-level** finding(s). Immediate action is recommended."
        )
    elif warning_issues:
        lines.append(
            f"The evaluation identified **{len(warning_issues)} warning-level** finding(s). "
            "The model is functional but could benefit from targeted improvements."
        )
    else:
        lines.append(
            "The model is demonstrating strong baseline performance with no critical "
            "robustness or fairness violations detected."
        )
    lines.append("")

    # ── Robustness Summary ──
    lines.append("---")
    lines.append("#### Robustness Analysis")
    lines.append(f"- **Baseline Score:** {robustness_details.get('baseline_score', 'N/A')}")
    lines.append(f"- **Noise Stability:** {robustness_details.get('stability_score', 'N/A')}%")
    lines.append(f"- **Adversarial Resilience:** {robustness_details.get('adversarial_resilience', 'N/A')}%")
    lines.append(f"- **Boundary Resilience:** {robustness_details.get('boundary_resilience', 'N/A')}%")
    cv = robustness_details.get("cv_stability", {})
    if cv and "mean" in cv:
        lines.append(f"- **CV Mean ± Std:** {cv['mean']:.4f} ± {cv['std']:.4f} ({cv.get('folds', '?')}-fold)")
    lines.append("")

    # ── Fairness Summary ──
    lines.append("---")
    lines.append("#### Fairness Analysis")
    fs = fairness_details.get("fairness_score")
    if fs is not None:
        lines.append(f"- **Fairness Score:** {fs}%")
        dpd = fairness_details.get("demographic_parity_diff")
        if dpd is not None:
            lines.append(f"- **Demographic Parity Diff:** {dpd:.4f}")
        eod = fairness_details.get("equalized_odds_diff")
        if eod is not None:
            lines.append(f"- **Equalized Odds Diff:** {eod:.4f}")
        group_rates = fairness_details.get("group_positive_rates", {})
        if group_rates:
            lines.append("- **Per-Group Positive Rates:**")
            for g, r in group_rates.items():
                lines.append(f"  - Group {g}: {r:.4f}")
    else:
        lines.append("- Fairness analysis was not applicable or was skipped.")
    lines.append("")

    # ── Drift Summary ──
    if drift_details and drift_details.get("feature_psi"):
        lines.append("---")
        lines.append("#### Data Drift Analysis")
        lines.append(f"- **Overall Severity:** {drift_details.get('severity', 'N/A').upper()}")
        high = drift_details.get("high_drift_features", [])
        if high:
            lines.append(f"- **High-Drift Features ({len(high)}):** {', '.join(high)}")
        else:
            lines.append("- No features exceed the PSI drift threshold.")
        lines.append("")

    # ── RICE Findings ──
    if rice_issues and rice_issues[0].get("score", 0) > 0:
        lines.append("---")
        lines.append("#### Priority Action Items (RICE)")
        for idx, issue in enumerate(rice_issues, 1):
            sev_emoji = {"critical": "🔴", "warning": "🟡", "info": "🟢"}.get(issue.get("severity"), "⚪")
            lines.append(f"**{idx}. {sev_emoji} {issue['title']}** (RICE: {issue['score']}, Severity: {issue.get('severity', 'N/A')})")
            lines.append(f"   {issue['description']}")
            if issue.get("remediation"):
                lines.append(f"   *Remediation:* {issue['remediation']}")
            lines.append("")

    # ── Generic Advice ──
    lines.append("---")
    lines.append("#### General Mitigation Strategies")
    lines.append("- **Data Augmentation:** Inject noise (Gaussian perturbations for tabular, blur for images) into training batches.")
    lines.append("- **Adversarial Training:** Use robust optimization techniques like PGD training.")
    lines.append("- **Fairness Interventions:** Use Fairlearn's ExponentiatedGradient or ThresholdOptimizer.")
    lines.append("- **Monitoring:** Schedule periodic re-evaluations and set up data drift alerts.")
    lines.append("")

    report = "\n".join(lines)
    logger.info("Audit report generated — grade=%s, composite=%.1f", grade, composite)
    return report
