"""
Main evaluation pipeline orchestrator.
Each step is isolated so a failure in one does not crash the entire pipeline.
Uploaded files are cleaned up in the finally block.
"""
import os
import time
from datetime import datetime, timezone
from sqlalchemy.orm import Session

from app.core.database import SessionLocal
from app.core.logging_config import get_logger
from app.services.ml_loaders import load_model, load_data, validate_target_column, preprocess_data
from app.services.robustness import evaluate_robustness
from app.services.fairness import evaluate_fairness
from app.services.explainability import generate_shap_importance
from app.services.drift_detection import detect_drift
from app.services.rice_engine import calculate_rice_priority
from app.services.llm_reporter import generate_audit_report
from app.models.eval_report import EvaluationTask

logger = get_logger(__name__)


def run_evaluation_pipeline(
    task_id: int,
    model_path: str,
    dataset_path: str,
    task_type: str,
    target_column: str,
    sensitive_attr: str,
):
    db: Session = SessionLocal()
    start_time = time.time()
    task = None

    try:
        task = db.query(EvaluationTask).filter(EvaluationTask.id == task_id).first()
        if not task:
            logger.error("Task %d not found — aborting pipeline", task_id)
            return

        task.status = "evaluating"
        task.updated_at = datetime.now(timezone.utc)
        db.commit()
        logger.info("Pipeline started for task %d (model=%s)", task_id, task.model_name)

        # ── 1. Load & Preprocess ──
        df = load_data(dataset_path)
        model = load_model(model_path)

        resolved_target = validate_target_column(df, target_column)
        task.target_column = resolved_target
        X, y = preprocess_data(df, resolved_target)

        # ── 2. Robustness ──
        robustness_metrics = {}
        try:
            robustness_metrics = evaluate_robustness(model, X, y, task_type)
            task.robustness_score = robustness_metrics.get("stability_score")
            task.robustness_details = robustness_metrics
            logger.info("Task %d — Robustness complete", task_id)
        except Exception as exc:
            logger.error("Task %d — Robustness failed: %s", task_id, exc, exc_info=True)
            task.robustness_details = {"error": str(exc)}

        # ── 3. Fairness ──
        fairness_metrics = {}
        try:
            if sensitive_attr and sensitive_attr in X.columns:
                fairness_metrics = evaluate_fairness(model, X, y, sensitive_attr, task_type)
                task.fairness_score = fairness_metrics.get("fairness_score")
                task.fairness_details = fairness_metrics
                logger.info("Task %d — Fairness complete", task_id)
            else:
                task.fairness_details = {"message": "Sensitive attribute missing or not in dataset."}
                logger.warning("Task %d — Sensitive attribute '%s' not found, skipping fairness", task_id, sensitive_attr)
        except Exception as exc:
            logger.error("Task %d — Fairness failed: %s", task_id, exc, exc_info=True)
            task.fairness_details = {"error": str(exc)}

        # ── 4. SHAP Explainability ──
        shap_importance = {}
        try:
            shap_importance = generate_shap_importance(model, X)
            task.explainability_details = shap_importance
            logger.info("Task %d — SHAP complete", task_id)
        except Exception as exc:
            logger.error("Task %d — SHAP failed: %s", task_id, exc, exc_info=True)
            task.explainability_details = {"error": str(exc)}

        # ── 5. Drift Detection ──
        drift_details = {}
        try:
            drift_details = detect_drift(X)
            task.data_drift_details = drift_details
            logger.info("Task %d — Drift complete (severity=%s)", task_id, drift_details.get("severity"))
        except Exception as exc:
            logger.error("Task %d — Drift failed: %s", task_id, exc, exc_info=True)
            task.data_drift_details = {"error": str(exc)}

        # ── 6. RICE Engine ──
        rice_items = []
        try:
            rice_items = calculate_rice_priority(
                robustness_metrics,
                fairness_metrics,
                drift_details,
                shap_importance=shap_importance if "error" not in shap_importance else None,
                sensitive_attr=sensitive_attr or None,
            )
            task.rice_priority_table = rice_items
            logger.info("Task %d — RICE complete (%d items)", task_id, len(rice_items))
        except Exception as exc:
            logger.error("Task %d — RICE failed: %s", task_id, exc, exc_info=True)
            task.rice_priority_table = []

        # ── 7. Audit Report ──
        try:
            llm_report = generate_audit_report(task_type, robustness_metrics, fairness_metrics, rice_items, drift_details)
            if task.robustness_details and isinstance(task.robustness_details, dict):
                details = dict(task.robustness_details)
                details["llm_audit_report"] = llm_report
                task.robustness_details = details
            logger.info("Task %d — Report generated", task_id)
        except Exception as exc:
            logger.error("Task %d — Report failed: %s", task_id, exc, exc_info=True)

        task.status = "completed"
        task.duration_seconds = round(time.time() - start_time, 2)
        task.updated_at = datetime.now(timezone.utc)
        db.commit()
        logger.info("Task %d completed in %.2fs", task_id, task.duration_seconds)

    except Exception as exc:
        logger.error("Task %d — Pipeline-level failure: %s", task_id, exc, exc_info=True)
        if task:
            task.status = "failed"
            task.duration_seconds = round(time.time() - start_time, 2)
            task.robustness_details = {"error": str(exc)}
            task.updated_at = datetime.now(timezone.utc)
            db.commit()
    finally:
        db.close()
        _cleanup_files(model_path, dataset_path)


def _cleanup_files(*paths):
    """Remove temporary upload files after evaluation."""
    for path in paths:
        try:
            if path and os.path.isfile(path):
                os.remove(path)
                logger.debug("Cleaned up upload: %s", path)
        except OSError as exc:
            logger.warning("Could not remove file %s: %s", path, exc)
