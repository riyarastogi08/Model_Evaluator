"""
Explainability module — SHAP-based global feature importance.
"""
import shap
import pandas as pd
import numpy as np
from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)


def generate_shap_importance(model, X: pd.DataFrame) -> dict:
    """
    Compute global feature importance using SHAP.
    Returns a dict mapping column name → mean |SHAP| value, sorted descending.
    """
    try:
        sample_size = min(settings.SHAP_SAMPLE_SIZE, len(X))
        X_sample = shap.sample(X, sample_size) if len(X) > sample_size else X

        # Auto-detect best explainer
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            logger.info("Using TreeExplainer for SHAP analysis")
        except Exception:
            explainer = shap.Explainer(model.predict, X_sample)
            shap_values = explainer(X_sample).values
            logger.info("Using generic Explainer for SHAP analysis (TreeExplainer not supported)")

        # Extract mean absolute SHAP value per feature
        if isinstance(shap_values, list):
            # Multiclass: list of arrays
            vals = np.abs(np.array(shap_values)).mean(axis=(0, 1))
        elif len(shap_values.shape) == 3:
            vals = np.abs(shap_values).mean(axis=(0, 2))
        else:
            vals = np.abs(shap_values).mean(axis=0)

        importance_dict = {col: round(float(v), 6) for col, v in zip(X.columns, vals)}
        # Sort by importance descending, top 15 features
        sorted_importance = dict(
            sorted(importance_dict.items(), key=lambda item: item[1], reverse=True)[:15]
        )

        logger.info("SHAP importance computed for %d features", len(sorted_importance))
        return sorted_importance

    except Exception as exc:
        logger.error("SHAP generation failed: %s", exc, exc_info=True)
        return {"error": f"SHAP generation failed: {str(exc)}"}
