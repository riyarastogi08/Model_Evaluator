"""
Model and data loading utilities with validation, preprocessing, and clear error messages.
"""
import os
import joblib
import pandas as pd
import numpy as np
from app.core.logging_config import get_logger

logger = get_logger(__name__)

MIN_ROWS = 10
MIN_COLS = 2


def load_data(filepath: str) -> pd.DataFrame:
    """Load a dataset from CSV or JSON with validation."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    try:
        if filepath.endswith(".csv"):
            df = pd.read_csv(filepath)
        elif filepath.endswith(".json"):
            df = pd.read_json(filepath)
        else:
            raise ValueError(f"Unsupported dataset format: {os.path.splitext(filepath)[1]}")
    except (pd.errors.ParserError, ValueError) as exc:
        raise ValueError(f"Failed to parse dataset — file may be corrupt or malformed: {exc}") from exc

    # Shape validation
    if df.shape[0] < MIN_ROWS:
        raise ValueError(
            f"Dataset has only {df.shape[0]} rows; minimum {MIN_ROWS} required for meaningful evaluation."
        )
    if df.shape[1] < MIN_COLS:
        raise ValueError(
            f"Dataset has only {df.shape[1]} columns; minimum {MIN_COLS} required (features + target)."
        )

    # Drop fully empty columns
    empty_cols = df.columns[df.isnull().all()].tolist()
    if empty_cols:
        logger.warning("Dropping entirely empty columns: %s", empty_cols)
        df = df.drop(columns=empty_cols)

    logger.info("Dataset loaded: %d rows × %d columns from %s", df.shape[0], df.shape[1], filepath)
    return df


def validate_target_column(df: pd.DataFrame, target_column: str) -> str:
    """
    Resolve which column is the target.
    If target_column is specified and exists, use it.
    Otherwise fall back to the last column and log a warning.
    Returns the resolved column name.
    """
    if target_column and target_column in df.columns:
        logger.info("Using user-specified target column: '%s'", target_column)
        return target_column

    if target_column and target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in dataset. "
            f"Available columns: {list(df.columns)}"
        )

    # Fallback: last column heuristic
    fallback = df.columns[-1]
    logger.warning(
        "No target column specified — falling back to last column: '%s'. "
        "Set 'target_column' explicitly to avoid this.", fallback
    )
    return fallback


def preprocess_data(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    Clean and encode the feature matrix (X) in-place.
    - Drops duplicate rows.
    - Imputes numeric NaNs with the column median.
    - Imputes categorical NaNs with the column mode.
    - Label-encodes object/category columns so sklearn models can consume them.
    The target column is left untouched.
    """
    X = df.drop(columns=[target_column])
    original_rows = len(X)

    # Drop duplicate rows
    X = X.drop_duplicates()
    if len(X) < original_rows:
        logger.info("Dropped %d duplicate rows", original_rows - len(X))

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # Impute numeric NaNs with median
    for col in numeric_cols:
        n_missing = X[col].isna().sum()
        if n_missing > 0:
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)
            logger.info("Imputed %d NaN(s) in numeric column '%s' with median %.4f", n_missing, col, median_val)

    # Impute categorical NaNs with mode, then label-encode
    for col in cat_cols:
        n_missing = X[col].isna().sum()
        if n_missing > 0:
            mode_val = X[col].mode()[0]
            X[col] = X[col].fillna(mode_val)
            logger.info("Imputed %d NaN(s) in categorical column '%s' with mode '%s'", n_missing, col, mode_val)
        # Label encode: convert categories to integer codes
        X[col] = X[col].astype("category").cat.codes
        logger.info("Label-encoded categorical column '%s'", col)

    # Re-align with original df index (after dedup)
    target_aligned = df.loc[X.index, target_column]
    logger.info(
        "Preprocessing complete — %d rows, %d features (%d numeric, %d categorical encoded)",
        len(X), len(X.columns), len(numeric_cols), len(cat_cols),
    )
    return X, target_aligned


def load_model(filepath: str):
    """Load a scikit-learn .pkl model with validation."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")

    try:
        if filepath.endswith(".pkl"):
            model = joblib.load(filepath)
        else:
            raise ValueError(
                f"Unsupported model format: {os.path.splitext(filepath)[1]}. "
                "Only scikit-learn .pkl models are supported currently."
            )
    except Exception as exc:
        raise ValueError(f"Failed to load model — file may be corrupt: {exc}") from exc

    # Basic sanity check: model should have a predict method
    if not hasattr(model, "predict"):
        raise ValueError("Loaded object does not have a .predict() method — not a valid ML model.")

    logger.info("Model loaded successfully from %s", filepath)
    return model
