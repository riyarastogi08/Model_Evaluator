"""
Centralized application configuration using Pydantic Settings.
All hard-coded values are consolidated here and can be overridden via
environment variables or a .env file.
"""
import os
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # ── Application ──
    APP_TITLE: str = "Automated Model Robustness & Fairness Evaluator"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False

    # ── Database ──
    DATABASE_URL: str = "sqlite:///./evaluator.db"

    # ── File Upload ──
    UPLOAD_DIR: str = "uploads"
    MAX_FILE_SIZE_MB: int = 200
    # Only .pkl is fully supported — do not add .h5/.pt unless loaders are implemented
    ALLOWED_MODEL_EXTENSIONS: List[str] = [".pkl"]
    ALLOWED_DATASET_EXTENSIONS: List[str] = [".csv", ".json"]

    # ── CORS ──
    # In production, replace "*" with your actual domain, e.g. ["https://yourdomain.com"]
    CORS_ORIGINS: List[str] = ["*"]

    # ── Evaluation Defaults ──
    SHAP_SAMPLE_SIZE: int = 100
    CV_FOLDS: int = 5
    NOISE_LEVEL: float = 0.1
    PSI_DRIFT_THRESHOLD: float = 0.2
    KS_DRIFT_THRESHOLD: float = 0.05   # p-value threshold for KS test

    # ── Logging ──
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "app.log"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


settings = Settings()

# Ensure the upload directory exists at import time
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
