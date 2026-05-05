"""
Structured logging configuration with rotating file handler + console output.
All modules should import `get_logger(__name__)` instead of using print().
"""
import logging
import sys
from logging.handlers import RotatingFileHandler
from app.core.config import settings


def get_logger(name: str) -> logging.Logger:
    """
    Returns a named logger with consistent formatting.
    Handlers are attached only once (to the root 'app' logger) to avoid duplicates.
    """
    logger = logging.getLogger(name)

    # Only configure the root 'app' logger once
    root = logging.getLogger("app")
    if not root.handlers:
        root.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console handler
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(formatter)
        root.addHandler(console)

        # Rotating file handler (5 MB per file, keep 3 backups)
        try:
            file_handler = RotatingFileHandler(
                settings.LOG_FILE,
                maxBytes=5 * 1024 * 1024,
                backupCount=3,
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)
        except (OSError, PermissionError):
            root.warning("Could not create log file; logging to console only.")

    return logger
