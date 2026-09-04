"""
Centralized logging configuration.
"""

import logging
import logging.config
from pathlib import Path

import yaml

from src.core.config import settings


def setup_logging(config_path: str = "configs/logging_config.yaml") -> None:
    """Initialize logging from a YAML configuration file."""
    path = Path(config_path)
    if path.exists():
        with open(path, "rt") as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=settings.log_level)
        logging.warning(
            f"Logging configuration file {config_path} not found. "
            "Using basic configuration."
        )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    logger = logging.getLogger(name)
    logger.setLevel(settings.log_level)
    return logger
