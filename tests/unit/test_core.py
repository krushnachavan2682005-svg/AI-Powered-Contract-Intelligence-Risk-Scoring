"""
Unit tests for the core module.
"""

import logging
import os

from src.core.config import Settings
from src.core.exceptions import ApplicationError, ConfigurationError
from src.core.logging import get_logger


def test_settings_default_load() -> None:
    """Test that settings load with default values if no env vars are set."""
    # Ensure no conflicting env vars
    os.environ.pop("APP_NAME", None)
    settings = Settings()
    assert settings.app_name == "contract-intelligence-risk"


def test_settings_env_override() -> None:
    """Test that settings can be overridden by environment variables."""
    os.environ["APP_NAME"] = "test-app"
    os.environ["ENVIRONMENT"] = "production"
    settings = Settings()
    assert settings.app_name == "test-app"
    assert settings.environment == "production"


def test_logger_helper() -> None:
    """Test that get_logger returns a valid logger."""
    logger = get_logger("test_logger")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_logger"


def test_custom_exceptions() -> None:
    """Test that custom exceptions behave correctly."""
    try:
        raise ConfigurationError("Missing config")
    except ApplicationError as e:
        assert str(e) == "Missing config"
        assert e.message == "Missing config"
