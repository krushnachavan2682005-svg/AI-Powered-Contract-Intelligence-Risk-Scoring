"""
Base exception hierarchy for the application.
"""


class ApplicationError(Exception):
    """Base exception for all custom application errors."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ConfigurationError(ApplicationError):
    """Exception raised for configuration or environment errors."""

    pass
