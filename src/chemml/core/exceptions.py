"""
ChemML Custom Exceptions

This module defines a hierarchy of custom exceptions for ChemML operations.
All ChemML-specific exceptions should inherit from ChemMLError.
"""

from typing import Optional


class ChemMLError(Exception):
    """Base exception for all ChemML operations."""

    def __init__(self, message: str, details: Optional[dict] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class ChemMLDataError(ChemMLError):
    """Raised when data validation or processing fails."""


class ChemMLModelError(ChemMLError):
    """Raised when model operations fail."""


class ChemMLConfigurationError(ChemMLError):
    """Raised when configuration issues occur."""


class ChemMLDependencyError(ChemMLError):
    """Raised when required dependencies are missing."""


class ChemMLFeaturizationError(ChemMLError):
    """Raised when molecular featurization fails."""


class ChemMLQuantumError(ChemMLError):
    """Raised when quantum computing operations fail."""


class ChemMLVisualizationError(ChemMLError):
    """Raised when visualization operations fail."""


class ChemMLIntegrationError(ChemMLError):
    """Raised when external library integration fails."""


class ChemMLValidationError(ChemMLError):
    """Raised when input validation fails."""


# Convenience functions for common error patterns
def raise_data_error(message: str, data_info: Optional[dict] = None) -> None:
    """Raise a ChemMLDataError with optional data information."""
    raise ChemMLDataError(message, data_info)


def raise_model_error(message: str, model_info: Optional[dict] = None) -> None:
    """Raise a ChemMLModelError with optional model information."""
    raise ChemMLModelError(message, model_info)


def raise_dependency_error(package: str, operation: str = "") -> None:
    """Raise a ChemMLDependencyError for missing packages."""
    message = f"Required package '{package}' is not available"
    if operation:
        message += f" for {operation}"
    details = {"missing_package": package, "operation": operation}
    raise ChemMLDependencyError(message, details)


def raise_config_error(config_key: str, expected_type: str = "") -> None:
    """Raise a ChemMLConfigurationError for configuration issues."""
    message = f"Configuration error for key '{config_key}'"
    if expected_type:
        message += f" (expected {expected_type})"
    details = {"config_key": config_key, "expected_type": expected_type}
    raise ChemMLConfigurationError(message, details)
