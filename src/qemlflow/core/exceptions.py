"""
QeMLflow Custom Exceptions

This module defines a hierarchy of custom exceptions for QeMLflow operations.
All QeMLflow-specific exceptions should inherit from QeMLflowError.
"""

from typing import Optional


class QeMLflowError(Exception):
    """Base exception for all QeMLflow operations."""

    def __init__(self, message: str, details: Optional[dict] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class QeMLflowDataError(QeMLflowError):
    """Raised when data validation or processing fails."""


class QeMLflowModelError(QeMLflowError):
    """Raised when model operations fail."""


class QeMLflowConfigurationError(QeMLflowError):
    """Raised when configuration issues occur."""


class QeMLflowDependencyError(QeMLflowError):
    """Raised when required dependencies are missing."""


class QeMLflowFeaturizationError(QeMLflowError):
    """Raised when molecular featurization fails."""


class QeMLflowQuantumError(QeMLflowError):
    """Raised when quantum computing operations fail."""


class QeMLflowVisualizationError(QeMLflowError):
    """Raised when visualization operations fail."""


class QeMLflowIntegrationError(QeMLflowError):
    """Raised when external library integration fails."""


class QeMLflowValidationError(QeMLflowError):
    """Raised when input validation fails."""


# Convenience functions for common error patterns
def raise_data_error(message: str, data_info: Optional[dict] = None) -> None:
    """Raise a QeMLflowDataError with optional data information."""
    raise QeMLflowDataError(message, data_info)


def raise_model_error(message: str, model_info: Optional[dict] = None) -> None:
    """Raise a QeMLflowModelError with optional model information."""
    raise QeMLflowModelError(message, model_info)


def raise_dependency_error(package: str, operation: str = "") -> None:
    """Raise a QeMLflowDependencyError for missing packages."""
    message = f"Required package '{package}' is not available"
    if operation:
        message += f" for {operation}"
    details = {"missing_package": package, "operation": operation}
    raise QeMLflowDependencyError(message, details)


def raise_config_error(config_key: str, expected_type: str = "") -> None:
    """Raise a QeMLflowConfigurationError for configuration issues."""
    message = f"Configuration error for key '{config_key}'"
    if expected_type:
        message += f" (expected {expected_type})"
    details = {"config_key": config_key, "expected_type": expected_type}
    raise QeMLflowConfigurationError(message, details)
