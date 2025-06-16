"""
Enhanced error handling and validation utilities for ChemML.
"""

import functools
import logging
import traceback
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Optional, TypeVar, Union, cast

# Type variables for generic functions
F = TypeVar("F", bound=Callable[..., Any])
T = TypeVar("T")


class ChemMLError(Exception):
    """Base exception for ChemML-specific errors."""

    pass


class MolecularValidationError(ChemMLError):
    """Error in molecular data validation."""

    pass


class ModelError(ChemMLError):
    """Error in model operations."""

    pass


class DataProcessingError(ChemMLError):
    """Error in data processing operations."""

    pass


class ConfigurationError(ChemMLError):
    """Error in configuration setup."""

    pass


def handle_exceptions(
    default_return: Any = None,
    exceptions: tuple = (Exception,),
    log_errors: bool = True,
    reraise: bool = False,
):
    """
    Decorator for graceful exception handling with logging.

    Args:
        default_return: Value to return on exception
        exceptions: Tuple of exception types to catch
        log_errors: Whether to log exceptions
        reraise: Whether to reraise after logging
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                if log_errors:
                    logger = logging.getLogger(func.__module__)
                    logger.error(
                        f"Error in {func.__name__}: {str(e)}\n"
                        f"Traceback: {traceback.format_exc()}"
                    )

                if reraise:
                    raise

                return default_return

        return cast(F, wrapper)

    return decorator


def validate_smiles(smiles: str) -> str:
    """
    Validate and standardize SMILES string.

    Args:
        smiles: SMILES string to validate

    Returns:
        Validated SMILES string

    Raises:
        MolecularValidationError: If SMILES is invalid
    """
    if not isinstance(smiles, str):
        raise MolecularValidationError(f"SMILES must be string, got {type(smiles)}")

    smiles = smiles.strip()

    if not smiles:
        raise MolecularValidationError("SMILES cannot be empty")

    # Basic validation - could be enhanced with RDKit
    invalid_chars = {"@", "#", "$", "%", "^", "&", "*", "!", "~", "`"}
    if any(char in smiles for char in invalid_chars):
        raise MolecularValidationError(f"SMILES contains invalid characters: {smiles}")

    return smiles


def validate_numeric_range(
    value: Union[int, float],
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    name: str = "value",
) -> Union[int, float]:
    """
    Validate numeric value is within specified range.

    Args:
        value: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        name: Name of the value for error messages

    Returns:
        Validated value

    Raises:
        ValueError: If value is out of range
    """
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric, got {type(value)}")

    if min_val is not None and value < min_val:
        raise ValueError(f"{name} must be >= {min_val}, got {value}")

    if max_val is not None and value > max_val:
        raise ValueError(f"{name} must be <= {max_val}, got {value}")

    return value


@contextmanager
def error_context(operation: str) -> Iterator[None]:
    """
    Context manager for consistent error handling and logging.

    Args:
        operation: Description of the operation being performed
    """
    logger = logging.getLogger(__name__)

    try:
        logger.debug(f"Starting operation: {operation}")
        yield
        logger.debug(f"Completed operation: {operation}")

    except Exception as e:
        logger.error(f"Failed operation '{operation}': {str(e)}")
        logger.debug(f"Full traceback: {traceback.format_exc()}")
        raise


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default on division by zero.

    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Value to return if denominator is zero

    Returns:
        Division result or default value
    """
    try:
        if abs(denominator) < 1e-10:  # Avoid floating point issues
            return default
        return numerator / denominator
    except (ZeroDivisionError, TypeError):
        return default


class RetryableError(Exception):
    """Exception that indicates an operation should be retried."""

    pass


def retry_on_failure(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (RetryableError,),
):
    """
    Decorator to retry functions on specific exceptions.

    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between attempts (seconds)
        backoff: Multiplier for delay on each attempt
        exceptions: Tuple of exception types to retry on
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import time

            current_delay = delay
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger = logging.getLogger(func.__module__)
                        logger.warning(
                            f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                            f"Retrying in {current_delay:.1f}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}"
                        )
                        raise last_exception

            # This should never be reached, but just in case
            raise last_exception or Exception("Unexpected retry failure")

        return cast(F, wrapper)

    return decorator
