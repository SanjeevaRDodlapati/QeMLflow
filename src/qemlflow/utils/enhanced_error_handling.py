"""
Enhanced Error Handling and User Experience
==========================================

Phase 2 implementation: Improved error messages, debugging tools,
and user-friendly interfaces for QeMLflow.

Features:
- Contextual error messages with solutions
- Debugging utilities and diagnostics
- User-friendly error reporting
- Auto-recovery mechanisms
- Performance monitoring and alerts

Usage:
    from qemlflow.utils.enhanced_error_handling import QeMLflowError, debug_context

    with debug_context("Model Training"):
        # Your code here
        pass
"""

import functools
import logging
import sys
import time
import traceback
import warnings
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional, Union


class QeMLflowError(Exception):
    """Enhanced QeMLflow error with context and solutions."""

    def __init__(
        self,
        message: str,
        context: Optional[str] = None,
        solutions: Optional[List[str]] = None,
        error_code: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        self.message = message
        self.context = context or "General"
        self.solutions = solutions or []
        self.error_code = error_code
        self.original_error = original_error

        # Create enhanced error message
        enhanced_message = self._format_error_message()
        super().__init__(enhanced_message)

    def _format_error_message(self) -> str:
        """Format comprehensive error message."""
        lines = [f"🚨 QeMLflow Error in {self.context}", "=" * 50, f"❌ {self.message}"]

        if self.error_code:
            lines.append(f"🔍 Error Code: {self.error_code}")

        if self.original_error:
            lines.extend(
                [
                    "",
                    "📋 Original Error:",
                    f"   {type(self.original_error).__name__}: {self.original_error}",
                ]
            )

        if self.solutions:
            lines.extend(
                [
                    "",
                    "💡 Suggested Solutions:",
                ]
            )
            for i, solution in enumerate(self.solutions, 1):
                lines.append(f"   {i}. {solution}")

        lines.append("=" * 50)
        return "\n".join(lines)


class ErrorRecovery:
    """Auto-recovery mechanisms for common issues."""

    @staticmethod
    def missing_dependency_handler(
        package_name: str, install_command: str = None
    ) -> Callable:
        """Handle missing dependency errors with auto-install option."""

        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except ImportError as e:
                    if package_name in str(e):
                        install_cmd = install_command or f"pip install {package_name}"
                        raise QeMLflowError(
                            f"Missing required dependency: {package_name}",
                            context="Dependency Check",
                            solutions=[
                                f"Install the package: {install_cmd}",
                                "Update your requirements.txt",
                                "Check your virtual environment",
                            ],
                            error_code="MISSING_DEPENDENCY",
                            original_error=e,
                        )
                    raise

            return wrapper

        return decorator

    @staticmethod
    def file_not_found_handler(func: Callable) -> Callable:
        """Handle file not found errors with helpful solutions."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except FileNotFoundError as e:
                filename = str(e).split("'")[1] if "'" in str(e) else "unknown"
                raise QeMLflowError(
                    f"Required file not found: {filename}",
                    context="File Access",
                    solutions=[
                        f"Check if the file exists: {filename}",
                        "Verify the file path is correct",
                        "Ensure you have read permissions",
                        "Run setup scripts if this is a configuration file",
                    ],
                    error_code="FILE_NOT_FOUND",
                    original_error=e,
                )

        return wrapper

    @staticmethod
    def configuration_error_handler(func: Callable) -> Callable:
        """Handle configuration errors with auto-fix suggestions."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except (KeyError, ValueError) as e:
                if "config" in str(e).lower():
                    raise QeMLflowError(
                        f"Configuration error: {e}",
                        context="Configuration",
                        solutions=[
                            "Check your configuration files",
                            "Run: python tools/assessment/health_check.py --fix-issues",
                            "Reset to default configuration",
                            "Update configuration schema",
                        ],
                        error_code="CONFIG_ERROR",
                        original_error=e,
                    )
                raise

        return wrapper


class PerformanceMonitor:
    """Performance monitoring and alerting system."""

    def __init__(self):
        self.metrics = {}
        self.thresholds = {
            "import_time": 2.0,  # seconds
            "memory_usage": 1024 * 1024 * 1024,  # 1GB
            "execution_time": 30.0,  # seconds
        }

    @contextmanager
    def monitor_performance(self, operation_name: str):
        """Monitor performance of an operation."""
        start_time = time.time()
        start_memory = self._get_memory_usage()

        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()

            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory

            self.metrics[operation_name] = {
                "execution_time": execution_time,
                "memory_delta": memory_delta,
                "timestamp": time.time(),
            }

            self._check_thresholds(operation_name, execution_time, memory_delta)

    def _get_memory_usage(self) -> int:
        """Get current memory usage."""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss
        except ImportError:
            return 0

    def _check_thresholds(self, operation: str, exec_time: float, memory: int):
        """Check if metrics exceed thresholds."""
        warnings_issued = []

        if exec_time > self.thresholds["execution_time"]:
            warnings_issued.append(
                f"⚠️ Slow execution: {operation} took {exec_time:.2f}s "
                f"(threshold: {self.thresholds['execution_time']}s)"
            )

        if memory > self.thresholds["memory_usage"]:
            warnings_issued.append(
                f"⚠️ High memory usage: {operation} used {memory/1024/1024:.1f}MB "
                f"(threshold: {self.thresholds['memory_usage']/1024/1024:.1f}MB)"
            )

        for warning in warnings_issued:
            warnings.warn(warning, UserWarning)


class DebugContext:
    """Enhanced debugging context manager."""

    def __init__(self, context_name: str, verbose: bool = False):
        self.context_name = context_name
        self.verbose = verbose
        self.logger = logging.getLogger(f"qemlflow.debug.{context_name}")
        self.performance_monitor = PerformanceMonitor()

    def __enter__(self):
        if self.verbose:
            print(f"🔍 Starting debug context: {self.context_name}")
        self.logger.info(f"Entering context: {self.context_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self._handle_exception(exc_type, exc_val, exc_tb)

        if self.verbose:
            print(f"✅ Completed debug context: {self.context_name}")
        self.logger.info(f"Exiting context: {self.context_name}")
        return False  # Don't suppress exceptions

    def _handle_exception(self, exc_type, exc_val, exc_tb):
        """Handle exceptions with enhanced debugging info."""
        self.logger.error(
            f"Exception in {self.context_name}: {exc_type.__name__}: {exc_val}",
            exc_info=(exc_type, exc_val, exc_tb),
        )

        if self.verbose:
            print(f"🚨 Exception in {self.context_name}:")
            print(f"   Type: {exc_type.__name__}")
            print(f"   Message: {exc_val}")
            print("   Traceback:")
            traceback.print_tb(exc_tb, file=sys.stdout)


# Convenience functions
@contextmanager
def debug_context(name: str, verbose: bool = False):
    """Create a debug context."""
    with DebugContext(name, verbose) as ctx:
        yield ctx


def enhance_function_errors(context: str = None):
    """Decorator to enhance function errors with better messages."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                with debug_context(context or func.__name__):
                    return func(*args, **kwargs)
            except QeMLflowError:
                # Re-raise QeMLflow errors as-is
                raise
            except Exception as e:
                # Convert other exceptions to QeMLflowError
                raise QeMLflowError(
                    f"Error in {func.__name__}: {e}",
                    context=context or func.__name__,
                    solutions=[
                        "Check function parameters",
                        "Verify input data format",
                        "Enable verbose mode for more details",
                    ],
                    original_error=e,
                )

        return wrapper

    return decorator


# Global performance monitor instance
global_performance_monitor = PerformanceMonitor()


class UserExperienceEnhancer:
    """Overall user experience enhancement utilities."""

    @staticmethod
    def create_user_friendly_traceback(exc_info: tuple) -> str:
        """Create user-friendly traceback with solutions."""
        exc_type, exc_val, exc_tb = exc_info

        lines = [
            "🔍 QeMLflow Debug Information",
            "=" * 40,
            f"Error Type: {exc_type.__name__}",
            f"Error Message: {exc_val}",
            "",
            "📋 Call Stack (most recent call last):",
        ]

        # Get relevant traceback frames
        tb_lines = traceback.format_tb(exc_tb)
        for i, line in enumerate(tb_lines[-3:]):  # Show last 3 frames
            lines.append(f"  {i+1}. {line.strip()}")

        # Add suggestions based on error type
        suggestions = UserExperienceEnhancer._get_error_suggestions(
            exc_type, str(exc_val)
        )
        if suggestions:
            lines.extend(
                [
                    "",
                    "💡 Suggestions:",
                ]
            )
            for suggestion in suggestions:
                lines.append(f"   • {suggestion}")

        return "\n".join(lines)

    @staticmethod
    def _get_error_suggestions(exc_type: type, message: str) -> List[str]:
        """Get contextual suggestions based on error type."""
        suggestions = []

        if exc_type == ImportError:
            suggestions.extend(
                [
                    "Install missing packages with pip",
                    "Check your virtual environment",
                    "Verify package spelling",
                ]
            )
        elif exc_type == FileNotFoundError:
            suggestions.extend(
                [
                    "Check file path and spelling",
                    "Ensure file exists and is accessible",
                    "Verify current working directory",
                ]
            )
        elif exc_type == ValueError:
            suggestions.extend(
                [
                    "Check input data format and types",
                    "Verify parameter values are in valid range",
                    "Review function documentation",
                ]
            )
        elif exc_type == KeyError:
            suggestions.extend(
                [
                    "Check dictionary keys exist",
                    "Verify configuration file format",
                    "Review required parameters",
                ]
            )

        return suggestions


# Setup logging configuration for better debugging
def setup_enhanced_logging(level: str = "INFO", log_file: Optional[str] = None):
    """Setup enhanced logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            *([logging.FileHandler(log_file)] if log_file else []),
        ],
    )

    # Set QeMLflow-specific loggers
    qemlflow_logger = logging.getLogger("qemlflow")
    qemlflow_logger.setLevel(getattr(logging, level.upper()))


if __name__ == "__main__":
    # Example usage
    setup_enhanced_logging("DEBUG")

    print("🧪 QeMLflow Enhanced Error Handling Test")

    # Test debug context
    with debug_context("Test Context", verbose=True):
        print("This is inside debug context")

    # Test performance monitoring
    with global_performance_monitor.monitor_performance("test_operation"):
        time.sleep(0.1)  # Simulate work

    print("✅ Test completed successfully!")
