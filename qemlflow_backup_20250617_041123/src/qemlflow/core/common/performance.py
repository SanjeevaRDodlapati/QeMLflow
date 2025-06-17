"""
Performance monitoring and profiling utilities for ChemML.
"""

import functools
import logging
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, TypeVar, cast

import psutil

F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""

    execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    peak_memory_mb: float = 0.0
    cpu_percent: float = 0.0
    function_name: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "execution_time": self.execution_time,
            "memory_usage_mb": self.memory_usage_mb,
            "peak_memory_mb": self.peak_memory_mb,
            "cpu_percent": self.cpu_percent,
            "function_name": self.function_name,
            "timestamp": self.timestamp,
        }


class PerformanceMonitor:
    """Global performance monitoring singleton."""

    _instance: Optional["PerformanceMonitor"] = None

    def __init__(self) -> None:
        self.metrics_history: Dict[str, list] = defaultdict(list)
        self.logger = logging.getLogger(__name__)
        self.process = psutil.Process()

    @classmethod
    def get_instance(cls) -> "PerformanceMonitor":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def record_metrics(self, metrics: PerformanceMetrics) -> None:
        """Record performance metrics."""
        self.metrics_history[metrics.function_name].append(metrics)

        # Log if performance is concerning
        if metrics.execution_time > 10.0:  # > 10 seconds
            self.logger.warning(
                f"Slow execution detected: {metrics.function_name} "
                f"took {metrics.execution_time:.2f}s"
            )

        if metrics.memory_usage_mb > 1000:  # > 1GB
            self.logger.warning(
                f"High memory usage detected: {metrics.function_name} "
                f"used {metrics.memory_usage_mb:.1f}MB"
            )

    def get_function_stats(self, function_name: str) -> Dict[str, float]:
        """Get aggregated stats for a function."""
        history = self.metrics_history.get(function_name, [])

        if not history:
            return {}

        times = [m.execution_time for m in history]
        memory = [m.memory_usage_mb for m in history]

        return {
            "call_count": len(history),
            "avg_time": sum(times) / len(times),
            "max_time": max(times),
            "min_time": min(times),
            "avg_memory": sum(memory) / len(memory),
            "max_memory": max(memory),
            "total_time": sum(times),
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary for all monitored functions."""
        summary = {}

        for func_name in self.metrics_history:
            summary[func_name] = self.get_function_stats(func_name)

        return summary

    def clear_history(self) -> None:
        """Clear all performance metrics history."""
        self.metrics_history.clear()


def monitor_performance(
    log_threshold: float = 5.0, memory_threshold: float = 500.0, enabled: bool = True
):
    """
    Decorator to monitor function performance.

    Args:
        log_threshold: Log warning if execution time exceeds this (seconds)
        memory_threshold: Log warning if memory usage exceeds this (MB)
        enabled: Whether monitoring is enabled
    """

    def decorator(func: F) -> F:
        if not enabled:
            return func

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            monitor = PerformanceMonitor.get_instance()

            # Start monitoring
            start_time = time.time()
            process = monitor.process

            try:
                # Get initial memory
                memory_info = process.memory_info()
                initial_memory = memory_info.rss / 1024 / 1024  # Convert to MB

                # Execute function
                result = func(*args, **kwargs)

                # Calculate metrics
                end_time = time.time()
                execution_time = end_time - start_time

                # Get final memory and CPU
                final_memory_info = process.memory_info()
                final_memory = final_memory_info.rss / 1024 / 1024
                memory_delta = final_memory - initial_memory

                try:
                    cpu_percent = process.cpu_percent()
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    cpu_percent = 0.0

                # Create metrics
                metrics = PerformanceMetrics(
                    execution_time=execution_time,
                    memory_usage_mb=memory_delta,
                    peak_memory_mb=final_memory,
                    cpu_percent=cpu_percent,
                    function_name=func.__name__,
                )

                # Record metrics
                monitor.record_metrics(metrics)

                # Log if thresholds exceeded
                logger = logging.getLogger(func.__module__)
                if execution_time > log_threshold:
                    logger.warning(
                        f"{func.__name__} execution time: {execution_time:.2f}s "
                        f"(threshold: {log_threshold}s)"
                    )

                if abs(memory_delta) > memory_threshold:
                    logger.warning(
                        f"{func.__name__} memory delta: {memory_delta:+.1f}MB "
                        f"(threshold: {memory_threshold}MB)"
                    )

                return result

            except Exception:
                # Record failed execution
                end_time = time.time()
                execution_time = end_time - start_time

                metrics = PerformanceMetrics(
                    execution_time=execution_time,
                    function_name=f"{func.__name__}_FAILED",
                )
                monitor.record_metrics(metrics)
                raise

        return cast(F, wrapper)

    return decorator


@contextmanager
def performance_context(operation_name: str) -> None:
    """
    Context manager for monitoring performance of code blocks.

    Args:
        operation_name: Name of the operation being monitored
    """
    monitor = PerformanceMonitor.get_instance()
    logger = logging.getLogger(__name__)

    start_time = time.time()
    process = monitor.process

    try:
        memory_info = process.memory_info()
        initial_memory = memory_info.rss / 1024 / 1024

        yield

        end_time = time.time()
        execution_time = end_time - start_time

        final_memory_info = process.memory_info()
        final_memory = final_memory_info.rss / 1024 / 1024
        memory_delta = final_memory - initial_memory

        try:
            cpu_percent = process.cpu_percent()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            cpu_percent = 0.0

        metrics = PerformanceMetrics(
            execution_time=execution_time,
            memory_usage_mb=memory_delta,
            peak_memory_mb=final_memory,
            cpu_percent=cpu_percent,
            function_name=operation_name,
        )

        monitor.record_metrics(metrics)

        logger.debug(
            f"Operation '{operation_name}' completed in {execution_time:.2f}s, "
            f"memory delta: {memory_delta:+.1f}MB"
        )

    except Exception as e:
        end_time = time.time()
        execution_time = end_time - start_time

        logger.error(
            f"Operation '{operation_name}' failed after {execution_time:.2f}s: {e}"
        )
        raise


def get_system_info() -> Dict[str, Any]:
    """Get current system performance information."""
    try:
        return {
            "cpu_count": psutil.cpu_count(),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_total_gb": psutil.virtual_memory().total / 1024**3,
            "memory_available_gb": psutil.virtual_memory().available / 1024**3,
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage_percent": psutil.disk_usage("/").percent,
        }
    except Exception as e:
        logging.getLogger(__name__).warning(f"Could not get system info: {e}")
        return {}


def log_performance_summary() -> None:
    """Log a summary of all performance metrics."""
    monitor = PerformanceMonitor.get_instance()
    summary = monitor.get_summary()

    logger = logging.getLogger(__name__)
    logger.info("=== Performance Summary ===")

    for func_name, stats in summary.items():
        logger.info(
            f"{func_name}: {stats['call_count']} calls, "
            f"avg: {stats['avg_time']:.2f}s, "
            f"total: {stats['total_time']:.2f}s, "
            f"max memory: {stats['max_memory']:.1f}MB"
        )

    system_info = get_system_info()
    if system_info:
        logger.info(
            f"System: CPU {system_info.get('cpu_percent', 0):.1f}%, "
            f"Memory {system_info.get('memory_percent', 0):.1f}%"
        )


# Convenience function to get the global monitor
def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    return PerformanceMonitor.get_instance()
