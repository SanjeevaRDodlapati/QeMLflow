"""
Integration Performance Monitoring
=================================

Comprehensive monitoring system for external model integrations
including performance metrics, usage analytics, and health monitoring.
"""

import json
import threading
import time
import warnings
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psutil


@dataclass
class IntegrationMetric:
    """Single integration performance metric."""

    model_name: str
    operation: str  # 'integration', 'prediction', 'training'
    timestamp: datetime
    duration_seconds: float
    memory_mb: float
    cpu_percent: float
    gpu_memory_mb: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    input_size: Optional[int] = None
    output_size: Optional[int] = None
    cache_hit: bool = False


@dataclass
class ModelHealthMetrics:
    """Aggregated health metrics for a model."""

    model_name: str
    total_uses: int = 0
    success_rate: float = 1.0
    avg_duration_seconds: float = 0.0
    avg_memory_mb: float = 0.0
    last_used: Optional[datetime] = None
    error_count: int = 0
    common_errors: Dict[str, int] = field(default_factory=dict)
    performance_trend: str = "stable"  # improving, stable, degrading


@dataclass
class SystemMetrics:
    """Current system resource metrics."""

    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_usage_percent: float
    gpu_count: int = 0
    gpu_memory_used_mb: float = 0.0
    gpu_memory_total_mb: float = 0.0


class IntegrationMetrics:
    """
    Performance monitoring system for external model integrations.

    Tracks integration time, memory usage, success rates, and system health.
    """

    def __init__(self, metrics_dir: Optional[str] = None):
        """Initialize the metrics system."""
        if metrics_dir:
            self.metrics_dir = Path(metrics_dir)
        else:
            self.metrics_dir = Path.home() / ".chemml" / "metrics"

        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        # Metric storage
        self.integration_metrics: List[IntegrationMetric] = []
        self.model_health: Dict[str, ModelHealthMetrics] = {}
        self.system_metrics: deque = deque(
            maxlen=1000
        )  # Keep last 1000 system snapshots

        # Performance tracking
        self._active_operations: Dict[str, Dict] = {}
        self._lock = threading.Lock()

        # Load existing metrics
        self._load_metrics()

        # Start background system monitoring
        self._start_system_monitoring()

    def _load_metrics(self):
        """Load metrics from disk."""
        try:
            # Load integration metrics
            integration_file = self.metrics_dir / "integration_metrics.json"
            if integration_file.exists():
                with open(integration_file) as f:
                    data = json.load(f)
                    self.integration_metrics = [
                        IntegrationMetric(
                            model_name=m["model_name"],
                            operation=m["operation"],
                            timestamp=datetime.fromisoformat(m["timestamp"]),
                            duration_seconds=m["duration_seconds"],
                            memory_mb=m["memory_mb"],
                            cpu_percent=m["cpu_percent"],
                            gpu_memory_mb=m.get("gpu_memory_mb", 0.0),
                            success=m["success"],
                            error_message=m.get("error_message"),
                            input_size=m.get("input_size"),
                            output_size=m.get("output_size"),
                            cache_hit=m.get("cache_hit", False),
                        )
                        for m in data
                    ]

            # Load model health metrics
            health_file = self.metrics_dir / "model_health.json"
            if health_file.exists():
                with open(health_file) as f:
                    data = json.load(f)
                    self.model_health = {
                        name: ModelHealthMetrics(
                            model_name=name,
                            total_uses=h["total_uses"],
                            success_rate=h["success_rate"],
                            avg_duration_seconds=h["avg_duration_seconds"],
                            avg_memory_mb=h["avg_memory_mb"],
                            last_used=(
                                datetime.fromisoformat(h["last_used"])
                                if h["last_used"]
                                else None
                            ),
                            error_count=h["error_count"],
                            common_errors=h.get("common_errors", {}),
                            performance_trend=h.get("performance_trend", "stable"),
                        )
                        for name, h in data.items()
                    }

        except Exception as e:
            warnings.warn(f"Failed to load metrics: {e}")

    def _save_metrics(self):
        """Save metrics to disk."""
        try:
            # Save integration metrics (last 10000 only)
            integration_file = self.metrics_dir / "integration_metrics.json"
            recent_metrics = self.integration_metrics[-10000:]  # Keep last 10k
            with open(integration_file, "w") as f:
                json.dump(
                    [
                        {
                            "model_name": m.model_name,
                            "operation": m.operation,
                            "timestamp": m.timestamp.isoformat(),
                            "duration_seconds": m.duration_seconds,
                            "memory_mb": m.memory_mb,
                            "cpu_percent": m.cpu_percent,
                            "gpu_memory_mb": m.gpu_memory_mb,
                            "success": m.success,
                            "error_message": m.error_message,
                            "input_size": m.input_size,
                            "output_size": m.output_size,
                            "cache_hit": m.cache_hit,
                        }
                        for m in recent_metrics
                    ],
                    f,
                    indent=2,
                )

            # Save model health metrics
            health_file = self.metrics_dir / "model_health.json"
            with open(health_file, "w") as f:
                json.dump(
                    {
                        name: {
                            "total_uses": h.total_uses,
                            "success_rate": h.success_rate,
                            "avg_duration_seconds": h.avg_duration_seconds,
                            "avg_memory_mb": h.avg_memory_mb,
                            "last_used": (
                                h.last_used.isoformat() if h.last_used else None
                            ),
                            "error_count": h.error_count,
                            "common_errors": h.common_errors,
                            "performance_trend": h.performance_trend,
                        }
                        for name, h in self.model_health.items()
                    },
                    f,
                    indent=2,
                )

        except Exception as e:
            warnings.warn(f"Failed to save metrics: {e}")

    def _start_system_monitoring(self):
        """Start background system monitoring."""

        def monitor_system():
            while True:
                try:
                    # Get system metrics
                    cpu_percent = psutil.cpu_percent(interval=1)
                    memory = psutil.virtual_memory()
                    disk = psutil.disk_usage("/")

                    # GPU metrics (if available)
                    gpu_count = 0
                    gpu_memory_used = 0.0
                    gpu_memory_total = 0.0

                    try:
                        import GPUtil

                        gpus = GPUtil.getGPUs()
                        gpu_count = len(gpus)
                        if gpus:
                            gpu_memory_used = sum(gpu.memoryUsed for gpu in gpus)
                            gpu_memory_total = sum(gpu.memoryTotal for gpu in gpus)
                    except ImportError:
                        pass

                    metric = SystemMetrics(
                        timestamp=datetime.now(),
                        cpu_percent=cpu_percent,
                        memory_percent=memory.percent,
                        memory_available_gb=memory.available / (1024**3),
                        disk_usage_percent=disk.percent,
                        gpu_count=gpu_count,
                        gpu_memory_used_mb=gpu_memory_used,
                        gpu_memory_total_mb=gpu_memory_total,
                    )

                    with self._lock:
                        self.system_metrics.append(metric)

                    time.sleep(60)  # Monitor every minute

                except Exception:
                    time.sleep(60)  # Continue monitoring even if errors occur

        # Start monitoring in background thread
        monitor_thread = threading.Thread(target=monitor_system, daemon=True)
        monitor_thread.start()

    @contextmanager
    def track_operation(
        self, model_name: str, operation: str, input_size: Optional[int] = None
    ):
        """
        Context manager for tracking operation performance.

        Args:
            model_name: Name of the model
            operation: Type of operation (integration, prediction, training)
            input_size: Size of input data
        """
        operation_id = f"{model_name}_{operation}_{int(time.time())}"

        # Record start metrics
        start_time = time.time()
        start_memory = psutil.virtual_memory().used / (1024**2)  # MB
        start_cpu = psutil.cpu_percent()

        # GPU metrics (if available)
        start_gpu_memory = 0.0
        try:
            import GPUtil

            gpus = GPUtil.getGPUs()
            if gpus:
                start_gpu_memory = sum(gpu.memoryUsed for gpu in gpus)
        except ImportError:
            pass

        success = True
        error_message = None

        try:
            yield
        except Exception as e:
            success = False
            error_message = str(e)
            raise
        finally:
            # Record end metrics
            end_time = time.time()
            end_memory = psutil.virtual_memory().used / (1024**2)  # MB
            end_cpu = psutil.cpu_percent()

            end_gpu_memory = 0.0
            try:
                import GPUtil

                gpus = GPUtil.getGPUs()
                if gpus:
                    end_gpu_memory = sum(gpu.memoryUsed for gpu in gpus)
            except ImportError:
                pass

            # Create metric
            metric = IntegrationMetric(
                model_name=model_name,
                operation=operation,
                timestamp=datetime.now(),
                duration_seconds=end_time - start_time,
                memory_mb=max(end_memory - start_memory, 0),
                cpu_percent=(start_cpu + end_cpu) / 2,
                gpu_memory_mb=max(end_gpu_memory - start_gpu_memory, 0),
                success=success,
                error_message=error_message,
                input_size=input_size,
            )

            # Record the metric
            self.record_metric(metric)

    def record_metric(self, metric: IntegrationMetric):
        """Record a performance metric."""
        with self._lock:
            self.integration_metrics.append(metric)

            # Update model health metrics
            if metric.model_name not in self.model_health:
                self.model_health[metric.model_name] = ModelHealthMetrics(
                    model_name=metric.model_name
                )

            health = self.model_health[metric.model_name]

            # Update aggregated metrics
            prev_total = health.total_uses
            health.total_uses += 1
            health.last_used = metric.timestamp

            # Update averages
            if prev_total > 0:
                health.avg_duration_seconds = (
                    health.avg_duration_seconds * prev_total + metric.duration_seconds
                ) / health.total_uses
                health.avg_memory_mb = (
                    health.avg_memory_mb * prev_total + metric.memory_mb
                ) / health.total_uses
            else:
                health.avg_duration_seconds = metric.duration_seconds
                health.avg_memory_mb = metric.memory_mb

            # Update success rate
            if not metric.success:
                health.error_count += 1
                if metric.error_message:
                    if metric.error_message not in health.common_errors:
                        health.common_errors[metric.error_message] = 0
                    health.common_errors[metric.error_message] += 1

            health.success_rate = (
                health.total_uses - health.error_count
            ) / health.total_uses

            # Update performance trend
            health.performance_trend = self._calculate_performance_trend(
                metric.model_name
            )

        # Save metrics periodically
        if len(self.integration_metrics) % 100 == 0:
            self._save_metrics()

    def _calculate_performance_trend(self, model_name: str) -> str:
        """Calculate performance trend for a model."""
        # Get recent metrics for this model
        recent_metrics = [
            m
            for m in self.integration_metrics[-100:]  # Last 100 operations
            if m.model_name == model_name and m.success
        ]

        if len(recent_metrics) < 10:
            return "stable"

        # Split into two halves and compare average performance
        mid = len(recent_metrics) // 2
        first_half = recent_metrics[:mid]
        second_half = recent_metrics[mid:]

        avg_duration_first = sum(m.duration_seconds for m in first_half) / len(
            first_half
        )
        avg_duration_second = sum(m.duration_seconds for m in second_half) / len(
            second_half
        )

        # Calculate relative change
        if avg_duration_first > 0:
            change = (avg_duration_second - avg_duration_first) / avg_duration_first

            if change < -0.1:  # 10% improvement
                return "improving"
            elif change > 0.1:  # 10% degradation
                return "degrading"

        return "stable"

    def get_model_performance_summary(self, model_name: str) -> Dict[str, Any]:
        """Get performance summary for a specific model."""
        if model_name not in self.model_health:
            return {"error": f"No metrics found for model '{model_name}'"}

        health = self.model_health[model_name]

        # Get recent metrics
        recent_metrics = [
            m for m in self.integration_metrics[-1000:] if m.model_name == model_name
        ]

        # Calculate additional stats
        operation_stats = defaultdict(lambda: {"count": 0, "avg_duration": 0.0})
        for metric in recent_metrics:
            op_stat = operation_stats[metric.operation]
            prev_count = op_stat["count"]
            op_stat["count"] += 1
            op_stat["avg_duration"] = (
                op_stat["avg_duration"] * prev_count + metric.duration_seconds
            ) / op_stat["count"]

        return {
            "model_name": model_name,
            "total_uses": health.total_uses,
            "success_rate": f"{health.success_rate:.1%}",
            "avg_duration_seconds": f"{health.avg_duration_seconds:.2f}",
            "avg_memory_mb": f"{health.avg_memory_mb:.1f}",
            "last_used": health.last_used.isoformat() if health.last_used else None,
            "error_count": health.error_count,
            "performance_trend": health.performance_trend,
            "operation_breakdown": dict(operation_stats),
            "common_errors": health.common_errors,
            "recent_activity": len(
                [
                    m
                    for m in recent_metrics
                    if m.timestamp > datetime.now() - timedelta(hours=24)
                ]
            ),
        }

    def get_system_health(self) -> Dict[str, Any]:
        """Get current system health metrics."""
        if not self.system_metrics:
            return {"error": "No system metrics available"}

        latest = self.system_metrics[-1]

        # Calculate averages over last hour
        hour_ago = datetime.now() - timedelta(hours=1)
        recent_metrics = [m for m in self.system_metrics if m.timestamp > hour_ago]

        if recent_metrics:
            avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
            avg_memory = sum(m.memory_percent for m in recent_metrics) / len(
                recent_metrics
            )
        else:
            avg_cpu = latest.cpu_percent
            avg_memory = latest.memory_percent

        return {
            "timestamp": latest.timestamp.isoformat(),
            "current": {
                "cpu_percent": f"{latest.cpu_percent:.1f}%",
                "memory_percent": f"{latest.memory_percent:.1f}%",
                "memory_available_gb": f"{latest.memory_available_gb:.1f}",
                "disk_usage_percent": f"{latest.disk_usage_percent:.1f}%",
                "gpu_count": latest.gpu_count,
                "gpu_memory_used_percent": (
                    f"{(latest.gpu_memory_used_mb / latest.gpu_memory_total_mb * 100):.1f}%"
                    if latest.gpu_memory_total_mb > 0
                    else "N/A"
                ),
            },
            "hourly_average": {
                "cpu_percent": f"{avg_cpu:.1f}%",
                "memory_percent": f"{avg_memory:.1f}%",
            },
            "status": self._assess_system_status(latest),
        }

    def _assess_system_status(self, metrics: SystemMetrics) -> str:
        """Assess overall system status."""
        issues = []

        if metrics.cpu_percent > 90:
            issues.append("high_cpu")
        if metrics.memory_percent > 90:
            issues.append("high_memory")
        if metrics.disk_usage_percent > 90:
            issues.append("high_disk")
        if (
            metrics.gpu_memory_total_mb > 0
            and metrics.gpu_memory_used_mb / metrics.gpu_memory_total_mb > 0.95
        ):
            issues.append("high_gpu_memory")

        if not issues:
            return "healthy"
        elif len(issues) == 1:
            return "warning"
        else:
            return "critical"

    def generate_performance_report(
        self, days: int = 7, model_name: Optional[str] = None
    ) -> str:
        """
        Generate a comprehensive performance report.

        Args:
            days: Number of days to include in the report
            model_name: Specific model to report on (None for all models)
        """
        cutoff_date = datetime.now() - timedelta(days=days)

        # Filter metrics by date and model
        filtered_metrics = [
            m
            for m in self.integration_metrics
            if m.timestamp > cutoff_date
            and (model_name is None or m.model_name == model_name)
        ]

        if not filtered_metrics:
            return f"No metrics found for the last {days} days"

        # Calculate overall statistics
        total_operations = len(filtered_metrics)
        successful_operations = len([m for m in filtered_metrics if m.success])
        success_rate = (
            successful_operations / total_operations if total_operations > 0 else 0
        )

        avg_duration = (
            sum(m.duration_seconds for m in filtered_metrics) / total_operations
        )
        avg_memory = sum(m.memory_mb for m in filtered_metrics) / total_operations

        # Group by model
        model_stats = defaultdict(
            lambda: {"count": 0, "success": 0, "total_duration": 0, "total_memory": 0}
        )

        for metric in filtered_metrics:
            stats = model_stats[metric.model_name]
            stats["count"] += 1
            if metric.success:
                stats["success"] += 1
            stats["total_duration"] += metric.duration_seconds
            stats["total_memory"] += metric.memory_mb

        # Generate report
        report = f"""
# Performance Report ({days} days)

## Overview
- **Total Operations**: {total_operations}
- **Success Rate**: {success_rate:.1%}
- **Average Duration**: {avg_duration:.2f} seconds
- **Average Memory Usage**: {avg_memory:.1f} MB

## Model Performance
"""

        for model, stats in sorted(model_stats.items()):
            model_success_rate = (
                stats["success"] / stats["count"] if stats["count"] > 0 else 0
            )
            avg_model_duration = stats["total_duration"] / stats["count"]
            avg_model_memory = stats["total_memory"] / stats["count"]

            health = self.model_health.get(model, ModelHealthMetrics(model_name=model))

            report += f"""
### {model}
- **Operations**: {stats["count"]}
- **Success Rate**: {model_success_rate:.1%}
- **Avg Duration**: {avg_model_duration:.2f}s
- **Avg Memory**: {avg_model_memory:.1f} MB
- **Trend**: {health.performance_trend}
"""

        # Add system health
        system_health = self.get_system_health()
        if "error" not in system_health:
            report += f"""
## System Health
- **Status**: {system_health["status"]}
- **CPU Usage**: {system_health["current"]["cpu_percent"]}
- **Memory Usage**: {system_health["current"]["memory_percent"]}
- **Available Memory**: {system_health["current"]["memory_available_gb"]} GB
- **GPU Count**: {system_health["current"]["gpu_count"]}
"""

        return report.strip()

    def cleanup_old_metrics(self, days_to_keep: int = 30):
        """Clean up old metrics to prevent excessive storage."""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)

        # Filter integration metrics
        old_count = len(self.integration_metrics)
        self.integration_metrics = [
            m for m in self.integration_metrics if m.timestamp > cutoff_date
        ]
        new_count = len(self.integration_metrics)

        print(f"Cleaned up {old_count - new_count} old integration metrics")

        # Save cleaned metrics
        self._save_metrics()


# Global instance
_metrics_instance = None


def get_metrics() -> IntegrationMetrics:
    """Get the global metrics instance."""
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = IntegrationMetrics()
    return _metrics_instance


# Decorators for easy metric tracking
def track_integration(model_name: str):
    """Decorator to track integration performance."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            metrics = get_metrics()
            with metrics.track_operation(model_name, "integration"):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def track_prediction(model_name: str):
    """Decorator to track prediction performance."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            metrics = get_metrics()
            input_size = (
                len(args[1]) if len(args) > 1 and hasattr(args[1], "__len__") else None
            )
            with metrics.track_operation(model_name, "prediction", input_size):
                return func(*args, **kwargs)

        return wrapper

    return decorator
