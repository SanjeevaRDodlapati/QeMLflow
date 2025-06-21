"""
Production Monitoring Module

This module provides comprehensive production monitoring capabilities including
application performance monitoring (APM), real-time health checks, performance
alerting, and user experience monitoring for enterprise-grade operations.
"""

import json
import logging
import psutil
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Deque
from uuid import uuid4


@dataclass
class MetricData:
    """Individual metric data point."""
    
    name: str
    value: Union[int, float]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    tags: Dict[str, str] = field(default_factory=dict)
    metric_type: str = "gauge"  # gauge, counter, histogram, timer
    unit: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class PerformanceMetrics:
    """Performance metrics collection."""
    
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    # System metrics
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_available_mb: float = 0.0
    disk_usage_percent: float = 0.0
    disk_free_gb: float = 0.0
    
    # Application metrics
    active_connections: int = 0
    response_time_ms: float = 0.0
    throughput_rps: float = 0.0
    error_rate: float = 0.0
    
    # Custom metrics
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    
    def add_custom_metric(self, name: str, value: float):
        """Add custom metric."""
        self.custom_metrics[name] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class AlertRule:
    """Alert rule configuration."""
    
    rule_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    metric_name: str = ""
    condition: str = ""  # gt, lt, gte, lte, eq
    threshold: float = 0.0
    duration_seconds: int = 300  # 5 minutes
    severity: str = "warning"  # info, warning, critical
    enabled: bool = True
    
    # Actions
    email_recipients: List[str] = field(default_factory=list)
    webhook_url: str = ""
    slack_channel: str = ""
    
    def evaluate(self, value: float) -> bool:
        """Evaluate if alert should trigger."""
        if not self.enabled:
            return False
            
        if self.condition == "gt":
            return value > self.threshold
        elif self.condition == "lt":
            return value < self.threshold
        elif self.condition == "gte":
            return value >= self.threshold
        elif self.condition == "lte":
            return value <= self.threshold
        elif self.condition == "eq":
            return abs(value - self.threshold) < 0.001
        
        return False


@dataclass
class Alert:
    """Active alert."""
    
    # Fields with default values
    name: str = ""
    message: str = ""
    severity: str = "warning"
    current_value: float = 0.0
    threshold: float = 0.0
    status: str = "active"  # active, resolved, acknowledged
    rule_id: str = ""
    webhook_url: str = ""
    slack_channel: str = ""
    resolved_at: Optional[str] = None
    
    # Fields with factory defaults (must come after simple defaults)
    alert_id: str = field(default_factory=lambda: str(uuid4()))
    triggered_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    email_recipients: List[str] = field(default_factory=list)
    
    def resolve(self):
        """Mark alert as resolved."""
        self.status = "resolved"
        self.resolved_at = datetime.now(timezone.utc).isoformat()
    
    def acknowledge(self):
        """Acknowledge alert."""
        self.status = "acknowledged"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class MetricsCollector:
    """Collects and stores metrics data."""
    
    def __init__(self, storage_dir: str = "metrics_data", max_data_points: int = 10000):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True, parents=True)
        
        self.max_data_points = max_data_points
        self.metrics_buffer: Dict[str, Deque[MetricData]] = defaultdict(lambda: deque(maxlen=max_data_points))
        self.logger = logging.getLogger(__name__)
        
        # Thread safety
        self._lock = threading.Lock()
        
        self.logger.info(f"Metrics collector initialized: {self.storage_dir}")
    
    def record_metric(self, name: str, value: Union[int, float], 
                     tags: Optional[Dict[str, str]] = None,
                     metric_type: str = "gauge", unit: str = ""):
        """Record a metric data point."""
        with self._lock:
            metric = MetricData(
                name=name,
                value=float(value),
                tags=tags or {},
                metric_type=metric_type,
                unit=unit
            )
            
            self.metrics_buffer[name].append(metric)
            
            # Persist to storage periodically
            if len(self.metrics_buffer[name]) % 100 == 0:
                self._persist_metrics(name)
    
    def get_metrics(self, name: str, start_time: Optional[str] = None,
                   end_time: Optional[str] = None, limit: int = 1000) -> List[MetricData]:
        """Get metrics data."""
        with self._lock:
            metrics = list(self.metrics_buffer[name])
            
            # Filter by time range if specified
            if start_time or end_time:
                filtered_metrics = []
                for metric in metrics:
                    metric_time = metric.timestamp
                    
                    if start_time and metric_time < start_time:
                        continue
                    if end_time and metric_time > end_time:
                        continue
                    
                    filtered_metrics.append(metric)
                
                metrics = filtered_metrics
            
            # Apply limit
            return metrics[-limit:] if limit else metrics
    
    def get_latest_metric(self, name: str) -> Optional[MetricData]:
        """Get latest metric value."""
        with self._lock:
            if name in self.metrics_buffer and self.metrics_buffer[name]:
                return self.metrics_buffer[name][-1]
        return None
    
    def get_metric_summary(self, name: str, window_minutes: int = 60) -> Dict[str, Any]:
        """Get metric summary statistics."""
        window_start = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
        window_start_str = window_start.isoformat()
        
        metrics = self.get_metrics(name, start_time=window_start_str)
        
        if not metrics:
            return {"count": 0}
        
        values = [m.value for m in metrics]
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "latest": values[-1] if values else None,
            "window_minutes": window_minutes
        }
    
    def _persist_metrics(self, name: str):
        """Persist metrics to storage."""
        try:
            filename = f"metrics_{name}_{datetime.now().strftime('%Y%m%d')}.jsonl"
            filepath = self.storage_dir / filename
            
            metrics_to_persist = list(self.metrics_buffer[name])
            
            with open(filepath, 'a') as f:
                for metric in metrics_to_persist:
                    f.write(json.dumps(metric.to_dict()) + '\n')
                    
        except Exception as e:
            self.logger.error(f"Failed to persist metrics for {name}: {e}")
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old metric files."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            for file_path in self.storage_dir.glob("metrics_*.jsonl"):
                if file_path.stat().st_mtime < cutoff_date.timestamp():
                    file_path.unlink()
                    self.logger.info(f"Cleaned up old metrics file: {file_path}")
                    
        except Exception as e:
            self.logger.error(f"Failed to cleanup old metrics data: {e}")


class AlertManager:
    """Manages alerting rules and notifications."""
    
    def __init__(self, metrics_collector: MetricsCollector, alert_dir: str = "alerts"):
        self.metrics_collector = metrics_collector
        self.alert_dir = Path(alert_dir)
        self.alert_dir.mkdir(exist_ok=True, parents=True)
        
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        
        self.logger = logging.getLogger(__name__)
        self._lock = threading.Lock()
        
        # Alert state tracking - store trigger time and count
        self._alert_states: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"triggered_at": None, "count": 0})
        
        self.logger.info("Alert manager initialized")
    
    def add_alert_rule(self, rule: AlertRule):
        """Add an alert rule."""
        with self._lock:
            self.alert_rules[rule.rule_id] = rule
            self.logger.info(f"Added alert rule: {rule.name}")
    
    def remove_alert_rule(self, rule_id: str):
        """Remove an alert rule."""
        with self._lock:
            if rule_id in self.alert_rules:
                rule = self.alert_rules.pop(rule_id)
                self.logger.info(f"Removed alert rule: {rule.name}")
    
    def check_alerts(self):
        """Check all alert rules against current metrics."""
        with self._lock:
            for rule in self.alert_rules.values():
                try:
                    self._check_rule(rule)
                except Exception as e:
                    self.logger.error(f"Error checking alert rule {rule.name}: {e}")
    
    def _check_rule(self, rule: AlertRule):
        """Check individual alert rule."""
        latest_metric = self.metrics_collector.get_latest_metric(rule.metric_name)
        
        if not latest_metric:
            return
        
        current_value = latest_metric.value
        should_trigger = rule.evaluate(current_value)
        
        rule_state = self._alert_states[rule.rule_id]
        
        if should_trigger:
            if rule_state["triggered_at"] is None:
                # First time triggering
                rule_state["triggered_at"] = datetime.now(timezone.utc).isoformat()
                rule_state["count"] = 1
            else:
                # Check if duration threshold is met
                triggered_time = datetime.fromisoformat(rule_state["triggered_at"])
                duration = datetime.now(timezone.utc) - triggered_time
                
                if duration.total_seconds() >= rule.duration_seconds:
                    # Trigger alert if not already active
                    if rule.rule_id not in self.active_alerts:
                        self._trigger_alert(rule, current_value)
        else:
            # Reset state if condition is no longer met
            if rule_state["triggered_at"] is not None:
                rule_state["triggered_at"] = None
                rule_state["count"] = 0
                
                # Resolve active alert if exists
                if rule.rule_id in self.active_alerts:
                    self._resolve_alert(rule.rule_id)
    
    def _trigger_alert(self, rule: AlertRule, current_value: float):
        """Trigger an alert."""
        alert = Alert(
            rule_id=rule.rule_id,
            name=rule.name,
            message=f"{rule.name}: {rule.metric_name} is {current_value} (threshold: {rule.threshold})",
            severity=rule.severity,
            current_value=current_value,
            threshold=rule.threshold
        )
        
        self.active_alerts[rule.rule_id] = alert
        self.alert_history.append(alert)
        
        # Send notifications
        self._send_alert_notifications(alert, rule)
        
        # Persist alert
        self._persist_alert(alert)
        
        self.logger.warning(f"Alert triggered: {alert.message}")
    
    def _resolve_alert(self, rule_id: str):
        """Resolve an active alert."""
        if rule_id in self.active_alerts:
            alert = self.active_alerts.pop(rule_id)
            alert.resolve()
            
            # Persist resolution
            self._persist_alert(alert)
            
            self.logger.info(f"Alert resolved: {alert.name}")
    
    def _send_alert_notifications(self, alert: Alert, rule: AlertRule):
        """Send alert notifications."""
        try:
            # Email notifications
            if rule.email_recipients:
                self._send_email_alert(alert, rule.email_recipients)
            
            # Webhook notifications
            if rule.webhook_url:
                self._send_webhook_alert(alert, rule.webhook_url)
            
            # Slack notifications
            if rule.slack_channel:
                self._send_slack_alert(alert, rule.slack_channel)
                
        except Exception as e:
            self.logger.error(f"Failed to send alert notifications: {e}")
    
    def _send_email_alert(self, alert: Alert, recipients: List[str]):
        """Send email alert (placeholder implementation)."""
        # In production, integrate with email service
        self.logger.info(f"Email alert sent to {recipients}: {alert.message}")
    
    def _send_webhook_alert(self, alert: Alert, webhook_url: str):
        """Send webhook alert (placeholder implementation)."""
        # In production, make HTTP POST to webhook URL
        self.logger.info(f"Webhook alert sent to {webhook_url}: {alert.message}")
    
    def _send_slack_alert(self, alert: Alert, slack_channel: str):
        """Send Slack alert (placeholder implementation)."""
        # In production, integrate with Slack API
        self.logger.info(f"Slack alert sent to {slack_channel}: {alert.message}")
    
    def _persist_alert(self, alert: Alert):
        """Persist alert to storage."""
        try:
            filename = f"alerts_{datetime.now().strftime('%Y%m%d')}.jsonl"
            filepath = self.alert_dir / filename
            
            with open(filepath, 'a') as f:
                f.write(json.dumps(alert.to_dict()) + '\n')
                
        except Exception as e:
            self.logger.error(f"Failed to persist alert: {e}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        with self._lock:
            return list(self.active_alerts.values())
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        cutoff_time_str = cutoff_time.isoformat()
        
        return [alert for alert in self.alert_history 
                if alert.triggered_at >= cutoff_time_str]


class PerformanceMonitor:
    """Main performance monitoring system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.metrics_collector = MetricsCollector(
            storage_dir=self.config.get('metrics_storage_dir', 'metrics_data'),
            max_data_points=self.config.get('max_data_points', 10000)
        )
        
        self.alert_manager = AlertManager(
            self.metrics_collector,
            alert_dir=self.config.get('alert_dir', 'alerts')
        )
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.monitor_interval = self.config.get('monitor_interval', 30)  # seconds
        
        # Performance tracking
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        
        self._setup_default_alerts()
        
        self.logger.info("Performance monitor initialized")
    
    def _setup_default_alerts(self):
        """Setup default alert rules."""
        default_rules = [
            AlertRule(
                name="High CPU Usage",
                metric_name="system.cpu_percent",
                condition="gt",
                threshold=80.0,
                duration_seconds=300,
                severity="warning"
            ),
            AlertRule(
                name="High Memory Usage",
                metric_name="system.memory_percent",
                condition="gt",
                threshold=85.0,
                duration_seconds=300,
                severity="critical"
            ),
            AlertRule(
                name="High Error Rate",
                metric_name="application.error_rate",
                condition="gt",
                threshold=5.0,
                duration_seconds=180,
                severity="critical"
            ),
            AlertRule(
                name="High Response Time",
                metric_name="application.response_time_ms",
                condition="gt",
                threshold=2000.0,
                duration_seconds=300,
                severity="warning"
            )
        ]
        
        for rule in default_rules:
            self.alert_manager.add_alert_rule(rule)
    
    def start_monitoring(self):
        """Start performance monitoring."""
        if self.is_monitoring:
            self.logger.warning("Monitoring already started")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.is_monitoring = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        self.logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Collect application metrics
                self._collect_application_metrics()
                
                # Check alerts
                self.alert_manager.check_alerts()
                
                # Sleep until next collection
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitor_interval)
    
    def _collect_system_metrics(self):
        """Collect system performance metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics_collector.record_metric(
                "system.cpu_percent", cpu_percent, 
                metric_type="gauge", unit="percent"
            )
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.metrics_collector.record_metric(
                "system.memory_percent", memory.percent,
                metric_type="gauge", unit="percent"
            )
            self.metrics_collector.record_metric(
                "system.memory_used_mb", memory.used / 1024 / 1024,
                metric_type="gauge", unit="MB"
            )
            self.metrics_collector.record_metric(
                "system.memory_available_mb", memory.available / 1024 / 1024,
                metric_type="gauge", unit="MB"
            )
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.metrics_collector.record_metric(
                "system.disk_usage_percent", disk_percent,
                metric_type="gauge", unit="percent"
            )
            self.metrics_collector.record_metric(
                "system.disk_free_gb", disk.free / 1024 / 1024 / 1024,
                metric_type="gauge", unit="GB"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
    
    def _collect_application_metrics(self):
        """Collect application performance metrics."""
        try:
            # Calculate uptime
            uptime_seconds = time.time() - self.start_time
            self.metrics_collector.record_metric(
                "application.uptime_seconds", uptime_seconds,
                metric_type="gauge", unit="seconds"
            )
            
            # Request rate
            if uptime_seconds > 0:
                request_rate = self.request_count / uptime_seconds
                self.metrics_collector.record_metric(
                    "application.request_rate", request_rate,
                    metric_type="gauge", unit="rps"
                )
            
            # Error rate
            if self.request_count > 0:
                error_rate = (self.error_count / self.request_count) * 100
                self.metrics_collector.record_metric(
                    "application.error_rate", error_rate,
                    metric_type="gauge", unit="percent"
                )
            
        except Exception as e:
            self.logger.error(f"Failed to collect application metrics: {e}")
    
    def record_request(self, duration_ms: float, success: bool = True):
        """Record a request for metrics."""
        self.request_count += 1
        
        if not success:
            self.error_count += 1
        
        self.metrics_collector.record_metric(
            "application.response_time_ms", duration_ms,
            metric_type="histogram", unit="ms"
        )
    
    def record_custom_metric(self, name: str, value: Union[int, float],
                           tags: Optional[Dict[str, str]] = None,
                           metric_type: str = "gauge", unit: str = ""):
        """Record a custom metric."""
        self.metrics_collector.record_metric(name, value, tags, metric_type, unit)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get current performance summary."""
        return {
            "system": {
                "cpu_percent": self._get_latest_metric_value("system.cpu_percent"),
                "memory_percent": self._get_latest_metric_value("system.memory_percent"),
                "disk_usage_percent": self._get_latest_metric_value("system.disk_usage_percent")
            },
            "application": {
                "uptime_seconds": self._get_latest_metric_value("application.uptime_seconds"),
                "request_rate": self._get_latest_metric_value("application.request_rate"),
                "error_rate": self._get_latest_metric_value("application.error_rate"),
                "response_time_ms": self._get_latest_metric_value("application.response_time_ms")
            },
            "alerts": {
                "active_count": len(self.alert_manager.get_active_alerts()),
                "active_alerts": [alert.to_dict() for alert in self.alert_manager.get_active_alerts()]
            }
        }
    
    def _get_latest_metric_value(self, name: str) -> Optional[float]:
        """Get latest metric value."""
        metric = self.metrics_collector.get_latest_metric(name)
        return metric.value if metric else None
    
    def get_metrics_collector(self) -> MetricsCollector:
        """Get metrics collector instance."""
        return self.metrics_collector
    
    def get_alert_manager(self) -> AlertManager:
        """Get alert manager instance."""
        return self.alert_manager


# Context manager for monitoring function execution
class monitor_function:
    """Context manager for monitoring function execution."""
    
    def __init__(self, monitor: PerformanceMonitor, function_name: str):
        self.monitor = monitor
        self.function_name = function_name
        self.start_time: Optional[float] = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration_ms = (time.time() - self.start_time) * 1000
            success = exc_type is None
            
            self.monitor.record_request(duration_ms, success)
            self.monitor.record_custom_metric(
                f"function.{self.function_name}.duration_ms", 
                duration_ms,
                metric_type="histogram",
                unit="ms"
            )


# Global monitor instance
_global_performance_monitor = None

def get_performance_monitor(config: Optional[Dict[str, Any]] = None) -> PerformanceMonitor:
    """Get global performance monitor instance."""
    global _global_performance_monitor
    
    if _global_performance_monitor is None:
        _global_performance_monitor = PerformanceMonitor(config)
    
    return _global_performance_monitor


# Decorator for automatic function monitoring
def monitor_performance(function_name: Optional[str] = None):
    """Decorator to automatically monitor function performance."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            name = function_name or f"{func.__module__}.{func.__name__}"
            monitor = get_performance_monitor()
            
            with monitor_function(monitor, name):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


# Health checker classes
class HealthChecker:
    """Simple health checker for basic health status."""
    
    def __init__(self, performance_monitor: Optional[PerformanceMonitor] = None):
        self.monitor = performance_monitor or get_performance_monitor()
        self.logger = logging.getLogger(__name__)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status."""
        summary = self.monitor.get_performance_summary()
        
        # Determine overall health
        health_status = "healthy"
        
        # Check system health
        if summary["system"]["cpu_percent"] and summary["system"]["cpu_percent"] > 90:
            health_status = "degraded"
        
        if summary["system"]["memory_percent"] and summary["system"]["memory_percent"] > 90:
            health_status = "degraded"
        
        # Check application health
        if summary["application"]["error_rate"] and summary["application"]["error_rate"] > 10:
            health_status = "unhealthy"
        
        # Check active alerts
        if summary["alerts"]["active_count"] > 0:
            critical_alerts = [alert for alert in summary["alerts"]["active_alerts"] 
                             if alert["severity"] == "critical"]
            if critical_alerts:
                health_status = "unhealthy"
            elif health_status == "healthy":
                health_status = "degraded"
        
        return {
            "status": health_status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": summary
        }
