"""
Resource Monitoring and Alerting Module

This module provides comprehensive resource monitoring including:
- Real-time resource utilization monitoring
- Performance metrics collection and analysis
- Alert management and notification system
- Resource utilization dashboards and reporting
"""

import json
import logging
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from threading import Lock
from typing import Any, Callable, Dict, List, Optional
from enum import Enum

import psutil


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertStatus(Enum):
    """Alert status states."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


@dataclass
class ResourceAlert:
    """Represents a resource monitoring alert."""
    id: str
    timestamp: float
    severity: AlertSeverity
    status: AlertStatus
    metric_name: str
    current_value: float
    threshold_value: float
    message: str
    source: str = "resource_monitor"
    acknowledged_at: Optional[float] = None
    resolved_at: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class MetricSnapshot:
    """Snapshot of system metrics at a point in time."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    network_io_bytes: int
    disk_io_bytes: int
    process_count: int
    load_average: List[float]
    custom_metrics: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class AlertManager:
    """
    Manages alert lifecycle including creation, acknowledgment, and resolution.
    """
    
    def __init__(self, alert_history_limit: int = 1000):
        self.alert_history_limit = alert_history_limit
        self.active_alerts: Dict[str, ResourceAlert] = {}
        self.alert_history: List[ResourceAlert] = []
        self.alert_handlers: List[Callable[[ResourceAlert], None]] = []
        self.logger = logging.getLogger(__name__)
        self._lock = Lock()
        
        # Alert suppression to prevent spam
        self.suppression_intervals: Dict[str, float] = {}
        self.default_suppression_interval = 300  # 5 minutes
    
    def add_alert_handler(self, handler: Callable[[ResourceAlert], None]) -> None:
        """Add a custom alert handler function."""
        self.alert_handlers.append(handler)
        self.logger.info(f"Added alert handler: {handler.__name__}")
    
    def create_alert(self, 
                    metric_name: str,
                    current_value: float,
                    threshold_value: float,
                    severity: AlertSeverity,
                    message: str,
                    **metadata) -> Optional[ResourceAlert]:
        """Create a new alert if not suppressed."""
        
        # Generate alert ID based on metric and threshold
        alert_id = f"{metric_name}_{threshold_value}_{severity.value}"
        
        # Check suppression
        if self._is_alert_suppressed(alert_id):
            return None
        
        with self._lock:
            # Check if similar alert already exists
            if alert_id in self.active_alerts:
                # Update existing alert
                existing_alert = self.active_alerts[alert_id]
                existing_alert.current_value = current_value
                existing_alert.timestamp = time.time()
                return existing_alert
            
            # Create new alert
            alert = ResourceAlert(
                id=alert_id,
                timestamp=time.time(),
                severity=severity,
                status=AlertStatus.ACTIVE,
                metric_name=metric_name,
                current_value=current_value,
                threshold_value=threshold_value,
                message=message,
                metadata=metadata
            )
            
            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)
            
            # Maintain history limit
            if len(self.alert_history) > self.alert_history_limit:
                self.alert_history.pop(0)
            
            # Set suppression
            self.suppression_intervals[alert_id] = time.time() + self.default_suppression_interval
        
        # Notify handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"Alert handler {handler.__name__} failed: {e}")
        
        self.logger.warning(f"Alert created: {alert.message}")
        return alert
    
    def acknowledge_alert(self, alert_id: str, user: str = "system") -> bool:
        """Acknowledge an active alert."""
        with self._lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_at = time.time()
                if alert.metadata is None:
                    alert.metadata = {}
                alert.metadata["acknowledged_by"] = user
                
                self.logger.info(f"Alert {alert_id} acknowledged by {user}")
                return True
            return False
    
    def resolve_alert(self, alert_id: str, user: str = "system") -> bool:
        """Resolve an active alert."""
        with self._lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = time.time()
                if alert.metadata is None:
                    alert.metadata = {}
                alert.metadata["resolved_by"] = user
                
                # Remove from active alerts
                del self.active_alerts[alert_id]
                
                self.logger.info(f"Alert {alert_id} resolved by {user}")
                return True
            return False
    
    def auto_resolve_alerts(self, metric_name: str, current_value: float) -> int:
        """Auto-resolve alerts when metric returns to normal."""
        resolved_count = 0
        
        with self._lock:
            alerts_to_resolve = []
            
            for alert_id, alert in self.active_alerts.items():
                if alert.metric_name == metric_name:
                    # Check if current value is back within threshold
                    if alert.severity in [AlertSeverity.WARNING, AlertSeverity.CRITICAL]:
                        if current_value < alert.threshold_value:
                            alerts_to_resolve.append(alert_id)
            
            # Resolve alerts
            for alert_id in alerts_to_resolve:
                if self.resolve_alert(alert_id, "auto_resolve"):
                    resolved_count += 1
        
        return resolved_count
    
    def _is_alert_suppressed(self, alert_id: str) -> bool:
        """Check if alert is currently suppressed."""
        if alert_id in self.suppression_intervals:
            return time.time() < self.suppression_intervals[alert_id]
        return False
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of current alert status."""
        with self._lock:
            active_by_severity = {}
            for alert in self.active_alerts.values():
                severity = alert.severity.value
                active_by_severity[severity] = active_by_severity.get(severity, 0) + 1
            
            recent_alerts = self.alert_history[-10:] if self.alert_history else []
            
            return {
                "total_active_alerts": len(self.active_alerts),
                "active_by_severity": active_by_severity,
                "total_historical_alerts": len(self.alert_history),
                "recent_alerts": [
                    {
                        "id": alert.id,
                        "timestamp": alert.timestamp,
                        "severity": alert.severity.value,
                        "status": alert.status.value,
                        "message": alert.message
                    }
                    for alert in recent_alerts
                ]
            }


class ResourceMonitor:
    """
    Comprehensive system resource monitoring with alerting capabilities.
    """
    
    def __init__(self, 
                 monitoring_interval: float = 10.0,
                 data_retention_hours: int = 24,
                 enable_alerting: bool = True):
        self.monitoring_interval = monitoring_interval
        self.data_retention_hours = data_retention_hours
        self.enable_alerting = enable_alerting
        
        self.logger = logging.getLogger(__name__)
        self.alert_manager = AlertManager() if enable_alerting else None
        
        # Monitoring data
        self.metric_snapshots: List[MetricSnapshot] = []
        self.custom_metrics: Dict[str, Callable[[], float]] = {}
        self._lock = Lock()
        
        # Monitoring control
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        
        # Alert thresholds
        self.alert_thresholds = {
            "cpu_percent": {"warning": 80.0, "critical": 95.0},
            "memory_percent": {"warning": 85.0, "critical": 95.0},
            "disk_usage_percent": {"warning": 80.0, "critical": 90.0},
            "load_average": {"warning": 5.0, "critical": 10.0}
        }
        
        # Performance baselines
        self.baselines: Dict[str, float] = {}
        self._calculate_initial_baselines()
    
    def _calculate_initial_baselines(self) -> None:
        """Calculate initial performance baselines."""
        try:
            # Sample metrics for baseline calculation
            samples = []
            for _ in range(5):
                snapshot = self._collect_metrics()
                samples.append(snapshot)
                time.sleep(1)
            
            # Calculate averages as baselines
            self.baselines = {
                "cpu_percent": sum(s.cpu_percent for s in samples) / len(samples),
                "memory_percent": sum(s.memory_percent for s in samples) / len(samples),
                "disk_usage_percent": sum(s.disk_usage_percent for s in samples) / len(samples)
            }
            
            self.logger.info(f"Established performance baselines: {self.baselines}")
            
        except Exception as e:
            self.logger.error(f"Failed to calculate baselines: {e}")
            self.baselines = {}
    
    def add_custom_metric(self, name: str, collector_func: Callable[[], float]) -> None:
        """Add a custom metric collector function."""
        self.custom_metrics[name] = collector_func
        self.logger.info(f"Added custom metric: {name}")
    
    def set_alert_threshold(self, metric_name: str, warning: float, critical: float) -> None:
        """Set alert thresholds for a metric."""
        self.alert_thresholds[metric_name] = {
            "warning": warning,
            "critical": critical
        }
        self.logger.info(f"Set alert thresholds for {metric_name}: warning={warning}, critical={critical}")
    
    def _collect_metrics(self) -> MetricSnapshot:
        """Collect comprehensive system metrics."""
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Disk metrics
        disk_usage = psutil.disk_usage('/')
        disk_usage_percent = (disk_usage.used / disk_usage.total) * 100
        
        # Network I/O
        try:
            network_io = psutil.net_io_counters()
            network_io_bytes = network_io.bytes_sent + network_io.bytes_recv
        except Exception:
            network_io_bytes = 0
        
        # Disk I/O
        try:
            disk_io = psutil.disk_io_counters()
            if disk_io:
                disk_io_bytes = disk_io.read_bytes + disk_io.write_bytes
            else:
                disk_io_bytes = 0
        except Exception:
            disk_io_bytes = 0
        
        # Process count
        process_count = len(psutil.pids())
        
        # Load average (Unix-like systems)
        try:
            load_average = list(psutil.getloadavg())
        except (AttributeError, OSError):
            load_average = []
        
        # Custom metrics
        custom_metrics = {}
        for name, collector in self.custom_metrics.items():
            try:
                custom_metrics[name] = collector()
            except Exception as e:
                self.logger.error(f"Custom metric {name} collection failed: {e}")
                custom_metrics[name] = 0.0
        
        return MetricSnapshot(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            disk_usage_percent=disk_usage_percent,
            network_io_bytes=network_io_bytes,
            disk_io_bytes=disk_io_bytes,
            process_count=process_count,
            load_average=load_average,
            custom_metrics=custom_metrics
        )
    
    def _check_alerts(self, snapshot: MetricSnapshot) -> None:
        """Check metrics against alert thresholds."""
        if not self.alert_manager:
            return
        
        # Standard metric alerts
        metrics_to_check = {
            "cpu_percent": snapshot.cpu_percent,
            "memory_percent": snapshot.memory_percent,
            "disk_usage_percent": snapshot.disk_usage_percent
        }
        
        # Add load average if available
        if snapshot.load_average:
            metrics_to_check["load_average"] = snapshot.load_average[0]
        
        # Add custom metrics
        metrics_to_check.update(snapshot.custom_metrics)
        
        for metric_name, current_value in metrics_to_check.items():
            if metric_name not in self.alert_thresholds:
                continue
            
            thresholds = self.alert_thresholds[metric_name]
            
            # Check critical threshold
            if current_value >= thresholds["critical"]:
                self.alert_manager.create_alert(
                    metric_name=metric_name,
                    current_value=current_value,
                    threshold_value=thresholds["critical"],
                    severity=AlertSeverity.CRITICAL,
                    message=f"{metric_name} is critically high: {current_value:.1f}% (threshold: {thresholds['critical']:.1f}%)"
                )
            
            # Check warning threshold
            elif current_value >= thresholds["warning"]:
                self.alert_manager.create_alert(
                    metric_name=metric_name,
                    current_value=current_value,
                    threshold_value=thresholds["warning"],
                    severity=AlertSeverity.WARNING,
                    message=f"{metric_name} is high: {current_value:.1f}% (threshold: {thresholds['warning']:.1f}%)"
                )
            
            # Auto-resolve if metric is back to normal
            else:
                self.alert_manager.auto_resolve_alerts(metric_name, current_value)
    
    def start_monitoring(self) -> None:
        """Start continuous resource monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._stop_monitoring.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        self.logger.info("Resource monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        if not self._monitoring:
            return
        
        self._stop_monitoring.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        self._monitoring = False
        self.logger.info("Resource monitoring stopped")
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while not self._stop_monitoring.wait(self.monitoring_interval):
            try:
                # Collect metrics
                snapshot = self._collect_metrics()
                
                with self._lock:
                    self.metric_snapshots.append(snapshot)
                    
                    # Maintain data retention
                    cutoff_time = time.time() - (self.data_retention_hours * 3600)
                    self.metric_snapshots = [
                        s for s in self.metric_snapshots 
                        if s.timestamp > cutoff_time
                    ]
                
                # Check for alerts
                if self.enable_alerting:
                    self._check_alerts(snapshot)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
    
    def get_current_metrics(self) -> Optional[MetricSnapshot]:
        """Get the most recent metrics snapshot."""
        with self._lock:
            return self.metric_snapshots[-1] if self.metric_snapshots else None
    
    def get_metrics_history(self, hours: int = 1) -> List[MetricSnapshot]:
        """Get metrics history for specified number of hours."""
        cutoff_time = time.time() - (hours * 3600)
        
        with self._lock:
            return [
                snapshot for snapshot in self.metric_snapshots
                if snapshot.timestamp > cutoff_time
            ]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        current = self.get_current_metrics()
        if not current:
            return {"error": "No metrics available"}
        
        # Calculate trends
        recent_history = self.get_metrics_history(1)  # Last hour
        trends = {}
        
        if len(recent_history) > 1:
            first = recent_history[0]
            trends = {
                "cpu_trend": current.cpu_percent - first.cpu_percent,
                "memory_trend": current.memory_percent - first.memory_percent,
                "disk_trend": current.disk_usage_percent - first.disk_usage_percent
            }
        
        # Performance vs baseline
        baseline_comparison = {}
        for metric, baseline in self.baselines.items():
            if hasattr(current, metric):
                current_value = getattr(current, metric)
                deviation = ((current_value - baseline) / baseline) * 100 if baseline > 0 else 0
                baseline_comparison[metric] = {
                    "current": current_value,
                    "baseline": baseline,
                    "deviation_percent": round(deviation, 1)
                }
        
        summary = {
            "timestamp": current.timestamp,
            "current_metrics": current.to_dict(),
            "trends_last_hour": trends,
            "baseline_comparison": baseline_comparison,
            "monitoring_active": self._monitoring,
            "data_points_collected": len(self.metric_snapshots)
        }
        
        # Add alert summary if available
        if self.alert_manager:
            summary["alerts"] = self.alert_manager.get_alert_summary()
        
        return summary
    
    def export_metrics(self, filepath: str, hours: int = 24) -> bool:
        """Export metrics data to JSON file."""
        try:
            metrics_data = self.get_metrics_history(hours)
            export_data = {
                "export_timestamp": time.time(),
                "hours_exported": hours,
                "total_snapshots": len(metrics_data),
                "metrics": [snapshot.to_dict() for snapshot in metrics_data]
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.logger.info(f"Exported {len(metrics_data)} metric snapshots to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {e}")
            return False
    
    def __del__(self):
        """Cleanup on destruction."""
        self.stop_monitoring()


class ResourceDashboard:
    """
    Resource utilization dashboard and reporting system.
    """
    
    def __init__(self, monitor: ResourceMonitor):
        self.monitor = monitor
        self.logger = logging.getLogger(__name__)
    
    def generate_html_report(self, output_path: str) -> bool:
        """Generate an HTML dashboard report."""
        try:
            summary = self.monitor.get_performance_summary()
            current = summary.get("current_metrics", {})
            
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>QeMLflow Resource Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .metric-card {{ border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin: 10px 0; }}
        .critical {{ border-color: #dc3545; background-color: #f8d7da; }}
        .warning {{ border-color: #ffc107; background-color: #fff3cd; }}
        .healthy {{ border-color: #28a745; background-color: #d1eddb; }}
        .metric-value {{ font-size: 24px; font-weight: bold; }}
        .timestamp {{ color: #666; font-size: 12px; }}
    </style>
</head>
<body>
    <h1>QeMLflow Resource Dashboard</h1>
    <div class="timestamp">Generated: {datetime.fromtimestamp(summary.get('timestamp', time.time())).strftime('%Y-%m-%d %H:%M:%S')}</div>
    
    <div class="metric-card {'critical' if current.get('cpu_percent', 0) > 90 else 'warning' if current.get('cpu_percent', 0) > 70 else 'healthy'}">
        <h3>CPU Usage</h3>
        <div class="metric-value">{current.get('cpu_percent', 0):.1f}%</div>
    </div>
    
    <div class="metric-card {'critical' if current.get('memory_percent', 0) > 90 else 'warning' if current.get('memory_percent', 0) > 80 else 'healthy'}">
        <h3>Memory Usage</h3>
        <div class="metric-value">{current.get('memory_percent', 0):.1f}%</div>
    </div>
    
    <div class="metric-card {'critical' if current.get('disk_usage_percent', 0) > 90 else 'warning' if current.get('disk_usage_percent', 0) > 80 else 'healthy'}">
        <h3>Disk Usage</h3>
        <div class="metric-value">{current.get('disk_usage_percent', 0):.1f}%</div>
    </div>
    
    <h2>Active Alerts</h2>
"""
            
            # Add alerts if available
            if "alerts" in summary:
                alerts = summary["alerts"]
                if alerts["total_active_alerts"] > 0:
                    html_content += f"<p>Total Active Alerts: {alerts['total_active_alerts']}</p>"
                    for alert in alerts.get("recent_alerts", []):
                        severity_class = alert["severity"]
                        html_content += f"""
                        <div class="metric-card {severity_class}">
                            <strong>{alert['severity'].upper()}</strong>: {alert['message']}
                            <div class="timestamp">{datetime.fromtimestamp(alert['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}</div>
                        </div>
                        """
                else:
                    html_content += "<p>No active alerts</p>"
            
            html_content += """
    <h2>System Information</h2>
    <div class="metric-card healthy">
        <p><strong>Monitoring Active:</strong> {}</p>
        <p><strong>Data Points Collected:</strong> {}</p>
    </div>
</body>
</html>
""".format(
                summary.get("monitoring_active", False),
                summary.get("data_points_collected", 0)
            )
            
            with open(output_path, 'w') as f:
                f.write(html_content)
            
            self.logger.info(f"Generated HTML dashboard: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to generate HTML report: {e}")
            return False
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get dashboard data in JSON format for web interfaces."""
        return self.monitor.get_performance_summary()


# Convenience functions
def create_resource_monitor(monitoring_interval: float = 10.0) -> ResourceMonitor:
    """Create a resource monitor with default configuration."""
    return ResourceMonitor(monitoring_interval=monitoring_interval)


def setup_basic_alerting(monitor: ResourceMonitor) -> None:
    """Set up basic alerting for common resource issues."""
    def log_alert(alert: ResourceAlert) -> None:
        """Simple alert handler that logs alerts."""
        logger = logging.getLogger(__name__)
        logger.warning(f"ALERT [{alert.severity.value.upper()}]: {alert.message}")
    
    if monitor.alert_manager:
        monitor.alert_manager.add_alert_handler(log_alert)
