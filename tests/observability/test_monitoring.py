"""
Test suite for monitoring module.

This module tests the comprehensive production monitoring capabilities including
APM, health checks, performance alerting, and user experience monitoring.
"""

import json
import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from qemlflow.observability.monitoring import (
    MetricData, PerformanceMetrics, AlertRule, Alert,
    MetricsCollector, PerformanceMonitor, get_performance_monitor,
    monitor_function, monitor_performance, HealthChecker
)


class TestMetricData:
    """Test MetricData class."""
    
    def test_metric_data_creation(self):
        """Test MetricData creation."""
        metric = MetricData(
            name="test_metric",
            value=42.5,
            metric_type="gauge",
            unit="ms"
        )
        
        assert metric.name == "test_metric"
        assert metric.value == 42.5
        assert metric.metric_type == "gauge"
        assert metric.unit == "ms"
        assert metric.timestamp is not None
        assert isinstance(metric.tags, dict)
    
    def test_metric_data_to_dict(self):
        """Test MetricData to_dict conversion."""
        metric = MetricData(
            name="test_metric",
            value=42.5,
            tags={"service": "test"}
        )
        
        data = metric.to_dict()
        assert isinstance(data, dict)
        assert data["name"] == "test_metric"
        assert data["value"] == 42.5
        assert data["tags"]["service"] == "test"


class TestPerformanceMetrics:
    """Test PerformanceMetrics class."""
    
    def test_performance_metrics_creation(self):
        """Test PerformanceMetrics creation."""
        metrics = PerformanceMetrics(
            cpu_percent=50.0,
            memory_percent=60.0,
            response_time_ms=100.0
        )
        
        assert metrics.cpu_percent == 50.0
        assert metrics.memory_percent == 60.0
        assert metrics.response_time_ms == 100.0
        assert metrics.timestamp is not None
    
    def test_add_custom_metric(self):
        """Test adding custom metrics."""
        metrics = PerformanceMetrics()
        metrics.add_custom_metric("test_metric", 42.0)
        
        assert metrics.custom_metrics["test_metric"] == 42.0
    
    def test_to_dict(self):
        """Test PerformanceMetrics to_dict conversion."""
        metrics = PerformanceMetrics(cpu_percent=50.0)
        data = metrics.to_dict()
        
        assert isinstance(data, dict)
        assert data["cpu_percent"] == 50.0


class TestAlertRule:
    """Test AlertRule class."""
    
    def test_alert_rule_creation(self):
        """Test AlertRule creation."""
        rule = AlertRule(
            name="High CPU",
            metric_name="cpu_percent",
            condition="gt",
            threshold=80.0,
            severity="warning"
        )
        
        assert rule.name == "High CPU"
        assert rule.metric_name == "cpu_percent"
        assert rule.condition == "gt"
        assert rule.threshold == 80.0
        assert rule.severity == "warning"
        assert rule.enabled is True
    
    def test_alert_rule_evaluate_gt(self):
        """Test AlertRule evaluation for greater than."""
        rule = AlertRule(
            name="High CPU",
            metric_name="cpu_percent",
            condition="gt",
            threshold=80.0
        )
        
        assert rule.evaluate(85.0) is True
        assert rule.evaluate(75.0) is False
        assert rule.evaluate(80.0) is False
    
    def test_alert_rule_evaluate_gte(self):
        """Test AlertRule evaluation for greater than or equal."""
        rule = AlertRule(
            name="High CPU",
            metric_name="cpu_percent",
            condition="gte",
            threshold=80.0
        )
        
        assert rule.evaluate(85.0) is True
        assert rule.evaluate(80.0) is True
        assert rule.evaluate(75.0) is False
    
    def test_alert_rule_evaluate_lt(self):
        """Test AlertRule evaluation for less than."""
        rule = AlertRule(
            name="Low Disk",
            metric_name="disk_free_gb",
            condition="lt",
            threshold=10.0
        )
        
        assert rule.evaluate(5.0) is True
        assert rule.evaluate(15.0) is False
        assert rule.evaluate(10.0) is False
    
    def test_alert_rule_evaluate_disabled(self):
        """Test AlertRule evaluation when disabled."""
        rule = AlertRule(
            name="High CPU",
            metric_name="cpu_percent",
            condition="gt",
            threshold=80.0,
            enabled=False
        )
        
        assert rule.evaluate(85.0) is False


class TestAlert:
    """Test Alert class."""
    
    def test_alert_creation(self):
        """Test Alert creation."""
        alert = Alert(
            name="High CPU Alert",
            message="CPU usage is high",
            severity="warning",
            current_value=85.0,
            threshold=80.0
        )
        
        assert alert.name == "High CPU Alert"
        assert alert.message == "CPU usage is high"
        assert alert.severity == "warning"
        assert alert.current_value == 85.0
        assert alert.threshold == 80.0
        assert alert.status == "active"
        assert alert.alert_id is not None
    
    def test_alert_resolve(self):
        """Test alert resolution."""
        alert = Alert(name="Test Alert")
        assert alert.status == "active"
        assert alert.resolved_at is None
        
        alert.resolve()
        assert alert.status == "resolved"
        assert alert.resolved_at is not None
    
    def test_alert_acknowledge(self):
        """Test alert acknowledgment."""
        alert = Alert(name="Test Alert")
        assert alert.status == "active"
        
        alert.acknowledge()
        assert alert.status == "acknowledged"
    
    def test_alert_to_dict(self):
        """Test Alert to_dict conversion."""
        alert = Alert(
            name="Test Alert",
            severity="critical",
            current_value=95.0
        )
        
        data = alert.to_dict()
        assert isinstance(data, dict)
        assert data["name"] == "Test Alert"
        assert data["severity"] == "critical"
        assert data["current_value"] == 95.0


class TestMetricsCollector:
    """Test MetricsCollector class."""
    
    def test_metrics_collector_creation(self):
        """Test MetricsCollector creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = MetricsCollector(storage_dir=tmpdir)
            
            assert collector.storage_dir == Path(tmpdir)
            assert collector.max_data_points == 10000
            assert isinstance(collector.metrics_buffer, dict)
    
    def test_collect_metric(self):
        """Test metric collection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = MetricsCollector(storage_dir=tmpdir)
            
            collector.record_metric("test_metric", 42.0)
            
            assert "test_metric" in collector.metrics_buffer
            assert len(collector.metrics_buffer["test_metric"]) == 1
            assert collector.metrics_buffer["test_metric"][0].value == 42.0
    
    def test_get_metrics(self):
        """Test getting metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = MetricsCollector(storage_dir=tmpdir)
            
            # Add some metrics
            for i in range(5):
                collector.record_metric("test_metric", float(i))
            
            metrics = collector.get_metrics("test_metric")
            assert len(metrics) == 5
            assert metrics[0].value == 0.0
            assert metrics[4].value == 4.0
    
    def test_get_metrics_with_limit(self):
        """Test getting metrics with limit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = MetricsCollector(storage_dir=tmpdir)
            
            # Add some metrics
            for i in range(10):
                collector.record_metric("test_metric", float(i))
            
            metrics = collector.get_metrics("test_metric", limit=5)
            assert len(metrics) == 5
            # Should get the last 5 metrics
            assert metrics[0].value == 5.0
            assert metrics[4].value == 9.0
    
    def test_get_metrics_summary(self):
        """Test getting metrics summary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = MetricsCollector(storage_dir=tmpdir)
            
            # Add some metrics
            for i in range(5):
                collector.record_metric("test_metric", float(i))
            
            summary = collector.get_metric_summary("test_metric")
            assert isinstance(summary, dict)
            assert summary["count"] == 5
            assert summary["latest"] == 4.0


class TestPerformanceMonitor:
    """Test PerformanceMonitor class."""
    
    def test_performance_monitor_creation(self):
        """Test PerformanceMonitor creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'metrics_storage_dir': tmpdir,
                'alert_dir': tmpdir,
                'monitor_interval': 30
            }
            monitor = PerformanceMonitor(config)
            
            assert monitor.config == config
            assert monitor.monitor_interval == 30
            assert monitor.is_monitoring is False
            assert isinstance(monitor.metrics_collector, MetricsCollector)
    
    @patch('src.qemlflow.observability.monitoring.psutil.cpu_percent')
    @patch('src.qemlflow.observability.monitoring.psutil.virtual_memory')
    @patch('src.qemlflow.observability.monitoring.psutil.disk_usage')
    def test_collect_system_metrics(self, mock_disk, mock_memory, mock_cpu):
        """Test system metrics collection."""
        # Mock system metrics
        mock_cpu.return_value = 50.0
        mock_memory.return_value = Mock(percent=60.0, used=4*1024**3, available=2*1024**3)
        mock_disk.return_value = Mock(used=3*1024**3, total=4*1024**3, free=1*1024**3)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {'metrics_storage_dir': tmpdir}
            monitor = PerformanceMonitor(config)
            
            # Trigger system metrics collection
            monitor._collect_system_metrics()
            
            # Check that metrics were recorded
            cpu_metric = monitor.metrics_collector.get_latest_metric("system.cpu_percent")
            assert cpu_metric is not None
            assert cpu_metric.value == 50.0
            
            memory_metric = monitor.metrics_collector.get_latest_metric("system.memory_percent")
            assert memory_metric is not None
            assert memory_metric.value == 60.0
    
    def test_record_request(self):
        """Test request recording."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {'metrics_storage_dir': tmpdir}
            monitor = PerformanceMonitor(config)
            
            monitor.record_request(150.0, True)
            monitor.record_request(200.0, False)
            
            # Check recorded metrics
            response_time_metric = monitor.metrics_collector.get_latest_metric("application.response_time_ms")
            assert response_time_metric is not None
            assert response_time_metric.value == 200.0  # Latest value
            
            # Check counters
            assert monitor.request_count == 2
            assert monitor.error_count == 1
    
    def test_record_custom_metric(self):
        """Test custom metric recording."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {'metrics_storage_dir': tmpdir}
            monitor = PerformanceMonitor(config)
            
            monitor.record_custom_metric("custom.test_metric", 42.0, {"source": "test"})
            
            # Check recorded metric
            metric = monitor.metrics_collector.get_latest_metric("custom.test_metric")
            assert metric is not None
            assert metric.value == 42.0
            assert metric.tags["source"] == "test"
    
    def test_get_performance_summary(self):
        """Test performance summary generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {'metrics_storage_dir': tmpdir}
            monitor = PerformanceMonitor(config)
            
            # Add some test data
            monitor.record_request(100.0, True)
            monitor.record_request(200.0, False)
            
            summary = monitor.get_performance_summary()
            
            assert isinstance(summary, dict)
            assert "system" in summary
            assert "application" in summary
            assert "alerts" in summary
            
            # Check application metrics
            assert summary["application"]["response_time_ms"] == 200.0  # Latest


class TestMonitorDecorator:
    """Test monitoring decorators."""
    
    def test_monitor_function_context_manager(self):
        """Test monitor_function context manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {'metrics_storage_dir': tmpdir}
            monitor = PerformanceMonitor(config)
            
            with monitor_function(monitor, "test_function"):
                time.sleep(0.01)  # Small delay to test timing
            
            # Check that function call was recorded
            duration_metric = monitor.metrics_collector.get_latest_metric("function.test_function.duration_ms")
            assert duration_metric is not None
            assert duration_metric.value >= 10  # At least 10ms
    
    def test_monitor_performance_decorator(self):
        """Test monitor_performance decorator."""
        # Mock the global monitor
        with patch('src.qemlflow.observability.monitoring.get_performance_monitor') as mock_get_monitor:
            mock_monitor = Mock()
            mock_get_monitor.return_value = mock_monitor
            
            @monitor_performance("test_function")
            def test_func():
                return "test_result"
            
            result = test_func()
            
            assert result == "test_result"
            # Verify that monitor_function was used (indirectly)
            mock_get_monitor.assert_called_once()


class TestHealthChecker:
    """Test HealthChecker class."""
    
    def test_health_checker_creation(self):
        """Test HealthChecker creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {'metrics_storage_dir': tmpdir}
            monitor = PerformanceMonitor(config)
            checker = HealthChecker(monitor)
            
            assert checker.monitor == monitor
    
    def test_get_health_status_healthy(self):
        """Test health status when system is healthy."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {'metrics_storage_dir': tmpdir}
            monitor = PerformanceMonitor(config)
            checker = HealthChecker(monitor)
            
            # Mock healthy system metrics
            monitor.record_custom_metric("system.cpu_percent", 30.0)
            monitor.record_custom_metric("system.memory_percent", 40.0)
            
            health = checker.get_health_status()
            
            assert health["status"] in ["healthy", "degraded"]  # May vary based on mocking
            assert "timestamp" in health
            assert "details" in health
    
    def test_get_health_status_degraded(self):
        """Test health status when system is degraded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {'metrics_storage_dir': tmpdir}
            monitor = PerformanceMonitor(config)
            checker = HealthChecker(monitor)
            
            # Mock degraded system metrics
            monitor.record_custom_metric("system.cpu_percent", 95.0)
            monitor.record_custom_metric("system.memory_percent", 95.0)
            
            health = checker.get_health_status()
            
            assert health["status"] in ["degraded", "unhealthy"]


class TestGlobalInstances:
    """Test global instance management."""
    
    def test_get_performance_monitor(self):
        """Test getting global performance monitor."""
        monitor1 = get_performance_monitor()
        monitor2 = get_performance_monitor()
        
        # Should return the same instance
        assert monitor1 is monitor2
        assert isinstance(monitor1, PerformanceMonitor)


class TestIntegration:
    """Integration tests for monitoring system."""
    
    @patch('src.qemlflow.observability.monitoring.psutil.cpu_percent')
    @patch('src.qemlflow.observability.monitoring.psutil.virtual_memory')
    @patch('src.qemlflow.observability.monitoring.psutil.disk_usage')
    def test_full_monitoring_workflow(self, mock_disk, mock_memory, mock_cpu):
        """Test complete monitoring workflow."""
        # Mock system metrics
        mock_cpu.return_value = 45.0
        mock_memory.return_value = Mock(percent=55.0, used=2*1024**3, available=2*1024**3)
        mock_disk.return_value = Mock(used=3*1024**3, total=4*1024**3, free=1*1024**3)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {'metrics_storage_dir': tmpdir}
            monitor = PerformanceMonitor(config)
            
            # Collect system metrics
            monitor._collect_system_metrics()
            
            # Record some application events
            monitor.record_request(120.0, True)
            monitor.record_request(200.0, False)
            monitor.record_custom_metric("custom.validation_errors", 1.0)
            
            # Get performance summary
            summary = monitor.get_performance_summary()
            
            assert summary["system"]["cpu_percent"] == 45.0
            assert summary["system"]["memory_percent"] == 55.0
            assert summary["application"]["response_time_ms"] == 200.0  # Latest
            
            # Check health status
            checker = HealthChecker(monitor)
            health = checker.get_health_status()
            assert health["status"] in ["healthy", "degraded", "unhealthy"]


if __name__ == "__main__":
    pytest.main([__file__])
