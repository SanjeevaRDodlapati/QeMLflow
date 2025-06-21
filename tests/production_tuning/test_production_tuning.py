"""
Tests for Production Performance Tuning Module
"""

import time
from unittest.mock import Mock, patch
from datetime import datetime

import pytest

from qemlflow.production_tuning import (
    ProductionPerformanceTuner,
    PerformanceMetrics,
    initialize_production_performance,
    get_production_performance_status,
    validate_production_performance
)


class TestPerformanceMetrics:
    """Test performance metrics data structure."""
    
    def test_performance_metrics_creation(self):
        """Test creating performance metrics."""
        timestamp = datetime.now()
        metrics = PerformanceMetrics(
            timestamp=timestamp,
            cpu_usage=0.5,
            memory_usage=0.6,
            memory_available=1024*1024*1024,
            disk_usage=0.3,
            network_io={'bytes_sent': 1000.0, 'bytes_recv': 2000.0},
            active_connections=10,
            response_time=0.5,
            throughput=100.0,
            error_rate=0.01
        )
        
        assert metrics.timestamp == timestamp
        assert metrics.cpu_usage == 0.5
        assert metrics.memory_usage == 0.6
        assert metrics.memory_available == 1024*1024*1024
        assert metrics.disk_usage == 0.3
        assert metrics.network_io['bytes_sent'] == 1000.0
        assert metrics.active_connections == 10
        assert metrics.response_time == 0.5
        assert metrics.throughput == 100.0
        assert metrics.error_rate == 0.01


class TestProductionPerformanceTuner:
    """Test production performance tuner."""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing."""
        return {
            'performance': {
                'cpu': {
                    'max_cpu_usage': 0.8,
                    'enable_affinity': True
                },
                'memory': {
                    'gc_threshold': 0.7,
                    'enable_monitoring': True
                },
                'io': {
                    'max_concurrent_io': 50
                }
            },
            'monitoring': {
                'metrics': {
                    'interval': 1
                }
            }
        }
    
    @pytest.fixture
    def tuner(self, mock_config):
        """Create performance tuner for testing."""
        return ProductionPerformanceTuner(mock_config)
    
    def test_initialization(self, tuner):
        """Test performance tuner initialization."""
        assert tuner.cpu_threshold == 0.8
        assert tuner.memory_threshold == 0.7
        assert tuner.io_threshold == 50
        assert tuner.gc_enabled is True
        assert tuner.cpu_affinity_enabled is True
        assert tuner.is_monitoring is False
        assert tuner.tuning_enabled is True
        assert tuner.metrics_history == []
    
    def test_initialization_default_config(self):
        """Test initialization with default config."""
        tuner = ProductionPerformanceTuner()
        assert tuner.cpu_threshold == 0.85
        assert tuner.memory_threshold == 0.8
        assert tuner.io_threshold == 100
    
    def test_start_stop_monitoring(self, tuner):
        """Test starting and stopping monitoring."""
        assert not tuner.is_monitoring
        
        tuner.start_monitoring()
        assert tuner.is_monitoring
        assert tuner.monitoring_thread is not None
        
        # Wait a bit for thread to start
        time.sleep(0.1)
        
        tuner.stop_monitoring()
        assert not tuner.is_monitoring
    
    def test_start_monitoring_already_running(self, tuner):
        """Test starting monitoring when already running."""
        tuner.start_monitoring()
        initial_thread = tuner.monitoring_thread
        
        # Try to start again
        tuner.start_monitoring()
        assert tuner.monitoring_thread == initial_thread
        
        tuner.stop_monitoring()
    
    @patch('qemlflow.production_tuning.psutil.cpu_percent')
    @patch('qemlflow.production_tuning.psutil.virtual_memory')
    @patch('qemlflow.production_tuning.psutil.disk_usage')
    @patch('qemlflow.production_tuning.psutil.net_io_counters')
    @patch('qemlflow.production_tuning.psutil.net_connections')
    def test_collect_metrics(self, mock_connections, mock_net_io, mock_disk, 
                           mock_memory, mock_cpu, tuner):
        """Test metrics collection."""
        # Mock system metrics
        mock_cpu.return_value = 50.0
        
        mock_memory_obj = Mock()
        mock_memory_obj.percent = 60.0
        mock_memory_obj.available = 1024*1024*1024
        mock_memory.return_value = mock_memory_obj
        
        mock_disk_obj = Mock()
        mock_disk_obj.percent = 30.0
        mock_disk.return_value = mock_disk_obj
        
        mock_net_obj = Mock()
        mock_net_obj.bytes_sent = 1000
        mock_net_obj.bytes_recv = 2000
        mock_net_obj.packets_sent = 10
        mock_net_obj.packets_recv = 20
        mock_net_io.return_value = mock_net_obj
        
        mock_connections.return_value = [Mock() for _ in range(5)]
        
        metrics = tuner._collect_metrics()
        
        assert metrics.cpu_usage == 0.5
        assert metrics.memory_usage == 0.6
        assert metrics.memory_available == 1024*1024*1024
        assert metrics.disk_usage == 0.3
        assert metrics.network_io['bytes_sent'] == 1000.0
        assert metrics.active_connections == 5
        assert isinstance(metrics.timestamp, datetime)
    
    @patch('qemlflow.production_tuning.psutil.cpu_percent')
    def test_collect_metrics_error_handling(self, mock_cpu, tuner):
        """Test metrics collection error handling."""
        mock_cpu.side_effect = Exception("System monitoring error")
        
        metrics = tuner._collect_metrics()
        
        # Should return default metrics on error
        assert metrics.cpu_usage == 0.0
        assert metrics.memory_usage == 0.0
        assert isinstance(metrics.timestamp, datetime)
    
    @patch('qemlflow.production_tuning.gc.collect')
    def test_optimize_memory(self, mock_gc_collect, tuner):
        """Test memory optimization."""
        mock_gc_collect.return_value = 100
        
        tuner._optimize_memory()
        
        mock_gc_collect.assert_called_once()
    
    @patch('qemlflow.production_tuning.psutil.cpu_count')
    def test_optimize_cpu(self, mock_cpu_count, tuner):
        """Test CPU optimization."""
        mock_cpu_count.return_value = 4
        
        # Should not raise exception regardless of platform support
        tuner._optimize_cpu()
    
    @patch('qemlflow.production_tuning.psutil.cpu_count')
    def test_optimize_cpu_no_affinity_support(self, mock_cpu_count, tuner):
        """Test CPU optimization without affinity support."""
        mock_cpu_count.return_value = 4
        
        # Should not raise exception even without affinity support
        tuner._optimize_cpu()
    
    def test_optimize_io(self, tuner):
        """Test I/O optimization."""
        # Should not raise exception
        tuner._optimize_io()
    
    @patch('qemlflow.production_tuning.gc.collect')
    def test_force_garbage_collection(self, mock_gc_collect, tuner):
        """Test forced garbage collection."""
        mock_gc_collect.side_effect = [10, 5, 2]  # Three generations
        
        tuner._force_garbage_collection()
        
        assert mock_gc_collect.call_count == 3
    
    @patch('qemlflow.production_tuning.gc.collect')
    def test_force_garbage_collection_error(self, mock_gc_collect, tuner):
        """Test garbage collection error handling."""
        mock_gc_collect.side_effect = Exception("GC error")
        
        # Should not raise exception
        tuner._force_garbage_collection()
    
    def test_auto_tune_high_memory(self, tuner):
        """Test auto-tuning with high memory usage."""
        metrics = PerformanceMetrics(
            timestamp=datetime.now(),
            cpu_usage=0.5,
            memory_usage=0.95,  # Very high memory usage to trigger GC
            memory_available=1024*1024,
            disk_usage=0.3,
            network_io={},
            active_connections=10,
            response_time=0.5,
            throughput=100.0,
            error_rate=0.01
        )
        
        with patch.object(tuner, '_optimize_memory') as mock_optimize_memory, \
             patch.object(tuner, '_force_garbage_collection') as mock_gc:
            
            tuner._auto_tune(metrics)
            
            mock_optimize_memory.assert_called_once()
            mock_gc.assert_called_once()
    
    def test_auto_tune_high_cpu(self, tuner):
        """Test auto-tuning with high CPU usage."""
        metrics = PerformanceMetrics(
            timestamp=datetime.now(),
            cpu_usage=0.9,  # High CPU usage
            memory_usage=0.5,
            memory_available=1024*1024,
            disk_usage=0.3,
            network_io={},
            active_connections=10,
            response_time=0.5,
            throughput=100.0,
            error_rate=0.01
        )
        
        with patch.object(tuner, '_optimize_cpu') as mock_optimize_cpu:
            tuner._auto_tune(metrics)
            mock_optimize_cpu.assert_called_once()
    
    def test_auto_tune_high_io(self, tuner):
        """Test auto-tuning with high I/O load."""
        metrics = PerformanceMetrics(
            timestamp=datetime.now(),
            cpu_usage=0.5,
            memory_usage=0.5,
            memory_available=1024*1024,
            disk_usage=0.3,
            network_io={},
            active_connections=100,  # High connection count
            response_time=0.5,
            throughput=100.0,
            error_rate=0.01
        )
        
        with patch.object(tuner, '_optimize_io') as mock_optimize_io:
            tuner._auto_tune(metrics)
            mock_optimize_io.assert_called_once()
    
    def test_get_performance_summary_no_data(self, tuner):
        """Test performance summary with no data."""
        summary = tuner.get_performance_summary()
        
        assert summary['status'] == 'no_data'
        assert 'message' in summary
    
    def test_get_performance_summary_healthy(self, tuner):
        """Test performance summary with healthy metrics."""
        # Add some healthy metrics
        for _ in range(5):
            metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_usage=0.4,  # Low CPU
                memory_usage=0.5,  # Low memory
                memory_available=1024*1024*1024,
                disk_usage=0.3,
                network_io={},
                active_connections=10,
                response_time=0.2,
                throughput=150.0,
                error_rate=0.001  # Low error rate
            )
            tuner.metrics_history.append(metrics)
        
        summary = tuner.get_performance_summary()
        
        assert summary['status'] == 'healthy'
        assert summary['metrics']['cpu_usage'] == 0.4
        assert summary['metrics']['memory_usage'] == 0.5
        assert summary['issues'] == []
    
    def test_get_performance_summary_warning(self, tuner):
        """Test performance summary with warning conditions."""
        # Add metrics with high resource usage
        for _ in range(5):
            metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_usage=0.9,  # High CPU
                memory_usage=0.9,  # High memory
                memory_available=1024*1024,
                disk_usage=0.3,
                network_io={},
                active_connections=10,
                response_time=0.5,
                throughput=100.0,
                error_rate=0.01
            )
            tuner.metrics_history.append(metrics)
        
        summary = tuner.get_performance_summary()
        
        assert summary['status'] == 'warning'
        assert len(summary['issues']) >= 2  # High CPU and memory
        assert len(summary['recommendations']) >= 2
    
    def test_get_performance_summary_critical(self, tuner):
        """Test performance summary with critical conditions."""
        # Add metrics with high error rate
        for _ in range(5):
            metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_usage=0.5,
                memory_usage=0.5,
                memory_available=1024*1024*1024,
                disk_usage=0.3,
                network_io={},
                active_connections=10,
                response_time=0.5,
                throughput=100.0,
                error_rate=0.1  # High error rate (10%)
            )
            tuner.metrics_history.append(metrics)
        
        summary = tuner.get_performance_summary()
        
        assert summary['status'] == 'critical'
        assert any('error rate' in issue.lower() for issue in summary['issues'])
    
    @patch('qemlflow.production_tuning.gc.set_threshold')
    @patch('qemlflow.production_tuning.psutil.cpu_count')
    def test_apply_production_optimizations(self, mock_cpu_count, mock_gc_threshold, tuner):
        """Test applying production optimizations."""
        mock_cpu_count.return_value = 4
        
        result = tuner.apply_production_optimizations()
        
        assert result['status'] == 'success'
        assert len(result['optimizations']) >= 2
        mock_gc_threshold.assert_called_once_with(700, 10, 10)
    
    def test_validate_production_readiness(self, tuner):
        """Test production readiness validation."""
        result = tuner.validate_production_readiness()
        
        assert 'ready' in result
        assert 'score' in result
        assert 'checks' in result
        assert 'summary' in result
        assert isinstance(result['ready'], bool)
        assert 0 <= result['score'] <= 1


class TestUtilityFunctions:
    """Test utility functions."""
    
    @patch('qemlflow.production_tuning.ProductionPerformanceTuner')
    def test_initialize_production_performance(self, mock_tuner_class):
        """Test production performance initialization."""
        mock_tuner = Mock()
        mock_tuner_class.return_value = mock_tuner
        
        config = {'test': 'config'}
        result = initialize_production_performance(config)
        
        mock_tuner_class.assert_called_once_with(config)
        mock_tuner.apply_production_optimizations.assert_called_once()
        mock_tuner.start_monitoring.assert_called_once()
        assert result == mock_tuner
    
    def test_get_production_performance_status(self):
        """Test getting production performance status."""
        mock_tuner = Mock()
        mock_summary = {'status': 'healthy'}
        mock_tuner.get_performance_summary.return_value = mock_summary
        
        result = get_production_performance_status(mock_tuner)
        
        assert result == mock_summary
        mock_tuner.get_performance_summary.assert_called_once()
    
    def test_validate_production_performance(self):
        """Test validating production performance."""
        mock_tuner = Mock()
        mock_validation = {'ready': True}
        mock_tuner.validate_production_readiness.return_value = mock_validation
        
        result = validate_production_performance(mock_tuner)
        
        assert result == mock_validation
        mock_tuner.validate_production_readiness.assert_called_once()


if __name__ == '__main__':
    pytest.main([__file__])
