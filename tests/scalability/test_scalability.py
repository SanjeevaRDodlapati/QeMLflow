"""
Tests for QeMLflow Scalability System

This module contains comprehensive tests for horizontal scaling, load balancing,
and resource optimization capabilities.
"""

import tempfile
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from qemlflow.scalability import (
    ScalingMetrics, ScalingDecision, InstanceInfo,
    ReactiveScalingStrategy, PredictiveScalingStrategy,
    LoadBalancer, ResourceOptimizer, ScalabilityManager,
    initialize_scalability_system, get_scalability_manager,
    shutdown_scalability_system
)


class TestScalingMetrics(unittest.TestCase):
    """Test ScalingMetrics data class."""
    
    def test_metrics_creation(self):
        """Test creating scaling metrics."""
        metrics = ScalingMetrics(
            timestamp=datetime.now(),
            cpu_utilization=50.0,
            memory_utilization=60.0,
            request_rate=100.0,
            response_time=200.0,
            error_rate=1.0,
            queue_length=10,
            instance_count=3
        )
        
        self.assertEqual(metrics.cpu_utilization, 50.0)
        self.assertEqual(metrics.memory_utilization, 60.0)
        self.assertEqual(metrics.instance_count, 3)
    
    def test_metrics_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = ScalingMetrics(
            timestamp=datetime.now(),
            cpu_utilization=50.0,
            memory_utilization=60.0,
            request_rate=100.0,
            response_time=200.0,
            error_rate=1.0,
            queue_length=10,
            instance_count=3
        )
        
        metrics_dict = metrics.to_dict()
        self.assertIn('timestamp', metrics_dict)
        self.assertEqual(metrics_dict['cpu_utilization'], 50.0)
        self.assertEqual(metrics_dict['instance_count'], 3)


class TestScalingDecision(unittest.TestCase):
    """Test ScalingDecision data class."""
    
    def test_decision_creation(self):
        """Test creating scaling decision."""
        metrics = ScalingMetrics(
            timestamp=datetime.now(),
            cpu_utilization=50.0,
            memory_utilization=60.0,
            request_rate=100.0,
            response_time=200.0,
            error_rate=1.0,
            queue_length=10,
            instance_count=3
        )
        
        decision = ScalingDecision(
            action='scale_up',
            reason='High CPU utilization',
            target_instances=5,
            confidence=0.8,
            metrics=metrics
        )
        
        self.assertEqual(decision.action, 'scale_up')
        self.assertEqual(decision.target_instances, 5)
        self.assertEqual(decision.confidence, 0.8)
    
    def test_decision_to_dict(self):
        """Test converting decision to dictionary."""
        metrics = ScalingMetrics(
            timestamp=datetime.now(),
            cpu_utilization=50.0,
            memory_utilization=60.0,
            request_rate=100.0,
            response_time=200.0,
            error_rate=1.0,
            queue_length=10,
            instance_count=3
        )
        
        decision = ScalingDecision(
            action='scale_up',
            reason='High CPU utilization',
            target_instances=5,
            confidence=0.8,
            metrics=metrics
        )
        
        decision_dict = decision.to_dict()
        self.assertEqual(decision_dict['action'], 'scale_up')
        self.assertEqual(decision_dict['target_instances'], 5)
        self.assertIn('metrics', decision_dict)


class TestInstanceInfo(unittest.TestCase):
    """Test InstanceInfo data class."""
    
    def test_instance_creation(self):
        """Test creating instance info."""
        instance = InstanceInfo(
            instance_id='test-instance-1',
            host='localhost',
            port=8000,
            status='healthy',
            cpu_usage=30.0,
            memory_usage=40.0,
            request_count=100,
            last_health_check=datetime.now()
        )
        
        self.assertEqual(instance.instance_id, 'test-instance-1')
        self.assertEqual(instance.host, 'localhost')
        self.assertEqual(instance.port, 8000)
        self.assertEqual(instance.status, 'healthy')
    
    def test_instance_to_dict(self):
        """Test converting instance to dictionary."""
        instance = InstanceInfo(
            instance_id='test-instance-1',
            host='localhost',
            port=8000,
            status='healthy',
            cpu_usage=30.0,
            memory_usage=40.0,
            request_count=100,
            last_health_check=datetime.now()
        )
        
        instance_dict = instance.to_dict()
        self.assertEqual(instance_dict['instance_id'], 'test-instance-1')
        self.assertEqual(instance_dict['host'], 'localhost')
        self.assertEqual(instance_dict['port'], 8000)


class TestReactiveScalingStrategy(unittest.TestCase):
    """Test ReactiveScalingStrategy."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.strategy = ReactiveScalingStrategy()
    
    def test_scale_up_decision(self):
        """Test scale up decision."""
        metrics = ScalingMetrics(
            timestamp=datetime.now(),
            cpu_utilization=85.0,
            memory_utilization=90.0,
            request_rate=200.0,
            response_time=800.0,
            error_rate=2.0,
            queue_length=50,
            instance_count=2
        )
        
        config = {
            'targets': {
                'cpu_utilization': 70,
                'memory_utilization': 75,
                'request_rate': 100,
                'response_time': 500
            },
            'thresholds': {
                'scale_up': 0.2,
                'scale_down': -0.3
            },
            'max_instances': 10,
            'min_instances': 1
        }
        
        decision = self.strategy.should_scale(metrics, config)
        self.assertEqual(decision.action, 'scale_up')
        self.assertGreater(decision.target_instances, metrics.instance_count)
    
    def test_scale_down_decision(self):
        """Test scale down decision."""
        metrics = ScalingMetrics(
            timestamp=datetime.now(),
            cpu_utilization=20.0,
            memory_utilization=25.0,
            request_rate=30.0,
            response_time=100.0,
            error_rate=0.1,
            queue_length=2,
            instance_count=5
        )
        
        config = {
            'targets': {
                'cpu_utilization': 70,
                'memory_utilization': 75,
                'request_rate': 100,
                'response_time': 500
            },
            'thresholds': {
                'scale_up': 0.2,
                'scale_down': -0.3
            },
            'max_instances': 10,
            'min_instances': 1
        }
        
        decision = self.strategy.should_scale(metrics, config)
        self.assertEqual(decision.action, 'scale_down')
        self.assertLess(decision.target_instances, metrics.instance_count)
    
    def test_no_action_decision(self):
        """Test no action decision."""
        metrics = ScalingMetrics(
            timestamp=datetime.now(),
            cpu_utilization=70.0,
            memory_utilization=75.0,
            request_rate=100.0,
            response_time=500.0,
            error_rate=1.0,
            queue_length=10,
            instance_count=3
        )
        
        config = {
            'targets': {
                'cpu_utilization': 70,
                'memory_utilization': 75,
                'request_rate': 100,
                'response_time': 500
            },
            'thresholds': {
                'scale_up': 0.2,
                'scale_down': -0.3
            },
            'max_instances': 10,
            'min_instances': 1
        }
        
        decision = self.strategy.should_scale(metrics, config)
        self.assertEqual(decision.action, 'no_action')
        self.assertEqual(decision.target_instances, metrics.instance_count)


class TestPredictiveScalingStrategy(unittest.TestCase):
    """Test PredictiveScalingStrategy."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.strategy = PredictiveScalingStrategy()
    
    def test_insufficient_data(self):
        """Test handling insufficient historical data."""
        metrics = ScalingMetrics(
            timestamp=datetime.now(),
            cpu_utilization=50.0,
            memory_utilization=60.0,
            request_rate=100.0,
            response_time=200.0,
            error_rate=1.0,
            queue_length=10,
            instance_count=3
        )
        
        decision = self.strategy.should_scale(metrics, {})
        self.assertEqual(decision.action, 'no_action')
        self.assertIn('Insufficient historical data', decision.reason)
    
    def test_predictive_scale_up(self):
        """Test predictive scale up."""
        # Add historical data showing increasing trend
        base_time = datetime.now() - timedelta(minutes=50)
        for i in range(15):
            historical_metrics = ScalingMetrics(
                timestamp=base_time + timedelta(minutes=i*5),
                cpu_utilization=30.0 + i * 5,  # Increasing trend
                memory_utilization=40.0 + i * 3,
                request_rate=50.0,
                response_time=200.0,
                error_rate=1.0,
                queue_length=5,
                instance_count=2
            )
            self.strategy.add_historical_metrics(historical_metrics)
        
        current_metrics = ScalingMetrics(
            timestamp=datetime.now(),
            cpu_utilization=75.0,
            memory_utilization=70.0,
            request_rate=50.0,
            response_time=200.0,
            error_rate=1.0,
            queue_length=5,
            instance_count=2
        )
        
        config = {
            'cpu_threshold': 80,
            'memory_threshold': 85,
            'max_instances': 10,
            'min_instances': 1
        }
        
        decision = self.strategy.should_scale(current_metrics, config)
        self.assertEqual(decision.action, 'scale_up')
    
    def test_predictive_scale_down(self):
        """Test predictive scale down."""
        # Add historical data showing decreasing trend
        base_time = datetime.now() - timedelta(minutes=50)
        for i in range(15):
            historical_metrics = ScalingMetrics(
                timestamp=base_time + timedelta(minutes=i*5),
                cpu_utilization=80.0 - i * 3,  # Decreasing trend
                memory_utilization=70.0 - i * 2,
                request_rate=50.0,
                response_time=200.0,
                error_rate=1.0,
                queue_length=5,
                instance_count=3
            )
            self.strategy.add_historical_metrics(historical_metrics)
        
        current_metrics = ScalingMetrics(
            timestamp=datetime.now(),
            cpu_utilization=35.0,
            memory_utilization=40.0,
            request_rate=50.0,
            response_time=200.0,
            error_rate=1.0,
            queue_length=5,
            instance_count=3
        )
        
        config = {
            'cpu_threshold': 80,
            'memory_threshold': 85,
            'max_instances': 10,
            'min_instances': 1
        }
        
        decision = self.strategy.should_scale(current_metrics, config)
        self.assertEqual(decision.action, 'scale_down')


class TestLoadBalancer(unittest.TestCase):
    """Test LoadBalancer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'algorithm': 'round_robin',
            'health_check': {'enabled': True}
        }
        self.load_balancer = LoadBalancer(self.config)
    
    def test_add_remove_instance(self):
        """Test adding and removing instances."""
        instance = InstanceInfo(
            instance_id='test-1',
            host='localhost',
            port=8000,
            status='healthy',
            cpu_usage=30.0,
            memory_usage=40.0,
            request_count=10,
            last_health_check=datetime.now()
        )
        
        # Add instance
        self.load_balancer.add_instance(instance)
        self.assertIn('test-1', self.load_balancer.instances)
        
        # Remove instance
        self.load_balancer.remove_instance('test-1')
        self.assertNotIn('test-1', self.load_balancer.instances)
    
    def test_round_robin_algorithm(self):
        """Test round-robin load balancing."""
        # Add multiple instances
        for i in range(3):
            instance = InstanceInfo(
                instance_id=f'test-{i}',
                host='localhost',
                port=8000 + i,
                status='healthy',
                cpu_usage=30.0,
                memory_usage=40.0,
                request_count=10,
                last_health_check=datetime.now()
            )
            self.load_balancer.add_instance(instance)
        
        # Test round-robin selection
        selected_instances = []
        for _ in range(6):
            instance = self.load_balancer.get_next_instance()
            if instance:
                selected_instances.append(instance.instance_id)
        
        # Should cycle through instances
        self.assertEqual(len(selected_instances), 6)
        expected_pattern = ['test-0', 'test-1', 'test-2'] * 2
        self.assertEqual(selected_instances, expected_pattern)
    
    def test_least_connections_algorithm(self):
        """Test least connections load balancing."""
        self.load_balancer.config['algorithm'] = 'least_connections'
        
        # Add instances with different request counts
        for i in range(3):
            instance = InstanceInfo(
                instance_id=f'test-{i}',
                host='localhost',
                port=8000 + i,
                status='healthy',
                cpu_usage=30.0,
                memory_usage=40.0,
                request_count=i * 10,  # Different request counts
                last_health_check=datetime.now()
            )
            self.load_balancer.add_instance(instance)
        
        # Should select instance with least connections
        selected = self.load_balancer.get_next_instance()
        self.assertIsNotNone(selected)
        if selected:
            self.assertEqual(selected.instance_id, 'test-0')  # Has 0 request_count
    
    def test_no_healthy_instances(self):
        """Test behavior when no healthy instances available."""
        # Add unhealthy instance
        instance = InstanceInfo(
            instance_id='test-1',
            host='localhost',
            port=8000,
            status='unhealthy',
            cpu_usage=30.0,
            memory_usage=40.0,
            request_count=10,
            last_health_check=datetime.now()
        )
        self.load_balancer.add_instance(instance)
        
        # Should return None when no healthy instances
        selected = self.load_balancer.get_next_instance()
        self.assertIsNone(selected)


class TestResourceOptimizer(unittest.TestCase):
    """Test ResourceOptimizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'cpu': {'optimization_level': 'balanced'},
            'memory': {'optimization_level': 'balanced'}
        }
        self.optimizer = ResourceOptimizer(self.config)
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    def test_get_resource_usage(self, mock_disk, mock_memory, mock_cpu):
        """Test getting resource usage."""
        # Mock return values
        mock_cpu.return_value = 50.0
        mock_memory.return_value = Mock(percent=60.0, available=1000000000)
        mock_disk.return_value = Mock(used=50000000, total=100000000, free=50000000)
        
        usage = self.optimizer.get_resource_usage()
        
        self.assertEqual(usage['cpu_utilization'], 50.0)
        self.assertEqual(usage['memory_utilization'], 60.0)
        self.assertEqual(usage['disk_utilization'], 50.0)
    
    def test_optimize_for_workload(self):
        """Test workload-specific optimization."""
        # Test different workload types
        workload_types = ['cpu_intensive', 'memory_intensive', 'io_intensive', 'balanced']
        
        for workload in workload_types:
            # Should not raise exception
            self.optimizer.optimize_for_workload(workload)
        
        # Test unknown workload type (should default to balanced)
        self.optimizer.optimize_for_workload('unknown')


class TestScalabilityManager(unittest.TestCase):
    """Test ScalabilityManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / 'scalability.yml'
        
        # Create test config
        with open(self.config_path, 'w') as f:
            f.write("""
horizontal_scaling:
  enabled: true
  min_instances: 1
  max_instances: 5
monitoring_interval: 1
scaling_cooldown: 5
""")
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_manager_initialization(self):
        """Test manager initialization."""
        manager = ScalabilityManager(str(self.config_path))
        
        self.assertIsNotNone(manager.reactive_strategy)
        self.assertIsNotNone(manager.predictive_strategy)
        self.assertIsNotNone(manager.load_balancer)
        self.assertIsNotNone(manager.resource_optimizer)
        self.assertFalse(manager.running)
    
    def test_start_stop_manager(self):
        """Test starting and stopping manager."""
        manager = ScalabilityManager(str(self.config_path))
        
        # Start manager
        manager.start()
        self.assertTrue(manager.running)
        
        # Stop manager
        manager.stop()
        self.assertFalse(manager.running)
    
    @patch('qemlflow.scalability.ResourceOptimizer.get_resource_usage')
    def test_collect_metrics(self, mock_get_usage):
        """Test metrics collection."""
        mock_get_usage.return_value = {
            'cpu_utilization': 50.0,
            'memory_utilization': 60.0,
            'disk_utilization': 40.0,
            'memory_available': 1000000000,
            'disk_free': 50000000
        }
        
        manager = ScalabilityManager(str(self.config_path))
        
        # Test metrics collection
        manager._collect_and_analyze_metrics()
        
        # Should have collected metrics
        self.assertGreater(len(manager.metrics_history), 0)
    
    def test_get_status(self):
        """Test getting manager status."""
        manager = ScalabilityManager(str(self.config_path))
        
        status = manager.get_status()
        
        self.assertIn('running', status)
        self.assertIn('instance_count', status)
        self.assertIn('instances', status)
        self.assertIn('resource_usage', status)
        self.assertIn('last_scaling_action', status)


class TestGlobalFunctions(unittest.TestCase):
    """Test global scalability functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Ensure clean state
        shutdown_scalability_system()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutdown_scalability_system()
    
    def test_initialize_scalability_system(self):
        """Test initializing scalability system."""
        manager = initialize_scalability_system()
        
        self.assertIsNotNone(manager)
        self.assertIsInstance(manager, ScalabilityManager)
        if manager:
            self.assertTrue(manager.running)
        
        # Should return same instance on subsequent calls
        manager2 = initialize_scalability_system()
        self.assertIs(manager, manager2)
    
    def test_get_scalability_manager(self):
        """Test getting scalability manager."""
        # Should return None initially
        manager = get_scalability_manager()
        self.assertIsNone(manager)
        
        # Initialize and test
        initialize_scalability_system()
        manager = get_scalability_manager()
        self.assertIsNotNone(manager)
    
    def test_shutdown_scalability_system(self):
        """Test shutting down scalability system."""
        # Initialize system
        initialize_scalability_system()
        manager = get_scalability_manager()
        self.assertIsNotNone(manager)
        if manager:
            self.assertTrue(manager.running)
        
        # Shutdown system
        shutdown_scalability_system()
        
        # Manager should be None after shutdown
        manager = get_scalability_manager()
        self.assertIsNone(manager)


class TestScalabilityIntegration(unittest.TestCase):
    """Integration tests for scalability system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / 'scalability.yml'
        
        # Create comprehensive config
        with open(self.config_path, 'w') as f:
            f.write("""
horizontal_scaling:
  enabled: true
  min_instances: 1
  max_instances: 5
  target_cpu_utilization: 70
  target_memory_utilization: 80

load_balancing:
  enabled: true
  algorithm: round_robin

resource_optimization:
  enabled: true
  cpu:
    optimization_level: balanced
  memory:
    optimization_level: balanced

auto_scaling:
  enabled: true
  targets:
    cpu_utilization: 70
    memory_utilization: 75
    request_rate: 100
    response_time: 500
  thresholds:
    scale_up: 0.2
    scale_down: -0.3

monitoring_interval: 2
scaling_cooldown: 10
""")
        
        shutdown_scalability_system()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutdown_scalability_system()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('qemlflow.scalability.ResourceOptimizer.get_resource_usage')
    def test_full_scaling_workflow(self, mock_get_usage):
        """Test complete scaling workflow."""
        mock_get_usage.return_value = {
            'cpu_utilization': 85.0,  # High CPU to trigger scale up
            'memory_utilization': 90.0,  # High memory
            'disk_utilization': 40.0,
            'memory_available': 1000000000,
            'disk_free': 50000000
        }
        
        # Initialize system
        manager = initialize_scalability_system(str(self.config_path))
        
        # Let it run for a short time
        time.sleep(3)
        
        # Check status
        status = manager.get_status()
        self.assertTrue(status['running'])
        
        # Verify metrics were collected
        self.assertGreater(len(manager.metrics_history), 0)
        
        # Shutdown
        shutdown_scalability_system()
    
    def test_load_balancer_integration(self):
        """Test load balancer integration."""
        manager = initialize_scalability_system(str(self.config_path))
        
        # Add some instances manually
        for i in range(3):
            instance = InstanceInfo(
                instance_id=f'test-{i}',
                host='localhost',
                port=8000 + i,
                status='healthy',
                cpu_usage=30.0 + i * 10,
                memory_usage=40.0 + i * 5,
                request_count=i * 5,
                last_health_check=datetime.now()
            )
            manager.load_balancer.add_instance(instance)
        
        # Test load balancing
        for _ in range(5):
            selected = manager.load_balancer.get_next_instance()
            self.assertIsNotNone(selected)
        
        shutdown_scalability_system()


if __name__ == '__main__':
    unittest.main()
