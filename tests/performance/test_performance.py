"""
Tests for QeMLflow Performance Optimization System

This module contains comprehensive tests for caching, performance optimization,
and monitoring capabilities.
"""

import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from qemlflow.performance import (
    CacheStats, PerformanceMetrics, MemoryCache, FileCache,
    CacheManager, PerformanceOptimizer, PerformanceMonitor,
    PerformanceManager, initialize_performance_system,
    get_performance_manager, shutdown_performance_system,
    cached, optimized
)


class TestCacheStats(unittest.TestCase):
    """Test CacheStats data class."""
    
    def test_stats_creation(self):
        """Test creating cache stats."""
        stats = CacheStats(
            hits=100,
            misses=20,
            evictions=5,
            size=50,
            max_size=100
        )
        
        self.assertEqual(stats.hits, 100)
        self.assertEqual(stats.misses, 20)
        self.assertEqual(stats.size, 50)
    
    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        # Normal case
        stats = CacheStats(hits=80, misses=20)
        self.assertEqual(stats.hit_rate, 0.8)
        
        # No requests case
        stats = CacheStats(hits=0, misses=0)
        self.assertEqual(stats.hit_rate, 0.0)
        
        # Only hits
        stats = CacheStats(hits=100, misses=0)
        self.assertEqual(stats.hit_rate, 1.0)
    
    def test_stats_to_dict(self):
        """Test converting stats to dictionary."""
        stats = CacheStats(hits=100, misses=20, size=50)
        stats_dict = stats.to_dict()
        
        self.assertIn('hits', stats_dict)
        self.assertIn('hit_rate', stats_dict)
        self.assertEqual(stats_dict['hits'], 100)
        self.assertEqual(stats_dict['hit_rate'], 0.8333333333333334)


class TestPerformanceMetrics(unittest.TestCase):
    """Test PerformanceMetrics data class."""
    
    def test_metrics_creation(self):
        """Test creating performance metrics."""
        metrics = PerformanceMetrics(
            timestamp=datetime.now(),
            cpu_utilization=50.0,
            memory_utilization=60.0,
            response_time=200.0,
            throughput=100.0,
            cache_hit_rate=0.85,
            error_rate=0.01
        )
        
        self.assertEqual(metrics.cpu_utilization, 50.0)
        self.assertEqual(metrics.cache_hit_rate, 0.85)
    
    def test_metrics_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = PerformanceMetrics(
            timestamp=datetime.now(),
            cpu_utilization=50.0,
            memory_utilization=60.0,
            response_time=200.0,
            throughput=100.0,
            cache_hit_rate=0.85,
            error_rate=0.01
        )
        
        metrics_dict = metrics.to_dict()
        self.assertIn('timestamp', metrics_dict)
        self.assertEqual(metrics_dict['cpu_utilization'], 50.0)


class TestMemoryCache(unittest.TestCase):
    """Test MemoryCache implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cache = MemoryCache(max_size=5, ttl=None)
    
    def test_basic_operations(self):
        """Test basic cache operations."""
        # Set and get
        self.assertTrue(self.cache.set('key1', 'value1'))
        self.assertEqual(self.cache.get('key1'), 'value1')
        
        # Get non-existent key
        self.assertIsNone(self.cache.get('key2'))
        
        # Delete key
        self.assertTrue(self.cache.delete('key1'))
        self.assertIsNone(self.cache.get('key1'))
    
    def test_lru_eviction(self):
        """Test LRU eviction policy."""
        # Fill cache to capacity
        for i in range(5):
            self.cache.set(f'key{i}', f'value{i}')
        
        # Add one more item - should evict oldest
        self.cache.set('key5', 'value5')
        
        # First item should be evicted
        self.assertIsNone(self.cache.get('key0'))
        self.assertEqual(self.cache.get('key5'), 'value5')
    
    def test_lru_ordering(self):
        """Test LRU ordering with access."""
        # Fill cache
        for i in range(5):
            self.cache.set(f'key{i}', f'value{i}')
        
        # Access first item (makes it most recent)
        self.cache.get('key0')
        
        # Add new item - should evict key1 (now oldest)
        self.cache.set('key5', 'value5')
        
        self.assertEqual(self.cache.get('key0'), 'value0')  # Should still exist
        self.assertIsNone(self.cache.get('key1'))  # Should be evicted
    
    def test_ttl_expiration(self):
        """Test TTL-based expiration."""
        cache_with_ttl = MemoryCache(max_size=10, ttl=1)  # 1 second TTL
        
        # Set value
        cache_with_ttl.set('key1', 'value1')
        self.assertEqual(cache_with_ttl.get('key1'), 'value1')
        
        # Wait for expiration
        time.sleep(1.1)
        self.assertIsNone(cache_with_ttl.get('key1'))
    
    def test_cache_stats(self):
        """Test cache statistics."""
        # Perform operations
        self.cache.set('key1', 'value1')
        self.cache.get('key1')  # Hit
        self.cache.get('key2')  # Miss
        
        stats = self.cache.stats()
        self.assertEqual(stats.hits, 1)
        self.assertEqual(stats.misses, 1)
        self.assertEqual(stats.size, 1)
    
    def test_clear_cache(self):
        """Test clearing cache."""
        # Add items
        self.cache.set('key1', 'value1')
        self.cache.set('key2', 'value2')
        
        # Clear
        self.assertTrue(self.cache.clear())
        
        # Verify empty
        self.assertIsNone(self.cache.get('key1'))
        self.assertEqual(self.cache.stats().size, 0)


class TestFileCache(unittest.TestCase):
    """Test FileCache implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.cache = FileCache(self.temp_dir, max_size_bytes=1024*1024)  # 1MB
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_basic_operations(self):
        """Test basic file cache operations."""
        # Set and get
        self.assertTrue(self.cache.set('key1', 'value1'))
        self.assertEqual(self.cache.get('key1'), 'value1')
        
        # Get non-existent key
        self.assertIsNone(self.cache.get('key2'))
        
        # Delete key
        self.assertTrue(self.cache.delete('key1'))
        self.assertIsNone(self.cache.get('key1'))
    
    def test_persistence(self):
        """Test cache persistence across instances."""
        # Set value in first cache instance
        self.cache.set('persistent_key', 'persistent_value')
        
        # Create new cache instance with same directory
        new_cache = FileCache(self.temp_dir, max_size_bytes=1024*1024)
        
        # Value should still be accessible
        self.assertEqual(new_cache.get('persistent_key'), 'persistent_value')
    
    def test_complex_data(self):
        """Test caching complex data structures."""
        complex_data = {
            'list': [1, 2, 3],
            'dict': {'nested': True},
            'tuple': (1, 2, 3)
        }
        
        self.assertTrue(self.cache.set('complex', complex_data))
        retrieved = self.cache.get('complex')
        
        self.assertEqual(retrieved, complex_data)
    
    def test_clear_cache(self):
        """Test clearing file cache."""
        # Add items
        self.cache.set('key1', 'value1')
        self.cache.set('key2', 'value2')
        
        # Clear
        self.assertTrue(self.cache.clear())
        
        # Verify empty
        self.assertIsNone(self.cache.get('key1'))


class TestCacheManager(unittest.TestCase):
    """Test CacheManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = {
            'caching': {
                'memory_cache': {
                    'enabled': True,
                    'max_size': '1MB',
                    'ttl': 3600
                },
                'file_cache': {
                    'enabled': True,
                    'cache_dir': str(self.temp_dir),
                    'max_size': '10MB'
                }
            }
        }
        self.cache_manager = CacheManager(self.config)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_multi_cache_operations(self):
        """Test operations across multiple cache backends."""
        # Test memory cache
        self.assertTrue(self.cache_manager.set('mem_key', 'mem_value', 'memory'))
        self.assertEqual(self.cache_manager.get('mem_key', 'memory'), 'mem_value')
        
        # Test file cache
        self.assertTrue(self.cache_manager.set('file_key', 'file_value', 'file'))
        self.assertEqual(self.cache_manager.get('file_key', 'file'), 'file_value')
    
    def test_cache_stats(self):
        """Test getting statistics from all caches."""
        # Add data to caches
        self.cache_manager.set('key1', 'value1', 'memory')
        self.cache_manager.set('key2', 'value2', 'file')
        
        # Get data (generate hits)
        self.cache_manager.get('key1', 'memory')
        self.cache_manager.get('key2', 'file')
        
        # Check stats
        stats = self.cache_manager.stats()
        self.assertIn('memory', stats)
        self.assertIn('file', stats)
        
        # Verify hit counts
        self.assertGreater(stats['memory'].hits, 0)
        self.assertGreater(stats['file'].hits, 0)
    
    def test_clear_all_caches(self):
        """Test clearing all caches."""
        # Add data to multiple caches
        self.cache_manager.set('key1', 'value1', 'memory')
        self.cache_manager.set('key2', 'value2', 'file')
        
        # Clear all
        self.assertTrue(self.cache_manager.clear())
        
        # Verify all cleared
        self.assertIsNone(self.cache_manager.get('key1', 'memory'))
        self.assertIsNone(self.cache_manager.get('key2', 'file'))


class TestPerformanceOptimizer(unittest.TestCase):
    """Test PerformanceOptimizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'optimization': {
                'cpu': {
                    'enabled': True,
                    'parallelization': {'enabled': True, 'max_workers': 'auto'},
                    'computation': {'vectorization': True, 'jit_compilation': True}
                },
                'memory': {
                    'enabled': True,
                    'management': {'garbage_collection_tuning': True}
                },
                'io': {
                    'enabled': True,
                    'file_io': {'async_io': True, 'read_ahead': True}
                }
            }
        }
        self.optimizer = PerformanceOptimizer(self.config)
    
    def test_function_optimization(self):
        """Test function optimization."""
        def test_function(x):
            return x * 2
        
        # Test auto optimization
        optimized_func = self.optimizer.optimize_function(test_function, 'auto')
        self.assertEqual(optimized_func(5), 10)
        
        # Test memoization
        memoized_func = self.optimizer.optimize_function(test_function, 'memoize')
        self.assertEqual(memoized_func(5), 10)
        self.assertEqual(memoized_func(5), 10)  # Should use cached result


class TestPerformanceMonitor(unittest.TestCase):
    """Test PerformanceMonitor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'monitoring': {
                'enabled': True,
                'metrics': {'collection_interval': 0.1},  # Fast for testing
                'thresholds': {
                    'response_time': {'warning': 1000, 'critical': 5000},
                    'cache_hit_rate': {'warning': 0.8, 'critical': 0.6}
                }
            }
        }
        self.monitor = PerformanceMonitor(self.config)
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.monitor.stop_monitoring()
    
    def test_metrics_collection(self):
        """Test metrics collection."""
        # Start monitoring
        self.monitor.start_monitoring()
        
        # Wait for metrics collection
        time.sleep(0.3)
        
        # Stop monitoring
        self.monitor.stop_monitoring()
        
        # Check metrics were collected
        self.assertGreater(len(self.monitor.metrics_history), 0)
        
        # Check metrics structure
        metrics = self.monitor.metrics_history[0]
        self.assertIsInstance(metrics.timestamp, datetime)
        self.assertIsInstance(metrics.cpu_utilization, float)
        self.assertIsInstance(metrics.memory_utilization, float)
    
    def test_metrics_summary(self):
        """Test metrics summary generation."""
        # Add some fake metrics
        for i in range(5):
            metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_utilization=50.0 + i,
                memory_utilization=60.0 + i,
                response_time=200.0 + i * 10,
                throughput=100.0 - i,
                cache_hit_rate=0.85 - i * 0.01,
                error_rate=0.01 + i * 0.001
            )
            self.monitor.metrics_history.append(metrics)
        
        # Get summary
        summary = self.monitor.get_metrics_summary()
        
        # Check summary structure
        self.assertIn('avg_cpu_utilization', summary)
        self.assertIn('avg_memory_utilization', summary)
        self.assertIn('metrics_count', summary)
        
        # Verify averages
        self.assertAlmostEqual(summary['avg_cpu_utilization'], 52.0, places=1)


class TestPerformanceManager(unittest.TestCase):
    """Test PerformanceManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_path = self.temp_dir / 'performance.yml'
        
        # Create test config
        with open(self.config_path, 'w') as f:
            f.write("""
performance:
  caching:
    memory_cache:
      enabled: true
      max_size: "256MB"
      ttl: 3600
    file_cache:
      enabled: true
      cache_dir: "./test_cache"
      max_size: "1GB"
  
  optimization:
    cpu:
      enabled: true
    memory:
      enabled: true
    io:
      enabled: true
  
  monitoring:
    enabled: true
    metrics:
      collection_interval: 1
""")
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_manager_initialization(self):
        """Test manager initialization."""
        manager = PerformanceManager(str(self.config_path))
        
        self.assertIsNotNone(manager.cache_manager)
        self.assertIsNotNone(manager.optimizer)
        self.assertIsNotNone(manager.monitor)
    
    def test_start_stop_manager(self):
        """Test starting and stopping manager."""
        manager = PerformanceManager(str(self.config_path))
        
        # Start manager
        manager.start()
        
        # Stop manager
        manager.stop()
    
    def test_get_status(self):
        """Test getting manager status."""
        manager = PerformanceManager(str(self.config_path))
        
        status = manager.get_status()
        
        self.assertIn('cache_stats', status)
        self.assertIn('performance_metrics', status)
        self.assertIn('config', status)


class TestGlobalFunctions(unittest.TestCase):
    """Test global performance functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Ensure clean state
        shutdown_performance_system()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutdown_performance_system()
    
    def test_initialize_performance_system(self):
        """Test initializing performance system."""
        manager = initialize_performance_system()
        
        self.assertIsNotNone(manager)
        self.assertIsInstance(manager, PerformanceManager)
        
        # Should return same instance on subsequent calls
        manager2 = initialize_performance_system()
        self.assertIs(manager, manager2)
    
    def test_get_performance_manager(self):
        """Test getting performance manager."""
        # Should return None initially
        manager = get_performance_manager()
        self.assertIsNone(manager)
        
        # Initialize and test
        initialize_performance_system()
        manager = get_performance_manager()
        self.assertIsNotNone(manager)
    
    def test_shutdown_performance_system(self):
        """Test shutting down performance system."""
        # Initialize system
        initialize_performance_system()
        manager = get_performance_manager()
        self.assertIsNotNone(manager)
        
        # Shutdown system
        shutdown_performance_system()
        
        # Manager should be None after shutdown
        manager = get_performance_manager()
        self.assertIsNone(manager)


class TestDecorators(unittest.TestCase):
    """Test performance decorators."""
    
    def setUp(self):
        """Set up test fixtures."""
        shutdown_performance_system()
        initialize_performance_system()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutdown_performance_system()
    
    def test_cached_decorator(self):
        """Test cached decorator."""
        call_count = 0
        
        @cached(cache_type='memory', ttl=3600)
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        
        # First call should execute function
        result1 = expensive_function(5)
        self.assertEqual(result1, 10)
        self.assertEqual(call_count, 1)
        
        # Second call should use cache
        result2 = expensive_function(5)
        self.assertEqual(result2, 10)
        self.assertEqual(call_count, 1)  # Should not increment
        
        # Different argument should execute function
        result3 = expensive_function(10)
        self.assertEqual(result3, 20)
        self.assertEqual(call_count, 2)
    
    def test_optimized_decorator(self):
        """Test optimized decorator."""
        @optimized(optimization_type='auto')
        def test_function(x):
            return x * 2
        
        # Function should work normally
        result = test_function(5)
        self.assertEqual(result, 10)


class TestPerformanceIntegration(unittest.TestCase):
    """Integration tests for performance system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_path = self.temp_dir / 'performance.yml'
        
        # Create comprehensive config
        with open(self.config_path, 'w') as f:
            f.write("""
performance:
  caching:
    memory_cache:
      enabled: true
      max_size: "512MB"
      ttl: 3600
    file_cache:
      enabled: true
      cache_dir: "./integration_cache"
      max_size: "2GB"
  
  optimization:
    cpu:
      enabled: true
      parallelization:
        enabled: true
        max_workers: auto
    memory:
      enabled: true
      management:
        garbage_collection_tuning: true
    io:
      enabled: true
      file_io:
        async_io: true
  
  monitoring:
    enabled: true
    metrics:
      collection_interval: 2
    thresholds:
      response_time:
        warning: 1000
        critical: 5000
      cache_hit_rate:
        warning: 0.8
        critical: 0.6
""")
        
        shutdown_performance_system()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutdown_performance_system()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_full_performance_workflow(self):
        """Test complete performance workflow."""
        # Initialize system
        manager = initialize_performance_system(str(self.config_path))
        
        # Test caching
        manager.cache_manager.set('test_key', 'test_value', 'memory')
        cached_value = manager.cache_manager.get('test_key', 'memory')
        self.assertEqual(cached_value, 'test_value')
        
        # Let monitoring run briefly
        time.sleep(3)
        
        # Check status
        status = manager.get_status()
        self.assertIn('cache_stats', status)
        self.assertIn('performance_metrics', status)
        
        # Verify cache stats
        cache_stats = status['cache_stats']
        self.assertIn('memory', cache_stats)
        
        # Shutdown
        shutdown_performance_system()
    
    def test_decorator_integration(self):
        """Test decorator integration with system."""
        initialize_performance_system(str(self.config_path))
        
        call_count = 0
        
        @cached(cache_type='memory')
        @optimized(optimization_type='auto')
        def integrated_function(x, y):
            nonlocal call_count
            call_count += 1
            time.sleep(0.05)  # Reduced sleep time
            return x + y
        
        # Test function with decorators
        result1 = integrated_function(1, 2)
        first_call_count = call_count
        
        result2 = integrated_function(1, 2)  # Should use cache
        second_call_count = call_count
        
        self.assertEqual(result1, 3)
        self.assertEqual(result2, 3)
        # Check that function was only called once due to caching
        self.assertEqual(first_call_count, 1)
        self.assertEqual(second_call_count, 1)  # Should not increment due to cache hit
        
        shutdown_performance_system()


if __name__ == '__main__':
    unittest.main()
