"""
Production Performance Tuning Module for QeMLflow

This module provides production-specific performance tuning capabilities including:
- Resource optimization
- Performance monitoring
- Automatic tuning based on metrics
- Production environment configuration
"""

import gc
import os
import psutil
import logging
import threading
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    memory_available: float
    disk_usage: float
    network_io: Dict[str, float]
    active_connections: int
    response_time: float
    throughput: float
    error_rate: float


class ProductionPerformanceTuner:
    """Production performance tuning system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize performance tuner."""
        self.config = config or {}
        self.performance_config = self.config.get('performance', {})
        self.monitoring_config = self.config.get('monitoring', {})
        
        # Performance thresholds
        self.cpu_threshold = self.performance_config.get('cpu', {}).get('max_cpu_usage', 0.85)
        self.memory_threshold = self.performance_config.get('memory', {}).get('gc_threshold', 0.8)
        self.io_threshold = self.performance_config.get('io', {}).get('max_concurrent_io', 100)
        
        # Monitoring state
        self.metrics_history: List[PerformanceMetrics] = []
        self.is_monitoring = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.tuning_enabled = True
        
        # Performance optimizations
        self.gc_enabled = self.performance_config.get('memory', {}).get('enable_monitoring', True)
        self.cpu_affinity_enabled = self.performance_config.get('cpu', {}).get('enable_affinity', True)
        
        logger.info("Production performance tuner initialized")
    
    def start_monitoring(self) -> None:
        """Start performance monitoring."""
        if self.is_monitoring:
            logger.warning("Performance monitoring already running")
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        interval = self.monitoring_config.get('metrics', {}).get('interval', 15)
        
        while self.is_monitoring:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Keep only recent metrics (last hour)
                cutoff_time = datetime.now() - timedelta(hours=1)
                self.metrics_history = [
                    m for m in self.metrics_history 
                    if m.timestamp > cutoff_time
                ]
                
                # Perform automatic tuning
                if self.tuning_enabled:
                    self._auto_tune(metrics)
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval)
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        try:
            # CPU metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_usage = memory.percent / 100.0
            memory_available = memory.available
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_usage = disk.percent / 100.0
            
            # Network metrics
            network = psutil.net_io_counters()
            network_io = {
                'bytes_sent': float(network.bytes_sent),
                'bytes_recv': float(network.bytes_recv),
                'packets_sent': float(network.packets_sent),
                'packets_recv': float(network.packets_recv)
            }
            
            # Connection count
            connections = len(psutil.net_connections())
            
            # Application-specific metrics (simplified)
            response_time = self._calculate_response_time()
            throughput = self._calculate_throughput()
            error_rate = self._calculate_error_rate()
            
            return PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_usage=cpu_usage / 100.0,
                memory_usage=memory_usage,
                memory_available=memory_available,
                disk_usage=disk_usage,
                network_io=network_io,
                active_connections=connections,
                response_time=response_time,
                throughput=throughput,
                error_rate=error_rate
            )
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            return PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_usage=0.0,
                memory_usage=0.0,
                memory_available=0,
                disk_usage=0.0,
                network_io={},
                active_connections=0,
                response_time=0.0,
                throughput=0.0,
                error_rate=0.0
            )
    
    def _calculate_response_time(self) -> float:
        """Calculate average response time (simplified)."""
        # In production, this would integrate with your application metrics
        return 0.5  # Placeholder
    
    def _calculate_throughput(self) -> float:
        """Calculate requests per second (simplified)."""
        # In production, this would integrate with your application metrics
        return 100.0  # Placeholder
    
    def _calculate_error_rate(self) -> float:
        """Calculate error rate (simplified)."""
        # In production, this would integrate with your application metrics
        return 0.01  # Placeholder
    
    def _auto_tune(self, metrics: PerformanceMetrics) -> None:
        """Perform automatic performance tuning based on metrics."""
        try:
            # Memory optimization
            if metrics.memory_usage > self.memory_threshold:
                self._optimize_memory()
            
            # CPU optimization
            if metrics.cpu_usage > self.cpu_threshold:
                self._optimize_cpu()
            
            # I/O optimization
            if metrics.active_connections > self.io_threshold:
                self._optimize_io()
            
            # Garbage collection optimization
            if self.gc_enabled and metrics.memory_usage > 0.9:
                self._force_garbage_collection()
                
        except Exception as e:
            logger.error(f"Error during auto-tuning: {e}")
    
    def _optimize_memory(self) -> None:
        """Optimize memory usage."""
        logger.info("Optimizing memory usage")
        
        # Force garbage collection
        if self.gc_enabled:
            collected = gc.collect()
            logger.info(f"Garbage collection freed {collected} objects")
        
        # Additional memory optimizations could go here
        # - Clear caches
        # - Reduce buffer sizes
        # - Optimize data structures
    
    def _optimize_cpu(self) -> None:
        """Optimize CPU usage."""
        logger.info("Optimizing CPU usage")
        
        # Set CPU affinity if enabled
        if self.cpu_affinity_enabled:
            try:
                # Distribute across available CPUs
                cpu_count = psutil.cpu_count()
                if cpu_count and cpu_count > 1 and hasattr(os, 'sched_setaffinity'):
                    os.sched_setaffinity(0, range(cpu_count))
            except (AttributeError, OSError):
                # CPU affinity not supported on this platform
                pass
        
        # Additional CPU optimizations could go here
        # - Adjust thread pool sizes
        # - Optimize algorithms
        # - Reduce computational complexity
    
    def _optimize_io(self) -> None:
        """Optimize I/O operations."""
        logger.info("Optimizing I/O operations")
        
        # Additional I/O optimizations could go here
        # - Adjust buffer sizes
        # - Batch operations
        # - Use async I/O
        # - Connection pooling
    
    def _force_garbage_collection(self) -> None:
        """Force garbage collection to free memory."""
        try:
            # Collect all generations
            collected = 0
            for generation in range(3):
                collected += gc.collect(generation)
            
            logger.info(f"Forced garbage collection freed {collected} objects")
            
        except Exception as e:
            logger.error(f"Error during garbage collection: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.metrics_history:
            return {
                'status': 'no_data',
                'message': 'No performance data available'
            }
        
        recent_metrics = self.metrics_history[-10:]  # Last 10 measurements
        
        avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
        avg_response_time = sum(m.response_time for m in recent_metrics) / len(recent_metrics)
        avg_throughput = sum(m.throughput for m in recent_metrics) / len(recent_metrics)
        avg_error_rate = sum(m.error_rate for m in recent_metrics) / len(recent_metrics)
        
        # Determine overall status
        status = 'healthy'
        issues = []
        
        if avg_cpu > self.cpu_threshold:
            status = 'warning'
            issues.append(f'High CPU usage: {avg_cpu:.1%}')
        
        if avg_memory > self.memory_threshold:
            status = 'warning'
            issues.append(f'High memory usage: {avg_memory:.1%}')
        
        if avg_error_rate > 0.05:  # 5%
            status = 'critical'
            issues.append(f'High error rate: {avg_error_rate:.1%}')
        
        return {
            'status': status,
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'cpu_usage': avg_cpu,
                'memory_usage': avg_memory,
                'response_time': avg_response_time,
                'throughput': avg_throughput,
                'error_rate': avg_error_rate
            },
            'issues': issues,
            'recommendations': self._get_recommendations(avg_cpu, avg_memory, avg_error_rate)
        }
    
    def _get_recommendations(self, cpu_usage: float, memory_usage: float, error_rate: float) -> List[str]:
        """Get performance recommendations."""
        recommendations = []
        
        if cpu_usage > 0.8:
            recommendations.append("Consider horizontal scaling or CPU optimization")
        
        if memory_usage > 0.8:
            recommendations.append("Consider increasing memory limits or optimizing memory usage")
        
        if error_rate > 0.05:
            recommendations.append("Investigate error sources and improve error handling")
        
        return recommendations
    
    def apply_production_optimizations(self) -> Dict[str, Any]:
        """Apply production-specific optimizations."""
        logger.info("Applying production optimizations")
        
        optimizations_applied = []
        
        try:
            # Memory optimizations
            if self.gc_enabled:
                # Set garbage collection thresholds for production
                gc.set_threshold(700, 10, 10)  # More aggressive collection
                optimizations_applied.append("Optimized garbage collection thresholds")
            
            # CPU optimizations
            if self.cpu_affinity_enabled:
                try:
                    # Set CPU affinity to use all available cores
                    cpu_count = psutil.cpu_count()
                    if cpu_count and hasattr(os, 'sched_setaffinity'):
                        os.sched_setaffinity(0, range(cpu_count))
                        optimizations_applied.append(f"Set CPU affinity to use all {cpu_count} cores")
                except (AttributeError, OSError) as e:
                    logger.warning(f"Could not set CPU affinity: {e}")
            
            # Environment variables for optimization
            os.environ['PYTHONOPTIMIZE'] = '1'  # Enable optimizations
            os.environ['PYTHONDONTWRITEBYTECODE'] = '1'  # Don't write .pyc files
            optimizations_applied.append("Set Python optimization environment variables")
            
            return {
                'status': 'success',
                'optimizations': optimizations_applied,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error applying production optimizations: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'optimizations': optimizations_applied,
                'timestamp': datetime.now().isoformat()
            }
    
    def validate_production_readiness(self) -> Dict[str, Any]:
        """Validate production readiness from performance perspective."""
        checks = {
            'memory_monitoring': self.gc_enabled,
            'cpu_optimization': self.cpu_affinity_enabled,
            'performance_thresholds': self.cpu_threshold < 1.0 and self.memory_threshold < 1.0,
            'monitoring_enabled': self.is_monitoring,
            'tuning_enabled': self.tuning_enabled
        }
        
        passed_checks = sum(checks.values())
        total_checks = len(checks)
        
        return {
            'ready': passed_checks == total_checks,
            'score': passed_checks / total_checks,
            'checks': checks,
            'summary': f"{passed_checks}/{total_checks} production readiness checks passed"
        }


# Utility functions for production performance tuning

def initialize_production_performance(config: Optional[Dict[str, Any]] = None) -> ProductionPerformanceTuner:
    """Initialize production performance tuning."""
    tuner = ProductionPerformanceTuner(config)
    tuner.apply_production_optimizations()
    tuner.start_monitoring()
    return tuner


def get_production_performance_status(tuner: ProductionPerformanceTuner) -> Dict[str, Any]:
    """Get current production performance status."""
    return tuner.get_performance_summary()


def validate_production_performance(tuner: ProductionPerformanceTuner) -> Dict[str, Any]:
    """Validate production performance readiness."""
    return tuner.validate_production_readiness()


# Create alias for easier import
ProductionTuner = ProductionPerformanceTuner

# Export public API
__all__ = [
    'ProductionPerformanceTuner',
    'ProductionTuner',  # Alias
    'PerformanceMetrics',
    'initialize_production_performance',
    'get_production_performance_status',
    'validate_production_performance'
]
