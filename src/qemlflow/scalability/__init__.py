"""
QeMLflow Scalability Module

This module provides horizontal scaling, load balancing, and resource optimization
capabilities for production deployment.
"""

import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ScalingMetrics:
    """Metrics for scaling decisions."""
    timestamp: datetime
    cpu_utilization: float
    memory_utilization: float
    request_rate: float
    response_time: float
    error_rate: float
    queue_length: int
    instance_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'cpu_utilization': self.cpu_utilization,
            'memory_utilization': self.memory_utilization,
            'request_rate': self.request_rate,
            'response_time': self.response_time,
            'error_rate': self.error_rate,
            'queue_length': self.queue_length,
            'instance_count': self.instance_count
        }


@dataclass
class ScalingDecision:
    """Scaling decision result."""
    action: str  # scale_up, scale_down, no_action
    reason: str
    target_instances: int
    confidence: float
    metrics: ScalingMetrics
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'action': self.action,
            'reason': self.reason,
            'target_instances': self.target_instances,
            'confidence': self.confidence,
            'metrics': self.metrics.to_dict()
        }


@dataclass
class InstanceInfo:
    """Information about a service instance."""
    instance_id: str
    host: str
    port: int
    status: str  # healthy, unhealthy, starting, stopping
    cpu_usage: float
    memory_usage: float
    request_count: int
    last_health_check: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'instance_id': self.instance_id,
            'host': self.host,
            'port': self.port,
            'status': self.status,
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'request_count': self.request_count,
            'last_health_check': self.last_health_check.isoformat()
        }


class ScalingStrategy(ABC):
    """Abstract base class for scaling strategies."""
    
    @abstractmethod
    def should_scale(self, metrics: ScalingMetrics, config: Dict[str, Any]) -> ScalingDecision:
        """Determine if scaling is needed."""
        pass


class ReactiveScalingStrategy(ScalingStrategy):
    """Reactive scaling based on current metrics."""
    
    def should_scale(self, metrics: ScalingMetrics, config: Dict[str, Any]) -> ScalingDecision:
        """Determine scaling based on current metrics."""
        try:
            targets = config.get('targets', {})
            thresholds = config.get('thresholds', {})
            
            # Calculate weighted score
            score = 0.0
            weights = {
                'cpu_utilization': 0.4,
                'memory_utilization': 0.3,
                'request_rate': 0.2,
                'response_time': 0.1
            }
            
            for metric, weight in weights.items():
                if metric in targets:
                    target = targets[metric]
                    current = getattr(metrics, metric)
                    
                    if metric == 'response_time':
                        # For response time, higher is worse
                        score += weight * (current / target - 1)
                    else:
                        # For other metrics, higher utilization means need to scale up
                        score += weight * (current / target - 1)
            
            # Determine action based on score
            scale_up_threshold = thresholds.get('scale_up', 0.2)
            scale_down_threshold = thresholds.get('scale_down', -0.3)
            
            if score > scale_up_threshold:
                target_instances = min(
                    metrics.instance_count + max(1, int(score * 2)),
                    config.get('max_instances', 10)
                )
                return ScalingDecision(
                    action='scale_up',
                    reason=f'High resource utilization (score: {score:.2f})',
                    target_instances=target_instances,
                    confidence=min(1.0, abs(score)),
                    metrics=metrics
                )
            elif score < scale_down_threshold:
                target_instances = max(
                    metrics.instance_count - max(1, int(abs(score) * 2)),
                    config.get('min_instances', 1)
                )
                return ScalingDecision(
                    action='scale_down',
                    reason=f'Low resource utilization (score: {score:.2f})',
                    target_instances=target_instances,
                    confidence=min(1.0, abs(score)),
                    metrics=metrics
                )
            else:
                return ScalingDecision(
                    action='no_action',
                    reason=f'Metrics within acceptable range (score: {score:.2f})',
                    target_instances=metrics.instance_count,
                    confidence=1.0 - abs(score),
                    metrics=metrics
                )
        
        except Exception as e:
            logger.error(f"Error in reactive scaling: {e}")
            return ScalingDecision(
                action='no_action',
                reason=f'Error in scaling decision: {e}',
                target_instances=metrics.instance_count,
                confidence=0.0,
                metrics=metrics
            )


class PredictiveScalingStrategy(ScalingStrategy):
    """Predictive scaling based on historical patterns."""
    
    def __init__(self):
        self.historical_metrics: List[ScalingMetrics] = []
        self.max_history = 1000
    
    def add_historical_metrics(self, metrics: ScalingMetrics):
        """Add metrics to historical data."""
        self.historical_metrics.append(metrics)
        if len(self.historical_metrics) > self.max_history:
            self.historical_metrics.pop(0)
    
    def should_scale(self, metrics: ScalingMetrics, config: Dict[str, Any]) -> ScalingDecision:
        """Determine scaling based on predicted future load."""
        try:
            self.add_historical_metrics(metrics)
            
            if len(self.historical_metrics) < 10:
                return ScalingDecision(
                    action='no_action',
                    reason='Insufficient historical data for prediction',
                    target_instances=metrics.instance_count,
                    confidence=0.0,
                    metrics=metrics
                )
            
            # Simple trend analysis
            recent_metrics = self.historical_metrics[-10:]
            cpu_trend = sum(m.cpu_utilization for m in recent_metrics[-5:]) / 5 - \
                       sum(m.cpu_utilization for m in recent_metrics[:5]) / 5
            
            memory_trend = sum(m.memory_utilization for m in recent_metrics[-5:]) / 5 - \
                          sum(m.memory_utilization for m in recent_metrics[:5]) / 5
            
            # Predict next values
            predicted_cpu = metrics.cpu_utilization + cpu_trend * 2
            predicted_memory = metrics.memory_utilization + memory_trend * 2
            
            # Make scaling decision based on predictions
            cpu_threshold = config.get('cpu_threshold', 80)
            memory_threshold = config.get('memory_threshold', 85)
            
            if predicted_cpu > cpu_threshold or predicted_memory > memory_threshold:
                target_instances = min(
                    metrics.instance_count + 1,
                    config.get('max_instances', 10)
                )
                return ScalingDecision(
                    action='scale_up',
                    reason=f'Predicted high utilization (CPU: {predicted_cpu:.1f}%, Memory: {predicted_memory:.1f}%)',
                    target_instances=target_instances,
                    confidence=0.7,
                    metrics=metrics
                )
            elif predicted_cpu < 30 and predicted_memory < 40 and metrics.instance_count > 1:
                target_instances = max(
                    metrics.instance_count - 1,
                    config.get('min_instances', 1)
                )
                return ScalingDecision(
                    action='scale_down',
                    reason=f'Predicted low utilization (CPU: {predicted_cpu:.1f}%, Memory: {predicted_memory:.1f}%)',
                    target_instances=target_instances,
                    confidence=0.6,
                    metrics=metrics
                )
            else:
                return ScalingDecision(
                    action='no_action',
                    reason=f'Predicted utilization within range (CPU: {predicted_cpu:.1f}%, Memory: {predicted_memory:.1f}%)',
                    target_instances=metrics.instance_count,
                    confidence=0.8,
                    metrics=metrics
                )
        
        except Exception as e:
            logger.error(f"Error in predictive scaling: {e}")
            return ScalingDecision(
                action='no_action',
                reason=f'Error in predictive scaling: {e}',
                target_instances=metrics.instance_count,
                confidence=0.0,
                metrics=metrics
            )


class LoadBalancer:
    """Load balancer for distributing requests across instances."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.instances: Dict[str, InstanceInfo] = {}
        self.current_index = 0
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    def add_instance(self, instance: InstanceInfo):
        """Add an instance to the load balancer."""
        with self.lock:
            self.instances[instance.instance_id] = instance
            self.logger.info(f"Added instance {instance.instance_id} to load balancer")
    
    def remove_instance(self, instance_id: str):
        """Remove an instance from the load balancer."""
        with self.lock:
            if instance_id in self.instances:
                del self.instances[instance_id]
                self.logger.info(f"Removed instance {instance_id} from load balancer")
    
    def get_next_instance(self) -> Optional[InstanceInfo]:
        """Get the next instance for request routing."""
        with self.lock:
            healthy_instances = [
                inst for inst in self.instances.values()
                if inst.status == 'healthy'
            ]
            
            if not healthy_instances:
                return None
            
            algorithm = self.config.get('algorithm', 'round_robin')
            
            if algorithm == 'round_robin':
                return self._round_robin(healthy_instances)
            elif algorithm == 'least_connections':
                return self._least_connections(healthy_instances)
            elif algorithm == 'weighted_round_robin':
                return self._weighted_round_robin(healthy_instances)
            else:
                return self._round_robin(healthy_instances)
    
    def _round_robin(self, instances: List[InstanceInfo]) -> InstanceInfo:
        """Round-robin load balancing."""
        instance = instances[self.current_index % len(instances)]
        self.current_index += 1
        return instance
    
    def _least_connections(self, instances: List[InstanceInfo]) -> InstanceInfo:
        """Least connections load balancing."""
        return min(instances, key=lambda x: x.request_count)
    
    def _weighted_round_robin(self, instances: List[InstanceInfo]) -> InstanceInfo:
        """Weighted round-robin based on resource usage."""
        # Weight based on inverse of resource usage
        weights = []
        for instance in instances:
            usage = (instance.cpu_usage + instance.memory_usage) / 2
            weight = max(0.1, 1.0 - usage / 100.0)
            weights.append(weight)
        
        # Select based on weights
        total_weight = sum(weights)
        r = hash(time.time()) % int(total_weight * 100)
        
        current_weight = 0.0
        for i, weight in enumerate(weights):
            current_weight += weight * 100
            if r < current_weight:
                return instances[i]
        
        return instances[-1]
    
    def update_instance_stats(self, instance_id: str, cpu_usage: float, 
                            memory_usage: float, request_count: int):
        """Update instance statistics."""
        with self.lock:
            if instance_id in self.instances:
                self.instances[instance_id].cpu_usage = cpu_usage
                self.instances[instance_id].memory_usage = memory_usage
                self.instances[instance_id].request_count = request_count
                self.instances[instance_id].last_health_check = datetime.now()


class ResourceOptimizer:
    """Resource optimization and management."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._setup_optimization()
    
    def _setup_optimization(self):
        """Setup resource optimization."""
        cpu_config = self.config.get('cpu', {})
        memory_config = self.config.get('memory', {})
        
        # CPU optimization
        if cpu_config.get('process_affinity', False):
            self._optimize_cpu_affinity()
        
        # Memory optimization
        if memory_config.get('optimization_level') == 'aggressive':
            self._optimize_memory_aggressive()
    
    def _optimize_cpu_affinity(self):
        """Optimize CPU affinity for better performance."""
        try:
            # Set CPU affinity to use all available cores
            cpu_count = psutil.cpu_count()
            if cpu_count:
                available_cpus = list(range(cpu_count))
                # Note: os.sched_setaffinity is Linux-specific
                # This would need platform-specific implementation
                self.logger.info(f"Available CPU cores: {available_cpus}")
        except Exception as e:
            self.logger.warning(f"Failed to set CPU affinity: {e}")
    
    def _optimize_memory_aggressive(self):
        """Aggressive memory optimization."""
        try:
            import gc
            # Force garbage collection
            gc.collect()
            
            # Optimize garbage collection thresholds
            gc.set_threshold(700, 10, 10)
            self.logger.info("Applied aggressive memory optimization")
        except Exception as e:
            self.logger.warning(f"Failed to optimize memory: {e}")
    
    def get_resource_usage(self) -> Dict[str, float]:
        """Get current resource usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                'cpu_utilization': cpu_percent,
                'memory_utilization': memory.percent,
                'disk_utilization': (disk.used / disk.total) * 100,
                'memory_available': memory.available,
                'disk_free': disk.free
            }
        except Exception as e:
            self.logger.error(f"Failed to get resource usage: {e}")
            return {
                'cpu_utilization': 0.0,
                'memory_utilization': 0.0,
                'disk_utilization': 0.0,
                'memory_available': 0,
                'disk_free': 0
            }
    
    def optimize_for_workload(self, workload_type: str):
        """Optimize resources for specific workload type."""
        optimizations = {
            'cpu_intensive': self._optimize_for_cpu,
            'memory_intensive': self._optimize_for_memory,
            'io_intensive': self._optimize_for_io,
            'balanced': self._optimize_balanced
        }
        
        if workload_type in optimizations:
            optimizations[workload_type]()
        else:
            self._optimize_balanced()
    
    def _optimize_for_cpu(self):
        """Optimize for CPU-intensive workloads."""
        self.logger.info("Optimizing for CPU-intensive workload")
        # Implementation would include thread pool optimization, etc.
        
    def _optimize_for_memory(self):
        """Optimize for memory-intensive workloads."""
        self.logger.info("Optimizing for memory-intensive workload")
        # Implementation would include memory pool optimization, etc.
        
    def _optimize_for_io(self):
        """Optimize for I/O-intensive workloads."""
        self.logger.info("Optimizing for I/O-intensive workload")
        # Implementation would include async I/O optimization, etc.
        
    def _optimize_balanced(self):
        """Balanced optimization."""
        self.logger.info("Applying balanced optimization")
        # Implementation would include general optimizations


class ScalabilityManager:
    """Main scalability management system."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.reactive_strategy = ReactiveScalingStrategy()
        self.predictive_strategy = PredictiveScalingStrategy()
        self.load_balancer = LoadBalancer(self.config.get('load_balancing', {}))
        self.resource_optimizer = ResourceOptimizer(self.config.get('resource_optimization', {}))
        
        # State
        self.running = False
        self.last_scaling_action = datetime.now()
        self.scaling_cooldown = timedelta(seconds=self.config.get('scaling_cooldown', 300))
        
        # Storage
        self.storage_dir = Path('scalability_data')
        self.storage_dir.mkdir(exist_ok=True)
        
        # Metrics collection
        self.metrics_history: List[ScalingMetrics] = []
        self.max_metrics_history = 1000
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load scalability configuration."""
        if config_path and Path(config_path).exists() and YAML_AVAILABLE:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                return config if isinstance(config, dict) else {}
        
        # Default configuration
        return {
            'horizontal_scaling': {
                'enabled': True,
                'min_instances': 1,
                'max_instances': 10,
                'target_cpu_utilization': 70,
                'target_memory_utilization': 80
            },
            'load_balancing': {
                'enabled': True,
                'algorithm': 'round_robin'
            },
            'resource_optimization': {
                'enabled': True,
                'cpu': {'optimization_level': 'balanced'},
                'memory': {'optimization_level': 'balanced'}
            }
        }
    
    def start(self):
        """Start the scalability manager."""
        if self.running:
            return
        
        self.running = True
        self.logger.info("Starting scalability manager")
        
        # Start monitoring loop
        self._start_monitoring_loop()
    
    def stop(self):
        """Stop the scalability manager."""
        if not self.running:
            return
        
        self.running = False
        self.logger.info("Stopping scalability manager")
    
    def _start_monitoring_loop(self):
        """Start the monitoring loop in a separate thread."""
        def monitoring_loop():
            while self.running:
                try:
                    self._collect_and_analyze_metrics()
                    time.sleep(self.config.get('monitoring_interval', 30))
                except Exception as e:
                    self.logger.error(f"Error in monitoring loop: {e}")
                    time.sleep(30)
        
        thread = threading.Thread(target=monitoring_loop, daemon=True)
        thread.start()
    
    def _collect_and_analyze_metrics(self):
        """Collect metrics and make scaling decisions."""
        try:
            # Collect current metrics
            resource_usage = self.resource_optimizer.get_resource_usage()
            
            metrics = ScalingMetrics(
                timestamp=datetime.now(),
                cpu_utilization=resource_usage['cpu_utilization'],
                memory_utilization=resource_usage['memory_utilization'],
                request_rate=self._get_request_rate(),
                response_time=self._get_response_time(),
                error_rate=self._get_error_rate(),
                queue_length=self._get_queue_length(),
                instance_count=len(self.load_balancer.instances)
            )
            
            # Add to history
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > self.max_metrics_history:
                self.metrics_history.pop(0)
            
            # Make scaling decisions
            self._make_scaling_decisions(metrics)
            
            # Save metrics
            self._save_metrics(metrics)
            
        except Exception as e:
            self.logger.error(f"Error collecting and analyzing metrics: {e}")
    
    def _make_scaling_decisions(self, metrics: ScalingMetrics):
        """Make scaling decisions based on metrics."""
        try:
            # Check cooldown
            if datetime.now() - self.last_scaling_action < self.scaling_cooldown:
                return
            
            # Get scaling decisions from different strategies
            reactive_decision = self.reactive_strategy.should_scale(
                metrics, self.config.get('auto_scaling', {})
            )
            
            predictive_decision = self.predictive_strategy.should_scale(
                metrics, self.config.get('auto_scaling', {})
            )
            
            # Combine decisions (reactive takes precedence)
            final_decision = reactive_decision
            if reactive_decision.action == 'no_action' and predictive_decision.action != 'no_action':
                final_decision = predictive_decision
            
            # Execute scaling action
            if final_decision.action != 'no_action':
                self._execute_scaling_action(final_decision)
            
        except Exception as e:
            self.logger.error(f"Error making scaling decisions: {e}")
    
    def _execute_scaling_action(self, decision: ScalingDecision):
        """Execute the scaling action."""
        try:
            current_instances = len(self.load_balancer.instances)
            target_instances = decision.target_instances
            
            if decision.action == 'scale_up' and target_instances > current_instances:
                instances_to_add = target_instances - current_instances
                self.logger.info(f"Scaling up: adding {instances_to_add} instances")
                self._add_instances(instances_to_add)
                
            elif decision.action == 'scale_down' and target_instances < current_instances:
                instances_to_remove = current_instances - target_instances
                self.logger.info(f"Scaling down: removing {instances_to_remove} instances")
                self._remove_instances(instances_to_remove)
            
            self.last_scaling_action = datetime.now()
            
            # Log decision
            self._log_scaling_decision(decision)
            
        except Exception as e:
            self.logger.error(f"Error executing scaling action: {e}")
    
    def _add_instances(self, count: int):
        """Add new instances."""
        # In a real implementation, this would interact with container orchestration
        # or cloud provider APIs to start new instances
        for i in range(count):
            instance_id = f"instance_{len(self.load_balancer.instances) + i + 1}"
            instance = InstanceInfo(
                instance_id=instance_id,
                host=f"host_{i + 1}",
                port=8000 + i,
                status='starting',
                cpu_usage=0.0,
                memory_usage=0.0,
                request_count=0,
                last_health_check=datetime.now()
            )
            self.load_balancer.add_instance(instance)
    
    def _remove_instances(self, count: int):
        """Remove instances."""
        # In a real implementation, this would gracefully shutdown instances
        instances_to_remove = list(self.load_balancer.instances.keys())[:count]
        for instance_id in instances_to_remove:
            self.load_balancer.remove_instance(instance_id)
    
    def _get_request_rate(self) -> float:
        """Get current request rate."""
        # Placeholder - would integrate with actual request metrics
        return 50.0
    
    def _get_response_time(self) -> float:
        """Get current response time."""
        # Placeholder - would integrate with actual response time metrics
        return 200.0
    
    def _get_error_rate(self) -> float:
        """Get current error rate."""
        # Placeholder - would integrate with actual error metrics
        return 1.0
    
    def _get_queue_length(self) -> int:
        """Get current queue length."""
        # Placeholder - would integrate with actual queue metrics
        return 10
    
    def _save_metrics(self, metrics: ScalingMetrics):
        """Save metrics to storage."""
        try:
            metrics_file = self.storage_dir / f"metrics_{datetime.now().strftime('%Y%m%d')}.jsonl"
            with open(metrics_file, 'a') as f:
                f.write(json.dumps(metrics.to_dict()) + '\n')
        except Exception as e:
            self.logger.error(f"Error saving metrics: {e}")
    
    def _log_scaling_decision(self, decision: ScalingDecision):
        """Log scaling decision."""
        try:
            decision_file = self.storage_dir / f"decisions_{datetime.now().strftime('%Y%m%d')}.jsonl"
            with open(decision_file, 'a') as f:
                f.write(json.dumps(decision.to_dict()) + '\n')
        except Exception as e:
            self.logger.error(f"Error logging scaling decision: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current scalability status."""
        return {
            'running': self.running,
            'instance_count': len(self.load_balancer.instances),
            'instances': [inst.to_dict() for inst in self.load_balancer.instances.values()],
            'resource_usage': self.resource_optimizer.get_resource_usage(),
            'last_scaling_action': self.last_scaling_action.isoformat(),
            'metrics_history_size': len(self.metrics_history)
        }


# Global scalability manager instance
_scalability_manager: Optional[ScalabilityManager] = None


def initialize_scalability_system(config_path: Optional[str] = None) -> ScalabilityManager:
    """Initialize the global scalability system."""
    global _scalability_manager
    if _scalability_manager is None:
        _scalability_manager = ScalabilityManager(config_path)
        _scalability_manager.start()
    return _scalability_manager


def get_scalability_manager() -> Optional[ScalabilityManager]:
    """Get the global scalability manager."""
    return _scalability_manager


def shutdown_scalability_system():
    """Shutdown the global scalability system."""
    global _scalability_manager
    if _scalability_manager:
        _scalability_manager.stop()
        _scalability_manager = None


if __name__ == "__main__":
    # Example usage
    manager = initialize_scalability_system("config/scalability.yml")
    
    try:
        # Run for a while
        time.sleep(300)  # 5 minutes
    finally:
        shutdown_scalability_system()
