"""
Auto-Scaling and Load Balancing Module

This module provides intelligent auto-scaling mechanisms including:
- Horizontal and vertical auto-scaling
- Resource-based scaling triggers
- Load balancing and distribution
- Scaling policies and decision engines
"""

import logging
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import Any, Callable, Dict, List, Optional

import psutil


class ScalingDirection(Enum):
    """Scaling direction enumeration."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"


class ScalingType(Enum):
    """Scaling type enumeration."""
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


@dataclass
class ScalingTrigger:
    """Defines conditions that trigger scaling actions."""
    metric_name: str
    threshold_up: float
    threshold_down: float
    duration_seconds: int = 60
    comparison_type: str = "percent"  # percent, absolute, ratio
    enabled: bool = True


@dataclass
class ScalingAction:
    """Represents a scaling action to be taken."""
    timestamp: float
    direction: ScalingDirection
    scaling_type: ScalingType
    trigger_metric: str
    trigger_value: float
    target_value: int
    reason: str
    executed: bool = False
    execution_time: Optional[float] = None
    result: Optional[Dict[str, Any]] = None


@dataclass
class ResourceMetrics:
    """Current resource metrics for scaling decisions."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    network_io_mbps: float = 0.0
    disk_io_mbps: float = 0.0
    active_connections: int = 0
    request_rate: float = 0.0
    response_time_ms: float = 0.0
    error_rate_percent: float = 0.0
    custom_metrics: Dict[str, float] = field(default_factory=dict)


class ScalingPolicy:
    """
    Defines scaling policies and decision logic.
    """
    
    def __init__(self, 
                 name: str,
                 min_instances: int = 1,
                 max_instances: int = 10,
                 target_cpu_percent: float = 70.0,
                 target_memory_percent: float = 80.0,
                 scale_up_cooldown: int = 300,  # 5 minutes
                 scale_down_cooldown: int = 600):  # 10 minutes
        self.name = name
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.target_cpu_percent = target_cpu_percent
        self.target_memory_percent = target_memory_percent
        self.scale_up_cooldown = scale_up_cooldown
        self.scale_down_cooldown = scale_down_cooldown
        
        self.logger = logging.getLogger(__name__)
        
        # Scaling triggers
        self.triggers: List[ScalingTrigger] = [
            ScalingTrigger("cpu_percent", target_cpu_percent + 10, target_cpu_percent - 10),
            ScalingTrigger("memory_percent", target_memory_percent + 10, target_memory_percent - 10)
        ]
        
        # Scaling history
        self.scaling_history: List[ScalingAction] = []
        self._lock = Lock()
    
    def add_trigger(self, trigger: ScalingTrigger) -> None:
        """Add a custom scaling trigger."""
        self.triggers.append(trigger)
        self.logger.info(f"Added scaling trigger for {trigger.metric_name}")
    
    def evaluate_scaling_decision(self, 
                                 metrics: ResourceMetrics, 
                                 current_instances: int) -> Optional[ScalingAction]:
        """Evaluate if scaling action is needed based on current metrics."""
        
        # Check cooldown periods
        last_action = self._get_last_scaling_action()
        if last_action and not self._is_cooldown_expired(last_action):
            return None
        
        # Evaluate each trigger
        for trigger in self.triggers:
            if not trigger.enabled:
                continue
            
            metric_value = self._get_metric_value(metrics, trigger.metric_name)
            if metric_value is None:
                continue
            
            # Check for scale up condition
            if metric_value > trigger.threshold_up and current_instances < self.max_instances:
                return ScalingAction(
                    timestamp=time.time(),
                    direction=ScalingDirection.SCALE_UP,
                    scaling_type=ScalingType.HORIZONTAL,
                    trigger_metric=trigger.metric_name,
                    trigger_value=metric_value,
                    target_value=current_instances + 1,
                    reason=f"{trigger.metric_name} ({metric_value:.1f}%) above threshold ({trigger.threshold_up:.1f}%)"
                )
            
            # Check for scale down condition
            elif metric_value < trigger.threshold_down and current_instances > self.min_instances:
                return ScalingAction(
                    timestamp=time.time(),
                    direction=ScalingDirection.SCALE_DOWN,
                    scaling_type=ScalingType.HORIZONTAL,
                    trigger_metric=trigger.metric_name,
                    trigger_value=metric_value,
                    target_value=current_instances - 1,
                    reason=f"{trigger.metric_name} ({metric_value:.1f}%) below threshold ({trigger.threshold_down:.1f}%)"
                )
        
        return None
    
    def _get_metric_value(self, metrics: ResourceMetrics, metric_name: str) -> Optional[float]:
        """Get metric value from metrics object."""
        if hasattr(metrics, metric_name):
            return getattr(metrics, metric_name)
        elif metric_name in metrics.custom_metrics:
            return metrics.custom_metrics[metric_name]
        else:
            return None
    
    def _get_last_scaling_action(self) -> Optional[ScalingAction]:
        """Get the most recent scaling action."""
        with self._lock:
            if self.scaling_history:
                return self.scaling_history[-1]
            return None
    
    def _is_cooldown_expired(self, last_action: ScalingAction) -> bool:
        """Check if cooldown period has expired."""
        current_time = time.time()
        
        if last_action.direction == ScalingDirection.SCALE_UP:
            cooldown = self.scale_up_cooldown
        else:
            cooldown = self.scale_down_cooldown
        
        return (current_time - last_action.timestamp) > cooldown
    
    def record_scaling_action(self, action: ScalingAction) -> None:
        """Record a scaling action in history."""
        with self._lock:
            self.scaling_history.append(action)
            # Keep only last 100 actions
            if len(self.scaling_history) > 100:
                self.scaling_history.pop(0)
    
    def get_scaling_statistics(self) -> Dict[str, Any]:
        """Get scaling statistics and history."""
        with self._lock:
            total_actions = len(self.scaling_history)
            scale_up_actions = sum(1 for action in self.scaling_history 
                                 if action.direction == ScalingDirection.SCALE_UP)
            scale_down_actions = total_actions - scale_up_actions
            
            recent_actions = self.scaling_history[-10:] if self.scaling_history else []
            
            return {
                "policy_name": self.name,
                "total_scaling_actions": total_actions,
                "scale_up_actions": scale_up_actions,
                "scale_down_actions": scale_down_actions,
                "recent_actions": [
                    {
                        "timestamp": action.timestamp,
                        "direction": action.direction.value,
                        "trigger": action.trigger_metric,
                        "reason": action.reason,
                        "executed": action.executed
                    }
                    for action in recent_actions
                ],
                "current_config": {
                    "min_instances": self.min_instances,
                    "max_instances": self.max_instances,
                    "target_cpu_percent": self.target_cpu_percent,
                    "target_memory_percent": self.target_memory_percent
                }
            }


class LoadBalancer:
    """
    Intelligent load balancer for distributing workload across instances.
    """
    
    def __init__(self, balancing_strategy: str = "round_robin"):
        self.balancing_strategy = balancing_strategy
        self.instances: List[Dict[str, Any]] = []
        self.current_index = 0
        self.logger = logging.getLogger(__name__)
        self._lock = Lock()
        
        # Instance health tracking
        self.health_checks: Dict[str, Dict[str, Any]] = {}
    
    def add_instance(self, instance_id: str, endpoint: str, weight: float = 1.0) -> None:
        """Add an instance to the load balancer."""
        with self._lock:
            instance = {
                "id": instance_id,
                "endpoint": endpoint,
                "weight": weight,
                "active": True,
                "connections": 0,
                "response_time_ms": 0.0,
                "error_count": 0,
                "last_request": None
            }
            self.instances.append(instance)
            self.health_checks[instance_id] = {
                "healthy": True,
                "last_check": time.time(),
                "consecutive_failures": 0
            }
        self.logger.info(f"Added instance {instance_id} to load balancer")
    
    def remove_instance(self, instance_id: str) -> bool:
        """Remove an instance from the load balancer."""
        with self._lock:
            for i, instance in enumerate(self.instances):
                if instance["id"] == instance_id:
                    del self.instances[i]
                    if instance_id in self.health_checks:
                        del self.health_checks[instance_id]
                    self.logger.info(f"Removed instance {instance_id} from load balancer")
                    return True
            return False
    
    def get_next_instance(self) -> Optional[Dict[str, Any]]:
        """Get the next instance based on balancing strategy."""
        with self._lock:
            healthy_instances = [inst for inst in self.instances 
                               if inst["active"] and self.health_checks.get(inst["id"], {}).get("healthy", False)]
            
            if not healthy_instances:
                return None
            
            if self.balancing_strategy == "round_robin":
                instance = healthy_instances[self.current_index % len(healthy_instances)]
                self.current_index += 1
                return instance
            
            elif self.balancing_strategy == "least_connections":
                return min(healthy_instances, key=lambda x: x["connections"])
            
            elif self.balancing_strategy == "weighted_round_robin":
                # Implement weighted selection based on instance weights
                total_weight = sum(inst["weight"] for inst in healthy_instances)
                if total_weight == 0:
                    return healthy_instances[0]
                
                # Simple weighted selection (can be improved with more sophisticated algorithm)
                import random
                weight_sum = 0
                random_value = random.random() * total_weight
                
                for instance in healthy_instances:
                    weight_sum += instance["weight"]
                    if random_value <= weight_sum:
                        return instance
                
                return healthy_instances[-1]
            
            elif self.balancing_strategy == "fastest_response":
                return min(healthy_instances, key=lambda x: x["response_time_ms"])
            
            else:
                # Default to round robin
                return healthy_instances[self.current_index % len(healthy_instances)]
    
    def update_instance_metrics(self, instance_id: str, 
                              response_time_ms: float, 
                              success: bool) -> None:
        """Update instance performance metrics."""
        with self._lock:
            for instance in self.instances:
                if instance["id"] == instance_id:
                    instance["response_time_ms"] = response_time_ms
                    instance["last_request"] = time.time()
                    
                    if success:
                        # Reset error count on success
                        if instance["error_count"] > 0:
                            instance["error_count"] = max(0, instance["error_count"] - 1)
                        
                        # Update health status
                        if instance_id in self.health_checks:
                            self.health_checks[instance_id]["consecutive_failures"] = 0
                            self.health_checks[instance_id]["healthy"] = True
                    else:
                        instance["error_count"] += 1
                        
                        # Update health status
                        if instance_id in self.health_checks:
                            self.health_checks[instance_id]["consecutive_failures"] += 1
                            # Mark unhealthy after 3 consecutive failures
                            if self.health_checks[instance_id]["consecutive_failures"] >= 3:
                                self.health_checks[instance_id]["healthy"] = False
                    
                    break
    
    def get_load_distribution(self) -> Dict[str, Any]:
        """Get current load distribution across instances."""
        with self._lock:
            total_connections = sum(inst["connections"] for inst in self.instances)
            
            distribution = []
            for instance in self.instances:
                load_percent = (instance["connections"] / total_connections * 100) if total_connections > 0 else 0
                
                distribution.append({
                    "instance_id": instance["id"],
                    "endpoint": instance["endpoint"],
                    "connections": instance["connections"],
                    "load_percent": round(load_percent, 1),
                    "response_time_ms": instance["response_time_ms"],
                    "error_count": instance["error_count"],
                    "healthy": self.health_checks.get(instance["id"], {}).get("healthy", False)
                })
            
            return {
                "total_instances": len(self.instances),
                "healthy_instances": sum(1 for inst in self.instances 
                                       if self.health_checks.get(inst["id"], {}).get("healthy", False)),
                "total_connections": total_connections,
                "distribution": distribution,
                "strategy": self.balancing_strategy
            }


class AutoScaler:
    """
    Comprehensive auto-scaling system combining policies and load balancing.
    """
    
    def __init__(self, 
                 scaling_policy: ScalingPolicy,
                 load_balancer: Optional[LoadBalancer] = None,
                 monitoring_interval: float = 30.0):
        self.scaling_policy = scaling_policy
        self.load_balancer = load_balancer or LoadBalancer()
        self.monitoring_interval = monitoring_interval
        
        self.logger = logging.getLogger(__name__)
        self.current_instances = scaling_policy.min_instances
        
        # Monitoring and control
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        
        # Callbacks for scaling actions
        self.scale_up_callback: Optional[Callable[[int], bool]] = None
        self.scale_down_callback: Optional[Callable[[int], bool]] = None
    
    def set_scaling_callbacks(self, 
                            scale_up_func: Callable[[int], bool],
                            scale_down_func: Callable[[int], bool]) -> None:
        """Set callbacks for actual scaling implementation."""
        self.scale_up_callback = scale_up_func
        self.scale_down_callback = scale_down_func
    
    def collect_metrics(self) -> ResourceMetrics:
        """Collect current resource metrics for scaling decisions."""
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        
        # Network I/O (simplified)
        try:
            network_io = psutil.net_io_counters()
            network_io_mbps = (network_io.bytes_sent + network_io.bytes_recv) / (1024 * 1024)
        except Exception:
            network_io_mbps = 0.0
        
        # Disk I/O (simplified)
        try:
            disk_io = psutil.disk_io_counters()
            disk_io_mbps = (disk_io.read_bytes + disk_io.write_bytes) / (1024 * 1024)
        except Exception:
            disk_io_mbps = 0.0
        
        return ResourceMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            network_io_mbps=network_io_mbps,
            disk_io_mbps=disk_io_mbps
        )
    
    def evaluate_and_execute_scaling(self) -> Optional[ScalingAction]:
        """Evaluate metrics and execute scaling decision if needed."""
        metrics = self.collect_metrics()
        scaling_action = self.scaling_policy.evaluate_scaling_decision(metrics, self.current_instances)
        
        if scaling_action:
            # Execute the scaling action
            success = self._execute_scaling_action(scaling_action)
            
            scaling_action.executed = success
            scaling_action.execution_time = time.time()
            
            # Record the action
            self.scaling_policy.record_scaling_action(scaling_action)
            
            if success:
                self.current_instances = scaling_action.target_value
                self.logger.info(f"Scaling action executed: {scaling_action.reason}")
            else:
                self.logger.error(f"Scaling action failed: {scaling_action.reason}")
        
        return scaling_action
    
    def _execute_scaling_action(self, action: ScalingAction) -> bool:
        """Execute the actual scaling action."""
        try:
            if action.direction == ScalingDirection.SCALE_UP:
                if self.scale_up_callback:
                    return self.scale_up_callback(action.target_value)
                else:
                    # Simulate scale up
                    self.logger.info(f"Simulated scale up to {action.target_value} instances")
                    return True
            
            elif action.direction == ScalingDirection.SCALE_DOWN:
                if self.scale_down_callback:
                    return self.scale_down_callback(action.target_value)
                else:
                    # Simulate scale down
                    self.logger.info(f"Simulated scale down to {action.target_value} instances")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Scaling execution failed: {e}")
            return False
    
    def start_monitoring(self) -> None:
        """Start automatic scaling monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._stop_monitoring.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        self.logger.info("Auto-scaling monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop automatic scaling monitoring."""
        if not self._monitoring:
            return
        
        self._stop_monitoring.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        self._monitoring = False
        self.logger.info("Auto-scaling monitoring stopped")
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while not self._stop_monitoring.wait(self.monitoring_interval):
            try:
                self.evaluate_and_execute_scaling()
            except Exception as e:
                self.logger.error(f"Error in auto-scaling monitoring: {e}")
    
    def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive auto-scaling status report."""
        metrics = self.collect_metrics()
        
        return {
            "timestamp": time.time(),
            "current_instances": self.current_instances,
            "target_instances": self.scaling_policy.min_instances,
            "monitoring_active": self._monitoring,
            "current_metrics": {
                "cpu_percent": metrics.cpu_percent,
                "memory_percent": metrics.memory_percent,
                "network_io_mbps": metrics.network_io_mbps,
                "disk_io_mbps": metrics.disk_io_mbps
            },
            "scaling_policy": self.scaling_policy.get_scaling_statistics(),
            "load_balancer": self.load_balancer.get_load_distribution()
        }
    
    def __del__(self):
        """Cleanup on destruction."""
        self.stop_monitoring()


# Convenience functions
def create_default_scaling_policy(name: str = "default") -> ScalingPolicy:
    """Create a default scaling policy with sensible defaults."""
    return ScalingPolicy(
        name=name,
        min_instances=1,
        max_instances=5,
        target_cpu_percent=70.0,
        target_memory_percent=80.0
    )


def create_auto_scaler(policy_name: str = "default", 
                      load_balancing_strategy: str = "round_robin") -> AutoScaler:
    """Create an auto-scaler with default configuration."""
    policy = create_default_scaling_policy(policy_name)
    load_balancer = LoadBalancer(load_balancing_strategy)
    return AutoScaler(policy, load_balancer)
