"""
Compute Resource Management Module

This module provides advanced compute resource optimization including:
- CPU usage optimization and monitoring
- GPU resource management and allocation
- Compute workload balancing
- Performance optimization and tuning
"""

import logging
import multiprocessing
import platform
import time
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Tuple

import psutil

# Try to import GPU libraries (optional)
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    GPUtil = None


@dataclass
class ComputeSnapshot:
    """Represents compute resource usage at a specific point in time."""
    timestamp: float
    cpu_count: int
    cpu_usage_percent: float
    cpu_freq: Dict[str, float]
    cpu_load_avg: List[float]
    cpu_times: Dict[str, float]
    memory_info: Dict[str, float]
    disk_io: Dict[str, int] = field(default_factory=dict)
    network_io: Dict[str, int] = field(default_factory=dict)
    gpu_info: List[Dict[str, Any]] = field(default_factory=list)


class CPUOptimizer:
    """
    CPU resource optimization and monitoring.
    """
    
    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.logger = logging.getLogger(__name__)
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        self.snapshots: List[ComputeSnapshot] = []
        self._lock = Lock()
        
        # CPU information
        self.cpu_count = psutil.cpu_count()
        self.cpu_count_logical = psutil.cpu_count(logical=True)
        self.cpu_freq_max = psutil.cpu_freq().max if psutil.cpu_freq() else None
    
    def take_snapshot(self) -> ComputeSnapshot:
        """Take a comprehensive CPU and system snapshot."""
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        cpu_freq = psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {}
        cpu_times = psutil.cpu_times()._asdict()
        
        # Load average (Unix-like systems only)
        try:
            load_avg = list(psutil.getloadavg())
        except (AttributeError, OSError):
            load_avg = []
        
        # Memory info
        memory = psutil.virtual_memory()
        memory_info = {
            "total_gb": round(memory.total / (1024**3), 2),
            "available_gb": round(memory.available / (1024**3), 2),
            "percent": memory.percent
        }
        
        # Disk I/O
        try:
            disk_io = psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else {}
        except Exception:
            disk_io = {}
        
        # Network I/O
        try:
            network_io = psutil.net_io_counters()._asdict() if psutil.net_io_counters() else {}
        except Exception:
            network_io = {}
        
        snapshot = ComputeSnapshot(
            timestamp=time.time(),
            cpu_count=self.cpu_count,
            cpu_usage_percent=cpu_percent,
            cpu_freq=cpu_freq,
            cpu_load_avg=load_avg,
            cpu_times=cpu_times,
            memory_info=memory_info,
            disk_io=disk_io,
            network_io=network_io
        )
        
        with self._lock:
            self.snapshots.append(snapshot)
            # Keep only last 100 snapshots
            if len(self.snapshots) > 100:
                self.snapshots.pop(0)
        
        return snapshot
    
    def get_optimal_worker_count(self, workload_type: str = "cpu_bound") -> int:
        """Get optimal number of workers for different workload types."""
        if workload_type == "cpu_bound":
            # For CPU-bound tasks, use number of physical cores
            return self.cpu_count
        elif workload_type == "io_bound":
            # For I/O-bound tasks, can use more workers
            return min(self.cpu_count_logical * 2, 32)
        elif workload_type == "mixed":
            # For mixed workloads, use a balanced approach
            return max(self.cpu_count, self.cpu_count_logical)
        else:
            # Default to logical CPU count
            return self.cpu_count_logical
    
    def optimize_process_affinity(self, process_id: Optional[int] = None) -> Dict[str, Any]:
        """Optimize CPU affinity for better performance."""
        try:
            process = psutil.Process(process_id) if process_id else psutil.Process()
            
            # Get current affinity
            current_affinity = process.cpu_affinity()
            
            # Recommend optimal affinity based on CPU usage
            snapshot = self.take_snapshot()
            
            if snapshot.cpu_usage_percent > 80:
                # High CPU usage - spread across all cores
                optimal_affinity = list(range(self.cpu_count_logical))
            else:
                # Normal usage - use physical cores for better performance
                optimal_affinity = list(range(self.cpu_count))
            
            # Apply optimization if different
            if set(current_affinity) != set(optimal_affinity):
                process.cpu_affinity(optimal_affinity)
                
                return {
                    "optimized": True,
                    "previous_affinity": current_affinity,
                    "new_affinity": optimal_affinity,
                    "performance_impact": "improved_cache_efficiency"
                }
            else:
                return {
                    "optimized": False,
                    "current_affinity": current_affinity,
                    "status": "already_optimal"
                }
                
        except Exception as e:
            self.logger.error(f"Failed to optimize process affinity: {e}")
            return {"error": str(e)}
    
    def get_performance_recommendations(self) -> List[str]:
        """Get performance optimization recommendations."""
        recommendations = []
        
        if not self.snapshots:
            self.take_snapshot()
        
        latest = self.snapshots[-1]
        
        # CPU usage recommendations
        if latest.cpu_usage_percent > 90:
            recommendations.append("CPU usage is very high (>90%). Consider scaling horizontally.")
        elif latest.cpu_usage_percent > 70:
            recommendations.append("CPU usage is high (>70%). Monitor for performance bottlenecks.")
        
        # Load average recommendations (Unix-like systems)
        if latest.cpu_load_avg:
            load_per_core = latest.cpu_load_avg[0] / self.cpu_count
            if load_per_core > 1.5:
                recommendations.append(f"System load is high ({load_per_core:.1f} per core). Consider load balancing.")
        
        # Memory recommendations
        if latest.memory_info["percent"] > 85:
            recommendations.append("Memory usage is high (>85%). This may impact CPU performance.")
        
        # Frequency recommendations
        if latest.cpu_freq and self.cpu_freq_max:
            current_freq = latest.cpu_freq.get("current", 0)
            if current_freq < self.cpu_freq_max * 0.8:
                recommendations.append("CPU frequency is below maximum. Check power management settings.")
        
        if not recommendations:
            recommendations.append("CPU performance appears optimal.")
        
        return recommendations
    
    def start_monitoring(self) -> None:
        """Start continuous CPU monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._stop_monitoring.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        self.logger.info("CPU monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop CPU monitoring."""
        if not self._monitoring:
            return
        
        self._stop_monitoring.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        self._monitoring = False
        self.logger.info("CPU monitoring stopped")
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while not self._stop_monitoring.wait(self.monitoring_interval):
            try:
                self.take_snapshot()
            except Exception as e:
                self.logger.error(f"Error in CPU monitoring: {e}")


class GPUManager:
    """
    GPU resource management and monitoring.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.gpu_available = GPU_AVAILABLE
        self._gpu_info: List[Dict[str, Any]] = []
        
        if self.gpu_available:
            self._initialize_gpu_info()
    
    def _initialize_gpu_info(self) -> None:
        """Initialize GPU information."""
        try:
            gpus = GPUtil.getGPUs()
            self._gpu_info = []
            
            for gpu in gpus:
                gpu_info = {
                    "id": gpu.id,
                    "name": gpu.name,
                    "driver": gpu.driver,
                    "memory_total": gpu.memoryTotal,
                    "memory_used": gpu.memoryUsed,
                    "memory_free": gpu.memoryFree,
                    "memory_util": gpu.memoryUtil,
                    "gpu_util": gpu.load,
                    "temperature": gpu.temperature
                }
                self._gpu_info.append(gpu_info)
                
        except Exception as e:
            self.logger.error(f"Failed to initialize GPU info: {e}")
            self.gpu_available = False
    
    def get_gpu_status(self) -> Dict[str, Any]:
        """Get current GPU status."""
        if not self.gpu_available:
            return {"available": False, "message": "No GPU or GPUtil library not available"}
        
        try:
            gpus = GPUtil.getGPUs()
            gpu_status = []
            
            for gpu in gpus:
                status = {
                    "id": gpu.id,
                    "name": gpu.name,
                    "memory_usage": {
                        "used_mb": gpu.memoryUsed,
                        "total_mb": gpu.memoryTotal,
                        "free_mb": gpu.memoryFree,
                        "utilization_percent": round(gpu.memoryUtil * 100, 1)
                    },
                    "gpu_utilization_percent": round(gpu.load * 100, 1),
                    "temperature_c": gpu.temperature,
                    "status": self._get_gpu_health_status(gpu)
                }
                gpu_status.append(status)
            
            return {
                "available": True,
                "gpu_count": len(gpus),
                "gpus": gpu_status
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get GPU status: {e}")
            return {"available": False, "error": str(e)}
    
    def _get_gpu_health_status(self, gpu) -> str:
        """Determine GPU health status."""
        if gpu.temperature > 85:
            return "overheating"
        elif gpu.memoryUtil > 0.9:
            return "memory_critical"
        elif gpu.load > 0.9:
            return "high_utilization"
        elif gpu.memoryUtil > 0.7 or gpu.load > 0.7:
            return "moderate_load"
        else:
            return "healthy"
    
    def allocate_gpu(self, memory_required_mb: Optional[int] = None) -> Optional[int]:
        """Allocate the best available GPU for a task."""
        if not self.gpu_available:
            return None
        
        try:
            gpus = GPUtil.getGPUs()
            
            # Find GPU with most free memory
            best_gpu = None
            max_free_memory = 0
            
            for gpu in gpus:
                if memory_required_mb and gpu.memoryFree < memory_required_mb:
                    continue
                
                if gpu.memoryFree > max_free_memory:
                    max_free_memory = gpu.memoryFree
                    best_gpu = gpu
            
            if best_gpu:
                self.logger.info(f"Allocated GPU {best_gpu.id} ({best_gpu.name}) with {best_gpu.memoryFree}MB free")
                return best_gpu.id
            else:
                self.logger.warning("No suitable GPU found for allocation")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to allocate GPU: {e}")
            return None
    
    def get_memory_recommendations(self) -> List[str]:
        """Get GPU memory optimization recommendations."""
        if not self.gpu_available:
            return ["GPU monitoring not available"]
        
        recommendations = []
        gpu_status = self.get_gpu_status()
        
        if not gpu_status.get("available", False):
            return ["GPU status unavailable"]
        
        for gpu in gpu_status["gpus"]:
            gpu_id = gpu["id"]
            memory_util = gpu["memory_usage"]["utilization_percent"]
            gpu_util = gpu["gpu_utilization_percent"]
            
            if memory_util > 90:
                recommendations.append(f"GPU {gpu_id}: Memory usage critical (>90%). Free memory immediately.")
            elif memory_util > 80:
                recommendations.append(f"GPU {gpu_id}: Memory usage high (>80%). Consider optimization.")
            
            if gpu_util < 50 and memory_util > 50:
                recommendations.append(f"GPU {gpu_id}: Low compute utilization with high memory usage. Check for memory leaks.")
        
        if not recommendations:
            recommendations.append("GPU memory usage appears optimal.")
        
        return recommendations


class ComputeManager:
    """
    Comprehensive compute resource management system.
    """
    
    def __init__(self, enable_cpu_monitoring: bool = True, enable_gpu_monitoring: bool = True):
        self.cpu_optimizer = CPUOptimizer() if enable_cpu_monitoring else None
        self.gpu_manager = GPUManager() if enable_gpu_monitoring else None
        self.logger = logging.getLogger(__name__)
        
        # Start monitoring
        if self.cpu_optimizer:
            self.cpu_optimizer.start_monitoring()
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        system_info = {
            "platform": platform.platform(),
            "architecture": platform.architecture(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "cpu": {
                "physical_cores": psutil.cpu_count(logical=False),
                "logical_cores": psutil.cpu_count(logical=True),
                "max_frequency": psutil.cpu_freq().max if psutil.cpu_freq() else None,
                "current_frequency": psutil.cpu_freq().current if psutil.cpu_freq() else None
            },
            "memory": {
                "total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "available_gb": round(psutil.virtual_memory().available / (1024**3), 2)
            }
        }
        
        if self.gpu_manager and self.gpu_manager.gpu_available:
            system_info["gpu"] = self.gpu_manager.get_gpu_status()
        
        return system_info
    
    def optimize_for_workload(self, workload_type: str, **kwargs) -> Dict[str, Any]:
        """Optimize system for specific workload type."""
        optimization_results = {
            "workload_type": workload_type,
            "optimizations_applied": []
        }
        
        if self.cpu_optimizer:
            # CPU optimizations
            optimal_workers = self.cpu_optimizer.get_optimal_worker_count(workload_type)
            optimization_results["optimal_worker_count"] = optimal_workers
            
            # Process affinity optimization
            affinity_result = self.cpu_optimizer.optimize_process_affinity()
            if affinity_result.get("optimized", False):
                optimization_results["optimizations_applied"].append("cpu_affinity")
        
        if self.gpu_manager and workload_type in ["gpu_compute", "machine_learning"]:
            # GPU allocation for GPU workloads
            memory_required = kwargs.get("gpu_memory_mb")
            allocated_gpu = self.gpu_manager.allocate_gpu(memory_required)
            if allocated_gpu is not None:
                optimization_results["allocated_gpu"] = allocated_gpu
                optimization_results["optimizations_applied"].append("gpu_allocation")
        
        return optimization_results
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        report = {
            "timestamp": time.time(),
            "system_info": self.get_system_info()
        }
        
        if self.cpu_optimizer:
            latest_snapshot = self.cpu_optimizer.take_snapshot()
            report["cpu_status"] = {
                "usage_percent": latest_snapshot.cpu_usage_percent,
                "load_average": latest_snapshot.cpu_load_avg,
                "memory_percent": latest_snapshot.memory_info["percent"]
            }
            report["cpu_recommendations"] = self.cpu_optimizer.get_performance_recommendations()
        
        if self.gpu_manager:
            report["gpu_status"] = self.gpu_manager.get_gpu_status()
            report["gpu_recommendations"] = self.gpu_manager.get_memory_recommendations()
        
        return report
    
    def execute_parallel_task(self, 
                            task_func: Callable,
                            task_args: List[Tuple],
                            workload_type: str = "cpu_bound",
                            max_workers: Optional[int] = None) -> List[Any]:
        """Execute tasks in parallel with optimal resource allocation."""
        if not max_workers:
            if self.cpu_optimizer:
                max_workers = self.cpu_optimizer.get_optimal_worker_count(workload_type)
            else:
                max_workers = multiprocessing.cpu_count()
        
        # Choose executor type based on workload
        if workload_type == "cpu_bound":
            executor_class = ProcessPoolExecutor
        else:
            executor_class = ThreadPoolExecutor
        
        results = []
        start_time = time.time()
        
        try:
            with executor_class(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_args = {executor.submit(task_func, *args): args for args in task_args}
                
                # Collect results as they complete
                for future in as_completed(future_to_args):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        self.logger.error(f"Task failed: {e}")
                        results.append({"error": str(e)})
        
        except Exception as e:
            self.logger.error(f"Parallel execution failed: {e}")
            raise
        
        execution_time = time.time() - start_time
        self.logger.info(f"Parallel execution completed in {execution_time:.2f}s with {max_workers} workers")
        
        return results
    
    def __del__(self):
        """Cleanup on destruction."""
        if hasattr(self, 'cpu_optimizer') and self.cpu_optimizer:
            self.cpu_optimizer.stop_monitoring()


# Convenience functions
def get_optimal_worker_count(workload_type: str = "cpu_bound") -> int:
    """Get optimal worker count for a workload type."""
    optimizer = CPUOptimizer()
    return optimizer.get_optimal_worker_count(workload_type)


def get_system_resources() -> Dict[str, Any]:
    """Get current system resource usage."""
    manager = ComputeManager()
    return manager.get_performance_report()
