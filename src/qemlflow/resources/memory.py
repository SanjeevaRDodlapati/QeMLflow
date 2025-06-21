"""
Intelligent Memory Management Module

This module provides advanced memory management capabilities including:
- Memory usage tracking and profiling
- Memory optimization utilities
- Memory leak detection
- Automatic memory cleanup
- Memory usage alerts and limits
"""

import gc
import logging
import psutil
import threading
import time
import tracemalloc
import weakref
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class MemorySnapshot:
    """Represents a memory usage snapshot at a specific point in time."""
    timestamp: float
    total_memory: int
    available_memory: int
    used_memory: int
    memory_percent: float
    process_memory: int
    process_memory_percent: float
    gc_stats: Dict[str, Any] = field(default_factory=dict)
    top_traces: List[Tuple[str, int]] = field(default_factory=list)


class MemoryProfiler:
    """
    Advanced memory profiler with leak detection and optimization recommendations.
    """
    
    def __init__(self, trace_limit: int = 25, enable_tracemalloc: bool = True):
        self.trace_limit = trace_limit
        self.enable_tracemalloc = enable_tracemalloc
        self.snapshots: List[MemorySnapshot] = []
        self.baseline_snapshot: Optional[MemorySnapshot] = None
        self._lock = Lock()
        self.logger = logging.getLogger(__name__)
        
        if self.enable_tracemalloc and not tracemalloc.is_tracing():
            tracemalloc.start(self.trace_limit)
    
    def take_snapshot(self, label: str = "") -> MemorySnapshot:
        """Take a comprehensive memory snapshot."""
        with self._lock:
            # System memory stats
            memory_info = psutil.virtual_memory()
            process = psutil.Process()
            process_memory_info = process.memory_info()
            
            # Garbage collection stats
            gc_stats = {
                'collections': gc.get_stats(),
                'counts': gc.get_count(),
                'threshold': gc.get_threshold()
            }
            
            # Memory traces (if enabled)
            top_traces = []
            if self.enable_tracemalloc and tracemalloc.is_tracing():
                snapshot = tracemalloc.take_snapshot()
                top_stats = snapshot.statistics('lineno')[:self.trace_limit]
                top_traces = [(str(stat.traceback), stat.size) for stat in top_stats]
            
            snapshot = MemorySnapshot(
                timestamp=time.time(),
                total_memory=memory_info.total,
                available_memory=memory_info.available,
                used_memory=memory_info.used,
                memory_percent=memory_info.percent,
                process_memory=process_memory_info.rss,
                process_memory_percent=process.memory_percent(),
                gc_stats=gc_stats,
                top_traces=top_traces
            )
            
            self.snapshots.append(snapshot)
            if label:
                self.logger.info(f"Memory snapshot taken: {label}")
            
            return snapshot
    
    def set_baseline(self) -> MemorySnapshot:
        """Set the current memory state as baseline for leak detection."""
        self.baseline_snapshot = self.take_snapshot("baseline")
        return self.baseline_snapshot
    
    def detect_leaks(self) -> Dict[str, Any]:
        """Detect potential memory leaks since baseline."""
        if not self.baseline_snapshot:
            self.set_baseline()
            return {"warning": "No baseline set, created new baseline"}
        
        current_snapshot = self.take_snapshot("leak_detection")
        
        memory_growth = current_snapshot.process_memory - self.baseline_snapshot.process_memory
        memory_growth_mb = memory_growth / (1024 * 1024)
        
        leak_analysis = {
            "memory_growth_bytes": memory_growth,
            "memory_growth_mb": round(memory_growth_mb, 2),
            "growth_percentage": round(
                (memory_growth / self.baseline_snapshot.process_memory) * 100, 2
            ),
            "potential_leak": memory_growth_mb > 50,  # Threshold: 50MB growth
            "gc_collections": current_snapshot.gc_stats['collections'],
            "object_count_changes": {}
        }
        
        # Compare object counts
        baseline_counts = self.baseline_snapshot.gc_stats['counts']
        current_counts = current_snapshot.gc_stats['counts']
        
        for i, (baseline, current) in enumerate(zip(baseline_counts, current_counts)):
            generation = f"generation_{i}"
            leak_analysis["object_count_changes"][generation] = {
                "baseline": baseline,
                "current": current,
                "growth": current - baseline
            }
        
        return leak_analysis
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get comprehensive memory usage summary."""
        if not self.snapshots:
            return {"error": "No snapshots available"}
        
        latest = self.snapshots[-1]
        
        summary = {
            "current_memory": {
                "total_gb": round(latest.total_memory / (1024**3), 2),
                "available_gb": round(latest.available_memory / (1024**3), 2),
                "used_gb": round(latest.used_memory / (1024**3), 2),
                "usage_percent": latest.memory_percent
            },
            "process_memory": {
                "rss_mb": round(latest.process_memory / (1024**2), 2),
                "usage_percent": latest.process_memory_percent
            },
            "gc_stats": latest.gc_stats,
            "recommendations": self._generate_recommendations(latest)
        }
        
        if len(self.snapshots) > 1:
            # Add trend analysis
            previous = self.snapshots[-2]
            memory_trend = latest.process_memory - previous.process_memory
            summary["trend"] = {
                "memory_change_mb": round(memory_trend / (1024**2), 2),
                "direction": "increasing" if memory_trend > 0 else "decreasing"
            }
        
        return summary
    
    def _generate_recommendations(self, snapshot: MemorySnapshot) -> List[str]:
        """Generate memory optimization recommendations."""
        recommendations = []
        
        if snapshot.memory_percent > 80:
            recommendations.append("System memory usage is high (>80%). Consider freeing memory.")
        
        if snapshot.process_memory_percent > 10:
            recommendations.append("Process memory usage is high (>10% of system). Consider optimization.")
        
        # Check garbage collection patterns
        gc_counts = snapshot.gc_stats['counts']
        if sum(gc_counts) > 1000:
            recommendations.append("High object count detected. Consider explicit garbage collection.")
        
        if not recommendations:
            recommendations.append("Memory usage appears optimal.")
        
        return recommendations


class MemoryOptimizer:
    """
    Memory optimization utilities and automatic cleanup mechanisms.
    """
    
    def __init__(self, auto_gc_threshold: float = 0.8, cleanup_interval: int = 300):
        self.auto_gc_threshold = auto_gc_threshold  # Memory percentage threshold for auto GC
        self.cleanup_interval = cleanup_interval  # Seconds between cleanup cycles
        self.logger = logging.getLogger(__name__)
        self._cleanup_registry: Set[weakref.ReferenceType] = set()
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
    
    def register_for_cleanup(self, obj: Any, cleanup_func: Optional[Callable] = None) -> None:
        """Register an object for automatic cleanup."""
        def default_cleanup(ref):
            self.logger.debug(f"Object {ref} was garbage collected")
        
        cleanup_callback = cleanup_func or default_cleanup
        weak_ref = weakref.ref(obj, cleanup_callback)
        self._cleanup_registry.add(weak_ref)
    
    def force_cleanup(self) -> Dict[str, Any]:
        """Force aggressive memory cleanup."""
        initial_memory = psutil.Process().memory_info().rss
        
        # Clear dead weak references
        dead_refs = [ref for ref in self._cleanup_registry if ref() is None]
        for ref in dead_refs:
            self._cleanup_registry.discard(ref)
        
        # Force garbage collection
        collected = []
        for generation in range(3):
            collected.append(gc.collect(generation))
        
        # Clear numpy caches if available
        try:
            if hasattr(np, 'testing'):
                # Clear numpy test caches
                pass
        except Exception:
            pass
        
        final_memory = psutil.Process().memory_info().rss
        memory_freed = initial_memory - final_memory
        
        return {
            "memory_freed_mb": round(memory_freed / (1024**2), 2),
            "gc_collected": collected,
            "dead_references_cleared": len(dead_refs)
        }
    
    def optimize_numpy_arrays(self, arrays: List[np.ndarray]) -> Dict[str, Any]:
        """Optimize numpy arrays for memory efficiency."""
        optimization_results = []
        total_savings = 0
        
        for i, arr in enumerate(arrays):
            if not isinstance(arr, np.ndarray):
                continue
            
            original_size = arr.nbytes
            
            # Check if we can use a smaller dtype
            if arr.dtype in [np.float64, np.complex128]:
                # Try to downcast to float32/complex64 if precision allows
                if arr.dtype == np.float64:
                    arr_32 = arr.astype(np.float32)
                    if np.allclose(arr, arr_32, rtol=1e-5):
                        arrays[i] = arr_32
                        savings = original_size - arr_32.nbytes
                        total_savings += savings
                        optimization_results.append({
                            "array_index": i,
                            "original_dtype": str(arr.dtype),
                            "optimized_dtype": str(arr_32.dtype),
                            "memory_saved_mb": round(savings / (1024**2), 2)
                        })
            
            # Check for memory layout optimization
            if not arrays[i].flags.c_contiguous:
                contiguous_array = np.ascontiguousarray(arrays[i])
                arrays[i] = contiguous_array
                optimization_results.append({
                    "array_index": i,
                    "optimization": "made_contiguous",
                    "performance_benefit": "improved_cache_efficiency"
                })
        
        return {
            "total_memory_saved_mb": round(total_savings / (1024**2), 2),
            "optimizations": optimization_results
        }
    
    def start_monitoring(self) -> None:
        """Start automatic memory monitoring and cleanup."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._stop_monitoring.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        self.logger.info("Memory monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop automatic memory monitoring."""
        if not self._monitoring:
            return
        
        self._stop_monitoring.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        self._monitoring = False
        self.logger.info("Memory monitoring stopped")
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while not self._stop_monitoring.wait(self.cleanup_interval):
            try:
                memory_percent = psutil.virtual_memory().percent / 100
                if memory_percent > self.auto_gc_threshold:
                    self.logger.warning(
                        f"Memory usage {memory_percent:.1%} exceeds threshold "
                        f"{self.auto_gc_threshold:.1%}. Running cleanup..."
                    )
                    cleanup_result = self.force_cleanup()
                    self.logger.info(f"Cleanup completed: {cleanup_result}")
            except Exception as e:
                self.logger.error(f"Error in memory monitoring: {e}")


class MemoryManager:
    """
    Comprehensive memory management system combining profiling and optimization.
    """
    
    def __init__(self, 
                 enable_profiling: bool = True,
                 enable_optimization: bool = True,
                 auto_cleanup: bool = True):
        self.profiler = MemoryProfiler() if enable_profiling else None
        self.optimizer = MemoryOptimizer() if enable_optimization else None
        self.auto_cleanup = auto_cleanup
        self.logger = logging.getLogger(__name__)
        
        if self.auto_cleanup and self.optimizer:
            self.optimizer.start_monitoring()
    
    @contextmanager
    def profile_block(self, block_name: str):
        """Context manager for profiling a code block."""
        if not self.profiler:
            yield
            return
        
        self.logger.info(f"Starting memory profiling for: {block_name}")
        start_snapshot = self.profiler.take_snapshot(f"{block_name}_start")
        
        try:
            yield
        finally:
            end_snapshot = self.profiler.take_snapshot(f"{block_name}_end")
            
            # Calculate memory usage for this block
            memory_delta = end_snapshot.process_memory - start_snapshot.process_memory
            memory_delta_mb = memory_delta / (1024**2)
            
            self.logger.info(
                f"Memory profiling completed for: {block_name}. "
                f"Memory change: {memory_delta_mb:.2f} MB"
            )
    
    def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive memory management status report."""
        report = {
            "timestamp": time.time(),
            "profiler_enabled": self.profiler is not None,
            "optimizer_enabled": self.optimizer is not None,
            "auto_cleanup": self.auto_cleanup
        }
        
        if self.profiler:
            report["memory_summary"] = self.profiler.get_memory_summary()
            report["leak_detection"] = self.profiler.detect_leaks()
        
        if self.optimizer:
            report["monitoring_active"] = self.optimizer._monitoring
            
        return report
    
    def cleanup_and_optimize(self) -> Dict[str, Any]:
        """Perform comprehensive cleanup and optimization."""
        results = {}
        
        if self.optimizer:
            results["cleanup"] = self.optimizer.force_cleanup()
        
        if self.profiler:
            results["memory_summary"] = self.profiler.get_memory_summary()
        
        return results
    
    def __del__(self):
        """Cleanup on destruction."""
        if hasattr(self, 'optimizer') and self.optimizer:
            self.optimizer.stop_monitoring()


# Convenience functions for easy access
def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage in a simple format."""
    memory_info = psutil.virtual_memory()
    process = psutil.Process()
    
    return {
        "system_memory_gb": round(memory_info.total / (1024**3), 2),
        "available_memory_gb": round(memory_info.available / (1024**3), 2),
        "memory_usage_percent": memory_info.percent,
        "process_memory_mb": round(process.memory_info().rss / (1024**2), 2),
        "process_memory_percent": round(process.memory_percent(), 2)
    }


def profile_function(func: Callable) -> Callable:
    """Decorator for profiling function memory usage."""
    def wrapper(*args, **kwargs):
        profiler = MemoryProfiler()
        profiler.set_baseline()
        
        try:
            result = func(*args, **kwargs)
        finally:
            leak_analysis = profiler.detect_leaks()
            if leak_analysis.get("potential_leak", False):
                logging.getLogger(__name__).warning(
                    f"Potential memory leak detected in {func.__name__}: "
                    f"{leak_analysis['memory_growth_mb']} MB growth"
                )
        
        return result
    
    return wrapper
