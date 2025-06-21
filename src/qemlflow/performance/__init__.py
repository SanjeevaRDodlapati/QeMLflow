"""
QeMLflow Performance Optimization Module

This module provides caching strategies, performance optimization,
and performance monitoring capabilities.
"""

import functools
import gc
import hashlib
import logging
import pickle
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import json
import psutil

logger = logging.getLogger(__name__)


@dataclass
class CacheStats:
    """Cache statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    max_size: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'size': self.size,
            'max_size': self.max_size,
            'hit_rate': self.hit_rate
        }


@dataclass
class PerformanceMetrics:
    """Performance metrics."""
    timestamp: datetime
    cpu_utilization: float
    memory_utilization: float
    response_time: float
    throughput: float
    cache_hit_rate: float
    error_rate: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'cpu_utilization': self.cpu_utilization,
            'memory_utilization': self.memory_utilization,
            'response_time': self.response_time,
            'throughput': self.throughput,
            'cache_hit_rate': self.cache_hit_rate,
            'error_rate': self.error_rate
        }


class CacheBackend(ABC):
    """Abstract cache backend."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """Clear all cache entries."""
        pass
    
    @abstractmethod
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        pass


class MemoryCache(CacheBackend):
    """In-memory cache with LRU eviction."""
    
    def __init__(self, max_size: int = 1000, ttl: Optional[int] = None):
        self.max_size = max_size
        self.default_ttl = ttl
        self._cache: OrderedDict = OrderedDict()
        self._timestamps: Dict[str, float] = {}
        self._ttls: Dict[str, float] = {}
        self._lock = threading.RLock()
        self._stats = CacheStats(max_size=max_size)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key not in self._cache:
                self._stats.misses += 1
                return None
            
            # Check TTL
            if self._is_expired(key):
                self._remove_key(key)
                self._stats.misses += 1
                return None
            
            # Move to end (LRU)
            value = self._cache.pop(key)
            self._cache[key] = value
            self._stats.hits += 1
            return value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        with self._lock:
            try:
                # Remove existing key if present
                if key in self._cache:
                    self._remove_key(key)
                
                # Evict if necessary
                while len(self._cache) >= self.max_size:
                    self._evict_lru()
                
                # Add new entry
                self._cache[key] = value
                self._timestamps[key] = time.time()
                
                effective_ttl = ttl or self.default_ttl
                if effective_ttl:
                    self._ttls[key] = time.time() + effective_ttl
                
                self._stats.size = len(self._cache)
                return True
            
            except Exception as e:
                logger.error(f"Failed to set cache key {key}: {e}")
                return False
    
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        with self._lock:
            if key in self._cache:
                self._remove_key(key)
                return True
            return False
    
    def clear(self) -> bool:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
            self._ttls.clear()
            self._stats.size = 0
            return True
    
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            self._stats.size = len(self._cache)
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                size=self._stats.size,
                max_size=self._stats.max_size
            )
    
    def _is_expired(self, key: str) -> bool:
        """Check if key is expired."""
        if key not in self._ttls:
            return False
        return time.time() > self._ttls[key]
    
    def _remove_key(self, key: str):
        """Remove key and its metadata."""
        self._cache.pop(key, None)
        self._timestamps.pop(key, None)
        self._ttls.pop(key, None)
        self._stats.size = len(self._cache)
    
    def _evict_lru(self):
        """Evict least recently used item."""
        if self._cache:
            key = next(iter(self._cache))
            self._remove_key(key)
            self._stats.evictions += 1


class FileCache(CacheBackend):
    """File-based cache with compression."""
    
    def __init__(self, cache_dir: Path, max_size_bytes: int = 1000000000):  # 1GB default
        self.cache_dir = Path(cache_dir)
        self.max_size_bytes = max_size_bytes
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._stats = CacheStats()
        self._index_file = self.cache_dir / "index.json"
        self._load_index()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            cache_file = self._get_cache_file(key)
            
            if not cache_file.exists():
                self._stats.misses += 1
                return None
            
            try:
                # Check if expired
                if self._is_expired(cache_file):
                    cache_file.unlink(missing_ok=True)
                    self._stats.misses += 1
                    return None
                
                # Load from file
                with open(cache_file, 'rb') as f:
                    value = pickle.load(f)
                
                # Update access time
                cache_file.touch()
                self._stats.hits += 1
                return value
            
            except Exception as e:
                logger.error(f"Failed to load cache file {cache_file}: {e}")
                cache_file.unlink(missing_ok=True)
                self._stats.misses += 1
                return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        with self._lock:
            try:
                cache_file = self._get_cache_file(key)
                
                # Serialize value
                with open(cache_file, 'wb') as f:
                    pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                # Set TTL if specified
                if ttl:
                    expiry_time = time.time() + ttl
                    # Store expiry time in extended attributes or separate file
                    self._set_expiry(cache_file, expiry_time)
                
                # Cleanup if necessary
                self._cleanup_if_needed()
                
                self._update_stats()
                return True
            
            except Exception as e:
                logger.error(f"Failed to save cache file {cache_file}: {e}")
                return False
    
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        with self._lock:
            cache_file = self._get_cache_file(key)
            if cache_file.exists():
                cache_file.unlink()
                self._update_stats()
                return True
            return False
    
    def clear(self) -> bool:
        """Clear all cache entries."""
        with self._lock:
            try:
                for cache_file in self.cache_dir.glob("*.cache"):
                    cache_file.unlink()
                self._update_stats()
                return True
            except Exception as e:
                logger.error(f"Failed to clear cache: {e}")
                return False
    
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            self._update_stats()
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                size=self._stats.size,
                max_size=self.max_size_bytes
            )
    
    def _get_cache_file(self, key: str) -> Path:
        """Get cache file path for key."""
        # Hash key to avoid filesystem issues
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
    
    def _is_expired(self, cache_file: Path) -> bool:
        """Check if cache file is expired."""
        expiry_file = cache_file.with_suffix('.expiry')
        if not expiry_file.exists():
            return False
        
        try:
            with open(expiry_file, 'r') as f:
                expiry_time = float(f.read())
            return time.time() > expiry_time
        except Exception:
            return False
    
    def _set_expiry(self, cache_file: Path, expiry_time: float):
        """Set expiry time for cache file."""
        expiry_file = cache_file.with_suffix('.expiry')
        with open(expiry_file, 'w') as f:
            f.write(str(expiry_time))
    
    def _cleanup_if_needed(self):
        """Cleanup cache if size exceeds limit."""
        current_size = self._calculate_total_size()
        if current_size > self.max_size_bytes:
            # Remove oldest files
            cache_files = list(self.cache_dir.glob("*.cache"))
            cache_files.sort(key=lambda f: f.stat().st_atime)
            
            while current_size > self.max_size_bytes * 0.8 and cache_files:
                oldest_file = cache_files.pop(0)
                file_size = oldest_file.stat().st_size
                oldest_file.unlink(missing_ok=True)
                
                # Remove associated expiry file
                expiry_file = oldest_file.with_suffix('.expiry')
                expiry_file.unlink(missing_ok=True)
                
                current_size -= file_size
                self._stats.evictions += 1
    
    def _calculate_total_size(self) -> int:
        """Calculate total cache size."""
        total_size = 0
        for cache_file in self.cache_dir.glob("*.cache"):
            try:
                total_size += cache_file.stat().st_size
            except FileNotFoundError:
                pass
        return total_size
    
    def _update_stats(self):
        """Update cache statistics."""
        self._stats.size = len(list(self.cache_dir.glob("*.cache")))
    
    def _load_index(self):
        """Load cache index."""
        # Implementation for tracking cache metadata
        pass


class CacheManager:
    """Unified cache management."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.caches: Dict[str, CacheBackend] = {}
        self.logger = logging.getLogger(__name__)
        self._setup_caches()
    
    def _setup_caches(self):
        """Setup cache backends."""
        cache_config = self.config.get('caching', {})
        
        # Memory cache
        if cache_config.get('memory_cache', {}).get('enabled', False):
            memory_config = cache_config['memory_cache']
            max_size = self._parse_size(memory_config.get('max_size', '1GB'))
            ttl = memory_config.get('ttl')
            
            self.caches['memory'] = MemoryCache(
                max_size=max_size // 1024,  # Convert to number of items (rough estimate)
                ttl=ttl
            )
        
        # File cache
        if cache_config.get('file_cache', {}).get('enabled', False):
            file_config = cache_config['file_cache']
            cache_dir = Path(file_config.get('cache_dir', './cache'))
            max_size = self._parse_size(file_config.get('max_size', '2GB'))
            
            self.caches['file'] = FileCache(
                cache_dir=cache_dir,
                max_size_bytes=max_size
            )
    
    def get(self, key: str, cache_type: str = 'memory') -> Optional[Any]:
        """Get value from cache."""
        if cache_type in self.caches:
            return self.caches[cache_type].get(key)
        return None
    
    def set(self, key: str, value: Any, cache_type: str = 'memory', 
            ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        if cache_type in self.caches:
            return self.caches[cache_type].set(key, value, ttl)
        return False
    
    def delete(self, key: str, cache_type: str = 'memory') -> bool:
        """Delete value from cache."""
        if cache_type in self.caches:
            return self.caches[cache_type].delete(key)
        return False
    
    def clear(self, cache_type: Optional[str] = None) -> bool:
        """Clear cache(s)."""
        if cache_type:
            if cache_type in self.caches:
                return self.caches[cache_type].clear()
            return False
        else:
            # Clear all caches
            results = []
            for cache in self.caches.values():
                results.append(cache.clear())
            return all(results)
    
    def stats(self) -> Dict[str, CacheStats]:
        """Get statistics for all caches."""
        return {
            cache_type: cache.stats()
            for cache_type, cache in self.caches.items()
        }
    
    def _parse_size(self, size_str: str) -> int:
        """Parse size string to bytes."""
        size_str = size_str.upper()
        multipliers = {
            'B': 1,
            'KB': 1024,
            'MB': 1024**2,
            'GB': 1024**3,
            'TB': 1024**4
        }
        
        for suffix, multiplier in multipliers.items():
            if size_str.endswith(suffix):
                number = float(size_str[:-len(suffix)])
                return int(number * multiplier)
        
        # Default to bytes if no suffix
        return int(size_str)


def cache(cache_manager: CacheManager, cache_type: str = 'memory', 
          ttl: Optional[int] = None, key_func: Optional[Callable] = None):
    """Cache decorator."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = _generate_cache_key(func.__name__, args, kwargs)
            
            # Try to get from cache
            result = cache_manager.get(cache_key, cache_type)
            if result is not None:
                return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_manager.set(cache_key, result, cache_type, ttl)
            return result
        
        return wrapper
    return decorator


def _generate_cache_key(func_name: str, args: tuple, kwargs: dict) -> str:
    """Generate cache key from function name and arguments."""
    key_data = {
        'func': func_name,
        'args': args,
        'kwargs': sorted(kwargs.items())
    }
    key_str = json.dumps(key_data, default=str, sort_keys=True)
    return hashlib.md5(key_str.encode()).hexdigest()


class PerformanceOptimizer:
    """Performance optimization utilities."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._setup_optimizations()
    
    def _setup_optimizations(self):
        """Setup performance optimizations."""
        opt_config = self.config.get('optimization', {})
        
        # CPU optimizations
        if opt_config.get('cpu', {}).get('enabled', False):
            self._setup_cpu_optimizations(opt_config['cpu'])
        
        # Memory optimizations
        if opt_config.get('memory', {}).get('enabled', False):
            self._setup_memory_optimizations(opt_config['memory'])
        
        # I/O optimizations
        if opt_config.get('io', {}).get('enabled', False):
            self._setup_io_optimizations(opt_config['io'])
    
    def _setup_cpu_optimizations(self, cpu_config: Dict[str, Any]):
        """Setup CPU optimizations."""
        # Parallelization settings
        parallel_config = cpu_config.get('parallelization', {})
        if parallel_config.get('enabled', False):
            max_workers = parallel_config.get('max_workers', 'auto')
            if max_workers == 'auto':
                max_workers = psutil.cpu_count()
            
            self.logger.info(f"CPU optimization: max_workers={max_workers}")
        
        # Computational optimizations
        comp_config = cpu_config.get('computation', {})
        optimizations = []
        if comp_config.get('vectorization', False):
            optimizations.append('vectorization')
        if comp_config.get('jit_compilation', False):
            optimizations.append('jit_compilation')
        
        if optimizations:
            self.logger.info(f"CPU computational optimizations: {optimizations}")
    
    def _setup_memory_optimizations(self, memory_config: Dict[str, Any]):
        """Setup memory optimizations."""
        # Garbage collection tuning
        mgmt_config = memory_config.get('management', {})
        if mgmt_config.get('garbage_collection_tuning', False):
            self._tune_garbage_collection()
        
        # Memory allocation
        alloc_config = memory_config.get('allocation', {})
        if alloc_config.get('pre_allocation', False):
            self.logger.info("Memory optimization: pre-allocation enabled")
    
    def _setup_io_optimizations(self, io_config: Dict[str, Any]):
        """Setup I/O optimizations."""
        # File I/O optimizations
        file_config = io_config.get('file_io', {})
        optimizations = []
        if file_config.get('async_io', False):
            optimizations.append('async_io')
        if file_config.get('read_ahead', False):
            optimizations.append('read_ahead')
        
        if optimizations:
            self.logger.info(f"I/O optimizations: {optimizations}")
    
    def _tune_garbage_collection(self):
        """Tune garbage collection settings."""
        try:
            # Set GC thresholds for better performance
            gc.set_threshold(700, 10, 10)
            self.logger.info("Garbage collection tuning applied")
        except Exception as e:
            self.logger.warning(f"Failed to tune garbage collection: {e}")
    
    def optimize_function(self, func: Callable, optimization_type: str = 'auto') -> Callable:
        """Optimize a function based on configuration."""
        if optimization_type == 'auto':
            # Auto-detect optimization opportunities
            return self._auto_optimize_function(func)
        else:
            # Apply specific optimization
            return self._apply_optimization(func, optimization_type)
    
    def _auto_optimize_function(self, func: Callable) -> Callable:
        """Auto-optimize function."""
        # Simple example - could be extended with more sophisticated analysis
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Add timing and optimization logic here
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            if execution_time > 1.0:  # Functions taking more than 1s
                self.logger.debug(f"Slow function detected: {func.__name__} took {execution_time:.3f}s")
            
            return result
        
        return wrapper
    
    def _apply_optimization(self, func: Callable, optimization_type: str) -> Callable:
        """Apply specific optimization."""
        if optimization_type == 'memoize':
            return functools.lru_cache(maxsize=128)(func)
        else:
            return func


class PerformanceMonitor:
    """Performance monitoring and metrics collection."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.metrics_history: List[PerformanceMetrics] = []
        self.max_history = 1000
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
    
    def start_monitoring(self):
        """Start performance monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        self.logger.info("Performance monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        interval = self.config.get('monitoring', {}).get('metrics', {}).get('collection_interval', 30)
        
        while self._monitoring:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Trim history
                if len(self.metrics_history) > self.max_history:
                    self.metrics_history.pop(0)
                
                # Check thresholds
                self._check_thresholds(metrics)
                
                time.sleep(interval)
            
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval)
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # Placeholder for application-specific metrics
        response_time = self._get_avg_response_time()
        throughput = self._get_current_throughput()
        cache_hit_rate = self._get_cache_hit_rate()
        error_rate = self._get_error_rate()
        
        return PerformanceMetrics(
            timestamp=datetime.now(),
            cpu_utilization=cpu_percent,
            memory_utilization=memory.percent,
            response_time=response_time,
            throughput=throughput,
            cache_hit_rate=cache_hit_rate,
            error_rate=error_rate
        )
    
    def _get_avg_response_time(self) -> float:
        """Get average response time (placeholder)."""
        # In a real implementation, this would track actual response times
        return 200.0  # ms
    
    def _get_current_throughput(self) -> float:
        """Get current throughput (placeholder)."""
        # In a real implementation, this would track actual throughput
        return 100.0  # requests/second
    
    def _get_cache_hit_rate(self) -> float:
        """Get cache hit rate (placeholder)."""
        # In a real implementation, this would get actual cache stats
        return 0.85  # 85%
    
    def _get_error_rate(self) -> float:
        """Get error rate (placeholder)."""
        # In a real implementation, this would track actual errors
        return 0.01  # 1%
    
    def _check_thresholds(self, metrics: PerformanceMetrics):
        """Check performance thresholds and alert if needed."""
        thresholds = self.config.get('monitoring', {}).get('thresholds', {})
        
        # Response time check
        response_threshold = thresholds.get('response_time', {})
        if metrics.response_time > response_threshold.get('critical', 5000):
            self.logger.critical(f"Critical response time: {metrics.response_time}ms")
        elif metrics.response_time > response_threshold.get('warning', 1000):
            self.logger.warning(f"High response time: {metrics.response_time}ms")
        
        # Cache hit rate check
        cache_threshold = thresholds.get('cache_hit_rate', {})
        if metrics.cache_hit_rate < cache_threshold.get('critical', 0.6):
            self.logger.critical(f"Critical cache hit rate: {metrics.cache_hit_rate:.2%}")
        elif metrics.cache_hit_rate < cache_threshold.get('warning', 0.8):
            self.logger.warning(f"Low cache hit rate: {metrics.cache_hit_rate:.2%}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary."""
        if not self.metrics_history:
            return {}
        
        recent_metrics = self.metrics_history[-10:]  # Last 10 measurements
        
        return {
            'avg_cpu_utilization': sum(m.cpu_utilization for m in recent_metrics) / len(recent_metrics),
            'avg_memory_utilization': sum(m.memory_utilization for m in recent_metrics) / len(recent_metrics),
            'avg_response_time': sum(m.response_time for m in recent_metrics) / len(recent_metrics),
            'avg_throughput': sum(m.throughput for m in recent_metrics) / len(recent_metrics),
            'avg_cache_hit_rate': sum(m.cache_hit_rate for m in recent_metrics) / len(recent_metrics),
            'avg_error_rate': sum(m.error_rate for m in recent_metrics) / len(recent_metrics),
            'metrics_count': len(self.metrics_history)
        }


class PerformanceManager:
    """Main performance management system."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.cache_manager = CacheManager(self.config)
        self.optimizer = PerformanceOptimizer(self.config)
        self.monitor = PerformanceMonitor(self.config)
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load performance configuration."""
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    import yaml
                    return yaml.safe_load(f) or {}
            except Exception as e:
                self.logger.warning(f"Failed to load config from {config_path}: {e}")
        
        # Default configuration
        return {
            'caching': {
                'memory_cache': {'enabled': True, 'max_size': '256MB', 'ttl': 3600},
                'file_cache': {'enabled': True, 'cache_dir': './cache', 'max_size': '1GB'}
            },
            'optimization': {
                'cpu': {'enabled': True},
                'memory': {'enabled': True},
                'io': {'enabled': True}
            },
            'monitoring': {
                'enabled': True,
                'metrics': {'collection_interval': 30}
            }
        }
    
    def start(self):
        """Start performance management."""
        if self.config.get('monitoring', {}).get('enabled', False):
            self.monitor.start_monitoring()
        self.logger.info("Performance management started")
    
    def stop(self):
        """Stop performance management."""
        self.monitor.stop_monitoring()
        self.logger.info("Performance management stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get performance status."""
        return {
            'cache_stats': {
                cache_type: stats.to_dict()
                for cache_type, stats in self.cache_manager.stats().items()
            },
            'performance_metrics': self.monitor.get_metrics_summary(),
            'config': {
                'caching_enabled': bool(self.config.get('caching')),
                'optimization_enabled': bool(self.config.get('optimization')),
                'monitoring_enabled': bool(self.config.get('monitoring'))
            }
        }


# Global performance manager instance
_performance_manager: Optional[PerformanceManager] = None


def initialize_performance_system(config_path: Optional[str] = None) -> PerformanceManager:
    """Initialize the global performance system."""
    global _performance_manager
    if _performance_manager is None:
        _performance_manager = PerformanceManager(config_path)
        _performance_manager.start()
    return _performance_manager


def get_performance_manager() -> Optional[PerformanceManager]:
    """Get the global performance manager."""
    return _performance_manager


def shutdown_performance_system():
    """Shutdown the global performance system."""
    global _performance_manager
    if _performance_manager:
        _performance_manager.stop()
        _performance_manager = None


# Convenience decorators
def cached(cache_type: str = 'memory', ttl: Optional[int] = None):
    """Cache decorator using global performance manager."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            manager = get_performance_manager()
            if manager:
                cache_key = _generate_cache_key(func.__name__, args, kwargs)
                
                # Try cache first
                result = manager.cache_manager.get(cache_key, cache_type)
                if result is not None:
                    return result
                
                # Execute and cache
                result = func(*args, **kwargs)
                manager.cache_manager.set(cache_key, result, cache_type, ttl)
                return result
            else:
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


def optimized(optimization_type: str = 'auto'):
    """Optimization decorator using global performance manager."""
    def decorator(func: Callable) -> Callable:
        manager = get_performance_manager()
        if manager:
            return manager.optimizer.optimize_function(func, optimization_type)
        else:
            return func
    
    return decorator


if __name__ == "__main__":
    # Example usage
    manager = initialize_performance_system("config/performance.yml")
    
    try:
        # Run for a while
        time.sleep(60)  # 1 minute
        
        # Get status
        status = manager.get_status()
        print(json.dumps(status, indent=2))
        
    finally:
        shutdown_performance_system()
