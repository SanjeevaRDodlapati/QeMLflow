"""
Metrics Module

This module provides metric types and utilities for comprehensive performance
and application monitoring including counters, gauges, histograms, and timers.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4


class MetricType(Enum):
    """Metric type enumeration."""
    COUNTER = "counter"
    GAUGE = "gauge" 
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class Metric:
    """Base metric data structure."""
    
    name: str
    value: Union[int, float]
    metric_type: MetricType
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['metric_type'] = self.metric_type.value
        return data


class BaseMetric(ABC):
    """Base class for all metric types."""
    
    def __init__(self, name: str, tags: Optional[Dict[str, str]] = None, unit: str = ""):
        self.name = name
        self.tags = tags or {}
        self.unit = unit
        self.created_at = datetime.now(timezone.utc)
    
    @abstractmethod
    def get_value(self) -> Union[int, float]:
        """Get current metric value."""
        pass
    
    @abstractmethod
    def get_metric_type(self) -> MetricType:
        """Get metric type."""
        pass
    
    def to_metric(self) -> Metric:
        """Convert to Metric dataclass."""
        return Metric(
            name=self.name,
            value=self.get_value(),
            metric_type=self.get_metric_type(),
            tags=self.tags.copy(),
            unit=self.unit
        )


class Counter(BaseMetric):
    """Counter metric - monotonically increasing value."""
    
    def __init__(self, name: str, tags: Optional[Dict[str, str]] = None, unit: str = ""):
        super().__init__(name, tags, unit)
        self._value = 0
    
    def increment(self, value: Union[int, float] = 1):
        """Increment counter by value."""
        if value < 0:
            raise ValueError("Counter increment must be positive")
        self._value += value
    
    def get_value(self) -> Union[int, float]:
        """Get current counter value."""
        return self._value
    
    def get_metric_type(self) -> MetricType:
        """Get metric type."""
        return MetricType.COUNTER
    
    def reset(self):
        """Reset counter to zero."""
        self._value = 0


class Gauge(BaseMetric):
    """Gauge metric - arbitrary value that can go up or down."""
    
    def __init__(self, name: str, tags: Optional[Dict[str, str]] = None, unit: str = ""):
        super().__init__(name, tags, unit)
        self._value = 0
    
    def set(self, value: Union[int, float]):
        """Set gauge value."""
        self._value = value
    
    def increment(self, value: Union[int, float] = 1):
        """Increment gauge by value."""
        self._value += value
    
    def decrement(self, value: Union[int, float] = 1):
        """Decrement gauge by value."""
        self._value -= value
    
    def get_value(self) -> Union[int, float]:
        """Get current gauge value."""
        return self._value
    
    def get_metric_type(self) -> MetricType:
        """Get metric type."""
        return MetricType.GAUGE


class Histogram(BaseMetric):
    """Histogram metric - tracks distribution of values."""
    
    def __init__(self, name: str, tags: Optional[Dict[str, str]] = None, unit: str = "",
                 buckets: Optional[List[float]] = None):
        super().__init__(name, tags, unit)
        
        # Default buckets if none provided
        self.buckets = buckets or [0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0]
        self.buckets.sort()
        
        # Bucket counts
        self._bucket_counts = {bucket: 0 for bucket in self.buckets}
        self._bucket_counts['inf'] = 0  # +Inf bucket
        
        # Raw observations
        self._observations: List[float] = []
        self._count = 0
        self._sum = 0.0
    
    def observe(self, value: Union[int, float]):
        """Observe a value."""
        value = float(value)
        
        self._observations.append(value)
        self._count += 1
        self._sum += value
        
        # Update bucket counts
        for bucket in self.buckets:
            if value <= bucket:
                self._bucket_counts[bucket] += 1
        
        # Always increment the +Inf bucket
        self._bucket_counts['inf'] += 1
    
    def get_value(self) -> Union[int, float]:
        """Get observation count."""
        return self._count
    
    def get_metric_type(self) -> MetricType:
        """Get metric type."""
        return MetricType.HISTOGRAM
    
    def get_count(self) -> int:
        """Get total number of observations."""
        return self._count
    
    def get_sum(self) -> float:
        """Get sum of all observed values."""
        return self._sum
    
    def get_mean(self) -> Optional[float]:
        """Get mean of observed values."""
        if self._count == 0:
            return None
        return self._sum / self._count
    
    def get_percentile(self, percentile: float) -> Optional[float]:
        """Get percentile value."""
        if not self._observations:
            return None
        
        sorted_obs = sorted(self._observations)
        index = int((percentile / 100.0) * len(sorted_obs))
        
        if index >= len(sorted_obs):
            index = len(sorted_obs) - 1
        
        return sorted_obs[index]
    
    def get_bucket_counts(self) -> Dict[Union[float, str], int]:
        """Get bucket counts."""
        return self._bucket_counts.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        if not self._observations:
            return {
                "count": 0,
                "sum": 0.0,
                "mean": None,
                "min": None,
                "max": None,
                "p50": None,
                "p95": None,
                "p99": None
            }
        
        return {
            "count": self._count,
            "sum": self._sum,
            "mean": self.get_mean(),
            "min": min(self._observations),
            "max": max(self._observations),
            "p50": self.get_percentile(50),
            "p95": self.get_percentile(95),
            "p99": self.get_percentile(99),
            "bucket_counts": self.get_bucket_counts()
        }


class Timer(BaseMetric):
    """Timer metric - measures duration of operations."""
    
    def __init__(self, name: str, tags: Optional[Dict[str, str]] = None, unit: str = "seconds"):
        super().__init__(name, tags, unit)
        self._histogram = Histogram(f"{name}_duration", tags, unit)
        self._active_timers: Dict[str, float] = {}
    
    def start(self, timer_id: Optional[str] = None) -> str:
        """Start a timer and return timer ID."""
        if timer_id is None:
            timer_id = str(uuid4())
        
        self._active_timers[timer_id] = time.time()
        return timer_id
    
    def stop(self, timer_id: str) -> Optional[float]:
        """Stop a timer and record duration."""
        if timer_id not in self._active_timers:
            return None
        
        start_time = self._active_timers.pop(timer_id)
        duration = time.time() - start_time
        
        self._histogram.observe(duration)
        return duration
    
    def time_function(self, func, *args, **kwargs):
        """Time a function execution."""
        timer_id = self.start()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            self.stop(timer_id)
    
    def get_value(self) -> Union[int, float]:
        """Get total number of timed operations."""
        return self._histogram.get_count()
    
    def get_metric_type(self) -> MetricType:
        """Get metric type."""
        return MetricType.TIMER
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get timing statistics."""
        return self._histogram.get_statistics()
    
    def get_mean_duration(self) -> Optional[float]:
        """Get mean duration."""
        return self._histogram.get_mean()
    
    def get_percentile_duration(self, percentile: float) -> Optional[float]:
        """Get percentile duration."""
        return self._histogram.get_percentile(percentile)


# Context manager for timing operations
class time_operation:
    """Context manager for timing operations."""
    
    def __init__(self, timer: Timer, timer_id: Optional[str] = None):
        self.timer = timer
        self.timer_id = timer_id
        self.actual_timer_id: Optional[str] = None
        self.duration: Optional[float] = None
    
    def __enter__(self):
        self.actual_timer_id = self.timer.start(self.timer_id)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.actual_timer_id:
            self.duration = self.timer.stop(self.actual_timer_id)
    
    def get_duration(self) -> Optional[float]:
        """Get measured duration."""
        return self.duration


# Decorator for timing functions
def time_function(timer: Timer):
    """Decorator to time function execution."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            return timer.time_function(func, *args, **kwargs)
        return wrapper
    return decorator
