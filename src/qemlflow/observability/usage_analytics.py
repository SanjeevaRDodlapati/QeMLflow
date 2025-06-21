"""
Usage Analytics Module

This module provides comprehensive usage analytics tracking including feature usage,
performance analytics, user behavior analysis, and usage reporting for enterprise-grade
platform optimization and user experience insights.
"""

import json
import logging
import time
import uuid
from collections import defaultdict, Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Tuple
import threading
import hashlib
import os
import sys
from functools import wraps

try:
    import psutil
except ImportError:
    psutil = None


@dataclass
class UsageEvent:
    """Individual usage event tracking."""
    
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = ""  # api_call, feature_usage, error, performance
    feature_name: str = ""
    user_id: str = ""  # anonymous by default
    session_id: str = ""
    
    # Timing information
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    duration_ms: float = 0.0
    
    # Context information
    platform: str = field(default_factory=lambda: sys.platform)
    python_version: str = field(default_factory=lambda: f"{sys.version_info.major}.{sys.version_info.minor}")
    qemlflow_version: str = "0.2.0"
    
    # Usage metrics
    input_size: int = 0
    output_size: int = 0
    memory_used_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    
    # Success/failure tracking
    success: bool = True
    error_type: str = ""
    error_message: str = ""
    
    # Custom metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class PerformanceMetrics:
    """Performance analytics data."""
    
    feature_name: str = ""
    measurement_window: str = "1h"  # 1h, 1d, 1w, 1m
    
    # Timing metrics
    avg_duration_ms: float = 0.0
    p50_duration_ms: float = 0.0
    p95_duration_ms: float = 0.0
    p99_duration_ms: float = 0.0
    min_duration_ms: float = 0.0
    max_duration_ms: float = 0.0
    
    # Usage metrics
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    success_rate: float = 0.0
    
    # Resource metrics
    avg_memory_mb: float = 0.0
    max_memory_mb: float = 0.0
    avg_cpu_percent: float = 0.0
    max_cpu_percent: float = 0.0
    
    # Error patterns
    error_types: Dict[str, int] = field(default_factory=dict)
    top_errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class UserBehaviorPattern:
    """User behavior analysis results."""
    
    pattern_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    pattern_type: str = ""  # workflow, usage_sequence, frequency_pattern
    pattern_name: str = ""
    
    # Pattern details
    frequency: int = 0
    confidence_score: float = 0.0
    user_segments: List[str] = field(default_factory=list)
    
    # Usage characteristics
    common_features: List[str] = field(default_factory=list)
    session_patterns: Dict[str, Any] = field(default_factory=dict)
    time_patterns: Dict[str, Any] = field(default_factory=dict)
    
    # Business impact
    impact_score: float = 0.0
    optimization_opportunities: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class UsageTracker:
    """Core usage tracking system."""
    
    def __init__(self, storage_dir: str = "usage_analytics", enabled: bool = True):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True, parents=True)
        self.enabled = enabled
        
        self.logger = logging.getLogger(__name__)
        self._lock = threading.Lock()
        
        # Event buffer for batch processing
        self._event_buffer: List[UsageEvent] = []
        self._buffer_size = 100
        self._last_flush = time.time()
        self._flush_interval = 60  # seconds
        
        # Session tracking
        self._session_id = str(uuid.uuid4())
        self._session_start = time.time()
        
        # User identification (anonymous by default)
        self._user_id = self._generate_anonymous_user_id()
        
        # Background flush thread
        self._flush_thread: Optional[threading.Thread] = None
        if self.enabled:
            self._start_background_flush()
    
    def _generate_anonymous_user_id(self) -> str:
        """Generate anonymous user ID based on system characteristics."""
        try:
            # Use system info to create consistent anonymous ID
            uname_info = os.uname()
            system_info = f"{os.getlogin()}-{uname_info.machine}-{uname_info.sysname}"
            return hashlib.sha256(system_info.encode()).hexdigest()[:16]
        except Exception:
            return "anonymous-user"
    
    def _start_background_flush(self):
        """Start background thread for periodic event flushing."""
        def flush_worker():
            while self.enabled:
                time.sleep(self._flush_interval)
                self._flush_events()
        
        self._flush_thread = threading.Thread(target=flush_worker, daemon=True)
        self._flush_thread.start()
    
    def track_event(self, event_type: str, feature_name: str, **kwargs) -> str:
        """Track a usage event."""
        if not self.enabled:
            return ""
        
        # Collect system metrics if available
        memory_mb = 0.0
        cpu_percent = 0.0
        
        if psutil:
            try:
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                cpu_percent = process.cpu_percent()
            except Exception:
                pass
        
        event = UsageEvent(
            event_type=event_type,
            feature_name=feature_name,
            user_id=self._user_id,
            session_id=self._session_id,
            memory_used_mb=memory_mb,
            cpu_usage_percent=cpu_percent,
            **kwargs
        )
        
        with self._lock:
            self._event_buffer.append(event)
            
            # Flush if buffer is full
            if len(self._event_buffer) >= self._buffer_size:
                self._flush_events()
        
        return event.event_id
    
    def track_performance(self, feature_name: str, duration_ms: float, success: bool = True, **kwargs) -> str:
        """Track performance metrics for a feature."""
        return self.track_event(
            event_type="performance",
            feature_name=feature_name,
            duration_ms=duration_ms,
            success=success,
            **kwargs
        )
    
    def track_feature_usage(self, feature_name: str, **kwargs) -> str:
        """Track feature usage."""
        return self.track_event(
            event_type="feature_usage",
            feature_name=feature_name,
            **kwargs
        )
    
    def track_api_call(self, api_endpoint: str, method: str = "GET", **kwargs) -> str:
        """Track API call usage."""
        return self.track_event(
            event_type="api_call",
            feature_name=f"{method} {api_endpoint}",
            **kwargs
        )
    
    def track_error(self, feature_name: str, error_type: str, error_message: str, **kwargs) -> str:
        """Track error occurrence."""
        return self.track_event(
            event_type="error",
            feature_name=feature_name,
            success=False,
            error_type=error_type,
            error_message=error_message,
            **kwargs
        )
    
    def _flush_events(self):
        """Flush buffered events to storage."""
        if not self._event_buffer:
            return
        
        try:
            # Create timestamped file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            events_file = self.storage_dir / f"events_{timestamp}_{uuid.uuid4().hex[:8]}.json"
            
            # Copy and clear buffer
            events_to_save = []
            with self._lock:
                events_to_save = [event.to_dict() for event in self._event_buffer]
                self._event_buffer.clear()
            
            # Save to file
            with open(events_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "session_id": self._session_id,
                    "session_start": self._session_start,
                    "flush_timestamp": time.time(),
                    "events": events_to_save
                }, f, indent=2)
            
            self._last_flush = time.time()
            self.logger.debug(f"Flushed {len(events_to_save)} events to {events_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to flush usage events: {e}")
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get current session summary."""
        session_duration = time.time() - self._session_start
        
        return {
            "session_id": self._session_id,
            "user_id": self._user_id,
            "session_duration_seconds": session_duration,
            "events_buffered": len(self._event_buffer),
            "last_flush": self._last_flush
        }
    
    def flush_and_close(self):
        """Flush remaining events and close tracker."""
        self.enabled = False
        self._flush_events()
        
        if self._flush_thread and self._flush_thread.is_alive():
            self._flush_thread.join(timeout=1.0)


class PerformanceAnalyzer:
    """Analyzes performance metrics from usage data."""
    
    def __init__(self, storage_dir: str = "usage_analytics"):
        self.storage_dir = Path(storage_dir)
        self.logger = logging.getLogger(__name__)
    
    def analyze_performance(self, time_window: str = "1d") -> List[PerformanceMetrics]:
        """Analyze performance metrics for specified time window."""
        cutoff_time = self._get_time_cutoff(time_window)
        events = self._load_events_since(cutoff_time)
        
        # Group events by feature
        feature_events = defaultdict(list)
        for event in events:
            if event.get("event_type") == "performance" or event.get("duration_ms", 0) > 0:
                feature_events[event.get("feature_name", "unknown")].append(event)
        
        # Calculate metrics for each feature
        performance_metrics = []
        for feature_name, feature_event_list in feature_events.items():
            if len(feature_event_list) < 5:  # Skip features with too few data points
                continue
            
            metrics = self._calculate_feature_performance(feature_name, feature_event_list, time_window)
            performance_metrics.append(metrics)
        
        return sorted(performance_metrics, key=lambda m: m.total_calls, reverse=True)
    
    def _get_time_cutoff(self, time_window: str) -> datetime:
        """Get cutoff time for analysis window."""
        now = datetime.now(timezone.utc)
        
        if time_window == "1h":
            return now - timedelta(hours=1)
        elif time_window == "1d":
            return now - timedelta(days=1)
        elif time_window == "1w":
            return now - timedelta(weeks=1)
        elif time_window == "1m":
            return now - timedelta(days=30)
        else:
            return now - timedelta(days=1)
    
    def _load_events_since(self, cutoff_time: datetime) -> List[Dict[str, Any]]:
        """Load events since specified cutoff time."""
        events = []
        
        try:
            for events_file in self.storage_dir.glob("events_*.json"):
                try:
                    with open(events_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Check if file is recent enough
                    file_time = datetime.fromtimestamp(data.get("flush_timestamp", 0), timezone.utc)
                    if file_time >= cutoff_time:
                        events.extend(data.get("events", []))
                
                except Exception as e:
                    self.logger.warning(f"Failed to load {events_file}: {e}")
            
        except Exception as e:
            self.logger.error(f"Failed to load usage events: {e}")
        
        return events
    
    def _calculate_feature_performance(self, feature_name: str, events: List[Dict[str, Any]], window: str) -> PerformanceMetrics:
        """Calculate performance metrics for a feature."""
        durations = [e.get("duration_ms", 0.0) for e in events if e.get("duration_ms", 0) > 0]
        memory_usage = [e.get("memory_used_mb", 0.0) for e in events if e.get("memory_used_mb", 0) > 0]
        cpu_usage = [e.get("cpu_usage_percent", 0.0) for e in events if e.get("cpu_usage_percent", 0) > 0]
        
        successful_events = [e for e in events if e.get("success", True)]
        failed_events = [e for e in events if not e.get("success", True)]
        
        # Calculate duration percentiles
        durations.sort()
        n = len(durations)
        
        metrics = PerformanceMetrics(
            feature_name=feature_name,
            measurement_window=window,
            total_calls=len(events),
            successful_calls=len(successful_events),
            failed_calls=len(failed_events),
            success_rate=len(successful_events) / len(events) if events else 0.0
        )
        
        if durations:
            metrics.avg_duration_ms = sum(durations) / len(durations)
            metrics.min_duration_ms = durations[0]
            metrics.max_duration_ms = durations[-1]
            metrics.p50_duration_ms = durations[int(n * 0.5)] if n > 0 else 0.0
            metrics.p95_duration_ms = durations[int(n * 0.95)] if n > 0 else 0.0
            metrics.p99_duration_ms = durations[int(n * 0.99)] if n > 0 else 0.0
        
        if memory_usage:
            metrics.avg_memory_mb = sum(memory_usage) / len(memory_usage)
            metrics.max_memory_mb = max(memory_usage)
        
        if cpu_usage:
            metrics.avg_cpu_percent = sum(cpu_usage) / len(cpu_usage)
            metrics.max_cpu_percent = max(cpu_usage)
        
        # Analyze error patterns
        error_types = Counter(e.get("error_type", "unknown") for e in failed_events if e.get("error_type"))
        metrics.error_types = dict(error_types.most_common(10))
        metrics.top_errors = [error for error, _ in error_types.most_common(5)]
        
        return metrics


class BehaviorAnalyzer:
    """Analyzes user behavior patterns."""
    
    def __init__(self, storage_dir: str = "usage_analytics"):
        self.storage_dir = Path(storage_dir)
        self.logger = logging.getLogger(__name__)
    
    def analyze_behavior_patterns(self, time_window: str = "1w") -> List[UserBehaviorPattern]:
        """Analyze user behavior patterns."""
        cutoff_time = self._get_time_cutoff(time_window)
        events = self._load_events_since(cutoff_time)
        
        patterns = []
        
        # Analyze workflow patterns
        workflow_patterns = self._analyze_workflow_patterns(events)
        patterns.extend(workflow_patterns)
        
        # Analyze usage frequency patterns
        frequency_patterns = self._analyze_frequency_patterns(events)
        patterns.extend(frequency_patterns)
        
        # Analyze time-based patterns
        time_patterns = self._analyze_time_patterns(events)
        patterns.extend(time_patterns)
        
        return patterns
    
    def _get_time_cutoff(self, time_window: str) -> datetime:
        """Get cutoff time for analysis window."""
        now = datetime.now(timezone.utc)
        
        if time_window == "1d":
            return now - timedelta(days=1)
        elif time_window == "1w":
            return now - timedelta(weeks=1)
        elif time_window == "1m":
            return now - timedelta(days=30)
        else:
            return now - timedelta(weeks=1)
    
    def _load_events_since(self, cutoff_time: datetime) -> List[Dict[str, Any]]:
        """Load events since specified cutoff time."""
        events = []
        
        try:
            for events_file in self.storage_dir.glob("events_*.json"):
                try:
                    with open(events_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Check if file is recent enough
                    file_time = datetime.fromtimestamp(data.get("flush_timestamp", 0), timezone.utc)
                    if file_time >= cutoff_time:
                        events.extend(data.get("events", []))
                
                except Exception as e:
                    self.logger.warning(f"Failed to load {events_file}: {e}")
            
        except Exception as e:
            self.logger.error(f"Failed to load usage events: {e}")
        
        return events
    
    def _analyze_workflow_patterns(self, events: List[Dict[str, Any]]) -> List[UserBehaviorPattern]:
        """Analyze common workflow patterns."""
        patterns = []
        
        # Group events by session
        session_events = defaultdict(list)
        for event in events:
            session_id = event.get("session_id", "unknown")
            session_events[session_id].append(event)
        
        # Find common feature sequences
        feature_sequences = []
        for session_id, session_event_list in session_events.items():
            # Sort by timestamp
            session_event_list.sort(key=lambda e: e.get("timestamp", ""))
            sequence = [e.get("feature_name", "unknown") for e in session_event_list]
            if len(sequence) >= 3:  # Only consider meaningful sequences
                feature_sequences.append(sequence)
        
        # Find common subsequences
        sequence_patterns = self._find_common_subsequences(feature_sequences)
        
        for pattern_seq, frequency in sequence_patterns.items():
            if frequency >= 3:  # Minimum frequency threshold
                pattern = UserBehaviorPattern(
                    pattern_type="workflow",
                    pattern_name=f"Workflow: {' → '.join(pattern_seq)}",
                    frequency=frequency,
                    confidence_score=min(frequency / len(feature_sequences), 1.0),
                    common_features=list(pattern_seq),
                    optimization_opportunities=[
                        "Consider creating workflow shortcuts",
                        "Optimize feature transitions",
                        "Add workflow templates"
                    ]
                )
                patterns.append(pattern)
        
        return patterns
    
    def _analyze_frequency_patterns(self, events: List[Dict[str, Any]]) -> List[UserBehaviorPattern]:
        """Analyze feature usage frequency patterns."""
        patterns = []
        
        # Count feature usage
        feature_counts = Counter(e.get("feature_name", "unknown") for e in events)
        total_events = len(events)
        
        # Identify high-frequency features
        for feature, count in feature_counts.most_common(10):
            if count >= 10:  # Minimum usage threshold
                pattern = UserBehaviorPattern(
                    pattern_type="frequency_pattern",
                    pattern_name=f"High Usage: {feature}",
                    frequency=count,
                    confidence_score=count / total_events,
                    common_features=[feature],
                    impact_score=count / total_events * 100,
                    optimization_opportunities=[
                        "Optimize performance for high-usage features",
                        "Consider advanced features for power users",
                        "Monitor for potential bottlenecks"
                    ]
                )
                patterns.append(pattern)
        
        return patterns
    
    def _analyze_time_patterns(self, events: List[Dict[str, Any]]) -> List[UserBehaviorPattern]:
        """Analyze time-based usage patterns."""
        patterns = []
        
        # Group events by hour of day
        hour_counts: Dict[int, int] = defaultdict(int)
        for event in events:
            try:
                timestamp = datetime.fromisoformat(event.get("timestamp", "").replace("Z", "+00:00"))
                hour = timestamp.hour
                hour_counts[hour] += 1
            except Exception:
                continue
        
        if hour_counts:
            # Find peak hours
            peak_hour = max(hour_counts.keys(), key=lambda h: hour_counts[h])
            peak_count = hour_counts[peak_hour]
            total_count = sum(hour_counts.values())
            
            if peak_count > total_count * 0.15:  # Peak hour has >15% of total usage
                pattern = UserBehaviorPattern(
                    pattern_type="time_pattern",
                    pattern_name=f"Peak Usage: {peak_hour}:00-{peak_hour+1}:00",
                    frequency=peak_count,
                    confidence_score=peak_count / total_count,
                    time_patterns={"peak_hour": peak_hour, "peak_percentage": peak_count / total_count},
                    optimization_opportunities=[
                        "Scale resources during peak hours",
                        "Optimize performance for peak times",
                        "Consider load balancing strategies"
                    ]
                )
                patterns.append(pattern)
        
        return patterns
    
    def _find_common_subsequences(self, sequences: List[List[str]], min_length: int = 2) -> Dict[tuple, int]:
        """Find common subsequences in feature usage sequences."""
        subsequence_counts: Dict[tuple, int] = {}
        
        for sequence in sequences:
            # Generate all subsequences of minimum length
            for i in range(len(sequence)):
                for j in range(i + min_length, min(i + 5, len(sequence) + 1)):  # Max length 5
                    subseq = tuple(sequence[i:j])
                    subsequence_counts[subseq] = subsequence_counts.get(subseq, 0) + 1
        
        # Return only subsequences that appear multiple times
        return {seq: count for seq, count in subsequence_counts.items() if count > 1}


def track_usage(feature_name: str, **kwargs):
    """Decorator for automatic usage tracking."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **func_kwargs):
            tracker = get_usage_tracker()
            start_time = time.time()
            
            try:
                result = func(*args, **func_kwargs)
                duration_ms = (time.time() - start_time) * 1000
                
                tracker.track_performance(
                    feature_name=feature_name,
                    duration_ms=duration_ms,
                    success=True,
                    **kwargs
                )
                
                return result
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                
                tracker.track_error(
                    feature_name=feature_name,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    duration_ms=duration_ms,
                    **kwargs
                )
                
                raise
        
        return wrapper
    return decorator


# Global usage tracker instance
_usage_tracker: Optional[UsageTracker] = None


def get_usage_tracker() -> UsageTracker:
    """Get global usage tracker instance."""
    global _usage_tracker
    if _usage_tracker is None:
        _usage_tracker = UsageTracker()
    return _usage_tracker


def initialize_usage_tracking(storage_dir: str = "usage_analytics", enabled: bool = True) -> UsageTracker:
    """Initialize usage tracking system."""
    global _usage_tracker
    if _usage_tracker is not None:
        _usage_tracker.flush_and_close()
    
    _usage_tracker = UsageTracker(storage_dir=storage_dir, enabled=enabled)
    return _usage_tracker


def shutdown_usage_tracking():
    """Shutdown usage tracking system."""
    global _usage_tracker
    if _usage_tracker is not None:
        _usage_tracker.flush_and_close()
        _usage_tracker = None
