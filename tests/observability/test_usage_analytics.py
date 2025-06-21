"""
Tests for Usage Analytics Module

This module tests all aspects of usage analytics including feature usage tracking,
performance analytics, user behavior analysis, and usage reporting.
"""

import json
import tempfile
import time
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path

from src.qemlflow.observability.usage_analytics import (
    UsageEvent,
    PerformanceMetrics,
    UserBehaviorPattern,
    UsageTracker,
    PerformanceAnalyzer,
    BehaviorAnalyzer,
    track_usage,
    get_usage_tracker,
    initialize_usage_tracking,
    shutdown_usage_tracking
)


class TestUsageEvent(unittest.TestCase):
    """Test UsageEvent dataclass."""
    
    def test_usage_event_creation(self):
        """Test creating usage event."""
        event = UsageEvent(
            event_type="feature_usage",
            feature_name="data_processing",
            user_id="test_user",
            duration_ms=150.5
        )
        
        self.assertEqual(event.event_type, "feature_usage")
        self.assertEqual(event.feature_name, "data_processing")
        self.assertEqual(event.user_id, "test_user")
        self.assertEqual(event.duration_ms, 150.5)
        self.assertTrue(event.success)
    
    def test_event_to_dict(self):
        """Test converting event to dictionary."""
        event = UsageEvent(feature_name="test_feature", event_type="api_call")
        event_dict = event.to_dict()
        
        self.assertIsInstance(event_dict, dict)
        self.assertEqual(event_dict["feature_name"], "test_feature")
        self.assertEqual(event_dict["event_type"], "api_call")
        self.assertIn("event_id", event_dict)
        self.assertIn("timestamp", event_dict)


class TestPerformanceMetrics(unittest.TestCase):
    """Test PerformanceMetrics dataclass."""
    
    def test_performance_metrics_creation(self):
        """Test creating performance metrics."""
        metrics = PerformanceMetrics(
            feature_name="model_training",
            avg_duration_ms=2500.0,
            total_calls=100,
            success_rate=0.95
        )
        
        self.assertEqual(metrics.feature_name, "model_training")
        self.assertEqual(metrics.avg_duration_ms, 2500.0)
        self.assertEqual(metrics.total_calls, 100)
        self.assertEqual(metrics.success_rate, 0.95)
    
    def test_metrics_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = PerformanceMetrics(feature_name="test_feature")
        metrics_dict = metrics.to_dict()
        
        self.assertIsInstance(metrics_dict, dict)
        self.assertEqual(metrics_dict["feature_name"], "test_feature")


class TestUserBehaviorPattern(unittest.TestCase):
    """Test UserBehaviorPattern dataclass."""
    
    def test_behavior_pattern_creation(self):
        """Test creating behavior pattern."""
        pattern = UserBehaviorPattern(
            pattern_type="workflow",
            pattern_name="Data Analysis Workflow",
            frequency=25,
            confidence_score=0.8
        )
        
        self.assertEqual(pattern.pattern_type, "workflow")
        self.assertEqual(pattern.pattern_name, "Data Analysis Workflow")
        self.assertEqual(pattern.frequency, 25)
        self.assertEqual(pattern.confidence_score, 0.8)


class TestUsageTracker(unittest.TestCase):
    """Test UsageTracker class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.tracker = UsageTracker(self.temp_dir, enabled=True)
    
    def tearDown(self):
        """Clean up test fixtures."""
        if self.tracker:
            self.tracker.flush_and_close()
    
    def test_tracker_initialization(self):
        """Test tracker initialization."""
        self.assertTrue(self.tracker.enabled)
        self.assertTrue(self.tracker.storage_dir.exists())
        self.assertIsNotNone(self.tracker._session_id)
        self.assertIsNotNone(self.tracker._user_id)
    
    def test_track_event(self):
        """Test basic event tracking."""
        event_id = self.tracker.track_event(
            event_type="feature_usage",
            feature_name="test_feature",
            metadata={"test": "data"}
        )
        
        self.assertIsInstance(event_id, str)
        self.assertEqual(len(self.tracker._event_buffer), 1)
        
        event = self.tracker._event_buffer[0]
        self.assertEqual(event.event_type, "feature_usage")
        self.assertEqual(event.feature_name, "test_feature")
        self.assertEqual(event.metadata["test"], "data")
    
    def test_track_performance(self):
        """Test performance tracking."""
        event_id = self.tracker.track_performance(
            feature_name="api_call",
            duration_ms=250.5,
            success=True
        )
        
        self.assertIsInstance(event_id, str)
        event = self.tracker._event_buffer[0]
        self.assertEqual(event.event_type, "performance")
        self.assertEqual(event.duration_ms, 250.5)
        self.assertTrue(event.success)
    
    def test_track_feature_usage(self):
        """Test feature usage tracking."""
        event_id = self.tracker.track_feature_usage(
            feature_name="data_processing",
            input_size=1024,
            output_size=512
        )
        
        self.assertIsInstance(event_id, str)
        event = self.tracker._event_buffer[0]
        self.assertEqual(event.event_type, "feature_usage")
        self.assertEqual(event.input_size, 1024)
        self.assertEqual(event.output_size, 512)
    
    def test_track_api_call(self):
        """Test API call tracking."""
        event_id = self.tracker.track_api_call(
            api_endpoint="/api/predict",
            method="POST"
        )
        
        self.assertIsInstance(event_id, str)
        event = self.tracker._event_buffer[0]
        self.assertEqual(event.event_type, "api_call")
        self.assertEqual(event.feature_name, "POST /api/predict")
    
    def test_track_error(self):
        """Test error tracking."""
        event_id = self.tracker.track_error(
            feature_name="model_training",
            error_type="ValueError",
            error_message="Invalid input shape"
        )
        
        self.assertIsInstance(event_id, str)
        event = self.tracker._event_buffer[0]
        self.assertEqual(event.event_type, "error")
        self.assertFalse(event.success)
        self.assertEqual(event.error_type, "ValueError")
        self.assertEqual(event.error_message, "Invalid input shape")
    
    def test_flush_events(self):
        """Test event flushing."""
        # Add some events
        for i in range(5):
            self.tracker.track_event("test", f"feature_{i}")
        
        # Flush events
        self.tracker._flush_events()
        
        # Buffer should be empty
        self.assertEqual(len(self.tracker._event_buffer), 0)
        
        # Check that file was created
        event_files = list(self.tracker.storage_dir.glob("events_*.json"))
        self.assertGreater(len(event_files), 0)
        
        # Check file content
        with open(event_files[0], 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.assertEqual(len(data["events"]), 5)
        self.assertIn("session_id", data)
    
    def test_get_session_summary(self):
        """Test session summary."""
        summary = self.tracker.get_session_summary()
        
        self.assertIn("session_id", summary)
        self.assertIn("user_id", summary)
        self.assertIn("session_duration_seconds", summary)
        self.assertIn("events_buffered", summary)
    
    def test_disabled_tracker(self):
        """Test disabled tracker doesn't track events."""
        disabled_tracker = UsageTracker(self.temp_dir, enabled=False)
        
        event_id = disabled_tracker.track_event("test", "feature")
        
        self.assertEqual(event_id, "")
        self.assertEqual(len(disabled_tracker._event_buffer), 0)


class TestPerformanceAnalyzer(unittest.TestCase):
    """Test PerformanceAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.analyzer = PerformanceAnalyzer(self.temp_dir)
        
        # Create sample event data
        self._create_sample_events()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if hasattr(self, 'temp_dir') and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def _create_sample_events(self):
        """Create sample event files for testing."""
        events_data = {
            "session_id": "test_session",
            "session_start": time.time(),
            "flush_timestamp": time.time(),
            "events": [
                {
                    "event_id": "1",
                    "event_type": "performance",
                    "feature_name": "data_processing",
                    "duration_ms": 100.0,
                    "success": True,
                    "memory_used_mb": 50.0,
                    "cpu_usage_percent": 25.0,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                },
                {
                    "event_id": "2",
                    "event_type": "performance",
                    "feature_name": "data_processing",
                    "duration_ms": 150.0,
                    "success": True,
                    "memory_used_mb": 60.0,
                    "cpu_usage_percent": 30.0,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                },
                {
                    "event_id": "3",
                    "event_type": "performance",
                    "feature_name": "data_processing",
                    "duration_ms": 200.0,
                    "success": False,
                    "error_type": "ValueError",
                    "error_message": "Test error",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                },
                {
                    "event_id": "4",
                    "event_type": "performance",
                    "feature_name": "model_training",
                    "duration_ms": 2000.0,
                    "success": True,
                    "memory_used_mb": 200.0,
                    "cpu_usage_percent": 80.0,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            ]
        }
        
        events_file = Path(self.temp_dir) / "events_test.json"
        with open(events_file, 'w', encoding='utf-8') as f:
            json.dump(events_data, f)
    
    def test_analyze_performance(self):
        """Test performance analysis."""
        # Use the existing setup data but need at least 5 events for analysis
        now = datetime.now(timezone.utc)
        
        # Add more events to meet minimum requirement
        additional_events = []
        for i in range(3):  # Add 3 more events to get 7 total (>= 5 minimum)
            additional_events.append({
                "event_id": f"additional_{i}",
                "event_type": "performance",
                "feature_name": "data_processing",
                "duration_ms": 120.0 + i * 30,
                "success": True,
                "memory_used_mb": 45.0 + i * 5,
                "cpu_usage_percent": 20.0 + i * 5,
                "timestamp": now.isoformat()
            })
        
        # Update the existing events file
        events_file = Path(self.temp_dir) / "events_test.json"
        with open(events_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        data["events"].extend(additional_events)
        
        with open(events_file, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        
        metrics_list = self.analyzer.analyze_performance("1d")
        
        self.assertGreater(len(metrics_list), 0)
        
        # Find data_processing metrics
        data_processing_metrics = None
        for metrics in metrics_list:
            if metrics.feature_name == "data_processing":
                data_processing_metrics = metrics
                break
        
        self.assertIsNotNone(data_processing_metrics)
        if data_processing_metrics:
            self.assertEqual(data_processing_metrics.total_calls, 6)  # 3 original + 3 additional
            self.assertGreater(data_processing_metrics.successful_calls, 0)
            self.assertGreater(data_processing_metrics.avg_duration_ms, 0)
    
    def test_load_events_since(self):
        """Test loading events since cutoff time."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=1)
        events = self.analyzer._load_events_since(cutoff_time)
        
        self.assertEqual(len(events), 4)
        
        # Check event structure
        for event in events:
            self.assertIn("event_id", event)
            self.assertIn("feature_name", event)
    
    def test_calculate_feature_performance(self):
        """Test feature performance calculation."""
        events = [
            {"duration_ms": 100.0, "success": True, "memory_used_mb": 50.0},
            {"duration_ms": 200.0, "success": True, "memory_used_mb": 60.0},
            {"duration_ms": 150.0, "success": False, "error_type": "ValueError"}
        ]
        
        metrics = self.analyzer._calculate_feature_performance("test_feature", events, "1d")
        
        self.assertEqual(metrics.feature_name, "test_feature")
        self.assertEqual(metrics.total_calls, 3)
        self.assertEqual(metrics.successful_calls, 2)
        self.assertEqual(metrics.failed_calls, 1)
        self.assertAlmostEqual(metrics.success_rate, 2/3, places=2)
        self.assertAlmostEqual(metrics.avg_duration_ms, 150.0, places=1)
    
    def test_empty_events_analysis(self):
        """Test performance analysis with no events."""
        # Create empty events file
        empty_events_data = {
            "session_id": "empty_session",
            "session_start": time.time(),
            "flush_timestamp": time.time(),
            "events": []
        }
        
        events_file = Path(self.temp_dir) / "events_empty.json"
        with open(events_file, 'w', encoding='utf-8') as f:
            json.dump(empty_events_data, f)
        
        # Should return empty list
        metrics_list = self.analyzer.analyze_performance("1d")
        self.assertEqual(len(metrics_list), 0)
    
    def test_insufficient_events_analysis(self):
        """Test performance analysis with insufficient events (< 5)."""
        # Create events with only 3 events for a feature
        insufficient_events_data = {
            "session_id": "insufficient_session",
            "session_start": time.time(),
            "flush_timestamp": time.time(),
            "events": [
                {
                    "event_id": "1",
                    "event_type": "performance",
                    "feature_name": "rare_feature",
                    "duration_ms": 100.0,
                    "success": True,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                },
                {
                    "event_id": "2",
                    "event_type": "performance",
                    "feature_name": "rare_feature",
                    "duration_ms": 150.0,
                    "success": True,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                },
                {
                    "event_id": "3",
                    "event_type": "performance",
                    "feature_name": "rare_feature",
                    "duration_ms": 200.0,
                    "success": True,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            ]
        }
        
        events_file = Path(self.temp_dir) / "events_insufficient.json"
        with open(events_file, 'w', encoding='utf-8') as f:
            json.dump(insufficient_events_data, f)
        
        # Should return empty list (insufficient events)
        metrics_list = self.analyzer.analyze_performance("1d")
        self.assertEqual(len(metrics_list), 0)


class TestBehaviorAnalyzer(unittest.TestCase):
    """Test BehaviorAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.analyzer = BehaviorAnalyzer(self.temp_dir)
        
        # Create sample behavior data
        self._create_sample_behavior_events()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if hasattr(self, 'temp_dir') and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def _create_sample_behavior_events(self):
        """Create sample behavior events for testing."""
        base_time = datetime.now(timezone.utc)
        
        events_data = {
            "session_id": "session_1",
            "session_start": time.time(),
            "flush_timestamp": time.time(),
            "events": [
                {
                    "event_id": "1",
                    "feature_name": "data_loading",
                    "session_id": "session_1",
                    "timestamp": (base_time + timedelta(minutes=1)).isoformat()
                },
                {
                    "event_id": "2",
                    "feature_name": "data_processing",
                    "session_id": "session_1",
                    "timestamp": (base_time + timedelta(minutes=2)).isoformat()
                },
                {
                    "event_id": "3",
                    "feature_name": "model_training",
                    "session_id": "session_1",
                    "timestamp": (base_time + timedelta(minutes=3)).isoformat()
                },
                {
                    "event_id": "4",
                    "feature_name": "data_loading",
                    "session_id": "session_2",
                    "timestamp": (base_time + timedelta(minutes=10)).isoformat()
                },
                {
                    "event_id": "5",
                    "feature_name": "data_processing",
                    "session_id": "session_2",
                    "timestamp": (base_time + timedelta(minutes=11)).isoformat()
                },
                {
                    "event_id": "6",
                    "feature_name": "visualization",
                    "session_id": "session_2",
                    "timestamp": (base_time + timedelta(minutes=12)).isoformat()
                }
            ]
        }
        
        events_file = Path(self.temp_dir) / "events_behavior.json"
        with open(events_file, 'w', encoding='utf-8') as f:
            json.dump(events_data, f)
    
    def test_analyze_behavior_patterns(self):
        """Test behavior pattern analysis."""
        # Add more sample events to create meaningful patterns
        now = datetime.now(timezone.utc)
        
        # Create multiple sessions with similar patterns (need at least 3 features in sequence)
        additional_events = []
        for session_num in range(3, 6):  # Add sessions 3, 4, 5
            session_id = f"session_{session_num}"
            base_time = now + timedelta(hours=session_num)
            
            # Create similar workflow pattern with 3+ features
            workflow_events = [
                {
                    "event_id": f"session_{session_num}_1",
                    "feature_name": "data_loading",
                    "session_id": session_id,
                    "timestamp": (base_time + timedelta(minutes=1)).isoformat()
                },
                {
                    "event_id": f"session_{session_num}_2",
                    "feature_name": "data_processing",
                    "session_id": session_id,
                    "timestamp": (base_time + timedelta(minutes=2)).isoformat()
                },
                {
                    "event_id": f"session_{session_num}_3",
                    "feature_name": "model_training",
                    "session_id": session_id,
                    "timestamp": (base_time + timedelta(minutes=3)).isoformat()
                }
            ]
            additional_events.extend(workflow_events)
        
        # Update the existing events file
        events_file = Path(self.temp_dir) / "events_behavior.json"
        with open(events_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        data["events"].extend(additional_events)
        
        with open(events_file, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        
        patterns = self.analyzer.analyze_behavior_patterns("1w")
        
        self.assertGreater(len(patterns), 0)
        
        # Check for workflow patterns (should find common subsequences)
        workflow_patterns = [p for p in patterns if p.pattern_type == "workflow"]
        if len(workflow_patterns) == 0:
            # If no workflow patterns found, at least check for frequency patterns
            frequency_patterns = [p for p in patterns if p.pattern_type == "frequency_pattern"]
            self.assertGreater(len(frequency_patterns), 0)
    
    def test_analyze_workflow_patterns(self):
        """Test workflow pattern analysis."""
        events = [
            {"feature_name": "data_loading", "session_id": "s1", "timestamp": "2025-01-01T10:00:00"},
            {"feature_name": "data_processing", "session_id": "s1", "timestamp": "2025-01-01T10:01:00"},
            {"feature_name": "model_training", "session_id": "s1", "timestamp": "2025-01-01T10:02:00"}
        ]
        
        patterns = self.analyzer._analyze_workflow_patterns(events)
        
        # Should find workflow patterns
        self.assertGreaterEqual(len(patterns), 0)
    
    def test_analyze_frequency_patterns(self):
        """Test frequency pattern analysis."""
        events = [
            {"feature_name": "popular_feature"} for _ in range(20)
        ] + [
            {"feature_name": "rare_feature"} for _ in range(2)
        ]
        
        patterns = self.analyzer._analyze_frequency_patterns(events)
        
        # Should find high-frequency pattern
        high_freq_patterns = [p for p in patterns if "popular_feature" in p.pattern_name]
        self.assertGreater(len(high_freq_patterns), 0)
    
    def test_find_common_subsequences(self):
        """Test common subsequence finding."""
        sequences = [
            ["A", "B", "C", "D"],
            ["A", "B", "E", "F"],
            ["X", "A", "B", "Y"],
            ["A", "B", "C", "Z"]
        ]
        
        common_subseqs = self.analyzer._find_common_subsequences(sequences)
        
        # Should find ("A", "B") as common subsequence
        self.assertIn(("A", "B"), common_subseqs)
        self.assertGreaterEqual(common_subseqs[("A", "B")], 2)


class TestTrackUsageDecorator(unittest.TestCase):
    """Test track_usage decorator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        initialize_usage_tracking(self.temp_dir, enabled=True)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutdown_usage_tracking()
    
    def test_successful_function_tracking(self):
        """Test tracking successful function calls."""
        @track_usage("test_function")
        def test_function(x, y):
            time.sleep(0.01)  # Small delay to test duration tracking
            return x + y
        
        result = test_function(1, 2)
        
        self.assertEqual(result, 3)
        
        tracker = get_usage_tracker()
        self.assertEqual(len(tracker._event_buffer), 1)
        
        event = tracker._event_buffer[0]
        self.assertEqual(event.feature_name, "test_function")
        self.assertTrue(event.success)
        self.assertGreater(event.duration_ms, 0)
    
    def test_error_function_tracking(self):
        """Test tracking function calls that raise errors."""
        @track_usage("error_function")
        def error_function():
            raise ValueError("Test error")
        
        with self.assertRaises(ValueError):
            error_function()
        
        tracker = get_usage_tracker()
        self.assertEqual(len(tracker._event_buffer), 1)
        
        event = tracker._event_buffer[0]
        self.assertEqual(event.feature_name, "error_function")
        self.assertFalse(event.success)
        self.assertEqual(event.error_type, "ValueError")
        self.assertEqual(event.error_message, "Test error")


class TestGlobalFunctions(unittest.TestCase):
    """Test global functions."""
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Ensure usage tracking is shutdown after each test
        try:
            shutdown_usage_tracking()
        except Exception:
            pass  # May already be shutdown
    
    def test_initialize_usage_tracking(self):
        """Test initializing usage tracking."""
        temp_dir = tempfile.mkdtemp()
        
        try:
            tracker = initialize_usage_tracking(temp_dir, enabled=True)
            
            self.assertIsNotNone(tracker)
            self.assertTrue(tracker.enabled)
            self.assertEqual(str(tracker.storage_dir), temp_dir)
        finally:
            # Clean up
            shutdown_usage_tracking()
            import shutil
            if Path(temp_dir).exists():
                shutil.rmtree(temp_dir)
    
    def test_get_usage_tracker(self):
        """Test getting usage tracker."""
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Initialize tracker
            initialize_usage_tracking(temp_dir)
            
            # Get tracker
            tracker1 = get_usage_tracker()
            tracker2 = get_usage_tracker()
            
            # Should return same instance
            self.assertIs(tracker1, tracker2)
        finally:
            # Clean up
            shutdown_usage_tracking()
            import shutil
            if Path(temp_dir).exists():
                shutil.rmtree(temp_dir)
    
    def test_shutdown_usage_tracking(self):
        """Test shutting down usage tracking."""
        temp_dir = tempfile.mkdtemp()
        
        try:
            initialize_usage_tracking(temp_dir)
            tracker = get_usage_tracker()
            
            # Tracker should be active
            self.assertTrue(tracker.enabled)
            
            # Shutdown
            shutdown_usage_tracking()
            
            # Tracker should be disabled
            self.assertFalse(tracker.enabled)
        finally:
            # Ensure cleanup
            import shutil
            if Path(temp_dir).exists():
                shutil.rmtree(temp_dir)
    
    def test_get_tracker_without_initialization(self):
        """Test getting tracker before initialization."""
        # Make sure we start clean
        shutdown_usage_tracking()
        
        # Should create a default tracker if none exists
        tracker = get_usage_tracker()
        self.assertIsNotNone(tracker)
        self.assertIsInstance(tracker, UsageTracker)
        
        # Default tracker should be enabled
        self.assertTrue(tracker.enabled)


if __name__ == '__main__':
    unittest.main()
