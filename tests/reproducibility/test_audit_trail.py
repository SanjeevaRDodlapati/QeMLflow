"""
Comprehensive test suite for the audit trail system.

Tests cover:
1. Audit event creation and logging
2. Data lineage tracking
3. Workflow tracking
4. Core functionality
5. Integration capabilities
"""

import json
import os
import shutil
import tempfile
import threading
import time
from unittest import TestCase

import pytest

from qemlflow.reproducibility.audit_trail import (
    AuditEvent,
    DataLineage,
    WorkflowStep,
    AuditTrailManager,
    audit_trail,
    audit_workflow,
    get_audit_manager,
    log_audit_event,
    track_data_lineage
)


class TestAuditEvent(TestCase):
    """Test AuditEvent dataclass functionality."""

    def test_audit_event_creation(self):
        """Test basic audit event creation."""
        event = AuditEvent(
            event_type="test",
            action="create",
            resource="test_resource",
            resource_type="file"
        )
        
        self.assertEqual(event.event_type, "test")
        self.assertEqual(event.action, "create")
        self.assertEqual(event.resource, "test_resource")
        self.assertEqual(event.resource_type, "file")
        
        # Check auto-generated fields
        self.assertIsNotNone(event.event_id)
        self.assertIsNotNone(event.timestamp)
        self.assertGreater(event.process_id, 0)
        self.assertGreater(event.thread_id, 0)

    def test_audit_event_serialization(self):
        """Test audit event JSON serialization."""
        event = AuditEvent(
            event_type="test",
            action="create",
            resource="test_resource",
            resource_type="file",
            metadata={"key": "value"}
        )
        
        # Convert to dict
        event_dict = event.to_dict()
        self.assertIsInstance(event_dict, dict)
        self.assertEqual(event_dict["event_type"], "test")
        self.assertEqual(event_dict["action"], "create")
        
        # Convert to JSON using dict
        event_json = json.dumps(event_dict)
        self.assertIsInstance(event_json, str)
        
        # Parse JSON back
        parsed = json.loads(event_json)
        self.assertEqual(parsed["event_type"], "test")
        self.assertEqual(parsed["action"], "create")

    def test_audit_event_checksum(self):
        """Test audit event checksum calculation."""
        event = AuditEvent(
            event_type="test",
            action="create",
            resource="test_resource",
            resource_type="file"
        )
        
        # Check that checksum is calculated
        self.assertIsNotNone(event.checksum)
        self.assertEqual(len(event.checksum), 64)  # SHA-256 hex length


class TestDataLineage(TestCase):
    """Test DataLineage tracking functionality."""

    def test_data_lineage_creation(self):
        """Test basic data lineage creation."""
        lineage = DataLineage(
            data_id="test_data_001",
            name="Test Dataset",
            path="/path/to/data.csv"
        )
        
        self.assertEqual(lineage.data_id, "test_data_001")
        self.assertEqual(lineage.name, "Test Dataset")
        self.assertEqual(lineage.path, "/path/to/data.csv")

    def test_data_lineage_relationships(self):
        """Test data lineage parent-child relationships."""
        parent = DataLineage(
            data_id="parent_001",
            name="Parent Dataset"
        )
        
        child = DataLineage(
            data_id="child_001",
            name="Child Dataset",
            transformation="data_processing"
        )
        
        # Add parent relationship
        child.add_parent(parent.data_id)
        
        self.assertIn(parent.data_id, child.parent_data)
        self.assertEqual(child.transformation, "data_processing")

    def test_data_lineage_serialization(self):
        """Test data lineage serialization."""
        lineage = DataLineage(
            data_id="test_data_001",
            name="Test Dataset",
            path="/path/to/data.csv"
        )
        
        # Convert to dict
        lineage_dict = lineage.to_dict()
        self.assertIsInstance(lineage_dict, dict)
        self.assertEqual(lineage_dict["data_id"], "test_data_001")
        self.assertEqual(lineage_dict["name"], "Test Dataset")


class TestWorkflowStep(TestCase):
    """Test WorkflowStep tracking functionality."""

    def test_workflow_step_creation(self):
        """Test basic workflow step creation."""
        step = WorkflowStep(
            step_id="step_001",
            name="Data Loading"
        )
        
        self.assertEqual(step.step_id, "step_001")
        self.assertEqual(step.name, "Data Loading")
        self.assertEqual(step.status, "running")

    def test_workflow_step_completion(self):
        """Test workflow step completion."""
        step = WorkflowStep(
            step_id="step_001",
            name="Test Step"
        )
        
        # Complete step
        step.complete(status="completed")
        
        self.assertEqual(step.status, "completed")
        self.assertIsNotNone(step.completed_at)
        self.assertGreaterEqual(step.duration, 0)

    def test_workflow_step_failure(self):
        """Test workflow step failure handling."""
        step = WorkflowStep(
            step_id="step_001",
            name="Test Step"
        )
        
        # Fail step
        error_message = "Test error"
        step.complete(status="failed", error_message=error_message)
        
        self.assertEqual(step.status, "failed")
        self.assertEqual(step.error_message, error_message)
        self.assertIsNotNone(step.completed_at)

    def test_workflow_step_serialization(self):
        """Test workflow step serialization."""
        step = WorkflowStep(
            step_id="step_001",
            name="Test Step"
        )
        
        # Convert to dict
        step_dict = step.to_dict()
        self.assertIsInstance(step_dict, dict)
        self.assertEqual(step_dict["step_id"], "step_001")
        self.assertEqual(step_dict["name"], "Test Step")


class TestAuditTrailManager(TestCase):
    """Test AuditTrailManager functionality."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = AuditTrailManager(audit_dir=self.temp_dir)

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)

    def test_manager_initialization(self):
        """Test audit trail manager initialization."""
        self.assertIsInstance(self.manager, AuditTrailManager)
        self.assertTrue(os.path.exists(self.temp_dir))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "events")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "lineage")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "workflows")))

    def test_log_event(self):
        """Test basic event logging."""
        event_id = self.manager.log_event(
            action="test_action",
            resource="test_resource",
            resource_type="test",
            metadata={"key": "value"}
        )
        
        self.assertIsNotNone(event_id)
        # Check that event file was created
        events_dir = os.path.join(self.temp_dir, "events")
        event_files = os.listdir(events_dir)
        self.assertGreater(len(event_files), 0)

    def test_data_lineage_tracking(self):
        """Test data lineage tracking."""
        lineage = self.manager.track_data_lineage(
            data_id="test_data_001",
            name="Test Dataset",
            path="/path/to/data.csv"
        )
        
        self.assertIsInstance(lineage, DataLineage)
        self.assertEqual(lineage.data_id, "test_data_001")

    def test_multiple_events(self):
        """Test logging multiple events."""
        # Log multiple events
        event_ids = []
        for i in range(5):
            event_id = self.manager.log_event(
                action=f"action_{i}",
                resource=f"resource_{i}",
                resource_type="test"
            )
            event_ids.append(event_id)
        
        # Check all events were logged
        self.assertEqual(len(event_ids), 5)
        self.assertEqual(len(set(event_ids)), 5)  # All unique

    def test_concurrent_logging(self):
        """Test concurrent event logging."""
        event_ids = []
        
        def log_events(thread_id, num_events=5):
            for i in range(num_events):
                event_id = self.manager.log_event(
                    action=f"thread_{thread_id}_action_{i}",
                    resource=f"resource_{i}",
                    thread_id=thread_id
                )
                event_ids.append(event_id)
        
        # Create multiple threads
        threads = []
        num_threads = 3
        events_per_thread = 5
        
        for i in range(num_threads):
            thread = threading.Thread(target=log_events, args=(i, events_per_thread))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all events were logged
        expected_total = num_threads * events_per_thread
        self.assertEqual(len(event_ids), expected_total)


class TestAuditDecorators(TestCase):
    """Test audit trail decorators functionality."""

    def test_audit_trail_decorator(self):
        """Test @audit_trail decorator functionality."""
        @audit_trail(action="test_function", resource_type="function")
        def test_function(x, y):
            return x + y
        
        # Execute function
        result = test_function(2, 3)
        self.assertEqual(result, 5)

    def test_audit_workflow_context(self):
        """Test audit_workflow context manager."""
        # Test workflow context usage
        with audit_workflow("test_workflow"):
            time.sleep(0.01)  # Simulate some work
        
        # If no exception, context manager worked
        self.assertTrue(True)

    def test_decorator_error_handling(self):
        """Test decorator error handling."""
        @audit_trail(action="failing_function")
        def failing_function():
            raise ValueError("Test error")
        
        # Execute function and expect error
        with self.assertRaises(ValueError):
            failing_function()


class TestStandaloneAPI(TestCase):
    """Test standalone API functions."""

    def test_log_audit_event_function(self):
        """Test standalone log_audit_event function."""
        event_id = log_audit_event(
            action="standalone_test",
            resource="test_resource",
            metadata={"test": True}
        )
        
        self.assertIsNotNone(event_id)

    def test_track_data_lineage_function(self):
        """Test standalone track_data_lineage function."""
        lineage = track_data_lineage(
            data_id="standalone_data_001",
            name="Standalone Test Data"
        )
        
        self.assertIsInstance(lineage, DataLineage)
        self.assertEqual(lineage.data_id, "standalone_data_001")

    def test_get_audit_manager_function(self):
        """Test get_audit_manager function."""
        manager = get_audit_manager()
        self.assertIsInstance(manager, AuditTrailManager)


class TestConfigurationIntegration(TestCase):
    """Test configuration file integration."""

    def test_configuration_file_exists(self):
        """Test that configuration file exists and is readable."""
        config_path = "/Users/sanjeevadodlapati/Downloads/Repos/QeMLflow/config/audit_trail.yml"
        
        if os.path.exists(config_path):
            # Try to read the config file
            with open(config_path, 'r') as f:
                content = f.read()
            
            self.assertGreater(len(content), 0)
            # Basic validation that it looks like YAML
            self.assertIn('storage:', content)
            self.assertIn('logging:', content)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
