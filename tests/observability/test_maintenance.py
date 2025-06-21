"""
Tests for Automated Maintenance Module

This module tests all aspects of automated maintenance including dependency updates,
security patches, cleanup processes, and health-based scaling.
"""

import json
import tempfile
import time
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from qemlflow.observability.maintenance import (
    MaintenanceTask,
    MaintenanceResult,
    SystemHealth,
    DependencyUpdateExecutor,
    SecurityPatchExecutor,
    CleanupExecutor,
    HealthBasedScalingExecutor,
    AutomatedMaintenanceManager,
    get_maintenance_manager,
    initialize_maintenance,
    shutdown_maintenance
)


class TestMaintenanceTask(unittest.TestCase):
    """Test MaintenanceTask dataclass."""
    
    def test_task_creation(self):
        """Test creating maintenance task."""
        task = MaintenanceTask(
            task_id="test_task",
            task_type="dependency_update",
            priority="high",
            description="Test dependency update",
            schedule="daily"
        )
        
        self.assertEqual(task.task_id, "test_task")
        self.assertEqual(task.task_type, "dependency_update")
        self.assertEqual(task.priority, "high")
        self.assertEqual(task.schedule, "daily")
        self.assertTrue(task.enabled)
        self.assertEqual(task.run_count, 0)
        self.assertIsInstance(task.created_at, datetime)
    
    def test_task_to_dict(self):
        """Test converting task to dictionary."""
        task = MaintenanceTask("test_task", "cleanup")
        task_dict = task.to_dict()
        
        self.assertIsInstance(task_dict, dict)
        self.assertEqual(task_dict["task_id"], "test_task")
        self.assertEqual(task_dict["task_type"], "cleanup")
        self.assertIn("created_at", task_dict)
        self.assertIsInstance(task_dict["created_at"], str)  # Should be ISO string


class TestMaintenanceResult(unittest.TestCase):
    """Test MaintenanceResult dataclass."""
    
    def test_result_creation(self):
        """Test creating maintenance result."""
        result = MaintenanceResult(
            task_id="test_task",
            status="success",
            message="Task completed successfully",
            duration_seconds=45.2
        )
        
        self.assertEqual(result.task_id, "test_task")
        self.assertEqual(result.status, "success")
        self.assertEqual(result.duration_seconds, 45.2)
        self.assertIsInstance(result.timestamp, datetime)
        self.assertEqual(len(result.errors), 0)
    
    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        result = MaintenanceResult(
            task_id="test_task", 
            status="failure", 
            message="Test failed", 
            duration_seconds=10.0
        )
        result_dict = result.to_dict()
        
        self.assertIsInstance(result_dict, dict)
        self.assertEqual(result_dict["status"], "failure")
        self.assertIn("timestamp", result_dict)


class TestSystemHealth(unittest.TestCase):
    """Test SystemHealth dataclass."""
    
    def test_system_health_creation(self):
        """Test creating system health object."""
        health = SystemHealth(
            cpu_usage_percent=75.0,
            memory_usage_percent=60.0,
            disk_usage_percent=80.0,
            active_connections=500,
            response_time_ms=200.0,
            error_rate_percent=2.5
        )
        
        self.assertEqual(health.cpu_usage_percent, 75.0)
        self.assertEqual(health.memory_usage_percent, 60.0)
        self.assertTrue(health.healthy)  # Default is healthy
        self.assertEqual(len(health.issues), 0)


class TestDependencyUpdateExecutor(unittest.TestCase):
    """Test DependencyUpdateExecutor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.executor = DependencyUpdateExecutor(self.temp_dir)
        
        # Create mock requirements.txt
        req_file = Path(self.temp_dir) / "requirements.txt"
        with open(req_file, 'w') as f:
            f.write("requests==2.25.1\nnumpy==1.21.0\n")
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_can_execute(self):
        """Test executor type checking."""
        dep_task = MaintenanceTask("test", "dependency_update")
        other_task = MaintenanceTask("test", "cleanup")
        
        self.assertTrue(self.executor.can_execute(dep_task))
        self.assertFalse(self.executor.can_execute(other_task))
    
    @patch('subprocess.run')
    def test_execute_no_requirements(self, mock_run):
        """Test execution when no requirements files exist."""
        # Remove requirements file
        (Path(self.temp_dir) / "requirements.txt").unlink()
        
        task = MaintenanceTask("test", "dependency_update")
        result = self.executor.execute(task)
        
        self.assertEqual(result.status, "skipped")
        self.assertIn("No dependency files found", result.message)
    
    @patch('subprocess.run')
    def test_execute_with_updates(self, mock_run):
        """Test execution with available updates."""
        # Mock pip list --outdated
        mock_outdated_result = Mock()
        mock_outdated_result.stdout = json.dumps([
            {"name": "requests", "version": "2.25.1", "latest_version": "2.28.0"},
            {"name": "numpy", "version": "1.21.0", "latest_version": "1.23.0"}
        ])
        mock_outdated_result.returncode = 0
        
        # Mock pip install
        mock_install_result = Mock()
        mock_install_result.returncode = 0
        
        mock_run.side_effect = [mock_outdated_result, mock_install_result, mock_install_result]
        
        task = MaintenanceTask("test", "dependency_update")
        result = self.executor.execute(task)
        
        self.assertEqual(result.status, "success")
        self.assertIn("Updated 2 packages", result.message)
        self.assertEqual(len(result.details["updated_packages"]), 2)
    
    @patch('subprocess.run')
    def test_execute_with_pip_error(self, mock_run):
        """Test execution when pip commands fail."""
        mock_run.side_effect = Exception("pip command failed")
        
        task = MaintenanceTask("test", "dependency_update")
        result = self.executor.execute(task)
        
        self.assertEqual(result.status, "failure")
        self.assertIn("Dependency update failed", result.message)
    
    @patch('subprocess.run')
    def test_execute_poetry_packages(self, mock_run):
        """Test execution with poetry packages."""
        # Remove requirements.txt to only test poetry
        (Path(self.temp_dir) / "requirements.txt").unlink()
        
        # Create pyproject.toml to indicate poetry project
        pyproject_file = Path(self.temp_dir) / "pyproject.toml"
        with open(pyproject_file, 'w') as f:
            f.write("[tool.poetry]\nname = \"test\"\nversion = \"0.1.0\"\n")
        
        # Mock poetry commands
        mock_version_result = Mock()
        mock_version_result.returncode = 0
        
        mock_dry_run_result = Mock()
        mock_dry_run_result.stdout = "Updating package1 (1.0.0) to (1.1.0)"
        mock_dry_run_result.returncode = 0
        
        mock_update_result = Mock()
        mock_update_result.returncode = 0
        
        mock_run.side_effect = [mock_version_result, mock_dry_run_result, mock_update_result]
        
        task = MaintenanceTask("test", "dependency_update")
        result = self.executor.execute(task)
        
        self.assertEqual(result.status, "success")
        self.assertIn("Updated", result.message)
    
    @patch('subprocess.run')
    def test_execute_no_updates_available(self, mock_run):
        """Test execution when no updates are available."""
        # Mock pip list --outdated returning empty list
        mock_outdated_result = Mock()
        mock_outdated_result.stdout = "[]"
        mock_outdated_result.returncode = 0
        
        mock_run.return_value = mock_outdated_result
        
        task = MaintenanceTask("test", "dependency_update")
        result = self.executor.execute(task)
        
        self.assertEqual(result.status, "skipped")
        self.assertIn("No updates", result.message)
    
    @patch('subprocess.run')
    def test_execute_malformed_pip_output(self, mock_run):
        """Test execution with malformed pip output."""
        # Mock pip list --outdated returning malformed JSON
        mock_outdated_result = Mock()
        mock_outdated_result.stdout = "invalid json"
        mock_outdated_result.returncode = 0
        
        mock_run.return_value = mock_outdated_result
        
        task = MaintenanceTask("test", "dependency_update")
        result = self.executor.execute(task)
        
        self.assertEqual(result.status, "skipped")  # Falls back to no updates
        self.assertIn("No updates", result.message)


class TestSecurityPatchExecutor(unittest.TestCase):
    """Test SecurityPatchExecutor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.executor = SecurityPatchExecutor(self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_can_execute(self):
        """Test executor type checking."""
        sec_task = MaintenanceTask("test", "security_patch")
        other_task = MaintenanceTask("test", "dependency_update")
        
        self.assertTrue(self.executor.can_execute(sec_task))
        self.assertFalse(self.executor.can_execute(other_task))
    
    @patch('subprocess.run')
    def test_execute_no_vulnerabilities(self, mock_run):
        """Test execution when no vulnerabilities found."""
        # Mock both pip-audit and safety returning no vulnerabilities
        mock_pip_result = Mock()
        mock_pip_result.stdout = '{"vulnerabilities": []}'
        mock_pip_result.returncode = 0
        
        mock_safety_result = Mock()
        mock_safety_result.stdout = '[]'  # Empty array for safety
        mock_safety_result.returncode = 0
        
        # First call is pip-audit, second is safety
        mock_run.side_effect = [mock_pip_result, mock_safety_result]
        
        task = MaintenanceTask("test", "security_patch")
        result = self.executor.execute(task)
        
        self.assertEqual(result.status, "success")
        self.assertIn("No security vulnerabilities found", result.message)
    
    @patch('subprocess.run')
    def test_execute_with_vulnerabilities(self, mock_run):
        """Test execution with vulnerabilities found."""
        # Mock pip-audit finding vulnerabilities
        mock_audit_result = Mock()
        mock_audit_result.stdout = json.dumps({
            "vulnerabilities": [
                {
                    "package": "requests",
                    "version": "2.25.1",
                    "id": "CVE-2023-1234",
                    "description": "Test vulnerability",
                    "severity": "high"
                }
            ]
        })
        mock_audit_result.returncode = 0
        
        # Mock safety with no results
        mock_safety_result = Mock()
        mock_safety_result.stdout = '[]'
        mock_safety_result.returncode = 0
        
        # Mock pip install (fix attempt)
        mock_install_result = Mock()
        mock_install_result.returncode = 0
        
        mock_run.side_effect = [mock_audit_result, mock_safety_result, mock_install_result]
        
        task = MaintenanceTask("test", "security_patch")
        result = self.executor.execute(task)
        
        self.assertIn(result.status, ["success", "partial", "failure"])
        self.assertIn("vulnerabilities", result.details)
    
    @patch('subprocess.run')
    def test_execute_pip_audit_not_available(self, mock_run):
        """Test execution when pip-audit is not available."""
        # Mock pip-audit not found
        mock_run.side_effect = [
            FileNotFoundError("pip-audit not found"),  # pip-audit fails
            Mock(stdout='[]', returncode=0)  # safety succeeds with no results
        ]
        
        task = MaintenanceTask("test", "security_patch")
        result = self.executor.execute(task)
        
        self.assertEqual(result.status, "success")
        self.assertIn("No security vulnerabilities found", result.message)
    
    @patch('subprocess.run')
    def test_execute_malformed_json_response(self, mock_run):
        """Test execution with malformed JSON response."""
        # Mock pip-audit returning malformed JSON
        mock_audit_result = Mock()
        mock_audit_result.stdout = '{"invalid": json'  # Malformed JSON
        mock_audit_result.returncode = 0
        
        mock_safety_result = Mock()
        mock_safety_result.stdout = '[]'
        mock_safety_result.returncode = 0
        
        mock_run.side_effect = [mock_audit_result, mock_safety_result]
        
        task = MaintenanceTask("test", "security_patch")
        result = self.executor.execute(task)
        
        # Should handle gracefully and continue with safety check
        self.assertEqual(result.status, "success")


class TestCleanupExecutor(unittest.TestCase):
    """Test CleanupExecutor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.executor = CleanupExecutor(self.temp_dir)
        
        # Create some files to clean up
        self._create_cleanup_files()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def _create_cleanup_files(self):
        """Create files that should be cleaned up."""
        # Create __pycache__ directory
        pycache_dir = Path(self.temp_dir) / "__pycache__"
        pycache_dir.mkdir()
        (pycache_dir / "test.pyc").touch()
        
        # Create .pyc files
        (Path(self.temp_dir) / "test.pyc").touch()
        
        # Create old log files
        logs_dir = Path(self.temp_dir) / "logs"
        logs_dir.mkdir()
        old_log = logs_dir / "old.log"
        old_log.touch()
        # Make it old using utime
        import os
        old_time = time.time() - (35 * 24 * 60 * 60)  # 35 days ago
        os.utime(old_log, (old_time, old_time))
    
    def test_can_execute(self):
        """Test executor type checking."""
        cleanup_task = MaintenanceTask("test", "cleanup")
        other_task = MaintenanceTask("test", "dependency_update")
        
        self.assertTrue(self.executor.can_execute(cleanup_task))
        self.assertFalse(self.executor.can_execute(other_task))
    
    def test_execute_cleanup(self):
        """Test cleanup execution."""
        task = MaintenanceTask("test", "cleanup")
        result = self.executor.execute(task)
        
        self.assertEqual(result.status, "success")
        self.assertIn("Cleaned up", result.message)
        self.assertIn("files_removed", result.details)
        self.assertIn("directories_removed", result.details)
        
        # Verify __pycache__ was removed
        self.assertFalse((Path(self.temp_dir) / "__pycache__").exists())
    
    def test_cleanup_pattern(self):
        """Test pattern-based cleanup."""
        stats = self.executor._cleanup_pattern("*.pyc")
        
        self.assertIsInstance(stats, dict)
        self.assertIn("files", stats)
        self.assertIn("directories", stats)
        self.assertIn("bytes", stats)
    
    def test_cleanup_with_permission_error(self):
        """Test cleanup with permission errors."""
        # Create a protected file (simulate permission error in _cleanup_pattern)
        with patch.object(self.executor, '_cleanup_pattern') as mock_cleanup:
            mock_cleanup.side_effect = PermissionError("Permission denied")
            
            task = MaintenanceTask("test", "cleanup")
            result = self.executor.execute(task)
            
            self.assertEqual(result.status, "failure")
            self.assertIn("Cleanup failed", result.message)
    
    def test_cleanup_empty_directory(self):
        """Test cleanup in empty directory."""
        # Create empty test directory
        empty_dir = tempfile.mkdtemp()
        empty_executor = CleanupExecutor(empty_dir)
        
        try:
            task = MaintenanceTask("test", "cleanup")
            result = empty_executor.execute(task)
            
            self.assertEqual(result.status, "success")
            self.assertIn("Cleaned up", result.message)
            self.assertEqual(result.details["files_removed"], 0)
        finally:
            import shutil
            shutil.rmtree(empty_dir)


class TestHealthBasedScalingExecutor(unittest.TestCase):
    """Test HealthBasedScalingExecutor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.executor = HealthBasedScalingExecutor()
    
    def test_can_execute(self):
        """Test executor type checking."""
        scaling_task = MaintenanceTask("test", "scaling")
        other_task = MaintenanceTask("test", "cleanup")
        
        self.assertTrue(self.executor.can_execute(scaling_task))
        self.assertFalse(self.executor.can_execute(other_task))
    
    @patch.object(HealthBasedScalingExecutor, '_get_system_health')
    def test_execute_healthy_system(self, mock_health):
        """Test execution with healthy system."""
        mock_health.return_value = SystemHealth(
            cpu_usage_percent=50.0,
            memory_usage_percent=60.0,
            disk_usage_percent=70.0,
            active_connections=100,
            response_time_ms=200.0,
            error_rate_percent=1.0
        )
        
        task = MaintenanceTask("test", "scaling")
        task.metadata = {"cpu_scale_threshold": 75.0}
        result = self.executor.execute(task)
        
        self.assertEqual(result.status, "success")
        self.assertIn("system_health", result.details)
    
    @patch.object(HealthBasedScalingExecutor, '_get_system_health')
    def test_execute_high_cpu(self, mock_health):
        """Test execution with high CPU usage."""
        mock_health.return_value = SystemHealth(
            cpu_usage_percent=85.0,  # High CPU
            memory_usage_percent=60.0,
            disk_usage_percent=70.0,
            active_connections=100,
            response_time_ms=200.0,
            error_rate_percent=1.0
        )
        
        task = MaintenanceTask("test", "scaling")
        task.metadata = {"cpu_scale_threshold": 75.0}
        result = self.executor.execute(task)
        
        self.assertEqual(result.status, "success")
        self.assertGreater(len(result.details["recommended_actions"]), 0)
    
    def test_determine_scaling_actions(self):
        """Test scaling action determination."""
        health = SystemHealth(
            cpu_usage_percent=85.0,
            memory_usage_percent=90.0,
            disk_usage_percent=70.0,
            active_connections=1500,
            response_time_ms=200.0,
            error_rate_percent=1.0
        )
        
        config = {
            "cpu_scale_threshold": 80.0,
            "memory_scale_threshold": 85.0,
            "connection_scale_threshold": 1000
        }
        
        actions = self.executor._determine_scaling_actions(health, config)
        
        self.assertGreater(len(actions), 0)
        # Should recommend scaling for CPU, memory, and connections
        action_types = [a["type"] for a in actions]
        self.assertIn("scale_up", action_types)
        self.assertIn("scale_out", action_types)


class TestAutomatedMaintenanceManager(unittest.TestCase):
    """Test AutomatedMaintenanceManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = AutomatedMaintenanceManager(storage_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.manager.stop_scheduler()
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_manager_initialization(self):
        """Test manager initialization."""
        self.assertIsNotNone(self.manager.config)
        self.assertGreater(len(self.manager.executors), 0)
        self.assertEqual(len(self.manager.tasks), 0)
        self.assertFalse(self.manager._scheduler_thread and self.manager._scheduler_thread.is_alive())
    
    def test_add_task(self):
        """Test adding maintenance task."""
        task = MaintenanceTask("test_task", "cleanup")
        
        success = self.manager.add_task(task)
        
        self.assertTrue(success)
        self.assertIn("test_task", self.manager.tasks)
        self.assertEqual(self.manager.tasks["test_task"].task_id, "test_task")
    
    def test_remove_task(self):
        """Test removing maintenance task."""
        task = MaintenanceTask("test_task", "cleanup")
        self.manager.add_task(task)
        
        success = self.manager.remove_task("test_task")
        
        self.assertTrue(success)
        self.assertNotIn("test_task", self.manager.tasks)
    
    def test_remove_nonexistent_task(self):
        """Test removing non-existent task."""
        success = self.manager.remove_task("nonexistent")
        
        self.assertFalse(success)
    
    def test_execute_task(self):
        """Test executing maintenance task."""
        task = MaintenanceTask("test_cleanup", "cleanup")
        self.manager.add_task(task)
        
        result = self.manager.execute_task("test_cleanup")
        
        self.assertIsNotNone(result)
        if result:  # Type guard for None check
            self.assertEqual(result.task_id, "test_cleanup")
            self.assertIn(result.status, ["success", "failure", "skipped"])
        
        # Check task was updated
        updated_task = self.manager.tasks["test_cleanup"]
        self.assertEqual(updated_task.run_count, 1)
        self.assertIsNotNone(updated_task.last_run)
    
    def test_execute_nonexistent_task(self):
        """Test executing non-existent task."""
        result = self.manager.execute_task("nonexistent")
        
        self.assertIsNone(result)
    
    def test_execute_disabled_task(self):
        """Test executing disabled task."""
        task = MaintenanceTask("disabled_task", "cleanup")
        task.enabled = False
        self.manager.add_task(task)
        
        result = self.manager.execute_task("disabled_task")
        
        self.assertIsNone(result)
    
    def test_get_task_status(self):
        """Test getting task status."""
        task = MaintenanceTask("test_task", "cleanup")
        self.manager.add_task(task)
        
        # Get specific task status
        status = self.manager.get_task_status("test_task")
        self.assertIn("task", status)
        self.assertEqual(status["task"]["task_id"], "test_task")
        
        # Get all tasks status
        all_status = self.manager.get_task_status()
        self.assertIn("total_tasks", all_status)
        self.assertEqual(all_status["total_tasks"], 1)
    
    def test_get_maintenance_summary(self):
        """Test getting maintenance summary."""
        # Add and execute a task
        task = MaintenanceTask("test_task", "cleanup")
        self.manager.add_task(task)
        self.manager.execute_task("test_task")
        
        summary = self.manager.get_maintenance_summary(days=1)
        
        self.assertIn("total_executions", summary)
        self.assertIn("successful_executions", summary)
        self.assertIn("by_task_type", summary)
        self.assertGreaterEqual(summary["total_executions"], 1)
    
    def test_scheduler_start_stop(self):
        """Test scheduler start and stop."""
        self.manager.start_scheduler()
        
        # Give it a moment to start
        time.sleep(0.1)
        
        self.assertTrue(self.manager._scheduler_thread and self.manager._scheduler_thread.is_alive())
        
        self.manager.stop_scheduler()
        
        # Should stop gracefully
        self.assertFalse(self.manager._scheduler_thread and self.manager._scheduler_thread.is_alive())
    
    def test_calculate_next_run(self):
        """Test next run calculation."""
        # Daily schedule
        daily_task = MaintenanceTask("daily", "cleanup", schedule="daily")
        self.manager._calculate_next_run(daily_task)
        self.assertIsNotNone(daily_task.next_run)
        
        # Manual schedule
        manual_task = MaintenanceTask("manual", "cleanup", schedule="manual")
        self.manager._calculate_next_run(manual_task)
        self.assertIsNone(manual_task.next_run)
    
    @patch('builtins.open', new_callable=MagicMock)
    @patch('json.load')
    def test_load_tasks(self, mock_json_load, mock_open):
        """Test loading tasks from storage."""
        # Mock task data
        mock_task_data = [{
            "task_id": "loaded_task",
            "task_type": "cleanup",
            "priority": "medium",
            "description": "",
            "schedule": "daily",
            "enabled": True,
            "created_at": "2023-01-01T00:00:00+00:00",
            "last_run": None,
            "next_run": None,
            "run_count": 0,
            "success_count": 0,
            "failure_count": 0,
            "last_result": None,
            "metadata": {}
        }]
        
        mock_json_load.return_value = mock_task_data
        mock_open.return_value.__enter__.return_value = MagicMock()
        
        # Create new manager (will trigger _load_tasks)
        with patch.object(Path, 'exists', return_value=True):
            manager = AutomatedMaintenanceManager(storage_dir=self.temp_dir)
        
        self.assertIn("loaded_task", manager.tasks)
    
    def test_add_duplicate_task(self):
        """Test adding duplicate task."""
        task1 = MaintenanceTask("duplicate_task", "cleanup")
        task2 = MaintenanceTask("duplicate_task", "dependency_update")
        
        success1 = self.manager.add_task(task1)
        success2 = self.manager.add_task(task2)
        
        self.assertTrue(success1)
        self.assertTrue(success2)  # Should overwrite
        
        # Should have the second task's type
        self.assertEqual(self.manager.tasks["duplicate_task"].task_type, "dependency_update")
    
    def test_execute_task_with_executor_error(self):
        """Test executing task when executor raises error."""
        task = MaintenanceTask("error_task", "cleanup")
        self.manager.add_task(task)
        
        # Find the cleanup executor and mock it specifically
        cleanup_executor = None
        for executor in self.manager.executors:
            if executor.can_execute(task):
                cleanup_executor = executor
                break
        
        self.assertIsNotNone(cleanup_executor, "No cleanup executor found")
        
        # Mock the specific executor to return a failure result
        with patch.object(cleanup_executor, 'execute') as mock_execute:
            mock_execute.return_value = MaintenanceResult(
                task_id="error_task",
                status="failure",
                message="Executor failed",
                duration_seconds=1.0
            )
            
            result = self.manager.execute_task("error_task")
            
            self.assertIsNotNone(result)
            if result:
                self.assertEqual(result.status, "failure")
                self.assertIn("failed", result.message.lower())
    
    def test_save_and_load_tasks(self):
        """Test saving and loading tasks."""
        # Add a task
        task = MaintenanceTask("persistent_task", "cleanup")
        task.metadata = {"test": "data"}
        self.manager.add_task(task)
        
        # Save tasks
        self.manager._save_tasks()
        
        # Create new manager (should load tasks)
        new_manager = AutomatedMaintenanceManager(storage_dir=self.temp_dir)
        
        self.assertIn("persistent_task", new_manager.tasks)
        self.assertEqual(new_manager.tasks["persistent_task"].task_type, "cleanup")
        self.assertEqual(new_manager.tasks["persistent_task"].metadata, {"test": "data"})
        
        new_manager.stop_scheduler()
    
    def test_invalid_schedule_calculation(self):
        """Test calculation with invalid schedule."""
        task = MaintenanceTask("invalid_schedule", "cleanup", schedule="invalid")
        self.manager._calculate_next_run(task)
        
        # Should set a default next run time (1 hour from now)
        self.assertIsNotNone(task.next_run)
        if task.next_run:
            # Verify it's approximately 1 hour from now
            expected_time = datetime.now(timezone.utc) + timedelta(hours=1)
            time_diff = abs((task.next_run - expected_time).total_seconds())
            self.assertLess(time_diff, 60)  # Within 1 minute tolerance
    
    def test_concurrent_task_execution(self):
        """Test that concurrent executions are handled properly."""
        import threading
        
        task = MaintenanceTask("concurrent_task", "cleanup")
        self.manager.add_task(task)
        
        results = []
        
        def execute_task():
            result = self.manager.execute_task("concurrent_task")
            results.append(result)
        
        # Start multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=execute_task)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # All should complete (though only one might actually execute due to implementation)
        self.assertEqual(len(results), 3)
        successful_results = [r for r in results if r is not None]
        self.assertGreaterEqual(len(successful_results), 1)


class TestGlobalFunctions(unittest.TestCase):
    """Test global maintenance functions."""
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutdown_maintenance()
    
    def test_get_maintenance_manager(self):
        """Test getting maintenance manager."""
        manager1 = get_maintenance_manager()
        manager2 = get_maintenance_manager()
        
        # Should return same instance
        self.assertIs(manager1, manager2)
        self.assertIsNotNone(manager1)
    
    def test_initialize_maintenance(self):
        """Test initializing maintenance system."""
        temp_dir = tempfile.mkdtemp()
        
        try:
            manager = initialize_maintenance(storage_dir=temp_dir)
            
            self.assertIsNotNone(manager)
            self.assertEqual(str(manager.storage_dir), temp_dir)
        finally:
            shutdown_maintenance()
            import shutil
            if Path(temp_dir).exists():
                shutil.rmtree(temp_dir)
    
    def test_shutdown_maintenance(self):
        """Test shutting down maintenance system."""
        # Initialize first
        manager = get_maintenance_manager()
        self.assertIsNotNone(manager)
        
        # Shutdown
        shutdown_maintenance()
        
        # Should be cleaned up
        # Note: Can't easily test this without accessing global variable


if __name__ == '__main__':
    unittest.main()
