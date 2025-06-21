"""
Tests for Code Health Metrics Module

This module tests all aspects of code health tracking including technical debt
analysis, code quality metrics, complexity monitoring, and maintenance scheduling.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.qemlflow.observability.code_health import (
    TechnicalDebt,
    CodeQualityMetrics,
    ComplexityMetrics,
    MaintenanceTask,
    TechnicalDebtAnalyzer,
    CodeQualityAnalyzer,
    ComplexityAnalyzer,
    MaintenanceScheduler,
    CodeHealthDashboard,
    get_code_health_dashboard
)


class TestTechnicalDebt(unittest.TestCase):
    """Test TechnicalDebt dataclass."""
    
    def test_technical_debt_creation(self):
        """Test creating technical debt item."""
        debt = TechnicalDebt(
            file_path="test.py",
            line_number=10,
            debt_type="code_smell",
            severity="medium",
            description="TODO: Refactor this method"
        )
        
        self.assertEqual(debt.file_path, "test.py")
        self.assertEqual(debt.line_number, 10)
        self.assertEqual(debt.debt_type, "code_smell")
        self.assertEqual(debt.severity, "medium")
        self.assertEqual(debt.status, "open")
    
    def test_debt_resolution(self):
        """Test resolving technical debt."""
        debt = TechnicalDebt()
        debt.resolve()
        
        self.assertEqual(debt.status, "resolved")
        self.assertIsNotNone(debt.resolved_at)
    
    def test_debt_to_dict(self):
        """Test converting debt to dictionary."""
        debt = TechnicalDebt(file_path="test.py", debt_type="bug_risk")
        debt_dict = debt.to_dict()
        
        self.assertIsInstance(debt_dict, dict)
        self.assertEqual(debt_dict["file_path"], "test.py")
        self.assertEqual(debt_dict["debt_type"], "bug_risk")


class TestCodeQualityMetrics(unittest.TestCase):
    """Test CodeQualityMetrics dataclass."""
    
    def test_quality_metrics_creation(self):
        """Test creating quality metrics."""
        metrics = CodeQualityMetrics(
            file_path="test.py",
            lines_of_code=100,
            cyclomatic_complexity=5.0,
            pylint_score=8.5
        )
        
        self.assertEqual(metrics.file_path, "test.py")
        self.assertEqual(metrics.lines_of_code, 100)
        self.assertEqual(metrics.cyclomatic_complexity, 5.0)
        self.assertEqual(metrics.pylint_score, 8.5)
    
    def test_metrics_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = CodeQualityMetrics(file_path="test.py")
        metrics_dict = metrics.to_dict()
        
        self.assertIsInstance(metrics_dict, dict)
        self.assertEqual(metrics_dict["file_path"], "test.py")


class TestComplexityMetrics(unittest.TestCase):
    """Test ComplexityMetrics dataclass."""
    
    def test_complexity_metrics_creation(self):
        """Test creating complexity metrics."""
        metrics = ComplexityMetrics(
            file_path="test.py",
            function_name="test_func",
            cyclomatic_complexity=3,
            cognitive_complexity=2
        )
        
        self.assertEqual(metrics.file_path, "test.py")
        self.assertEqual(metrics.function_name, "test_func")
        self.assertEqual(metrics.cyclomatic_complexity, 3)
        self.assertEqual(metrics.cognitive_complexity, 2)


class TestMaintenanceTask(unittest.TestCase):
    """Test MaintenanceTask dataclass."""
    
    def test_maintenance_task_creation(self):
        """Test creating maintenance task."""
        task = MaintenanceTask(
            task_type="dependency_update",
            title="Update packages",
            priority="high"
        )
        
        self.assertEqual(task.task_type, "dependency_update")
        self.assertEqual(task.title, "Update packages")
        self.assertEqual(task.priority, "high")
        self.assertEqual(task.status, "scheduled")
    
    def test_task_completion(self):
        """Test completing task."""
        task = MaintenanceTask()
        task.complete()
        
        self.assertEqual(task.status, "completed")
        self.assertIsNotNone(task.completed_at)


class TestTechnicalDebtAnalyzer(unittest.TestCase):
    """Test TechnicalDebtAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.analyzer = TechnicalDebtAnalyzer(self.temp_dir)
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        self.assertEqual(str(self.analyzer.project_root), self.temp_dir)
        self.assertIn("TODO", self.analyzer.debt_patterns)
        self.assertIn("FIXME", self.analyzer.debt_patterns)
    
    def test_analyze_file_with_comments(self):
        """Test analyzing file with debt comments."""
        # Create test file with debt comments
        test_file = Path(self.temp_dir) / "test.py"
        test_content = '''
def test_function():
    # TODO: Implement this properly
    # FIXME: This is broken
    # HACK: Temporary solution
    pass
'''
        test_file.write_text(test_content)
        
        debts = self.analyzer.analyze_file(test_file)
        
        self.assertEqual(len(debts), 4)  # TODO, FIXME, HACK, TEMP (from "Temporary")
        
        # Check TODO debt
        todo_debt = next(d for d in debts if "TODO" in d.description)
        self.assertEqual(todo_debt.debt_type, "maintainability")
        self.assertEqual(todo_debt.severity, "low")
        
        # Check FIXME debt
        fixme_debt = next(d for d in debts if "FIXME" in d.description)
        self.assertEqual(fixme_debt.debt_type, "bug_risk")
        self.assertEqual(fixme_debt.severity, "medium")
    
    def test_analyze_long_function(self):
        """Test analyzing file with long function."""
        # Create test file with long function
        test_file = Path(self.temp_dir) / "test.py"
        long_function = "def long_function():\n" + "    pass\n" * 60  # 61 lines total
        test_file.write_text(long_function)
        
        debts = self.analyzer.analyze_file(test_file)
        
        # Should detect long method
        long_method_debts = [d for d in debts if "Long function" in d.description]
        self.assertTrue(len(long_method_debts) > 0)
    
    def test_analyze_nonexistent_file(self):
        """Test analyzing non-existent file."""
        non_existent = Path(self.temp_dir) / "nonexistent.py"
        debts = self.analyzer.analyze_file(non_existent)
        
        self.assertEqual(len(debts), 0)
    
    def test_analyze_project(self):
        """Test analyzing entire project."""
        # Create multiple test files
        (Path(self.temp_dir) / "file1.py").write_text("# TODO: Fix this\npass")
        (Path(self.temp_dir) / "file2.py").write_text("# FIXME: Broken\npass")
        
        debts = self.analyzer.analyze_project()
        
        self.assertGreaterEqual(len(debts), 2)


class TestCodeQualityAnalyzer(unittest.TestCase):
    """Test CodeQualityAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.analyzer = CodeQualityAnalyzer(self.temp_dir)
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        self.assertEqual(str(self.analyzer.project_root), self.temp_dir)
    
    def test_analyze_python_file(self):
        """Test analyzing Python file."""
        # Create test Python file
        test_file = Path(self.temp_dir) / "test.py"
        test_content = '''
import os
import sys

def simple_function(a, b):
    """A simple function."""
    if a > b:
        return a
    else:
        return b

class SimpleClass:
    """A simple class."""
    def method(self):
        return "hello"
'''
        test_file.write_text(test_content)
        
        metrics = self.analyzer.analyze_file(test_file)
        
        self.assertEqual(metrics.file_path, str(test_file.relative_to(Path(self.temp_dir))))
        self.assertGreater(metrics.lines_of_code, 0)
        self.assertGreater(metrics.source_lines, 0)
        self.assertGreater(metrics.import_count, 0)
    
    def test_analyze_non_python_file(self):
        """Test analyzing non-Python file."""
        test_file = Path(self.temp_dir) / "test.txt"
        test_file.write_text("Hello, world!")
        
        metrics = self.analyzer.analyze_file(test_file)
        
        # Should still create metrics but with limited analysis
        self.assertEqual(metrics.file_path, str(test_file.relative_to(Path(self.temp_dir))))


class TestComplexityAnalyzer(unittest.TestCase):
    """Test ComplexityAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.analyzer = ComplexityAnalyzer(self.temp_dir)
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        self.assertEqual(str(self.analyzer.project_root), self.temp_dir)
    
    def test_analyze_simple_function(self):
        """Test analyzing simple function."""
        test_file = Path(self.temp_dir) / "test.py"
        test_content = '''
def simple_function(x):
    if x > 0:
        return x * 2
    else:
        return 0
'''
        test_file.write_text(test_content)
        
        metrics_list = self.analyzer.analyze_file(test_file)
        
        self.assertEqual(len(metrics_list), 1)
        metrics = metrics_list[0]
        self.assertEqual(metrics.function_name, "simple_function")
        self.assertGreater(metrics.cyclomatic_complexity, 0)
    
    def test_analyze_complex_function(self):
        """Test analyzing complex function."""
        test_file = Path(self.temp_dir) / "test.py"
        test_content = '''
def complex_function(a, b, c, d, e):
    if a > 0:
        if b > 0:
            if c > 0:
                if d > 0:
                    if e > 0:
                        return a + b + c + d + e
                    else:
                        return a + b + c + d
                else:
                    return a + b + c
            else:
                return a + b
        else:
            return a
    else:
        return 0
'''
        test_file.write_text(test_content)
        
        metrics_list = self.analyzer.analyze_file(test_file)
        
        self.assertEqual(len(metrics_list), 1)
        metrics = metrics_list[0]
        self.assertEqual(metrics.function_name, "complex_function")
        self.assertGreater(metrics.cyclomatic_complexity, 5)  # Should be high complexity


class TestMaintenanceScheduler(unittest.TestCase):
    """Test MaintenanceScheduler class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.scheduler = MaintenanceScheduler(self.temp_dir)
    
    def test_scheduler_initialization(self):
        """Test scheduler initialization."""
        # The storage_dir should be within the temp_dir
        self.assertTrue(str(self.scheduler.storage_dir).startswith(self.temp_dir))
    
    def test_schedule_task(self):
        """Test scheduling maintenance task."""
        task = MaintenanceTask(
            task_type="dependency_update",
            title="Update pytest",
            priority="medium"
        )
        
        task_id = self.scheduler.schedule_task(task)
        
        # Should return task ID as string
        self.assertIsInstance(task_id, str)
        self.assertEqual(task.task_type, "dependency_update")
        self.assertEqual(task.title, "Update pytest")
        self.assertEqual(task.priority, "medium")
    
    def test_get_scheduled_tasks(self):
        """Test getting maintenance summary."""
        # Schedule some tasks
        task1 = MaintenanceTask(title="Task 1", priority="high")
        task2 = MaintenanceTask(title="Task 2", priority="low")
        
        self.scheduler.schedule_task(task1)
        self.scheduler.schedule_task(task2)
        
        summary = self.scheduler.get_maintenance_summary()
        
        self.assertGreaterEqual(summary["scheduled_tasks"], 2)
    
    def test_complete_task(self):
        """Test completing scheduled task."""
        task = MaintenanceTask(title="Test Task")
        task_id = self.scheduler.schedule_task(task)
        
        completed = self.scheduler.complete_task(task_id)
        
        self.assertTrue(completed)
        self.assertEqual(task.status, "completed")
    
    def test_check_outdated_dependencies(self):
        """Test checking for outdated dependencies."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout='[{"name": "requests", "version": "2.25.1", "latest_version": "2.28.0"}]'
            )
            
            task_ids = self.scheduler.schedule_dependency_updates()
            
            self.assertIsInstance(task_ids, list)


class TestCodeHealthDashboard(unittest.TestCase):
    """Test CodeHealthDashboard class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.dashboard = CodeHealthDashboard(self.temp_dir)
    
    def test_dashboard_initialization(self):
        """Test dashboard initialization."""
        self.assertEqual(str(self.dashboard.project_root), self.temp_dir)
        self.assertIsNotNone(self.dashboard.debt_analyzer)
        self.assertIsNotNone(self.dashboard.quality_analyzer)
        self.assertIsNotNone(self.dashboard.complexity_analyzer)
        self.assertIsNotNone(self.dashboard.maintenance_scheduler)
    
    @patch('src.qemlflow.observability.code_health.TechnicalDebtAnalyzer')
    @patch('src.qemlflow.observability.code_health.CodeQualityAnalyzer')
    @patch('src.qemlflow.observability.code_health.ComplexityAnalyzer')
    def test_analyze_code_health(self, mock_complexity, mock_quality, mock_debt):
        """Test analyzing code health."""
        # Mock analyzers
        mock_debt.return_value.analyze_project.return_value = [
            TechnicalDebt(file_path="test.py", debt_type="code_smell")
        ]
        mock_quality.return_value.analyze_file.return_value = CodeQualityMetrics(file_path="test.py")
        mock_complexity.return_value.analyze_file.return_value = [
            ComplexityMetrics(file_path="test.py", function_name="test_func")
        ]
        
        # Create test file
        test_file = Path(self.temp_dir) / "test.py"
        test_file.write_text("def test_func(): pass")
        
        dashboard = CodeHealthDashboard(self.temp_dir)
        health_data = dashboard.analyze_project_health()
        
        self.assertIn("technical_debt", health_data)
        self.assertIn("code_quality", health_data)
        self.assertIn("complexity_analysis", health_data)
        self.assertIn("project_overview", health_data)
    
    def test_save_report(self):
        """Test saving health report."""
        # Test by running a full analysis which saves a report
        test_file = Path(self.temp_dir) / "test.py"
        test_file.write_text("def test_func(): pass")
        
        # Run analysis which saves report internally
        self.dashboard.analyze_project_health()
        
        # Check that report files were created
        reports_dir = Path(self.temp_dir) / "code_health"
        self.assertTrue(reports_dir.exists())
        
        latest_file = reports_dir / "latest_report.json"
        self.assertTrue(latest_file.exists())


class TestGlobalFunctions(unittest.TestCase):
    """Test global functions."""
    
    def test_get_code_health_dashboard(self):
        """Test getting global dashboard instance."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dashboard1 = get_code_health_dashboard(temp_dir)
            dashboard2 = get_code_health_dashboard(temp_dir)
            
            # Should return same instance
            self.assertIs(dashboard1, dashboard2)


if __name__ == '__main__':
    unittest.main()
