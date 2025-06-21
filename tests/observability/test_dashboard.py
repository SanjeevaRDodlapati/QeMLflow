"""
Tests for Dashboard & Reporting Module

This module tests comprehensive monitoring dashboards, automated reporting,
trend analysis, and performance benchmarking capabilities.
"""

import json
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from qemlflow.observability.dashboard import (
    DashboardWidget,
    DashboardLayout,
    ReportSchedule,
    TrendAnalysis,
    MonitoringDataSource,
    ChartGenerator,
    TrendAnalyzer,
    ReportGenerator,
    DashboardManager,
    get_dashboard_manager,
    initialize_dashboard_system,
    shutdown_dashboard_system
)


class TestDashboardWidget(unittest.TestCase):
    """Test DashboardWidget dataclass."""
    
    def test_widget_creation(self):
        """Test creating dashboard widget."""
        widget = DashboardWidget(
            widget_id="test_widget",
            widget_type="chart",
            title="Test Chart",
            description="Test chart widget",
            data_source="monitoring"
        )
        
        self.assertEqual(widget.widget_id, "test_widget")
        self.assertEqual(widget.widget_type, "chart")
        self.assertEqual(widget.title, "Test Chart")
        self.assertEqual(widget.refresh_interval, 60)
        self.assertIsInstance(widget.created_at, datetime)
    
    def test_widget_to_dict(self):
        """Test converting widget to dictionary."""
        widget = DashboardWidget("test_widget", "metric", "Test Metric")
        widget_dict = widget.to_dict()
        
        self.assertIsInstance(widget_dict, dict)
        self.assertEqual(widget_dict["widget_id"], "test_widget")
        self.assertEqual(widget_dict["widget_type"], "metric")
        self.assertIn("created_at", widget_dict)
        self.assertIsInstance(widget_dict["created_at"], str)


class TestDashboardLayout(unittest.TestCase):
    """Test DashboardLayout dataclass."""
    
    def test_layout_creation(self):
        """Test creating dashboard layout."""
        layout = DashboardLayout(
            dashboard_id="test_dashboard",
            name="Test Dashboard",
            description="Test dashboard layout"
        )
        
        self.assertEqual(layout.dashboard_id, "test_dashboard")
        self.assertEqual(layout.name, "Test Dashboard")
        self.assertEqual(len(layout.widgets), 0)
        self.assertIsInstance(layout.created_at, datetime)
    
    def test_add_widget(self):
        """Test adding widget to dashboard."""
        layout = DashboardLayout("test", "Test")
        widget = DashboardWidget("widget1", "chart", "Chart 1")
        
        layout.add_widget(widget)
        
        self.assertEqual(len(layout.widgets), 1)
        self.assertEqual(layout.widgets[0].widget_id, "widget1")
        self.assertIsNotNone(layout.updated_at)
    
    def test_remove_widget(self):
        """Test removing widget from dashboard."""
        layout = DashboardLayout("test", "Test")
        widget = DashboardWidget("widget1", "chart", "Chart 1")
        layout.add_widget(widget)
        
        # Remove existing widget
        success = layout.remove_widget("widget1")
        self.assertTrue(success)
        self.assertEqual(len(layout.widgets), 0)
        
        # Try to remove non-existent widget
        success = layout.remove_widget("widget2")
        self.assertFalse(success)


class TestReportSchedule(unittest.TestCase):
    """Test ReportSchedule dataclass."""
    
    def test_schedule_creation(self):
        """Test creating report schedule."""
        schedule = ReportSchedule(
            schedule_id="daily_report",
            report_type="system_summary",
            frequency="daily",
            recipients=["admin@test.com"],
            format="html"
        )
        
        self.assertEqual(schedule.schedule_id, "daily_report")
        self.assertEqual(schedule.frequency, "daily")
        self.assertTrue(schedule.enabled)
        self.assertEqual(len(schedule.recipients), 1)


class TestTrendAnalysis(unittest.TestCase):
    """Test TrendAnalysis dataclass."""
    
    def test_trend_analysis_creation(self):
        """Test creating trend analysis."""
        analysis = TrendAnalysis(
            metric_name="cpu_usage",
            time_period="24h",
            trend_direction="increasing",
            trend_strength=0.8,
            change_rate=15.5,
            statistical_significance=0.95
        )
        
        self.assertEqual(analysis.metric_name, "cpu_usage")
        self.assertEqual(analysis.trend_direction, "increasing")
        self.assertEqual(analysis.trend_strength, 0.8)
        self.assertEqual(len(analysis.anomalies_detected), 0)


class TestMonitoringDataSource(unittest.TestCase):
    """Test MonitoringDataSource class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_source = MonitoringDataSource(self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_get_available_metrics_empty(self):
        """Test getting available metrics from empty directory."""
        metrics = self.data_source.get_available_metrics()
        self.assertEqual(len(metrics), 0)
    
    def test_get_available_metrics_with_data(self):
        """Test getting available metrics with data files."""
        # Create test metric files
        metric_file = Path(self.temp_dir) / "cpu_usage.json"
        with open(metric_file, 'w') as f:
            json.dump([], f)
        
        metrics = self.data_source.get_available_metrics()
        self.assertIn("cpu_usage", metrics)
    
    def test_get_data_no_file(self):
        """Test getting data when metric file doesn't exist."""
        query = {
            "metric_type": "nonexistent",
            "time_range": "1h",
            "aggregation": "avg"
        }
        
        result = self.data_source.get_data(query)
        
        self.assertIn("metric_type", result)
        self.assertEqual(result["metric_type"], "nonexistent")
        self.assertEqual(len(result["data"]), 0)
    
    def test_get_data_with_file(self):
        """Test getting data from existing metric file."""
        # Create test data
        test_data = [
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "value": 50.0,
                "cpu_percent": 45.0
            }
        ]
        
        metric_file = Path(self.temp_dir) / "cpu_usage.json"
        with open(metric_file, 'w') as f:
            json.dump(test_data, f)
        
        query = {
            "metric_type": "cpu_usage",
            "time_range": "1h",
            "aggregation": "avg"
        }
        
        result = self.data_source.get_data(query)
        
        self.assertEqual(result["metric_type"], "cpu_usage")
        self.assertIn("data", result)
        self.assertGreater(len(result["data"]), 0)
    
    def test_calculate_cutoff_time(self):
        """Test cutoff time calculation."""
        # Test hours
        cutoff_1h = self.data_source._calculate_cutoff_time("1h")
        expected_1h = datetime.now(timezone.utc) - timedelta(hours=1)
        self.assertAlmostEqual(
            cutoff_1h.timestamp(), 
            expected_1h.timestamp(), 
            delta=60
        )
        
        # Test days
        cutoff_1d = self.data_source._calculate_cutoff_time("1d")
        expected_1d = datetime.now(timezone.utc) - timedelta(days=1)
        self.assertAlmostEqual(
            cutoff_1d.timestamp(), 
            expected_1d.timestamp(), 
            delta=60
        )
        
        # Test invalid format (should default to 1h)
        cutoff_invalid = self.data_source._calculate_cutoff_time("invalid")
        expected_invalid = datetime.now(timezone.utc) - timedelta(hours=1)
        self.assertAlmostEqual(
            cutoff_invalid.timestamp(),
            expected_invalid.timestamp(),
            delta=60
        )


class TestChartGenerator(unittest.TestCase):
    """Test ChartGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.chart_generator = ChartGenerator(self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_generate_chart_empty_data(self):
        """Test generating chart with empty data."""
        widget = DashboardWidget("test_chart", "chart", "Test Chart")
        widget.config = {"chart_type": "line", "y_axis": "value"}
        
        data = {"data": []}
        
        result = self.chart_generator.generate_chart(widget, data)
        
        # Should return empty string for empty data
        self.assertEqual(result, "")
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_generate_matplotlib_chart(self, mock_close, mock_savefig):
        """Test generating chart with matplotlib."""
        widget = DashboardWidget("test_chart", "chart", "Test Chart")
        widget.config = {"chart_type": "line", "y_axis": "value"}
        
        data = {
            "data": [
                {"timestamp": "2023-01-01T00:00:00Z", "value": 10},
                {"timestamp": "2023-01-01T01:00:00Z", "value": 20}
            ]
        }
        
        # Mock PLOTLY_AVAILABLE as False
        with patch('src.qemlflow.observability.dashboard.PLOTLY_AVAILABLE', False):
            result = self.chart_generator.generate_chart(widget, data)
        
        self.assertIn("test_chart.png", result)
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()


class TestTrendAnalyzer(unittest.TestCase):
    """Test TrendAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.trend_analyzer = TrendAnalyzer()
    
    def test_analyze_trends_empty_data(self):
        """Test trend analysis with empty data."""
        result = self.trend_analyzer.analyze_trends([], "test_metric", "24h")
        
        self.assertEqual(result.metric_name, "test_metric")
        self.assertEqual(result.trend_direction, "unknown")
        self.assertEqual(result.trend_strength, 0.0)
    
    def test_analyze_trends_increasing(self):
        """Test trend analysis with increasing data."""
        data = [
            {"timestamp": "2023-01-01T00:00:00Z", "value": 10},
            {"timestamp": "2023-01-01T01:00:00Z", "value": 20},
            {"timestamp": "2023-01-01T02:00:00Z", "value": 30},
            {"timestamp": "2023-01-01T03:00:00Z", "value": 40},
            {"timestamp": "2023-01-01T04:00:00Z", "value": 50}
        ]
        
        result = self.trend_analyzer.analyze_trends(data, "test_metric", "24h")
        
        self.assertEqual(result.metric_name, "test_metric")
        self.assertEqual(result.trend_direction, "increasing")
        self.assertGreater(result.trend_strength, 0.8)
        self.assertGreater(result.change_rate, 0)
    
    def test_analyze_trends_stable(self):
        """Test trend analysis with stable data."""
        data = [
            {"timestamp": "2023-01-01T00:00:00Z", "value": 50},
            {"timestamp": "2023-01-01T01:00:00Z", "value": 51},
            {"timestamp": "2023-01-01T02:00:00Z", "value": 49},
            {"timestamp": "2023-01-01T03:00:00Z", "value": 50},
            {"timestamp": "2023-01-01T04:00:00Z", "value": 50}
        ]
        
        result = self.trend_analyzer.analyze_trends(data, "test_metric", "24h")
        
        self.assertEqual(result.metric_name, "test_metric")
        self.assertEqual(result.trend_direction, "stable")
        self.assertLess(result.trend_strength, 0.2)
    
    def test_detect_anomalies(self):
        """Test anomaly detection."""
        # Create series with outliers
        values = [10, 12, 11, 13, 10, 100, 11, 12]  # 100 is an outlier
        series = pd.Series(values)
        
        anomalies = self.trend_analyzer._detect_anomalies(series)
        
        self.assertGreater(len(anomalies), 0)
        # The outlier value should be detected
        outlier_values = [a["value"] for a in anomalies]
        self.assertIn(100.0, outlier_values)


class TestReportGenerator(unittest.TestCase):
    """Test ReportGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.template_dir = Path(self.temp_dir) / "templates"
        self.output_dir = Path(self.temp_dir) / "reports"
        self.report_generator = ReportGenerator(
            str(self.output_dir), 
            str(self.template_dir)
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_generate_report_html(self):
        """Test generating HTML report."""
        data = {
            "timestamp": datetime.now().isoformat(),
            "metrics": [
                {"name": "CPU Usage", "value": "75%", "status": "OK"},
                {"name": "Memory Usage", "value": "60%", "status": "OK"}
            ]
        }
        
        result = self.report_generator.generate_report("test", data, "html")
        
        self.assertNotEqual(result, "")
        self.assertTrue(Path(result).exists())
        self.assertIn("test_report_", result)
        self.assertIn(".html", result)
    
    def test_generate_report_json(self):
        """Test generating JSON report."""
        data = {
            "timestamp": datetime.now().isoformat(),
            "metrics": [{"name": "test", "value": "100"}]
        }
        
        result = self.report_generator.generate_report("test", data, "json")
        
        self.assertNotEqual(result, "")
        self.assertTrue(Path(result).exists())
        self.assertIn(".json", result)
        
        # Verify JSON content
        with open(result, 'r') as f:
            report_content = json.load(f)
        
        self.assertEqual(report_content["report_type"], "test")
        self.assertIn("metrics", report_content)


class TestDashboardManager(unittest.TestCase):
    """Test DashboardManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = DashboardManager(self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_manager_initialization(self):
        """Test dashboard manager initialization."""
        self.assertIsNotNone(self.manager.chart_generator)
        self.assertIsNotNone(self.manager.trend_analyzer)
        self.assertIsNotNone(self.manager.report_generator)
        self.assertEqual(len(self.manager.dashboards), 0)
    
    def test_create_dashboard(self):
        """Test creating dashboard."""
        dashboard = self.manager.create_dashboard(
            "test_dashboard",
            "Test Dashboard",
            "Test dashboard description"
        )
        
        self.assertEqual(dashboard.dashboard_id, "test_dashboard")
        self.assertEqual(dashboard.name, "Test Dashboard")
        self.assertIn("test_dashboard", self.manager.dashboards)
    
    def test_add_widget(self):
        """Test adding widget to dashboard."""
        # Create dashboard first
        self.manager.create_dashboard("test_dashboard", "Test")
        
        widget = DashboardWidget("test_widget", "chart", "Test Chart")
        success = self.manager.add_widget("test_dashboard", widget)
        
        self.assertTrue(success)
        dashboard = self.manager.dashboards["test_dashboard"]
        self.assertEqual(len(dashboard.widgets), 1)
        self.assertEqual(dashboard.widgets[0].widget_id, "test_widget")
    
    def test_add_widget_nonexistent_dashboard(self):
        """Test adding widget to non-existent dashboard."""
        widget = DashboardWidget("test_widget", "chart", "Test Chart")
        success = self.manager.add_widget("nonexistent", widget)
        
        self.assertFalse(success)
    
    def test_remove_widget(self):
        """Test removing widget from dashboard."""
        # Create dashboard and add widget
        self.manager.create_dashboard("test_dashboard", "Test")
        widget = DashboardWidget("test_widget", "chart", "Test Chart")
        self.manager.add_widget("test_dashboard", widget)
        
        # Remove widget
        success = self.manager.remove_widget("test_dashboard", "test_widget")
        
        self.assertTrue(success)
        dashboard = self.manager.dashboards["test_dashboard"]
        self.assertEqual(len(dashboard.widgets), 0)
    
    def test_refresh_dashboard_nonexistent(self):
        """Test refreshing non-existent dashboard."""
        result = self.manager.refresh_dashboard("nonexistent")
        
        self.assertIn("error", result)
        self.assertEqual(result["error"], "Dashboard not found")
    
    def test_refresh_dashboard_empty(self):
        """Test refreshing empty dashboard."""
        self.manager.create_dashboard("test_dashboard", "Test")
        
        result = self.manager.refresh_dashboard("test_dashboard")
        
        self.assertEqual(result["dashboard_id"], "test_dashboard")
        self.assertEqual(result["name"], "Test")
        self.assertEqual(len(result["widgets"]), 0)
        self.assertIn("last_updated", result)
    
    def test_add_data_source(self):
        """Test adding data source."""
        mock_data_source = Mock()
        self.manager.add_data_source("test_source", mock_data_source)
        
        self.assertIn("test_source", self.manager.data_sources)
        self.assertEqual(self.manager.data_sources["test_source"], mock_data_source)
    
    def test_generate_dashboard_report(self):
        """Test generating dashboard report."""
        # Create dashboard with widget
        self.manager.create_dashboard("test_dashboard", "Test")
        widget = DashboardWidget("test_widget", "metric", "Test Metric")
        self.manager.add_widget("test_dashboard", widget)
        
        # Mock data source
        mock_data_source = Mock()
        mock_data_source.get_data.return_value = {
            "value": "100%",
            "status": "OK"
        }
        self.manager.add_data_source("monitoring", mock_data_source)
        
        result = self.manager.generate_dashboard_report("test_dashboard", "html")
        
        self.assertNotEqual(result, "")
        self.assertTrue(Path(result).exists())


class TestGlobalFunctions(unittest.TestCase):
    """Test global dashboard functions."""
    
    def tearDown(self):
        """Clean up after tests."""
        shutdown_dashboard_system()
    
    def test_get_dashboard_manager(self):
        """Test getting dashboard manager."""
        manager1 = get_dashboard_manager()
        manager2 = get_dashboard_manager()
        
        # Should return same instance
        self.assertIs(manager1, manager2)
        self.assertIsNotNone(manager1)
    
    def test_initialize_dashboard_system(self):
        """Test initializing dashboard system."""
        temp_dir = tempfile.mkdtemp()
        
        try:
            manager = initialize_dashboard_system(temp_dir)
            
            self.assertIsNotNone(manager)
            self.assertEqual(str(manager.storage_dir), temp_dir)
            # Should have monitoring data source by default
            self.assertIn("monitoring", manager.data_sources)
        finally:
            shutdown_dashboard_system()
            import shutil
            if Path(temp_dir).exists():
                shutil.rmtree(temp_dir)
    
    def test_shutdown_dashboard_system(self):
        """Test shutting down dashboard system."""
        # Initialize first
        manager = get_dashboard_manager()
        self.assertIsNotNone(manager)
        
        # Shutdown
        shutdown_dashboard_system()
        
        # Should be cleaned up
        # Note: Can't easily test this without accessing global variable


if __name__ == '__main__':
    unittest.main()
