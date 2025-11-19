"""
Dashboard & Reporting Module

This module provides comprehensive monitoring dashboards, automated reporting,
trend analysis, and performance benchmarking capabilities for enterprise-grade
observability and operational intelligence.
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import matplotlib.pyplot as plt
import pandas as pd
from jinja2 import Environment, FileSystemLoader

try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


@dataclass
class DashboardWidget:
    """Represents a dashboard widget component."""
    widget_id: str
    widget_type: str  # 'chart', 'metric', 'table', 'alert'
    title: str
    description: str = ""
    data_source: str = ""
    refresh_interval: int = 60  # seconds
    config: Dict[str, Any] = field(default_factory=dict)
    position: Dict[str, int] = field(default_factory=lambda: {"row": 0, "col": 0, "width": 1, "height": 1})
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert widget to dictionary."""
        data = asdict(self)
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat() if value else None
        return data


@dataclass
class DashboardLayout:
    """Dashboard layout configuration."""
    dashboard_id: str
    name: str
    description: str = ""
    widgets: List[DashboardWidget] = field(default_factory=list)
    layout_config: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = None
    
    def add_widget(self, widget: DashboardWidget):
        """Add widget to dashboard."""
        self.widgets.append(widget)
        self.updated_at = datetime.now(timezone.utc)
    
    def remove_widget(self, widget_id: str) -> bool:
        """Remove widget from dashboard."""
        initial_count = len(self.widgets)
        self.widgets = [w for w in self.widgets if w.widget_id != widget_id]
        if len(self.widgets) < initial_count:
            self.updated_at = datetime.now(timezone.utc)
            return True
        return False


@dataclass
class ReportSchedule:
    """Automated report scheduling configuration."""
    schedule_id: str
    report_type: str
    frequency: str  # 'hourly', 'daily', 'weekly', 'monthly'
    recipients: List[str] = field(default_factory=list)
    format: str = "html"  # 'html', 'pdf', 'json'
    enabled: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrendAnalysis:
    """Trend analysis results."""
    metric_name: str
    time_period: str
    trend_direction: str  # 'increasing', 'decreasing', 'stable', 'volatile'
    trend_strength: float  # 0.0 to 1.0
    change_rate: float
    statistical_significance: float
    anomalies_detected: List[Dict[str, Any]] = field(default_factory=list)
    predictions: Dict[str, Any] = field(default_factory=dict)
    confidence_intervals: Dict[str, float] = field(default_factory=dict)


class DataSource(ABC):
    """Abstract base class for dashboard data sources."""
    
    @abstractmethod
    def get_data(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve data for dashboard widgets."""
        pass
    
    @abstractmethod
    def get_available_metrics(self) -> List[str]:
        """Get list of available metrics."""
        pass


class MonitoringDataSource(DataSource):
    """Data source for monitoring metrics."""
    
    def __init__(self, storage_dir: str = "metrics_data"):
        self.storage_dir = Path(storage_dir)
        self.logger = logging.getLogger(__name__)
    
    def get_data(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve monitoring data."""
        try:
            metric_type = query.get("metric_type", "")
            time_range = query.get("time_range", "1h")
            aggregation = query.get("aggregation", "avg")
            
            # Load metric data
            data = self._load_metric_data(metric_type, time_range)
            
            # Apply aggregation
            aggregated_data = self._aggregate_data(data, aggregation)
            
            return {
                "metric_type": metric_type,
                "time_range": time_range,
                "data": aggregated_data,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        except Exception as e:
            self.logger.error(f"Failed to get monitoring data: {e}")
            return {"error": str(e)}
    
    def get_available_metrics(self) -> List[str]:
        """Get available monitoring metrics."""
        try:
            metrics = []
            if self.storage_dir.exists():
                for file_path in self.storage_dir.glob("*.json"):
                    metrics.append(file_path.stem)
            return metrics
        except Exception as e:
            self.logger.error(f"Failed to get available metrics: {e}")
            return []
    
    def _load_metric_data(self, metric_type: str, time_range: str) -> List[Dict[str, Any]]:
        """Load metric data from storage."""
        data = []
        try:
            metric_file = self.storage_dir / f"{metric_type}.json"
            if metric_file.exists():
                with open(metric_file, 'r') as f:
                    all_data = json.load(f)
                
                # Filter by time range
                cutoff_time = self._calculate_cutoff_time(time_range)
                data = [
                    item for item in all_data
                    if datetime.fromisoformat(item.get("timestamp", "")).replace(tzinfo=timezone.utc) >= cutoff_time
                ]
        
        except Exception as e:
            self.logger.error(f"Failed to load metric data: {e}")
        
        return data
    
    def _calculate_cutoff_time(self, time_range: str) -> datetime:
        """Calculate cutoff time for data filtering."""
        now = datetime.now(timezone.utc)
        
        try:
            if time_range.endswith('h'):
                hours = int(time_range[:-1])
                return now - timedelta(hours=hours)
            elif time_range.endswith('d'):
                days = int(time_range[:-1])
                return now - timedelta(days=days)
            elif time_range.endswith('w'):
                weeks = int(time_range[:-1])
                return now - timedelta(weeks=weeks)
            else:
                return now - timedelta(hours=1)  # Default to 1 hour
        except (ValueError, IndexError):
            return now - timedelta(hours=1)  # Default to 1 hour on error
    
    def _aggregate_data(self, data: List[Dict[str, Any]], aggregation: str) -> List[Dict[str, Any]]:
        """Aggregate data points."""
        if not data:
            return []
        
        if aggregation == "raw":
            return data
        
        # Group data by time buckets for aggregation
        df = pd.DataFrame(data)
        if df.empty:
            return []
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Aggregate by time buckets
        if aggregation == "avg":
            aggregated = df.resample('5min').mean()
        elif aggregation == "max":
            aggregated = df.resample('5min').max()
        elif aggregation == "min":
            aggregated = df.resample('5min').min()
        elif aggregation == "sum":
            aggregated = df.resample('5min').sum()
        else:
            return data
        
        # Convert back to list of dictionaries
        result = []
        for timestamp, row in aggregated.iterrows():
            entry = {"timestamp": str(timestamp)}
            entry.update(row.to_dict())
            result.append(entry)
        
        return result


class ChartGenerator:
    """Generates charts for dashboard widgets."""
    
    def __init__(self, output_dir: str = "dashboard_charts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def generate_chart(self, widget: DashboardWidget, data: Dict[str, Any]) -> str:
        """Generate chart based on widget configuration."""
        try:
            chart_type = widget.config.get("chart_type", "line")
            
            if PLOTLY_AVAILABLE:
                return self._generate_plotly_chart(widget, data, chart_type)
            else:
                return self._generate_matplotlib_chart(widget, data, chart_type)
        
        except Exception as e:
            self.logger.error(f"Failed to generate chart for {widget.widget_id}: {e}")
            return ""
    
    def _generate_plotly_chart(self, widget: DashboardWidget, data: Dict[str, Any], chart_type: str) -> str:
        """Generate interactive chart using Plotly."""
        chart_data = data.get("data", [])
        if not chart_data:
            return ""
        
        df = pd.DataFrame(chart_data)
        
        if chart_type == "line":
            fig = px.line(df, x="timestamp", y=widget.config.get("y_axis", "value"),
                         title=widget.title)
        elif chart_type == "bar":
            fig = px.bar(df, x="timestamp", y=widget.config.get("y_axis", "value"),
                        title=widget.title)
        elif chart_type == "scatter":
            fig = px.scatter(df, x="timestamp", y=widget.config.get("y_axis", "value"),
                           title=widget.title)
        elif chart_type == "histogram":
            fig = px.histogram(df, x=widget.config.get("x_axis", "value"),
                             title=widget.title)
        else:
            fig = px.line(df, x="timestamp", y=widget.config.get("y_axis", "value"),
                         title=widget.title)
        
        # Save as HTML
        chart_file = self.output_dir / f"{widget.widget_id}.html"
        fig.write_html(str(chart_file))
        
        return str(chart_file)
    
    def _generate_matplotlib_chart(self, widget: DashboardWidget, data: Dict[str, Any], chart_type: str) -> str:
        """Generate static chart using Matplotlib."""
        chart_data = data.get("data", [])
        if not chart_data:
            return ""
        
        df = pd.DataFrame(chart_data)
        
        plt.figure(figsize=(10, 6))
        
        if chart_type == "line":
            plt.plot(df["timestamp"], df.get(widget.config.get("y_axis", "value"), []))
        elif chart_type == "bar":
            plt.bar(df["timestamp"], df.get(widget.config.get("y_axis", "value"), []))
        elif chart_type == "scatter":
            plt.scatter(df["timestamp"], df.get(widget.config.get("y_axis", "value"), []))
        elif chart_type == "histogram":
            plt.hist(df.get(widget.config.get("x_axis", "value"), []))
        
        plt.title(widget.title)
        plt.xlabel(widget.config.get("x_label", "Time"))
        plt.ylabel(widget.config.get("y_label", "Value"))
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save as PNG
        chart_file = self.output_dir / f"{widget.widget_id}.png"
        plt.savefig(str(chart_file))
        plt.close()
        
        return str(chart_file)


class TrendAnalyzer:
    """Analyzes trends in time series data."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_trends(self, data: List[Dict[str, Any]], metric_name: str, 
                      time_period: str = "24h") -> TrendAnalysis:
        """Analyze trends in metric data."""
        try:
            df = pd.DataFrame(data)
            if df.empty:
                return self._empty_trend_analysis(metric_name, time_period)
            
            # Prepare data
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            # Extract numeric values
            values = df.select_dtypes(include=[float, int])
            if values.empty:
                return self._empty_trend_analysis(metric_name, time_period)
            
            # Analyze trend for first numeric column
            value_column = values.columns[0]
            series = values[value_column].dropna()
            
            if len(series) < 2:
                return self._empty_trend_analysis(metric_name, time_period)
            
            # Calculate trend metrics
            trend_direction = self._calculate_trend_direction(series)
            trend_strength = self._calculate_trend_strength(series)
            change_rate = self._calculate_change_rate(series)
            significance = self._calculate_statistical_significance(series)
            anomalies = self._detect_anomalies(series)
            
            return TrendAnalysis(
                metric_name=metric_name,
                time_period=time_period,
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                change_rate=change_rate,
                statistical_significance=significance,
                anomalies_detected=anomalies,
                predictions={},
                confidence_intervals={}
            )
        
        except Exception as e:
            self.logger.error(f"Failed to analyze trends: {e}")
            return self._empty_trend_analysis(metric_name, time_period)
    
    def _empty_trend_analysis(self, metric_name: str, time_period: str) -> TrendAnalysis:
        """Return empty trend analysis."""
        return TrendAnalysis(
            metric_name=metric_name,
            time_period=time_period,
            trend_direction="unknown",
            trend_strength=0.0,
            change_rate=0.0,
            statistical_significance=0.0
        )
    
    def _calculate_trend_direction(self, series: pd.Series) -> str:
        """Calculate overall trend direction."""
        if len(series) < 2:
            return "unknown"
        
        # Calculate linear regression slope
        x = pd.Series(range(len(series)))
        y = series
        
        # Calculate slope using linear regression formula
        n = len(series)
        slope = ((n * (x * y).sum()) - (x.sum() * y.sum())) / ((n * (x * x).sum()) - (x.sum() ** 2))
        
        # Normalize slope by the mean to get percentage change per time unit
        if y.mean() != 0:
            normalized_slope = slope / y.mean()
        else:
            normalized_slope = slope
        
        if normalized_slope > 0.02:  # More than 2% change per unit time
            return "increasing"
        elif normalized_slope < -0.02:  # More than 2% change per unit time
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_trend_strength(self, series: pd.Series) -> float:
        """Calculate trend strength (0.0 to 1.0)."""
        if len(series) < 2:
            return 0.0
        
        # Calculate R-squared for linear trend fit
        x = pd.Series(range(len(series)))
        y = series
        
        # Calculate slope and intercept
        n = len(series)
        if n < 2:
            return 0.0
            
        slope = ((n * (x * y).sum()) - (x.sum() * y.sum())) / ((n * (x * x).sum()) - (x.sum() ** 2))
        intercept = (y.sum() - slope * x.sum()) / n
        
        # Calculate R-squared
        y_pred = slope * x + intercept
        ss_res = ((y - y_pred) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        
        if ss_tot == 0:
            return 0.0
            
        r_squared = 1 - (ss_res / ss_tot)
        return max(0.0, min(1.0, r_squared))
    
    def _calculate_change_rate(self, series: pd.Series) -> float:
        """Calculate percentage change rate."""
        if len(series) < 2:
            return 0.0
        
        first_value = series.iloc[0]
        last_value = series.iloc[-1]
        
        if first_value == 0:
            return 0.0
        
        return ((last_value - first_value) / first_value) * 100
    
    def _calculate_statistical_significance(self, series: pd.Series) -> float:
        """Calculate statistical significance of trend."""
        if len(series) < 3:
            return 0.0
        
        # Simple approach: use correlation coefficient as significance
        x = range(len(series))
        correlation = pd.Series(x).corr(series)
        return abs(correlation) if not pd.isna(correlation) else 0.0
    
    def _detect_anomalies(self, series: pd.Series) -> List[Dict[str, Any]]:
        """Detect anomalies in time series."""
        anomalies = []
        
        if len(series) < 5:
            return anomalies
        
        # Simple outlier detection using IQR
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        for idx, value in series.items():
            if value < lower_bound or value > upper_bound:
                anomalies.append({
                    "index": str(idx),
                    "value": float(value),
                    "type": "outlier",
                    "severity": "high" if abs(value - series.median()) > 2 * series.std() else "medium"
                })
        
        return anomalies


class ReportGenerator:
    """Generates automated reports."""
    
    def __init__(self, output_dir: str = "reports", template_dir: str = "templates"):
        self.output_dir = Path(output_dir)
        self.template_dir = Path(template_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.template_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Initialize Jinja2 environment
        self.jinja_env = Environment(loader=FileSystemLoader(str(self.template_dir)))
    
    def generate_report(self, report_type: str, data: Dict[str, Any], 
                       format: str = "html") -> str:
        """Generate report in specified format."""
        try:
            template_name = f"{report_type}_report.{format}.j2"
            template_path = self.template_dir / template_name
            
            # Create default template if it doesn't exist
            if not template_path.exists():
                self._create_default_template(template_path, report_type, format)
            
            # Load template
            template = self.jinja_env.get_template(template_name)
            
            # Render report
            template_data = {**data, 'report_type': report_type}
            report_content = template.render(**template_data)
            
            # Save report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = self.output_dir / f"{report_type}_report_{timestamp}.{format}"
            
            with open(report_file, 'w') as f:
                f.write(report_content)
            
            self.logger.info(f"Generated report: {report_file}")
            return str(report_file)
        
        except Exception as e:
            self.logger.error(f"Failed to generate report: {e}")
            return ""
    
    def _create_default_template(self, template_path: Path, report_type: str, format: str):
        """Create default report template."""
        if format == "html":
            template_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{report_type.title()} Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f4f4f4; padding: 20px; border-radius: 5px; }}
        .metric {{ margin: 10px 0; padding: 10px; border-left: 3px solid #007cba; }}
        .chart {{ margin: 20px 0; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{report_type.title()} Report</h1>
        <p>Generated on: {{{{ timestamp }}}}</p>
    </div>
    
    <h2>Summary</h2>
    <div class="metric">
        <strong>Total Metrics:</strong> {{{{ metrics|length if metrics else 0 }}}}
    </div>
    
    {{% if metrics %}}
    <h2>Metrics</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>Value</th>
            <th>Status</th>
        </tr>
        {{% for metric in metrics %}}
        <tr>
            <td>{{{{ metric.name }}}}</td>
            <td>{{{{ metric.value }}}}</td>
            <td>{{{{ metric.status or 'N/A' }}}}</td>
        </tr>
        {{% endfor %}}
    </table>
    {{% endif %}}
    
    {{% if charts %}}
    <h2>Charts</h2>
    {{% for chart in charts %}}
    <div class="chart">
        <h3>{{{{ chart.title }}}}</h3>
        <img src="{{{{ chart.path }}}}" alt="{{{{ chart.title }}}}" style="max-width: 100%;">
    </div>
    {{% endfor %}}
    {{% endif %}}
</body>
</html>
"""
        elif format == "json":
            template_content = """
{
    "report_type": "{{ report_type }}",
    "timestamp": "{{ timestamp }}",
    "metrics": {{ metrics|tojson if metrics else '[]' }},
    "charts": {{ charts|tojson if charts else '[]' }},
    "summary": {
        "total_metrics": {{ metrics|length if metrics else 0 }}
    }
}
"""
        else:
            template_content = f"""
{report_type.upper()} REPORT
Generated on: {{{{ timestamp }}}}

SUMMARY:
Total Metrics: {{{{ metrics|length if metrics else 0 }}}}

{{% if metrics %}}
METRICS:
{{% for metric in metrics %}}
- {{{{ metric.name }}}}: {{{{ metric.value }}}} ({{{{ metric.status or 'N/A' }}}})
{{% endfor %}}
{{% endif %}}
"""
        
        with open(template_path, 'w') as f:
            f.write(template_content.strip())


class DashboardManager:
    """Manages dashboard layouts and widgets."""
    
    def __init__(self, storage_dir: str = "dashboard_data"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.dashboards: Dict[str, DashboardLayout] = {}
        self.data_sources: Dict[str, DataSource] = {}
        self.chart_generator = ChartGenerator()
        self.trend_analyzer = TrendAnalyzer()
        self.report_generator = ReportGenerator()
        self.logger = logging.getLogger(__name__)
        
        # Load existing dashboards
        self._load_dashboards()
    
    def add_data_source(self, name: str, data_source: DataSource):
        """Add data source for dashboard widgets."""
        self.data_sources[name] = data_source
    
    def create_dashboard(self, dashboard_id: str, name: str, description: str = "") -> DashboardLayout:
        """Create new dashboard."""
        dashboard = DashboardLayout(
            dashboard_id=dashboard_id,
            name=name,
            description=description
        )
        
        self.dashboards[dashboard_id] = dashboard
        self._save_dashboard(dashboard)
        return dashboard
    
    def add_widget(self, dashboard_id: str, widget: DashboardWidget) -> bool:
        """Add widget to dashboard."""
        if dashboard_id in self.dashboards:
            self.dashboards[dashboard_id].add_widget(widget)
            self._save_dashboard(self.dashboards[dashboard_id])
            return True
        return False
    
    def remove_widget(self, dashboard_id: str, widget_id: str) -> bool:
        """Remove widget from dashboard."""
        if dashboard_id in self.dashboards:
            success = self.dashboards[dashboard_id].remove_widget(widget_id)
            if success:
                self._save_dashboard(self.dashboards[dashboard_id])
            return success
        return False
    
    def refresh_dashboard(self, dashboard_id: str) -> Dict[str, Any]:
        """Refresh dashboard data and generate updated content."""
        if dashboard_id not in self.dashboards:
            return {"error": "Dashboard not found"}
        
        dashboard = self.dashboards[dashboard_id]
        dashboard_data = {
            "dashboard_id": dashboard_id,
            "name": dashboard.name,
            "description": dashboard.description,
            "widgets": [],
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
        
        for widget in dashboard.widgets:
            widget_data = self._refresh_widget(widget)
            dashboard_data["widgets"].append(widget_data)
        
        return dashboard_data
    
    def _refresh_widget(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Refresh individual widget data."""
        widget_data = widget.to_dict()
        
        try:
            # Get data from appropriate data source
            data_source_name = widget.data_source or "monitoring"
            if data_source_name in self.data_sources:
                data_source = self.data_sources[data_source_name]
                query = widget.config.get("query", {})
                raw_data = data_source.get_data(query)
                
                widget_data["data"] = raw_data
                
                # Generate chart if widget is chart type
                if widget.widget_type == "chart":
                    chart_path = self.chart_generator.generate_chart(widget, raw_data)
                    widget_data["chart_path"] = chart_path
                
                # Analyze trends if configured
                if widget.config.get("analyze_trends", False):
                    trend_data = raw_data.get("data", [])
                    trend_analysis = self.trend_analyzer.analyze_trends(
                        trend_data, widget.title, widget.config.get("trend_period", "24h")
                    )
                    widget_data["trend_analysis"] = asdict(trend_analysis)
                
                widget.last_updated = datetime.now(timezone.utc)
            
        except Exception as e:
            self.logger.error(f"Failed to refresh widget {widget.widget_id}: {e}")
            widget_data["error"] = str(e)
        
        return widget_data
    
    def generate_dashboard_report(self, dashboard_id: str, format: str = "html") -> str:
        """Generate report for dashboard."""
        dashboard_data = self.refresh_dashboard(dashboard_id)
        
        # Prepare report data
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "dashboard": dashboard_data,
            "metrics": [],
            "charts": []
        }
        
        # Extract metrics and charts
        for widget_data in dashboard_data.get("widgets", []):
            if widget_data.get("widget_type") == "metric":
                report_data["metrics"].append({
                    "name": widget_data.get("title", ""),
                    "value": widget_data.get("data", {}).get("value", "N/A"),
                    "status": widget_data.get("data", {}).get("status", "")
                })
            elif widget_data.get("widget_type") == "chart":
                report_data["charts"].append({
                    "title": widget_data.get("title", ""),
                    "path": widget_data.get("chart_path", "")
                })
        
        return self.report_generator.generate_report("dashboard", report_data, format)
    
    def _load_dashboards(self):
        """Load existing dashboards from storage."""
        try:
            dashboards_file = self.storage_dir / "dashboards.json"
            if dashboards_file.exists():
                with open(dashboards_file, 'r') as f:
                    dashboards_data = json.load(f)
                
                for dashboard_data in dashboards_data:
                    dashboard = DashboardLayout(
                        dashboard_id=dashboard_data["dashboard_id"],
                        name=dashboard_data["name"],
                        description=dashboard_data.get("description", ""),
                        layout_config=dashboard_data.get("layout_config", {})
                    )
                    
                    # Load widgets
                    for widget_data in dashboard_data.get("widgets", []):
                        widget = DashboardWidget(
                            widget_id=widget_data["widget_id"],
                            widget_type=widget_data["widget_type"],
                            title=widget_data["title"],
                            description=widget_data.get("description", ""),
                            data_source=widget_data.get("data_source", ""),
                            refresh_interval=widget_data.get("refresh_interval", 60),
                            config=widget_data.get("config", {}),
                            position=widget_data.get("position", {})
                        )
                        dashboard.add_widget(widget)
                    
                    self.dashboards[dashboard.dashboard_id] = dashboard
        
        except Exception as e:
            self.logger.error(f"Failed to load dashboards: {e}")
    
    def _save_dashboard(self, dashboard: DashboardLayout):
        """Save dashboard to storage."""
        try:
            dashboards_file = self.storage_dir / "dashboards.json"
            
            # Load existing dashboards
            dashboards_data = []
            if dashboards_file.exists():
                with open(dashboards_file, 'r') as f:
                    dashboards_data = json.load(f)
            
            # Update or add current dashboard
            dashboard_dict = asdict(dashboard)
            # Convert datetime objects
            for key, value in dashboard_dict.items():
                if isinstance(value, datetime):
                    dashboard_dict[key] = value.isoformat() if value else None
            
            # Handle widgets
            for widget_dict in dashboard_dict.get("widgets", []):
                for key, value in widget_dict.items():
                    if isinstance(value, datetime):
                        widget_dict[key] = value.isoformat() if value else None
            
            # Remove existing entry and add updated one
            dashboards_data = [d for d in dashboards_data if d.get("dashboard_id") != dashboard.dashboard_id]
            dashboards_data.append(dashboard_dict)
            
            # Save to file
            with open(dashboards_file, 'w') as f:
                json.dump(dashboards_data, f, indent=2)
        
        except Exception as e:
            self.logger.error(f"Failed to save dashboard: {e}")


# Global dashboard manager instance
_dashboard_manager: Optional[DashboardManager] = None


def get_dashboard_manager() -> DashboardManager:
    """Get global dashboard manager instance."""
    global _dashboard_manager
    if _dashboard_manager is None:
        _dashboard_manager = DashboardManager()
    return _dashboard_manager


def initialize_dashboard_system(storage_dir: str = "dashboard_data") -> DashboardManager:
    """Initialize dashboard system."""
    global _dashboard_manager
    _dashboard_manager = DashboardManager(storage_dir=storage_dir)
    
    # Add default data sources
    _dashboard_manager.add_data_source("monitoring", MonitoringDataSource())
    
    return _dashboard_manager


def shutdown_dashboard_system():
    """Shutdown dashboard system."""
    global _dashboard_manager
    _dashboard_manager = None
