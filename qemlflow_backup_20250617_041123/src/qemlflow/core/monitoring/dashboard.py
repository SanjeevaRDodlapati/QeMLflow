"""
ChemML Smart Performance Dashboard
=================================

Real-time performance monitoring and visualization for ChemML operations.
Builds on the existing PerformanceMonitor to provide actionable insights.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..common.performance import PerformanceMonitor


class PerformanceDashboard:
    """Smart dashboard for ChemML performance monitoring."""

    def __init__(self, output_dir: str = "./performance_reports") -> None:
        self.monitor = PerformanceMonitor.get_instance()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def generate_real_time_report(self) -> Dict[str, Any]:
        """Generate comprehensive real-time performance report."""
        summary = self.monitor.get_summary()

        return {
            "timestamp": datetime.now().isoformat(),
            "system_info": self._get_system_metrics(),
            "performance_summary": summary,
            "top_functions_by_time": self._get_slowest_functions(summary),
            "memory_hotspots": self._get_memory_intensive_ops(summary),
            "optimization_suggestions": self._suggest_optimizations(summary),
            "health_score": self._calculate_health_score(summary),
        }

    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get current system performance metrics."""
        import psutil

        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage("/").percent,
            "active_processes": len(psutil.pids()),
            "available_memory_gb": psutil.virtual_memory().available / (1024**3),
        }

    def _get_slowest_functions(
        self, summary: Dict[str, Any], top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """Identify functions with highest execution times."""
        if not summary:
            return []

        functions_by_time = []
        for func_name, stats in summary.items():
            if isinstance(stats, dict) and "total_time" in stats:
                functions_by_time.append(
                    {
                        "function": func_name,
                        "total_time": stats["total_time"],
                        "avg_time": stats.get("avg_time", 0),
                        "call_count": stats.get("call_count", 0),
                        "efficiency_score": self._calculate_efficiency_score(stats),
                    }
                )

        return sorted(functions_by_time, key=lambda x: x["total_time"], reverse=True)[
            :top_n
        ]

    def _get_memory_intensive_ops(
        self, summary: Dict[str, Any], top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """Identify memory-intensive operations."""
        if not summary:
            return []

        memory_ops = []
        for func_name, stats in summary.items():
            if isinstance(stats, dict) and "max_memory" in stats:
                memory_ops.append(
                    {
                        "function": func_name,
                        "max_memory_mb": stats["max_memory"],
                        "avg_memory_mb": stats.get("avg_memory", 0),
                        "call_count": stats.get("call_count", 0),
                        "memory_efficiency": self._calculate_memory_efficiency(stats),
                    }
                )

        return sorted(memory_ops, key=lambda x: x["max_memory_mb"], reverse=True)[
            :top_n
        ]

    def _suggest_optimizations(self, summary: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate intelligent optimization suggestions."""
        suggestions = []

        if not summary:
            return [
                {
                    "type": "info",
                    "message": "No performance data available yet. Run some ChemML operations to see suggestions.",
                }
            ]

        # Analyze for slow functions
        slow_functions = self._get_slowest_functions(summary, 3)
        for func in slow_functions:
            if (
                func["avg_time"] > 5.0
            ):  # Functions taking more than 5 seconds on average
                suggestions.append(
                    {
                        "type": "performance",
                        "function": func["function"],
                        "message": f"Function '{func['function']}' averaging {func['avg_time']:.2f}s per call. Consider caching or optimization.",
                    }
                )

        # Analyze for memory usage
        memory_ops = self._get_memory_intensive_ops(summary, 3)
        for op in memory_ops:
            if op["max_memory_mb"] > 500:  # Functions using more than 500MB
                suggestions.append(
                    {
                        "type": "memory",
                        "function": op["function"],
                        "message": f"Function '{op['function']}' using up to {op['max_memory_mb']:.1f}MB. Consider batch processing or streaming.",
                    }
                )

        # General suggestions
        if len(summary) > 10:
            suggestions.append(
                {
                    "type": "general",
                    "message": "Consider implementing function-level caching for frequently called operations.",
                }
            )

        return (
            suggestions
            if suggestions
            else [
                {
                    "type": "success",
                    "message": "Performance looks good! All operations running efficiently.",
                }
            ]
        )

    def _calculate_efficiency_score(self, stats: Dict[str, Any]) -> float:
        """Calculate efficiency score for a function (0-100)."""
        avg_time = stats.get("avg_time", 0)
        call_count = stats.get("call_count", 1)

        # Lower average time and higher call count indicate better efficiency
        if avg_time == 0:
            return 100.0

        # Normalize score (this is a simple heuristic)
        base_score = min(100, (1.0 / avg_time) * 10)
        frequency_bonus = min(20, call_count / 10)

        return min(100.0, base_score + frequency_bonus)

    def _calculate_memory_efficiency(self, stats: Dict[str, Any]) -> float:
        """Calculate memory efficiency score (0-100)."""
        max_memory = stats.get("max_memory", 0)
        avg_memory = stats.get("avg_memory", 0)

        if max_memory == 0:
            return 100.0

        # Lower memory usage indicates better efficiency
        efficiency = max(0, 100 - (max_memory / 10))  # Rough heuristic
        consistency = 100 - abs(max_memory - avg_memory) if avg_memory > 0 else 50

        return (efficiency + consistency) / 2

    def _calculate_health_score(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall system health score."""
        if not summary:
            return {
                "score": 100,
                "status": "Excellent",
                "details": "No performance issues detected",
            }

        total_functions = len(summary)
        slow_functions = len(
            [f for f in self._get_slowest_functions(summary) if f["avg_time"] > 2.0]
        )
        memory_intensive = len(
            [
                f
                for f in self._get_memory_intensive_ops(summary)
                if f["max_memory_mb"] > 200
            ]
        )

        # Calculate health score
        slow_penalty = (
            (slow_functions / total_functions) * 30 if total_functions > 0 else 0
        )
        memory_penalty = (
            (memory_intensive / total_functions) * 20 if total_functions > 0 else 0
        )

        score = max(0, 100 - slow_penalty - memory_penalty)

        if score >= 90:
            status = "Excellent"
        elif score >= 75:
            status = "Good"
        elif score >= 60:
            status = "Fair"
        else:
            status = "Needs Attention"

        return {
            "score": round(score, 1),
            "status": status,
            "details": f"Analyzed {total_functions} functions. {slow_functions} slow, {memory_intensive} memory-intensive.",
        }

    def generate_html_dashboard(self, save_to_file: bool = True) -> str:
        """Generate an HTML dashboard for performance monitoring."""
        report = self.generate_real_time_report()

        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ChemML Performance Dashboard</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f7;
            color: #1d1d1f;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
            border: 1px solid #e5e5e7;
        }}
        .metric-title {{
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 15px;
            color: #1d1d1f;
        }}
        .health-score {{
            font-size: 48px;
            font-weight: 700;
            text-align: center;
            margin: 20px 0;
        }}
        .health-excellent {{ color: #30d158; }}
        .health-good {{ color: #32d74b; }}
        .health-fair {{ color: #ff9500; }}
        .health-attention {{ color: #ff3b30; }}
        .system-metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 15px;
        }}
        .system-metric {{
            text-align: center;
            padding: 10px;
            background: #f9f9f9;
            border-radius: 8px;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: 600;
            margin-bottom: 5px;
        }}
        .metric-label {{
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
        }}
        .function-list {{
            max-height: 300px;
            overflow-y: auto;
        }}
        .function-item {{
            display: flex;
            justify-content: between;
            align-items: center;
            padding: 10px;
            border-bottom: 1px solid #e5e5e7;
        }}
        .function-name {{
            flex: 1;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 14px;
        }}
        .function-metric {{
            font-weight: 600;
            color: #666;
        }}
        .suggestions-list {{
            list-style: none;
            padding: 0;
        }}
        .suggestion-item {{
            padding: 12px;
            margin-bottom: 10px;
            border-radius: 8px;
            border-left: 4px solid;
        }}
        .suggestion-performance {{ background: #fff3cd; border-color: #ff9500; }}
        .suggestion-memory {{ background: #f8d7da; border-color: #ff3b30; }}
        .suggestion-general {{ background: #d1ecf1; border-color: #007bff; }}
        .suggestion-success {{ background: #d4edda; border-color: #30d158; }}
        .timestamp {{
            text-align: center;
            color: #666;
            font-size: 14px;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧬 ChemML Performance Dashboard</h1>
            <p>Real-time monitoring and optimization insights</p>
        </div>

        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-title">📊 System Health</div>
                <div class="health-score health-{report['health_score']['status'].lower().replace(' ', '-')}">
                    {report['health_score']['score']}%
                </div>
                <div style="text-align: center;">
                    <strong>{report['health_score']['status']}</strong><br>
                    <small>{report['health_score']['details']}</small>
                </div>
            </div>

            <div class="metric-card">
                <div class="metric-title">🖥️ System Metrics</div>
                <div class="system-metrics">
                    <div class="system-metric">
                        <div class="metric-value">{report['system_info']['cpu_percent']:.1f}%</div>
                        <div class="metric-label">CPU</div>
                    </div>
                    <div class="system-metric">
                        <div class="metric-value">{report['system_info']['memory_percent']:.1f}%</div>
                        <div class="metric-label">Memory</div>
                    </div>
                    <div class="system-metric">
                        <div class="metric-value">{report['system_info']['available_memory_gb']:.1f}GB</div>
                        <div class="metric-label">Available</div>
                    </div>
                </div>
            </div>
        </div>

        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-title">⏱️ Slowest Functions</div>
                <div class="function-list">
                    {"".join([f'''
                    <div class="function-item">
                        <div class="function-name">{func['function']}</div>
                        <div class="function-metric">{func['avg_time']:.2f}s</div>
                    </div>
                    ''' for func in report['top_functions_by_time'][:5]])}
                </div>
            </div>

            <div class="metric-card">
                <div class="metric-title">🧠 Memory Usage</div>
                <div class="function-list">
                    {"".join([f'''
                    <div class="function-item">
                        <div class="function-name">{func['function']}</div>
                        <div class="function-metric">{func['max_memory_mb']:.1f}MB</div>
                    </div>
                    ''' for func in report['memory_hotspots'][:5]])}
                </div>
            </div>
        </div>

        <div class="metric-card">
            <div class="metric-title">💡 Optimization Suggestions</div>
            <ul class="suggestions-list">
                {"".join([f'''
                <li class="suggestion-item suggestion-{suggestion.get('type', 'general')}">
                    {suggestion['message']}
                </li>
                ''' for suggestion in report['optimization_suggestions']])}
            </ul>
        </div>

        <div class="timestamp">
            Last updated: {report['timestamp']}
        </div>
    </div>
</body>
</html>
        """

        if save_to_file:
            dashboard_path = self.output_dir / "performance_dashboard.html"
            with open(dashboard_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            print(f"📊 Performance dashboard saved to: {dashboard_path}")

        return html_content

    def save_report(self, filename: Optional[str] = None) -> Path:
        """Save performance report to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_report_{timestamp}.json"

        report_path = self.output_dir / filename
        report = self.generate_real_time_report()

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)

        print(f"📄 Performance report saved to: {report_path}")
        return report_path

    def start_monitoring(
        self, interval_seconds: int = 60, duration_minutes: Optional[int] = None
    ):
        """Start continuous performance monitoring."""
        print(
            f"🚀 Starting ChemML performance monitoring (interval: {interval_seconds}s)"
        )

        start_time = time.time()
        iteration = 0

        try:
            while True:
                iteration += 1

                # Generate and save report
                self.save_report(f"monitoring_report_{iteration:04d}.json")
                self.generate_html_dashboard()

                print(f"📊 Monitoring iteration {iteration} completed")

                # Check duration limit
                if duration_minutes and (time.time() - start_time) >= (
                    duration_minutes * 60
                ):
                    print(f"⏰ Monitoring completed after {duration_minutes} minutes")
                    break

                # Wait for next iteration
                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            print(f"\n🛑 Monitoring stopped by user after {iteration} iterations")


# Convenience function for quick dashboard generation
def create_performance_dashboard() -> PerformanceDashboard:
    """Create and return a performance dashboard instance."""
    return PerformanceDashboard()


def show_performance_dashboard() -> None:
    """Generate and display performance dashboard."""
    dashboard = create_performance_dashboard()
    dashboard.generate_html_dashboard()

    # Try to open in browser
    try:
        import webbrowser

        dashboard_path = dashboard.output_dir / "performance_dashboard.html"
        webbrowser.open(f"file://{dashboard_path.absolute()}")
        print("🌐 Performance dashboard opened in browser")
    except Exception:
        print(
            "📊 Performance dashboard generated. Open performance_reports/performance_dashboard.html in your browser."
        )
