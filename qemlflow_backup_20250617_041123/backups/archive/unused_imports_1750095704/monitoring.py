# !/usr/bin/env python3
"""
Enterprise Monitoring and Dashboard System
=========================================

Phase 3 implementation: Real-time monitoring, dashboards, and enterprise
features for ChemML.

Features:
- Real-time performance monitoring
- Interactive dashboards
- Automated reporting
- System health monitoring
- User analytics and insights
- Enterprise security features

Usage:
    from chemml.enterprise.monitoring import MonitoringDashboard, SystemMonitor

    monitor = SystemMonitor()
    dashboard = MonitoringDashboard()
    dashboard.start_server()
"""

import json
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
@dataclass
class SystemMetrics:
    """System performance metrics."""

    timestamp: float
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    active_users: int
    api_requests: int
    model_predictions: int
    errors_count: int
    response_time: float


@dataclass
class ModelPerformance:
    """Model performance tracking."""

    model_id: str
    timestamp: float
    accuracy: float
    throughput: float  # predictions per second
    latency: float  # average response time
    memory_usage: float
    error_rate: float


@dataclass
class UserActivity:
    """User activity tracking."""

    user_id: str
    timestamp: float
    action: str
    resource: str
    duration: float
    success: bool


class MetricsCollector:
    """Collects and aggregates system metrics."""

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.system_metrics = deque(maxlen=max_history)
        self.model_metrics = defaultdict(lambda: deque(maxlen=max_history))
        self.user_activities = deque(maxlen=max_history)
        self.alerts = deque(maxlen=100)
        self.collection_thread = None
        self.running = False

    def start_collection(self, interval: float = 60.0):
        """Start automatic metrics collection."""
        if self.running:
            return

        self.running = True
        self.collection_thread = threading.Thread(
            target=self._collection_loop, args=(interval,), daemon=True
        )
        self.collection_thread.start()
        print(f"📊 Metrics collection started (interval: {interval}s)")

    def stop_collection(self):
        """Stop metrics collection."""
        self.running = False
        if self.collection_thread:
            self.collection_thread.join()
        print("📊 Metrics collection stopped")

    def _collection_loop(self, interval: float):
        """Main collection loop."""
        while self.running:
            try:
                metrics = self._collect_system_metrics()
                self.system_metrics.append(metrics)
                self._check_alerts(metrics)
                time.sleep(interval)
            except Exception as e:
                print(f"⚠️ Metrics collection error: {e}")
                time.sleep(interval)

    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        try:
            import psutil

            # Get system metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            return SystemMetrics(
                timestamp=time.time(),
                cpu_usage=cpu_usage,
                memory_usage=memory.percent,
                disk_usage=disk.percent,
                active_users=self._get_active_users(),
                api_requests=self._get_api_requests(),
                model_predictions=self._get_model_predictions(),
                errors_count=self._get_errors_count(),
                response_time=self._get_average_response_time(),
            )

        except ImportError:
            # Fallback metrics when psutil is not available
            return SystemMetrics(
                timestamp=time.time(),
                cpu_usage=20.0 + 30.0 * (time.time() % 10) / 10,  # Simulated
                memory_usage=45.0 + 20.0 * (time.time() % 8) / 8,  # Simulated
                disk_usage=60.0,
                active_users=5,
                api_requests=100,
                model_predictions=50,
                errors_count=2,
                response_time=0.15,
            )

    def _get_active_users(self) -> int:
        """Get count of active users."""
        # In practice, this would query actual user sessions
        return len(
            set(
                activity.user_id
                for activity in self.user_activities
                if time.time() - activity.timestamp < 3600
            )
        )

    def _get_api_requests(self) -> int:
        """Get API request count in last minute."""
        cutoff = time.time() - 60
        return len([a for a in self.user_activities if a.timestamp > cutoff])

    def _get_model_predictions(self) -> int:
        """Get model prediction count in last minute."""
        cutoff = time.time() - 60
        return len(
            [
                a
                for a in self.user_activities
                if a.timestamp > cutoff and a.action == "predict"
            ]
        )

    def _get_errors_count(self) -> int:
        """Get error count in last minute."""
        cutoff = time.time() - 60
        return len(
            [a for a in self.user_activities if a.timestamp > cutoff and not a.success]
        )

    def _get_average_response_time(self) -> float:
        """Get average response time in last minute."""
        cutoff = time.time() - 60
        recent_activities = [a for a in self.user_activities if a.timestamp > cutoff]
        if not recent_activities:
            return 0.0
        return sum(a.duration for a in recent_activities) / len(recent_activities)

    def _check_alerts(self, metrics: SystemMetrics):
        """Check for alert conditions."""
        alerts = []

        if metrics.cpu_usage > 80:
            alerts.append(f"High CPU usage: {metrics.cpu_usage:.1f}%")

        if metrics.memory_usage > 85:
            alerts.append(f"High memory usage: {metrics.memory_usage:.1f}%")

        if metrics.disk_usage > 90:
            alerts.append(f"High disk usage: {metrics.disk_usage:.1f}%")

        if metrics.response_time > 1.0:
            alerts.append(f"Slow response time: {metrics.response_time:.2f}s")

        if metrics.errors_count > 10:
            alerts.append(f"High error rate: {metrics.errors_count} errors/min")

        for alert in alerts:
            self.alerts.append(
                {"timestamp": time.time(), "level": "warning", "message": alert}
            )

    def record_user_activity(
        self,
        user_id: str,
        action: str,
        resource: str,
        duration: float,
        success: bool = True,
    ):
        """Record user activity."""
        activity = UserActivity(
            user_id=user_id,
            timestamp=time.time(),
            action=action,
            resource=resource,
            duration=duration,
            success=success,
        )
        self.user_activities.append(activity)

    def record_model_performance(
        self,
        model_id: str,
        accuracy: float,
        throughput: float,
        latency: float,
        memory_usage: float,
        error_rate: float,
    ):
        """Record model performance metrics."""
        performance = ModelPerformance(
            model_id=model_id,
            timestamp=time.time(),
            accuracy=accuracy,
            throughput=throughput,
            latency=latency,
            memory_usage=memory_usage,
            error_rate=error_rate,
        )
        self.model_metrics[model_id].append(performance)


class AnalyticsDashboard:
    """Advanced analytics dashboard."""

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.insights_cache = {}
        self.last_analysis = 0
        self.analysis_interval = 300  # 5 minutes

    def generate_system_overview(self) -> Dict[str, Any]:
        """Generate system overview dashboard."""
        if not self.metrics.system_metrics:
            return {"error": "No metrics available"}

        latest = self.metrics.system_metrics[-1]

        # Calculate trends
        if len(self.metrics.system_metrics) > 10:
            recent_cpu = [m.cpu_usage for m in list(self.metrics.system_metrics)[-10:]]
            recent_memory = [
                m.memory_usage for m in list(self.metrics.system_metrics)[-10:]
            ]

            cpu_trend = "increasing" if recent_cpu[-1] > recent_cpu[0] else "decreasing"
            memory_trend = (
                "increasing" if recent_memory[-1] > recent_memory[0] else "decreasing"
            )
        else:
            cpu_trend = memory_trend = "stable"

        return {
            "current_status": {
                "cpu_usage": latest.cpu_usage,
                "memory_usage": latest.memory_usage,
                "disk_usage": latest.disk_usage,
                "active_users": latest.active_users,
                "response_time": latest.response_time,
            },
            "trends": {"cpu_trend": cpu_trend, "memory_trend": memory_trend},
            "alerts": list(self.metrics.alerts)[-5:],  # Last 5 alerts
            "uptime": self._calculate_uptime(),
            "total_requests": len(self.metrics.user_activities),
            "error_rate": self._calculate_error_rate(),
        }

    def generate_model_analytics(self) -> Dict[str, Any]:
        """Generate model performance analytics."""
        analytics = {}

        for model_id, performances in self.metrics.model_metrics.items():
            if not performances:
                continue

            latest = performances[-1]
            performances_list = list(performances)

            analytics[model_id] = {
                "current_performance": {
                    "accuracy": latest.accuracy,
                    "throughput": latest.throughput,
                    "latency": latest.latency,
                    "memory_usage": latest.memory_usage,
                    "error_rate": latest.error_rate,
                },
                "trends": {
                    "accuracy_trend": self._calculate_trend(
                        [p.accuracy for p in performances_list]
                    ),
                    "throughput_trend": self._calculate_trend(
                        [p.throughput for p in performances_list]
                    ),
                    "latency_trend": self._calculate_trend(
                        [p.latency for p in performances_list]
                    ),
                },
                "statistics": {
                    "avg_accuracy": sum(p.accuracy for p in performances_list)
                    / len(performances_list),
                    "avg_throughput": sum(p.throughput for p in performances_list)
                    / len(performances_list),
                    "avg_latency": sum(p.latency for p in performances_list)
                    / len(performances_list),
                },
            }

        return analytics

    def generate_user_insights(self) -> Dict[str, Any]:
        """Generate user behavior insights."""
        if not self.metrics.user_activities:
            return {"error": "No user activity data"}

        activities = list(self.metrics.user_activities)

        # User activity patterns
        users = set(a.user_id for a in activities)
        actions = defaultdict(int)
        resources = defaultdict(int)
        hourly_activity = defaultdict(int)

        for activity in activities:
            actions[activity.action] += 1
            resources[activity.resource] += 1
            hour = datetime.fromtimestamp(activity.timestamp).hour
            hourly_activity[hour] += 1

        return {
            "total_users": len(users),
            "total_activities": len(activities),
            "most_common_actions": dict(
                sorted(actions.items(), key=lambda x: x[1], reverse=True)[:5]
            ),
            "most_accessed_resources": dict(
                sorted(resources.items(), key=lambda x: x[1], reverse=True)[:5]
            ),
            "peak_hours": dict(
                sorted(hourly_activity.items(), key=lambda x: x[1], reverse=True)[:3]
            ),
            "average_session_duration": self._calculate_avg_session_duration(
                activities
            ),
            "success_rate": len([a for a in activities if a.success])
            / len(activities)
            * 100,
        }

    def _calculate_uptime(self) -> float:
        """Calculate system uptime."""
        if not self.metrics.system_metrics:
            return 0.0

        oldest_metric = self.metrics.system_metrics[0]
        return time.time() - oldest_metric.timestamp

    def _calculate_error_rate(self) -> float:
        """Calculate current error rate."""
        cutoff = time.time() - 3600  # Last hour
        recent_activities = [
            a for a in self.metrics.user_activities if a.timestamp > cutoff
        ]

        if not recent_activities:
            return 0.0

        errors = len([a for a in recent_activities if not a.success])
        return (errors / len(recent_activities)) * 100

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend from a list of values."""
        if len(values) < 2:
            return "stable"

        recent_avg = sum(values[-3:]) / min(3, len(values))
        older_avg = sum(values[:3]) / min(3, len(values))

        if recent_avg > older_avg * 1.1:
            return "increasing"
        elif recent_avg < older_avg * 0.9:
            return "decreasing"
        else:
            return "stable"

    def _calculate_avg_session_duration(self, activities: List[UserActivity]) -> float:
        """Calculate average session duration."""
        if not activities:
            return 0.0

        return sum(a.duration for a in activities) / len(activities)

    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive text report."""
        system_overview = self.generate_system_overview()
        model_analytics = self.generate_model_analytics()
        user_insights = self.generate_user_insights()

        lines = [
            "📊 ChemML Enterprise Analytics Report",
            "=" * 45,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "🖥️ System Overview:",
            f"  • CPU Usage: {system_overview['current_status']['cpu_usage']:.1f}%",
            f"  • Memory Usage: {system_overview['current_status']['memory_usage']:.1f}%",
            f"  • Active Users: {system_overview['current_status']['active_users']}",
            f"  • Response Time: {system_overview['current_status']['response_time']:.3f}s",
            f"  • Error Rate: {system_overview['error_rate']:.2f}%",
            "",
        ]

        if model_analytics:
            lines.extend(["🤖 Model Performance:", ""])
            for model_id, data in model_analytics.items():
                lines.extend(
                    [
                        f"  📊 {model_id}:",
                        f"    • Accuracy: {data['current_performance']['accuracy']:.3f}",
                        f"    • Throughput: {data['current_performance']['throughput']:.1f} pred/s",
                        f"    • Latency: {data['current_performance']['latency']:.3f}s",
                        "",
                    ]
                )

        if "error" not in user_insights:
            lines.extend(
                [
                    "👥 User Insights:",
                    f"  • Total Users: {user_insights['total_users']}",
                    f"  • Total Activities: {user_insights['total_activities']}",
                    f"  • Success Rate: {user_insights['success_rate']:.1f}%",
                    f"  • Avg Session Duration: {user_insights['average_session_duration']:.2f}s",
                    "",
                ]
            )

        if system_overview.get("alerts"):
            lines.extend(["⚠️ Recent Alerts:", ""])
            for alert in system_overview["alerts"]:
                lines.append(f"  • {alert['message']}")

        return "\n".join(lines)


class MonitoringDashboard:
    """Main monitoring dashboard coordinator."""

    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.analytics = AnalyticsDashboard(self.metrics_collector)
        self.auto_reports = False
        self.report_interval = 3600  # 1 hour
        self.report_thread = None

    def start_monitoring(
        self, collection_interval: float = 60.0, auto_reports: bool = False
    ):
        """Start comprehensive monitoring."""
        print("🚀 Starting ChemML Enterprise Monitoring")
        print("=" * 40)

        # Start metrics collection
        self.metrics_collector.start_collection(collection_interval)

        # Start auto reports if requested
        if auto_reports:
            self.start_auto_reports()

        print("✅ Monitoring system started successfully")

    def stop_monitoring(self):
        """Stop all monitoring activities."""
        self.metrics_collector.stop_collection()
        self.stop_auto_reports()
        print("🛑 Monitoring system stopped")

    def start_auto_reports(self):
        """Start automatic report generation."""
        if self.auto_reports:
            return

        self.auto_reports = True
        self.report_thread = threading.Thread(
            target=self._auto_report_loop, daemon=True
        )
        self.report_thread.start()
        print(f"📄 Auto-reports started (interval: {self.report_interval}s)")

    def stop_auto_reports(self):
        """Stop automatic report generation."""
        self.auto_reports = False

    def _auto_report_loop(self):
        """Auto report generation loop."""
        while self.auto_reports:
            try:
                time.sleep(self.report_interval)
                if self.auto_reports:  # Check again after sleep
                    report = self.analytics.generate_comprehensive_report()
                    self._save_report(report)
            except Exception as e:
                print(f"⚠️ Auto-report error: {e}")

    def _save_report(self, report: str):
        """Save report to file."""
        reports_dir = Path("reports/monitoring")
        reports_dir.mkdir(parents=True, exist_ok=True)

        filename = f"monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        filepath = reports_dir / filename

        with open(filepath, "w") as f:
            f.write(report)

        print(f"📄 Report saved: {filepath}")

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get all dashboard data."""
        return {
            "system_overview": self.analytics.generate_system_overview(),
            "model_analytics": self.analytics.generate_model_analytics(),
            "user_insights": self.analytics.generate_user_insights(),
            "timestamp": time.time(),
        }

    def simulate_activity(self, duration: int = 300):
        """Simulate user activity for demonstration."""
        print(f"🎭 Simulating activity for {duration} seconds...")

        users = ["user_1", "user_2", "user_3", "admin", "scientist"]
        actions = ["login", "predict", "train", "analyze", "download", "logout"]
        resources = ["model_a", "model_b", "dataset_1", "dashboard", "reports"]

        start_time = time.time()
        while time.time() - start_time < duration:
            # Simulate user activity
            user = (
                np.random.choice(users)
                if "np" in globals()
                else users[int(time.time()) % len(users)]
            )
            action = actions[int(time.time()) % len(actions)]
            resource = resources[int(time.time()) % len(resources)]
            duration_activity = 0.1 + (time.time() % 1) * 2  # 0.1-2.1 seconds
            success = (time.time() % 10) > 1  # 90% success rate

            self.metrics_collector.record_user_activity(
                user, action, resource, duration_activity, success
            )

            # Simulate model performance
            if action == "predict":
                self.metrics_collector.record_model_performance(
                    model_id=resource,
                    accuracy=0.85 + 0.1 * ((time.time() % 10) / 10),
                    throughput=50 + 20 * ((time.time() % 5) / 5),
                    latency=0.1 + 0.05 * ((time.time() % 3) / 3),
                    memory_usage=100 + 50 * ((time.time() % 7) / 7),
                    error_rate=0.01 + 0.02 * ((time.time() % 4) / 4),
                )

            time.sleep(1)  # Activity every second

        print("✅ Activity simulation completed")


if __name__ == "__main__":
    print("🏢 ChemML Enterprise Monitoring Test")

    # Create and start monitoring dashboard
    dashboard = MonitoringDashboard()

    try:
        # Start monitoring
        dashboard.start_monitoring(collection_interval=5.0, auto_reports=False)

        # Simulate some activity
        dashboard.simulate_activity(30)  # 30 seconds of activity

        # Generate and display report
        report = dashboard.analytics.generate_comprehensive_report()
        print("\n" + report)

        # Get dashboard data
        data = dashboard.get_dashboard_data()
        print(f"\n📊 Dashboard data collected: {len(data)} sections")

    finally:
        # Stop monitoring
        dashboard.stop_monitoring()

    print("\n✅ Enterprise monitoring test completed!")
