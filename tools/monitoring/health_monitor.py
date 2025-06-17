#!/usr/bin/env python3
"""
QeMLflow Health Monitoring and Alerting System
============================================

Monitors codebase health metrics and provides actionable insights.
Can be integrated with CI/CD pipelines for automated quality gates.
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path


class HealthMonitor:
    """Monitor and analyze QeMLflow codebase health metrics."""

    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.reports_dir = self.base_dir / "reports" / "health"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def get_current_metrics(self):
        """Get current health metrics from health tracker."""
        import subprocess

        try:
            result = subprocess.run(
                [
                    "python",
                    str(self.base_dir / "tools/linting/health_tracker.py"),
                    "--report",
                ],
                capture_output=True,
                text=True,
                cwd=self.base_dir,
            )

            if result.returncode != 0:
                raise Exception(f"Health tracker failed: {result.stderr}")

            # Parse health score from output
            output = result.stdout
            health_score = 0
            total_issues = 0
            test_coverage = 0

            for line in output.split("\n"):
                if "Health Score:" in line:
                    health_score = float(line.split(":")[1].split("/")[0].strip())
                elif "Total Issues:" in line:
                    total_issues = int(line.split(":")[1].strip())
                elif "Test Coverage:" in line:
                    test_coverage = float(line.split(":")[1].replace("%", "").strip())

            return {
                "timestamp": datetime.now().isoformat(),
                "health_score": health_score,
                "total_issues": total_issues,
                "test_coverage": test_coverage,
                "raw_output": output,
            }

        except Exception as e:
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "health_score": 0,
                "total_issues": 9999,
                "test_coverage": 0,
            }

    def analyze_trends(self, current_metrics):
        """Analyze health trends over time."""
        history_file = self.reports_dir / "health_history.json"

        # Load historical data
        history = []
        if history_file.exists():
            with open(history_file, "r") as f:
                history = json.load(f)

        # Add current metrics
        history.append(current_metrics)

        # Keep only last 30 days
        cutoff_date = datetime.now() - timedelta(days=30)
        history = [
            h for h in history if datetime.fromisoformat(h["timestamp"]) > cutoff_date
        ]

        # Save updated history
        with open(history_file, "w") as f:
            json.dump(history, f, indent=2)

        return self._calculate_trends(history)

    def _calculate_trends(self, history):
        """Calculate trend analysis from historical data."""
        if len(history) < 2:
            return {"trend": "insufficient_data", "change": 0}

        recent = history[-3:]  # Last 3 measurements
        health_scores = [h.get("health_score", 0) for h in recent]

        if len(health_scores) >= 2:
            change = health_scores[-1] - health_scores[0]
            if change > 2:
                trend = "improving"
            elif change < -2:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "stable"
            change = 0

        return {"trend": trend, "change": change, "history_count": len(history)}

    def generate_alerts(self, metrics, trends):
        """Generate alerts based on metrics and trends."""
        alerts = []

        # Critical health score
        if metrics["health_score"] < 50:
            alerts.append(
                {
                    "level": "critical",
                    "message": f"Health score critically low: {metrics['health_score']}/100",
                    "action": "Immediate action required - run auto-fix tools",
                }
            )
        elif metrics["health_score"] < 70:
            alerts.append(
                {
                    "level": "warning",
                    "message": f"Health score below target: {metrics['health_score']}/100",
                    "action": "Schedule maintenance cycle to address issues",
                }
            )

        # High issue count
        if metrics["total_issues"] > 1500:
            alerts.append(
                {
                    "level": "warning",
                    "message": f"High issue count: {metrics['total_issues']}",
                    "action": "Run comprehensive linting and auto-fix",
                }
            )

        # Low test coverage
        if metrics["test_coverage"] < 60:
            alerts.append(
                {
                    "level": "warning",
                    "message": f"Test coverage below target: {metrics['test_coverage']}%",
                    "action": "Add tests for uncovered modules",
                }
            )

        # Declining trend
        if trends["trend"] == "declining":
            alerts.append(
                {
                    "level": "warning",
                    "message": f"Health declining: {trends['change']:.1f} point drop",
                    "action": "Investigate recent changes causing quality decline",
                }
            )

        return alerts

    def generate_recommendations(self, metrics, trends, alerts):
        """Generate actionable recommendations."""
        recommendations = []

        # Based on health score
        if metrics["health_score"] < 70:
            recommendations.extend(
                [
                    "Run auto-fix: python tools/linting/comprehensive_linter.py --auto-fix",
                    "Address critical issues: python tools/linting/critical_fixes.py",
                    "Review and fix import organization issues",
                ]
            )

        # Based on trends
        if trends["trend"] == "declining":
            recommendations.extend(
                [
                    "Investigate recent commits for quality regressions",
                    "Run incremental refactor: python tools/linting/incremental_refactor.py",
                    "Schedule team code review session",
                ]
            )

        # Based on test coverage
        if metrics["test_coverage"] < 70:
            recommendations.extend(
                [
                    "Add unit tests for uncovered modules",
                    "Review test collection issues",
                    "Run: pytest --cov-report=html to identify gaps",
                ]
            )

        # General maintenance
        if metrics["total_issues"] > 1000:
            recommendations.extend(
                [
                    "Schedule weekly maintenance cycle",
                    "Consider pair programming for complex refactoring",
                    "Update linting configuration for stricter standards",
                ]
            )

        return recommendations

    def create_report(self, output_format="console"):
        """Create comprehensive health report."""
        print("🏥 QeMLflow Health Monitor")
        print("=" * 50)

        # Get current metrics
        print("📊 Collecting current metrics...")
        metrics = self.get_current_metrics()

        if "error" in metrics:
            print(f"❌ Error collecting metrics: {metrics['error']}")
            return False

        # Analyze trends
        print("📈 Analyzing trends...")
        trends = self.analyze_trends(metrics)

        # Generate alerts and recommendations
        alerts = self.generate_alerts(metrics, trends)
        recommendations = self.generate_recommendations(metrics, trends, alerts)

        # Display results
        print(f"\n📊 Current Status ({metrics['timestamp'][:10]}):")
        print(f"   Health Score: {metrics['health_score']}/100")
        print(f"   Total Issues: {metrics['total_issues']}")
        print(f"   Test Coverage: {metrics['test_coverage']}%")

        print(f"\n📈 Trend Analysis:")
        print(f"   Direction: {trends['trend']}")
        print(f"   Change: {trends['change']:+.1f} points")
        print(f"   History: {trends['history_count']} measurements")

        if alerts:
            print(f"\n🚨 Alerts ({len(alerts)}):")
            for alert in alerts:
                icon = "🔴" if alert["level"] == "critical" else "⚠️"
                print(f"   {icon} {alert['message']}")
                print(f"      → {alert['action']}")
        else:
            print("\n✅ No alerts - system healthy")

        if recommendations:
            print(f"\n💡 Recommendations ({len(recommendations)}):")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")

        # Save detailed report
        report_data = {
            "metrics": metrics,
            "trends": trends,
            "alerts": alerts,
            "recommendations": recommendations,
            "generated_at": datetime.now().isoformat(),
        }

        report_file = (
            self.reports_dir
            / f"health_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_file, "w") as f:
            json.dump(report_data, f, indent=2)

        print(f"\n📄 Detailed report saved: {report_file}")

        # Return success status
        critical_alerts = [a for a in alerts if a["level"] == "critical"]
        return len(critical_alerts) == 0


def main():
    parser = argparse.ArgumentParser(description="QeMLflow Health Monitoring")
    parser.add_argument(
        "--format", choices=["console", "json"], default="console", help="Output format"
    )
    parser.add_argument(
        "--exit-on-critical",
        action="store_true",
        help="Exit with error code if critical issues found",
    )

    args = parser.parse_args()

    monitor = HealthMonitor()
    success = monitor.create_report(args.format)

    if args.exit_on_critical and not success:
        print("\n🔴 Critical issues detected - exiting with error code")
        sys.exit(1)

    print("\n🎉 Health monitoring completed")


if __name__ == "__main__":
    main()
