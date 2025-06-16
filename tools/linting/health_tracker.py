#!/usr/bin/env python3
"""
Code Health Tracking Dashboard for ChemML
=========================================

This script creates a health tracking dashboard that monitors code quality
metrics over time and generates reports for continuous improvement.

Features:
1. Track health scores over time
2. Generate trend analysis
3. Create visual dashboards (if matplotlib available)
4. Monitor pre-commit compliance
5. Track technical debt accumulation

Usage:
    python tools/linting/health_tracker.py [--update] [--dashboard] [--report]
"""

import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

try:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

@dataclass
class HealthSnapshot:
    """A snapshot of code health at a specific time."""

    timestamp: str
    health_score: float
    total_issues: int
    issues_by_severity: Dict[str, int]
    issues_by_tool: Dict[str, int]
    files_checked: int
    auto_fixable_count: int

    # Additional metrics
    complexity_score: Optional[float] = None
    security_score: Optional[float] = None
    coverage_percentage: Optional[float] = None
    technical_debt_minutes: Optional[int] = None

@dataclass
class HealthTrend:
    """Analysis of health trends over time."""

    period_days: int
    score_change: float
    issue_count_change: int
    trend_direction: str  # 'improving', 'declining', 'stable'
    key_improvements: List[str]
    key_concerns: List[str]

class HealthTracker:
    """Code health tracking and dashboard generation."""

    def __init__(self, root_path: Optional[Path] = None):
        self.root = root_path or Path(__file__).parent.parent.parent
        self.health_dir = self.root / "reports" / "health"
        self.health_dir.mkdir(parents=True, exist_ok=True)

        self.history_file = self.health_dir / "health_history.json"
        self.dashboard_dir = self.health_dir / "dashboards"
        self.dashboard_dir.mkdir(parents=True, exist_ok=True)

    def capture_current_health(self) -> HealthSnapshot:
        """Capture current code health snapshot."""
        print("📊 Capturing current health snapshot...")

        # Run comprehensive linter
        try:
            linter_path = self.root / "tools" / "linting" / "comprehensive_linter.py"
            result = subprocess.run(
                ["python", str(linter_path), "--format", "json", "--quiet"],
                capture_output=True,
                text=True,
                cwd=self.root,
            )

            if result.returncode == 0 and result.stdout:
                # Parse clean JSON output
                data = json.loads(result.stdout.strip())
                summary = data.get("summary", {})

                snapshot = HealthSnapshot(
                    timestamp=datetime.now().isoformat(),
                    health_score=summary.get("health_score", 0.0),
                    total_issues=summary.get("total_issues", 0),
                    issues_by_severity=data.get("issues_by_severity", {}),
                    issues_by_tool=data.get("issues_by_tool", {}),
                    files_checked=summary.get("total_files_checked", 0),
                    auto_fixable_count=summary.get("auto_fixable_count", 0),
                )

                # Try to get additional metrics
                snapshot.complexity_score = self._calculate_complexity_score()
                snapshot.security_score = self._calculate_security_score()
                snapshot.coverage_percentage = self._get_test_coverage()
                snapshot.technical_debt_minutes = self._estimate_technical_debt(
                    snapshot
                )

                return snapshot

        except Exception as e:
            print(f"Error capturing health snapshot: {e}")

        # Fallback snapshot if linter fails
        return HealthSnapshot(
            timestamp=datetime.now().isoformat(),
            health_score=0.0,
            total_issues=0,
            issues_by_severity={},
            issues_by_tool={},
            files_checked=0,
            auto_fixable_count=0,
        )

    def _calculate_complexity_score(self) -> Optional[float]:
        """Calculate complexity score from code quality enhancer."""
        try:
            enhancer_path = self.root / "tools" / "linting" / "code_quality_enhancer.py"
            result = subprocess.run(
                ["python", str(enhancer_path), "--complexity"],
                capture_output=True,
                text=True,
                cwd=self.root,
            )

            if result.returncode == 0:
                # Parse output to estimate complexity score
                output = result.stdout
                if "complexity issues" in output:
                    import re

                    match = re.search(r"Found (\d+) complexity issues", output)
                    if match:
                        issues = int(match.group(1))
                        # Simple scoring: fewer issues = higher score
                        return max(0, 100 - (issues * 2))

        except Exception:
            pass

        return None

    def _calculate_security_score(self) -> Optional[float]:
        """Calculate security score."""
        try:
            enhancer_path = self.root / "tools" / "linting" / "code_quality_enhancer.py"
            result = subprocess.run(
                ["python", str(enhancer_path), "--security"],
                capture_output=True,
                text=True,
                cwd=self.root,
            )

            if result.returncode == 0:
                output = result.stdout
                if "security issues" in output:
                    import re

                    match = re.search(r"Found (\d+) security issues", output)
                    if match:
                        issues = int(match.group(1))
                        # Simple scoring: no issues = 100, each issue reduces score
                        return max(0, 100 - (issues * 5))

        except Exception:
            pass

        return None

    def _get_test_coverage(self) -> Optional[float]:
        """Get test coverage percentage."""
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "--cov=src", "--cov-report=term-missing"],
                capture_output=True,
                text=True,
                cwd=self.root,
            )

            if result.stdout:
                import re

                # Look for coverage percentage in output
                match = re.search(r"TOTAL.*?(\d+)%", result.stdout)
                if match:
                    return float(match.group(1))

        except Exception:
            pass

        return None

    def _estimate_technical_debt(self, snapshot: HealthSnapshot) -> Optional[int]:
        """Estimate technical debt in minutes to fix."""
        # Simple estimation based on issue counts and severity
        debt_minutes = 0

        # Time estimates per issue type (in minutes)
        time_per_issue = {
            "error": 15,  # 15 minutes per error
            "warning": 5,  # 5 minutes per warning
            "info": 2,  # 2 minutes per info
        }

        for severity, count in snapshot.issues_by_severity.items():
            debt_minutes += count * time_per_issue.get(severity, 3)

        return debt_minutes if debt_minutes > 0 else None

    def save_snapshot(self, snapshot: HealthSnapshot):
        """Save health snapshot to history."""
        history = self.load_history()
        history.append(asdict(snapshot))

        # Keep only last 365 days of history
        cutoff_date = datetime.now() - timedelta(days=365)
        history = [
            h for h in history if datetime.fromisoformat(h["timestamp"]) > cutoff_date
        ]

        with open(self.history_file, "w") as f:
            json.dump(history, f, indent=2)

        print(f"💾 Health snapshot saved ({len(history)} total records)")

    def load_history(self) -> List[Dict]:
        """Load health history from file."""
        if self.history_file.exists():
            try:
                with open(self.history_file) as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading health history: {e}")

        return []

    def analyze_trends(self, days: int = 30) -> HealthTrend:
        """Analyze health trends over specified period."""
        history = self.load_history()

        if len(history) < 2:
            return HealthTrend(
                period_days=days,
                score_change=0.0,
                issue_count_change=0,
                trend_direction="stable",
                key_improvements=[],
                key_concerns=[],
            )

        # Filter to period
        cutoff_date = datetime.now() - timedelta(days=days)
        period_history = [
            h for h in history if datetime.fromisoformat(h["timestamp"]) > cutoff_date
        ]

        if len(period_history) < 2:
            period_history = history[-2:]  # Use last 2 records

        # Calculate changes
        first_snapshot = period_history[0]
        last_snapshot = period_history[-1]

        score_change = last_snapshot["health_score"] - first_snapshot["health_score"]
        issue_change = last_snapshot["total_issues"] - first_snapshot["total_issues"]

        # Determine trend direction
        if score_change > 2:
            direction = "improving"
        elif score_change < -2:
            direction = "declining"
        else:
            direction = "stable"

        # Identify key changes
        improvements = []
        concerns = []

        if score_change > 0:
            improvements.append(f"Health score improved by {score_change:.1f} points")
        elif score_change < 0:
            concerns.append(f"Health score declined by {abs(score_change):.1f} points")

        if issue_change < 0:
            improvements.append(f"Reduced issues by {abs(issue_change)}")
        elif issue_change > 0:
            concerns.append(f"Increased issues by {issue_change}")

        return HealthTrend(
            period_days=len(period_history),
            score_change=score_change,
            issue_count_change=issue_change,
            trend_direction=direction,
            key_improvements=improvements,
            key_concerns=concerns,
        )

    def generate_dashboard(self) -> Optional[Path]:
        """Generate visual dashboard if matplotlib is available."""
        if not HAS_MATPLOTLIB:
            print("📊 Matplotlib not available - skipping visual dashboard")
            return None

        history = self.load_history()
        if len(history) < 2:
            print("📊 Insufficient data for dashboard (need at least 2 snapshots)")
            return None

        # Prepare data
        timestamps = [datetime.fromisoformat(h["timestamp"]) for h in history]
        health_scores = [h["health_score"] for h in history]
        total_issues = [h["total_issues"] for h in history]

        # Create dashboard
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("ChemML Code Health Dashboard", fontsize=16, fontweight="bold")

        # Health score over time
        ax1.plot(timestamps, health_scores, marker="o", linewidth=2, markersize=4)
        ax1.set_title("Health Score Over Time")
        ax1.set_ylabel("Health Score")
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))

        # Issue count over time
        ax2.plot(
            timestamps, total_issues, marker="s", color="red", linewidth=2, markersize=4
        )
        ax2.set_title("Total Issues Over Time")
        ax2.set_ylabel("Issue Count")
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))

        # Latest issue breakdown by severity
        if history:
            latest = history[-1]
            severities = list(latest["issues_by_severity"].keys())
            counts = list(latest["issues_by_severity"].values())

            if severities and counts:
                colors = {"error": "red", "warning": "orange", "info": "yellow"}
                pie_colors = [colors.get(s, "gray") for s in severities]

                ax3.pie(counts, labels=severities, autopct="%1.1f%%", colors=pie_colors)
                ax3.set_title("Current Issues by Severity")

        # Latest issue breakdown by tool
        if history:
            latest = history[-1]
            tools = list(latest["issues_by_tool"].keys())
            tool_counts = list(latest["issues_by_tool"].values())

            if tools and tool_counts:
                ax4.bar(
                    tools,
                    tool_counts,
                    color=["blue", "green", "purple", "brown"][: len(tools)],
                )
                ax4.set_title("Current Issues by Tool")
                ax4.set_ylabel("Issue Count")
                ax4.tick_params(axis="x", rotation=45)

        # Adjust layout and save
        plt.tight_layout()

        dashboard_file = (
            self.dashboard_dir
            / f"health_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
        plt.savefig(dashboard_file, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"📊 Dashboard saved to {dashboard_file}")
        return dashboard_file

    def generate_text_report(self) -> str:
        """Generate comprehensive text report."""
        current_snapshot = self.capture_current_health()
        trend_7d = self.analyze_trends(7)
        trend_30d = self.analyze_trends(30)

        report = []
        report.append("=" * 80)
        report.append("🏥 ChemML Code Health Report")
        report.append("=" * 80)
        report.append(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Current Status
        report.append("📊 Current Health Status:")
        report.append(
            f"  • Overall Health Score: {current_snapshot.health_score:.1f}/100"
        )
        report.append(f"  • Total Issues: {current_snapshot.total_issues}")
        report.append(f"  • Files Checked: {current_snapshot.files_checked}")
        report.append(f"  • Auto-fixable Issues: {current_snapshot.auto_fixable_count}")

        if current_snapshot.complexity_score:
            report.append(
                f"  • Complexity Score: {current_snapshot.complexity_score:.1f}/100"
            )
        if current_snapshot.security_score:
            report.append(
                f"  • Security Score: {current_snapshot.security_score:.1f}/100"
            )
        if current_snapshot.coverage_percentage:
            report.append(
                f"  • Test Coverage: {current_snapshot.coverage_percentage:.1f}%"
            )
        if current_snapshot.technical_debt_minutes:
            hours = current_snapshot.technical_debt_minutes // 60
            minutes = current_snapshot.technical_debt_minutes % 60
            report.append(f"  • Technical Debt: {hours}h {minutes}m")

        report.append("")

        # 7-day trend
        report.append("📈 7-Day Trend:")
        report.append(f"  • Direction: {trend_7d.trend_direction.title()}")
        report.append(f"  • Score Change: {trend_7d.score_change:+.1f}")
        report.append(f"  • Issue Count Change: {trend_7d.issue_count_change:+d}")

        if trend_7d.key_improvements:
            report.append("  • Improvements:")
            for improvement in trend_7d.key_improvements:
                report.append(f"    - {improvement}")

        if trend_7d.key_concerns:
            report.append("  • Concerns:")
            for concern in trend_7d.key_concerns:
                report.append(f"    - {concern}")

        report.append("")

        # 30-day trend
        report.append("📈 30-Day Trend:")
        report.append(f"  • Direction: {trend_30d.trend_direction.title()}")
        report.append(f"  • Score Change: {trend_30d.score_change:+.1f}")
        report.append(f"  • Issue Count Change: {trend_30d.issue_count_change:+d}")

        report.append("")

        # Recommendations
        report.append("💡 Recommendations:")

        if current_snapshot.health_score < 70:
            report.append("  • 🚨 Health score below 70 - immediate attention needed")

        if current_snapshot.auto_fixable_count > 0:
            report.append(
                f"  • 🔧 Run auto-fix to resolve {current_snapshot.auto_fixable_count} fixable issues"
            )

        if trend_7d.trend_direction == "declining":
            report.append(
                "  • 📉 Recent decline in code quality - investigate recent changes"
            )

        if (
            current_snapshot.technical_debt_minutes
            and current_snapshot.technical_debt_minutes > 120
        ):
            report.append(
                "  • ⏰ High technical debt - consider dedicating time to issue resolution"
            )

        if (
            current_snapshot.coverage_percentage
            and current_snapshot.coverage_percentage < 80
        ):
            report.append("  • 🧪 Test coverage below 80% - add more tests")

        report.append("")
        report.append("=" * 80)

        return "\n".join(report)

    def update_and_report(self):
        """Update health snapshot and generate report."""
        # Capture current health
        snapshot = self.capture_current_health()
        self.save_snapshot(snapshot)

        # Generate report
        report = self.generate_text_report()
        print(report)

        # Save report to file
        report_file = (
            self.health_dir
            / f"health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        with open(report_file, "w") as f:
            f.write(report)

        print(f"📄 Full report saved to {report_file}")

        # Generate dashboard if possible
        dashboard_file = self.generate_dashboard()

        return snapshot, report_file, dashboard_file

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Code health tracking and dashboard for ChemML"
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Update health snapshot and generate report",
    )
    parser.add_argument(
        "--dashboard", action="store_true", help="Generate visual dashboard"
    )
    parser.add_argument(
        "--report", action="store_true", help="Generate text report only"
    )

    args = parser.parse_args()

    if not any([args.update, args.dashboard, args.report]):
        args.update = True  # Default action

    tracker = HealthTracker()

    if args.update:
        tracker.update_and_report()
    elif args.report:
        report = tracker.generate_text_report()
        print(report)
    elif args.dashboard:
        tracker.generate_dashboard()

if __name__ == "__main__":
    main()
