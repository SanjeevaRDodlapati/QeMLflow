#!/usr/bin/env python3
"""
QeMLflow Status Dashboard
======================

Provides a comprehensive dashboard view of the QeMLflow project status,
including workflows, documentation, and system health.
"""

import json
import sys
from datetime import datetime
from pathlib import Path


def load_latest_monitoring_results():
    """Load the latest monitoring results from logs."""
    logs_dir = Path("logs")
    if not logs_dir.exists():
        return None

    monitoring_files = list(logs_dir.glob("monitoring_*.json"))
    if not monitoring_files:
        return None

    # Get the most recent monitoring file
    latest_file = max(monitoring_files, key=lambda p: p.stat().st_mtime)

    try:
        with open(latest_file, "r") as f:
            return json.load(f)
    except Exception:
        return None


def print_dashboard_header():
    """Print the dashboard header."""
    print("🔍 QeMLflow System Status Dashboard")
    print("=" * 60)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()


def print_quick_status():
    """Print quick status checks."""
    print("⚡ Quick Status Checks")
    print("-" * 30)

    # Check if key files exist
    checks = [
        ("GitHub Workflows", Path(".github/workflows").exists()),
        ("Documentation Config", Path(".config/mkdocs.yml").exists()),
        ("Source Code", Path("src").exists()),
        ("Documentation", Path("docs").exists()),
        ("Tests", Path("tests").exists()),
        ("Requirements", Path("requirements.txt").exists()),
        ("Monitoring Script", Path("scripts/monitoring/automated_monitor.py").exists()),
    ]

    for name, status in checks:
        emoji = "✅" if status else "❌"
        print(f"   {emoji} {name}")

    print()


def print_monitoring_results(results):
    """Print detailed monitoring results."""
    if not results:
        print("📊 Detailed Monitoring Results")
        print("-" * 30)
        print("   ⚠️ No monitoring results available")
        print("   💡 Run: python scripts/monitoring/automated_monitor.py")
        print()
        return

    print("📊 Detailed Monitoring Results")
    print("-" * 30)
    print(f"   Last Update: {results.get('timestamp', 'Unknown')}")
    print(
        f"   Overall Status: {get_status_emoji(results.get('overall_status', 'unknown'))} {results.get('overall_status', 'Unknown').upper()}"
    )
    print()

    # Component status
    components = [
        ("GitHub Actions", results.get("workflows", {}).get("overall", "unknown")),
        (
            "Documentation Site",
            results.get("documentation", {}).get("overall", "unknown"),
        ),
        ("Releases", results.get("releases", {}).get("overall", "unknown")),
        ("Repository Health", results.get("repository", {}).get("overall", "unknown")),
    ]

    for name, status in components:
        emoji = get_status_emoji(status)
        print(f"   {emoji} {name}: {status.upper()}")

    # Show recent issues
    issues = results.get("issues", [])
    if issues:
        print()
        print("   ⚠️ Recent Issues:")
        for issue in issues[-3:]:  # Show last 3 issues
            severity_emoji = "🔴" if issue["severity"] == "error" else "🟡"
            print(f"      {severity_emoji} [{issue['component']}] {issue['message']}")

    print()


def print_workflow_status(results):
    """Print detailed workflow status."""
    print("⚙️ GitHub Actions Status")
    print("-" * 30)

    workflows = results.get("workflows", {}) if results else {}

    if not workflows or workflows.get("overall") == "unknown":
        print("   ❓ Workflow status unknown")
        print("   💡 Check: https://github.com/SanjeevaRDodlapati/QeMLflow/actions")
    else:
        recent_runs = workflows.get("recent_runs", [])
        if recent_runs:
            print("   Recent Workflow Runs:")
            for run in recent_runs[:5]:  # Show last 5 runs
                status_emoji = get_status_emoji(run.get("status", "unknown"))
                print(
                    f"      {status_emoji} {run.get('name', 'Unknown')} - {run.get('status', 'unknown')}"
                )

        workflow_list = workflows.get("workflows", [])
        if workflow_list:
            print()
            print("   Available Workflows:")
            for workflow in workflow_list:
                state_emoji = "✅" if workflow.get("state") == "active" else "⚠️"
                print(f"      {state_emoji} {workflow.get('name', 'Unknown')}")

    print()


def print_documentation_status(results):
    """Print documentation status."""
    print("📚 Documentation Status")
    print("-" * 30)

    docs = results.get("documentation", {}) if results else {}

    if not docs or docs.get("overall") == "unknown":
        print("   ❓ Documentation status unknown")
    else:
        print(f"   URL: {docs.get('url', 'Unknown')}")
        print(f"   Status: HTTP {docs.get('status_code', 'Unknown')}")
        print(f"   Response Time: {docs.get('response_time', 'Unknown')}s")

        content_checks = docs.get("content_checks", {})
        if content_checks:
            print("   Content Checks:")
            for check, passed in content_checks.items():
                emoji = "✅" if passed else "❌"
                print(f"      {emoji} {check.replace('_', ' ').title()}")

    print()


def print_action_items():
    """Print recommended action items."""
    print("🎯 Recommended Actions")
    print("-" * 30)

    actions = [
        "Run monitoring: python scripts/monitoring/automated_monitor.py",
        "Check workflows: Visit GitHub Actions page",
        "Test locally: mkdocs serve (for documentation)",
        "Quick check: python scripts/development/quick_status_check.py",
    ]

    for i, action in enumerate(actions, 1):
        print(f"   {i}. {action}")

    print()


def print_useful_links():
    """Print useful links."""
    print("🔗 Useful Links")
    print("-" * 30)
    print("   • GitHub Repository: https://github.com/SanjeevaRDodlapati/QeMLflow")
    print("   • GitHub Actions: https://github.com/SanjeevaRDodlapati/QeMLflow/actions")
    print("   • Documentation Site: https://sanjeevardodlapati.github.io/QeMLflow/")
    print("   • Releases: https://github.com/SanjeevaRDodlapati/QeMLflow/releases")
    print()


def get_status_emoji(status):
    """Get emoji for status."""
    emoji_map = {
        "success": "✅",
        "warning": "⚠️",
        "error": "❌",
        "failure": "❌",
        "unknown": "❓",
    }
    return emoji_map.get(status.lower(), "❓")


def main():
    """Generate the status dashboard."""
    print_dashboard_header()

    # Load monitoring results
    results = load_latest_monitoring_results()

    # Print sections
    print_quick_status()
    print_monitoring_results(results)
    print_workflow_status(results)
    print_documentation_status(results)
    print_action_items()
    print_useful_links()

    # Exit with appropriate code based on status
    if results:
        overall_status = results.get("overall_status", "unknown")
        if overall_status == "success":
            print("🎉 All systems operational!")
            sys.exit(0)
        elif overall_status == "warning":
            print("⚠️ Some issues detected")
            sys.exit(1)
        else:
            print("🚨 Critical issues detected")
            sys.exit(2)
    else:
        print("❓ Status unknown - run monitoring system")
        sys.exit(1)


if __name__ == "__main__":
    main()
