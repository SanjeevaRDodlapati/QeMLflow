#!/usr/bin/env python3
"""
GitHub Actions Progress Tracker
Monitor the success of our workflow fixes in real-time
"""

import json
import time
from datetime import datetime

from github_actions_monitor import GitHubActionsMonitor


def track_workflow_progress():
    """Track GitHub Actions progress after our fixes."""
    print("🔍 GITHUB ACTIONS PROGRESS TRACKER")
    print("=" * 60)
    print("Monitoring workflow success after typing import fixes...")
    print()

    monitor = GitHubActionsMonitor(".")

    # Track our commit
    target_commit = "2da00e47"

    for iteration in range(10):  # Monitor for up to 10 iterations
        print(f"🔄 Check #{iteration + 1} - {datetime.now().strftime('%H:%M:%S')}")

        runs = monitor.get_workflow_runs(15)
        our_commit_runs = [r for r in runs if r["head_sha"].startswith(target_commit)]

        if our_commit_runs:
            print(f"📊 Workflows for our commit ({target_commit}):")

            success_count = 0
            failure_count = 0
            in_progress_count = 0

            for run in our_commit_runs:
                status = run.get("conclusion") or "in_progress"
                emoji = (
                    "✅" if status == "success" else "❌" if status == "failure" else "🔄"
                )

                if status == "success":
                    success_count += 1
                elif status == "failure":
                    failure_count += 1
                else:
                    in_progress_count += 1

                print(f"   {emoji} {run['name']:<30} | {status}")

            print(f"\n📈 SUMMARY:")
            print(f"   ✅ Success: {success_count}")
            print(f"   ❌ Failed:  {failure_count}")
            print(f"   🔄 Running: {in_progress_count}")

            success_rate = (
                success_count / (success_count + failure_count) * 100
                if (success_count + failure_count) > 0
                else 0
            )
            print(f"   📊 Success Rate: {success_rate:.1f}%")

            if in_progress_count == 0:
                print(f"\n🎯 ALL WORKFLOWS COMPLETED!")
                if success_rate >= 80:
                    print("🎉 EXCELLENT! High success rate achieved!")
                elif success_rate >= 60:
                    print("👍 GOOD! Significant improvement!")
                else:
                    print("⚠️  Still needs work, but progress made!")
                break

        print("-" * 60)

        if iteration < 9:  # Don't sleep on last iteration
            time.sleep(30)  # Wait 30 seconds between checks

    return monitor


if __name__ == "__main__":
    track_workflow_progress()
