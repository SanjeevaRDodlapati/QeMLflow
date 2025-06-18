#!/usr/bin/env python3
"""
Continuous GitHub Actions Monitor
================================

Continuously monitors the current CI/CD pipeline to see if our fixes are working.
"""

import sys
import time
from datetime import datetime

from github_actions_monitor import GitHubActionsMonitor


def monitor_current_pipeline():
    """Monitor the current pipeline progress."""
    monitor = GitHubActionsMonitor(".")

    print("🔄 Starting continuous monitoring of GitHub Actions...")
    print(f"📁 Repository: {monitor.repo_owner}/{monitor.repo_name}")
    print("=" * 60)

    previous_status = {}
    check_count = 0

    while check_count < 20:  # Monitor for up to 20 checks (10 minutes)
        try:
            check_count += 1
            runs = monitor.get_workflow_runs(5)

            print(f"\n🔍 Check #{check_count} at {datetime.now().strftime('%H:%M:%S')}")
            print("-" * 40)

            status_changed = False
            for run in runs[:3]:  # Check top 3 runs
                run_id = run["id"]
                current_status = run.get("conclusion") or run.get("status")
                name = run["name"]

                if run_id not in previous_status:
                    print(f"🆕 NEW: {name} - {current_status}")
                    status_changed = True
                elif previous_status[run_id] != current_status:
                    print(f"🔄 CHANGED: {name}")
                    print(f"   {previous_status[run_id]} → {current_status}")
                    status_changed = True
                else:
                    print(f"⏸️  SAME: {name} - {current_status}")

                previous_status[run_id] = current_status

                # Check if this is our target pipeline
                if "CI/CD" in name and current_status in ["success", "failure"]:
                    print(f"\n🎯 TARGET PIPELINE COMPLETED!")
                    print(f"   Name: {name}")
                    print(f"   Result: {current_status}")
                    print(f"   SHA: {run['head_sha'][:8]}")

                    if current_status == "success":
                        print("🎉 SUCCESS! Our fixes are working!")
                        return True
                    else:
                        print("❌ Still failing. Need further investigation.")
                        return False

            if not status_changed:
                print("   (No changes detected)")

            if check_count < 20:
                print(f"\n⏳ Waiting 30 seconds before next check... ({check_count}/20)")
                time.sleep(30)

        except KeyboardInterrupt:
            print("\n\n⏹️  Monitoring stopped by user")
            break
        except Exception as e:
            print(f"\n❌ Error during monitoring: {e}")
            time.sleep(10)

    print(f"\n📊 Monitoring completed after {check_count} checks")
    return None


if __name__ == "__main__":
    result = monitor_current_pipeline()

    if result is True:
        print("\n✅ Mission accomplished! GitHub Actions are working!")
        sys.exit(0)
    elif result is False:
        print("\n❌ Issues still remain. Further fixes needed.")
        sys.exit(1)
    else:
        print("\n⏱️  Monitoring timed out. Check manually.")
        sys.exit(2)
