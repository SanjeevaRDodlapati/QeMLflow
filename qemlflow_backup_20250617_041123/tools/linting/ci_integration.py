"""
CI/CD Integration Script for ChemML Linting Framework
===================================================

This script provides CI/CD integration for the linting framework,
including automated quality checks and reporting.

Features:
1. Run comprehensive linting checks
2. Enforce quality gates
3. Generate CI-friendly reports
4. Auto-fix in CI when safe
5. Track quality metrics over time

Usage in CI:
    python tools/linting/ci_integration.py --check --fail-on-errors
    python tools/linting/ci_integration.py --auto-fix --dry-run
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class CILintingIntegration:
    """CI/CD integration for linting framework."""

    def __init__(self, root_path: Optional[Path] = None):
        self.root = root_path or Path(__file__).parent.parent.parent
        self.tools_dir = self.root / "tools" / "linting"
        self.reports_dir = self.root / "reports" / "linting"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Quality gates - configurable thresholds
        self.quality_gates = {
            "min_health_score": 75.0,
            "max_critical_issues": 0,
            "max_error_issues": 10,
            "max_total_issues": 100,
        }

    def run_comprehensive_check(self) -> Tuple[bool, Dict]:
        """Run comprehensive linting check and return pass/fail status."""
        print("🔍 Running comprehensive linting check for CI...")

        try:
            # Run comprehensive linter
            linter_script = self.tools_dir / "comprehensive_linter.py"
            result = subprocess.run(
                ["python", str(linter_script), "--format", "json"],
                capture_output=True,
                text=True,
                cwd=self.root,
            )

            if result.returncode != 0:
                print(f"❌ Linter execution failed: {result.stderr}")
                return False, {"error": "Linter execution failed"}

            # Parse results
            output_lines = result.stdout.strip().split("\n")
            json_data = None

            for line in output_lines:
                if line.startswith("{") and '"health_score"' in line:
                    json_data = json.loads(line)
                    break

            if not json_data:
                print("❌ Could not parse linter output")
                return False, {"error": "Could not parse linter output"}

            # Check quality gates
            passed, gate_results = self._check_quality_gates(json_data)

            # Save CI report
            ci_report = {
                "timestamp": datetime.now().isoformat(),
                "passed": passed,
                "linting_results": json_data,
                "quality_gates": gate_results,
                "ci_summary": self._generate_ci_summary(json_data, gate_results),
            }

            report_file = (
                self.reports_dir
                / f"ci_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(report_file, "w") as f:
                json.dump(ci_report, f, indent=2)

            return passed, ci_report

        except Exception as e:
            print(f"❌ Error during CI check: {e}")
            return False, {"error": str(e)}

    def _check_quality_gates(self, linting_results: Dict) -> Tuple[bool, Dict]:
        """Check if results pass quality gates."""
        gates_passed = True
        gate_results = {}

        # Check health score
        health_score = linting_results.get("health_score", 0)
        min_health = self.quality_gates["min_health_score"]
        health_passed = health_score >= min_health
        gates_passed = gates_passed and health_passed

        gate_results["health_score"] = {
            "value": health_score,
            "threshold": min_health,
            "passed": health_passed,
            "message": f"Health score: {health_score:.1f} (min: {min_health})",
        }

        # Check critical issues (F821, F822, E999, etc.)
        issues_by_severity = linting_results.get("issues_by_severity", {})
        error_count = issues_by_severity.get("error", 0)

        # Critical issues check
        max_critical = self.quality_gates["max_critical_issues"]
        critical_passed = error_count <= max_critical
        gates_passed = gates_passed and critical_passed

        gate_results["critical_issues"] = {
            "value": error_count,
            "threshold": max_critical,
            "passed": critical_passed,
            "message": f"Critical issues: {error_count} (max: {max_critical})",
        }

        # Total error issues check
        max_errors = self.quality_gates["max_error_issues"]
        error_passed = error_count <= max_errors
        gates_passed = gates_passed and error_passed

        gate_results["error_issues"] = {
            "value": error_count,
            "threshold": max_errors,
            "passed": error_passed,
            "message": f"Error issues: {error_count} (max: {max_errors})",
        }

        # Total issues check
        total_issues = linting_results.get("total_issues", 0)
        max_total = self.quality_gates["max_total_issues"]
        total_passed = total_issues <= max_total
        gates_passed = gates_passed and total_passed

        gate_results["total_issues"] = {
            "value": total_issues,
            "threshold": max_total,
            "passed": total_passed,
            "message": f"Total issues: {total_issues} (max: {max_total})",
        }

        return gates_passed, gate_results

    def _generate_ci_summary(self, linting_results: Dict, gate_results: Dict) -> Dict:
        """Generate CI-friendly summary."""
        return {
            "overall_status": (
                "PASS" if all(g["passed"] for g in gate_results.values()) else "FAIL"
            ),
            "health_score": linting_results.get("health_score", 0),
            "total_issues": linting_results.get("total_issues", 0),
            "auto_fixable": linting_results.get("auto_fixable_count", 0),
            "failed_gates": [
                name for name, result in gate_results.items() if not result["passed"]
            ],
        }

    def run_auto_fix_safe(self, dry_run: bool = True) -> bool:
        """Run safe auto-fixes in CI environment."""
        print(f"🔧 Running safe auto-fix (dry_run={dry_run})...")

        try:
            # Run critical fixes (safe for CI)
            critical_fixes_script = self.tools_dir / "critical_fixes.py"
            args = ["python", str(critical_fixes_script)]
            if dry_run:
                args.append("--dry-run")

            result = subprocess.run(args, capture_output=True, text=True, cwd=self.root)

            if result.returncode != 0:
                print(f"❌ Critical fixes failed: {result.stderr}")
                return False

            print(result.stdout)

            # Run formatter fixes if not dry run
            if not dry_run:
                print("🎨 Running formatting fixes...")

                # Black formatting
                subprocess.run(
                    ["black", "src/", "tests/", "tools/", "scripts/", "examples/"],
                    cwd=self.root,
                    check=False,
                )

                # isort import sorting
                subprocess.run(
                    ["isort", "src/", "tests/", "tools/", "scripts/", "examples/"],
                    cwd=self.root,
                    check=False,
                )

            return True

        except Exception as e:
            print(f"❌ Error during auto-fix: {e}")
            return False

    def generate_ci_output(self, ci_report: Dict):
        """Generate CI-friendly output."""
        summary = ci_report.get("ci_summary", {})
        gate_results = ci_report.get("quality_gates", {})

        print("\n" + "=" * 60)
        print("🏗️ CI LINTING REPORT")
        print("=" * 60)

        # Overall status
        status = summary.get("overall_status", "UNKNOWN")
        status_emoji = "✅" if status == "PASS" else "❌"
        print(f"{status_emoji} Overall Status: {status}")

        # Key metrics
        print(f"📊 Health Score: {summary.get('health_score', 0):.1f}/100")
        print(f"🐛 Total Issues: {summary.get('total_issues', 0)}")
        print(f"🔧 Auto-fixable: {summary.get('auto_fixable', 0)}")

        # Gate status
        print("\n🚪 Quality Gates:")
        for gate_name, gate_result in gate_results.items():
            gate_emoji = "✅" if gate_result["passed"] else "❌"
            print(f"  {gate_emoji} {gate_result['message']}")

        # Failed gates
        failed_gates = summary.get("failed_gates", [])
        if failed_gates:
            print(f"\n❌ Failed Gates: {', '.join(failed_gates)}")

        print("=" * 60)

        # Set CI environment variables for downstream tools
        if status == "PASS":
            print("export CHEMML_LINTING_STATUS=pass")
            print("export CHEMML_HEALTH_SCORE=" + str(summary.get("health_score", 0)))
        else:
            print("export CHEMML_LINTING_STATUS=fail")
            print("export CHEMML_FAILED_GATES=" + ",".join(failed_gates))

    def run_pre_commit_check(self) -> bool:
        """Run pre-commit checks."""
        print("🪝 Running pre-commit checks...")

        try:
            result = subprocess.run(
                ["pre-commit", "run", "--all-files"],
                capture_output=True,
                text=True,
                cwd=self.root,
            )

            print(result.stdout)
            if result.stderr:
                print(result.stderr)

            return result.returncode == 0

        except Exception as e:
            print(f"❌ Pre-commit check failed: {e}")
            return False

    def update_health_tracking(self):
        """Update health tracking in CI."""
        try:
            health_tracker_script = self.tools_dir / "health_tracker.py"
            result = subprocess.run(
                ["python", str(health_tracker_script), "--update"],
                capture_output=True,
                text=True,
                cwd=self.root,
            )

            if result.returncode == 0:
                print("📊 Health tracking updated")
            else:
                print(f"⚠️ Health tracking update failed: {result.stderr}")

        except Exception as e:
            print(f"⚠️ Error updating health tracking: {e}")


def main():
    """Main entry point for CI integration."""
    import argparse

    parser = argparse.ArgumentParser(
        description="CI/CD integration for ChemML linting framework"
    )
    parser.add_argument(
        "--check", action="store_true", help="Run comprehensive quality check"
    )
    parser.add_argument("--auto-fix", action="store_true", help="Run safe auto-fixes")
    parser.add_argument(
        "--pre-commit", action="store_true", help="Run pre-commit checks"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Run in dry-run mode (no changes)"
    )
    parser.add_argument(
        "--fail-on-errors",
        action="store_true",
        help="Exit with error code if quality gates fail",
    )
    parser.add_argument(
        "--update-health", action="store_true", help="Update health tracking"
    )

    args = parser.parse_args()

    ci_integration = CILintingIntegration()

    exit_code = 0

    # Run auto-fix if requested
    if args.auto_fix:
        success = ci_integration.run_auto_fix_safe(dry_run=args.dry_run)
        if not success and args.fail_on_errors:
            exit_code = 1

    # Run pre-commit checks if requested
    if args.pre_commit:
        success = ci_integration.run_pre_commit_check()
        if not success and args.fail_on_errors:
            exit_code = 1

    # Run comprehensive check if requested
    if args.check:
        passed, ci_report = ci_integration.run_comprehensive_check()
        ci_integration.generate_ci_output(ci_report)

        if not passed and args.fail_on_errors:
            exit_code = 1

    # Update health tracking if requested
    if args.update_health:
        ci_integration.update_health_tracking()

    # Default action if no specific action requested
    if not any([args.check, args.auto_fix, args.pre_commit, args.update_health]):
        # Run full CI pipeline
        print("🏗️ Running full CI linting pipeline...")

        # 1. Run auto-fix (dry-run first, then real if safe)
        if ci_integration.run_auto_fix_safe(dry_run=True):
            ci_integration.run_auto_fix_safe(dry_run=False)

        # 2. Run comprehensive check
        passed, ci_report = ci_integration.run_comprehensive_check()
        ci_integration.generate_ci_output(ci_report)

        # 3. Update health tracking
        ci_integration.update_health_tracking()

        if not passed and args.fail_on_errors:
            exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
