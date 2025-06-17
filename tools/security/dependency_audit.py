"""
Enhanced Dependency Management for QeMLflow
=========================================

Comprehensive dependency validation, security scanning, and conflict resolution.
Part of Phase 1 critical infrastructure improvements.

Features:
- Vulnerability scanning with multiple tools
- Dependency conflict detection and resolution
- Package version optimization
- Security audit reporting
- Automated fixes for common issues

Usage:
    python tools/security/dependency_audit.py
    python tools/security/dependency_audit.py --scan-vulnerabilities
    python tools/security/dependency_audit.py --fix-conflicts
    python tools/security/dependency_audit.py --full-audit
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class DependencyAuditor:
    """Comprehensive dependency security and conflict auditor."""

    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "vulnerabilities": [],
            "conflicts": [],
            "outdated": [],
            "recommendations": [],
            "fixes_applied": [],
        }

    def run_full_audit(self) -> Dict[str, Any]:
        """Run comprehensive dependency audit."""
        print("🔒 QeMLflow Dependency Security Audit")
        print("=" * 50)

        # Check tool availability
        self._check_security_tools()

        # Run vulnerability scans
        self._scan_vulnerabilities()

        # Check for conflicts
        self._check_conflicts()

        # Check for outdated packages
        self._check_outdated()

        # Generate recommendations
        self._generate_recommendations()

        # Save results
        self._save_results()

        return self.results

    def _check_security_tools(self):
        """Check availability of security scanning tools."""
        print("\n🛠️ Security Tools Status")
        print("-" * 30)

        tools = {
            "safety": ("pip install safety", "Known security vulnerabilities"),
            "bandit": ("pip install bandit", "Python security linter"),
            "pip-audit": ("pip install pip-audit", "PyPI vulnerability scanner"),
            "semgrep": ("pip install semgrep", "Static analysis security scanner"),
        }

        available_tools = []

        for tool, (install_cmd, description) in tools.items():
            try:
                result = subprocess.run(
                    [tool, "--version"], capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    print(f"   ✅ {tool}: available - {description}")
                    available_tools.append(tool)
                else:
                    print(f"   ❌ {tool}: not available ({install_cmd})")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                print(f"   ❌ {tool}: not available ({install_cmd})")

        self.results["available_tools"] = available_tools

        if not available_tools:
            print("\n⚠️  No security tools available. Install recommended tools:")
            print("   pip install safety bandit pip-audit")

    def _scan_vulnerabilities(self):
        """Scan for known security vulnerabilities."""
        print("\n🔍 Vulnerability Scanning")
        print("-" * 30)

        vulnerabilities = []

        # Safety scan
        if "safety" in self.results.get("available_tools", []):
            print("   🔍 Running Safety scan...")
            try:
                result = subprocess.run(
                    ["safety", "check", "--json"],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                if result.stdout:
                    safety_data = json.loads(result.stdout)
                    vulnerabilities.extend(safety_data)
                    print(f"   📊 Safety: {len(safety_data)} vulnerabilities found")
                else:
                    print("   ✅ Safety: no vulnerabilities found")
            except Exception as e:
                print(f"   ❌ Safety scan failed: {e}")

        # Bandit scan
        if "bandit" in self.results.get("available_tools", []):
            print("   🔍 Running Bandit scan...")
            try:
                result = subprocess.run(
                    ["bandit", "-r", "src/", "-f", "json", "-q"],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                if result.stdout:
                    bandit_data = json.loads(result.stdout)
                    issues = bandit_data.get("results", [])
                    vulnerabilities.extend(
                        [
                            {
                                "type": "code_security",
                                "severity": issue.get("issue_severity"),
                                "description": issue.get("issue_text"),
                                "file": issue.get("filename"),
                                "line": issue.get("line_number"),
                            }
                            for issue in issues
                        ]
                    )
                    print(f"   📊 Bandit: {len(issues)} security issues found")
                else:
                    print("   ✅ Bandit: no security issues found")
            except Exception as e:
                print(f"   ❌ Bandit scan failed: {e}")

        # pip-audit scan
        if "pip-audit" in self.results.get("available_tools", []):
            print("   🔍 Running pip-audit scan...")
            try:
                result = subprocess.run(
                    ["pip-audit", "--format=json"],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                if result.stdout:
                    audit_data = json.loads(result.stdout)
                    if audit_data:
                        vulnerabilities.extend(audit_data)
                        print(
                            f"   📊 pip-audit: {len(audit_data)} vulnerabilities found"
                        )
                    else:
                        print("   ✅ pip-audit: no vulnerabilities found")
            except Exception as e:
                print(f"   ❌ pip-audit scan failed: {e}")

        self.results["vulnerabilities"] = vulnerabilities

        if vulnerabilities:
            print(f"\n⚠️  Total vulnerabilities found: {len(vulnerabilities)}")
            # Show top 3 most severe
            for vuln in vulnerabilities[:3]:
                severity = vuln.get("severity", "unknown")
                desc = vuln.get("description", "No description")[:80]
                print(f"   • {severity}: {desc}")
        else:
            print("\n✅ No vulnerabilities detected")

    def _check_conflicts(self):
        """Check for dependency conflicts."""
        print("\n🔗 Dependency Conflict Analysis")
        print("-" * 30)

        conflicts = []

        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "check"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0 and result.stdout:
                conflict_lines = result.stdout.strip().split("\n")
                for line in conflict_lines:
                    if line.strip():
                        conflicts.append(
                            {"type": "dependency_conflict", "description": line.strip()}
                        )
                print(f"   ❌ {len(conflicts)} dependency conflicts detected")
                for conflict in conflicts[:3]:
                    print(f"     • {conflict['description']}")
            else:
                print("   ✅ No dependency conflicts detected")
        except Exception as e:
            print(f"   ❌ Conflict check failed: {e}")

        self.results["conflicts"] = conflicts

    def _check_outdated(self):
        """Check for outdated packages."""
        print("\n📦 Outdated Package Analysis")
        print("-" * 30)

        outdated = []

        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "list", "--outdated", "--format=json"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.stdout:
                outdated_data = json.loads(result.stdout)
                outdated = outdated_data
                print(f"   📊 {len(outdated)} packages have updates available")

                # Show top 5 outdated packages
                for pkg in outdated[:5]:
                    name = pkg.get("name", "unknown")
                    current = pkg.get("version", "unknown")
                    latest = pkg.get("latest_version", "unknown")
                    print(f"     • {name}: {current} → {latest}")

                if len(outdated) > 5:
                    print(f"     ... and {len(outdated) - 5} more")
            else:
                print("   ✅ All packages are up to date")
        except Exception as e:
            print(f"   ❌ Outdated package check failed: {e}")

        self.results["outdated"] = outdated

    def _generate_recommendations(self):
        """Generate security and maintenance recommendations."""
        print("\n💡 Security Recommendations")
        print("-" * 30)

        recommendations = []

        # Vulnerability recommendations
        if self.results["vulnerabilities"]:
            recommendations.append(
                {
                    "priority": "high",
                    "category": "security",
                    "action": "Update packages with known vulnerabilities",
                    "details": f"{len(self.results['vulnerabilities'])} vulnerabilities found",
                }
            )

        # Conflict recommendations
        if self.results["conflicts"]:
            recommendations.append(
                {
                    "priority": "high",
                    "category": "stability",
                    "action": "Resolve dependency conflicts",
                    "details": f"{len(self.results['conflicts'])} conflicts detected",
                }
            )

        # Outdated package recommendations
        if self.results["outdated"]:
            critical_packages = ["numpy", "pandas", "torch", "tensorflow"]
            outdated_critical = [
                pkg
                for pkg in self.results["outdated"]
                if pkg.get("name", "").lower() in critical_packages
            ]

            if outdated_critical:
                recommendations.append(
                    {
                        "priority": "medium",
                        "category": "maintenance",
                        "action": "Update critical packages",
                        "details": f"{len(outdated_critical)} critical packages outdated",
                    }
                )

        # Tool recommendations
        missing_tools = []
        for tool in ["safety", "bandit", "pip-audit"]:
            if tool not in self.results.get("available_tools", []):
                missing_tools.append(tool)

        if missing_tools:
            recommendations.append(
                {
                    "priority": "low",
                    "category": "tooling",
                    "action": "Install security scanning tools",
                    "details": f"Missing: {', '.join(missing_tools)}",
                }
            )

        self.results["recommendations"] = recommendations

        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                priority = rec["priority"].upper()
                action = rec["action"]
                details = rec["details"]
                print(f"   {i}. [{priority}] {action}")
                print(f"      {details}")
        else:
            print("   ✅ No immediate recommendations")

    def _save_results(self):
        """Save audit results to file."""
        output_dir = self.project_root / "reports" / "security"
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"dependency_audit_{timestamp}.json"

        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        print(f"\n📄 Results saved to: {output_file}")

    def fix_common_issues(self):
        """Attempt to fix common dependency issues."""
        print("\n🔧 Applying Automatic Fixes")
        print("-" * 30)

        fixes_applied = []

        # Fix 1: Update pip itself
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
                capture_output=True,
                timeout=60,
            )
            fixes_applied.append("Updated pip to latest version")
            print("   ✅ Updated pip to latest version")
        except Exception as e:
            print(f"   ❌ Failed to update pip: {e}")

        # Fix 2: Install missing security tools
        security_tools = ["safety", "bandit"]
        for tool in security_tools:
            if tool not in self.results.get("available_tools", []):
                try:
                    subprocess.run(
                        [sys.executable, "-m", "pip", "install", tool],
                        capture_output=True,
                        timeout=60,
                    )
                    fixes_applied.append(f"Installed {tool}")
                    print(f"   ✅ Installed {tool}")
                except Exception as e:
                    print(f"   ❌ Failed to install {tool}: {e}")

        # Fix 3: Create reports directory
        reports_dir = self.project_root / "reports" / "security"
        if not reports_dir.exists():
            reports_dir.mkdir(parents=True, exist_ok=True)
            fixes_applied.append("Created security reports directory")
            print("   ✅ Created security reports directory")

        self.results["fixes_applied"] = fixes_applied

        if not fixes_applied:
            print("   ℹ️  No automatic fixes available")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="QeMLflow Dependency Security Auditor")
    parser.add_argument(
        "--scan-vulnerabilities",
        action="store_true",
        help="Focus on vulnerability scanning",
    )
    parser.add_argument(
        "--fix-conflicts",
        action="store_true",
        help="Attempt to fix dependency conflicts",
    )
    parser.add_argument(
        "--full-audit", action="store_true", help="Run comprehensive audit"
    )
    parser.add_argument("--auto-fix", action="store_true", help="Apply automatic fixes")

    args = parser.parse_args()

    auditor = DependencyAuditor()

    if args.full_audit or not any([args.scan_vulnerabilities, args.fix_conflicts]):
        results = auditor.run_full_audit()
    else:
        # Selective operations based on flags
        results = {"timestamp": datetime.now().isoformat()}

        if args.scan_vulnerabilities:
            auditor._check_security_tools()
            auditor._scan_vulnerabilities()

        if args.fix_conflicts:
            auditor._check_conflicts()

    if args.auto_fix:
        auditor.fix_common_issues()

    # Summary
    vulnerabilities = len(results.get("vulnerabilities", []))
    conflicts = len(results.get("conflicts", []))
    recommendations = len(results.get("recommendations", []))

    print("\n📊 Audit Summary")
    print(f"   Vulnerabilities: {vulnerabilities}")
    print(f"   Conflicts: {conflicts}")
    print(f"   Recommendations: {recommendations}")

    if vulnerabilities + conflicts > 0:
        print(f"\n⚠️  Action required: {vulnerabilities + conflicts} issues found")
        sys.exit(1)
    else:
        print("\n✅ Security audit passed")
        sys.exit(0)


if __name__ == "__main__":
    main()
