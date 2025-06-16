#!/usr/bin/env python3
"""
Quick Workflow Status Checker
=============================

Provides a fast status check of GitHub Actions workflows and documentation
without requiring API calls. Useful for quick local checks.
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path


def check_workflow_files():
    """Check if workflow files exist and are properly configured."""
    workflow_dir = Path(".github/workflows")
    issues = []

    if not workflow_dir.exists():
        issues.append("❌ .github/workflows directory not found")
        return issues

    expected_workflows = ["ci-cd.yml", "docs.yml", "release.yml", "monitoring.yml"]

    for workflow in expected_workflows:
        workflow_path = workflow_dir / workflow
        if not workflow_path.exists():
            issues.append(f"⚠️ Missing workflow: {workflow}")
        else:
            issues.append(f"✅ Found workflow: {workflow}")

    return issues


def check_documentation_setup():
    """Check if documentation files are properly set up."""
    issues = []

    # Check mkdocs.yml
    if Path("mkdocs.yml").exists():
        issues.append("✅ mkdocs.yml configuration found")
    else:
        issues.append("❌ mkdocs.yml not found")

    # Check docs directory
    docs_dir = Path("docs")
    if docs_dir.exists():
        issues.append("✅ docs/ directory found")

        # Check key documentation files
        key_files = [
            "index.md",
            "getting-started/quick-start.md",
            "user-guide/overview.md",
        ]
        for file in key_files:
            if (docs_dir / file).exists():
                issues.append(f"✅ Found: docs/{file}")
            else:
                issues.append(f"⚠️ Missing: docs/{file}")
    else:
        issues.append("❌ docs/ directory not found")

    return issues


def check_git_status():
    """Check git repository status."""
    issues = []

    try:
        # Check if we're in a git repository
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"], capture_output=True, text=True, cwd="."
        )

        if result.returncode != 0:
            issues.append("❌ Not in a git repository")
            return issues

        issues.append("✅ Git repository detected")

        # Check current branch
        result = subprocess.run(
            ["git", "branch", "--show-current"], capture_output=True, text=True
        )

        if result.returncode == 0:
            branch = result.stdout.strip()
            issues.append(f"📍 Current branch: {branch}")

        # Check if there are uncommitted changes
        result = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True
        )

        if result.returncode == 0:
            if result.stdout.strip():
                issues.append("⚠️ Uncommitted changes detected")
            else:
                issues.append("✅ Working directory clean")

        # Check remote status
        result = subprocess.run(["git", "remote", "-v"], capture_output=True, text=True)

        if result.returncode == 0 and result.stdout.strip():
            issues.append("✅ Git remote configured")
        else:
            issues.append("⚠️ No git remote found")

    except FileNotFoundError:
        issues.append("❌ Git not installed or not in PATH")
    except Exception as e:
        issues.append(f"❌ Git check error: {e}")

    return issues


def check_python_environment():
    """Check Python environment and dependencies."""
    issues = []

    # Check Python version
    python_version = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    issues.append(f"🐍 Python version: {python_version}")

    # Check if we're in a virtual environment
    if hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    ):
        issues.append("✅ Virtual environment detected")
    else:
        issues.append("⚠️ Not in a virtual environment")

    # Check key files
    if Path("requirements.txt").exists():
        issues.append("✅ requirements.txt found")
    else:
        issues.append("⚠️ requirements.txt not found")

    if Path("pyproject.toml").exists():
        issues.append("✅ pyproject.toml found")
    else:
        issues.append("⚠️ pyproject.toml not found")

    return issues


def main():
    """Run quick status checks."""
    print("⚡ ChemML Quick Status Checker")
    print("=" * 50)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    all_issues = []

    # Run checks
    print("🔧 Checking GitHub Workflows...")
    workflow_issues = check_workflow_files()
    all_issues.extend(workflow_issues)
    for issue in workflow_issues:
        print(f"   {issue}")
    print()

    print("📚 Checking Documentation Setup...")
    doc_issues = check_documentation_setup()
    all_issues.extend(doc_issues)
    for issue in doc_issues:
        print(f"   {issue}")
    print()

    print("📦 Checking Python Environment...")
    python_issues = check_python_environment()
    all_issues.extend(python_issues)
    for issue in python_issues:
        print(f"   {issue}")
    print()

    print("📋 Checking Git Status...")
    git_issues = check_git_status()
    all_issues.extend(git_issues)
    for issue in git_issues:
        print(f"   {issue}")
    print()

    # Summary
    error_count = len([i for i in all_issues if i.startswith("❌")])
    warning_count = len([i for i in all_issues if i.startswith("⚠️")])
    success_count = len([i for i in all_issues if i.startswith("✅")])

    print("📊 Summary:")
    print(f"   ✅ Success: {success_count}")
    print(f"   ⚠️ Warnings: {warning_count}")
    print(f"   ❌ Errors: {error_count}")
    print()

    if error_count > 0:
        print("🚨 Critical issues detected!")
        print(
            "💡 Run 'python scripts/monitoring/automated_monitor.py' for detailed analysis"
        )
        sys.exit(2)
    elif warning_count > 0:
        print("⚠️ Some issues detected, but system should work")
        sys.exit(1)
    else:
        print("🎉 All basic checks passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
