#!/usr/bin/env python3
"""
Final Migration Validation Report
=================================

This script generates a comprehensive final report on the ChemML to QeMLflow migration,
summarizing all aspects of the migration and providing recommendations.

Author: Final Validation System
Date: June 17, 2025
"""

import importlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def run_command(cmd: List[str], cwd: str = None) -> Dict[str, Any]:
    """Run a command and return result."""
    try:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }
    except Exception as e:
        return {"success": False, "error": str(e), "returncode": -1}


def generate_final_report():
    """Generate comprehensive final migration report."""

    # Get root directory
    root_path = Path(__file__).parent.parent.parent
    report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = {
        "migration_report": {
            "title": "ChemML to QeMLflow Migration - Final Validation Report",
            "timestamp": report_time,
            "version": "1.0.0",
            "migration_date": "2025-06-17",
        }
    }

    # 1. Git Repository Status
    print("📊 Analyzing Git Repository Status...")
    git_status = run_command(["git", "status", "--porcelain"], str(root_path))
    git_log = run_command(["git", "log", "--oneline", "-10"], str(root_path))
    git_remote = run_command(["git", "remote", "-v"], str(root_path))

    report["git_analysis"] = {
        "working_directory_clean": git_status["success"]
        and not git_status["stdout"].strip(),
        "recent_commits": git_log["stdout"].split("\n")[:5]
        if git_log["success"]
        else [],
        "remote_configured": "origin" in git_remote["stdout"]
        if git_remote["success"]
        else False,
        "untracked_files": [
            line.strip()
            for line in git_status["stdout"].split("\n")
            if line.strip() and line.startswith("??")
        ]
        if git_status["success"]
        else [],
    }

    # 2. QeMLflow Package Analysis
    print("📦 Analyzing QeMLflow Package...")
    try:
        # Add src to path
        src_path = root_path / "src"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))

        import qemlflow

        package_info = {
            "import_successful": True,
            "version": getattr(qemlflow, "__version__", "Unknown"),
            "package_path": str(Path(qemlflow.__file__).parent),
            "main_modules": [],
        }

        # Test core modules
        core_modules = ["utils", "config"]
        for module in core_modules:
            try:
                importlib.import_module(f"qemlflow.{module}")
                package_info["main_modules"].append(module)
            except ImportError:
                pass

        report["package_analysis"] = package_info

    except Exception as e:
        report["package_analysis"] = {"import_successful": False, "error": str(e)}

    # 3. File Structure Analysis
    print("📁 Analyzing File Structure...")

    critical_files = [
        "src/qemlflow/__init__.py",
        "setup.py",
        "pyproject.toml",
        "requirements.txt",
        "README.md",
        ".gitignore",
    ]

    critical_dirs = ["src/qemlflow", "tests", "docs", "examples", "tools", "notebooks"]

    structure_analysis = {
        "critical_files": {},
        "critical_directories": {},
        "backup_present": (root_path / "qemlflow_backup_20250617_041123").exists(),
    }

    for file_path in critical_files:
        full_path = root_path / file_path
        structure_analysis["critical_files"][file_path] = {
            "exists": full_path.exists(),
            "size": full_path.stat().st_size if full_path.exists() else 0,
        }

    for dir_path in critical_dirs:
        full_path = root_path / dir_path
        if full_path.exists():
            file_count = len(list(full_path.rglob("*")))
            structure_analysis["critical_directories"][dir_path] = {
                "exists": True,
                "file_count": file_count,
            }
        else:
            structure_analysis["critical_directories"][dir_path] = {"exists": False}

    report["structure_analysis"] = structure_analysis

    # 4. Content Analysis
    print("🔍 Analyzing Content Migration...")

    # Check for remaining chemml references in critical files
    chemml_refs = []
    qemlflow_refs = []

    for file_path in critical_files:
        full_path = root_path / file_path
        if not full_path.exists():
            continue

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read().lower()

            if "chemml" in content:
                chemml_refs.append(file_path)

            if "qemlflow" in content:
                qemlflow_refs.append(file_path)

        except Exception:
            continue

    report["content_analysis"] = {
        "chemml_references_in_critical_files": chemml_refs,
        "qemlflow_references_in_critical_files": qemlflow_refs,
        "migration_completeness": len(qemlflow_refs)
        / max(len(critical_files), 1)
        * 100,
    }

    # 5. Environment Analysis
    print("🌍 Analyzing Environment...")

    env_analysis = {
        "python_version": sys.version,
        "qemlflow_env_exists": (root_path / "qemlflow_env").exists(),
        "old_chemml_env_exists": (root_path / "chemml_env").exists(),
    }

    # Test environment
    if env_analysis["qemlflow_env_exists"]:
        python_exe = root_path / "qemlflow_env" / "bin" / "python"
        if not python_exe.exists():
            python_exe = root_path / "qemlflow_env" / "Scripts" / "python.exe"

        if python_exe.exists():
            env_test = run_command(
                [str(python_exe), "-c", "import qemlflow; print(qemlflow.__version__)"]
            )
            env_analysis["environment_test"] = {
                "success": env_test["success"],
                "version": env_test["stdout"].strip() if env_test["success"] else None,
            }

    report["environment_analysis"] = env_analysis

    # 6. Migration Tools Analysis
    print("🔧 Analyzing Migration Tools...")

    migration_tools = (
        list((root_path / "tools" / "migration").glob("*.py"))
        if (root_path / "tools" / "migration").exists()
        else []
    )
    validation_tools = (
        list((root_path / "tools" / "validation").glob("*.py"))
        if (root_path / "tools" / "validation").exists()
        else []
    )

    report["tools_analysis"] = {
        "migration_tools": [tool.name for tool in migration_tools],
        "validation_tools": [tool.name for tool in validation_tools],
        "tools_count": len(migration_tools) + len(validation_tools),
    }

    # 7. Overall Assessment
    print("🎯 Generating Overall Assessment...")

    # Calculate scores
    scores = {
        "git_status": 100 if report["git_analysis"]["working_directory_clean"] else 70,
        "package_import": 100 if report["package_analysis"]["import_successful"] else 0,
        "structure_completeness": sum(
            1 for f in structure_analysis["critical_files"].values() if f["exists"]
        )
        / len(critical_files)
        * 100,
        "content_migration": report["content_analysis"]["migration_completeness"],
        "environment_setup": 100
        if env_analysis.get("environment_test", {}).get("success")
        else 80,
    }

    overall_score = sum(scores.values()) / len(scores)

    # Determine status
    if overall_score >= 95:
        status = "EXCELLENT"
        recommendation = (
            "Migration is complete and excellent. Ready for production use."
        )
    elif overall_score >= 85:
        status = "GOOD"
        recommendation = (
            "Migration is substantially complete. Minor issues may need attention."
        )
    elif overall_score >= 70:
        status = "ACCEPTABLE"
        recommendation = "Migration is functional but some issues need to be addressed."
    else:
        status = "NEEDS_WORK"
        recommendation = (
            "Migration has significant issues that need immediate attention."
        )

    report["final_assessment"] = {
        "overall_score": round(overall_score, 1),
        "individual_scores": scores,
        "status": status,
        "recommendation": recommendation,
    }

    # 8. Recommendations
    recommendations = []

    if not report["git_analysis"]["working_directory_clean"]:
        recommendations.append("Clean up untracked files and commit remaining changes")

    if chemml_refs:
        recommendations.append(
            f"Review ChemML references in critical files: {chemml_refs}"
        )

    if not env_analysis.get("environment_test", {}).get("success"):
        recommendations.append("Test and fix virtual environment setup")

    if overall_score < 95:
        recommendations.append(
            "Run comprehensive test suite to identify remaining issues"
        )

    if not recommendations:
        recommendations.append(
            "Migration appears complete. Consider final documentation update."
        )

    report["recommendations"] = recommendations

    # Save report
    report_file = root_path / "tools" / "validation" / "final_migration_report.json"
    report_file.parent.mkdir(parents=True, exist_ok=True)

    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 80)
    print("🎉 FINAL MIGRATION VALIDATION REPORT")
    print("=" * 80)
    print(f"Overall Score: {overall_score:.1f}/100 ({status})")
    print(
        f"Package Import: {'✅' if report['package_analysis']['import_successful'] else '❌'}"
    )
    print(
        f"Git Status: {'✅' if report['git_analysis']['working_directory_clean'] else '⚠️'}"
    )
    print(f"Structure: {scores['structure_completeness']:.1f}% complete")
    print(f"Content Migration: {scores['content_migration']:.1f}% complete")

    if report["environment_analysis"].get("environment_test"):
        env_status = (
            "✅"
            if report["environment_analysis"]["environment_test"]["success"]
            else "❌"
        )
        print(f"Environment: {env_status}")

    print(f"\n📋 Recommendation: {recommendation}")

    if recommendations:
        print("\n🔧 Action Items:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")

    print(f"\n📄 Detailed report saved to: {report_file}")

    return report


def main():
    """Main entry point."""
    try:
        report = generate_final_report()

        # Return appropriate exit code
        if report["final_assessment"]["overall_score"] >= 85:
            return 0
        else:
            return 1

    except Exception as e:
        print(f"❌ Error generating report: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
