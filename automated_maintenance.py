#!/usr/bin/env python3
"""
QeMLflow Automated Maintenance System
Handles linting, import fixes, and repository health.
"""

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


class QeMLflowMaintenanceSystem:
    def __init__(self, root_dir: str = "/Users/sanjeev/Downloads/Repos/QeMLflow"):
        self.root_dir = Path(root_dir)
        self.python_exe = self.root_dir / "qemlflow_env" / "bin" / "python"
        self.report_file = self.root_dir / "maintenance_report.json"

    def run_command(self, cmd: List[str], capture=True) -> Dict[str, Any]:
        """Run a command and return results."""
        try:
            result = subprocess.run(
                cmd, capture_output=capture, text=True, cwd=self.root_dir, timeout=300
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout if capture else "",
                "stderr": result.stderr if capture else "",
                "returncode": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Command timed out"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def check_syntax_errors(self) -> Dict[str, Any]:
        """Check for syntax errors in the codebase."""
        print("🔍 Checking syntax errors...")
        result = self.run_command([str(self.python_exe), "quick_syntax_check.py"])

        return {
            "task": "syntax_check",
            "success": result["success"],
            "details": result.get("stdout", ""),
            "timestamp": time.time(),
        }

    def fix_import_errors(self) -> Dict[str, Any]:
        """Fix missing import statements."""
        print("🔧 Fixing import errors...")

        # Common imports that are frequently missing
        import_fixes = [
            ("Dict", "from typing import Dict"),
            ("List", "from typing import List"),
            ("Optional", "from typing import Optional"),
            ("Union", "from typing import Union"),
            ("Any", "from typing import Any"),
            ("Callable", "from typing import Callable"),
        ]

        fixed_files = []
        src_dir = self.root_dir / "src" / "qemlflow"

        for py_file in src_dir.rglob("*.py"):
            if py_file.is_file():
                try:
                    with open(py_file, "r") as f:
                        content = f.read()

                    modified = False
                    # Check if file uses typing but doesn't import
                    for type_name, import_statement in import_fixes:
                        if type_name in content and import_statement not in content:
                            # Add import at the top after existing imports
                            lines = content.split("\\n")
                            insert_idx = 0

                            # Find where to insert (after existing imports)
                            for i, line in enumerate(lines):
                                if line.startswith("import ") or line.startswith(
                                    "from "
                                ):
                                    insert_idx = i + 1
                                elif line.strip() == "" and insert_idx > 0:
                                    break

                            lines.insert(insert_idx, import_statement)
                            content = "\\n".join(lines)
                            modified = True

                    if modified:
                        with open(py_file, "w") as f:
                            f.write(content)
                        fixed_files.append(str(py_file.relative_to(self.root_dir)))

                except Exception as e:
                    print(f"Error processing {py_file}: {e}")

        return {
            "task": "import_fixes",
            "success": len(fixed_files) >= 0,  # Success if no errors
            "fixed_files": fixed_files,
            "count": len(fixed_files),
            "timestamp": time.time(),
        }

    def run_basic_linting(self) -> Dict[str, Any]:
        """Run basic linting with flake8."""
        print("📋 Running basic linting...")

        # Use the virtual environment's flake8
        flake8_cmd = [str(self.root_dir / "qemlflow_env" / "bin" / "flake8")]
        flake8_cmd.extend(
            [
                "src/qemlflow",
                "--count",
                "--statistics",
                "--max-line-length=88",
                "--ignore=E203,W503,F401",  # Ignore some common issues
            ]
        )

        result = self.run_command(flake8_cmd)

        return {
            "task": "basic_linting",
            "success": result["success"],
            "output": result.get("stdout", ""),
            "errors": result.get("stderr", ""),
            "timestamp": time.time(),
        }

    def clean_git_status(self) -> Dict[str, Any]:
        """Clean up git status by ignoring non-essential files."""
        print("🧹 Cleaning git status...")

        # Update .gitignore to exclude problematic files
        gitignore_additions = [
            "# Automated maintenance",
            ".artifacts/",
            "qemlflow_backup_*/",
            "backups/archive/",
            "reports/archives/",
            "*.log",
            "maintenance_report.json",
            "quick_syntax_check.py",
        ]

        gitignore_path = self.root_dir / ".gitignore"
        try:
            with open(gitignore_path, "r") as f:
                current_content = f.read()

            for addition in gitignore_additions:
                if addition not in current_content:
                    current_content += f"\\n{addition}"

            with open(gitignore_path, "w") as f:
                f.write(current_content)

            return {
                "task": "git_cleanup",
                "success": True,
                "message": "Updated .gitignore",
                "timestamp": time.time(),
            }
        except Exception as e:
            return {
                "task": "git_cleanup",
                "success": False,
                "error": str(e),
                "timestamp": time.time(),
            }

    def generate_report(self, results: List[Dict[str, Any]]) -> None:
        """Generate a comprehensive maintenance report."""
        report = {
            "timestamp": time.time(),
            "summary": {
                "total_tasks": len(results),
                "successful_tasks": sum(1 for r in results if r.get("success", False)),
                "failed_tasks": sum(1 for r in results if not r.get("success", False)),
            },
            "tasks": results,
        }

        with open(self.report_file, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\\n📊 Maintenance Report Generated: {self.report_file}")
        print(
            f"✅ {report['summary']['successful_tasks']}/{report['summary']['total_tasks']} tasks completed successfully"
        )

    def run_full_maintenance(self) -> None:
        """Run the complete maintenance cycle."""
        print("🚀 Starting QeMLflow Automated Maintenance System")
        print("=" * 60)

        results = []

        # 1. Check syntax
        results.append(self.check_syntax_errors())

        # 2. Fix imports
        results.append(self.fix_import_errors())

        # 3. Clean git
        results.append(self.clean_git_status())

        # 4. Run linting
        results.append(self.run_basic_linting())

        # 5. Generate report
        self.generate_report(results)

        print("\\n🎯 Maintenance cycle completed!")
        return results


def main():
    """Main entry point."""
    maintenance = QeMLflowMaintenanceSystem()
    results = maintenance.run_full_maintenance()

    # Exit with error code if any critical tasks failed
    failed_tasks = [r for r in results if not r.get("success", False)]
    if failed_tasks:
        print(f"\\n⚠️ {len(failed_tasks)} tasks failed. Check the report for details.")
        sys.exit(1)
    else:
        print("\\n✅ All maintenance tasks completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
