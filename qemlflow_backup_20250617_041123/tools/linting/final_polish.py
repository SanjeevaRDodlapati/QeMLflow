#!/usr/bin/env python3
"""
Final Polish Tool - Address remaining high-impact issues
"""

import re
import subprocess
import sys
from pathlib import Path


class FinalPolish:
    """Apply final polishing fixes to maximize code quality."""

    def __init__(self, workspace_root: str = None):
        self.workspace_root = Path(workspace_root) if workspace_root else Path.cwd()

    def run_auto_fixes(self):
        """Run auto-fixable improvements."""
        print("🔧 Running auto-fix tools...")

        # Run black for formatting
        print("📝 Running black formatter...")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "black", "--line-length", "88", "."],
                cwd=self.workspace_root,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                print("✅ Black formatting completed")
            else:
                print(f"⚠️ Black formatting issues: {result.stderr}")
        except Exception as e:
            print(f"❌ Error running black: {e}")

        # Run isort for import organization
        print("📦 Running isort for imports...")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "isort", "--profile", "black", "."],
                cwd=self.workspace_root,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                print("✅ Import organization completed")
            else:
                print(f"⚠️ Isort issues: {result.stderr}")
        except Exception as e:
            print(f"❌ Error running isort: {e}")

    def fix_whitespace_issues(self):
        """Fix whitespace and formatting issues."""
        print("🧹 Fixing whitespace issues...")

        python_files = []
        for py_file in self.workspace_root.rglob("*.py"):
            # Skip problematic directories
            if any(
                excluded in py_file.parts
                for excluded in {".git", "__pycache__", "chemml_env", "site", "build"}
            ):
                continue
            python_files.append(py_file)

        fixes = 0
        for file_path in python_files[:50]:  # Limit to first 50 files
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                original_content = content

                # Fix trailing whitespace
                lines = content.split("\n")
                for i, line in enumerate(lines):
                    lines[i] = line.rstrip()
                content = "\n".join(lines)

                # Remove multiple blank lines (more than 2)
                content = re.sub(r"\n\n\n+", "\n\n", content)

                if content != original_content:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(content)
                    fixes += 1

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

        print(f"✅ Fixed whitespace issues in {fixes} files")
        return fixes

    def run_final_check(self):
        """Run a final health check."""
        print("🏥 Running final health check...")
        try:
            result = subprocess.run(
                [sys.executable, "tools/linting/health_tracker.py", "--update"],
                cwd=self.workspace_root,
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                print("✅ Health check completed")
                # Extract health score from output
                output = result.stdout
                if "Health Score:" in output:
                    score_line = [
                        line for line in output.split("\n") if "Health Score:" in line
                    ]
                    if score_line:
                        print(f"📊 {score_line[0].strip()}")
            else:
                print(f"⚠️ Health check issues: {result.stderr}")

        except Exception as e:
            print(f"❌ Error running health check: {e}")

    def run_final_polish(self):
        """Run all final polishing steps."""
        print("🚀 Starting final code quality polish...")
        print("=" * 60)

        # Run auto-fixes
        self.run_auto_fixes()
        print()

        # Fix whitespace issues
        self.fix_whitespace_issues()
        print()

        # Final health check
        self.run_final_check()
        print()

        print("✨ Final polish completed!")
        print("=" * 60)


def main():
    """Main function."""
    polish = FinalPolish()
    polish.run_final_polish()


if __name__ == "__main__":
    main()
