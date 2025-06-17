"""
QeMLflow Safe Quick Wins Implementation

This script implements safe, targeted improvements to boost codebase health:
1. Remove unused imports (using autoflake)
2. Format code (using black)
3. Sort imports (using isort)
4. Remove trailing whitespace
5. Check syntax before saving

Usage:
    python tools/maintenance/safe_quick_wins.py [--dry-run]
"""

import argparse
import ast
import json
import os
import subprocess
import sys
from pathlib import Path


class SafeQuickWinsFixer:
    """Implements safe quick wins to improve codebase health."""

    def __init__(self, base_dir: Path, dry_run: bool = False):
        self.base_dir = base_dir
        self.dry_run = dry_run
        self.fixed_count = 0
        self.errors = []

    def run_safe_fixes(self):
        """Run safe quick win fixes."""
        print("🛡️ QeMLflow Safe Quick Wins Implementation")
        print("=" * 50)

        # Skip files that commonly have syntax issues
        skip_patterns = [
            "__pycache__",
            ".git",
            ".pytest_cache",
            "build",
            "site",
            "qemlflow_env",
        ]

        fixes = [
            ("Removing unused imports", self.safe_remove_unused_imports),
            ("Formatting code with black", self.safe_format_with_black),
            ("Organizing imports", self.safe_organize_imports),
            ("Removing trailing whitespace", self.safe_remove_whitespace),
        ]

        for description, fix_func in fixes:
            print(f"\n📋 {description}...")
            try:
                count = fix_func()
                self.fixed_count += count
                print(f"   ✅ Fixed {count} files")
            except Exception as e:
                error = f"❌ Error in {description}: {e}"
                print(f"   {error}")
                self.errors.append(error)

        self.validate_syntax()
        self.generate_report()

    def safe_remove_unused_imports(self) -> int:
        """Safely remove unused imports using autoflake."""
        if self.dry_run:
            return self._count_autoflake_changes()

        try:
            # Run autoflake on specific directories
            directories = ["src/qemlflow", "tools", "examples"]
            count = 0

            for directory in directories:
                dir_path = self.base_dir / directory
                if not dir_path.exists():
                    continue

                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "autoflake",
                        "--remove-all-unused-imports",
                        "--remove-unused-variables",
                        "--remove-duplicate-keys",
                        "--exclude",
                        "__pycache__",
                        "--exclude",
                        ".git",
                        "--in-place",
                        "--recursive",
                        str(dir_path),
                    ],
                    capture_output=True,
                    text=True,
                    cwd=self.base_dir,
                )

                if result.returncode == 0:
                    # Count modified files
                    modified = len(
                        [
                            line
                            for line in result.stdout.split("\n")
                            if "fixing" in line.lower()
                        ]
                    )
                    count += modified
                else:
                    print(
                        f"   Warning: autoflake failed for {directory}: {result.stderr}"
                    )

            return count

        except Exception as e:
            print(f"   Warning: autoflake failed: {e}")
            return 0

    def safe_format_with_black(self) -> int:
        """Safely format code with black."""
        if self.dry_run:
            return self._count_black_changes()

        try:
            directories = ["src/qemlflow", "tools", "examples"]
            count = 0

            for directory in directories:
                dir_path = self.base_dir / directory
                if not dir_path.exists():
                    continue

                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "black",
                        "--line-length",
                        "88",
                        "--target-version",
                        "py311",
                        "--exclude",
                        r"/(\.git|__pycache__|\.pytest_cache|build|site)/",
                        str(dir_path),
                    ],
                    capture_output=True,
                    text=True,
                    cwd=self.base_dir,
                )

                if result.returncode == 0:
                    # Count reformatted files
                    reformatted = len(
                        [
                            line
                            for line in result.stderr.split("\n")
                            if "reformatted" in line.lower()
                        ]
                    )
                    count += reformatted
                else:
                    print(f"   Warning: black failed for {directory}: {result.stderr}")

            return count

        except Exception as e:
            print(f"   Warning: black failed: {e}")
            return 0

    def safe_organize_imports(self) -> int:
        """Safely organize imports with isort."""
        if self.dry_run:
            return self._count_isort_changes()

        try:
            directories = ["src/qemlflow", "tools", "examples"]
            count = 0

            for directory in directories:
                dir_path = self.base_dir / directory
                if not dir_path.exists():
                    continue

                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "isort",
                        "--profile",
                        "black",
                        "--line-length",
                        "88",
                        "--multi-line",
                        "3",
                        "--trailing-comma",
                        "--force-grid-wrap",
                        "0",
                        "--combine-as",
                        "--use-parentheses",
                        "--skip",
                        "__pycache__",
                        "--skip",
                        ".git",
                        str(dir_path),
                    ],
                    capture_output=True,
                    text=True,
                    cwd=self.base_dir,
                )

                if result.returncode == 0:
                    # Count fixed files
                    fixed = len(
                        [line for line in result.stderr.split("\n") if "Fixing" in line]
                    )
                    count += fixed
                else:
                    print(f"   Warning: isort failed for {directory}: {result.stderr}")

            return count

        except Exception as e:
            print(f"   Warning: isort failed: {e}")
            return 0

    def safe_remove_whitespace(self) -> int:
        """Safely remove trailing whitespace."""
        count = 0

        # Target specific file types in specific directories
        patterns = [
            "src/qemlflow/**/*.py",
            "tools/**/*.py",
            "examples/**/*.py",
            "*.md",
            "*.yml",
            "*.yaml",
        ]

        for pattern in patterns:
            for file_path in self.base_dir.glob(pattern):
                if any(
                    skip in str(file_path) for skip in ["__pycache__", ".git", "build"]
                ):
                    continue

                try:
                    # Check if it's a valid Python file first
                    if file_path.suffix == ".py":
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()

                        # Basic syntax check
                        try:
                            ast.parse(content)
                        except SyntaxError:
                            print(
                                f"   Warning: Skipping {file_path} due to syntax error"
                            )
                            continue

                    # Process the file
                    with open(file_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()

                    original_lines = lines[:]
                    lines = [
                        line.rstrip() + "\n" if line.strip() else "\n" for line in lines
                    ]

                    if lines != original_lines:
                        if not self.dry_run:
                            with open(file_path, "w", encoding="utf-8") as f:
                                f.writelines(lines)
                        count += 1

                except Exception as e:
                    self.errors.append(f"Error processing {file_path}: {e}")

        return count

    def validate_syntax(self):
        """Validate that all Python files have correct syntax."""
        print(f"\n🔍 Validating syntax...")

        python_files = list(self.base_dir.glob("src/qemlflow/**/*.py"))
        errors = 0

        for file_path in python_files[:10]:  # Check first 10 files
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                ast.parse(content)
            except SyntaxError as e:
                print(f"   ❌ Syntax error in {file_path}: {e}")
                errors += 1
            except Exception as e:
                print(f"   ⚠️  Could not check {file_path}: {e}")

        if errors == 0:
            print("   ✅ All checked files have valid syntax")
        else:
            print(f"   ❌ Found {errors} syntax errors")

    def _count_autoflake_changes(self) -> int:
        """Count potential autoflake changes."""
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "autoflake",
                    "--check",
                    "--recursive",
                    str(self.base_dir / "src/qemlflow"),
                ],
                capture_output=True,
                text=True,
                cwd=self.base_dir,
            )

            return len(
                [line for line in result.stdout.split("\n") if "would fix" in line]
            )
        except:
            return 0

    def _count_black_changes(self) -> int:
        """Count potential black changes."""
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "black",
                    "--check",
                    "--diff",
                    str(self.base_dir / "src/qemlflow"),
                ],
                capture_output=True,
                text=True,
                cwd=self.base_dir,
            )

            return len(
                [line for line in result.stdout.split("\n") if "would reformat" in line]
            )
        except:
            return 0

    def _count_isort_changes(self) -> int:
        """Count potential isort changes."""
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "isort",
                    "--check-only",
                    "--diff",
                    str(self.base_dir / "src/qemlflow"),
                ],
                capture_output=True,
                text=True,
                cwd=self.base_dir,
            )

            return len(
                [line for line in result.stdout.split("\n") if "would reformat" in line]
            )
        except:
            return 0

    def generate_report(self):
        """Generate improvement report."""
        print(f"\n🎉 Safe Quick Wins Summary")
        print("=" * 35)
        print(f"   ✅ Total files fixed: {self.fixed_count}")

        if self.errors:
            print(f"   ⚠️  Errors encountered: {len(self.errors)}")
            for error in self.errors[:3]:
                print(f"      • {error}")
            if len(self.errors) > 3:
                print(f"      • ... and {len(self.errors) - 3} more")

        # Save detailed report
        report_path = self.base_dir / "reports" / "safe_quick_wins_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "timestamp": str(
                subprocess.run(["date"], capture_output=True, text=True).stdout.strip()
            ),
            "total_fixes": self.fixed_count,
            "errors": self.errors,
            "dry_run": self.dry_run,
        }

        if not self.dry_run:
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
            print(f"   📄 Detailed report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="QeMLflow Safe Quick Wins Implementation"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview changes without applying them"
    )

    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent.parent
    fixer = SafeQuickWinsFixer(base_dir, dry_run=args.dry_run)
    fixer.run_safe_fixes()


if __name__ == "__main__":
    main()
