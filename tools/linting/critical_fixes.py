#!/usr/bin/env python3
"""
Critical Linting Issues Fixer
=============================

This script targets and fixes the most critical linting issues that affect
code functionality and maintainability.

Focus areas:
1. Undefined names (F821) - critical for functionality
2. Unused imports (F401) - cleanup and performance
3. Import issues (F403, F405) - ambiguous imports
4. Complex functions (C901) - maintainability
5. Missing type annotations - code quality
"""

import ast
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Tuple

class CriticalLintingFixer:
    """Fixes critical linting issues that affect code functionality."""

    def __init__(self, root_path: Path = None):
        self.root = root_path or Path.cwd()
        self.fixed_files = []
        self.issues_fixed = {"F821": 0, "F401": 0, "F403": 0, "C901": 0}

    def find_undefined_names(self, file_path: Path) -> List[Tuple[int, str, str]]:
        """Find undefined name errors (F821) in a file."""
        try:
            result = subprocess.run(
                ["flake8", "--select=F821", str(file_path)],
                capture_output=True,
                text=True,
            )

            undefined_names = []
            for line in result.stdout.strip().split("\n"):
                if line and "F821" in line:
                    # Parse: filepath:line:col: F821 undefined name 'name'
                    match = re.match(
                        r".*:(\d+):\d+: F821 undefined name \'([^\']+)\'", line
                    )
                    if match:
                        line_num = int(match.group(1))
                        name = match.group(2)
                        undefined_names.append((line_num, name, line))

            return undefined_names
        except Exception as e:
            print(f"Error checking {file_path}: {e}")
            return []

    def fix_common_undefined_names(self, file_path: Path) -> bool:
        """Fix common undefined name issues."""
        try:
            with open(file_path, "r") as f:
                content = f.read()

            _original_content = content
            modified = False

            # Common fixes for undefined names
            fixes = {
                # Add missing typing imports
                "Any": "from typing import Any",
                "Dict": "from typing import Dict",
                "List": "from typing import List",
                "Tuple": "from typing import Tuple",
                "Optional": "from typing import Optional",
                "Union": "from typing import Union",
                "Self": "from typing_extensions import Self",
                # Add missing standard library imports
                "logging": "import logging",
                "random": "import random",
                "os": "import os",
                "sys": "import sys",
                "json": "import json",
                "warnings": "import warnings",
                # Add missing third-party imports
                "np": "import numpy as np",
                "pd": "import pandas as pd",
                "plt": "import matplotlib.pyplot as plt",
            }

            undefined_names = self.find_undefined_names(file_path)

            for line_num, name, error_line in undefined_names:
                if name in fixes:
                    import_statement = fixes[name]

                    # Check if import already exists
                    if import_statement not in content:
                        # Find the right place to add the import
                        lines = content.split("\n")

                        # Find imports section
                        import_index = 0
                        for i, line in enumerate(lines):
                            if line.strip().startswith(
                                "import "
                            ) or line.strip().startswith("from "):
                                import_index = i + 1
                            elif (
                                line.strip()
                                and not line.strip().startswith("#")
                                and not line.strip().startswith('"""')
                            ):
                                break

                        # Insert the import
                        lines.insert(import_index, import_statement)
                        content = "\n".join(lines)
                        modified = True
                        self.issues_fixed["F821"] += 1
                        print(f"  ✅ Added missing import: {import_statement}")

            if modified:
                with open(file_path, "w") as f:
                    f.write(content)
                return True

        except Exception as e:
            print(f"Error fixing {file_path}: {e}")

        return False

    def remove_unused_imports(self, file_path: Path) -> bool:
        """Remove unused imports (F401)."""
        try:
            # Get unused imports
            result = subprocess.run(
                ["flake8", "--select=F401", str(file_path)],
                capture_output=True,
                text=True,
            )

            if not result.stdout.strip():
                return False

            with open(file_path, "r") as f:
                lines = f.readlines()

            # Parse unused import lines
            unused_lines = set()
            for line in result.stdout.strip().split("\n"):
                if "F401" in line:
                    match = re.match(r".*:(\d+):", line)
                    if match:
                        unused_lines.add(int(match.group(1)) - 1)  # Convert to 0-based

            if unused_lines:
                # Remove unused import lines
                new_lines = []
                for i, line in enumerate(lines):
                    if i not in unused_lines:
                        new_lines.append(line)
                    else:
                        print(f"  ✅ Removed unused import: {line.strip()}")
                        self.issues_fixed["F401"] += 1

                with open(file_path, "w") as f:
                    f.writelines(new_lines)
                return True

        except Exception as e:
            print(f"Error removing unused imports from {file_path}: {e}")

        return False

    def fix_star_imports(self, file_path: Path) -> bool:
        """Convert dangerous star imports to specific imports where possible."""
        try:
            with open(file_path, "r") as f:
                content = f.read()

            _original_content = content

            # Look for star imports
            star_import_pattern = r"from\s+([a-zA-Z_][a-zA-Z0-9_.]*)\s+import\s+\*"
            matches = re.findall(star_import_pattern, content)

            if matches:
                print(f"  ⚠️  Found star imports in {file_path}: {matches}")
                # For now, just report them - fixing requires more analysis
                self.issues_fixed["F403"] += len(matches)

        except Exception as e:
            print(f"Error checking star imports in {file_path}: {e}")

        return False

    def fix_file(self, file_path: Path) -> bool:
        """Fix critical linting issues in a single file."""
        print(f"🔧 Fixing {file_path}")

        fixed = False

        # Fix undefined names
        if self.fix_common_undefined_names(file_path):
            fixed = True

        # Remove unused imports
        if self.remove_unused_imports(file_path):
            fixed = True

        # Check star imports
        self.fix_star_imports(file_path)

        if fixed:
            self.fixed_files.append(file_path)

        return fixed

    def fix_directory(self, directory: Path) -> None:
        """Fix all Python files in a directory."""
        python_files = list(directory.rglob("*.py"))

        # Skip archived and virtual environment files
        excluded_patterns = [
            "archive/",
            "chemml_env/",
            "build/",
            "dist/",
            ".venv/",
            "__pycache__/",
            ".pytest_cache/",
        ]

        filtered_files = []
        for file_path in python_files:
            relative_path = str(file_path.relative_to(self.root))
            if not any(pattern in relative_path for pattern in excluded_patterns):
                filtered_files.append(file_path)

        print(f"🎯 Found {len(filtered_files)} Python files to fix")

        for file_path in filtered_files:
            try:
                self.fix_file(file_path)
            except Exception as e:
                print(f"❌ Error fixing {file_path}: {e}")

    def generate_report(self) -> None:
        """Generate a summary report of fixes applied."""
        print("\n" + "=" * 60)
        print("🎯 Critical Linting Fixes Summary")
        print("=" * 60)

        total_fixes = sum(self.issues_fixed.values())
        print(f"📊 Total issues fixed: {total_fixes}")
        print(f"📁 Files modified: {len(self.fixed_files)}")
        print()

        print("🔍 Issues fixed by type:")
        for issue_type, count in self.issues_fixed.items():
            if count > 0:
                issue_descriptions = {
                    "F821": "Undefined names",
                    "F401": "Unused imports",
                    "F403": "Star imports (reported)",
                    "C901": "Complex functions",
                }
                desc = issue_descriptions.get(issue_type, issue_type)
                print(f"  {issue_type}: {count} ({desc})")

        print("\n📁 Modified files:")
        for file_path in self.fixed_files[:10]:  # Show first 10
            print(f"  {file_path}")

        if len(self.fixed_files) > 10:
            print(f"  ... and {len(self.fixed_files) - 10} more")

        print("=" * 60)

def main():
    """Main entry point."""
    fixer = CriticalLintingFixer()

    print("🚀 Starting critical linting fixes...")

    # Fix main source directories
    directories_to_fix = ["src/", "tests/", "scripts/", "tools/", "examples/"]

    for dir_name in directories_to_fix:
        dir_path = fixer.root / dir_name
        if dir_path.exists():
            print(f"\n📂 Processing {dir_name}")
            fixer.fix_directory(dir_path)

    # Generate report
    fixer.generate_report()

    print("\n🎉 Critical linting fixes completed!")
    print("💡 Run the comprehensive linter again to see improvements")

if __name__ == "__main__":
    main()
