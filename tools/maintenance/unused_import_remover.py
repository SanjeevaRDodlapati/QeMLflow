#!/usr/bin/env python3
"""
Automated Unused Import Remover for QeMLflow
==========================================

Safely removes unused imports (F401 errors) using AST analysis.
This tool:
1. Parses Python files to understand import usage
2. Identifies truly unused imports
3. Preserves imports that might be used in eval(), exec(), or string contexts
4. Creates backups before making changes
5. Validates syntax after each change
"""

import ast
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple


class UnusedImportRemover:
    """Safe unused import removal with comprehensive analysis."""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.backup_dir = (
            self.project_root / "backups" / f"unused_imports_{int(time.time())}"
        )
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # Imports to preserve even if they appear unused
        self.preserve_imports = {
            # Common eval/exec imports
            "numpy",
            "np",
            "pandas",
            "pd",
            "matplotlib",
            "plt",
            # Framework imports that might be used dynamically
            "torch",
            "tensorflow",
            "sklearn",
            "rdkit",
            # QeMLflow core imports
            "qemlflow",
            "QeMLflowError",
            # Special imports that create side effects
            "__version__",
            "__all__",
        }

        # Patterns that indicate import usage in strings/dynamic contexts
        self.dynamic_usage_patterns = [
            r"eval\s*\(",
            r"exec\s*\(",
            r"getattr\s*\(",
            r"hasattr\s*\(",
            r"importlib",
            r"__import__",
            r"globals\(\)",
            r"locals\(\)",
        ]

    def get_f401_errors(self) -> List[Tuple[str, int, str]]:
        """Get all F401 errors from flake8."""
        try:
            cmd = [
                "python",
                "-m",
                "flake8",
                "--select=F401",
                "--format=%(path)s:%(row)d:%(col)d:%(text)s",
                "src/",
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=self.project_root
            )

            errors = []
            for line in result.stdout.strip().split("\n"):
                if line and ":" in line:
                    parts = line.split(":")
                    if len(parts) >= 4:
                        file_path = parts[0]
                        line_num = int(parts[1])
                        error_msg = ":".join(parts[3:]).strip()
                        errors.append((file_path, line_num, error_msg))

            return errors
        except Exception as e:
            print(f"Error getting F401 errors: {e}")
            return []

    def analyze_file_imports(self, file_path: str) -> Tuple[List[str], Set[str]]:
        """Analyze a Python file to find imports and their usage."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Parse AST
            tree = ast.parse(content)

            # Find all imports
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name.split(".")[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module.split(".")[0])
                    for alias in node.names:
                        imports.append(alias.name)

            # Find used names
            used_names = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    used_names.add(node.id)
                elif isinstance(node, ast.Attribute):
                    # Handle attribute access like numpy.array
                    if isinstance(node.value, ast.Name):
                        used_names.add(node.value.id)

            # Check for dynamic usage patterns
            for pattern in self.dynamic_usage_patterns:
                if re.search(pattern, content):
                    # If file has dynamic usage, be more conservative
                    return imports, used_names.union(set(imports))

            return imports, used_names

        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return [], set()

    def is_safe_to_remove(
        self, import_name: str, file_path: str, line_content: str
    ) -> bool:
        """Determine if an import is safe to remove."""
        # Never remove preserved imports
        if any(preserve in import_name.lower() for preserve in self.preserve_imports):
            return False

        # Check if it's an __all__ import
        if "__all__" in line_content or "import *" in line_content:
            return False

        # Be conservative with framework imports
        framework_imports = [
            "sklearn",
            "torch",
            "tensorflow",
            "rdkit",
            "numpy",
            "pandas",
        ]
        if any(fw in import_name.lower() for fw in framework_imports):
            return False

        # Check file content for string usage
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Look for import name in strings or comments
            import_base = import_name.split(".")[-1]
            if f'"{import_base}"' in content or f"'{import_base}'" in content:
                return False

        except Exception:
            pass

        return True

    def remove_unused_import_line(self, file_path: str, line_num: int) -> bool:
        """Remove a specific import line from a file."""
        try:
            # Create backup
            backup_path = self.backup_dir / Path(file_path).name
            shutil.copy2(file_path, backup_path)

            # Read file
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Remove the line (convert to 0-based index)
            if 0 <= line_num - 1 < len(lines):
                removed_line = lines[line_num - 1].strip()
                lines.pop(line_num - 1)

                # Write back
                with open(file_path, "w", encoding="utf-8") as f:
                    f.writelines(lines)

                # Validate syntax
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        ast.parse(f.read())
                    print(f"✅ Removed: {removed_line} from {file_path}")
                    return True
                except SyntaxError:
                    # Restore from backup
                    shutil.copy2(backup_path, file_path)
                    print(f"❌ Syntax error after removing line, restored: {file_path}")
                    return False

        except Exception as e:
            print(f"Error removing import from {file_path}: {e}")
            return False

        return False

    def process_file(self, file_path: str, errors: List[Tuple[int, str]]) -> int:
        """Process a single file to remove unused imports."""
        print(f"\n🔍 Processing: {file_path}")

        # Sort errors by line number in descending order to avoid line number shifts
        errors.sort(key=lambda x: x[0], reverse=True)

        removed_count = 0
        for line_num, error_msg in errors:
            # Extract import name from error message
            import_match = re.search(r"'([^']+)' imported but unused", error_msg)
            if not import_match:
                continue

            import_name = import_match.group(1)

            # Read the actual line to check content
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                if 0 <= line_num - 1 < len(lines):
                    line_content = lines[line_num - 1].strip()
                else:
                    continue
            except Exception:
                continue

            # Check if safe to remove
            if self.is_safe_to_remove(import_name, file_path, line_content):
                if self.remove_unused_import_line(file_path, line_num):
                    removed_count += 1
            else:
                print(f"🛡️  Preserved (safety): {import_name} in {file_path}")

        return removed_count

    def run(self) -> Tuple[int, int]:
        """Run the unused import removal process."""
        print("🚀 Starting Unused Import Removal")
        print(f"📁 Project root: {self.project_root}")
        print(f"💾 Backups stored in: {self.backup_dir}")

        # Get all F401 errors
        print("\n📊 Analyzing F401 errors...")
        errors = self.get_f401_errors()

        if not errors:
            print("✅ No F401 errors found!")
            return 0, 0

        print(f"📋 Found {len(errors)} F401 errors")

        # Group errors by file
        files_errors = {}
        for file_path, line_num, error_msg in errors:
            if file_path not in files_errors:
                files_errors[file_path] = []
            files_errors[file_path].append((line_num, error_msg))

        # Process each file
        total_removed = 0
        total_files = len(files_errors)

        for i, (file_path, file_errors) in enumerate(files_errors.items(), 1):
            print(f"\n[{i}/{total_files}] Processing {file_path}")
            removed = self.process_file(file_path, file_errors)
            total_removed += removed

        return total_removed, len(errors)


def main():
    """Main function to run unused import removal."""
    import argparse

    parser = argparse.ArgumentParser(description="Remove unused imports safely")
    parser.add_argument(
        "--project-root", default=os.getcwd(), help="Project root directory"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    project_root = args.project_root
    remover = UnusedImportRemover(project_root)
    removed, total = remover.run()

    print(f"\n🎉 COMPLETION SUMMARY")
    print(f"📊 Total F401 errors found: {total}")
    print(f"🗑️  Successfully removed: {removed}")
    print(f"🛡️  Preserved for safety: {total - removed}")
    print(
        f"✅ Success rate: {(removed/total*100):.1f}%"
        if total > 0
        else "✅ No errors to fix"
    )

    if removed > 0:
        print(f"\n🔄 Running flake8 again to verify...")
        try:
            result = subprocess.run(
                ["python", "-m", "flake8", "--select=F401", "--count", "src/"],
                capture_output=True,
                text=True,
                cwd=project_root,
            )
            remaining = result.stdout.strip()
            print(f"📈 Remaining F401 errors: {remaining}")
        except Exception:
            print("⚠️  Could not verify remaining errors")


if __name__ == "__main__":
    main()
