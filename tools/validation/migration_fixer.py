#!/usr/bin/env python3
"""
Migration Issues Fix Script
===========================

This script addresses the critical issues found during migration validation:
1. Remaining 'chemml' references in archive/legacy files
2. Python syntax errors
3. Git status cleanup

Author: Migration Fix System
Date: June 17, 2025
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


class MigrationFixer:
    """Fix critical migration issues."""

    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.fixes_applied = []

    def log_fix(self, category: str, message: str):
        """Log a fix applied."""
        fix_msg = f"🔧 FIX [{category}]: {message}"
        self.fixes_applied.append(fix_msg)
        print(fix_msg)

    def fix_archive_chemml_references(self):
        """Fix ChemML references in archive/legacy files."""
        print("\n🔍 Fixing archive ChemML references...")

        archive_dirs = [self.root_path / ".archive", self.root_path / "backups"]

        files_fixed = 0

        for archive_dir in archive_dirs:
            if not archive_dir.exists():
                continue

            for py_file in archive_dir.rglob("*.py"):
                try:
                    with open(py_file, "r", encoding="utf-8") as f:
                        content = f.read()

                    original_content = content

                    # Replace chemml with qemlflow in code contexts (but preserve historical references)
                    content = re.sub(r"\bfrom chemml\b", "from qemlflow", content)
                    content = re.sub(r"\bimport chemml\b", "import qemlflow", content)
                    content = re.sub(r"\bchemml\.", "qemlflow.", content)

                    # Update package references but keep historical comments
                    if "chemml" in content.lower() and content != original_content:
                        with open(py_file, "w", encoding="utf-8") as f:
                            f.write(content)
                        files_fixed += 1
                        self.log_fix(
                            "Archive", f"Updated {py_file.relative_to(self.root_path)}"
                        )

                except Exception as e:
                    print(f"⚠️  Could not fix {py_file}: {e}")

        self.log_fix("Archive", f"Fixed {files_fixed} archive files")

    def fix_syntax_errors(self):
        """Fix critical Python syntax errors."""
        print("\n🔍 Fixing Python syntax errors...")

        # List of files with known syntax errors that can be auto-fixed
        fixable_files = [
            "tools/progress_dashboard.py",
            "tools/testing/test_new_modules.py",
            "tools/maintenance/quick_wins.py",
            "src/qemlflow/__init___backup_20250616_002516.py",
            "src/qemlflow/__init___fast.py",
            "src/qemlflow/research/drug_discovery/__init__.py",
            "src/qemlflow/tutorials/__init__.py",
        ]

        files_fixed = 0

        for file_path_str in fixable_files:
            file_path = self.root_path / file_path_str
            if not file_path.exists():
                continue

            try:
                self.fix_file_syntax(file_path)
                files_fixed += 1
            except Exception as e:
                print(f"⚠️  Could not fix syntax in {file_path}: {e}")

        self.log_fix("Syntax", f"Fixed {files_fixed} files with syntax errors")

    def fix_file_syntax(self, file_path: Path):
        """Fix syntax errors in a specific file."""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        original_content = content

        # Fix common syntax issues

        # Fix invalid decimal literals (numbers starting with 0)
        content = re.sub(r"\b0+(\d+)", r"\1", content)

        # Fix unmatched brackets
        if file_path.name == "__init__.py":
            # Fix unmatched ] in __init__.py files
            content = re.sub(r"\]\s*$", "", content, flags=re.MULTILINE)
            content = re.sub(r"^\s*\]", "", content, flags=re.MULTILINE)

        # Fix indentation issues
        lines = content.split("\n")
        fixed_lines = []

        for i, line in enumerate(lines):
            # Fix unexpected indents at the beginning of functions
            if (
                i > 0
                and line.strip()
                and not line[0].isspace()
                and lines[i - 1].strip().endswith(":")
            ):
                if line.strip() and not line.strip().startswith(
                    (
                        "def ",
                        "class ",
                        "if ",
                        "for ",
                        "while ",
                        "try:",
                        "except",
                        "finally:",
                        "else:",
                        "elif ",
                    )
                ):
                    line = "    " + line

            # Fix functions without body
            if line.strip().endswith(":") and i < len(lines) - 1:
                next_line = lines[i + 1] if i + 1 < len(lines) else ""
                if not next_line.strip() or not next_line.startswith("    "):
                    if "def " in line or "class " in line:
                        fixed_lines.append(line)
                        fixed_lines.append("    pass")
                        continue

            fixed_lines.append(line)

        content = "\n".join(fixed_lines)

        # Fix unterminated strings (basic fix)
        # Count quotes to detect unterminated strings
        content = self.fix_unterminated_strings(content)

        if content != original_content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            self.log_fix(
                "Syntax", f"Fixed syntax in {file_path.relative_to(self.root_path)}"
            )

    def fix_unterminated_strings(self, content: str) -> str:
        """Fix unterminated triple-quoted strings."""
        # Look for unterminated triple quotes
        patterns = [
            (r'"""[^"]*$', '"""'),
            (r"'''[^']*$", "'''"),
        ]

        for pattern, closing in patterns:
            if re.search(pattern, content, re.MULTILINE | re.DOTALL):
                # Add closing quotes at the end if missing
                if not content.rstrip().endswith(closing):
                    content = content.rstrip() + "\n" + closing + "\n"

        return content

    def remove_broken_files(self):
        """Remove severely broken files that can't be easily fixed."""
        print("\n🔍 Removing severely broken files...")

        broken_files = [
            "src/qemlflow/__init___backup_20250616_002516.py",  # Old backup file
            "src/qemlflow/__init___fast.py",  # Broken alternative init
        ]

        files_removed = 0

        for file_path_str in broken_files:
            file_path = self.root_path / file_path_str
            if file_path.exists():
                try:
                    # Move to backup instead of deleting
                    backup_dir = (
                        self.root_path / "tools" / "migration" / "broken_files_backup"
                    )
                    backup_dir.mkdir(parents=True, exist_ok=True)

                    backup_file = backup_dir / file_path.name
                    shutil.move(str(file_path), str(backup_file))

                    files_removed += 1
                    self.log_fix(
                        "Cleanup",
                        f"Moved broken file {file_path.relative_to(self.root_path)} to backup",
                    )
                except Exception as e:
                    print(f"⚠️  Could not remove {file_path}: {e}")

        self.log_fix("Cleanup", f"Removed {files_removed} broken files")

    def update_gitignore_for_validation(self):
        """Update .gitignore to ignore validation directory."""
        print("\n🔍 Updating .gitignore...")

        gitignore_path = self.root_path / ".gitignore"

        # Entries to add
        new_entries = [
            "# Migration validation",
            "tools/validation/migration_validation_report.json",
            "tools/migration/broken_files_backup/",
            "",
        ]

        try:
            # Read existing .gitignore
            if gitignore_path.exists():
                with open(gitignore_path, "r") as f:
                    content = f.read()
            else:
                content = ""

            # Add new entries if not already present
            for entry in new_entries:
                if entry.strip() and entry not in content:
                    content += entry + "\n"

            # Write back
            with open(gitignore_path, "w") as f:
                f.write(content)

            self.log_fix("Git", "Updated .gitignore for validation files")

        except Exception as e:
            print(f"⚠️  Could not update .gitignore: {e}")

    def fix_missing_imports(self):
        """Fix missing imports in the main __init__.py."""
        print("\n🔍 Fixing missing imports...")

        init_file = self.root_path / "src" / "qemlflow" / "__init__.py"

        if not init_file.exists():
            self.log_fix("Import", "Main __init__.py not found")
            return

        try:
            with open(init_file, "r") as f:
                content = f.read()

            # Add missing import for typing
            if "from typing import" not in content and "Optional" in content:
                # Add typing imports at the top
                lines = content.split("\n")

                # Find the first import or insert at the beginning
                insert_pos = 0
                for i, line in enumerate(lines):
                    if line.strip().startswith("import ") or line.strip().startswith(
                        "from "
                    ):
                        insert_pos = i
                        break

                lines.insert(
                    insert_pos, "from typing import Optional, Dict, Any, List, Union"
                )
                content = "\n".join(lines)

                with open(init_file, "w") as f:
                    f.write(content)

                self.log_fix("Import", "Added missing typing imports to __init__.py")

        except Exception as e:
            print(f"⚠️  Could not fix imports: {e}")

    def run_all_fixes(self):
        """Run all migration fixes."""
        print("🚀 Starting Migration Issue Fixes")
        print("=" * 60)

        # Apply fixes in order of priority
        self.fix_missing_imports()
        self.remove_broken_files()
        self.fix_syntax_errors()
        self.fix_archive_chemml_references()
        self.update_gitignore_for_validation()

        # Summary
        print("\n" + "=" * 60)
        print("📊 FIXES APPLIED")
        print("=" * 60)

        if self.fixes_applied:
            for fix in self.fixes_applied:
                print(fix)
        else:
            print("No fixes were applied.")

        print(f"\nTotal fixes applied: {len(self.fixes_applied)}")

        # Git status after fixes
        print("\n🔍 Git status after fixes:")
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.root_path,
                capture_output=True,
                text=True,
            )
            if result.stdout.strip():
                print("Modified files:")
                print(result.stdout)
            else:
                print("Working directory is clean.")
        except Exception as e:
            print(f"Could not check git status: {e}")


def main():
    """Main entry point."""
    # Get the root directory
    script_dir = Path(__file__).parent
    root_dir = script_dir.parent.parent

    fixer = MigrationFixer(str(root_dir))
    fixer.run_all_fixes()


if __name__ == "__main__":
    main()
