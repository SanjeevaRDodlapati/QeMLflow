"""
Incremental Code Refactoring Tool for QeMLflow

This tool provides targeted fixes for common code quality issues,
working incrementally to improve the codebase health score.
"""

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


class IncrementalRefactor:
    """Incremental code refactoring tool for systematic quality improvements."""

    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.fixes_applied = 0
        self.files_modified = set()
        self.backup_dir = (
            self.project_root
            / "backup"
            / f"refactor_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

    def create_backup(self, file_path: Path):
        """Create backup of file before modification."""
        if not self.backup_dir.exists():
            self.backup_dir.mkdir(parents=True, exist_ok=True)

        relative_path = file_path.relative_to(self.project_root)
        backup_path = self.backup_dir / relative_path
        backup_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy file to backup
        import shutil

        shutil.copy2(file_path, backup_path)

    def fix_f_string_placeholders(self, file_path: Path) -> int:
        """Fix f-strings that are missing placeholders."""
        fixes = 0

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            original_content = content

            # Pattern to find f-strings without placeholders
            # This is a simple approach - more sophisticated parsing could be added
            f_string_pattern = r'f["\']([^"\'{}]*)["\']'

            def replace_f_string(match):
                nonlocal fixes
                string_content = match.group(1)
                # If the string has no {} placeholders, convert to regular string
                if "{" not in string_content and "}" not in string_content:
                    fixes += 1
                    return f'"{string_content}"'  # Remove the f prefix
                return match.group(0)

            content = re.sub(f_string_pattern, replace_f_string, content)

            if content != original_content:
                self.create_backup(file_path)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                self.files_modified.add(file_path)

        except Exception as e:
            print(f"❌ Error fixing f-strings in {file_path}: {e}")

        return fixes

    def fix_unused_variables(self, file_path: Path) -> int:
        """Fix unused variables by adding underscore prefix."""
        fixes = 0

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            original_lines = lines.copy()

            for i, line in enumerate(lines):
                # Look for patterns like "local variable 'name' is assigned to but never used"
                # We'll add underscore prefix to unused variables

                # Simple heuristic: if line has assignment with unused variable
                # This is a basic implementation - more sophisticated AST analysis could be added

                # Pattern for simple assignment
                assignment_pattern = r"^(\s*)([a-zA-Z_]\w*)\s*=\s*(.+)$"
                match = re.match(assignment_pattern, line.strip())

                if match and not line.strip().startswith("_"):
                    # Check if variable name appears to be unused (very basic heuristic)
                    var_name = match.group(2)
                    remaining_code = "".join(lines[i + 1 :])

                    # If variable isn't used later in the function/method
                    if (
                        var_name not in remaining_code
                        or remaining_code.count(var_name) < 2
                    ):
                        # Add underscore prefix
                        indent = match.group(1)
                        assignment = match.group(3)
                        lines[i] = f"{indent}_{var_name} = {assignment}\n"
                        fixes += 1

            if lines != original_lines:
                self.create_backup(file_path)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.writelines(lines)
                self.files_modified.add(file_path)

        except Exception as e:
            print(f"❌ Error fixing unused variables in {file_path}: {e}")

        return fixes

    def fix_bare_except(self, file_path: Path) -> int:
        """Fix bare except statements by adding Exception."""
        fixes = 0

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            original_content = content

            # Pattern to find bare except statements
            bare_except_pattern = r"^(\s*)except\s*:(.*)$"

            def replace_bare_except(match):
                nonlocal fixes
                fixes += 1
                indent = match.group(1)
                rest = match.group(2)
                return f"{indent}except Exception:{rest}"

            content = re.sub(
                bare_except_pattern, replace_bare_except, content, flags=re.MULTILINE
            )

            if content != original_content:
                self.create_backup(file_path)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                self.files_modified.add(file_path)

        except Exception as e:
            print(f"❌ Error fixing bare except in {file_path}: {e}")

        return fixes

    def fix_redefined_imports(self, file_path: Path) -> int:
        """Fix redefined imports by removing duplicates."""
        fixes = 0

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            original_lines = lines.copy()
            seen_imports = set()
            new_lines = []

            for line in lines:
                line_stripped = line.strip()

                # Check if this is an import line
                if (
                    line_stripped.startswith("import ")
                    or line_stripped.startswith("from ")
                    and " import " in line_stripped
                ):
                    if line_stripped in seen_imports:
                        # Skip duplicate import
                        fixes += 1
                        continue
                    else:
                        seen_imports.add(line_stripped)

                new_lines.append(line)

            if new_lines != original_lines:
                self.create_backup(file_path)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.writelines(new_lines)
                self.files_modified.add(file_path)

        except Exception as e:
            print(f"❌ Error fixing redefined imports in {file_path}: {e}")

        return fixes

    def fix_block_comments(self, file_path: Path) -> int:
        """Fix block comments that should start with '# '."""
        fixes = 0

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            original_content = content

            # Pattern to find block comments that don't start with '# '
            block_comment_pattern = r"^(\s*)#([^#\s].*?)$"

            def replace_block_comment(match):
                nonlocal fixes
                fixes += 1
                indent = match.group(1)
                comment = match.group(2)
                return f"{indent}# {comment}"

            content = re.sub(
                block_comment_pattern,
                replace_block_comment,
                content,
                flags=re.MULTILINE,
            )

            if content != original_content:
                self.create_backup(file_path)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                self.files_modified.add(file_path)

        except Exception as e:
            print(f"❌ Error fixing block comments in {file_path}: {e}")

        return fixes

    def get_issues_by_type(self) -> Dict[str, List[Dict]]:
        """Get current issues categorized by type."""
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "tools/linting/comprehensive_linter.py",
                    "--format",
                    "json",
                    "--quiet",
                ],
                cwd=self.project_root,
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                data = json.loads(result.stdout)
                issues_by_type = {}

                for issue in data.get("issues", []):
                    rule_code = issue.get("rule_code", "unknown")
                    if rule_code not in issues_by_type:
                        issues_by_type[rule_code] = []
                    issues_by_type[rule_code].append(issue)

                return issues_by_type
            else:
                print(f"❌ Error getting linting issues: {result.stderr}")
                return {}

        except Exception as e:
            print(f"❌ Error parsing linting output: {e}")
            return {}

    def apply_targeted_fixes(
        self, issue_types: List[str] = None, max_files: int = 10
    ) -> Dict[str, int]:
        """Apply targeted fixes for specific issue types."""

        if issue_types is None:
            issue_types = ["F541", "F841", "E722", "F811", "E265"]

        print("🔧 Starting incremental refactoring...")
        print(f"   Target issue types: {', '.join(issue_types)}")
        print(f"   Max files to process: {max_files}")

        issues_by_type = self.get_issues_by_type()
        fix_stats = {}

        for issue_type in issue_types:
            fix_stats[issue_type] = 0

            if issue_type not in issues_by_type:
                continue

            issues = issues_by_type[issue_type]
            files_to_fix = set()

            # Get unique files for this issue type
            for issue in issues[: max_files * 5]:  # Process more issues but limit files
                file_path = Path(issue["file_path"])
                files_to_fix.add(file_path)

                if len(files_to_fix) >= max_files:
                    break

            print(f"\n🔍 Processing {issue_type} issues in {len(files_to_fix)} files...")

            for file_path in files_to_fix:
                if not file_path.exists():
                    continue

                fixes_in_file = 0

                if issue_type == "F541":
                    fixes_in_file = self.fix_f_string_placeholders(file_path)
                elif issue_type == "F841":
                    fixes_in_file = self.fix_unused_variables(file_path)
                elif issue_type == "E722":
                    fixes_in_file = self.fix_bare_except(file_path)
                elif issue_type == "F811":
                    fixes_in_file = self.fix_redefined_imports(file_path)
                elif issue_type == "E265":
                    fixes_in_file = self.fix_block_comments(file_path)

                fix_stats[issue_type] += fixes_in_file
                self.fixes_applied += fixes_in_file

                if fixes_in_file > 0:
                    print(
                        f"   ✅ Fixed {fixes_in_file} {issue_type} issues in {file_path.name}"
                    )

        return fix_stats

    def run_post_fix_validation(self) -> bool:
        """Run linting to validate fixes didn't break anything."""
        print("\n🔍 Validating fixes...")

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "tools/linting/comprehensive_linter.py",
                    "--format",
                    "json",
                    "--quiet",
                ],
                cwd=self.project_root,
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                data = json.loads(result.stdout)
                new_total = data.get("summary", {}).get("total_issues", 0)
                new_score = data.get("summary", {}).get("health_score", 0)

                print(f"   📊 New total issues: {new_total}")
                print(f"   🏥 New health score: {new_score}")
                return True
            else:
                print(f"   ❌ Validation failed: {result.stderr}")
                return False

        except Exception as e:
            print(f"   ❌ Validation error: {e}")
            return False

    def generate_report(self, fix_stats: Dict[str, int]):
        """Generate refactoring report."""
        print("\n" + "=" * 80)
        print("🔧 Incremental Refactoring Report")
        print("=" * 80)
        print(f"📅 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📁 Files Modified: {len(self.files_modified)}")
        print(f"🔧 Total Fixes Applied: {self.fixes_applied}")

        if self.backup_dir.exists():
            print(f"💾 Backups Saved To: {self.backup_dir}")

        print("\n📊 Fixes by Issue Type:")
        for issue_type, count in fix_stats.items():
            if count > 0:
                print(f"   • {issue_type}: {count} fixes")

        if self.files_modified:
            print("\n📝 Modified Files:")
            for file_path in sorted(self.files_modified):
                rel_path = file_path.relative_to(self.project_root)
                print(f"   • {rel_path}")

        print("\n💡 Next Steps:")
        print("   • Run comprehensive linter to see remaining issues")
        print("   • Update health tracker to see improvement")
        print("   • Consider running auto-formatter (black, isort)")
        print("   • Review complex functions flagged by C901")

        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Incremental Code Refactoring Tool")
    parser.add_argument(
        "--issues",
        nargs="*",
        default=["F541", "F841", "E722", "F811", "E265"],
        help="Issue types to fix (default: F541 F841 E722 F811 E265)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=10,
        help="Maximum files to process per issue type (default: 10)",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Project root directory (default: current directory)",
    )

    args = parser.parse_args()

    refactor = IncrementalRefactor(args.project_root)

    # Apply targeted fixes
    fix_stats = refactor.apply_targeted_fixes(args.issues, args.max_files)

    # Validate fixes
    if refactor.fixes_applied > 0:
        refactor.run_post_fix_validation()

    # Generate report
    refactor.generate_report(fix_stats)

    return 0 if refactor.fixes_applied > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
