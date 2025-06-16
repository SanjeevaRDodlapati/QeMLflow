#!/usr/bin/env python3
"""
Conservative Incremental Refactor Tool

Applies safe, targeted fixes for code quality issues with validation.
"""

import argparse
import ast
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

class ConservativeRefactor:
    """Conservative code refactoring with safety checks."""

    def __init__(self, workspace_root: str = None):
        self.workspace_root = Path(workspace_root) if workspace_root else Path.cwd()
        self.fixes_applied = []
        self.errors_encountered = []

    def validate_syntax(self, content: str) -> bool:
        """Validate that the content has valid Python syntax."""
        try:
            ast.parse(content)
            return True
        except SyntaxError:
            return False

    def fix_f_string_placeholders(self, content: str) -> Tuple[str, int]:
        """Fix F541: f-string is missing placeholders."""
        fixes = 0
        lines = content.split("\n")

        for i, line in enumerate(lines):
            # Pattern: f"text without any {placeholders}"
            if 'f"' in line or "f'" in line:
                # Simple pattern match for f-strings without placeholders
                pattern = r'f(["\'])([^{}]*?)\1'

                def replace_f_string(match):
                    quote = match.group(1)
                    text = match.group(2)
                    # Only replace if there are no braces (placeholders)
                    if "{" not in text and "}" not in text:
                        return f"{quote}{text}{quote}"  # Remove f prefix
                    return match.group(0)  # Keep as is

                new_line = re.sub(pattern, replace_f_string, line)
                if new_line != line:
                    lines[i] = new_line
                    fixes += 1

        return "\n".join(lines), fixes

    def fix_bare_except(self, content: str) -> Tuple[str, int]:
        """Fix E722: do not use bare 'except'."""
        # Pattern: except: -> except Exception:
        original_matches = len(re.findall(r"(\s+)except\s*:", content))
        new_content = re.sub(r"(\s+)except\s*:", r"\1except Exception:", content)
        new_matches = len(re.findall(r"(\s+)except\s*:", new_content))
        fixes = original_matches - new_matches
        return new_content, fixes

    def fix_comment_spacing(self, content: str) -> Tuple[str, int]:
        """Fix E265: block comment should start with '# '."""
        # Pattern: #comment -> # comment (only if not already properly spaced)
        original_matches = len(re.findall(r"(\n\s*)#([a-zA-Z])", content))
        new_content = re.sub(r"(\n\s*)#([a-zA-Z])", r"\1# \2", content)
        new_matches = len(re.findall(r"(\n\s*)#([a-zA-Z])", new_content))
        fixes = original_matches - new_matches
        return new_content, fixes

    def apply_safe_fixes(self, content: str) -> Tuple[str, Dict[str, int]]:
        """Apply only the safest fixes that are unlikely to break code."""
        fix_counts = {}
        current_content = content

        # Apply fixes in order of safety (most safe first)
        current_content, count = self.fix_comment_spacing(current_content)
        fix_counts["comment_spacing"] = count

        current_content, count = self.fix_f_string_placeholders(current_content)
        fix_counts["f_string_placeholders"] = count

        current_content, count = self.fix_bare_except(current_content)
        fix_counts["bare_except"] = count

        return current_content, fix_counts

    def process_file(self, file_path: Path) -> bool:
        """Process a single file with conservative fixes."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                original_content = f.read()

            # Skip if file is too large (>50KB) to be safe
            if len(original_content) > 50_000:
                return False

            # Validate original syntax
            if not self.validate_syntax(original_content):
                self.errors_encountered.append(f"Invalid syntax in {file_path}")
                return False

            # Apply fixes
            new_content, fix_counts = self.apply_safe_fixes(original_content)

            # Validate new syntax
            if not self.validate_syntax(new_content):
                self.errors_encountered.append(
                    f"Fix introduced syntax error in {file_path}"
                )
                return False

            # Only write if there were changes
            total_fixes = sum(fix_counts.values())
            if total_fixes > 0:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(new_content)

                self.fixes_applied.append(
                    {"file": str(file_path), "fixes": fix_counts, "total": total_fixes}
                )
                return True

        except Exception as e:
            self.errors_encountered.append(f"Error processing {file_path}: {e}")
            return False

        return False

    def run_conservative_fixes(self, max_files: int = 15):
        """Run conservative fixes on a subset of files."""
        # Find Python files, excluding problematic directories
        exclude_dirs = {
            ".git",
            "__pycache__",
            ".pytest_cache",
            "build",
            "dist",
            "egg-info",
            ".tox",
            ".venv",
            "venv",
            "chemml_env",
            "site",
        }

        python_files = []
        for py_file in self.workspace_root.rglob("*.py"):
            # Skip if in excluded directory
            if any(excluded in py_file.parts for excluded in exclude_dirs):
                continue
            # Skip archived files with known syntax errors
            if "archived" in py_file.parts:
                continue
            python_files.append(py_file)

        # Sort by size (smaller files first for safety)
        python_files.sort(key=lambda f: f.stat().st_size)

        # Limit to max_files
        files_to_process = python_files[:max_files]

        print(f"Processing {len(files_to_process)} files (limited to {max_files})...")

        success_count = 0
        for file_path in files_to_process:
            try:
                rel_path = file_path.relative_to(self.workspace_root)
                if self.process_file(file_path):
                    success_count += 1
                    print(f"✓ Fixed: {rel_path}")
            except Exception as e:
                print(f"- Error with {file_path}: {e}")

        # Summary
        total_fixes = sum(fix["total"] for fix in self.fixes_applied)
        print("\nSummary:")
        print(f"  Files processed: {len(files_to_process)}")
        print(f"  Files modified: {success_count}")
        print(f"  Total fixes applied: {total_fixes}")

        if self.errors_encountered:
            print(f"  Errors encountered: {len(self.errors_encountered)}")
            for error in self.errors_encountered[:3]:  # Show first 3 errors
                print(f"    - {error}")

        return success_count

def main():
    """Main function to run conservative refactoring."""
    parser = argparse.ArgumentParser(description="Conservative incremental refactoring")
    parser.add_argument(
        "--max-files", type=int, default=15, help="Maximum number of files to process"
    )
    parser.add_argument(
        "--workspace", type=str, default=None, help="Workspace root directory"
    )

    args = parser.parse_args()

    refactor = ConservativeRefactor(args.workspace)
    success_count = refactor.run_conservative_fixes(args.max_files)

    print(f"\nConservative refactoring completed. Modified {success_count} files.")
    return success_count

if __name__ == "__main__":
    main()
