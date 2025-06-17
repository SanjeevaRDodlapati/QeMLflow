#!/usr/bin/env python3
"""
Targeted Issue Fixer - Focus on high-impact issues
"""

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


class TargetedFixer:
    """Fix high-impact issues that improve code quality most."""

    def __init__(self, workspace_root: str = None):
        self.workspace_root = Path(workspace_root) if workspace_root else Path.cwd()
        self.fixes_applied = 0

    def get_current_issues(self) -> Dict:
        """Get current issues from the linter."""
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "tools/linting/comprehensive_linter.py",
                    "--format",
                    "json",
                    "--quiet",
                ],
                capture_output=True,
                text=True,
                cwd=self.workspace_root,
            )

            if result.returncode == 0:
                return json.loads(result.stdout)
            else:
                print(f"Error running linter: {result.stderr}")
                return {}
        except Exception as e:
            print(f"Error getting issues: {e}")
            return {}

    def fix_f_string_issues(self, issues: List[Dict]) -> int:
        """Fix F541 f-string placeholder issues."""
        fixes = 0
        f541_issues = [issue for issue in issues if issue.get("rule_code") == "F541"]

        # Group by file
        files_to_fix = {}
        for issue in f541_issues:
            file_path = issue["file_path"]
            if file_path not in files_to_fix:
                files_to_fix[file_path] = []
            files_to_fix[file_path].append(issue)

        for file_path, file_issues in files_to_fix.items():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                lines = content.split("\n")
                file_fixes = 0

                for issue in file_issues:
                    line_num = issue["line_number"] - 1  # Convert to 0-based
                    if 0 <= line_num < len(lines):
                        line = lines[line_num]
                        # Simple fix: remove f prefix from f-strings without placeholders
                        if 'f"' in line and "{" not in line:
                            lines[line_num] = line.replace('f"', '"')
                            file_fixes += 1
                        elif "f'" in line and "{" not in line:
                            lines[line_num] = line.replace("f'", "'")
                            file_fixes += 1

                if file_fixes > 0:
                    new_content = "\n".join(lines)
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(new_content)
                    fixes += file_fixes
                    print(f"Fixed {file_fixes} f-string issues in {file_path}")

            except Exception as e:
                print(f"Error fixing f-strings in {file_path}: {e}")

        return fixes

    def fix_unused_variables(self, issues: List[Dict]) -> int:
        """Fix F841 unused variable issues by prefixing with underscore."""
        fixes = 0
        f841_issues = [issue for issue in issues if issue.get("rule_code") == "F841"]

        # Group by file
        files_to_fix = {}
        for issue in f841_issues:
            file_path = issue["file_path"]
            if file_path not in files_to_fix:
                files_to_fix[file_path] = []
            files_to_fix[file_path].append(issue)

        for file_path, file_issues in files_to_fix.items():
            if len(file_issues) > 10:  # Skip files with too many issues
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                lines = content.split("\n")
                file_fixes = 0

                for issue in file_issues:
                    line_num = issue["line_number"] - 1  # Convert to 0-based
                    if 0 <= line_num < len(lines):
                        line = lines[line_num]
                        # Extract variable name from message like "local variable 'var' is assigned to but never used"
                        match = re.search(
                            r"local variable '(\w+)' is assigned", issue["message"]
                        )
                        if match:
                            var_name = match.group(1)
                            # Only prefix if not already prefixed and line contains assignment
                            if not var_name.startswith("_") and f"{var_name} =" in line:
                                lines[line_num] = line.replace(
                                    f"{var_name} =", f"_{var_name} =", 1
                                )
                                file_fixes += 1

                if file_fixes > 0:
                    new_content = "\n".join(lines)
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(new_content)
                    fixes += file_fixes
                    print(f"Fixed {file_fixes} unused variable issues in {file_path}")

            except Exception as e:
                print(f"Error fixing unused variables in {file_path}: {e}")

        return fixes

    def run_targeted_fixes(self):
        """Run targeted fixes for high-impact issues."""
        print("Getting current linting issues...")
        data = self.get_current_issues()

        if not data or "issues" not in data:
            print("No issues data available")
            return 0

        issues = data["issues"]
        print(f"Found {len(issues)} total issues")

        # Count issues by type
        issue_counts = {}
        for issue in issues:
            rule_code = issue.get("rule_code", "unknown")
            issue_counts[rule_code] = issue_counts.get(rule_code, 0) + 1

        print("Top issues:")
        for rule_code, count in sorted(
            issue_counts.items(), key=lambda x: x[1], reverse=True
        )[:10]:
            print(f"  {rule_code}: {count}")

        total_fixes = 0

        # Fix F541 (f-string placeholders) - high impact, low risk
        print("\nFixing f-string placeholder issues...")
        fixes = self.fix_f_string_issues(issues)
        total_fixes += fixes
        print(f"Applied {fixes} f-string fixes")

        # Fix F841 (unused variables) - medium impact, low risk
        print("\nFixing unused variable issues...")
        fixes = self.fix_unused_variables(issues)
        total_fixes += fixes
        print(f"Applied {fixes} unused variable fixes")

        print(f"\nTotal fixes applied: {total_fixes}")
        return total_fixes


def main():
    """Main function."""
    fixer = TargetedFixer()
    fixes = fixer.run_targeted_fixes()

    if fixes > 0:
        print(f"\nTargeted fixes completed. Applied {fixes} fixes.")
        print("Run the comprehensive linter to see the updated status.")
    else:
        print("No fixes were applied.")


if __name__ == "__main__":
    main()
