#!/usr/bin/env python3
"""
SAFE syntax error fixer for QeMLflow.
Conservative approach with extensive safety checks and backups.
"""

import ast
import os
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


def backup_file(file_path):
    """Create a backup of the file before modification."""
    backup_path = f"{file_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(file_path, backup_path)
    return backup_path


def check_python_syntax(file_path):
    """Check if a Python file has valid syntax."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        ast.parse(content)
        return True, None
    except SyntaxError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Other error: {e}"


def get_syntax_error_details(file_path):
    """Get detailed syntax error information using flake8."""
    try:
        result = subprocess.run(
            ["flake8", file_path, "--select=E999", "--format=%(row)d:%(col)d:%(text)s"],
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def fix_unterminated_docstring_conservative(file_path):
    """Conservative fix for unterminated docstrings with safety checks."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        original_content = content

        # Only fix if we can clearly identify the issue
        error_details = get_syntax_error_details(file_path)

        if "unterminated triple-quoted string literal" in error_details:
            lines = content.split("\n")

            # Find lines with opening triple quotes
            quote_positions = []
            for i, line in enumerate(lines):
                if '"""' in line:
                    # Count quotes in this line
                    quote_count = line.count('"""')
                    quote_positions.append((i, quote_count))

            # If we have an odd total number of triple quotes, add one at the end
            total_quotes = sum(count for _, count in quote_positions)
            if total_quotes % 2 == 1:
                # Add closing quote at the end of the file
                content = content.rstrip() + '\n"""\n'

        # Fix obvious character encoding issues (conservative)
        if "invalid character" in error_details:
            content = content.replace("²", "**2")
            content = content.replace("🎯", "target")
            content = content.replace("🎓", "graduation")

        # Fix missing opening docstring (very conservative)
        if (
            content.strip()
            and not content.startswith('"""')
            and not content.startswith("#")
        ):
            first_line = content.split("\n")[0]
            if (
                first_line.strip().endswith("for QeMLflow tests.")
                or first_line.strip().endswith("and workflows.")
                or first_line.strip().endswith("module.")
            ):
                content = '"""\n' + content

        if content != original_content:
            # Validate the fix before writing
            try:
                ast.parse(content)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                return True
            except SyntaxError:
                # If our fix creates new syntax errors, don't apply it
                return False

        return False

    except Exception as e:
        print(f"Error in conservative fix for {file_path}: {e}")
        return False


def fix_invalid_decimal_literals(file_path):
    """Fix invalid decimal literals like 1.2.3."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        original_content = content

        # Only fix if flake8 specifically reports this issue
        error_details = get_syntax_error_details(file_path)
        if "invalid decimal literal" in error_details:
            # Replace patterns like 1.2.3 with 1_2_3 (valid Python identifier)
            content = re.sub(r"(\d+)\.(\d+)\.(\d+)", r"\1_\2_\3", content)

        if content != original_content:
            # Validate the fix
            try:
                ast.parse(content)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                return True
            except SyntaxError:
                return False

        return False

    except Exception as e:
        print(f"Error fixing decimal literals in {file_path}: {e}")
        return False


def safe_syntax_fix():
    """Perform safe syntax fixes with backups and validation."""
    print("🛡️  SAFE SYNTAX ERROR FIXER")
    print("=" * 50)
    print("Features:")
    print("• Creates backups before modification")
    print("• Conservative pattern matching")
    print("• Validates syntax after each fix")
    print("• Rolls back if fixes create new errors")
    print()

    # Get files with syntax errors
    try:
        result = subprocess.run(
            ["flake8", "src/", "--select=E999", "--format=%(path)s"],
            capture_output=True,
            text=True,
        )

        syntax_error_files = []
        for line in result.stdout.strip().split("\n"):
            if line.startswith("src/") and line not in syntax_error_files:
                syntax_error_files.append(line)

    except Exception as e:
        print(f"Error getting syntax error files: {e}")
        return

    if not syntax_error_files:
        print("✅ No syntax errors found!")
        return

    print(f"Found {len(syntax_error_files)} files with syntax errors")
    print()

    fixed_count = 0
    failed_count = 0
    backups_created = []

    for file_path in syntax_error_files:
        print(f"Processing: {file_path}")

        # Check current syntax
        is_valid, error = check_python_syntax(file_path)
        if is_valid:
            print("  ✅ Already valid")
            continue

        print(f"  ❌ Error: {(error or 'Unknown')[:80]}...")

        # Create backup
        try:
            backup_path = backup_file(file_path)
            backups_created.append((file_path, backup_path))
            print(f"  💾 Backup created: {backup_path}")
        except Exception as e:
            print(f"  ❌ Failed to create backup: {e}")
            failed_count += 1
            continue

        # Try conservative fixes
        fixed = False

        # Fix 1: Unterminated docstrings
        if fix_unterminated_docstring_conservative(file_path):
            is_valid_after, _ = check_python_syntax(file_path)
            if is_valid_after:
                print("  ✅ Fixed unterminated docstring")
                fixed = True

        # Fix 2: Invalid decimal literals
        if not fixed and fix_invalid_decimal_literals(file_path):
            is_valid_after, _ = check_python_syntax(file_path)
            if is_valid_after:
                print("  ✅ Fixed invalid decimal literal")
                fixed = True

        if fixed:
            fixed_count += 1
        else:
            # Restore from backup if no fix worked
            try:
                shutil.copy2(backup_path, file_path)
                print("  🔄 Restored from backup (no safe fix found)")
            except Exception as e:
                print(f"  ❌ Failed to restore backup: {e}")
            failed_count += 1

        print()

    print("🎯 SUMMARY:")
    print(f"  • Files processed: {len(syntax_error_files)}")
    print(f"  • Successfully fixed: {fixed_count}")
    print(f"  • Need manual attention: {failed_count}")
    print(f"  • Success rate: {fixed_count/len(syntax_error_files)*100:.1f}%")
    print(f"  • Backups created: {len(backups_created)}")

    if backups_created:
        print()
        print("💾 Backup files created:")
        for original, backup in backups_created:
            print(f"  • {backup}")

    if failed_count > 0:
        print()
        print("📝 Files needing manual attention:")
        for file_path in syntax_error_files:
            is_valid, error = check_python_syntax(file_path)
            if not is_valid:
                print(f"  • {file_path}: {(error or 'Unknown')[:60]}...")


if __name__ == "__main__":
    safe_syntax_fix()
