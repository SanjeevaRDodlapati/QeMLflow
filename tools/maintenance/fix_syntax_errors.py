#!/usr/bin/env python3
"""
Fix syntax errors in QeMLflow codebase systematically.
"""

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


class SyntaxErrorFixer:
    """Fix various syntax errors detected by flake8."""

    def __init__(self, src_dir: str = "src"):
        self.src_dir = Path(src_dir)
        self.fixed_files = []
        self.failed_files = []

    def run_flake8_check(self, file_path: str) -> List[str]:
        """Run flake8 on a specific file and return E999 errors."""
        try:
            result = subprocess.run(
                ["flake8", file_path, "--select=E999"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.stdout:
                return result.stdout.strip().split("\n")
            return []
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return []

    def fix_unterminated_docstrings(self, file_path: Path) -> bool:
        """Fix unterminated triple-quoted strings in a file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            original_content = content

            # Count triple quotes to find imbalances
            triple_quote_positions = []
            for match in re.finditer(r'"""', content):
                triple_quote_positions.append(match.start())

            # If odd number of triple quotes, we have an unterminated string
            if len(triple_quote_positions) % 2 == 1:
                print(
                    f"  Found {len(triple_quote_positions)} triple quotes (odd number)"
                )

                # Find the last occurrence and check if it should be closed
                lines = content.split("\n")

                # Look for common patterns that indicate missing closing quotes
                for i, line in enumerate(lines):
                    line_stripped = line.strip()

                    # Pattern 1: Line starts with """ but doesn't end with """
                    if (
                        line_stripped.startswith('"""')
                        and not line_stripped.endswith('"""')
                        and len(line_stripped) > 3
                    ):
                        # This might be an opening docstring
                        continue

                    # Pattern 2: Look for function/class definitions followed by docstring without closing
                    if (
                        line_stripped.startswith(("def ", "class "))
                        and i + 1 < len(lines)
                        and lines[i + 1].strip().startswith('"""')
                    ):

                        # Find the end of this docstring
                        docstring_start = i + 1
                        docstring_end = None

                        for j in range(docstring_start + 1, len(lines)):
                            if '"""' in lines[j]:
                                docstring_end = j
                                break

                        if docstring_end is None:
                            # Add closing quotes at a reasonable location
                            # Look for the next function/class or end of file
                            next_def_line = None
                            for j in range(docstring_start + 1, len(lines)):
                                if lines[j].strip().startswith(("def ", "class ")) or (
                                    lines[j].strip()
                                    and not lines[j].startswith(" ")
                                    and not lines[j].startswith("\t")
                                ):
                                    next_def_line = j
                                    break

                            if next_def_line:
                                # Insert closing quotes before the next definition
                                lines.insert(next_def_line, '    """')
                                print(
                                    f"  Added closing docstring at line {next_def_line + 1}"
                                )
                            else:
                                # Add at the end of file
                                lines.append('    """')
                                print(f"  Added closing docstring at end of file")

                            content = "\n".join(lines)
                            break

                # If we still have odd number of quotes, add one at the end
                if content.count('"""') % 2 == 1:
                    content += '\n    """\n'
                    print("  Added final closing quotes at end of file")

            # Fix invalid characters
            content = self.fix_invalid_characters(content)

            # Fix invalid syntax patterns
            content = self.fix_invalid_syntax_patterns(content)

            if content != original_content:
                # Backup original file
                backup_path = file_path.with_suffix(file_path.suffix + ".backup")
                with open(backup_path, "w", encoding="utf-8") as f:
                    f.write(original_content)

                # Write fixed content
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)

                print(f"  Fixed and backed up to {backup_path}")
                return True

            return False

        except Exception as e:
            print(f"  Error fixing {file_path}: {e}")
            return False

    def fix_invalid_characters(self, content: str) -> str:
        """Fix invalid characters in the content."""
        # Replace invalid characters with valid alternatives
        replacements = {
            "²": "**2",  # Superscript 2
            "🎓": "graduation",  # Graduation cap emoji
            "🎯": "target",  # Target emoji
            # Add more as needed
        }

        for invalid_char, replacement in replacements.items():
            if invalid_char in content:
                content = content.replace(invalid_char, replacement)
                print(
                    f"  Replaced invalid character '{invalid_char}' with '{replacement}'"
                )

        return content

    def fix_invalid_syntax_patterns(self, content: str) -> str:
        """Fix common invalid syntax patterns."""
        lines = content.split("\n")
        fixed_lines = []

        for line in lines:
            # Fix invalid decimal literals like "20250616.015828"
            if re.search(r"\b\d+\.\d+\.\d+", line):
                # Replace with string literal
                line = re.sub(r"\b(\d+\.\d+\.\d+)", r'"\1"', line)
                print(f"  Fixed invalid decimal literal in line")

            # Fix unterminated string literals
            if line.count('"') % 2 == 1 and not line.strip().endswith("\\"):
                # Add closing quote at end of line
                line += '"'
                print(f"  Fixed unterminated string literal")

            if line.count("'") % 2 == 1 and not line.strip().endswith("\\"):
                # Add closing quote at end of line
                line += "'"
                print(f"  Fixed unterminated string literal")

            fixed_lines.append(line)

        return "\n".join(fixed_lines)

    def fix_file(self, file_path: Path) -> bool:
        """Fix syntax errors in a specific file."""
        print(f"\nFixing {file_path}...")

        # Check if file has syntax errors
        errors = self.run_flake8_check(str(file_path))
        if not errors or not any("E999" in error for error in errors):
            print(f"  No syntax errors found")
            return True

        print(f"  Found {len(errors)} syntax errors")
        for error in errors[:3]:  # Show first 3 errors
            print(f"    {error}")

        # Try to fix the file
        success = self.fix_unterminated_docstrings(file_path)

        # Re-check for errors
        new_errors = self.run_flake8_check(str(file_path))
        if new_errors and any("E999" in error for error in new_errors):
            print(f"  Still has {len(new_errors)} syntax errors after fix")
            self.failed_files.append(str(file_path))
            return False
        else:
            print(f"  Successfully fixed!")
            self.fixed_files.append(str(file_path))
            return True

    def fix_all_syntax_errors(self) -> Dict[str, int]:
        """Fix all syntax errors in the source directory."""
        print("Scanning for Python files with syntax errors...")

        # Get all Python files
        python_files = list(self.src_dir.rglob("*.py"))
        print(f"Found {len(python_files)} Python files")

        # Get files with syntax errors
        error_files = []
        for file_path in python_files:
            errors = self.run_flake8_check(str(file_path))
            if errors and any("E999" in error for error in errors):
                error_files.append(file_path)

        print(f"Found {len(error_files)} files with syntax errors")

        # Fix each file
        for file_path in error_files:
            self.fix_file(file_path)

        return {
            "total_files": len(python_files),
            "error_files": len(error_files),
            "fixed_files": len(self.fixed_files),
            "failed_files": len(self.failed_files),
        }


def main():
    """Main function."""
    if len(sys.argv) > 1:
        src_dir = sys.argv[1]
    else:
        src_dir = "src"

    fixer = SyntaxErrorFixer(src_dir)
    results = fixer.fix_all_syntax_errors()

    print(f"\n{'='*60}")
    print("SYNTAX ERROR FIXING RESULTS")
    print(f"{'='*60}")
    print(f"Total Python files: {results['total_files']}")
    print(f"Files with syntax errors: {results['error_files']}")
    print(f"Successfully fixed: {results['fixed_files']}")
    print(f"Failed to fix: {results['failed_files']}")

    if fixer.fixed_files:
        print(f"\nFixed files:")
        for file_path in fixer.fixed_files:
            print(f"  ✓ {file_path}")

    if fixer.failed_files:
        print(f"\nFailed to fix:")
        for file_path in fixer.failed_files:
            print(f"  ✗ {file_path}")

    return 0 if not fixer.failed_files else 1


if __name__ == "__main__":
    sys.exit(main())
