"""
Syntax Error Fixer - Targeted fixes for critical syntax errors
"""

import ast
import os
import re
import sys
from pathlib import Path


class SyntaxErrorFixer:
    """Fix critical syntax errors that prevent parsing."""

    def __init__(self, workspace_root=None):
        self.workspace_root = Path(workspace_root) if workspace_root else Path.cwd()
        self.fixes_applied = []

    def fix_broken_try_except(self, content: str) -> str:
        """Fix broken try-except blocks."""
        # Pattern: try block without except or finally
        lines = content.split("\n")
        fixed_lines = []
        i = 0

        while i < len(lines):
            line = lines[i]

            # Look for try: statements
            if re.match(r"\s*try\s*:", line):
                # Check if there's a proper except/finally block following
                try_indent = len(line) - len(line.lstrip())
                found_except_or_finally = False
                j = i + 1

                # Skip the try block content
                while j < len(lines):
                    next_line = lines[j]
                    if not next_line.strip():  # Empty line
                        j += 1
                        continue

                    next_indent = len(next_line) - len(next_line.lstrip())

                    # If we're back to the same indent level or less
                    if next_indent <= try_indent:
                        # Check if it's except/finally
                        if re.match(r"\s*(except|finally)", next_line):
                            found_except_or_finally = True
                        break
                    j += 1

                # If no except/finally found, add a basic except block
                if not found_except_or_finally:
                    fixed_lines.append(line)
                    # Add try block content
                    k = i + 1
                    while k < len(lines) and k < j:
                        fixed_lines.append(lines[k])
                        k += 1
                    # Add except block
                    indent_str = " " * (try_indent + 4)
                    fixed_lines.append(" " * try_indent + "except Exception:")
                    fixed_lines.append(indent_str + "pass")
                    i = j - 1
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)
            i += 1

        return "\n".join(fixed_lines)

    def fix_indentation_errors(self, content: str) -> str:
        """Fix basic indentation errors."""
        lines = content.split("\n")
        fixed_lines = []

        for i, line in enumerate(lines):
            # Fix mixed tabs and spaces (convert tabs to 4 spaces)
            line = line.expandtabs(4)

            # Fix unindent errors - look for lines that don't match expected indentation
            if line.strip():
                # Simple heuristic: if line starts with 'src_path' but should be '_src_path'
                line = line.replace("src_path =", "_src_path =")

            fixed_lines.append(line)

        return "\n".join(fixed_lines)

    def fix_variable_errors(self, content: str) -> str:
        """Fix undefined variable errors."""
        # Fix _src_path vs src_path inconsistency
        content = re.sub(r"(\s+)src_path(\s*=)", r"\1_src_path\2", content)
        content = re.sub(
            r"sys\.path\.insert\(0,\s*str\(src_path\)\)",
            r"sys.path.insert(0, str(_src_path))",
            content,
        )

        return content

    def fix_file(self, file_path: Path) -> bool:
        """Fix syntax errors in a specific file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                original_content = f.read()

            # Apply fixes
            content = original_content
            content = self.fix_variable_errors(content)
            content = self.fix_indentation_errors(content)
            content = self.fix_broken_try_except(content)

            # Test if the file can be parsed
            try:
                ast.parse(content)
                # If successful, write the fixed content
                if content != original_content:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(content)
                    self.fixes_applied.append(str(file_path))
                    return True
            except SyntaxError:
                # If still has syntax errors, don't write
                print(f"Warning: Could not fully fix syntax errors in {file_path}")
                return False

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return False

        return False

    def fix_syntax_errors(self, files_with_errors=None):
        """Fix syntax errors in files."""
        if files_with_errors is None:
            # Default list of files with E999 errors from linter output
            files_with_errors = [
                "examples/archived/advanced_integration_features_demo.py",
                "examples/archived/boltz_integration_demo.py",
                "examples/archived/standalone_advanced_features_demo.py",
                "examples/integrations/framework/registry_demo.py",
                "scripts/utilities/setup_wandb_integration.py",
                "src/chemml/core/pipeline.py",
            ]

        print("Fixing critical syntax errors...")
        fixed_count = 0

        for file_path in files_with_errors:
            full_path = self.workspace_root / file_path
            if full_path.exists():
                if self.fix_file(full_path):
                    fixed_count += 1
                    print(f"Fixed: {file_path}")
                else:
                    print(f"Failed to fix: {file_path}")
            else:
                print(f"File not found: {file_path}")

        print(f"\nFixed {fixed_count} files with syntax errors")
        return fixed_count


def main():
    """Main function to run syntax error fixes."""
    import argparse

    parser = argparse.ArgumentParser(description="Fix critical syntax errors")
    parser.add_argument(
        "--workspace", type=str, default=None, help="Workspace root directory"
    )
    parser.add_argument("--files", nargs="*", help="Specific files to fix")

    args = parser.parse_args()

    fixer = SyntaxErrorFixer(args.workspace)
    fixed_count = fixer.fix_syntax_errors(args.files)

    print(f"\nSyntax Error Fixer completed. Fixed {fixed_count} files.")
    return fixed_count


if __name__ == "__main__":
    main()
