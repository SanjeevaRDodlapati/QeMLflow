#!/usr/bin/env python3
"""
ChemML API Standardization Tool
Fixes common API inconsistencies including bare except clauses and parameter naming.
"""

import argparse
import ast
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

class BareExceptFixer(ast.NodeTransformer):
    """AST transformer to fix bare except clauses."""

    def __init__(self):
        self.fixes_made = []

    def visit_ExceptHandler(self, node):
        """Visit except handlers and fix bare except clauses."""
        if node.type is None:  # This is a bare except
            # Convert to Exception
            new_node = ast.ExceptHandler(
                type=ast.Name(id="Exception", ctx=ast.Load()),
                name=node.name,
                body=node.body,
                lineno=node.lineno,
                col_offset=node.col_offset,
            )
            self.fixes_made.append(f"Line {node.lineno}: Fixed bare except clause")
            return new_node
        return self.generic_visit(node)

def fix_bare_except_in_file(filepath: str) -> Tuple[bool, List[str]]:
    """Fix bare except clauses in a Python file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Parse the AST
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            return False, [f"Syntax error: {e}"]

        # Apply transformations
        fixer = BareExceptFixer()
        new_tree = fixer.visit(tree)

        if fixer.fixes_made:
            # Convert back to source code
            import astor

            new_content = astor.to_source(new_tree)

            # Write the fixed content
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(new_content)

            return True, fixer.fixes_made

        return False, []

    except Exception as e:
        return False, [f"Error processing file: {e}"]

def find_parameter_inconsistencies(filepath: str) -> List[Dict]:
    """Find parameter naming inconsistencies in a Python file."""
    inconsistencies = []

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        tree = ast.parse(content)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check for data parameter variations
                data_params = []
                for arg in node.args.args:
                    arg_name = arg.arg
                    if any(
                        pattern in arg_name.lower() for pattern in ["data", "dataset"]
                    ):
                        data_params.append(arg_name)

                if len(data_params) > 1:
                    inconsistencies.append(
                        {
                            "type": "parameter_naming",
                            "function": node.name,
                            "line": node.lineno,
                            "issue": f"Multiple data parameters: {data_params}",
                            "suggestion": "Use consistent naming: 'data' for primary dataset",
                        }
                    )

        return inconsistencies

    except Exception as e:
        return [{"error": f"Error analyzing {filepath}: {e}"}]

def find_missing_type_annotations(filepath: str) -> List[Dict]:
    """Find functions missing type annotations."""
    missing_annotations = []

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        tree = ast.parse(content)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Skip private methods and dunder methods
                if node.name.startswith("_"):
                    continue

                # Check return annotation
                if node.returns is None:
                    missing_annotations.append(
                        {
                            "type": "missing_return_annotation",
                            "function": node.name,
                            "line": node.lineno,
                            "suggestion": "Add return type annotation",
                        }
                    )

                # Check parameter annotations
                for arg in node.args.args:
                    if arg.annotation is None and arg.arg != "self":
                        missing_annotations.append(
                            {
                                "type": "missing_param_annotation",
                                "function": node.name,
                                "parameter": arg.arg,
                                "line": node.lineno,
                                "suggestion": f"Add type annotation for parameter '{arg.arg}'",
                            }
                        )

        return missing_annotations

    except Exception as e:
        return [{"error": f"Error analyzing {filepath}: {e}"}]

def main():
    """Main function to run API standardization."""
    parser = argparse.ArgumentParser(description="ChemML API Standardization Tool")
    parser.add_argument(
        "--fix-except", action="store_true", help="Fix bare except clauses"
    )
    parser.add_argument(
        "--check-params",
        action="store_true",
        help="Check parameter naming inconsistencies",
    )
    parser.add_argument(
        "--check-types", action="store_true", help="Check missing type annotations"
    )
    parser.add_argument("--target-files", nargs="+", help="Specific files to process")
    parser.add_argument("--all", action="store_true", help="Run all checks and fixes")

    args = parser.parse_args()

    if args.all:
        args.fix_except = args.check_params = args.check_types = True

    # Get target files
    if args.target_files:
        target_files = args.target_files
    else:
        # Find all Python files in src/chemml
        src_dir = Path("src/chemml")
        if src_dir.exists():
            target_files = list(src_dir.rglob("*.py"))
        else:
            print("❌ src/chemml directory not found")
            return

    print("🔧 ChemML API Standardization")
    print("=" * 40)

    total_fixes = 0

    for filepath in target_files:
        filepath = str(filepath)
        if not filepath.endswith(".py"):
            continue

        print(f"\n📁 Processing: {filepath}")

        # Fix bare except clauses
        if args.fix_except:
            fixed, fixes = fix_bare_except_in_file(filepath)
            if fixed:
                print(f"  ✅ Fixed {len(fixes)} bare except clauses")
                for fix in fixes:
                    print(f"    • {fix}")
                total_fixes += len(fixes)
            else:
                if fixes:  # Error messages
                    print(f"  ❌ Errors: {fixes}")

        # Check parameter naming
        if args.check_params:
            param_issues = find_parameter_inconsistencies(filepath)
            if param_issues:
                print(f"  ⚠️  Found {len(param_issues)} parameter naming issues")
                for issue in param_issues[:3]:  # Show first 3
                    if "error" not in issue:
                        print(
                            f"    • {issue['function']}:{issue['line']} - {issue['issue']}"
                        )

        # Check type annotations
        if args.check_types:
            type_issues = find_missing_type_annotations(filepath)
            if type_issues:
                missing_count = len(type_issues)
                print(f"  📝 Missing {missing_count} type annotations")
                # Show summary only for brevity

    print(f"\n✅ Standardization complete! Made {total_fixes} fixes.")

if __name__ == "__main__":
    main()
