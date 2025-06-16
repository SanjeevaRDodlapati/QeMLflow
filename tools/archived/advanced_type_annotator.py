#!/usr/bin/env python3
"""
Advanced Type Annotation Enhancement Tool
Automatically adds type annotations to ChemML functions based on context analysis.
"""

import argparse
import ast
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union


class SmartTypeAnnotator(ast.NodeTransformer):
    """Intelligently adds type annotations to function definitions."""

    TYPE_MAPPINGS = {
        # Common data types
        "data": "Union[pd.DataFrame, np.ndarray]",
        "X": "Union[pd.DataFrame, np.ndarray]",
        "y": "Union[pd.Series, np.ndarray]",
        "features": "Union[pd.DataFrame, np.ndarray]",
        "target": "Union[pd.Series, np.ndarray]",
        # Molecular data
        "smiles": "Union[str, List[str]]",
        "molecules": "List[Mol]",
        "molecular_data": "List[Mol]",
        "mol": "Mol",
        # File paths
        "filepath": "str",
        "path": "str",
        "filename": "str",
        "save_path": "Optional[str]",
        "output_dir": "Optional[str]",
        # Configuration
        "config": "Dict[str, Any]",
        "params": "Dict[str, Any]",
        "settings": "Dict[str, Any]",
        # Models
        "model": "Any",
        "estimator": "Any",
        "classifier": "Any",
        "regressor": "Any",
        # Numeric types
        "threshold": "float",
        "alpha": "float",
        "learning_rate": "float",
        "n_estimators": "int",
        "max_depth": "int",
        "random_state": "int",
        # Boolean flags
        "verbose": "bool",
        "debug": "bool",
        "normalize": "bool",
        "fit_intercept": "bool",
    }

    RETURN_TYPE_PATTERNS = {
        # Prediction/transformation methods
        ("predict", "transform", "fit_transform"): "Union[pd.DataFrame, np.ndarray]",
        # Training methods
        ("fit", "train"): "Self",
        # I/O methods
        ("save", "export", "write"): "None",
        ("load", "read", "import"): "Any",
        # Calculation methods
        ("calculate", "compute", "score"): "Union[float, np.ndarray]",
        # Getter methods
        ("get_", "extract_", "obtain_"): "Any",
        # Boolean checks
        ("is_", "has_", "can_", "should_", "check_"): "bool",
        # Visualization
        ("plot", "show", "display", "visualize"): "Optional[Any]",
    }

    def __init__(self):
        self.changes_made = []
        self.import_additions = set()

    def visit_FunctionDef(self, node):
        """Add type annotations to function definitions."""
        changes_in_function = []

        # Skip private methods and already annotated functions
        if node.name.startswith("_"):
            return self.generic_visit(node)

        # Add parameter annotations
        for arg in node.args.args:
            if arg.arg == "self":
                continue

            if arg.annotation is None:
                suggested_type = self._get_parameter_type(arg.arg)
                if suggested_type:
                    arg.annotation = ast.parse(suggested_type, mode="eval").body
                    changes_in_function.append(f"Added {arg.arg}: {suggested_type}")
                    self._add_required_imports(suggested_type)

        # Add return annotation
        if node.returns is None:
            suggested_return = self._get_return_type(node.name, node)
            if suggested_return:
                node.returns = ast.parse(suggested_return, mode="eval").body
                changes_in_function.append(f"Added return: {suggested_return}")
                self._add_required_imports(suggested_return)

        if changes_in_function:
            self.changes_made.append(
                {
                    "function": node.name,
                    "line": node.lineno,
                    "changes": changes_in_function,
                }
            )

        return self.generic_visit(node)

    def _get_parameter_type(self, param_name: str) -> Optional[str]:
        """Get suggested type for parameter based on name."""
        # Direct mapping
        if param_name in self.TYPE_MAPPINGS:
            return self.TYPE_MAPPINGS[param_name]

        # Pattern matching
        param_lower = param_name.lower()

        if any(pattern in param_lower for pattern in ["data", "df", "dataframe"]):
            return "pd.DataFrame"
        elif any(pattern in param_lower for pattern in ["array", "matrix"]):
            return "np.ndarray"
        elif param_lower.endswith("_list") or param_lower.endswith("s"):
            return "List[Any]"
        elif "dict" in param_lower or "mapping" in param_lower:
            return "Dict[str, Any]"
        elif any(pattern in param_lower for pattern in ["count", "num_", "n_", "size"]):
            return "int"
        elif any(
            pattern in param_lower for pattern in ["rate", "ratio", "score", "weight"]
        ):
            return "float"
        elif any(pattern in param_lower for pattern in ["flag", "enable", "disable"]):
            return "bool"
        elif "file" in param_lower or "path" in param_lower:
            return "str"

        return "Any"  # Default fallback

    def _get_return_type(self, func_name: str, node: ast.FunctionDef) -> Optional[str]:
        """Get suggested return type based on function name and body analysis."""
        func_lower = func_name.lower()

        # Check patterns
        for patterns, return_type in self.RETURN_TYPE_PATTERNS.items():
            if any(pattern in func_lower for pattern in patterns):
                return return_type

        # Analyze return statements
        return_analysis = self._analyze_return_statements(node)
        if return_analysis:
            return return_analysis

        return "Any"  # Default

    def _analyze_return_statements(self, node: ast.FunctionDef) -> Optional[str]:
        """Analyze return statements to infer return type."""
        return_types = set()

        for stmt in ast.walk(node):
            if isinstance(stmt, ast.Return) and stmt.value:
                if isinstance(stmt.value, ast.Dict):
                    return_types.add("Dict[str, Any]")
                elif isinstance(stmt.value, ast.List):
                    return_types.add("List[Any]")
                elif isinstance(stmt.value, ast.Tuple):
                    return_types.add("Tuple[Any, ...]")
                elif isinstance(stmt.value, ast.Constant):
                    if isinstance(stmt.value.value, bool):
                        return_types.add("bool")
                    elif isinstance(stmt.value.value, int):
                        return_types.add("int")
                    elif isinstance(stmt.value.value, float):
                        return_types.add("float")
                    elif isinstance(stmt.value.value, str):
                        return_types.add("str")
                elif isinstance(stmt.value, ast.Name) and stmt.value.id == "None":
                    return_types.add("None")

        if len(return_types) == 1:
            return list(return_types)[0]
        elif len(return_types) > 1:
            return f"Union[{', '.join(sorted(return_types))}]"

        return None

    def _add_required_imports(self, type_hint: str):
        """Track required imports for type hints."""
        if "pd." in type_hint:
            self.import_additions.add("import pandas as pd")
        if "np." in type_hint:
            self.import_additions.add("import numpy as np")
        if (
            "List" in type_hint
            or "Dict" in type_hint
            or "Union" in type_hint
            or "Optional" in type_hint
        ):
            self.import_additions.add(
                "from typing import List, Dict, Union, Optional, Any"
            )
        if "Mol" in type_hint:
            self.import_additions.add("from rdkit import Chem")
            self.import_additions.add("Mol = Chem.Mol")


def add_type_annotations_to_file(filepath: str, backup: bool = True) -> Dict:
    """Add type annotations to a Python file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Create backup if requested
        if backup:
            backup_path = f"{filepath}.typing_backup"
            with open(backup_path, "w", encoding="utf-8") as f:
                f.write(content)

        # Parse and annotate
        tree = ast.parse(content)
        annotator = SmartTypeAnnotator()
        new_tree = annotator.visit(tree)

        if annotator.changes_made or annotator.import_additions:
            # Add required imports at the top
            if annotator.import_additions:
                import_lines = "\n".join(sorted(annotator.import_additions)) + "\n\n"
                # Find where to insert imports (after docstring)
                lines = content.split("\n")
                insert_index = 0

                # Skip module docstring
                if lines and (lines[0].startswith('"""') or lines[0].startswith("'''")):
                    for i, line in enumerate(lines[1:], 1):
                        if line.strip().endswith('"""') or line.strip().endswith("'''"):
                            insert_index = i + 1
                            break

                lines.insert(insert_index, import_lines.rstrip())
                content = "\n".join(lines)

            # Convert AST back to code
            import astor

            new_content = astor.to_source(new_tree)

            # Write annotated content
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(new_content)

            return {
                "success": True,
                "changes": annotator.changes_made,
                "imports_added": list(annotator.import_additions),
            }

        return {"success": False, "message": "No annotations needed"}

    except Exception as e:
        return {"success": False, "error": str(e)}


def main():
    """Main function for type annotation enhancement."""
    parser = argparse.ArgumentParser(description="Advanced Type Annotation Enhancement")
    parser.add_argument("--target-files", nargs="+", help="Specific files to annotate")
    parser.add_argument(
        "--low-coverage-only",
        action="store_true",
        help="Only process files with <50% annotation coverage",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be changed"
    )

    args = parser.parse_args()

    print("📝 Advanced Type Annotation Enhancement")
    print("=" * 45)

    if args.target_files:
        target_files = args.target_files
    else:
        # Find Python files in src/chemml
        src_dir = Path("src/chemml")
        target_files = list(src_dir.rglob("*.py"))

        if args.low_coverage_only:
            # Filter to low-coverage files (would need previous analysis)
            low_coverage_files = [
                "src/chemml/core/utils.py",
                "src/chemml/research/generative.py",
                "src/chemml/research/quantum.py",
                "src/chemml/integrations/pipeline.py",
            ]
            target_files = [f for f in target_files if str(f) in low_coverage_files]

    total_changes = 0
    total_functions = 0

    for filepath in target_files[:5]:  # Limit to first 5 for demonstration
        filepath = str(filepath)
        if not filepath.endswith(".py"):
            continue

        print(f"\n📁 Processing: {filepath}")

        if args.dry_run:
            # Analyze without making changes
            try:
                with open(filepath, "r") as f:
                    content = f.read()
                tree = ast.parse(content)
                annotator = SmartTypeAnnotator()
                annotator.visit(tree)

                if annotator.changes_made:
                    print(
                        f"  💡 Would add annotations to {len(annotator.changes_made)} functions"
                    )
                    for change in annotator.changes_made[:3]:
                        print(
                            f"    • {change['function']}: {len(change['changes'])} annotations"
                        )
                else:
                    print("  ✅ No annotations needed")
            except Exception as e:
                print(f"  ❌ Error: {e}")
        else:
            result = add_type_annotations_to_file(filepath, backup=True)
            if result["success"]:
                changes = result.get("changes", [])
                total_changes += len(changes)
                total_functions += len(changes)
                print(f"  ✅ Added annotations to {len(changes)} functions")

                if result.get("imports_added"):
                    print(f"  📦 Added imports: {len(result['imports_added'])}")
            else:
                if "error" in result:
                    print(f"  ❌ Error: {result['error']}")
                else:
                    print(f"  ✅ {result.get('message', 'No changes needed')}")

    if not args.dry_run:
        print("\n✅ Enhancement complete!")
        print(f"📊 Total functions annotated: {total_functions}")
        print(f"📊 Total annotations added: {total_changes}")


if __name__ == "__main__":
    main()
