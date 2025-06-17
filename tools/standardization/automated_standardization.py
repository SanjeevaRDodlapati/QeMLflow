"""
Automated Parameter Standardization Tool
Automatically fixes parameter naming inconsistencies in QeMLflow codebase.
"""

import argparse
import ast
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple


class ParameterStandardizer(ast.NodeTransformer):
    """AST transformer that standardizes parameter names."""

    # Standard naming conventions
    PARAMETER_MAPPINGS = {
        # Data parameters
        "patient_data": "data",
        "stratum_data": "data",
        "materials_data": "data",
        "environmental_data": "data",
        "reaction_data": "data",
        "molecular_data": "molecules",
        "training_data": "X_train",
        "test_data": "X_test",
        "validation_data": "X_val",
        # File parameters
        "filename": "filepath",
        "file_path": "filepath",
        "log_file": "filepath",
        "output_file": "filepath",
        "input_file": "filepath",
        "base_path": "base_filepath",
        # Model parameters
        "model_type": "model_type",  # Already correct
        "estimator": "model",
        "classifier": "model",
        "regressor": "model",
        # Type parameters
        "algorithm": "algorithm_type",
        "method": "method_type",
    }

    def __init__(self):
        self.changes_made = []
        self.current_function = None

    def visit_FunctionDef(self, node):
        """Visit function definitions and standardize parameter names."""
        self.current_function = node.name

        # Create new argument list with standardized names
        new_args = []
        changes_in_function = []

        for arg in node.args.args:
            old_name = arg.arg
            new_name = self.PARAMETER_MAPPINGS.get(old_name, old_name)

            if new_name != old_name:
                # Create new argument with standardized name
                new_arg = ast.arg(arg=new_name, annotation=arg.annotation)
                changes_in_function.append((old_name, new_name))
                self.changes_made.append(
                    {
                        "function": self.current_function,
                        "line": node.lineno,
                        "old_param": old_name,
                        "new_param": new_name,
                    }
                )
            else:
                new_arg = arg

            new_args.append(new_arg)

        # Update the function's arguments
        node.args.args = new_args

        # Update parameter references in function body if names changed
        if changes_in_function:
            node = self._update_parameter_references(node, dict(changes_in_function))

        return self.generic_visit(node)

    def _update_parameter_references(self, func_node, name_mappings):
        """Update parameter references in function body."""

        class ParameterReferenceUpdater(ast.NodeTransformer):
            def visit_Name(self, node):
                if isinstance(node.ctx, ast.Load) and node.id in name_mappings:
                    node.id = name_mappings[node.id]
                return node

        updater = ParameterReferenceUpdater()
        return updater.visit(func_node)


def standardize_parameters_in_file(
    filepath: str, backup: bool = True
) -> Tuple[bool, List[Dict]]:
    """Standardize parameter names in a Python file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Create backup if requested
        if backup:
            backup_path = (
                f"{filepath}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            shutil.copy2(filepath, backup_path)

        # Parse and transform AST
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            return False, [{"error": f"Syntax error: {e}"}]

        standardizer = ParameterStandardizer()
        new_tree = standardizer.visit(tree)

        if standardizer.changes_made:
            # Convert back to source code
            import astor

            new_content = astor.to_source(new_tree)

            # Write the standardized content
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(new_content)

            return True, standardizer.changes_made

        return False, []

    except Exception as e:
        return False, [{"error": f"Error processing file: {e}"}]


def create_type_annotation_suggestions(filepath: str) -> List[str]:
    """Create type annotation suggestions for a file."""
    suggestions = []

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        tree = ast.parse(content)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Skip private methods
                if node.name.startswith("_"):
                    continue

                # Check for missing return annotation
                if node.returns is None:
                    suggested_return = _suggest_return_type(node.name)
                    suggestions.append(f"def {node.name}(...) -> {suggested_return}:")

                # Check for missing parameter annotations
                for arg in node.args.args:
                    if arg.annotation is None and arg.arg != "self":
                        suggested_type = _suggest_parameter_type(arg.arg)
                        suggestions.append(f"  {arg.arg}: {suggested_type}")

    except Exception:
        pass

    return suggestions


def _suggest_return_type(function_name: str) -> str:
    """Suggest return type based on function name."""
    name_lower = function_name.lower()

    if any(pattern in name_lower for pattern in ["predict", "transform"]):
        return "Union[pd.DataFrame, np.ndarray]"
    elif any(pattern in name_lower for pattern in ["fit", "train"]):
        return "Self"
    elif any(pattern in name_lower for pattern in ["save", "export", "write"]):
        return "None"
    elif any(pattern in name_lower for pattern in ["load", "read"]):
        return "Any"
    elif any(pattern in name_lower for pattern in ["calculate", "compute"]):
        return "Union[float, np.ndarray]"
    elif any(pattern in name_lower for pattern in ["is_", "has_", "check_"]):
        return "bool"
    else:
        return "Any"


def _suggest_parameter_type(param_name: str) -> str:
    """Suggest parameter type based on name."""
    name_lower = param_name.lower()

    if name_lower in ["data", "X", "features"]:
        return "Union[pd.DataFrame, np.ndarray]"
    elif name_lower in ["y", "target", "labels"]:
        return "Union[pd.Series, np.ndarray]"
    elif "smiles" in name_lower:
        return "Union[str, List[str]]"
    elif "molecules" in name_lower:
        return "List[Mol]"
    elif "model" in name_lower:
        return "Any"
    elif "config" in name_lower:
        return "Dict[str, Any]"
    elif name_lower in ["filepath", "filename", "path"]:
        return "str"
    elif "threshold" in name_lower or "alpha" in name_lower:
        return "float"
    elif "n_" in name_lower or "num_" in name_lower:
        return "int"
    elif name_lower in ["verbose", "debug", "enable"]:
        return "bool"
    else:
        return "Any"


def process_priority_files(src_dir: str = "src/qemlflow") -> Dict:
    """Process the highest priority files for standardization."""
    priority_files = [
        "src/qemlflow/core/data.py",
        "src/qemlflow/core/models.py",
        "src/qemlflow/core/evaluation.py",
        "src/qemlflow/research/drug_discovery.py",
        "src/qemlflow/research/clinical_research.py",
        "src/qemlflow/integrations/deepchem_integration.py",
    ]

    results = {
        "files_processed": 0,
        "total_changes": 0,
        "changes_by_file": {},
        "errors": [],
    }

    for filepath in priority_files:
        if Path(filepath).exists():
            success, changes = standardize_parameters_in_file(filepath)
            if success:
                results["files_processed"] += 1
                results["total_changes"] += len(changes)
                results["changes_by_file"][filepath] = changes
                print(f"✅ {filepath}: {len(changes)} parameters standardized")
            else:
                if changes:  # Error messages
                    results["errors"].extend(changes)
                    print(f"❌ {filepath}: {changes}")
        else:
            print(f"⚠️  {filepath}: File not found")

    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Automated Parameter Standardization")
    parser.add_argument("--target-files", nargs="+", help="Specific files to process")
    parser.add_argument(
        "--priority-only", action="store_true", help="Process only high-priority files"
    )
    parser.add_argument(
        "--no-backup", action="store_true", help="Skip creating backup files"
    )
    parser.add_argument(
        "--add-types", action="store_true", help="Also suggest type annotations"
    )

    args = parser.parse_args()

    print("🔧 Automated Parameter Standardization")
    print("=" * 45)

    if args.priority_only:
        results = process_priority_files()
        print(f"\n✅ Processed {results['files_processed']} priority files")
        print(f"📊 Total changes: {results['total_changes']}")

        if results["errors"]:
            print(f"❌ Errors: {len(results['errors'])}")

    elif args.target_files:
        total_changes = 0
        for filepath in args.target_files:
            success, changes = standardize_parameters_in_file(
                filepath, backup=not args.no_backup
            )
            if success:
                total_changes += len(changes)
                print(f"✅ {filepath}: {len(changes)} changes")

                if args.add_types:
                    suggestions = create_type_annotation_suggestions(filepath)
                    if suggestions:
                        print(f"  💡 Type annotation suggestions: {len(suggestions)}")
            else:
                print(f"❌ {filepath}: Failed")

        print(f"\n✅ Total changes: {total_changes}")

    else:
        print("Please specify --priority-only or --target-files")


if __name__ == "__main__":
    main()
