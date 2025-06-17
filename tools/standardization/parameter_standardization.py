"""
Parameter Naming Standardization Script
Identifies and suggests fixes for inconsistent parameter naming patterns.
"""

import argparse
import ast
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple


class ParameterAnalyzer(ast.NodeVisitor):
    """Analyzes parameter naming patterns in Python files."""

    def __init__(self):
        self.data_parameters = defaultdict(list)
        self.model_parameters = defaultdict(list)
        self.file_parameters = defaultdict(list)
        self.type_parameters = defaultdict(list)
        self.all_parameters = []

    def visit_FunctionDef(self, node):
        """Visit function definitions and analyze parameters."""
        for arg in node.args.args:
            param_name = arg.arg
            param_info = {
                "name": param_name,
                "function": node.name,
                "line": node.lineno,
            }

            self.all_parameters.append(param_info)

            # Categorize parameters
            if self._is_data_parameter(param_name):
                self.data_parameters[param_name].append(param_info)
            elif self._is_model_parameter(param_name):
                self.model_parameters[param_name].append(param_info)
            elif self._is_file_parameter(param_name):
                self.file_parameters[param_name].append(param_info)
            elif self._is_type_parameter(param_name):
                self.type_parameters[param_name].append(param_info)

        self.generic_visit(node)

    def _is_data_parameter(self, name: str) -> bool:
        """Check if parameter is data-related."""
        data_patterns = [
            "data",
            "dataset",
            "molecular_data",
            "molecules",
            "training_data",
            "test_data",
            "validation_data",
        ]
        return any(pattern in name.lower() for pattern in data_patterns)

    def _is_model_parameter(self, name: str) -> bool:
        """Check if parameter is model-related."""
        model_patterns = ["model", "models", "estimator", "classifier", "regressor"]
        return any(pattern in name.lower() for pattern in model_patterns)

    def _is_file_parameter(self, name: str) -> bool:
        """Check if parameter is file-related."""
        file_patterns = ["file", "path", "filename", "filepath", "save_path"]
        return any(pattern in name.lower() for pattern in file_patterns)

    def _is_type_parameter(self, name: str) -> bool:
        """Check if parameter is type-related."""
        type_patterns = ["type", "kind", "method", "algorithm"]
        return any(pattern in name.lower() for pattern in type_patterns)


def analyze_parameter_consistency(filepath: str) -> Dict:
    """Analyze parameter naming consistency in a file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        tree = ast.parse(content)
        analyzer = ParameterAnalyzer()
        analyzer.visit(tree)

        return {
            "filepath": filepath,
            "data_parameters": dict(analyzer.data_parameters),
            "model_parameters": dict(analyzer.model_parameters),
            "file_parameters": dict(analyzer.file_parameters),
            "type_parameters": dict(analyzer.type_parameters),
            "total_parameters": len(analyzer.all_parameters),
        }

    except Exception as e:
        return {"filepath": filepath, "error": str(e)}


def generate_standardization_suggestions(analysis: Dict) -> List[Dict]:
    """Generate suggestions for parameter standardization."""
    suggestions = []

    # Data parameter suggestions
    data_params = analysis.get("data_parameters", {})
    if len(data_params) > 1:
        # Find most common data parameter name
        param_counts = {
            name: len(occurrences) for name, occurrences in data_params.items()
        }
        most_common = max(param_counts, key=param_counts.get)

        for param_name, occurrences in data_params.items():
            if param_name != most_common and param_name != "data":
                suggestions.append(
                    {
                        "type": "data_parameter_standardization",
                        "current": param_name,
                        "suggested": "data",
                        "occurrences": len(occurrences),
                        "locations": [
                            (occ["function"], occ["line"]) for occ in occurrences
                        ],
                    }
                )

    # File parameter suggestions
    file_params = analysis.get("file_parameters", {})
    for param_name, occurrences in file_params.items():
        if (
            param_name in ["filename", "file_path", "path"]
            and "filepath" in file_params
        ):
            continue  # Skip if filepath already exists
        elif param_name not in ["filepath", "save_path", "config_path"]:
            suggested = "filepath"
            if "save" in param_name.lower():
                suggested = "save_path"
            elif "config" in param_name.lower():
                suggested = "config_path"

            suggestions.append(
                {
                    "type": "file_parameter_standardization",
                    "current": param_name,
                    "suggested": suggested,
                    "occurrences": len(occurrences),
                    "locations": [
                        (occ["function"], occ["line"]) for occ in occurrences
                    ],
                }
            )

    return suggestions


def create_standardization_report(src_dir: str) -> Dict:
    """Create a comprehensive standardization report."""
    src_path = Path(src_dir)
    if not src_path.exists():
        return {"error": f"Directory {src_dir} does not exist"}

    python_files = list(src_path.rglob("*.py"))
    all_analyses = []
    all_suggestions = []

    for filepath in python_files:
        analysis = analyze_parameter_consistency(str(filepath))
        if "error" not in analysis:
            all_analyses.append(analysis)
            suggestions = generate_standardization_suggestions(analysis)
            all_suggestions.extend(suggestions)

    # Summary statistics
    total_files = len(all_analyses)
    total_data_params = sum(len(a.get("data_parameters", {})) for a in all_analyses)
    total_suggestions = len(all_suggestions)

    # Group suggestions by type
    suggestions_by_type = defaultdict(list)
    for suggestion in all_suggestions:
        suggestions_by_type[suggestion["type"]].append(suggestion)

    return {
        "summary": {
            "files_analyzed": total_files,
            "total_data_parameters": total_data_params,
            "total_suggestions": total_suggestions,
        },
        "suggestions_by_type": dict(suggestions_by_type),
        "detailed_analyses": all_analyses,
    }


def print_standardization_report(report: Dict):
    """Print a formatted standardization report."""
    if "error" in report:
        print(f"❌ Error: {report['error']}")
        return

    summary = report["summary"]
    suggestions = report["suggestions_by_type"]

    print("🔧 Parameter Naming Standardization Report")
    print("=" * 50)
    print(f"📊 Files analyzed: {summary['files_analyzed']}")
    print(f"📊 Total data parameters found: {summary['total_data_parameters']}")
    print(f"📊 Standardization suggestions: {summary['total_suggestions']}")

    if summary["total_suggestions"] == 0:
        print("✅ No standardization issues found!")
        return

    print("\n💡 Standardization Suggestions:")
    print("-" * 30)

    for suggestion_type, suggestion_list in suggestions.items():
        print(f"\n{suggestion_type.replace('_', ' ').title()}:")
        for suggestion in suggestion_list[:5]:  # Show first 5
            current = suggestion["current"]
            suggested = suggestion["suggested"]
            count = suggestion["occurrences"]
            print(f"  • {current} → {suggested} ({count} occurrences)")

    print(f"\n📝 Total suggestions: {summary['total_suggestions']}")
    if summary["total_suggestions"] > 10:
        print("   (Showing first few suggestions per category)")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Parameter Naming Standardization")
    parser.add_argument(
        "--src-dir", default="src/qemlflow", help="Source directory to analyze"
    )
    parser.add_argument(
        "--save-report", action="store_true", help="Save detailed report to JSON file"
    )

    args = parser.parse_args()

    report = create_standardization_report(args.src_dir)
    print_standardization_report(report)

    if args.save_report:
        import json

        with open("parameter_standardization_report.json", "w") as f:
            json.dump(report, f, indent=2)
        print("\n💾 Detailed report saved to parameter_standardization_report.json")


if __name__ == "__main__":
    main()
