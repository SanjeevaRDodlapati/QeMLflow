#!/usr/bin/env python3
"""
Type Annotation Enhancement Tool
Adds missing type annotations to ChemML functions.
"""

import argparse
import ast
import re
from pathlib import Path
from typing import Dict, List, Optional, Set


class TypeAnnotationAnalyzer(ast.NodeVisitor):
    """Analyzes and suggests type annotations for functions."""

    def __init__(self):
        self.missing_annotations = []
        self.function_info = []

    def visit_FunctionDef(self, node):
        """Analyze function definitions for missing type annotations."""
        # Skip private methods and special methods
        if node.name.startswith("_"):
            self.generic_visit(node)
            return

        function_info = {
            "name": node.name,
            "line": node.lineno,
            "has_return_annotation": node.returns is not None,
            "parameters": [],
            "missing_param_annotations": [],
            "missing_return_annotation": node.returns is None,
        }

        # Analyze parameters
        for arg in node.args.args:
            param_info = {
                "name": arg.arg,
                "has_annotation": arg.annotation is not None,
                "suggested_type": self._suggest_type_from_name(arg.arg),
            }
            function_info["parameters"].append(param_info)

            # Skip 'self' parameter
            if arg.arg == "self":
                continue

            if arg.annotation is None:
                function_info["missing_param_annotations"].append(arg.arg)
                self.missing_annotations.append(
                    {
                        "type": "parameter",
                        "function": node.name,
                        "parameter": arg.arg,
                        "line": node.lineno,
                        "suggested_type": param_info["suggested_type"],
                    }
                )

        # Check return annotation
        if node.returns is None:
            suggested_return = self._suggest_return_type(node)
            function_info["suggested_return_type"] = suggested_return
            self.missing_annotations.append(
                {
                    "type": "return",
                    "function": node.name,
                    "line": node.lineno,
                    "suggested_type": suggested_return,
                }
            )

        self.function_info.append(function_info)
        self.generic_visit(node)

    def _suggest_type_from_name(self, param_name: str) -> str:
        """Suggest type annotation based on parameter name."""
        name_lower = param_name.lower()

        # Common patterns
        if name_lower in ["data", "X", "features"]:
            return "Union[pd.DataFrame, np.ndarray]"
        elif name_lower in ["y", "target", "labels"]:
            return "Union[pd.Series, np.ndarray]"
        elif "smiles" in name_lower:
            return "Union[str, List[str]]"
        elif "molecules" in name_lower or "molecular" in name_lower:
            return "List[Mol]"  # RDKit Mol objects
        elif "model" in name_lower:
            return "Any"  # Could be various ML models
        elif "config" in name_lower:
            return "Dict[str, Any]"
        elif name_lower in ["filepath", "filename", "path"]:
            return "str"
        elif name_lower in ["save_path", "output_dir"]:
            return "Optional[str]"
        elif "threshold" in name_lower or "alpha" in name_lower:
            return "float"
        elif "n_" in name_lower or "num_" in name_lower:
            return "int"
        elif name_lower.endswith("_list") or name_lower.endswith("s"):
            return "List[Any]"
        elif name_lower in ["verbose", "debug", "enable"]:
            return "bool"
        else:
            return "Any"

    def _suggest_return_type(self, node: ast.FunctionDef) -> str:
        """Suggest return type based on function analysis."""
        function_name = node.name.lower()

        # Analyze return statements
        return_types = set()
        for stmt in ast.walk(node):
            if isinstance(stmt, ast.Return) and stmt.value:
                if isinstance(stmt.value, ast.Dict):
                    return_types.add("Dict[str, Any]")
                elif isinstance(stmt.value, ast.List):
                    return_types.add("List[Any]")
                elif isinstance(stmt.value, ast.Tuple):
                    return_types.add("Tuple[Any, ...]")
                elif isinstance(stmt.value, (ast.Constant, ast.Num, ast.Str)):
                    if isinstance(
                        (
                            stmt.value.value
                            if hasattr(stmt.value, "value")
                            else stmt.value.n
                        ),
                        bool,
                    ):
                        return_types.add("bool")
                    elif isinstance(
                        (
                            stmt.value.value
                            if hasattr(stmt.value, "value")
                            else stmt.value.n
                        ),
                        int,
                    ):
                        return_types.add("int")
                    elif isinstance(
                        (
                            stmt.value.value
                            if hasattr(stmt.value, "value")
                            else stmt.value.s
                        ),
                        str,
                    ):
                        return_types.add("str")
                    else:
                        return_types.add("float")

        if return_types:
            if len(return_types) == 1:
                return list(return_types)[0]
            else:
                return f"Union[{', '.join(sorted(return_types))}]"

        # Pattern-based suggestions
        if any(
            pattern in function_name
            for pattern in ["predict", "transform", "fit_transform"]
        ):
            return "Union[pd.DataFrame, np.ndarray]"
        elif any(pattern in function_name for pattern in ["fit", "train"]):
            return "Self"
        elif any(pattern in function_name for pattern in ["save", "export"]):
            return "None"
        elif any(pattern in function_name for pattern in ["load", "read"]):
            return "Any"
        elif any(pattern in function_name for pattern in ["calculate", "compute"]):
            return "Union[float, np.ndarray]"
        elif any(pattern in function_name for pattern in ["get_", "extract_"]):
            return "Any"
        elif any(pattern in function_name for pattern in ["is_", "has_", "check_"]):
            return "bool"
        else:
            return "Any"


def analyze_type_annotations(filepath: str) -> Dict:
    """Analyze type annotations in a file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        tree = ast.parse(content)
        analyzer = TypeAnnotationAnalyzer()
        analyzer.visit(tree)

        return {
            "filepath": filepath,
            "missing_annotations": analyzer.missing_annotations,
            "function_info": analyzer.function_info,
            "annotation_coverage": calculate_annotation_coverage(
                analyzer.function_info
            ),
        }

    except Exception as e:
        return {"filepath": filepath, "error": str(e)}


def calculate_annotation_coverage(function_info: List[Dict]) -> Dict:
    """Calculate type annotation coverage statistics."""
    if not function_info:
        return {"parameter_coverage": 0, "return_coverage": 0, "overall_coverage": 0}

    total_params = sum(len(f["parameters"]) for f in function_info)
    annotated_params = sum(
        sum(1 for p in f["parameters"] if p["has_annotation"]) for f in function_info
    )

    total_functions = len(function_info)
    annotated_returns = sum(1 for f in function_info if f["has_return_annotation"])

    param_coverage = (annotated_params / total_params * 100) if total_params > 0 else 0
    return_coverage = (
        (annotated_returns / total_functions * 100) if total_functions > 0 else 0
    )
    overall_coverage = (
        (
            (annotated_params + annotated_returns)
            / (total_params + total_functions)
            * 100
        )
        if (total_params + total_functions) > 0
        else 0
    )

    return {
        "parameter_coverage": round(param_coverage, 1),
        "return_coverage": round(return_coverage, 1),
        "overall_coverage": round(overall_coverage, 1),
        "total_functions": total_functions,
        "total_parameters": total_params,
    }


def generate_type_annotation_suggestions(analysis: Dict) -> List[str]:
    """Generate code suggestions for adding type annotations."""
    suggestions = []
    missing = analysis.get("missing_annotations", [])

    # Group by function
    functions = {}
    for item in missing:
        func_name = item["function"]
        if func_name not in functions:
            functions[func_name] = {
                "parameters": [],
                "return": None,
                "line": item["line"],
            }

        if item["type"] == "parameter":
            functions[func_name]["parameters"].append(
                f"{item['parameter']}: {item['suggested_type']}"
            )
        elif item["type"] == "return":
            functions[func_name]["return"] = item["suggested_type"]

    for func_name, info in functions.items():
        suggestion = f"# Function: {func_name} (line {info['line']})\n"
        if info["parameters"]:
            suggestion += (
                f"# Add parameter annotations: {', '.join(info['parameters'])}\n"
            )
        if info["return"]:
            suggestion += f"# Add return annotation: -> {info['return']}\n"
        suggestions.append(suggestion)

    return suggestions


def create_type_annotation_report(src_dir: str) -> Dict:
    """Create a comprehensive type annotation report."""
    src_path = Path(src_dir)
    if not src_path.exists():
        return {"error": f"Directory {src_dir} does not exist"}

    python_files = list(src_path.rglob("*.py"))
    all_analyses = []
    total_coverage = {
        "param_total": 0,
        "param_annotated": 0,
        "func_total": 0,
        "func_annotated": 0,
    }

    for filepath in python_files:
        analysis = analyze_type_annotations(str(filepath))
        if "error" not in analysis:
            all_analyses.append(analysis)
            coverage = analysis["annotation_coverage"]
            total_coverage["param_total"] += coverage.get("total_parameters", 0)
            total_coverage["func_total"] += coverage.get("total_functions", 0)

    # Calculate overall coverage
    overall_param_coverage = 0
    overall_return_coverage = 0

    total_param_weighted = 0
    total_return_weighted = 0

    for analysis in all_analyses:
        if "annotation_coverage" in analysis:
            coverage = analysis["annotation_coverage"]
            total_params = coverage.get("total_parameters", 0)
            total_funcs = coverage.get("total_functions", 0)

            if total_params > 0:
                total_param_weighted += coverage["parameter_coverage"] * total_params
            if total_funcs > 0:
                total_return_weighted += coverage["return_coverage"] * total_funcs

    if total_coverage["param_total"] > 0:
        overall_param_coverage = total_param_weighted / total_coverage["param_total"]
    if total_coverage["func_total"] > 0:
        overall_return_coverage = total_return_weighted / total_coverage["func_total"]

    return {
        "summary": {
            "files_analyzed": len(all_analyses),
            "total_functions": total_coverage["func_total"],
            "total_parameters": total_coverage["param_total"],
            "overall_parameter_coverage": round(overall_param_coverage, 1),
            "overall_return_coverage": round(overall_return_coverage, 1),
        },
        "detailed_analyses": all_analyses,
    }


def print_type_annotation_report(report: Dict):
    """Print a formatted type annotation report."""
    if "error" in report:
        print(f"❌ Error: {report['error']}")
        return

    summary = report["summary"]

    print("📝 Type Annotation Coverage Report")
    print("=" * 40)
    print(f"📊 Files analyzed: {summary['files_analyzed']}")
    print(f"📊 Total functions: {summary['total_functions']}")
    print(f"📊 Total parameters: {summary['total_parameters']}")
    print(f"📊 Parameter annotation coverage: {summary['overall_parameter_coverage']}%")
    print(f"📊 Return annotation coverage: {summary['overall_return_coverage']}%")

    # Show files with lowest coverage
    analyses = report["detailed_analyses"]
    low_coverage_files = [
        a for a in analyses if a["annotation_coverage"]["overall_coverage"] < 50
    ]

    if low_coverage_files:
        print(
            f"\n⚠️  Files with low annotation coverage ({len(low_coverage_files)} files):"
        )
        for analysis in sorted(
            low_coverage_files,
            key=lambda x: x["annotation_coverage"]["overall_coverage"],
        )[:5]:
            filepath = Path(analysis["filepath"]).name
            coverage = analysis["annotation_coverage"]["overall_coverage"]
            print(f"  • {filepath}: {coverage}%")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Type Annotation Analysis")
    parser.add_argument(
        "--src-dir", default="src/chemml", help="Source directory to analyze"
    )
    parser.add_argument(
        "--save-report", action="store_true", help="Save detailed report to JSON file"
    )
    parser.add_argument(
        "--show-suggestions",
        action="store_true",
        help="Show annotation suggestions for low-coverage files",
    )

    args = parser.parse_args()

    report = create_type_annotation_report(args.src_dir)
    print_type_annotation_report(report)

    if args.save_report:
        import json

        with open("type_annotation_report.json", "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n💾 Detailed report saved to type_annotation_report.json")


if __name__ == "__main__":
    main()
