ChemML API Consistency Analysis
==============================

Analyze and standardize API patterns across the ChemML codebase.
"""

import ast
import inspect
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

class APIAnalyzer:
    """Analyze API consistency across ChemML modules."""

    def __init__(self, source_dir: str = "src/chemml"):
        self.source_dir = Path(source_dir)
        self.api_patterns = defaultdict(list)
        self.inconsistencies = []
        self.recommendations = []

    def analyze_api_consistency(self) -> Dict[str, Any]:
        """Run comprehensive API consistency analysis."""
        print("🔍 ChemML API Consistency Analysis")
        print("=" * 50)

        # Analyze different API aspects
        self.analyze_method_naming_patterns()
        self.analyze_parameter_patterns()
        self.analyze_return_type_patterns()
        self.analyze_class_interfaces()
        self.analyze_error_handling_patterns()

        # Generate recommendations
        self.generate_recommendations()

        return self.generate_report()

    def analyze_method_naming_patterns(self):
        """Analyze method naming conventions."""
        print("\n📝 Method Naming Patterns")
        print("-" * 30)

        naming_patterns = defaultdict(list)

        for py_file in self.source_dir.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue

            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    tree = ast.parse(f.read())

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        method_name = node.name

                        # Categorize naming patterns
                        if method_name.startswith("_"):
                            naming_patterns["private"].append(method_name)
                        elif method_name.startswith("get_"):
                            naming_patterns["getter"].append(method_name)
                        elif method_name.startswith("set_"):
                            naming_patterns["setter"].append(method_name)
                        elif method_name.startswith("is_") or method_name.startswith(
                            "has_"
                        ):
                            naming_patterns["boolean"].append(method_name)
                        elif method_name.startswith("calculate_"):
                            naming_patterns["calculation"].append(method_name)
                        elif method_name.startswith("process_"):
                            naming_patterns["processing"].append(method_name)
                        elif method_name.startswith("load_"):
                            naming_patterns["loading"].append(method_name)
                        elif method_name.startswith("save_"):
                            naming_patterns["saving"].append(method_name)
                        else:
                            naming_patterns["other"].append(method_name)

            except Exception as e:
                continue

        # Analyze patterns
        for pattern, methods in naming_patterns.items():
            count = len(methods)
            print(f"  {pattern}: {count} methods")
            if count > 5:  # Show examples for common patterns
                examples = list(set(methods))[:5]
                print(f"    Examples: {', '.join(examples)}")

        self.api_patterns["naming"] = naming_patterns

        # Check for inconsistencies
        inconsistent_names = []

        # Check for mixed naming in similar functions
        calculation_methods = naming_patterns["calculation"]
        processing_methods = naming_patterns["processing"]

        if len(calculation_methods) > 0 and len(processing_methods) > 0:
            # Look for semantic overlap
            calc_stems = {
                name.replace("calculate_", "") for name in calculation_methods
            }
            proc_stems = {name.replace("process_", "") for name in processing_methods}

            overlap = calc_stems.intersection(proc_stems)
            if overlap:
                inconsistent_names.extend(overlap)

        if inconsistent_names:
            self.inconsistencies.append(
                {
                    "type": "naming_inconsistency",
                    "description": "Mixed naming patterns for similar functionality",
                    "examples": inconsistent_names[:5],
                }
            )

    def analyze_parameter_patterns(self):
        """Analyze parameter naming and type patterns."""
        print("\n🔧 Parameter Patterns")
        print("-" * 30)

        param_patterns = defaultdict(Counter)

        for py_file in self.source_dir.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue

            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    tree = ast.parse(f.read())

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        for arg in node.args.args:
                            param_name = arg.arg

                            # Skip 'self' and 'cls'
                            if param_name in ["self", "cls"]:
                                continue

                            # Categorize parameter patterns
                            if "data" in param_name.lower():
                                param_patterns["data_params"][param_name] += 1
                            elif (
                                "file" in param_name.lower()
                                or "path" in param_name.lower()
                            ):
                                param_patterns["file_params"][param_name] += 1
                            elif "config" in param_name.lower():
                                param_patterns["config_params"][param_name] += 1
                            elif "model" in param_name.lower():
                                param_patterns["model_params"][param_name] += 1
                            elif param_name.endswith("_type") or param_name.endswith(
                                "_method"
                            ):
                                param_patterns["type_params"][param_name] += 1
                            else:
                                param_patterns["other_params"][param_name] += 1

            except Exception:
                continue

        # Report common parameter names
        for category, counter in param_patterns.items():
            if len(counter) > 0:
                top_params = counter.most_common(5)
                print(
                    f"  {category}: {', '.join([f'{name}({count})' for name, count in top_params])}"
                )

        self.api_patterns["parameters"] = param_patterns

        # Check for parameter inconsistencies
        data_params = param_patterns["data_params"]
        if len(data_params) > 3:
            common_variations = ["data", "dataset", "df", "dataframe", "input_data"]
            actual_variations = [
                name
                for name in data_params.keys()
                if any(var in name.lower() for var in common_variations)
            ]

            if len(set(actual_variations)) > 2:
                self.inconsistencies.append(
                    {
                        "type": "parameter_inconsistency",
                        "description": "Inconsistent naming for data parameters",
                        "examples": actual_variations[:5],
                    }
                )

    def analyze_return_type_patterns(self):
        """Analyze return type annotations and patterns."""
        print("\n↩️  Return Type Patterns")
        print("-" * 30)

        return_patterns = defaultdict(list)

        for py_file in self.source_dir.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue

            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        if node.returns:
                            # Extract return type annotation
                            return_annotation = (
                                ast.unparse(node.returns)
                                if hasattr(ast, "unparse")
                                else str(node.returns)
                            )
                            return_patterns["annotated"].append(return_annotation)
                        else:
                            return_patterns["unannotated"].append(node.name)

            except Exception:
                continue

        # Report return type patterns
        print(f"  Annotated functions: {len(return_patterns['annotated'])}")
        print(f"  Unannotated functions: {len(return_patterns['unannotated'])}")

        if return_patterns["annotated"]:
            # Count common return types
            type_counter = Counter(return_patterns["annotated"])
            common_types = type_counter.most_common(5)
            print(
                f"  Common return types: {', '.join([f'{t}({c})' for t, c in common_types])}"
            )

        self.api_patterns["return_types"] = return_patterns

        # Check for return type consistency
        annotation_ratio = len(return_patterns["annotated"]) / (
            len(return_patterns["annotated"]) + len(return_patterns["unannotated"])
        )
        if annotation_ratio < 0.7:  # Less than 70% annotated
            self.inconsistencies.append(
                {
                    "type": "return_type_inconsistency",
                    "description": f"Low return type annotation coverage: {annotation_ratio:.1%}",
                    "examples": return_patterns["unannotated"][:5],
                }
            )

    def analyze_class_interfaces(self):
        """Analyze class interface patterns."""
        print("\n🏗️  Class Interface Patterns")
        print("-" * 30)

        class_patterns = defaultdict(list)

        for py_file in self.source_dir.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue

            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    tree = ast.parse(f.read())

                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        class_name = node.name
                        methods = [
                            n.name for n in node.body if isinstance(n, ast.FunctionDef)
                        ]

                        # Categorize classes by interface patterns
                        if "__init__" in methods:
                            class_patterns["with_init"].append(class_name)

                        if any(method.startswith("fit") for method in methods):
                            class_patterns["ml_interface"].append(class_name)

                        if any(
                            method in ["predict", "transform", "fit_transform"]
                            for method in methods
                        ):
                            class_patterns["sklearn_like"].append(class_name)

                        if any(
                            method in ["__enter__", "__exit__"] for method in methods
                        ):
                            class_patterns["context_manager"].append(class_name)

                        if any(method.startswith("_") for method in methods):
                            class_patterns["with_private"].append(class_name)

                        class_patterns["all_classes"].append(
                            {
                                "name": class_name,
                                "methods": methods,
                                "method_count": len(methods),
                            }
                        )

            except Exception:
                continue

        # Report class patterns
        for pattern, classes in class_patterns.items():
            if pattern != "all_classes":
                print(f"  {pattern}: {len(classes)} classes")

        # Analyze method consistency in similar classes
        ml_classes = class_patterns["ml_interface"]
        if len(ml_classes) > 3:
            # Check for consistent ML interface patterns
            print(f"  ML-like classes found: {len(ml_classes)}")

        self.api_patterns["classes"] = class_patterns

    def analyze_error_handling_patterns(self):
        """Analyze error handling consistency."""
        print("\n⚠️  Error Handling Patterns")
        print("-" * 30)

        error_patterns = defaultdict(list)

        for py_file in self.source_dir.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue

            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.Try):
                        # Check exception handling patterns
                        for handler in node.handlers:
                            if handler.type:
                                exc_type = (
                                    ast.unparse(handler.type)
                                    if hasattr(ast, "unparse")
                                    else str(handler.type)
                                )
                                error_patterns["handled_exceptions"].append(exc_type)
                            else:
                                error_patterns["bare_except"].append(str(py_file))

                    elif isinstance(node, ast.Raise):
                        if node.exc:
                            exc_type = (
                                ast.unparse(node.exc)
                                if hasattr(ast, "unparse")
                                else str(node.exc)
                            )
                            error_patterns["raised_exceptions"].append(exc_type)

            except Exception:
                continue

        # Report error handling patterns
        handled_count = len(error_patterns["handled_exceptions"])
        bare_except_count = len(error_patterns["bare_except"])
        raised_count = len(error_patterns["raised_exceptions"])

        print(f"  Exception handlers: {handled_count}")
        print(f"  Bare except clauses: {bare_except_count}")
        print(f"  Explicit raises: {raised_count}")

        if handled_count > 0:
            exc_counter = Counter(error_patterns["handled_exceptions"])
            common_exceptions = exc_counter.most_common(3)
            print(
                f"  Common exceptions: {', '.join([f'{e}({c})' for e, c in common_exceptions])}"
            )

        self.api_patterns["errors"] = error_patterns

        # Check for error handling consistency
        if bare_except_count > 0:
            self.inconsistencies.append(
                {
                    "type": "error_handling_inconsistency",
                    "description": f"Found {bare_except_count} bare except clauses",
                    "examples": error_patterns["bare_except"][:3],
                }
            )

    def generate_recommendations(self):
        """Generate specific recommendations for API improvements."""
        print("\n💡 Generating Recommendations")
        print("-" * 30)

        # Naming convention recommendations
        naming_patterns = self.api_patterns["naming"]
        if len(naming_patterns["other"]) > len(naming_patterns["getter"]) + len(
            naming_patterns["setter"]
        ):
            self.recommendations.append(
                {
                    "category": "naming",
                    "priority": "medium",
                    "title": "Standardize method naming conventions",
                    "description": "Many methods use inconsistent naming. Consider adopting standard prefixes like get_, set_, calculate_, process_.",
                    "action": "Create naming convention guide and refactor inconsistent methods",
                }
            )

        # Parameter naming recommendations
        param_patterns = self.api_patterns["parameters"]
        data_param_count = len(param_patterns["data_params"])
        if data_param_count > 5:
            self.recommendations.append(
                {
                    "category": "parameters",
                    "priority": "high",
                    "title": "Standardize data parameter naming",
                    "description": "Multiple naming patterns for data parameters reduce API consistency.",
                    "action": 'Standardize on "data" or "dataset" for primary data parameters',
                }
            )

        # Return type recommendations
        return_patterns = self.api_patterns["return_types"]
        annotation_ratio = len(return_patterns["annotated"]) / (
            len(return_patterns["annotated"]) + len(return_patterns["unannotated"])
        )
        if annotation_ratio < 0.8:
            self.recommendations.append(
                {
                    "category": "type_hints",
                    "priority": "high",
                    "title": "Improve type annotation coverage",
                    "description": f"Only {annotation_ratio:.1%} of functions have return type annotations.",
                    "action": "Add type hints to all public methods for better IDE support and documentation",
                }
            )

        # Class interface recommendations
        class_patterns = self.api_patterns["classes"]
        ml_classes = class_patterns["ml_interface"]
        sklearn_classes = class_patterns["sklearn_like"]

        if len(ml_classes) > len(sklearn_classes):
            self.recommendations.append(
                {
                    "category": "interfaces",
                    "priority": "medium",
                    "title": "Adopt sklearn-like interfaces",
                    "description": "Some ML classes don't follow sklearn conventions (fit/predict/transform).",
                    "action": "Standardize ML classes to use sklearn-compatible interfaces",
                }
            )

        # Error handling recommendations
        error_patterns = self.api_patterns["errors"]
        if len(error_patterns["bare_except"]) > 0:
            self.recommendations.append(
                {
                    "category": "error_handling",
                    "priority": "high",
                    "title": "Replace bare except clauses",
                    "description": "Bare except clauses can hide important errors.",
                    "action": "Replace with specific exception types or Exception base class",
                }
            )

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive API analysis report."""
        print("\n" + "=" * 50)
        print("📊 API Consistency Report Summary")
        print("=" * 50)

        # Summary statistics
        total_inconsistencies = len(self.inconsistencies)
        high_priority_recs = len(
            [r for r in self.recommendations if r["priority"] == "high"]
        )
        medium_priority_recs = len(
            [r for r in self.recommendations if r["priority"] == "medium"]
        )

        print(f"Total inconsistencies found: {total_inconsistencies}")
        print(f"High priority recommendations: {high_priority_recs}")
        print(f"Medium priority recommendations: {medium_priority_recs}")

        # Show top recommendations
        if self.recommendations:
            print("\n🎯 Top Priority Actions:")
            high_priority = [r for r in self.recommendations if r["priority"] == "high"]
            for i, rec in enumerate(high_priority[:3], 1):
                print(f"  {i}. {rec['title']}")
                print(f"     {rec['description']}")

        # Show critical inconsistencies
        if self.inconsistencies:
            print("\n⚠️  Critical Issues:")
            for issue in self.inconsistencies[:3]:
                print(f"  • {issue['description']}")
                if "examples" in issue:
                    print(
                        f"    Examples: {', '.join(str(e) for e in issue['examples'][:3])}"
                    )

        print("\n✅ Analysis complete! See recommendations for improvement actions.")

        return {
            "patterns": dict(self.api_patterns),
            "inconsistencies": self.inconsistencies,
            "recommendations": self.recommendations,
            "summary": {
                "total_inconsistencies": total_inconsistencies,
                "high_priority_recommendations": high_priority_recs,
                "medium_priority_recommendations": medium_priority_recs,
            },
        }

    def save_report(self, output_file: str = "api_consistency_report.json"):
        """Save the analysis report to file."""
        report = self.generate_report()
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n💾 Report saved to {output_file}")

if __name__ == "__main__":
    analyzer = APIAnalyzer()
    report = analyzer.analyze_api_consistency()
    analyzer.save_report()
