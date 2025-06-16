#!/usr/bin/env python3
"""
ChemML Unified Development Tools
===============================

Consolidated development utilities for performance optimization,
type annotation, and code standardization.

This tool combines functionality from:
- performance_optimizer.py
- advanced_import_optimizer.py
- ultra_fast_optimizer.py
- advanced_type_annotator.py
- parameter_standardization.py

Usage:
    python tools/development/unified_optimizer.py --help
    python tools/development/unified_optimizer.py --optimize-imports
    python tools/development/unified_optimizer.py --analyze-performance
    python tools/development/unified_optimizer.py --standardize-code
"""

import argparse
import ast
import importlib
import json
import os
import sys
import time
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

class PerformanceProfiler:
    """Profile and optimize performance bottlenecks."""

    def __init__(self):
        self.import_times = {}
        self.failed_imports = []
        self.recommendations = []

    def measure_time(self, func):
        """Decorator to measure function execution time."""

        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            return result, execution_time

        return wrapper

    def profile_imports(self, modules: List[str]) -> Dict[str, float]:
        """Profile import times for given modules."""
        results = {}

        for module in modules:
            start_time = time.time()
            try:
                importlib.import_module(module)
                end_time = time.time()
                results[module] = end_time - start_time
            except ImportError as e:
                self.failed_imports.append((module, str(e)))
                results[module] = -1  # Indicate failure

        return results

    def analyze_chemml_performance(self) -> Dict[str, Any]:
        """Analyze ChemML performance characteristics."""
        chemml_modules = [
            "chemml",
            "chemml.core",
            "chemml.integrations",
            "chemml.integrations.core",
            "chemml.integrations.adapters",
            "chemml.preprocessing",
            "chemml.models",
        ]

        import_times = self.profile_imports(chemml_modules)

        # Analyze results
        slow_imports = {k: v for k, v in import_times.items() if v > 0.1}
        failed_imports = {k: v for k, v in import_times.items() if v == -1}

        return {
            "import_times": import_times,
            "slow_imports": slow_imports,
            "failed_imports": failed_imports,
            "total_time": sum(v for v in import_times.values() if v > 0),
            "recommendations": self._generate_performance_recommendations(import_times),
        }

    def _generate_performance_recommendations(
        self, import_times: Dict[str, float]
    ) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []

        slow_modules = [k for k, v in import_times.items() if v > 0.1]
        if slow_modules:
            recommendations.append(
                f"Consider lazy loading for slow modules: {', '.join(slow_modules)}"
            )

        if sum(v for v in import_times.values() if v > 0) > 1.0:
            recommendations.append(
                "Total import time > 1s. Consider import optimization."
            )

        return recommendations

class ImportOptimizer:
    """Optimize import patterns and reduce startup time."""

    def __init__(self):
        self.import_graph = {}
        self.unused_imports = []
        self.circular_imports = []

    def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze imports in a Python file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)
            imports = self._extract_imports(tree)

            return {
                "file": str(file_path),
                "imports": imports,
                "import_count": len(imports),
                "recommendations": self._generate_import_recommendations(imports),
            }
        except Exception as e:
            return {"error": str(e), "file": str(file_path)}

    def _extract_imports(self, tree: ast.AST) -> List[Dict[str, str]]:
        """Extract import statements from AST."""
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(
                        {
                            "type": "import",
                            "module": alias.name,
                            "alias": alias.asname,
                            "line": node.lineno,
                        }
                    )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(
                        {
                            "type": "from_import",
                            "module": module,
                            "name": alias.name,
                            "alias": alias.asname,
                            "line": node.lineno,
                        }
                    )

        return imports

    def _generate_import_recommendations(self, imports: List[Dict]) -> List[str]:
        """Generate recommendations for import optimization."""
        recommendations = []

        # Check for potential issues
        if len(imports) > 20:
            recommendations.append("High number of imports. Consider lazy loading.")

        # Check for common optimization patterns
        stdlib_imports = [
            imp for imp in imports if self._is_stdlib_import(imp["module"])
        ]
        if len(stdlib_imports) > 10:
            recommendations.append("Many stdlib imports. Group them together.")

        return recommendations

    def _is_stdlib_import(self, module_name: str) -> bool:
        """Check if module is from Python standard library."""
        stdlib_modules = {
            "os",
            "sys",
            "json",
            "time",
            "datetime",
            "pathlib",
            "typing",
            "collections",
            "itertools",
            "functools",
            "importlib",
            "ast",
        }
        return module_name.split(".")[0] in stdlib_modules

    def optimize_imports_in_file(self, file_path: Path) -> Dict[str, Any]:
        """Optimize imports in a specific file."""
        analysis = self.analyze_file(file_path)

        if "error" in analysis:
            return analysis

        # Generate optimized import structure
        imports = analysis["imports"]
        optimized = self._reorganize_imports(imports)

        return {
            "file": str(file_path),
            "original_imports": len(imports),
            "optimized_structure": optimized,
            "recommendations": analysis["recommendations"],
        }

    def _reorganize_imports(self, imports: List[Dict]) -> Dict[str, List[str]]:
        """Reorganize imports into logical groups."""
        groups = {"stdlib": [], "third_party": [], "local": [], "chemml": []}

        for imp in imports:
            module = imp["module"]
            if self._is_stdlib_import(module):
                groups["stdlib"].append(self._format_import(imp))
            elif module.startswith("chemml"):
                groups["chemml"].append(self._format_import(imp))
            elif "." not in module or module.split(".")[0] in [
                "numpy",
                "pandas",
                "torch",
                "sklearn",
            ]:
                groups["third_party"].append(self._format_import(imp))
            else:
                groups["local"].append(self._format_import(imp))

        return groups

    def _format_import(self, imp: Dict) -> str:
        """Format import statement."""
        if imp["type"] == "import":
            if imp["alias"]:
                return f"import {imp['module']} as {imp['alias']}"
            else:
                return f"import {imp['module']}"
        else:  # from_import
            if imp["alias"]:
                return f"from {imp['module']} import {imp['name']} as {imp['alias']}"
            else:
                return f"from {imp['module']} import {imp['name']}"

class CodeStandardizer:
    """Standardize code formatting and style."""

    def __init__(self):
        self.style_issues = []
        self.fixes_applied = []

    def analyze_code_quality(self, file_path: Path) -> Dict[str, Any]:
        """Analyze code quality and style issues."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            issues = []

            # Check line length
            for i, line in enumerate(content.splitlines(), 1):
                if len(line) > 100:
                    issues.append(f"Line {i}: Line too long ({len(line)} chars)")

            # Check for TODO/FIXME comments
            for i, line in enumerate(content.splitlines(), 1):
                if "TODO" in line or "FIXME" in line:
                    issues.append(f"Line {i}: TODO/FIXME comment found")

            # Check for missing docstrings
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    if not ast.get_docstring(node):
                        issues.append(
                            f"Line {node.lineno}: Missing docstring for {node.name}"
                        )

            return {
                "file": str(file_path),
                "issues": issues,
                "issue_count": len(issues),
                "recommendations": self._generate_style_recommendations(issues),
            }

        except Exception as e:
            return {"error": str(e), "file": str(file_path)}

    def _generate_style_recommendations(self, issues: List[str]) -> List[str]:
        """Generate style improvement recommendations."""
        recommendations = []

        long_lines = [issue for issue in issues if "Line too long" in issue]
        if long_lines:
            recommendations.append(f"Fix {len(long_lines)} long lines")

        missing_docs = [issue for issue in issues if "Missing docstring" in issue]
        if missing_docs:
            recommendations.append(
                f"Add docstrings to {len(missing_docs)} functions/classes"
            )

        todos = [issue for issue in issues if "TODO/FIXME" in issue]
        if todos:
            recommendations.append(f"Address {len(todos)} TODO/FIXME comments")

        return recommendations

class UnifiedOptimizer:
    """Main optimizer combining all development tools."""

    def __init__(self):
        self.profiler = PerformanceProfiler()
        self.import_optimizer = ImportOptimizer()
        self.standardizer = CodeStandardizer()

    def optimize_project(self, project_path: Path) -> Dict[str, Any]:
        """Run complete optimization analysis on project."""
        print(f"🔧 Optimizing ChemML project at {project_path}")

        results = {
            "performance": self.profiler.analyze_chemml_performance(),
            "imports": self._analyze_project_imports(project_path),
            "code_quality": self._analyze_project_quality(project_path),
            "summary": {},
        }

        # Generate summary
        results["summary"] = self._generate_summary(results)

        return results

    def _analyze_project_imports(self, project_path: Path) -> Dict[str, Any]:
        """Analyze imports across the project."""
        python_files = list(project_path.rglob("*.py"))

        results = {}
        for file_path in python_files[:10]:  # Limit to first 10 files for demo
            try:
                results[str(file_path)] = self.import_optimizer.analyze_file(file_path)
            except Exception as e:
                results[str(file_path)] = {"error": str(e)}

        return results

    def _analyze_project_quality(self, project_path: Path) -> Dict[str, Any]:
        """Analyze code quality across the project."""
        python_files = list(project_path.rglob("*.py"))

        results = {}
        total_issues = 0

        for file_path in python_files[:5]:  # Limit to first 5 files for demo
            try:
                analysis = self.standardizer.analyze_code_quality(file_path)
                results[str(file_path)] = analysis
                total_issues += analysis.get("issue_count", 0)
            except Exception as e:
                results[str(file_path)] = {"error": str(e)}

        results["summary"] = {"total_issues": total_issues}
        return results

    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate optimization summary."""
        perf = results.get("performance", {})
        _imports = results.get("imports", {})
        quality = results.get("code_quality", {})

        return {
            "performance_score": self._calculate_performance_score(perf),
            "import_optimization_needed": len(perf.get("slow_imports", {})) > 0,
            "code_quality_issues": quality.get("summary", {}).get("total_issues", 0),
            "recommendations": (
                perf.get("recommendations", [])
                + ["Consider import optimization if needed"]
                + ["Review code quality issues"]
            )[
                :5
            ],  # Top 5 recommendations
        }

    def _calculate_performance_score(self, perf_data: Dict) -> float:
        """Calculate performance score (0-100)."""
        if not perf_data:
            return 0.0

        total_time = perf_data.get("total_time", 0)
        failed_count = len(perf_data.get("failed_imports", {}))

        # Simple scoring algorithm
        time_score = max(0, 100 - (total_time * 50))  # Penalize slow imports
        failure_score = max(0, 100 - (failed_count * 20))  # Penalize failures

        return (time_score + failure_score) / 2

def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="ChemML Unified Development Tools")
    parser.add_argument(
        "--optimize-imports", action="store_true", help="Optimize import patterns"
    )
    parser.add_argument(
        "--analyze-performance", action="store_true", help="Analyze performance"
    )
    parser.add_argument(
        "--standardize-code", action="store_true", help="Check code standards"
    )
    parser.add_argument(
        "--full-optimization", action="store_true", help="Run all optimizations"
    )
    parser.add_argument(
        "--project-path", type=str, default=".", help="Project path to analyze"
    )

    args = parser.parse_args()

    project_path = Path(args.project_path).resolve()
    optimizer = UnifiedOptimizer()

    if args.full_optimization or not any(
        [args.optimize_imports, args.analyze_performance, args.standardize_code]
    ):
        print("🚀 Running full optimization analysis...")
        results = optimizer.optimize_project(project_path)

        print("\n📊 Optimization Results:")
        print(
            f"   Performance Score: {results['summary']['performance_score']:.1f}/100"
        )
        print(f"   Code Quality Issues: {results['summary']['code_quality_issues']}")
        print(
            f"   Import Optimization Needed: {results['summary']['import_optimization_needed']}"
        )

        print("\n🎯 Top Recommendations:")
        for i, rec in enumerate(results["summary"]["recommendations"], 1):
            print(f"   {i}. {rec}")

    else:
        if args.analyze_performance:
            print("📈 Analyzing performance...")
            perf_results = optimizer.profiler.analyze_chemml_performance()
            print(f"   Total import time: {perf_results['total_time']:.3f}s")
            print(f"   Slow modules: {len(perf_results['slow_imports'])}")

        if args.optimize_imports:
            print("🔄 Analyzing imports...")
            # Analyze a few key files
            key_files = [
                project_path / "src" / "chemml" / "__init__.py",
                project_path / "src" / "chemml" / "integrations" / "__init__.py",
            ]

            for file_path in key_files:
                if file_path.exists():
                    result = optimizer.import_optimizer.analyze_file(file_path)
                    print(
                        f"   {file_path.name}: {result.get('import_count', 0)} imports"
                    )

        if args.standardize_code:
            print("📝 Checking code standards...")
            # Sample a few files for quality check
            python_files = list(project_path.rglob("*.py"))[:3]
            total_issues = 0

            for file_path in python_files:
                result = optimizer.standardizer.analyze_code_quality(file_path)
                issues = result.get("issue_count", 0)
                total_issues += issues
                print(f"   {file_path.name}: {issues} issues")

            print(f"   Total issues in sample: {total_issues}")

    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()
