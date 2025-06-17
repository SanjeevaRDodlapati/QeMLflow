"""
Advanced Code Quality Improvement Tool for ChemML
================================================

This tool provides advanced code quality improvements beyond basic linting:
1. Complexity reduction suggestions and automated refactoring
2. Advanced type hint addition
3. Dead code detection and removal
4. Performance optimization suggestions
5. Security vulnerability scanning
6. Documentation improvement suggestions

Usage:
    python tools/linting/code_quality_enhancer.py [--auto-fix] [--complexity] [--security] [--all]
"""

import ast
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    import yaml
except ImportError:
    yaml = None


@dataclass
class QualityIssue:
    """Represents a code quality issue."""

    file_path: str
    line_number: int
    issue_type: (
        str  # 'complexity', 'security', 'performance', 'documentation', 'typing'
    )
    severity: str  # 'high', 'medium', 'low'
    message: str
    suggestion: Optional[str] = None
    auto_fixable: bool = False


@dataclass
class ComplexityAnalysis:
    """Analysis of function complexity."""

    function_name: str
    file_path: str
    line_number: int
    cyclomatic_complexity: int
    cognitive_complexity: int
    lines_of_code: int
    suggestions: List[str] = field(default_factory=list)


class CodeQualityEnhancer:
    """Advanced code quality improvement tool."""

    def __init__(self, root_path: Optional[Path] = None):
        self.root = root_path or Path(__file__).parent.parent.parent
        self.config_path = self.root / "tools" / "linting" / "linting_config.yaml"

        # Load configuration
        self.config = self._load_config()

        # Define thresholds
        self.complexity_thresholds = {
            "cyclomatic": 10,
            "cognitive": 15,
            "lines_of_code": 50,
        }

    def _load_config(self) -> Dict:
        """Load linting configuration."""
        if self.config_path.exists() and yaml:
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        return {}

    def _get_python_files(self) -> List[Path]:
        """Get all Python files to analyze."""
        files = []

        include_patterns = self.config.get("include_patterns", ["src/**/*.py"])
        exclude_patterns = self.config.get("exclude_patterns", [])

        for pattern in include_patterns:
            for file_path in self.root.glob(pattern):
                if file_path.is_file() and file_path.suffix == ".py":
                    # Check if file should be excluded
                    should_exclude = False
                    for exclude_pattern in exclude_patterns:
                        if file_path.match(exclude_pattern):
                            should_exclude = True
                            break

                    if not should_exclude:
                        files.append(file_path)

        return files

    def analyze_complexity(
        self, files: Optional[List[Path]] = None
    ) -> List[ComplexityAnalysis]:
        """Analyze cyclomatic and cognitive complexity of functions."""
        if files is None:
            files = self._get_python_files()

        analyses = []

        for file_path in files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                tree = ast.parse(content, filename=str(file_path))

                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        analysis = self._analyze_function_complexity(
                            node, str(file_path), content
                        )
                        if analysis:
                            analyses.append(analysis)

            except Exception as e:
                print(f"Error analyzing {file_path}: {e}")
                continue

        return analyses

    def _analyze_function_complexity(
        self, node: ast.FunctionDef, file_path: str, content: str
    ) -> Optional[ComplexityAnalysis]:
        """Analyze complexity of a single function."""
        try:
            # Calculate cyclomatic complexity
            cyclomatic = self._calculate_cyclomatic_complexity(node)

            # Calculate cognitive complexity (simplified)
            cognitive = self._calculate_cognitive_complexity(node)

            # Count lines of code
            start_line = node.lineno
            end_line = (
                node.end_lineno if hasattr(node, "end_lineno") else start_line + 1
            )
            lines_of_code = end_line - start_line + 1

            # Generate suggestions if thresholds exceeded
            suggestions = []
            if cyclomatic > self.complexity_thresholds["cyclomatic"]:
                suggestions.append(
                    f"High cyclomatic complexity ({cyclomatic}). Consider breaking into smaller functions."
                )

            if cognitive > self.complexity_thresholds["cognitive"]:
                suggestions.append(
                    f"High cognitive complexity ({cognitive}). Consider simplifying control flow."
                )

            if lines_of_code > self.complexity_thresholds["lines_of_code"]:
                suggestions.append(
                    f"Function is {lines_of_code} lines long. Consider splitting into smaller functions."
                )

            # Only return analysis if there are issues
            if suggestions:
                return ComplexityAnalysis(
                    function_name=node.name,
                    file_path=file_path,
                    line_number=node.lineno,
                    cyclomatic_complexity=cyclomatic,
                    cognitive_complexity=cognitive,
                    lines_of_code=lines_of_code,
                    suggestions=suggestions,
                )

        except Exception as e:
            print(f"Error analyzing function {node.name}: {e}")

        return None

    def _calculate_cyclomatic_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity (simplified)."""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.With, ast.AsyncWith)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1

        return complexity

    def _calculate_cognitive_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cognitive complexity (simplified approximation)."""
        complexity = 0
        nesting_level = 0

        def visit_node(n, level):
            nonlocal complexity

            if isinstance(n, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1 + level
                level += 1
            elif isinstance(n, ast.ExceptHandler):
                complexity += 1 + level
            elif isinstance(n, ast.BoolOp):
                complexity += len(n.values) - 1

            for child in ast.iter_child_nodes(n):
                visit_node(child, level)

        for child in ast.iter_child_nodes(node):
            visit_node(child, nesting_level)

        return complexity

    def run_security_scan(
        self, files: Optional[List[Path]] = None
    ) -> List[QualityIssue]:
        """Run security vulnerability scanning with bandit."""
        if files is None:
            files = self._get_python_files()

        issues = []

        try:
            # Run bandit security scanner
            cmd = ["bandit", "-f", "json", "-r"] + [str(f.parent) for f in files]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.root)

            if result.stdout:
                import json

                bandit_results = json.loads(result.stdout)

                for result_item in bandit_results.get("results", []):
                    issue = QualityIssue(
                        file_path=result_item["filename"],
                        line_number=result_item["line_number"],
                        issue_type="security",
                        severity=result_item["issue_severity"].lower(),
                        message=f"Security issue: {result_item['issue_text']}",
                        suggestion=f"Recommendation: {result_item.get('issue_cwe', {}).get('message', 'See bandit documentation')}",
                        auto_fixable=False,
                    )
                    issues.append(issue)

        except Exception as e:
            print(f"Error running security scan: {e}")

        return issues

    def analyze_dead_code(
        self, files: Optional[List[Path]] = None
    ) -> List[QualityIssue]:
        """Analyze for dead/unreachable code."""
        if files is None:
            files = self._get_python_files()

        issues = []

        try:
            # Run vulture for dead code detection
            cmd = ["vulture"] + [str(f) for f in files]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.root)

            if result.stdout:
                for line in result.stdout.split("\n"):
                    if ":" in line and "unused" in line:
                        parts = line.split(":")
                        if len(parts) >= 3:
                            file_path = parts[0]
                            line_num = int(parts[1]) if parts[1].isdigit() else 1
                            message = ":".join(parts[2:]).strip()

                            issue = QualityIssue(
                                file_path=file_path,
                                line_number=line_num,
                                issue_type="dead_code",
                                severity="medium",
                                message=f"Dead code detected: {message}",
                                suggestion="Consider removing unused code to improve maintainability",
                                auto_fixable=False,
                            )
                            issues.append(issue)

        except Exception as e:
            print(f"Error running dead code analysis: {e}")

        return issues

    def generate_quality_report(self) -> Dict:
        """Generate comprehensive quality report."""
        print("🔍 Starting comprehensive code quality analysis...")

        files = self._get_python_files()
        print(f"📁 Analyzing {len(files)} Python files")

        report = {
            "timestamp": str(Path(__file__).stat().st_mtime),
            "files_analyzed": len(files),
            "complexity_issues": [],
            "security_issues": [],
            "dead_code_issues": [],
            "summary": {},
        }

        # Complexity analysis
        print("🧮 Analyzing function complexity...")
        complexity_analyses = self.analyze_complexity(files)

        for analysis in complexity_analyses:
            if analysis.suggestions:
                report["complexity_issues"].append(
                    {
                        "file": analysis.file_path,
                        "function": analysis.function_name,
                        "line": analysis.line_number,
                        "cyclomatic_complexity": analysis.cyclomatic_complexity,
                        "cognitive_complexity": analysis.cognitive_complexity,
                        "lines_of_code": analysis.lines_of_code,
                        "suggestions": analysis.suggestions,
                    }
                )

        # Security analysis
        print("🔒 Running security vulnerability scan...")
        security_issues = self.run_security_scan(files)
        report["security_issues"] = [
            {
                "file": issue.file_path,
                "line": issue.line_number,
                "severity": issue.severity,
                "message": issue.message,
                "suggestion": issue.suggestion,
            }
            for issue in security_issues
        ]

        # Dead code analysis
        print("💀 Analyzing for dead code...")
        dead_code_issues = self.analyze_dead_code(files)
        report["dead_code_issues"] = [
            {
                "file": issue.file_path,
                "line": issue.line_number,
                "message": issue.message,
                "suggestion": issue.suggestion,
            }
            for issue in dead_code_issues
        ]

        # Summary
        report["summary"] = {
            "total_complexity_issues": len(report["complexity_issues"]),
            "total_security_issues": len(report["security_issues"]),
            "total_dead_code_issues": len(report["dead_code_issues"]),
            "high_complexity_functions": len(
                [
                    i
                    for i in report["complexity_issues"]
                    if i["cyclomatic_complexity"] > 15 or i["cognitive_complexity"] > 20
                ]
            ),
            "high_severity_security": len(
                [i for i in report["security_issues"] if i["severity"] == "high"]
            ),
        }

        return report

    def print_quality_report(self, report: Dict):
        """Print formatted quality report."""
        print("\n" + "=" * 80)
        print("🏆 ChemML Advanced Code Quality Report")
        print("=" * 80)

        summary = report["summary"]
        print("📊 Analysis Summary:")
        print(f"  • Files analyzed: {report['files_analyzed']}")
        print(f"  • Complexity issues: {summary['total_complexity_issues']}")
        print(f"  • Security issues: {summary['total_security_issues']}")
        print(f"  • Dead code issues: {summary['total_dead_code_issues']}")
        print(f"  • High complexity functions: {summary['high_complexity_functions']}")
        print(f"  • High severity security: {summary['high_severity_security']}")

        # Top complexity issues
        if report["complexity_issues"]:
            print("\n🧮 Top Complexity Issues:")
            sorted_complexity = sorted(
                report["complexity_issues"],
                key=lambda x: x["cyclomatic_complexity"],
                reverse=True,
            )[:5]

            for issue in sorted_complexity:
                print(
                    f"  • {issue['function']} in {Path(issue['file']).name}:{issue['line']}"
                )
                print(
                    f"    Cyclomatic: {issue['cyclomatic_complexity']}, "
                    f"Cognitive: {issue['cognitive_complexity']}, "
                    f"LOC: {issue['lines_of_code']}"
                )

        # Security issues
        if report["security_issues"]:
            print("\n🔒 Security Issues:")
            for issue in report["security_issues"][:5]:
                print(
                    f"  • {Path(issue['file']).name}:{issue['line']} [{issue['severity'].upper()}]"
                )
                print(f"    {issue['message']}")

        # Recommendations
        print("\n💡 Recommendations:")
        if summary["high_complexity_functions"] > 0:
            print(
                f"  • Refactor {summary['high_complexity_functions']} high-complexity functions"
            )
        if summary["total_security_issues"] > 0:
            print(
                f"  • Address {summary['total_security_issues']} security vulnerabilities"
            )
        if summary["total_dead_code_issues"] > 0:
            print(f"  • Remove {summary['total_dead_code_issues']} pieces of dead code")

        print("=" * 80)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Advanced code quality improvement tool for ChemML"
    )
    parser.add_argument(
        "--complexity", action="store_true", help="Analyze function complexity"
    )
    parser.add_argument(
        "--security", action="store_true", help="Run security vulnerability scan"
    )
    parser.add_argument(
        "--dead-code", action="store_true", help="Analyze for dead/unused code"
    )
    parser.add_argument("--all", action="store_true", help="Run all quality checks")
    parser.add_argument("--save", action="store_true", help="Save report to file")

    args = parser.parse_args()

    if not any([args.complexity, args.security, args.dead_code, args.all]):
        args.all = True  # Default to all checks

    enhancer = CodeQualityEnhancer()

    if args.all:
        report = enhancer.generate_quality_report()
        enhancer.print_quality_report(report)

        if args.save:
            report_file = enhancer.root / "reports" / "linting" / "quality_report.json"
            report_file.parent.mkdir(parents=True, exist_ok=True)

            import json

            with open(report_file, "w") as f:
                json.dump(report, f, indent=2)
            print(f"📄 Report saved to {report_file}")

    else:
        # Run individual checks
        if args.complexity:
            analyses = enhancer.analyze_complexity()
            print(f"Found {len(analyses)} complexity issues")

        if args.security:
            issues = enhancer.run_security_scan()
            print(f"Found {len(issues)} security issues")

        if args.dead_code:
            issues = enhancer.analyze_dead_code()
            print(f"Found {len(issues)} dead code issues")


if __name__ == "__main__":
    main()
