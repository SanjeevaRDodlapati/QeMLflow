"""
Comprehensive Linting Framework for ChemML
==========================================

A comprehensive linting framework that provides:
1. Multi-tool linting analysis (flake8, black, isort, mypy)
2. Intelligent issue categorization and prioritization
3. Auto-fix capabilities for common issues
4. Reporting and tracking of linting health
5. Integration with CI/CD pipeline

This framework helps main        if not quiet:
            print("🔍 Running mypy...")
        mypy_issues = self._run_mypy(files)
        all_issues.extend(mypy_issues)

        # Populate report
        report.issues = all_issues
        report.total_issues = len(all_issues)lity across the entire ChemML codebase.
"""

import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

try:
    import yaml
except ImportError:
    yaml = None


@dataclass
class LintingIssue:
    """Represents a single linting issue."""

    file_path: str
    line_number: int
    column: int
    rule_code: str
    message: str
    severity: str  # 'error', 'warning', 'info'
    tool: str  # 'flake8', 'black', 'isort', 'mypy'
    auto_fixable: bool = False


@dataclass
class LintingReport:
    """Comprehensive linting report."""

    timestamp: datetime
    total_files_checked: int
    total_issues: int = 0
    issues_by_severity: Dict[str, int] = field(default_factory=dict)
    issues_by_tool: Dict[str, int] = field(default_factory=dict)
    issues_by_category: Dict[str, int] = field(default_factory=dict)
    issues: List[LintingIssue] = field(default_factory=list)
    auto_fixable_count: int = 0
    health_score: float = 0.0


class ComprehensiveLinter:
    """Advanced linting framework for ChemML."""

    def __init__(self, root_path: Optional[Path] = None):
        self.root = (
            root_path or Path(__file__).parent.parent.parent
        )  # Go up to ChemML root
        self.config_path = self.root / "tools" / "linting" / "linting_config.yaml"
        self.reports_dir = self.root / "reports" / "linting"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Load configuration
        self.config = self._load_config()

        # Define file patterns to check
        self.include_patterns = self.config.get(
            "include_patterns",
            [
                "src/**/*.py",
                "tests/**/*.py",
                "scripts/**/*.py",
                "tools/**/*.py",
                "examples/**/*.py",
            ],
        )

        # Define patterns to exclude
        self.exclude_patterns = self.config.get(
            "exclude_patterns",
            [
                "chemml_env/**",
                "build/**",
                "dist/**",
                ".venv/**",
                "__pycache__/**",
                "*.egg-info/**",
                "archive/**",  # Exclude legacy/archived code
                "docs/_build/**",
                "site/**",
            ],
        )

    def _load_config(self) -> Dict:
        """Load linting configuration."""
        if self.config_path.exists() and yaml is not None:
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f)

        # Default configuration
        return {
            "tools": {
                "flake8": {
                    "enabled": True,
                    "severity_map": {
                        "E": "error",
                        "W": "warning",
                        "F": "error",
                        "C": "warning",
                        "B": "warning",
                        "N": "warning",
                    },
                },
                "black": {"enabled": True, "auto_fix": True},
                "isort": {"enabled": True, "auto_fix": True},
                "mypy": {"enabled": True, "strict_mode": False},
            },
            "severity_weights": {"error": 1.0, "warning": 0.5, "info": 0.1},
        }

    def _get_python_files(self) -> List[Path]:
        """Get list of Python files to check."""
        files = []

        for pattern in self.include_patterns:
            files.extend(self.root.glob(pattern))

        # Filter out excluded patterns
        filtered_files = []
        for file_path in files:
            relative_path = file_path.relative_to(self.root)
            excluded = any(
                relative_path.match(pattern) for pattern in self.exclude_patterns
            )
            if not excluded and file_path.is_file():
                filtered_files.append(file_path)

        return filtered_files

    def _run_flake8(self, files: List[Path]) -> List[LintingIssue]:
        """Run flake8 linting."""
        issues = []
        if not self.config["tools"]["flake8"]["enabled"]:
            return issues

        try:
            # Run flake8 with JSON-like output format
            cmd = ["flake8", "--format=%(path)s:%(row)d:%(col)d: %(code)s %(text)s"] + [
                str(f) for f in files
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.root)

            severity_map = self.config["tools"]["flake8"]["severity_map"]

            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue

                try:
                    # Parse flake8 output: path:line:col: code message
                    # Use regex to handle complex parsing more robustly
                    import re

                    match = re.match(r"^([^:]+):(\d+):(\d+):\s+(\w+)\s+(.*)$", line)
                    if match:
                        file_path = match.group(1)
                        line_num = int(match.group(2))
                        col_num = int(match.group(3))
                        code = match.group(4)
                        message = match.group(5)

                        # Determine severity
                        severity = severity_map.get(code[0], "warning")

                        # Check if auto-fixable
                        auto_fixable = code in ["F401", "W391", "E302", "E303"]

                        issues.append(
                            LintingIssue(
                                file_path=file_path,
                                line_number=line_num,
                                column=col_num,
                                rule_code=code,
                                message=message,
                                severity=severity,
                                tool="flake8",
                                auto_fixable=auto_fixable,
                            )
                        )
                except (ValueError, IndexError, AttributeError) as e:
                    # Skip lines that don't match the expected format
                    continue

        except Exception as e:
            print(f"Error running flake8: {e}")

        return issues

    def _run_black(self, files: List[Path]) -> List[LintingIssue]:
        """Run black formatting check."""
        issues = []
        if not self.config["tools"]["black"]["enabled"]:
            return issues

        try:
            # Run black in check mode
            cmd = ["black", "--check", "--diff"] + [str(f) for f in files]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.root)

            if result.returncode != 0:
                # Parse diff output to find files that need formatting
                output_lines = result.stderr.split("\n")
                for line in output_lines:
                    if line.startswith("would reformat"):
                        file_path = line.split()[-1]
                        issues.append(
                            LintingIssue(
                                file_path=file_path,
                                line_number=1,
                                column=1,
                                rule_code="BLACK001",
                                message="File needs black formatting",
                                severity="warning",
                                tool="black",
                                auto_fixable=True,
                            )
                        )

        except Exception as e:
            print(f"Error running black: {e}")

        return issues

    def _run_isort(self, files: List[Path]) -> List[LintingIssue]:
        """Run isort import sorting check."""
        issues = []
        if not self.config["tools"]["isort"]["enabled"]:
            return issues

        try:
            # Run isort in check mode
            cmd = ["isort", "--check-only", "--diff"] + [str(f) for f in files]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.root)

            if result.returncode != 0:
                # Parse output to find files with import issues
                output_lines = result.stdout.split("\n")
                current_file = None
                for line in output_lines:
                    if line.startswith("ERROR:"):
                        # Extract file path from error message
                        if "Imports are incorrectly sorted" in line:
                            # Find file path in the line
                            parts = line.split()
                            for i, part in enumerate(parts):
                                if part.endswith(".py"):
                                    current_file = part
                                    break

                        if current_file:
                            issues.append(
                                LintingIssue(
                                    file_path=current_file,
                                    line_number=1,
                                    column=1,
                                    rule_code="ISORT001",
                                    message="Imports are incorrectly sorted",
                                    severity="warning",
                                    tool="isort",
                                    auto_fixable=True,
                                )
                            )

        except Exception as e:
            print(f"Error running isort: {e}")

        return issues

    def _run_mypy(self, files: List[Path]) -> List[LintingIssue]:
        """Run mypy type checking."""
        issues = []
        if not self.config["tools"]["mypy"]["enabled"]:
            return issues

        try:
            # Run mypy on specific files
            cmd = ["mypy"] + [str(f) for f in files]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.root)

            if result.returncode != 0:
                # Parse mypy output
                output_lines = result.stdout.split("\n")
                for line in output_lines:
                    if ":" in line and (
                        "error:" in line or "warning:" in line or "note:" in line
                    ):
                        try:
                            # Parse line format: file:line:col: severity: message
                            parts = line.split(":", 3)
                            if len(parts) >= 4:
                                file_path = parts[0]
                                line_num = int(parts[1]) if parts[1].isdigit() else 1
                                col = int(parts[2]) if parts[2].isdigit() else 1
                                rest = parts[3].strip()

                                if rest.startswith("error:"):
                                    severity = "error"
                                    message = rest[6:].strip()
                                    rule_code = "MYPY001"
                                elif rest.startswith("warning:"):
                                    severity = "warning"
                                    message = rest[8:].strip()
                                    rule_code = "MYPY002"
                                elif rest.startswith("note:"):
                                    severity = "info"
                                    message = rest[5:].strip()
                                    rule_code = "MYPY003"
                                else:
                                    continue

                                issues.append(
                                    LintingIssue(
                                        file_path=file_path,
                                        line_number=line_num,
                                        column=col,
                                        rule_code=rule_code,
                                        message=message,
                                        severity=severity,
                                        tool="mypy",
                                        auto_fixable=False,
                                    )
                                )
                        except (ValueError, IndexError):
                            # Skip malformed lines
                            continue

        except Exception as e:
            print(f"Error running mypy: {e}")

        return issues

    def _categorize_issues(self, issues: List[LintingIssue]) -> Dict[str, int]:
        """Categorize issues by type."""
        categories = {
            "import_issues": 0,
            "formatting": 0,
            "complexity": 0,
            "unused_variables": 0,
            "type_errors": 0,
            "style_violations": 0,
            "other": 0,
        }

        for issue in issues:
            code = issue.rule_code

            if code.startswith("F4") or code == "ISORT001":
                categories["import_issues"] += 1
            elif code == "BLACK001" or code.startswith("E"):
                categories["formatting"] += 1
            elif code == "C901":
                categories["complexity"] += 1
            elif code in ["F841", "F401"]:
                categories["unused_variables"] += 1
            elif code.startswith("F8") or code.startswith("MYPY"):
                categories["type_errors"] += 1
            elif code.startswith("W") or code.startswith("N"):
                categories["style_violations"] += 1
            else:
                categories["other"] += 1

        return categories

    def _calculate_health_score(self, report: LintingReport) -> float:
        """Calculate overall codebase health score (0-100)."""
        if report.total_files_checked == 0:
            return 100.0

        # Calculate weighted issue count
        weights = self.config["severity_weights"]
        weighted_issues = (
            report.issues_by_severity.get("error", 0) * weights["error"]
            + report.issues_by_severity.get("warning", 0) * weights["warning"]
            + report.issues_by_severity.get("info", 0) * weights["info"]
        )

        # Calculate issues per file
        issues_per_file = weighted_issues / report.total_files_checked

        # Convert to health score (lower issues = higher score)
        # Assume 0 issues per file = 100%, 10+ issues per file = 0%
        health_score = max(0, 100 - (issues_per_file * 10))

        return round(health_score, 1)

    def run_comprehensive_analysis(self, quiet: bool = False) -> LintingReport:
        """Run comprehensive linting analysis."""
        if not quiet:
            print("🔍 Starting comprehensive linting analysis...")

        # Get files to check
        files = self._get_python_files()
        if not quiet:
            print(f"📁 Found {len(files)} Python files to check")

        # Initialize report
        report = LintingReport(timestamp=datetime.now(), total_files_checked=len(files))

        # Run linting tools
        all_issues = []

        if not quiet:
            print("🔧 Running flake8...")
        flake8_issues = self._run_flake8(files)
        all_issues.extend(flake8_issues)

        if not quiet:
            print("🎨 Running black...")
        black_issues = self._run_black(files)
        all_issues.extend(black_issues)

        if not quiet:
            print("📦 Running isort...")
        isort_issues = self._run_isort(files)
        all_issues.extend(isort_issues)

        if not quiet:
            print("🔍 Running mypy...")
        mypy_issues = self._run_mypy(files)
        all_issues.extend(mypy_issues)

        # Populate report
        report.issues = all_issues
        report.total_issues = len(all_issues)

        # Calculate statistics
        report.issues_by_severity = {}
        report.issues_by_tool = {}
        report.auto_fixable_count = 0

        for issue in all_issues:
            # By severity
            report.issues_by_severity[issue.severity] = (
                report.issues_by_severity.get(issue.severity, 0) + 1
            )

            # By tool
            report.issues_by_tool[issue.tool] = (
                report.issues_by_tool.get(issue.tool, 0) + 1
            )

            # Auto-fixable count
            if issue.auto_fixable:
                report.auto_fixable_count += 1

        # Categorize issues
        report.issues_by_category = self._categorize_issues(all_issues)

        # Calculate health score
        report.health_score = self._calculate_health_score(report)

        return report

    def auto_fix_issues(self, dry_run: bool = True) -> Dict[str, int]:
        """Auto-fix common linting issues."""
        fixed_count = {"black": 0, "isort": 0, "manual": 0}

        if not dry_run:
            # Run black formatting
            if self.config["tools"]["black"].get("auto_fix", False):
                try:
                    files = self._get_python_files()
                    cmd = ["black"] + [str(f) for f in files]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode == 0:
                        # Count files that were reformatted
                        fixed_count["black"] = len(
                            [
                                line
                                for line in result.stderr.split("\n")
                                if "reformatted" in line
                            ]
                        )
                except Exception as e:
                    print(f"Error running black auto-fix: {e}")

            # Run isort import sorting
            if self.config["tools"]["isort"].get("auto_fix", False):
                try:
                    files = self._get_python_files()
                    cmd = ["isort"] + [str(f) for f in files]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode == 0:
                        fixed_count["isort"] = len(
                            [
                                line
                                for line in result.stdout.split("\n")
                                if "Fixing" in line
                            ]
                        )
                except Exception as e:
                    print(f"Error running isort auto-fix: {e}")

        return fixed_count

    def generate_report(
        self, report: LintingReport, format_type: str = "console"
    ) -> str:
        """Generate formatted report."""
        if format_type == "console":
            return self._generate_console_report(report)
        elif format_type == "json":
            return self._generate_json_report(report)
        else:
            raise ValueError(
                f"Unsupported format: {format_type} (html not implemented yet)"
            )

    def _generate_console_report(self, report: LintingReport) -> str:
        """Generate console-friendly report."""
        lines = []
        lines.append("=" * 80)
        lines.append("🔍 ChemML Comprehensive Linting Report")
        lines.append("=" * 80)
        lines.append(f"📅 Generated: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"📁 Files checked: {report.total_files_checked}")
        lines.append(f"🚨 Total issues: {report.total_issues}")
        lines.append(f"🏥 Health score: {report.health_score}/100")
        lines.append("")

        # Issues by severity
        lines.append("📊 Issues by Severity:")
        for severity, count in report.issues_by_severity.items():
            lines.append(f"  {severity.capitalize()}: {count}")
        lines.append("")

        # Issues by tool
        lines.append("🔧 Issues by Tool:")
        for tool, count in report.issues_by_tool.items():
            lines.append(f"  {tool}: {count}")
        lines.append("")

        # Issues by category
        lines.append("📂 Issues by Category:")
        for category, count in report.issues_by_category.items():
            if count > 0:
                lines.append(f"  {category.replace('_', ' ').title()}: {count}")
        lines.append("")

        # Auto-fixable issues
        lines.append(f"🔧 Auto-fixable issues: {report.auto_fixable_count}")
        lines.append("")

        # Health assessment
        if report.health_score >= 90:
            assessment = "🟢 Excellent - Code quality is very high"
        elif report.health_score >= 75:
            assessment = "🟡 Good - Minor issues to address"
        elif report.health_score >= 50:
            assessment = "🟠 Fair - Several issues need attention"
        else:
            assessment = "🔴 Poor - Significant improvements needed"

        lines.append(f"📈 Health Assessment: {assessment}")
        lines.append("")

        # Recommendations
        lines.append("💡 Recommendations:")
        if report.auto_fixable_count > 0:
            lines.append(
                f"  • Run auto-fix to resolve {report.auto_fixable_count} fixable issues"
            )

        if report.issues_by_category.get("complexity", 0) > 0:
            lines.append("  • Refactor complex functions to improve maintainability")

        if report.issues_by_category.get("import_issues", 0) > 0:
            lines.append("  • Clean up unused imports and organize import statements")

        if report.issues_by_severity.get("error", 0) > 0:
            lines.append("  • Address error-level issues first for code stability")

        lines.append("=" * 80)

        return "\n".join(lines)

    def _generate_json_report(self, report: LintingReport) -> str:
        """Generate JSON report."""
        data = {
            "timestamp": report.timestamp.isoformat(),
            "summary": {
                "total_files_checked": report.total_files_checked,
                "total_issues": report.total_issues,
                "health_score": report.health_score,
                "auto_fixable_count": report.auto_fixable_count,
            },
            "issues_by_severity": report.issues_by_severity,
            "issues_by_tool": report.issues_by_tool,
            "issues_by_category": report.issues_by_category,
            "issues": [
                {
                    "file_path": issue.file_path,
                    "line_number": issue.line_number,
                    "column": issue.column,
                    "rule_code": issue.rule_code,
                    "message": issue.message,
                    "severity": issue.severity,
                    "tool": issue.tool,
                    "auto_fixable": issue.auto_fixable,
                }
                for issue in report.issues
            ],
        }
        return json.dumps(data, indent=2)

    def save_report(self, report: LintingReport, format_type: str = "json") -> Path:
        """Save report to file."""
        timestamp = report.timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"linting_report_{timestamp}.{format_type}"
        report_path = self.reports_dir / filename

        content = self.generate_report(report, format_type)
        with open(report_path, "w") as f:
            f.write(content)

        return report_path


def main():
    """Main entry point for comprehensive linting."""
    import argparse

    parser = argparse.ArgumentParser(description="ChemML Comprehensive Linting")
    parser.add_argument(
        "--auto-fix", action="store_true", help="Auto-fix common issues"
    )
    parser.add_argument(
        "--format", choices=["console", "json"], default="console", help="Report format"
    )
    parser.add_argument("--save", action="store_true", help="Save report to file")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output (useful for JSON mode)",
    )

    args = parser.parse_args()

    linter = ComprehensiveLinter()

    # Suppress progress output in JSON mode or when explicitly requested
    quiet_mode = args.format == "json" or args.quiet

    # Run analysis
    if not quiet_mode:
        print("🔍 Starting comprehensive linting analysis...")
    report = linter.run_comprehensive_analysis(quiet=quiet_mode)

    # Auto-fix if requested
    if args.auto_fix:
        if not quiet_mode:
            print("\n🔧 Running auto-fix...")
        fixed = linter.auto_fix_issues(dry_run=False)
        if not quiet_mode:
            print(f"✅ Fixed issues: {sum(fixed.values())}")

        # Re-run analysis after fixes
        if not quiet_mode:
            print("🔄 Re-analyzing after fixes...")
        report = linter.run_comprehensive_analysis(quiet=quiet_mode)

    # Generate and display report
    formatted_report = linter.generate_report(report, args.format)

    # In JSON mode, only output JSON to stdout
    if args.format == "json":
        print(formatted_report)
    else:
        print(formatted_report)

    # Save report if requested
    if args.save:
        report_path = linter.save_report(report, args.format)
        if not quiet_mode and args.format != "json":
            print(f"\n💾 Report saved to: {report_path}")


if __name__ == "__main__":
    main()
