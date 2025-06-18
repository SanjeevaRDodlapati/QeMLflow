#!/usr/bin/env python3
"""
Robust Multi-Layer Linting Framework
===================================

A defensive linting system that uses multiple overlapping tools and cross-validation
to ensure no issues are missed. Implements redundancy, consensus, and verification.

Architecture:
1. Multiple linters for each category
2. Cross-validation between tools
3. Consensus-based issue reporting
4. Tool failure detection and recovery
5. Progressive escalation for disagreements
"""

import ast
import json
import py_compile
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union


@dataclass
class LintingIssue:
    """Enhanced linting issue with source tool tracking."""

    file_path: str
    line_number: int
    column: int
    rule_code: str
    message: str
    severity: str
    tool: str
    auto_fixable: bool = False
    confidence: float = 1.0  # Tool confidence in this issue
    detected_by: List[str] = field(default_factory=list)  # Which tools found this

    def __hash__(self):
        """Allow issues to be used in sets for deduplication."""
        return hash((self.file_path, self.line_number, self.column, self.rule_code))


@dataclass
class ToolResult:
    """Result from a single linting tool."""

    tool_name: str
    success: bool
    issues: List[LintingIssue]
    execution_time: float
    error_message: str = ""
    exit_code: int = 0


@dataclass
class ConsensusReport:
    """Multi-tool consensus report."""

    timestamp: datetime
    total_files_checked: int
    tools_run: List[str]
    tools_failed: List[str]

    # Issue analysis
    total_issues: int = 0
    consensus_issues: int = 0  # Agreed upon by multiple tools
    disputed_issues: int = 0  # Found by only one tool

    # Tool agreement analysis
    tool_agreement_score: float = 0.0  # How much tools agree (0-1)
    reliability_score: float = 0.0  # Overall reliability (0-1)

    # Categorized results
    issues_by_category: Dict[str, int] = field(default_factory=dict)
    issues_by_severity: Dict[str, int] = field(default_factory=dict)
    issues_by_consensus: Dict[str, List[LintingIssue]] = field(default_factory=dict)

    # All issues with metadata
    all_issues: List[LintingIssue] = field(default_factory=list)

    health_score: float = 0.0


class RobustLintingFramework:
    """Multi-layer defensive linting framework."""

    def __init__(self, root_path: Optional[Path] = None):
        self.root = root_path or Path.cwd()
        self.backup_dir = (
            self.root
            / "backups"
            / f"robust_lint_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # Configure linting tools by category
        self.syntax_tools = ["python_compile", "ast_parse", "pylint_syntax"]
        self.style_tools = ["flake8", "pycodestyle", "autopep8_check"]
        self.logic_tools = ["flake8", "pylint", "bandit_logic"]
        self.type_tools = ["mypy", "pytype", "pyre"]
        self.format_tools = ["black", "autopep8", "yapf"]
        self.import_tools = ["isort", "flake8", "import_linter"]
        self.complexity_tools = ["flake8", "pylint", "radon", "xenon"]
        self.security_tools = ["bandit", "safety", "semgrep"]

        # Tool availability cache
        self.available_tools = {}
        self._check_tool_availability()

        # Consensus thresholds
        self.consensus_threshold = 0.6  # 60% of tools must agree
        self.reliability_threshold = 0.8  # 80% tool success rate required

    def _check_tool_availability(self):
        """Check which tools are actually available."""
        potential_tools = [
            "flake8",
            "black",
            "isort",
            "mypy",
            "pylint",
            "bandit",
            "autopep8",
            "yapf",
            "pytype",
            "pyre",
            "radon",
            "xenon",
            "safety",
            "semgrep",
            "pycodestyle",
        ]

        for tool in potential_tools:
            try:
                result = subprocess.run(
                    [tool, "--version"], capture_output=True, text=True, timeout=5
                )
                self.available_tools[tool] = result.returncode == 0
            except (subprocess.TimeoutExpired, FileNotFoundError):
                self.available_tools[tool] = False

        # Always available (built-in)
        self.available_tools["python_compile"] = True
        self.available_tools["ast_parse"] = True

    def validate_syntax_multiple(self, file_path: Path) -> List[ToolResult]:
        """Validate syntax using multiple methods."""
        results = []

        # Method 1: Python compile
        start_time = datetime.now()
        try:
            py_compile.compile(str(file_path), doraise=True)
            results.append(
                ToolResult(
                    tool_name="python_compile",
                    success=True,
                    issues=[],
                    execution_time=(datetime.now() - start_time).total_seconds(),
                )
            )
        except Exception as e:
            issue = LintingIssue(
                file_path=str(file_path),
                line_number=getattr(e, "lineno", 1),
                column=getattr(e, "offset", 1),
                rule_code="E999",
                message=str(e),
                severity="error",
                tool="python_compile",
                confidence=1.0,
                detected_by=["python_compile"],
            )
            results.append(
                ToolResult(
                    tool_name="python_compile",
                    success=True,
                    issues=[issue],
                    execution_time=(datetime.now() - start_time).total_seconds(),
                )
            )

        # Method 2: AST parsing
        start_time = datetime.now()
        try:
            with open(file_path, "r") as f:
                content = f.read()
            ast.parse(content, filename=str(file_path))
            results.append(
                ToolResult(
                    tool_name="ast_parse",
                    success=True,
                    issues=[],
                    execution_time=(datetime.now() - start_time).total_seconds(),
                )
            )
        except SyntaxError as e:
            issue = LintingIssue(
                file_path=str(file_path),
                line_number=e.lineno or 1,
                column=e.offset or 1,
                rule_code="E999",
                message=str(e),
                severity="error",
                tool="ast_parse",
                confidence=0.9,
                detected_by=["ast_parse"],
            )
            results.append(
                ToolResult(
                    tool_name="ast_parse",
                    success=True,
                    issues=[issue],
                    execution_time=(datetime.now() - start_time).total_seconds(),
                )
            )
        except Exception as e:
            results.append(
                ToolResult(
                    tool_name="ast_parse",
                    success=False,
                    issues=[],
                    execution_time=(datetime.now() - start_time).total_seconds(),
                    error_message=str(e),
                )
            )

        return results

    def run_tool_safely(
        self, tool_name: str, file_paths: List[Path], extra_args: List[str] = None
    ) -> ToolResult:
        """Run a single tool with comprehensive error handling."""
        start_time = datetime.now()
        extra_args = extra_args or []

        if not self.available_tools.get(tool_name, False):
            return ToolResult(
                tool_name=tool_name,
                success=False,
                issues=[],
                execution_time=0,
                error_message=f"Tool {tool_name} not available",
            )

        try:
            # Build command
            cmd = [tool_name] + extra_args + [str(f) for f in file_paths]

            # Run with timeout
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.root,
                timeout=300,  # 5 minute timeout
            )

            execution_time = (datetime.now() - start_time).total_seconds()

            # Parse output based on tool
            issues = self._parse_tool_output(tool_name, result.stdout, result.stderr)

            return ToolResult(
                tool_name=tool_name,
                success=True,
                issues=issues,
                execution_time=execution_time,
                exit_code=result.returncode,
            )

        except subprocess.TimeoutExpired:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                issues=[],
                execution_time=(datetime.now() - start_time).total_seconds(),
                error_message="Tool execution timeout",
            )
        except Exception as e:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                issues=[],
                execution_time=(datetime.now() - start_time).total_seconds(),
                error_message=str(e),
            )

    def _parse_tool_output(
        self, tool_name: str, stdout: str, stderr: str
    ) -> List[LintingIssue]:
        """Parse tool output into standardized issues."""
        issues = []

        if tool_name == "flake8":
            # Parse flake8 format: file:line:col: code message
            for line in stdout.strip().split("\n"):
                if not line:
                    continue
                match = re.match(r"^([^:]+):(\d+):(\d+):\s+(\w+)\s+(.*)$", line)
                if match:
                    file_path, line_num, col, code, message = match.groups()
                    severity = "error" if code.startswith(("E999", "F")) else "warning"
                    issues.append(
                        LintingIssue(
                            file_path=file_path,
                            line_number=int(line_num),
                            column=int(col),
                            rule_code=code,
                            message=message,
                            severity=severity,
                            tool=tool_name,
                            auto_fixable=code in ["F401", "W391", "E302", "E303"],
                            confidence=0.9,
                            detected_by=[tool_name],
                        )
                    )

        elif tool_name == "black":
            # Parse black stderr for reformatting needs
            if stderr and "would reformat" in stderr:
                for line in stderr.split("\n"):
                    if "would reformat" in line:
                        file_path = line.split()[-1]
                        issues.append(
                            LintingIssue(
                                file_path=file_path,
                                line_number=1,
                                column=1,
                                rule_code="BLACK001",
                                message="File needs black formatting",
                                severity="warning",
                                tool=tool_name,
                                auto_fixable=True,
                                confidence=1.0,
                                detected_by=[tool_name],
                            )
                        )

        elif tool_name == "mypy":
            # Parse mypy output: file:line: error/warning: message
            for line in stdout.strip().split("\n"):
                if ":" in line and ("error:" in line or "warning:" in line):
                    parts = line.split(":", 3)
                    if len(parts) >= 4:
                        file_path = parts[0]
                        line_num = int(parts[1]) if parts[1].isdigit() else 1
                        col = int(parts[2]) if parts[2].isdigit() else 1
                        rest = parts[3].strip()

                        if rest.startswith("error:"):
                            severity = "error"
                            message = rest[6:].strip()
                            code = "MYPY001"
                        elif rest.startswith("warning:"):
                            severity = "warning"
                            message = rest[8:].strip()
                            code = "MYPY002"
                        else:
                            continue

                        issues.append(
                            LintingIssue(
                                file_path=file_path,
                                line_number=line_num,
                                column=col,
                                rule_code=code,
                                message=message,
                                severity=severity,
                                tool=tool_name,
                                confidence=0.8,
                                detected_by=[tool_name],
                            )
                        )

        elif tool_name == "pylint":
            # Parse pylint JSON output if available, otherwise text
            try:
                if stdout.strip().startswith("[") or stdout.strip().startswith("{"):
                    # JSON format
                    data = json.loads(stdout)
                    for item in data:
                        issues.append(
                            LintingIssue(
                                file_path=item.get("path", ""),
                                line_number=item.get("line", 1),
                                column=item.get("column", 1),
                                rule_code=item.get("message-id", "PYLINT"),
                                message=item.get("message", ""),
                                severity=self._pylint_type_to_severity(
                                    item.get("type", "warning")
                                ),
                                tool=tool_name,
                                confidence=0.8,
                                detected_by=[tool_name],
                            )
                        )
            except json.JSONDecodeError:
                # Text format fallback
                for line in stdout.strip().split("\n"):
                    if ":" in line and any(
                        level in line for level in ["ERROR", "WARNING", "INFO"]
                    ):
                        # Basic text parsing for pylint
                        pass  # Implement if needed

        elif tool_name == "bandit":
            # Parse bandit JSON output
            try:
                if stdout.strip():
                    data = json.loads(stdout)
                    for result in data.get("results", []):
                        issues.append(
                            LintingIssue(
                                file_path=result.get("filename", ""),
                                line_number=result.get("line_number", 1),
                                column=1,
                                rule_code=result.get("test_id", "BANDIT"),
                                message=result.get("issue_text", ""),
                                severity=result.get(
                                    "issue_severity", "warning"
                                ).lower(),
                                tool=tool_name,
                                confidence=float(
                                    result.get("issue_confidence", "MEDIUM")
                                    .replace("HIGH", "0.9")
                                    .replace("MEDIUM", "0.7")
                                    .replace("LOW", "0.5")
                                ),
                                detected_by=[tool_name],
                            )
                        )
            except json.JSONDecodeError:
                pass

        # Add more tool parsers as needed...

        return issues

    def _pylint_type_to_severity(self, pylint_type: str) -> str:
        """Convert pylint message type to severity."""
        mapping = {
            "error": "error",
            "warning": "warning",
            "refactor": "info",
            "convention": "info",
            "info": "info",
        }
        return mapping.get(pylint_type.lower(), "warning")

    def find_consensus_issues(
        self, tool_results: List[ToolResult]
    ) -> Dict[str, List[LintingIssue]]:
        """Find issues that multiple tools agree on."""
        all_issues = []
        for result in tool_results:
            if result.success:
                all_issues.extend(result.issues)

        # Group similar issues (same file, similar line, similar type)
        issue_groups = {}
        for issue in all_issues:
            # Create a key for similar issues
            key = (
                issue.file_path,
                issue.line_number // 5 * 5,
                issue.rule_code[:1],
            )  # Group by file, line range, and rule type
            if key not in issue_groups:
                issue_groups[key] = []
            issue_groups[key].append(issue)

        # Classify by consensus level
        consensus_issues = {
            "strong_consensus": [],  # 3+ tools agree
            "moderate_consensus": [],  # 2 tools agree
            "single_tool": [],  # Only 1 tool found it
            "disputed": [],  # Tools disagree on severity
        }

        for group in issue_groups.values():
            if len(group) >= 3:
                # Merge into single issue with high confidence
                merged = self._merge_issues(group)
                merged.confidence = 0.95
                merged.detected_by = [issue.tool for issue in group]
                consensus_issues["strong_consensus"].append(merged)
            elif len(group) == 2:
                merged = self._merge_issues(group)
                merged.confidence = 0.8
                merged.detected_by = [issue.tool for issue in group]
                consensus_issues["moderate_consensus"].append(merged)
            else:
                issue = group[0]
                issue.confidence = 0.6
                consensus_issues["single_tool"].append(issue)

        return consensus_issues

    def _merge_issues(self, issues: List[LintingIssue]) -> LintingIssue:
        """Merge similar issues from multiple tools."""
        # Use the most specific/detailed issue as base
        base_issue = max(issues, key=lambda x: len(x.message))

        # Combine information
        all_tools = [issue.tool for issue in issues]
        all_codes = [issue.rule_code for issue in issues]

        # Choose the most severe severity
        severities = [issue.severity for issue in issues]
        if "error" in severities:
            severity = "error"
        elif "warning" in severities:
            severity = "warning"
        else:
            severity = "info"

        return LintingIssue(
            file_path=base_issue.file_path,
            line_number=base_issue.line_number,
            column=base_issue.column,
            rule_code=f"{base_issue.rule_code}+{len(set(all_codes))-1}",  # Indicate multiple rules
            message=f"{base_issue.message} (detected by {len(all_tools)} tools)",
            severity=severity,
            tool="consensus",
            auto_fixable=any(issue.auto_fixable for issue in issues),
            confidence=min(
                1.0, 0.6 + 0.15 * len(all_tools)
            ),  # Higher confidence with more tools
            detected_by=all_tools,
        )

    def run_comprehensive_analysis(self, file_paths: List[Path]) -> ConsensusReport:
        """Run comprehensive multi-tool analysis."""
        print("🔍 Starting robust multi-tool linting analysis...")

        # Phase 1: Syntax validation (critical)
        print("🔧 Phase 1: Multi-method syntax validation...")
        syntax_results = []
        for file_path in file_paths:
            syntax_results.extend(self.validate_syntax_multiple(file_path))

        # Phase 2: Style and logic analysis
        print("🎨 Phase 2: Multi-tool style and logic analysis...")
        tool_results = []

        # Run available tools
        available_linters = [
            tool
            for tool in ["flake8", "black", "isort", "mypy", "pylint", "bandit"]
            if self.available_tools.get(tool, False)
        ]

        for tool in available_linters:
            print(f"  Running {tool}...")
            if tool == "flake8":
                result = self.run_tool_safely(
                    tool,
                    file_paths,
                    ["--format=%(path)s:%(row)d:%(col)d: %(code)s %(text)s"],
                )
            elif tool == "black":
                result = self.run_tool_safely(tool, file_paths, ["--check", "--diff"])
            elif tool == "isort":
                result = self.run_tool_safely(
                    tool, file_paths, ["--check-only", "--diff"]
                )
            elif tool == "mypy":
                result = self.run_tool_safely(tool, file_paths)
            elif tool == "pylint":
                result = self.run_tool_safely(
                    tool, file_paths, ["--output-format=json"]
                )
            elif tool == "bandit":
                result = self.run_tool_safely(tool, file_paths, ["--format", "json"])
            else:
                result = self.run_tool_safely(tool, file_paths)

            tool_results.append(result)

        # Combine with syntax results
        all_results = syntax_results + tool_results

        # Phase 3: Consensus analysis
        print("🤝 Phase 3: Cross-validation and consensus analysis...")
        consensus_issues = self.find_consensus_issues(all_results)

        # Phase 4: Generate comprehensive report
        print("📊 Phase 4: Generating consensus report...")

        successful_tools = [r.tool_name for r in all_results if r.success]
        failed_tools = [r.tool_name for r in all_results if not r.success]

        all_issues = []
        for category_issues in consensus_issues.values():
            all_issues.extend(category_issues)

        # Calculate agreement score
        total_possible_agreements = len(file_paths) * len(available_linters)
        actual_agreements = sum(
            len(group) for group in consensus_issues.values() if len(group) > 1
        )
        agreement_score = actual_agreements / max(total_possible_agreements, 1)

        # Calculate reliability score
        reliability_score = len(successful_tools) / max(len(all_results), 1)

        # Calculate health score with consensus weighting
        strong_consensus_weight = len(consensus_issues["strong_consensus"]) * 0.1
        moderate_consensus_weight = len(consensus_issues["moderate_consensus"]) * 0.05
        single_tool_weight = len(consensus_issues["single_tool"]) * 0.02

        total_weighted_issues = (
            strong_consensus_weight + moderate_consensus_weight + single_tool_weight
        )
        health_score = max(0, 100 - total_weighted_issues)

        report = ConsensusReport(
            timestamp=datetime.now(),
            total_files_checked=len(file_paths),
            tools_run=successful_tools,
            tools_failed=failed_tools,
            total_issues=len(all_issues),
            consensus_issues=len(consensus_issues["strong_consensus"])
            + len(consensus_issues["moderate_consensus"]),
            disputed_issues=len(consensus_issues["single_tool"]),
            tool_agreement_score=agreement_score,
            reliability_score=reliability_score,
            issues_by_consensus=consensus_issues,
            all_issues=all_issues,
            health_score=health_score,
        )

        return report

    def generate_consensus_report(self, report: ConsensusReport) -> str:
        """Generate detailed consensus report."""
        lines = []
        lines.append("=" * 80)
        lines.append("🔍 ROBUST MULTI-TOOL LINTING REPORT")
        lines.append("=" * 80)
        lines.append(f"📅 Generated: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"📁 Files analyzed: {report.total_files_checked}")
        lines.append(f"🛠️ Tools successful: {len(report.tools_run)}")
        lines.append(f"❌ Tools failed: {len(report.tools_failed)}")
        lines.append("")

        # Tool reliability
        lines.append("🎯 TOOL RELIABILITY:")
        lines.append(f"  Agreement Score: {report.tool_agreement_score:.1%}")
        lines.append(f"  Reliability Score: {report.reliability_score:.1%}")
        lines.append(f"  Overall Health: {report.health_score:.1f}/100")
        lines.append("")

        # Consensus analysis
        lines.append("🤝 CONSENSUS ANALYSIS:")
        lines.append(
            f"  Strong Consensus (3+ tools): {len(report.issues_by_consensus.get('strong_consensus', []))}"
        )
        lines.append(
            f"  Moderate Consensus (2 tools): {len(report.issues_by_consensus.get('moderate_consensus', []))}"
        )
        lines.append(
            f"  Single Tool Detection: {len(report.issues_by_consensus.get('single_tool', []))}"
        )
        lines.append(f"  Total Issues: {report.total_issues}")
        lines.append("")

        # Failed tools warning
        if report.tools_failed:
            lines.append("⚠️ FAILED TOOLS:")
            for tool in report.tools_failed:
                lines.append(f"  ❌ {tool}")
            lines.append("")

        # High confidence issues
        strong_consensus = report.issues_by_consensus.get("strong_consensus", [])
        if strong_consensus:
            lines.append("🔥 HIGH CONFIDENCE ISSUES (Multiple Tools Agree):")
            for issue in strong_consensus[:10]:  # Show top 10
                lines.append(
                    f"  {issue.file_path}:{issue.line_number} - {issue.rule_code}: {issue.message}"
                )
                lines.append(
                    f"    Detected by: {', '.join(issue.detected_by)} (confidence: {issue.confidence:.1%})"
                )
            if len(strong_consensus) > 10:
                lines.append(f"  ... and {len(strong_consensus) - 10} more")
            lines.append("")

        # Recommendations
        lines.append("💡 RECOMMENDATIONS:")
        if report.reliability_score < 0.8:
            lines.append(
                "  ⚠️ Low tool reliability - consider installing missing linters"
            )
        if len(strong_consensus) > 0:
            lines.append(
                f"  🔧 Address {len(strong_consensus)} high-confidence issues first"
            )
        if (
            len(report.issues_by_consensus.get("single_tool", []))
            > len(strong_consensus) * 2
        ):
            lines.append("  🤔 Many single-tool detections - manual review recommended")

        lines.append("=" * 80)

        return "\n".join(lines)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Robust Multi-Tool Linting Framework")
    parser.add_argument("files", nargs="*", help="Python files to analyze")
    parser.add_argument(
        "--scan-project", action="store_true", help="Scan entire project"
    )

    args = parser.parse_args()

    framework = RobustLintingFramework()

    if args.scan_project:
        # Scan entire project
        file_paths = list(Path.cwd().glob("src/**/*.py"))
        file_paths.extend(list(Path.cwd().glob("tools/**/*.py")))
        file_paths.extend(list(Path.cwd().glob("scripts/**/*.py")))
    else:
        file_paths = [Path(f) for f in args.files] if args.files else [Path.cwd()]

    # Filter to existing Python files
    python_files = [f for f in file_paths if f.exists() and f.suffix == ".py"]

    if not python_files:
        print("No Python files found to analyze")
        return

    report = framework.run_comprehensive_analysis(python_files)
    print(framework.generate_consensus_report(report))


if __name__ == "__main__":
    main()
