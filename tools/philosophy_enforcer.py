"""
QeMLflow Philosophy Enforcement System
====================================

Automated system to ensure codebase continues to align with core philosophy.
Implements continuous monitoring of philosophy compliance.
"""

import ast
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


@dataclass
class PhilosophyViolation:
    """Represents a violation of core philosophy principles."""

    principle: str
    severity: str  # "low", "medium", "high"
    file_path: str
    line_number: int
    description: str
    suggestion: str


class PhilosophyEnforcer:
    """
    Automated enforcement of QeMLflow core philosophy principles.

    Monitors code for violations of:
    - Lean core principles (minimal dependencies, clear separation)
    - Robust design (proper error handling, type hints)
    - Performance standards (import times, memory usage)
    - Scientific rigor (testing, documentation)
    """

    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.core_path = self.repo_path / "src" / "qemlflow" / "core"
        self.violations = []

        # Philosophy rules
        self.philosophy_rules = {
            "lean_core": self._check_lean_core_violations,
            "robust_design": self._check_robust_design_violations,
            "performance_standards": self._check_performance_violations,
            "scientific_rigor": self._check_scientific_rigor_violations,
            "modular_excellence": self._check_modular_excellence_violations,
        }

    def run_philosophy_audit(self) -> Dict[str, List[PhilosophyViolation]]:
        """Run comprehensive philosophy compliance audit."""
        print("🔍 Running QeMLflow Philosophy Compliance Audit...")

        audit_results = {}

        for principle, checker in self.philosophy_rules.items():
            print(f"   Checking {principle}...")
            violations = checker()
            audit_results[principle] = violations
            self.violations.extend(violations)

        return audit_results

    def _check_lean_core_violations(self) -> List[PhilosophyViolation]:
        """Check for violations of lean core principles."""
        violations = []

        # Check for wildcard imports in core
        for py_file in self.core_path.rglob("*.py"):
            with open(py_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for i, line in enumerate(lines, 1):
                if re.search(r"from .* import \*", line) and "__init__.py" not in str(
                    py_file
                ):
                    violations.append(
                        PhilosophyViolation(
                            principle="Lean Core",
                            severity="medium",
                            file_path=str(py_file),
                            line_number=i,
                            description="Wildcard import in core module",
                            suggestion="Use explicit imports for better namespace control",
                        )
                    )

        # Check core module sizes (should be focused)
        for py_file in self.core_path.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue

            line_count = sum(1 for _ in open(py_file, "r", encoding="utf-8"))
            if line_count > 1000:
                violations.append(
                    PhilosophyViolation(
                        principle="Lean Core",
                        severity="low",
                        file_path=str(py_file),
                        line_number=1,
                        description=f"Large core module ({line_count} lines)",
                        suggestion="Consider splitting into smaller, focused modules",
                    )
                )

        return violations

    def _check_robust_design_violations(self) -> List[PhilosophyViolation]:
        """Check for violations of robust design principles."""
        violations = []

        for py_file in self.core_path.rglob("*.py"):
            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    lines = content.splitlines()

                tree = ast.parse(content)

                # Check for bare except clauses
                for node in ast.walk(tree):
                    if isinstance(node, ast.ExceptHandler) and node.type is None:
                        violations.append(
                            PhilosophyViolation(
                                principle="Robust Design",
                                severity="high",
                                file_path=str(py_file),
                                line_number=node.lineno,
                                description="Bare except clause",
                                suggestion="Use specific exception types for better error handling",
                            )
                        )

                # Check for missing type hints in public methods
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and not node.name.startswith(
                        "_"
                    ):
                        if not node.returns and node.name != "__init__":
                            violations.append(
                                PhilosophyViolation(
                                    principle="Robust Design",
                                    severity="medium",
                                    file_path=str(py_file),
                                    line_number=node.lineno,
                                    description=f"Missing return type hint for {node.name}",
                                    suggestion="Add return type hint for better IDE support",
                                )
                            )

            except Exception as e:
                violations.append(
                    PhilosophyViolation(
                        principle="Robust Design",
                        severity="low",
                        file_path=str(py_file),
                        line_number=1,
                        description=f"Could not parse file: {e}",
                        suggestion="Check file syntax",
                    )
                )

        return violations

    def _check_performance_violations(self) -> List[PhilosophyViolation]:
        """Check for performance standard violations."""
        violations = []

        # Test import time
        try:
            result = subprocess.run(
                [
                    "python3",
                    "-c",
                    'import time; start=time.time(); import sys; sys.path.append("src"); import qemlflow; print(f"{time.time()-start:.2f}")',
                ],
                capture_output=True,
                text=True,
                cwd=self.repo_path,
            )

            if result.returncode == 0:
                import_time = float(result.stdout.strip().split()[-1])
                if import_time > 5.0:
                    violations.append(
                        PhilosophyViolation(
                            principle="Performance Standards",
                            severity="high",
                            file_path="src/qemlflow/__init__.py",
                            line_number=1,
                            description=f"Import time {import_time:.2f}s exceeds 5s target",
                            suggestion="Optimize lazy loading and reduce eager imports",
                        )
                    )
        except Exception as e:
            violations.append(
                PhilosophyViolation(
                    principle="Performance Standards",
                    severity="medium",
                    file_path="src/qemlflow/__init__.py",
                    line_number=1,
                    description=f"Could not measure import time: {e}",
                    suggestion="Check import structure",
                )
            )

        return violations

    def _check_scientific_rigor_violations(self) -> List[PhilosophyViolation]:
        """Check for scientific rigor violations."""
        violations = []

        # Check for missing docstrings in core classes
        for py_file in self.core_path.rglob("*.py"):
            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    content = f.read()

                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        if not ast.get_docstring(node):
                            violations.append(
                                PhilosophyViolation(
                                    principle="Scientific Rigor",
                                    severity="medium",
                                    file_path=str(py_file),
                                    line_number=node.lineno,
                                    description=f"Missing docstring for class {node.name}",
                                    suggestion="Add comprehensive docstring with usage examples",
                                )
                            )

            except Exception:
                continue

        return violations

    def _check_modular_excellence_violations(self) -> List[PhilosophyViolation]:
        """Check for modular excellence violations."""
        violations = []

        # Check for circular imports (simplified check)
        for py_file in self.core_path.rglob("*.py"):
            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    content = f.read()

                # Look for relative imports that might cause cycles
                if re.search(r"from \.\. import", content):
                    violations.append(
                        PhilosophyViolation(
                            principle="Modular Excellence",
                            severity="medium",
                            file_path=str(py_file),
                            line_number=1,
                            description="Potential circular import with parent package",
                            suggestion="Use absolute imports or restructure dependencies",
                        )
                    )

            except Exception:
                continue

        return violations

    def generate_philosophy_report(self) -> str:
        """Generate comprehensive philosophy compliance report."""
        if not self.violations:
            self.run_philosophy_audit()

        # Count violations by severity
        severity_counts = {"high": 0, "medium": 0, "low": 0}
        principle_counts = {}

        for violation in self.violations:
            severity_counts[violation.severity] += 1
            principle_counts[violation.principle] = (
                principle_counts.get(violation.principle, 0) + 1
            )

        # Calculate compliance score
        total_violations = len(self.violations)
        severity_weights = {"high": 3, "medium": 2, "low": 1}
        weighted_violations = sum(severity_weights[v.severity] for v in self.violations)
        max_possible_score = 100
        compliance_score = max(0, max_possible_score - weighted_violations)

        report = f"""
🧬 QeMLflow Philosophy Compliance Report
=========================================

📊 Overall Compliance Score: {compliance_score}/100

🎯 Violations Summary:
  • High Severity: {severity_counts['high']}
  • Medium Severity: {severity_counts['medium']}
  • Low Severity: {severity_counts['low']}
  • Total: {total_violations}

📋 Violations by Principle:"""

        for principle, count in principle_counts.items():
            report += f"\n  • {principle}: {count}"

        if self.violations:
            report += "\n\n🔍 Detailed Violations:\n"
            for i, violation in enumerate(self.violations[:10], 1):  # Show first 10
                report += f"\n{i}. {violation.principle} - {violation.severity.upper()}"
                report += f"\n   File: {violation.file_path}:{violation.line_number}"
                report += f"\n   Issue: {violation.description}"
                report += f"\n   Fix: {violation.suggestion}\n"

            if len(self.violations) > 10:
                report += f"\n... and {len(self.violations) - 10} more violations"
        else:
            report += "\n\n🎉 No philosophy violations found! Excellent compliance."

        report += f"\n\nGenerated: {subprocess.run(['date'], capture_output=True, text=True).stdout.strip()}"

        return report

    def get_philosophy_metrics(self) -> Dict[str, Any]:
        """Get quantitative philosophy compliance metrics."""
        if not self.violations:
            self.run_philosophy_audit()

        severity_counts = {"high": 0, "medium": 0, "low": 0}
        for violation in self.violations:
            severity_counts[violation.severity] += 1

        # Calculate compliance score
        severity_weights = {"high": 3, "medium": 2, "low": 1}
        weighted_violations = sum(severity_weights[v.severity] for v in self.violations)
        compliance_score = max(0, 100 - weighted_violations)

        return {
            "compliance_score": compliance_score,
            "total_violations": len(self.violations),
            "severity_breakdown": severity_counts,
            "principles_status": {
                "lean_core": "PASS"
                if severity_counts["high"] == 0
                else "NEEDS_ATTENTION",
                "robust_design": "PASS"
                if severity_counts["high"] == 0
                else "NEEDS_ATTENTION",
                "performance_standards": "PASS"
                if severity_counts["high"] == 0
                else "NEEDS_ATTENTION",
                "scientific_rigor": "PASS"
                if severity_counts["medium"] + severity_counts["high"] == 0
                else "NEEDS_ATTENTION",
                "modular_excellence": "PASS"
                if severity_counts["high"] == 0
                else "NEEDS_ATTENTION",
            },
        }


def run_philosophy_check() -> None:
    """Quick philosophy compliance check."""
    enforcer = PhilosophyEnforcer()
    print(enforcer.generate_philosophy_report())


if __name__ == "__main__":
    run_philosophy_check()
