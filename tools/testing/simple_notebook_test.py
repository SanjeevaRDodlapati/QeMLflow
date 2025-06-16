"""
Simple and Robust Notebook Validator
====================================

Quick validation of Day 6 and Day 7 notebooks for:
- File structure and JSON validity
- Import statements and dependencies
- Basic syntax validation
- Mock implementation detection

Usage:
    python simple_notebook_test.py [--quick]
"""

import ast
import json
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple


class SimpleNotebookValidator:
    def __init__(self):
        self.base_path = Path("notebooks/quickstart_bootcamp/days")
        self.day6_path = self.base_path / "day_06"
        self.day7_path = self.base_path / "day_07"

        # Track results
        self.results = {
            "total_tested": 0,
            "passed": 0,
            "failed": 0,
            "details": {},
            "summary": [],
        }

        # Known quantum libraries that should be optional
        self.quantum_libs = {
            "qiskit",
            "openfermion",
            "pyscf",
            "cirq",
            "pennylane",
            "pyquil",
            "braket",
            "tensorflow_quantum",
            "qulacs",
        }

    def load_notebook(self, notebook_path: Path) -> Tuple[bool, Dict]:
        """Load and validate notebook JSON structure"""
        try:
            with open(notebook_path, "r", encoding="utf-8") as f:
                notebook = json.load(f)

            # Basic structure validation
            required_fields = ["cells", "metadata", "nbformat"]
            for field in required_fields:
                if field not in notebook:
                    return False, {"error": f"Missing {field}"}

            return True, notebook
        except Exception as e:
            return False, {"error": f"Failed to load: {str(e)}"}

    def extract_code_cells(self, notebook: Dict) -> List[str]:
        """Extract code from all code cells"""
        code_cells = []
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") == "code":
                source = cell.get("source", [])
                if isinstance(source, list):
                    code = "".join(source)
                else:
                    code = str(source)
                code_cells.append(code)
        return code_cells

    def check_syntax(self, code: str) -> Tuple[bool, str]:
        """Check if code has valid Python syntax"""
        if not code.strip():
            return True, "Empty cell"

        try:
            ast.parse(code)
            return True, "Valid syntax"
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        except Exception as e:
            return False, f"Parse error: {e}"

    def extract_imports(self, code: str) -> List[str]:
        """Extract import statements from code"""
        imports = []

        # Use regex for simple import extraction
        patterns = [r"^import\\s+([^\\s#]+)", r"^from\\s+([^\\s#]+)\\s+import"]

        for line in code.split("\\n"):
            line = line.strip()
            for pattern in patterns:
                match = re.match(pattern, line)
                if match:
                    import_name = match.group(1)
                    imports.append(import_name)

        return imports

    def analyze_imports(self, imports: List[str]) -> Dict[str, List[str]]:
        """Categorize imports"""
        analysis = {"standard": [], "quantum": [], "problematic": [], "local": []}

        for imp in imports:
            base_imp = imp.split(".")[0]

            # Check for cross-notebook imports
            if ".ipynb" in imp or "day_0" in imp:
                analysis["problematic"].append(imp)
            elif base_imp in self.quantum_libs:
                analysis["quantum"].append(imp)
            elif base_imp in ["chemml", "src"]:
                analysis["local"].append(imp)
            else:
                analysis["standard"].append(imp)

        return analysis

    def check_mocks(self, code: str) -> List[str]:
        """Check for mock implementations"""
        mock_indicators = []

        patterns = [
            r"class\\s+Mock\\w*",
            r"def\\s+mock_\\w*",
            r"#\\s*Mock\\s+implementation",
            r"#\\s*Placeholder",
            r"raise\\s+NotImplementedError",
            r"pass\\s*#.*[Mm]ock",
        ]

        for pattern in patterns:
            if re.search(pattern, code, re.IGNORECASE):
                mock_indicators.append(pattern)

        return mock_indicators

    def validate_notebook(self, notebook_path: Path) -> Dict[str, Any]:
        """Validate a single notebook"""
        print(f"\\n📓 Testing: {notebook_path.name}")

        result = {
            "name": notebook_path.name,
            "path": str(notebook_path),
            "status": "UNKNOWN",
            "issues": [],
            "warnings": [],
            "stats": {
                "total_cells": 0,
                "code_cells": 0,
                "syntax_errors": 0,
                "import_issues": 0,
                "mock_implementations": 0,
            },
        }

        # Load notebook
        loaded, notebook_data = self.load_notebook(notebook_path)
        if not loaded:
            result["status"] = "FAIL"
            result["issues"].append(notebook_data["error"])
            return result

        # Extract code cells
        code_cells = self.extract_code_cells(notebook_data)
        result["stats"]["total_cells"] = len(notebook_data.get("cells", []))
        result["stats"]["code_cells"] = len(code_cells)

        # Process each code cell
        for i, code in enumerate(code_cells):
            cell_num = i + 1

            # Check syntax
            syntax_ok, syntax_msg = self.check_syntax(code)
            if not syntax_ok:
                result["issues"].append(f"Cell {cell_num}: {syntax_msg}")
                result["stats"]["syntax_errors"] += 1

            # Check imports
            imports = self.extract_imports(code)
            if imports:
                import_analysis = self.analyze_imports(imports)

                # Report problematic imports
                if import_analysis["problematic"]:
                    for imp in import_analysis["problematic"]:
                        result["issues"].append(
                            f"Cell {cell_num}: Problematic import '{imp}'"
                        )
                        result["stats"]["import_issues"] += 1

                # Check quantum imports for try/except
                if import_analysis["quantum"]:
                    has_try_except = "try:" in code and "except" in code
                    if not has_try_except:
                        for imp in import_analysis["quantum"]:
                            result["warnings"].append(
                                f"Cell {cell_num}: Quantum import '{imp}' not in try/except"
                            )

            # Check for mocks
            mocks = self.check_mocks(code)
            if mocks:
                result["stats"]["mock_implementations"] += len(mocks)

        # Determine final status
        if result["stats"]["syntax_errors"] > 0 or result["stats"]["import_issues"] > 0:
            result["status"] = "FAIL"
        else:
            result["status"] = "PASS"

        return result

    def run_validation(self) -> Dict[str, Any]:
        """Run validation on all notebooks"""
        print("🔍 Starting Simple Notebook Validation")
        print("=" * 50)

        # Collect all notebooks
        notebooks = []
        for day_path in [self.day6_path, self.day7_path]:
            if day_path.exists():
                notebooks.extend(sorted(day_path.glob("*.ipynb")))

        if not notebooks:
            print("❌ No notebooks found!")
            return self.results

        print(f"Found {len(notebooks)} notebooks to validate")

        # Validate each notebook
        for notebook_path in notebooks:
            try:
                result = self.validate_notebook(notebook_path)
                self.results["details"][notebook_path.name] = result
                self.results["total_tested"] += 1

                if result["status"] == "PASS":
                    self.results["passed"] += 1
                    print(f"✅ {notebook_path.name}: PASS")
                else:
                    self.results["failed"] += 1
                    print(f"❌ {notebook_path.name}: FAIL")
                    if result["issues"]:
                        print(f"   Issues: {len(result['issues'])}")
                        for issue in result["issues"][:3]:  # Show first 3 issues
                            print(f"     • {issue}")
                        if len(result["issues"]) > 3:
                            print(f"     ... and {len(result['issues']) - 3} more")

                if result["warnings"]:
                    print(f"   ⚠️  Warnings: {len(result['warnings'])}")

            except Exception as e:
                print(f"❌ Error testing {notebook_path.name}: {e}")
                self.results["failed"] += 1

        return self.results

    def generate_report(self) -> str:
        """Generate final validation report"""
        lines = []
        lines.append("\\n" + "=" * 60)
        lines.append("NOTEBOOK VALIDATION REPORT")
        lines.append("=" * 60)
        lines.append(f"Total Notebooks: {self.results['total_tested']}")
        lines.append(f"Passed: {self.results['passed']}")
        lines.append(f"Failed: {self.results['failed']}")

        if self.results["total_tested"] > 0:
            success_rate = (self.results["passed"] / self.results["total_tested"]) * 100
            lines.append(f"Success Rate: {success_rate:.1f}%")

        lines.append("\\n" + "-" * 40)
        lines.append("DETAILED RESULTS:")
        lines.append("-" * 40)

        for name, result in self.results["details"].items():
            lines.append(f"\\n📓 {name}: {result['status']}")
            stats = result["stats"]
            lines.append(f"   Code cells: {stats['code_cells']}")
            lines.append(f"   Syntax errors: {stats['syntax_errors']}")
            lines.append(f"   Import issues: {stats['import_issues']}")
            lines.append(f"   Mock implementations: {stats['mock_implementations']}")
            lines.append(f"   Warnings: {len(result['warnings'])}")

        lines.append("\\n" + "-" * 40)
        lines.append("RECOMMENDATIONS:")
        lines.append("-" * 40)

        if self.results["failed"] == 0:
            lines.append("✅ All notebooks passed basic validation!")
            lines.append("• Notebooks should be ready for educational use")
        else:
            lines.append("⚠️  Some notebooks have issues that need attention:")
            lines.append("• Fix syntax errors before deployment")
            lines.append("• Resolve problematic import statements")

        lines.append("• Ensure quantum library imports use try/except blocks")
        lines.append("• Test notebooks in clean environments")
        lines.append(
            "• Consider adding more mock implementations for missing libraries"
        )

        return "\\n".join(lines)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Validate Day 6 and Day 7 notebooks")
    parser.add_argument("--quick", action="store_true", help="Quick validation mode")
    _args = parser.parse_args()

    # Run validation
    validator = SimpleNotebookValidator()
    results = validator.run_validation()

    # Generate and display report
    report = validator.generate_report()
    print(report)

    # Save report
    with open("notebook_validation_report.txt", "w") as f:
        f.write(report)
    print("\\n📄 Report saved to: notebook_validation_report.txt")

    # Exit with appropriate code
    sys.exit(0 if results["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
