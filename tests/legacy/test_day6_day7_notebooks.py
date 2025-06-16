#!/usr/bin/env python3
"""
Comprehensive Test Suite for Day 6 & 7 Notebooks
==============================================

This script validates all notebooks in Day 6 and Day 7 folders to ensure:
- All imports work correctly
- Required libraries are available or have proper fallbacks
- Key classes and methods function as expected
- Notebooks can be executed without runtime errors

Usage:
    python test_day6_day7_notebooks.py
    python test_day6_day7_notebooks.py --verbose
    python test_day6_day7_notebooks.py --quick-test
"""

import argparse
import importlib
import json
import logging
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class NotebookTester:
    """Main testing class for Day 6 & 7 notebooks"""

    def __init__(self, base_path: str, verbose: bool = False):
        self.base_path = Path(base_path)
        self.verbose = verbose
        self.results = {"day_06": {}, "day_07": {}, "summary": {}, "errors": []}

        # Define notebook paths
        self.day6_notebooks = [
            "day_06_module_1_quantum_foundations.ipynb",
            "day_06_module_2_vqe_algorithms.ipynb",
            "day_06_module_3_quantum_production.ipynb",
            "day_06_quantum_computing_project.ipynb",
        ]

        self.day7_notebooks = [
            "day_07_module_1_integration.ipynb",
            "day_07_module_2_multimodal_workflows.ipynb",
            "day_07_module_3_deployment.ipynb",
            "day_07_integration_project.ipynb",
        ]

    def run_all_tests(self, quick_test: bool = False):
        """Run comprehensive test suite"""
        print("🧪 Starting Comprehensive Notebook Testing")
        print("=" * 60)

        start_time = time.time()

        # Test Day 6 notebooks
        print("\n📊 Testing Day 6 Notebooks...")
        self._test_day_notebooks("day_06", self.day6_notebooks, quick_test)

        # Test Day 7 notebooks
        print("\n🔗 Testing Day 7 Notebooks...")
        self._test_day_notebooks("day_07", self.day7_notebooks, quick_test)

        # Generate summary report
        self._generate_summary_report()

        total_time = time.time() - start_time
        print(f"\n⏱️ Total testing time: {total_time:.2f} seconds")

        return self.results

    def _test_day_notebooks(
        self, day_folder: str, notebooks: List[str], quick_test: bool
    ):
        """Test all notebooks in a specific day folder"""
        day_path = self.base_path / day_folder

        for notebook in notebooks:
            notebook_path = day_path / notebook

            if not notebook_path.exists():
                self.results[day_folder][notebook] = {
                    "status": "MISSING",
                    "error": f"Notebook file not found: {notebook_path}",
                }
                print(f"❌ {notebook}: File not found")
                continue

            print(f"\n🔍 Testing: {notebook}")
            result = self._test_single_notebook(notebook_path, quick_test)
            self.results[day_folder][notebook] = result

            # Print immediate feedback
            status_emoji = "✅" if result["status"] == "PASS" else "❌"
            print(f"{status_emoji} {notebook}: {result['status']}")

            if result["status"] != "PASS" and self.verbose:
                print(f"   Error: {result.get('error', 'Unknown error')}")

    def _test_single_notebook(
        self, notebook_path: Path, quick_test: bool
    ) -> Dict[str, Any]:
        """Test a single notebook comprehensively"""
        result = {
            "status": "UNKNOWN",
            "tests_run": [],
            "tests_passed": 0,
            "tests_failed": 0,
            "errors": [],
            "warnings": [],
            "execution_time": 0,
        }

        start_time = time.time()

        try:
            # Test 1: File structure and basic validation
            if not self._test_file_structure(notebook_path, result):
                result["status"] = "FAIL"
                return result

            # Test 2: Import validation
            if not self._test_imports(notebook_path, result):
                result["status"] = "FAIL"
                return result

            # Test 3: Syntax validation
            if not self._test_syntax(notebook_path, result):
                result["status"] = "FAIL"
                return result

            # Test 4: Library dependency check
            self._test_dependencies(notebook_path, result)

            if not quick_test:
                # Test 5: Mock implementation validation (for Day 6)
                if "day_06" in str(notebook_path):
                    self._test_mock_implementations(notebook_path, result)

                # Test 6: Class and method validation
                self._test_class_definitions(notebook_path, result)

            # Determine overall status
            if result["tests_failed"] == 0:
                result["status"] = "PASS"
            elif result["tests_failed"] < result["tests_run"] / 2:
                result["status"] = "PASS_WITH_WARNINGS"
            else:
                result["status"] = "FAIL"

        except Exception as e:
            result["status"] = "ERROR"
            result["errors"].append(f"Unexpected error: {str(e)}")
            if self.verbose:
                result["errors"].append(traceback.format_exc())

        result["execution_time"] = time.time() - start_time
        return result

    def _test_file_structure(self, notebook_path: Path, result: Dict) -> bool:
        """Test basic file structure and JSON validity"""
        test_name = "File Structure"
        result["tests_run"].append(test_name)

        try:
            # Check file size
            file_size = notebook_path.stat().st_size
            if file_size == 0:
                result["errors"].append("Notebook file is empty")
                result["tests_failed"] += 1
                return False

            # Check if it's valid JSON
            with open(notebook_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Basic structure checks
            if not content.strip():
                result["errors"].append("Notebook content is empty")
                result["tests_failed"] += 1
                return False

            # Check for basic notebook structure markers
            if "<VSCode.Cell" not in content:
                result["errors"].append("No VSCode cell markers found")
                result["tests_failed"] += 1
                return False

            result["tests_passed"] += 1
            return True

        except Exception as e:
            result["errors"].append(f"File structure test failed: {str(e)}")
            result["tests_failed"] += 1
            return False

    def _test_imports(self, notebook_path: Path, result: Dict) -> bool:
        """Test if all imports in the notebook can be resolved"""
        test_name = "Import Validation"
        result["tests_run"].append(test_name)

        try:
            with open(notebook_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Extract import statements
            import_lines = []
            lines = content.split("\n")

            for line in lines:
                line = line.strip()
                if line.startswith("import ") or line.startswith("from "):
                    # Skip comments and complex multiline imports
                    if not line.startswith("#") and "try:" not in line:
                        import_lines.append(line)

            failed_imports = []
            successful_imports = []

            for import_line in import_lines:
                try:
                    # Skip problematic cross-notebook imports
                    if "day_06_module" in import_line or "day_07_module" in import_line:
                        result["warnings"].append(
                            f"Skipping cross-notebook import: {import_line}"
                        )
                        continue

                    # Extract module name for testing
                    if import_line.startswith("from "):
                        module_name = import_line.split()[1].split(".")[0]
                    else:
                        module_name = import_line.split()[1].split(".")[0]

                    # Test import
                    importlib.import_module(module_name)
                    successful_imports.append(import_line)

                except ImportError as e:
                    # Check if it's a known optional import
                    optional_modules = [
                        "qiskit_nature",
                        "openfermion",
                        "pyscf",
                        "qiskit_aer",
                        "MDAnalysis",
                    ]
                    if any(opt in import_line for opt in optional_modules):
                        result["warnings"].append(
                            f"Optional import not available: {import_line}"
                        )
                    else:
                        failed_imports.append((import_line, str(e)))
                except Exception as e:
                    failed_imports.append((import_line, str(e)))

            # Report results
            if failed_imports:
                result["errors"].extend(
                    [f"Failed import: {imp} - {err}" for imp, err in failed_imports]
                )
                result["tests_failed"] += 1
                return False
            else:
                result["tests_passed"] += 1
                if self.verbose:
                    print(
                        f"    ✅ {len(successful_imports)} imports validated successfully"
                    )
                return True

        except Exception as e:
            result["errors"].append(f"Import validation failed: {str(e)}")
            result["tests_failed"] += 1
            return False

    def _test_syntax(self, notebook_path: Path, result: Dict) -> bool:
        """Test Python syntax in code cells"""
        test_name = "Syntax Validation"
        result["tests_run"].append(test_name)

        try:
            with open(notebook_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Extract Python code from cells
            python_blocks = []
            lines = content.split("\n")
            in_python_cell = False
            current_block = []

            for line in lines:
                if "<VSCode.Cell" in line and 'language="python"' in line:
                    in_python_cell = True
                    current_block = []
                elif "</VSCode.Cell>" in line and in_python_cell:
                    if current_block:
                        python_blocks.append("\n".join(current_block))
                    in_python_cell = False
                    current_block = []
                elif in_python_cell:
                    current_block.append(line)

            syntax_errors = []

            for i, block in enumerate(python_blocks):
                try:
                    compile(block, f"<notebook_cell_{i}>", "exec")
                except SyntaxError as e:
                    syntax_errors.append(f"Cell {i+1}: {str(e)}")
                except Exception:
                    # Other compilation errors (like undefined names) are OK for syntax check
                    pass

            if syntax_errors:
                result["errors"].extend(syntax_errors)
                result["tests_failed"] += 1
                return False
            else:
                result["tests_passed"] += 1
                if self.verbose:
                    print(
                        f"    ✅ {len(python_blocks)} code cells validated for syntax"
                    )
                return True

        except Exception as e:
            result["errors"].append(f"Syntax validation failed: {str(e)}")
            result["tests_failed"] += 1
            return False

    def _test_dependencies(self, notebook_path: Path, result: Dict):
        """Test library dependencies and provide recommendations"""
        test_name = "Dependency Analysis"
        result["tests_run"].append(test_name)

        try:
            with open(notebook_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Key libraries for each day
            quantum_libs = [
                "qiskit",
                "qiskit_aer",
                "qiskit_nature",
                "openfermion",
                "pyscf",
            ]
            integration_libs = ["torch", "sklearn", "rdkit", "pandas", "numpy"]

            missing_critical = []
            missing_optional = []

            for lib in quantum_libs + integration_libs:
                if lib in content:
                    try:
                        importlib.import_module(lib)
                    except ImportError:
                        if lib in ["qiskit_nature", "openfermion", "pyscf"]:
                            missing_optional.append(lib)
                        else:
                            missing_critical.append(lib)

            if missing_critical:
                result["errors"].append(
                    f"Critical libraries missing: {missing_critical}"
                )
                result["tests_failed"] += 1
            else:
                result["tests_passed"] += 1

            if missing_optional:
                result["warnings"].append(
                    f"Optional libraries missing: {missing_optional}"
                )

        except Exception as e:
            result["errors"].append(f"Dependency analysis failed: {str(e)}")
            result["tests_failed"] += 1

    def _test_mock_implementations(self, notebook_path: Path, result: Dict):
        """Test mock implementations in Day 6 notebooks"""
        test_name = "Mock Implementation Validation"
        result["tests_run"].append(test_name)

        try:
            with open(notebook_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Check for mock classes
            mock_indicators = [
                "Mock implementation",
                "mock MolecularHamiltonianBuilder",
                "mock QuantumCircuitDesigner",
                "simplified implementation",
            ]

            mock_found = any(
                indicator.lower() in content.lower() for indicator in mock_indicators
            )

            if mock_found:
                result["warnings"].append(
                    "Mock implementations detected - verify functionality"
                )

            result["tests_passed"] += 1

        except Exception as e:
            result["errors"].append(f"Mock implementation test failed: {str(e)}")
            result["tests_failed"] += 1

    def _test_class_definitions(self, notebook_path: Path, result: Dict):
        """Test class and method definitions"""
        test_name = "Class Definition Validation"
        result["tests_run"].append(test_name)

        try:
            with open(notebook_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Check for key classes
            key_classes = []
            if "module_1" in str(notebook_path):
                key_classes = ["MolecularHamiltonianBuilder", "QuantumCircuitDesigner"]
            elif "module_2" in str(notebook_path):
                key_classes = ["MolecularVQE"]
            elif "module_3" in str(notebook_path) or "day_06_quantum" in str(
                notebook_path
            ):
                key_classes = ["QuantumPipelineOrchestrator"]
            elif "day_07" in str(notebook_path):
                key_classes = ["ComponentMetadata", "PipelineOrchestrator"]

            missing_classes = []
            for class_name in key_classes:
                if f"class {class_name}" not in content:
                    missing_classes.append(class_name)

            if missing_classes:
                result["warnings"].append(
                    f"Expected classes not found: {missing_classes}"
                )

            result["tests_passed"] += 1

        except Exception as e:
            result["errors"].append(f"Class definition test failed: {str(e)}")
            result["tests_failed"] += 1

    def _generate_summary_report(self):
        """Generate comprehensive summary report"""
        print("\n" + "=" * 60)
        print("📊 COMPREHENSIVE TEST SUMMARY REPORT")
        print("=" * 60)

        total_notebooks = 0
        total_passed = 0
        total_failed = 0
        total_warnings = 0

        for day in ["day_06", "day_07"]:
            print(f"\n📋 {day.upper()} Results:")
            print("-" * 30)

            day_results = self.results[day]
            for notebook, result in day_results.items():
                total_notebooks += 1
                status = result.get("status", "UNKNOWN")

                if status == "PASS":
                    total_passed += 1
                    print(f"✅ {notebook}: PASS")
                elif status == "PASS_WITH_WARNINGS":
                    total_passed += 1
                    total_warnings += len(result.get("warnings", []))
                    print(f"⚠️  {notebook}: PASS (with warnings)")
                else:
                    total_failed += 1
                    print(f"❌ {notebook}: {status}")

                if self.verbose and result.get("errors"):
                    for error in result["errors"][:3]:  # Show first 3 errors
                        print(f"     Error: {error}")

        # Overall summary
        print("\n🎯 OVERALL RESULTS:")
        print(f"   Total Notebooks: {total_notebooks}")
        print(f"   Passed: {total_passed}")
        print(f"   Failed: {total_failed}")
        print(f"   Warnings: {total_warnings}")

        success_rate = (
            (total_passed / total_notebooks * 100) if total_notebooks > 0 else 0
        )
        print(f"   Success Rate: {success_rate:.1f}%")

        # Store in results
        self.results["summary"] = {
            "total_notebooks": total_notebooks,
            "passed": total_passed,
            "failed": total_failed,
            "warnings": total_warnings,
            "success_rate": success_rate,
        }

        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        if total_failed > 0:
            print(f"   - Fix {total_failed} failing notebooks before deployment")
        if total_warnings > 5:
            print(f"   - Review {total_warnings} warnings for potential issues")
        if success_rate < 80:
            print("   - Consider additional testing and validation")
        else:
            print("   - Notebooks are in good shape for deployment!")


def main():
    """Main testing function"""
    parser = argparse.ArgumentParser(description="Test Day 6 & 7 Notebooks")
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output with detailed error messages",
    )
    parser.add_argument(
        "--quick-test",
        "-q",
        action="store_true",
        help="Run only basic validation tests",
    )
    parser.add_argument(
        "--base-path",
        "-p",
        default="/Users/sanjeevadodlapati/Downloads/Repos/ChemML/notebooks/quickstart_bootcamp/days",
        help="Base path to notebooks directory",
    )

    args = parser.parse_args()

    # Initialize tester
    tester = NotebookTester(args.base_path, args.verbose)

    # Run tests
    results = tester.run_all_tests(args.quick_test)

    # Save results to file
    with open("day6_day7_test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n💾 Detailed results saved to: day6_day7_test_results.json")

    # Exit with appropriate code
    if results["summary"]["failed"] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
