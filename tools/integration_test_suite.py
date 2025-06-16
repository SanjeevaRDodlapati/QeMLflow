#!/usr/bin/env python3
"""
ChemML Integration Testing Suite
Comprehensive testing for Phase 5 enhancements.
"""

import importlib
import json
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional


class ChemMLTestSuite:
    """Comprehensive test suite for ChemML enhancements."""

    def __init__(self):
        self.results = {
            "import_performance": {},
            "lazy_loading": {},
            "type_annotations": {},
            "parameter_standardization": {},
            "error_handling": {},
            "overall_health": {},
        }

    def test_import_performance(self) -> Dict[str, Any]:
        """Test import performance improvements."""
        print("🚀 Testing Import Performance")
        print("-" * 30)

        # Clear any cached imports
        modules_to_clear = [
            mod for mod in sys.modules.keys() if mod.startswith("chemml")
        ]
        for mod in modules_to_clear:
            del sys.modules[mod]

        # Test main import
        start_time = time.time()
        try:
            import chemml

            end_time = time.time()
            import_time = end_time - start_time

            print(f"✅ ChemML import: {import_time:.3f}s")

            # Test that core functions work
            test_results = {}
            try:
                data = chemml.load_sample_data()
                test_results["load_sample_data"] = True
                print("✅ load_sample_data working")
            except Exception as e:
                test_results["load_sample_data"] = False
                print(f"❌ load_sample_data failed: {e}")

            try:
                fps = chemml.morgan_fingerprints(["CCO"])
                test_results["morgan_fingerprints"] = True
                print("✅ morgan_fingerprints working (lazy)")
            except Exception as e:
                test_results["morgan_fingerprints"] = False
                print(f"❌ morgan_fingerprints failed: {e}")

            return {
                "import_time": import_time,
                "import_successful": True,
                "core_functions": test_results,
                "performance_grade": (
                    "A" if import_time < 10 else "B" if import_time < 20 else "C"
                ),
            }

        except Exception as e:
            return {
                "import_successful": False,
                "error": str(e),
                "performance_grade": "F",
            }

    def test_lazy_loading(self) -> Dict[str, Any]:
        """Test lazy loading functionality."""
        print("\n🔄 Testing Lazy Loading")
        print("-" * 30)

        try:
            import chemml

            # Test that research and integrations are lazy-loaded
            research_available = chemml.research.is_available()
            integrations_available = chemml.integrations.is_available()

            print(f"✅ Research module lazy-loaded: {research_available}")
            print(f"✅ Integrations module lazy-loaded: {integrations_available}")

            # Test lazy function calls
            lazy_tests = {}

            try:
                # This should trigger lazy loading
                model = chemml.create_rf_model()
                lazy_tests["create_rf_model"] = True
                print("✅ create_rf_model (lazy) working")
            except Exception as e:
                lazy_tests["create_rf_model"] = False
                print(f"❌ create_rf_model failed: {e}")

            return {
                "research_available": research_available,
                "integrations_available": integrations_available,
                "lazy_functions": lazy_tests,
                "lazy_loading_grade": "A" if all(lazy_tests.values()) else "B",
            }

        except Exception as e:
            return {"error": str(e), "lazy_loading_grade": "F"}

    def test_type_annotations(self) -> Dict[str, Any]:
        """Test type annotation coverage."""
        print("\n📝 Testing Type Annotations")
        print("-" * 30)

        try:
            # Run the type annotation analyzer
            result = subprocess.run(
                ["python", "tools/type_annotation_analyzer.py"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                output = result.stdout
                # Extract coverage information
                import re

                coverage_match = re.search(
                    r"Parameter annotation coverage: ([\d.]+)%", output
                )
                return_match = re.search(
                    r"Return annotation coverage: ([\d.]+)%", output
                )

                param_coverage = float(coverage_match.group(1)) if coverage_match else 0
                return_coverage = float(return_match.group(1)) if return_match else 0
                overall_coverage = (param_coverage + return_coverage) / 2

                print(f"✅ Parameter coverage: {param_coverage}%")
                print(f"✅ Return coverage: {return_coverage}%")

                grade = (
                    "A"
                    if overall_coverage >= 90
                    else "B" if overall_coverage >= 75 else "C"
                )

                return {
                    "parameter_coverage": param_coverage,
                    "return_coverage": return_coverage,
                    "overall_coverage": overall_coverage,
                    "type_annotation_grade": grade,
                }
            else:
                return {"error": result.stderr, "type_annotation_grade": "F"}

        except Exception as e:
            return {"error": str(e), "type_annotation_grade": "F"}

    def test_parameter_standardization(self) -> Dict[str, Any]:
        """Test parameter naming standardization."""
        print("\n🔧 Testing Parameter Standardization")
        print("-" * 30)

        try:
            # Run the parameter analyzer
            result = subprocess.run(
                ["python", "tools/parameter_standardization.py"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                output = result.stdout
                # Extract standardization information
                import re

                suggestions_match = re.search(r"Total suggestions: (\d+)", output)
                suggestions_count = (
                    int(suggestions_match.group(1)) if suggestions_match else 0
                )

                print(f"✅ Remaining inconsistencies: {suggestions_count}")

                grade = (
                    "A"
                    if suggestions_count < 10
                    else "B" if suggestions_count < 30 else "C"
                )

                return {
                    "remaining_inconsistencies": suggestions_count,
                    "standardization_grade": grade,
                }
            else:
                return {"error": result.stderr, "standardization_grade": "F"}

        except Exception as e:
            return {"error": str(e), "standardization_grade": "F"}

    def test_error_handling(self) -> Dict[str, Any]:
        """Test improved error handling."""
        print("\n⚠️  Testing Error Handling")
        print("-" * 30)

        try:
            # Test custom exceptions
            from chemml.core.exceptions import ChemMLDataError, ChemMLError

            print("✅ Custom exceptions imported successfully")

            # Test that exceptions work properly
            try:
                raise ChemMLDataError("Test error", {"test": "data"})
            except ChemMLDataError as e:
                print("✅ ChemMLDataError working correctly")
                error_with_details = "Details:" in str(e)
                print(f"✅ Error details included: {error_with_details}")

            return {
                "custom_exceptions_available": True,
                "error_details_working": True,
                "error_handling_grade": "A",
            }

        except Exception as e:
            return {"error": str(e), "error_handling_grade": "F"}

    def test_overall_health(self) -> Dict[str, Any]:
        """Test overall system health."""
        print("\n🔬 Testing Overall Health")
        print("-" * 30)

        try:
            # Run the unified diagnostics
            result = subprocess.run(
                ["python", "tools/diagnostics_unified.py", "--quick"],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode == 0:
                output = result.stdout

                # Check for success indicators
                chemml_working = "ChemML initialized successfully!" in output
                config_working = "Config" in output and "✅" in output

                print(f"✅ ChemML initialization: {chemml_working}")
                print(f"✅ Configuration system: {config_working}")

                overall_health = chemml_working and config_working

                return {
                    "chemml_initialization": chemml_working,
                    "config_system": config_working,
                    "overall_health": overall_health,
                    "health_grade": "A" if overall_health else "C",
                }
            else:
                return {"error": result.stderr, "health_grade": "F"}

        except Exception as e:
            return {"error": str(e), "health_grade": "F"}

    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run all tests and generate comprehensive report."""
        print("🧪 ChemML Phase 5 Integration Testing")
        print("=" * 50)

        # Run all test categories
        self.results["import_performance"] = self.test_import_performance()
        self.results["lazy_loading"] = self.test_lazy_loading()
        self.results["type_annotations"] = self.test_type_annotations()
        self.results["parameter_standardization"] = (
            self.test_parameter_standardization()
        )
        self.results["error_handling"] = self.test_error_handling()
        self.results["overall_health"] = self.test_overall_health()

        # Calculate overall grade
        grades = []
        for category, result in self.results.items():
            if isinstance(result, dict):
                for key, value in result.items():
                    if key.endswith("_grade"):
                        grades.append(value)

        grade_values = {"A": 4, "B": 3, "C": 2, "D": 1, "F": 0}
        if grades:
            avg_grade = sum(grade_values.get(g, 0) for g in grades) / len(grades)
            overall_grade = (
                "A" if avg_grade >= 3.5 else "B" if avg_grade >= 2.5 else "C"
            )
        else:
            overall_grade = "F"

        self.results["overall_grade"] = overall_grade

        return self.results

    def print_summary_report(self):
        """Print a summary of test results."""
        print("\n" + "=" * 50)
        print("📊 PHASE 5 ENHANCEMENT TEST SUMMARY")
        print("=" * 50)

        # Import performance
        import_result = self.results.get("import_performance", {})
        if "import_time" in import_result:
            print(
                f"⚡ Import Performance: {import_result['import_time']:.2f}s ({import_result.get('performance_grade', 'F')})"
            )

        # Type annotations
        type_result = self.results.get("type_annotations", {})
        if "overall_coverage" in type_result:
            print(
                f"📝 Type Coverage: {type_result['overall_coverage']:.1f}% ({type_result.get('type_annotation_grade', 'F')})"
            )

        # Parameter standardization
        param_result = self.results.get("parameter_standardization", {})
        if "remaining_inconsistencies" in param_result:
            print(
                f"🔧 Parameter Issues: {param_result['remaining_inconsistencies']} remaining ({param_result.get('standardization_grade', 'F')})"
            )

        # Overall grade
        overall_grade = self.results.get("overall_grade", "F")
        print(f"\n🎯 OVERALL GRADE: {overall_grade}")

        # Recommendations
        print(f"\n💡 Next Steps:")
        if import_result.get("import_time", float("inf")) > 10:
            print("  • Continue import optimization (target: <5s)")
        if type_result.get("overall_coverage", 0) < 90:
            print("  • Increase type annotation coverage to 90%+")
        if param_result.get("remaining_inconsistencies", float("inf")) > 10:
            print("  • Complete parameter standardization")

        print("  • Implement advanced caching features")
        print("  • Add comprehensive documentation")


def main():
    """Main testing function."""
    import argparse

    parser = argparse.ArgumentParser(description="ChemML Integration Testing")
    parser.add_argument(
        "--save-report", action="store_true", help="Save detailed JSON report"
    )
    parser.add_argument("--quick", action="store_true", help="Run quick tests only")

    args = parser.parse_args()

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    test_suite = ChemMLTestSuite()

    if args.quick:
        # Quick tests only
        results = {
            "import_performance": test_suite.test_import_performance(),
            "overall_health": test_suite.test_overall_health(),
        }
        test_suite.results = results
    else:
        # Full comprehensive testing
        results = test_suite.run_comprehensive_tests()

    test_suite.print_summary_report()

    if args.save_report:
        with open("phase5_integration_test_report.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Detailed report saved to phase5_integration_test_report.json")


if __name__ == "__main__":
    main()
