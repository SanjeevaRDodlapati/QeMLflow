"""
ChemML Bootcamp Integration Test Suite
=====================================

Comprehensive testing framework to validate all optimization components work together:
- Modular notebook architecture (Days 5-7)
- Simplified assessment framework
- Streamlined documentation
- Multi-pace learning tracks

Usage:
    python integration_test_suite.py --test-all
    python integration_test_suite.py --test-notebooks
    python integration_test_suite.py --test-assessment
    python integration_test_suite.py --test-docs
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


class BootcampIntegrationTester:
    """Comprehensive integration testing for bootcamp optimizations."""

    def __init__(self, bootcamp_dir: str):
        self.bootcamp_dir = Path(bootcamp_dir)
        self.test_results = {
            "notebooks": {},
            "assessment": {},
            "documentation": {},
            "learning_tracks": {},
            "overall": {"passed": 0, "failed": 0, "warnings": 0},
        }

    def run_all_tests(self) -> Dict[str, Any]:
        """Run comprehensive integration test suite."""
        print("🧪 Starting ChemML Bootcamp Integration Test Suite")
        print("=" * 60)

        # Test 1: Modular Notebook Architecture
        print("\n📓 Testing Modular Notebook Architecture...")
        self.test_notebook_modularization()

        # Test 2: Assessment Framework Integration
        print("\n📊 Testing Simplified Assessment Framework...")
        self.test_assessment_integration()

        # Test 3: Documentation Streamlining
        print("\n📚 Testing Streamlined Documentation...")
        self.test_documentation_integration()

        # Test 4: Multi-Pace Learning Tracks
        print("\n🎯 Testing Multi-Pace Learning Tracks...")
        self.test_learning_tracks()

        # Test 5: Cross-Component Integration
        print("\n🔗 Testing Cross-Component Integration...")
        self.test_cross_component_integration()

        # Generate final report
        self.generate_integration_report()

        return self.test_results

    def test_notebook_modularization(self):
        """Test modular notebook architecture for Days 5-7."""
        print("  ✓ Checking modular notebook structure...")

        # Expected modular notebooks
        expected_modules = {
            "day_05": [
                "day_05_module_1_foundations.ipynb",
                "day_05_module_2_advanced.ipynb",
                "day_05_module_3_production.ipynb",
            ],
            "day_06": [
                "day_06_module_1_quantum_foundations.ipynb",
                "day_06_module_2_vqe_algorithms.ipynb",
                "day_06_module_3_quantum_production.ipynb",
            ],
            "day_07": [
                "day_07_module_1_integration.ipynb",
                "day_07_module_2_multimodal_workflows.ipynb",
                "day_07_module_3_deployment.ipynb",
            ],
        }

        for day, modules in expected_modules.items():
            day_results = {
                "modules_found": 0,
                "modules_expected": len(modules),
                "issues": [],
            }

            for module in modules:
                module_path = self.bootcamp_dir / module
                if module_path.exists():
                    day_results["modules_found"] += 1
                    # Test notebook structure
                    self._test_notebook_structure(module_path, day_results)
                else:
                    day_results["issues"].append(f"Missing module: {module}")

            self.test_results["notebooks"][day] = day_results

            # Report results
            found = day_results["modules_found"]
            expected = day_results["modules_expected"]
            if found == expected:
                print(f"    ✅ {day.upper()}: {found}/{expected} modules found")
                self.test_results["overall"]["passed"] += 1
            else:
                print(f"    ❌ {day.upper()}: {found}/{expected} modules found")
                self.test_results["overall"]["failed"] += 1
                for issue in day_results["issues"]:
                    print(f"      - {issue}")

    def _test_notebook_structure(self, notebook_path: Path, results: Dict):
        """Test individual notebook structure and content."""
        try:
            with open(notebook_path, "r", encoding="utf-8") as f:
                nb = nbformat.read(f, as_version=4)

            # Check for navigation cells
            has_navigation = any(
                "Previous Module" in str(cell.get("source", ""))
                or "Next Module" in str(cell.get("source", ""))
                for cell in nb.cells
            )

            # Check for progress tracking
            has_progress = any(
                "progress" in str(cell.get("source", "")).lower()
                or "checkpoint" in str(cell.get("source", "")).lower()
                for cell in nb.cells
            )

            # Check cell count (should be reasonable - not too dense)
            cell_count = len(nb.cells)
            if cell_count > 100:  # Arbitrary threshold for "too dense"
                results["issues"].append(
                    f"{notebook_path.name}: {cell_count} cells (may be too dense)"
                )

            if not has_navigation:
                results["issues"].append(
                    f"{notebook_path.name}: Missing navigation elements"
                )

            if not has_progress:
                results["issues"].append(
                    f"{notebook_path.name}: Missing progress tracking"
                )

        except Exception as e:
            results["issues"].append(
                f"{notebook_path.name}: Error reading notebook - {str(e)}"
            )

    def test_assessment_integration(self):
        """Test simplified assessment framework."""
        print("  ✓ Checking simplified assessment framework...")

        assessment_dir = self.bootcamp_dir / "assessment"
        expected_files = [
            "simple_progress_tracker.py",
            "daily_checkpoints.md",
            "completion_badges.py",
        ]

        assessment_results = {
            "files_found": 0,
            "files_expected": len(expected_files),
            "issues": [],
        }

        for file in expected_files:
            file_path = assessment_dir / file
            if file_path.exists():
                assessment_results["files_found"] += 1

                # Test file content for key features
                if file.endswith(".py"):
                    self._test_python_file(file_path, assessment_results)
                elif file.endswith(".md"):
                    self._test_markdown_file(file_path, assessment_results)
            else:
                assessment_results["issues"].append(f"Missing file: {file}")

        self.test_results["assessment"] = assessment_results

        # Report results
        found = assessment_results["files_found"]
        expected = assessment_results["files_expected"]
        if found == expected:
            print(f"    ✅ Assessment: {found}/{expected} files found")
            self.test_results["overall"]["passed"] += 1
        else:
            print(f"    ❌ Assessment: {found}/{expected} files found")
            self.test_results["overall"]["failed"] += 1

    def _test_python_file(self, file_path: Path, results: Dict):
        """Test Python file for basic syntax and key functions."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Check file size (should be reasonable - not over-engineered)
            line_count = len(content.splitlines())
            if line_count > 300:  # Should be under 300 lines for simplified framework
                results["issues"].append(
                    f"{file_path.name}: {line_count} lines (may be over-engineered)"
                )

            # Basic syntax check
            compile(content, file_path.name, "exec")

        except SyntaxError as e:
            results["issues"].append(f"{file_path.name}: Syntax error - {str(e)}")
        except Exception as e:
            results["issues"].append(f"{file_path.name}: Error reading file - {str(e)}")

    def _test_markdown_file(self, file_path: Path, results: Dict):
        """Test Markdown file for basic structure."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Check for basic markdown structure
            if not content.strip():
                results["issues"].append(f"{file_path.name}: Empty file")

            # Check for headers
            if not any(line.startswith("#") for line in content.splitlines()):
                results["issues"].append(f"{file_path.name}: No headers found")

        except Exception as e:
            results["issues"].append(f"{file_path.name}: Error reading file - {str(e)}")

    def test_documentation_integration(self):
        """Test streamlined documentation architecture."""
        print("  ✓ Checking streamlined documentation...")

        docs_dir = self.bootcamp_dir.parent / "docs"
        expected_docs = ["GET_STARTED.md", "LEARNING_PATHS.md", "REFERENCE.md"]

        doc_results = {
            "docs_found": 0,
            "docs_expected": len(expected_docs),
            "issues": [],
        }

        for doc in expected_docs:
            doc_path = docs_dir / doc
            if doc_path.exists():
                doc_results["docs_found"] += 1
                self._test_documentation_quality(doc_path, doc_results)
            else:
                doc_results["issues"].append(f"Missing documentation: {doc}")

        # Check for redundant files that should have been removed
        redundant_patterns = [
            "documentation_assessment",
            "documentation_integration",
            "documentation_organization",
            "validation_testing",
        ]

        if docs_dir.exists():
            for item in docs_dir.iterdir():
                if item.is_file() and any(
                    pattern in item.name for pattern in redundant_patterns
                ):
                    doc_results["issues"].append(
                        f"Redundant file still exists: {item.name}"
                    )

        self.test_results["documentation"] = doc_results

        # Report results
        found = doc_results["docs_found"]
        expected = doc_results["docs_expected"]
        if (
            found == expected
            and len([i for i in doc_results["issues"] if "Redundant" in i]) == 0
        ):
            print(f"    ✅ Documentation: {found}/{expected} core docs found")
            self.test_results["overall"]["passed"] += 1
        else:
            print(f"    ❌ Documentation: {found}/{expected} core docs found")
            self.test_results["overall"]["failed"] += 1

    def _test_documentation_quality(self, doc_path: Path, results: Dict):
        """Test documentation quality and structure."""
        try:
            with open(doc_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Check minimum length (should have substantial content)
            if len(content) < 500:
                results["issues"].append(
                    f"{doc_path.name}: Too short (may be incomplete)"
                )

            # Check for bootcamp mentions (should be integrated)
            if "bootcamp" not in content.lower():
                results["issues"].append(
                    f"{doc_path.name}: No bootcamp integration found"
                )

            # Check for clear structure
            headers = [line for line in content.splitlines() if line.startswith("#")]
            if len(headers) < 3:
                results["issues"].append(
                    f"{doc_path.name}: Limited structure (few headers)"
                )

        except Exception as e:
            results["issues"].append(f"{doc_path.name}: Error reading file - {str(e)}")

    def test_learning_tracks(self):
        """Test multi-pace learning track implementation."""
        print("  ✓ Checking multi-pace learning tracks...")

        # This would test the learning paths implementation
        # For now, check if LEARNING_PATHS.md contains track information
        docs_dir = self.bootcamp_dir.parent / "docs"
        learning_paths_file = docs_dir / "LEARNING_PATHS.md"

        track_results = {"tracks_implemented": False, "issues": []}

        if learning_paths_file.exists():
            try:
                with open(learning_paths_file, "r", encoding="utf-8") as f:
                    content = f.read()

                # Check for track mentions
                track_keywords = ["fast track", "complete track", "flexible track"]
                tracks_found = sum(
                    1
                    for keyword in track_keywords
                    if keyword.lower() in content.lower()
                )

                if tracks_found >= 2:
                    track_results["tracks_implemented"] = True
                    print(f"    ✅ Learning Tracks: {tracks_found}/3 tracks found")
                    self.test_results["overall"]["passed"] += 1
                else:
                    track_results["issues"].append(
                        f"Only {tracks_found}/3 learning tracks found"
                    )
                    print(f"    ❌ Learning Tracks: {tracks_found}/3 tracks found")
                    self.test_results["overall"]["failed"] += 1

            except Exception as e:
                track_results["issues"].append(
                    f"Error reading learning paths: {str(e)}"
                )
                print(f"    ❌ Learning Tracks: Error reading file")
                self.test_results["overall"]["failed"] += 1
        else:
            track_results["issues"].append("LEARNING_PATHS.md not found")
            print(f"    ❌ Learning Tracks: LEARNING_PATHS.md not found")
            self.test_results["overall"]["failed"] += 1

        self.test_results["learning_tracks"] = track_results

    def test_cross_component_integration(self):
        """Test how all components work together."""
        print("  ✓ Testing cross-component integration...")

        integration_results = {"issues": [], "integration_score": 0}

        # Test 1: Navigation between components
        # Check if documentation references notebooks
        docs_dir = self.bootcamp_dir.parent / "docs"
        get_started = docs_dir / "GET_STARTED.md"

        if get_started.exists():
            with open(get_started, "r", encoding="utf-8") as f:
                content = f.read()

            if "notebooks" in content.lower():
                integration_results["integration_score"] += 1
            else:
                integration_results["issues"].append(
                    "GET_STARTED.md doesn't reference notebooks"
                )

        # Test 2: Assessment integration with notebooks
        # Check if modular notebooks reference assessment
        day_05_mod1 = self.bootcamp_dir / "day_05_module_1_foundations.ipynb"
        if day_05_mod1.exists():
            try:
                with open(day_05_mod1, "r", encoding="utf-8") as f:
                    nb_content = f.read()

                if (
                    "progress" in nb_content.lower()
                    or "checkpoint" in nb_content.lower()
                ):
                    integration_results["integration_score"] += 1
                else:
                    integration_results["issues"].append(
                        "Notebooks missing assessment integration"
                    )
            except Exception:
                integration_results["issues"].append("Error reading modular notebook")

        # Test 3: Documentation cross-references
        learning_paths = docs_dir / "LEARNING_PATHS.md"
        if learning_paths.exists():
            with open(learning_paths, "r", encoding="utf-8") as f:
                content = f.read()

            if "get_started" in content.lower() or "reference" in content.lower():
                integration_results["integration_score"] += 1
            else:
                integration_results["issues"].append(
                    "LEARNING_PATHS.md missing cross-references"
                )

        # Evaluate integration score
        max_score = 3
        score = integration_results["integration_score"]

        if score >= max_score * 0.8:  # 80% or better
            print(f"    ✅ Integration: {score}/{max_score} components integrated")
            self.test_results["overall"]["passed"] += 1
        else:
            print(f"    ⚠️  Integration: {score}/{max_score} components integrated")
            self.test_results["overall"]["warnings"] += 1

        self.test_results["integration"] = integration_results

    def generate_integration_report(self):
        """Generate comprehensive integration test report."""
        print("\n" + "=" * 60)
        print("📋 INTEGRATION TEST REPORT")
        print("=" * 60)

        total_tests = (
            self.test_results["overall"]["passed"]
            + self.test_results["overall"]["failed"]
            + self.test_results["overall"]["warnings"]
        )

        passed = self.test_results["overall"]["passed"]
        failed = self.test_results["overall"]["failed"]
        warnings = self.test_results["overall"]["warnings"]

        print(f"\n📊 Test Summary:")
        print(f"  ✅ Passed: {passed}")
        print(f"  ❌ Failed: {failed}")
        print(f"  ⚠️  Warnings: {warnings}")
        print(
            f"  📈 Success Rate: {(passed / total_tests * 100):.1f}%"
            if total_tests > 0
            else "No tests run"
        )

        # Detailed results
        print(f"\n📓 Notebook Modularization:")
        for day, results in self.test_results["notebooks"].items():
            found = results["modules_found"]
            expected = results["modules_expected"]
            status = "✅" if found == expected else "❌"
            print(f"  {status} {day.upper()}: {found}/{expected} modules")
            for issue in results.get("issues", []):
                print(f"    - {issue}")

        print(f"\n📊 Assessment Framework:")
        if "assessment" in self.test_results:
            assessment = self.test_results["assessment"]
            found = assessment["files_found"]
            expected = assessment["files_expected"]
            status = "✅" if found == expected else "❌"
            print(f"  {status} Files: {found}/{expected}")
            for issue in assessment.get("issues", []):
                print(f"    - {issue}")

        print(f"\n📚 Documentation:")
        if "documentation" in self.test_results:
            docs = self.test_results["documentation"]
            found = docs["docs_found"]
            expected = docs["docs_expected"]
            status = "✅" if found == expected else "❌"
            print(f"  {status} Core Docs: {found}/{expected}")
            for issue in docs.get("issues", []):
                print(f"    - {issue}")

        print(f"\n🎯 Learning Tracks:")
        if "learning_tracks" in self.test_results:
            tracks = self.test_results["learning_tracks"]
            status = "✅" if tracks["tracks_implemented"] else "❌"
            print(f"  {status} Multi-pace tracks implemented")
            for issue in tracks.get("issues", []):
                print(f"    - {issue}")

        print(f"\n🔗 Integration:")
        if "integration" in self.test_results:
            integration = self.test_results["integration"]
            score = integration["integration_score"]
            print(f"  📈 Integration Score: {score}/3")
            for issue in integration.get("issues", []):
                print(f"    - {issue}")

        # Save results to file
        report_file = self.bootcamp_dir / "integration_test_report.json"
        with open(report_file, "w") as f:
            json.dump(self.test_results, f, indent=2)

        print(f"\n💾 Full report saved to: {report_file}")

        # Overall assessment
        success_rate = (passed / total_tests * 100) if total_tests > 0 else 0
        print(f"\n🎯 Overall Integration Status:")
        if success_rate >= 80:
            print("  🟢 EXCELLENT - Ready for user testing")
        elif success_rate >= 60:
            print("  🟡 GOOD - Minor issues to address")
        elif success_rate >= 40:
            print("  🟠 FAIR - Several issues need attention")
        else:
            print("  🔴 POOR - Major integration problems")


def main():
    """Main entry point for integration testing."""
    parser = argparse.ArgumentParser(
        description="ChemML Bootcamp Integration Test Suite"
    )
    parser.add_argument(
        "--test-all", action="store_true", help="Run all integration tests"
    )
    parser.add_argument(
        "--test-notebooks",
        action="store_true",
        help="Test notebook modularization only",
    )
    parser.add_argument(
        "--test-assessment", action="store_true", help="Test assessment framework only"
    )
    parser.add_argument(
        "--test-docs", action="store_true", help="Test documentation only"
    )
    parser.add_argument(
        "--bootcamp-dir", default=".", help="Path to bootcamp directory"
    )

    args = parser.parse_args()

    # Default to running all tests if no specific test is requested
    if not any(
        [args.test_all, args.test_notebooks, args.test_assessment, args.test_docs]
    ):
        args.test_all = True

    tester = BootcampIntegrationTester(args.bootcamp_dir)

    if args.test_all:
        results = tester.run_all_tests()
    else:
        # Run specific tests
        if args.test_notebooks:
            tester.test_notebook_modularization()
        if args.test_assessment:
            tester.test_assessment_integration()
        if args.test_docs:
            tester.test_documentation_integration()

        tester.generate_integration_report()

    # Exit with error code if tests failed
    failed_tests = tester.test_results["overall"]["failed"]
    sys.exit(1 if failed_tests > 0 else 0)


if __name__ == "__main__":
    main()
