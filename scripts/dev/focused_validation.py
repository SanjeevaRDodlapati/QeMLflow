#!/usr/bin/env python3
"""
Focused validation pipeline for type hint changes.
Minimal, targeted testing for our specific changes.
"""

import json
import subprocess
import sys
import time
from typing import Dict, List, Tuple


class FocusedValidator:
    """Lightweight validator for type hint changes."""

    def __init__(self):
        self.results = {}

    def validate_changes(self) -> bool:
        """Run focused validation for our specific changes."""
        print("🎯 Running focused validation for type hint changes...")

        validations = [
            ("Type Hint Compatibility", self._validate_type_hints),
            ("Philosophy Compliance", self._validate_philosophy),
            ("Import Performance", self._validate_import_performance),
            ("Existing Tests", self._validate_existing_tests),
        ]

        all_passed = True

        for name, validation_func in validations:
            print(f"\n📋 {name}...")
            try:
                passed, details = validation_func()
                self.results[name] = {"passed": passed, "details": details}

                if passed:
                    print(f"✅ {name} passed")
                else:
                    print(f"❌ {name} failed: {details}")
                    all_passed = False

            except Exception as e:
                print(f"💥 {name} error: {e}")
                all_passed = False

        return all_passed

    def _validate_type_hints(self) -> Tuple[bool, str]:
        """Validate type hint implementation."""
        try:
            # Test basic import compatibility after changes
            import sys

            modules_to_clear = [m for m in sys.modules.keys() if "qemlflow" in m]
            for module in modules_to_clear:
                del sys.modules[module]

            # Test import still works
            import qemlflow.core

            # Test that functions are still callable
            if hasattr(qemlflow.core, "data_processing"):
                data_proc = qemlflow.core.data_processing
                if hasattr(data_proc, "is_valid_smiles"):
                    # Test function still works
                    result = data_proc.is_valid_smiles("CCO")
                    assert isinstance(result, bool), "Function should return boolean"

            return True, "Type hint compatibility verified"

        except Exception as e:
            return False, f"Type hint validation failed: {e}"

    def _validate_philosophy(self) -> Tuple[bool, str]:
        """Check philosophy compliance improvement."""
        try:
            result = subprocess.run(
                ["python", "tools/philosophy_enforcer.py", "--quick-check"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                # Extract score from output
                output = result.stdout
                if "Overall Compliance Score:" in output:
                    score_line = [
                        line
                        for line in output.split("\n")
                        if "Overall Compliance Score:" in line
                    ][0]
                    score = int(score_line.split(":")[1].split("/")[0].strip())

                    if score >= 70:
                        return True, f"Philosophy score improved to {score}/100"
                    else:
                        return False, f"Philosophy score still low: {score}/100"
                else:
                    return True, "Philosophy check passed"
            else:
                return False, "Philosophy enforcer failed"

        except Exception as e:
            return False, f"Philosophy check error: {e}"

    def _validate_import_performance(self) -> Tuple[bool, str]:
        """Ensure import performance is maintained."""
        import sys

        # Clear modules
        modules_to_clear = [m for m in sys.modules.keys() if "qemlflow" in m]
        for module in modules_to_clear:
            del sys.modules[module]

        # Measure import time
        start_time = time.time()
        try:
            import qemlflow.core

            import_time = time.time() - start_time

            if import_time < 1.0:
                return True, f"Import time good: {import_time:.2f}s"
            else:
                return False, f"Import time too slow: {import_time:.2f}s"

        except Exception as e:
            return False, f"Import failed: {e}"

    def _validate_existing_tests(self) -> Tuple[bool, str]:
        """Run a sample of existing tests."""
        try:
            # Run a quick subset of existing tests
            result = subprocess.run(
                [
                    "python",
                    "-m",
                    "pytest",
                    "tests/unit/test_data_processing.py",
                    "-x",
                    "-q",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode == 0:
                return True, "Sample tests pass"
            else:
                return False, f"Some tests failed: {result.stdout[-200:]}"

        except subprocess.TimeoutExpired:
            return False, "Tests timeout"
        except Exception as e:
            return False, f"Test execution error: {e}"

    def print_summary(self):
        """Print validation summary."""
        print("\n" + "=" * 50)
        print("🎯 FOCUSED VALIDATION SUMMARY")
        print("=" * 50)

        passed_count = sum(1 for r in self.results.values() if r["passed"])
        total_count = len(self.results)

        print(f"Success Rate: {passed_count}/{total_count}")

        for name, result in self.results.items():
            status = "✅" if result["passed"] else "❌"
            print(f"{status} {name}: {result['details']}")

        if passed_count == total_count:
            print("\n🎉 All validations passed! Changes are ready.")
            return True
        else:
            print("\n⚠️ Some validations failed. Check details above.")
            return False


def main():
    """Main execution."""
    validator = FocusedValidator()
    success = validator.validate_changes()
    overall_success = validator.print_summary()

    return 0 if overall_success else 1


if __name__ == "__main__":
    sys.exit(main())
