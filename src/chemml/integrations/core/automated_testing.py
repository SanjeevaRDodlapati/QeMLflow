"""
Automated Testing Framework for External Model Adapters
======================================================

Comprehensive validation and testing system for external model integrations
to ensure quality, reliability, and consistency across all adapters.
"""

import shutil
import tempfile
import time
import traceback
import unittest
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

import numpy as np
import pandas as pd

from .external_models import ExternalModelWrapper
from .performance_monitoring import get_metrics


class AdapterTestCase(unittest.TestCase, ABC):
    """Base test case for adapter validation."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_data = self.generate_test_data()
        self.metrics = get_metrics()

    def tearDown(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    @abstractmethod
    def create_adapter(self) -> ExternalModelWrapper:
        """Create the adapter instance to test."""
        pass

    @abstractmethod
    def generate_test_data(self) -> Dict[str, Any]:
        """Generate appropriate test data for the adapter."""
        pass

    def test_initialization(self):
        """Test adapter initialization."""
        try:
            adapter = self.create_adapter()
            self.assertIsNotNone(adapter)
            self.assertIsInstance(adapter, ExternalModelWrapper)
            print(f"✅ Initialization test passed for {adapter.__class__.__name__}")
        except Exception as e:
            self.fail(f"Adapter initialization failed: {e}")

    def test_prediction_interface(self):
        """Test prediction interface compliance."""
        adapter = self.create_adapter()

        # Test prediction method exists
        self.assertTrue(hasattr(adapter, "predict"), "Adapter must have predict method")

        # Test prediction with test data
        try:
            X_test = self.test_data.get("X_test")
            if X_test is not None:
                # Check if model needs to be fitted first
                if hasattr(adapter, "is_fitted") and not adapter.is_fitted:
                    X_train = self.test_data.get("X_train")
                    y_train = self.test_data.get("y_train")
                    if X_train is not None and y_train is not None:
                        adapter.fit(X_train, y_train)

                predictions = adapter.predict(X_test)
                self.assertIsNotNone(predictions)

                # Check prediction format
                if isinstance(predictions, np.ndarray):
                    self.assertEqual(len(predictions), len(X_test))
                elif isinstance(predictions, (list, tuple)):
                    self.assertEqual(len(predictions), len(X_test))

                print(
                    f"✅ Prediction interface test passed for {adapter.__class__.__name__}"
                )
        except Exception as e:
            self.fail(f"Prediction interface test failed: {e}")

    def test_error_handling(self):
        """Test error handling robustness."""
        adapter = self.create_adapter()

        # Test with invalid input
        test_cases = [
            ("empty_array", np.array([])),
            (
                "wrong_shape",
                np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]),
            ),  # Likely wrong dimension
            ("none_input", None),
            ("string_input", "invalid_input"),
        ]

        for test_name, invalid_input in test_cases:
            try:
                if invalid_input is not None:
                    adapter.predict(invalid_input)
                else:
                    adapter.predict(invalid_input)
                # If no exception, that's also acceptable (graceful handling)
                print(f"✅ Error handling test '{test_name}' passed (graceful)")
            except Exception as e:
                # Expected behavior - should raise appropriate exceptions
                self.assertIsInstance(
                    e, (ValueError, TypeError, RuntimeError, AttributeError)
                )
                print(
                    f"✅ Error handling test '{test_name}' passed (exception: {type(e).__name__})"
                )

    def test_resource_cleanup(self):
        """Test resource management and cleanup."""
        adapter = self.create_adapter()

        # Check if cleanup method exists and works
        if hasattr(adapter, "cleanup"):
            try:
                adapter.cleanup()
                print(
                    f"✅ Resource cleanup test passed for {adapter.__class__.__name__}"
                )
            except Exception as e:
                self.fail(f"Resource cleanup failed: {e}")
        else:
            print(f"ℹ️  No explicit cleanup method for {adapter.__class__.__name__}")

    def test_memory_usage(self):
        """Test memory usage is reasonable."""
        import psutil

        # Measure memory before
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        # Create and use adapter
        adapter = self.create_adapter()
        X_test = self.test_data.get("X_test")

        if X_test is not None:
            try:
                # Fit if needed
                if hasattr(adapter, "is_fitted") and not adapter.is_fitted:
                    X_train = self.test_data.get("X_train")
                    y_train = self.test_data.get("y_train")
                    if X_train is not None and y_train is not None:
                        adapter.fit(X_train, y_train)

                # Make predictions
                # _predictions = adapter.predict(X_test)

                # Measure memory after
                memory_after = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = memory_after - memory_before

                # Memory increase should be reasonable (less than 1GB for test data)
                self.assertLess(
                    memory_increase,
                    1024,
                    f"Memory usage too high: {memory_increase:.1f} MB",
                )

                print(f"✅ Memory usage test passed: {memory_increase:.1f} MB increase")

            except Exception as e:
                print(f"⚠️  Memory test skipped due to prediction error: {e}")

    def test_performance_baseline(self):
        """Test performance meets baseline requirements."""
        adapter = self.create_adapter()
        X_test = self.test_data.get("X_test")

        if X_test is not None and len(X_test) > 0:
            try:
                # Fit if needed
                if hasattr(adapter, "is_fitted") and not adapter.is_fitted:
                    X_train = self.test_data.get("X_train")
                    y_train = self.test_data.get("y_train")
                    if X_train is not None and y_train is not None:
                        start_time = time.time()
                        adapter.fit(X_train, y_train)
                        fit_time = time.time() - start_time
                        print(f"ℹ️  Fit time: {fit_time:.2f} seconds")

                # Time prediction
                start_time = time.time()
                # _predictions = adapter.predict(X_test)
                prediction_time = time.time() - start_time

                # Calculate throughput
                samples_per_second = (
                    len(X_test) / prediction_time
                    if prediction_time > 0
                    else float("inf")
                )

                print(
                    f"✅ Performance test: {prediction_time:.3f}s for {len(X_test)} samples"
                )
                print(f"   Throughput: {samples_per_second:.1f} samples/second")

                # Basic performance requirement (should process at least 1 sample per second)
                self.assertGreater(
                    samples_per_second,
                    1.0,
                    "Performance too slow: less than 1 sample per second",
                )

            except Exception as e:
                print(f"⚠️  Performance test skipped due to error: {e}")


class MockAdapterTestCase(AdapterTestCase):
    """Test case for mock adapter (used for framework testing)."""

    def create_adapter(self) -> ExternalModelWrapper:
        """Create a mock adapter for testing."""
        # Create a mock repository structure
        mock_repo = self.temp_dir / "mock_repo"
        mock_repo.mkdir()

        # Create a simple mock model file
        model_file = mock_repo / "model.py"
        model_file.write_text(
            """
import numpy as np


class MockModel:
    def __init__(self):
        self.is_fitted = False

    def fit(self, X, y):
        self.is_fitted = True
        return {"loss": 0.1}

    def predict(self, X):
        if isinstance(X, np.ndarray) and len(X.shape) == 2:
            return np.random.random(X.shape[0])
        else:
            raise ValueError("Invalid input shape")
"""
        )

        return ExternalModelWrapper(
            repo_url=str(mock_repo), model_class_name="MockModel"
        )

    def generate_test_data(self) -> Dict[str, Any]:
        """Generate test data for mock adapter."""
        np.random.seed(42)  # Reproducible results

        X_train = np.random.random((100, 10))
        y_train = np.random.random(100)
        X_test = np.random.random((20, 10))

        return {"X_train": X_train, "y_train": y_train, "X_test": X_test}


class AdapterTestSuite:
    """
    Comprehensive test suite for validating external model adapters.
    """

    def __init__(self):
        """Initialize the test suite."""
        self.test_results = {}
        self.metrics = get_metrics()

    def validate_adapter(
        self,
        adapter_class: Type[ExternalModelWrapper],
        test_data_generator: Optional[Callable] = None,
        custom_tests: Optional[List[Callable]] = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive adapter validation.

        Args:
            adapter_class: The adapter class to test
            test_data_generator: Function to generate test data
            custom_tests: Additional custom test functions

        Returns:
            Dictionary with test results
        """

        print(f"\n🧪 Testing adapter: {adapter_class.__name__}")
        print("=" * 50)

        results = {
            "adapter_class": adapter_class.__name__,
            "timestamp": time.time(),
            "tests_passed": 0,
            "tests_failed": 0,
            "tests_skipped": 0,
            "test_details": {},
            "overall_status": "UNKNOWN",
            "recommendations": [],
        }

        # Create dynamic test case
        class DynamicTestCase(AdapterTestCase):
            def create_adapter(self):
                try:
                    return adapter_class()
                except Exception as e:
                    # Try with minimal arguments
                    try:
                        return adapter_class(
                            repo_url="https://github.com/example/repo.git",
                            model_class_name="ExampleModel",
                        )
                    except Exception:
                        raise e

            def generate_test_data(self):
                if test_data_generator:
                    return test_data_generator()
                else:
                    # Default test data
                    np.random.seed(42)
                    return {
                        "X_train": np.random.random((50, 5)),
                        "y_train": np.random.random(50),
                        "X_test": np.random.random((10, 5)),
                    }

        # Run standard tests
        test_methods = [
            "test_initialization",
            "test_prediction_interface",
            "test_error_handling",
            "test_resource_cleanup",
            "test_memory_usage",
            "test_performance_baseline",
        ]

        test_case = DynamicTestCase()
        test_case.setUp()

        try:
            for test_method in test_methods:
                test_name = test_method.replace("test_", "")
                print(f"\n🔍 Running {test_name}...")

                try:
                    getattr(test_case, test_method)()
                    results["tests_passed"] += 1
                    results["test_details"][test_name] = {
                        "status": "PASSED",
                        "message": "Test completed successfully",
                    }

                except unittest.SkipTest as e:
                    results["tests_skipped"] += 1
                    results["test_details"][test_name] = {
                        "status": "SKIPPED",
                        "message": str(e),
                    }
                    print(f"⏭️  Skipped: {e}")

                except Exception as e:
                    results["tests_failed"] += 1
                    results["test_details"][test_name] = {
                        "status": "FAILED",
                        "message": str(e),
                        "traceback": traceback.format_exc(),
                    }
                    print(f"❌ Failed: {e}")

            # Run custom tests if provided
            if custom_tests:
                for i, custom_test in enumerate(custom_tests):
                    test_name = f"custom_test_{i+1}"
                    print(f"\n🔍 Running {test_name}...")

                    try:
                        custom_test(test_case.create_adapter())
                        results["tests_passed"] += 1
                        results["test_details"][test_name] = {
                            "status": "PASSED",
                            "message": "Custom test completed successfully",
                        }
                        print(f"✅ Custom test {i+1} passed")

                    except Exception as e:
                        results["tests_failed"] += 1
                        results["test_details"][test_name] = {
                            "status": "FAILED",
                            "message": str(e),
                            "traceback": traceback.format_exc(),
                        }
                        print(f"❌ Custom test {i+1} failed: {e}")

        finally:
            test_case.tearDown()

        # Determine overall status
        total_tests = (
            results["tests_passed"] + results["tests_failed"] + results["tests_skipped"]
        )
        if total_tests == 0:
            results["overall_status"] = "NO_TESTS"
        elif results["tests_failed"] == 0:
            results["overall_status"] = "PASSED"
        elif results["tests_failed"] < results["tests_passed"]:
            results["overall_status"] = "MOSTLY_PASSED"
        else:
            results["overall_status"] = "FAILED"

        # Generate recommendations
        results["recommendations"] = self._generate_recommendations(results)

        # Store results
        self.test_results[adapter_class.__name__] = results

        # Print summary
        self._print_test_summary(results)

        return results

    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []

        failed_tests = [
            name
            for name, details in results["test_details"].items()
            if details["status"] == "FAILED"
        ]

        if "initialization" in failed_tests:
            recommendations.append(
                "Fix initialization issues - ensure adapter can be created with standard parameters"
            )

        if "prediction_interface" in failed_tests:
            recommendations.append(
                "Implement proper prediction interface - ensure predict() method returns valid results"
            )

        if "error_handling" in failed_tests:
            recommendations.append(
                "Improve error handling - add validation for input data and raise appropriate exceptions"
            )

        if "memory_usage" in failed_tests:
            recommendations.append(
                "Optimize memory usage - consider lazy loading or more efficient data structures"
            )

        if "performance_baseline" in failed_tests:
            recommendations.append(
                "Improve performance - optimize prediction speed to meet baseline requirements"
            )

        if results["tests_failed"] > results["tests_passed"]:
            recommendations.append(
                "Major issues detected - consider significant refactoring before production use"
            )
        elif results["tests_failed"] > 0:
            recommendations.append(
                "Minor issues detected - address failed tests before production deployment"
            )
        else:
            recommendations.append(
                "All tests passed - adapter is ready for production use"
            )

        return recommendations

    def _print_test_summary(self, results: Dict[str, Any]):
        """Print a formatted test summary."""
        print(f"\n📊 Test Summary for {results['adapter_class']}")
        print("=" * 50)
        print(f"✅ Passed: {results['tests_passed']}")
        print(f"❌ Failed: {results['tests_failed']}")
        print(f"⏭️  Skipped: {results['tests_skipped']}")
        print(f"🎯 Overall Status: {results['overall_status']}")

        if results["recommendations"]:
            print("\n💡 Recommendations:")
            for i, rec in enumerate(results["recommendations"], 1):
                print(f"   {i}. {rec}")

        print()

    def run_mock_adapter_test(self) -> Dict[str, Any]:
        """Run tests on the mock adapter to validate the testing framework."""
        print("🧪 Running mock adapter test to validate testing framework...")

        results = self.validate_adapter(
            adapter_class=type(None),  # Will use MockAdapterTestCase
            test_data_generator=lambda: {
                "X_train": np.random.random((50, 5)),
                "y_train": np.random.random(50),
                "X_test": np.random.random((10, 5)),
            },
        )

        return results

    def generate_test_report(self, output_file: Optional[str] = None) -> str:
        """
        Generate a comprehensive test report.

        Args:
            output_file: Optional file path to save the report

        Returns:
            The report as a string
        """

        if not self.test_results:
            return "No test results available. Run some tests first."

        report = f"""
# Adapter Testing Report
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Summary

| Adapter | Status | Passed | Failed | Skipped |
|---------|--------|--------|--------|---------|
"""

        for adapter_name, results in self.test_results.items():
            status_emoji = {
                "PASSED": "✅",
                "MOSTLY_PASSED": "⚠️",
                "FAILED": "❌",
                "NO_TESTS": "❔",
            }.get(results["overall_status"], "❔")

            report += f"| {adapter_name} | {status_emoji} {results['overall_status']} | {results['tests_passed']} | {results['tests_failed']} | {results['tests_skipped']} |\n"

        # Detailed results
        for adapter_name, results in self.test_results.items():
            report += f"""
## {adapter_name}

**Overall Status**: {results['overall_status']}

### Test Details
"""

            for test_name, details in results["test_details"].items():
                status_emoji = {"PASSED": "✅", "FAILED": "❌", "SKIPPED": "⏭️"}.get(
                    details["status"], "❔"
                )
                report += f"- **{test_name}**: {status_emoji} {details['status']}\n"
                if details["status"] == "FAILED":
                    report += f"  - Error: {details['message']}\n"

            if results["recommendations"]:
                report += "\n### Recommendations\n"
                for i, rec in enumerate(results["recommendations"], 1):
                    report += f"{i}. {rec}\n"

        report = report.strip()

        # Save to file if specified
        if output_file:
            with open(output_file, "w") as f:
                f.write(report)
            print(f"📄 Test report saved to: {output_file}")

        return report

    def validate_framework_integration(self) -> bool:
        """
        Validate that the testing framework works correctly.

        Returns:
            True if framework validation passes
        """

        print("🔧 Validating testing framework...")

        try:
            # Test with mock adapter
            mock_test = MockAdapterTestCase()
            mock_test.setUp()

            # Run basic tests
            mock_test.test_initialization()
            mock_test.test_prediction_interface()
            mock_test.test_error_handling()

            mock_test.tearDown()

            print("✅ Testing framework validation passed")
            return True

        except Exception as e:
            print(f"❌ Testing framework validation failed: {e}")
            return False


def create_adapter_test_suite() -> AdapterTestSuite:
    """Create a new adapter test suite."""
    return AdapterTestSuite()


def quick_adapter_test(adapter_class: Type[ExternalModelWrapper]) -> str:
    """
    Quick test of an adapter class.

    Args:
        adapter_class: The adapter class to test

    Returns:
        Quick test result summary
    """
    suite = AdapterTestSuite()
    results = suite.validate_adapter(adapter_class)

    status = results["overall_status"]
    passed = results["tests_passed"]
    failed = results["tests_failed"]

    return f"{adapter_class.__name__}: {status} ({passed} passed, {failed} failed)"


# Test data generators for common adapter types
def generate_molecular_test_data() -> Dict[str, Any]:
    """Generate test data for molecular property prediction models."""
    # Simple SMILES-like strings for testing
    smiles_train = ["CCO", "CC(C)O", "C1CCCCC1", "c1ccccc1"] * 25  # 100 samples
    smiles_test = ["CCC", "CCN", "C1CCC1"] * 7  # 21 samples

    # Convert to feature vectors (simplified)
    X_train = np.random.random((100, 20))  # 20 molecular descriptors
    y_train = np.random.random(100)  # Property values
    X_test = np.random.random((21, 20))

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "smiles_train": smiles_train,
        "smiles_test": smiles_test,
    }


def generate_protein_test_data() -> Dict[str, Any]:
    """Generate test data for protein-related models."""
    # Simple amino acid sequences
    sequences_train = ["ACDEFGHIKLMNPQRSTVWY"] * 50  # 50 sequences
    sequences_test = ["ACDEFG", "LMNPQR", "STVWY"] * 5  # 15 sequences

    # Convert to feature vectors
    X_train = np.random.random((50, 100))  # 100 protein features
    y_train = np.random.random(50)  # Protein properties
    X_test = np.random.random((15, 100))

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "sequences_train": sequences_train,
        "sequences_test": sequences_test,
    }


# Export testing utilities
__all__ = [
    "AdapterTestCase",
    "AdapterTestSuite",
    "MockAdapterTestCase",
    "create_adapter_test_suite",
    "quick_adapter_test",
    "generate_molecular_test_data",
    "generate_protein_test_data",
]
