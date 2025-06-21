"""
Backward Compatibility Testing Framework

This module provides comprehensive backward compatibility testing including:
- Regression testing for API changes
- Compatibility matrix generation
- Integration testing across versions
- Automated compatibility validation
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

from .compatibility import APISnapshot, APICompatibilityChecker
from .versioning import VersionManager


@dataclass
class CompatibilityTest:
    """Represents a backward compatibility test case."""
    
    name: str
    description: str
    test_code: str
    expected_result: Any
    minimum_version: str
    test_type: str = "functional"  # 'functional', 'performance', 'api'
    tags: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class CompatibilityTestResult:
    """Results from running a compatibility test."""
    
    test_name: str
    version: str
    success: bool
    execution_time: float
    error_message: str = ""
    actual_result: Any = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'test_name': self.test_name,
            'version': self.version,
            'success': self.success,
            'execution_time': self.execution_time,
            'error_message': self.error_message,
            'actual_result': self.actual_result,
            'metadata': self.metadata
        }


class CompatibilityMatrix:
    """Manages compatibility testing across multiple versions."""
    
    def __init__(self, matrix_file: str = "compatibility_matrix.json"):
        self.matrix_file = Path(matrix_file)
        self.logger = logging.getLogger(__name__)
        self.version_manager = VersionManager()
        
        # Load existing matrix
        self.matrix: Dict[str, Dict[str, bool]] = self._load_matrix()
        
        # Test registry
        self.tests: Dict[str, CompatibilityTest] = {}
        self.test_results: Dict[str, List[CompatibilityTestResult]] = {}
    
    def _load_matrix(self) -> Dict[str, Dict[str, bool]]:
        """Load compatibility matrix from file."""
        if not self.matrix_file.exists():
            return {}
        
        try:
            with open(self.matrix_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return cast(Dict[str, Dict[str, bool]], data.get('matrix', {}))
        except Exception as e:
            self.logger.error(f"Failed to load compatibility matrix: {e}")
            return {}
    
    def _save_matrix(self) -> None:
        """Save compatibility matrix to file."""
        try:
            data = {
                'matrix': self.matrix,
                'last_updated': datetime.now().isoformat(),
                'total_combinations': sum(len(versions) for versions in self.matrix.values())
            }
            
            with open(self.matrix_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            self.logger.debug(f"Saved compatibility matrix with {len(self.matrix)} versions")
        except Exception as e:
            self.logger.error(f"Failed to save compatibility matrix: {e}")
    
    def register_test(self, test: CompatibilityTest) -> None:
        """Register a new compatibility test."""
        self.tests[test.name] = test
        if test.name not in self.test_results:
            self.test_results[test.name] = []
        
        self.logger.info(f"Registered compatibility test: {test.name}")
    
    def set_compatibility(self, from_version: str, to_version: str, compatible: bool) -> None:
        """Set compatibility between two versions."""
        if from_version not in self.matrix:
            self.matrix[from_version] = {}
        
        self.matrix[from_version][to_version] = compatible
        self._save_matrix()
        
        self.logger.info(f"Set compatibility: {from_version} -> {to_version} = {compatible}")
    
    def is_compatible(self, from_version: str, to_version: str) -> Optional[bool]:
        """Check if one version is compatible with another."""
        if from_version in self.matrix:
            return self.matrix[from_version].get(to_version)
        return None
    
    def get_compatible_versions(self, base_version: str) -> List[str]:
        """Get all versions compatible with the base version."""
        if base_version not in self.matrix:
            return []
        
        return [
            version for version, compatible in self.matrix[base_version].items()
            if compatible
        ]
    
    def run_compatibility_test(self, test_name: str, version: str) -> CompatibilityTestResult:
        """Run a specific compatibility test against a version."""
        if test_name not in self.tests:
            raise ValueError(f"Test '{test_name}' not found")
        
        test = self.tests[test_name]
        start_time = datetime.now()
        
        try:
            # Execute test code in isolated environment
            result = self._execute_test_code(test.test_code, version)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Check if result matches expectation
            success = self._compare_results(result, test.expected_result)
            
            test_result = CompatibilityTestResult(
                test_name=test_name,
                version=version,
                success=success,
                execution_time=execution_time,
                actual_result=result
            )
            
            if not success:
                test_result.error_message = f"Expected {test.expected_result}, got {result}"
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            test_result = CompatibilityTestResult(
                test_name=test_name,
                version=version,
                success=False,
                execution_time=execution_time,
                error_message=str(e)
            )
        
        # Store result
        self.test_results[test_name].append(test_result)
        
        self.logger.info(f"Test {test_name} on {version}: {'PASS' if test_result.success else 'FAIL'}")
        return test_result
    
    def _execute_test_code(self, code: str, version: str) -> Any:
        """Execute test code in isolated environment."""
        # For now, execute in current environment
        # In production, this could use docker containers or virtual environments
        
        local_vars = {'__version__': version}
        global_vars = {'__builtins__': __builtins__}
        
        try:
            exec(code, global_vars, local_vars)
            return local_vars.get('result', None)
        except Exception as e:
            raise RuntimeError(f"Test execution failed: {e}")
    
    def _compare_results(self, actual: Any, expected: Any) -> bool:
        """Compare actual and expected test results."""
        if expected is None:
            return True  # No specific expectation
        
        if callable(expected):
            # Expected result is a validation function
            try:
                return bool(expected(actual))
            except Exception:
                return False
        
        # Direct comparison
        try:
            return bool(actual == expected)
        except Exception:
            return False
    
    def run_test_suite(self, versions: Optional[List[str]] = None) -> Dict[str, List[CompatibilityTestResult]]:
        """Run all tests against specified versions."""
        if versions is None:
            versions = [str(self.version_manager.current_version)]
        
        results: Dict[str, List[CompatibilityTestResult]] = {}
        
        for test_name in self.tests:
            results[test_name] = []
            for version in versions:
                try:
                    result = self.run_compatibility_test(test_name, version)
                    results[test_name].append(result)
                except Exception as e:
                    self.logger.error(f"Failed to run test {test_name} on {version}: {e}")
        
        return results
    
    def generate_compatibility_report(self, versions: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate comprehensive compatibility report."""
        if versions is None:
            versions = list(self.matrix.keys())
        
        # Run tests
        test_results = self.run_test_suite(versions)
        
        # Analyze results
        total_tests = len(self.tests) * len(versions)
        passed_tests = sum(
            1 for test_list in test_results.values()
            for result in test_list
            if result.success
        )
        
        # Version compatibility summary
        version_compatibility = {}
        for version in versions:
            version_results = [
                result for test_list in test_results.values()
                for result in test_list
                if result.version == version
            ]
            
            version_compatibility[version] = {
                'total_tests': len(version_results),
                'passed': sum(1 for r in version_results if r.success),
                'failed': sum(1 for r in version_results if not r.success),
                'success_rate': (
                    sum(1 for r in version_results if r.success) / len(version_results) * 100
                    if version_results else 0
                )
            }
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'overall_success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            'versions_tested': versions,
            'version_compatibility': version_compatibility,
            'test_results': {
                name: [result.to_dict() for result in results]
                for name, results in test_results.items()
            },
            'compatibility_matrix': self.matrix
        }


class RegressionTestRunner:
    """Runs regression tests to ensure backward compatibility."""
    
    def __init__(self, test_directory: str = "tests/compatibility"):
        self.test_directory = Path(test_directory)
        self.test_directory.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.api_snapshot = APISnapshot()
        self.compatibility_checker = APICompatibilityChecker()
        self.matrix = CompatibilityMatrix()
    
    def create_regression_test(self, name: str, description: str, 
                             test_function: str, expected_behavior: Any,
                             minimum_version: str) -> None:
        """Create a new regression test."""
        
        test = CompatibilityTest(
            name=name,
            description=description,
            test_code=test_function,
            expected_result=expected_behavior,
            minimum_version=minimum_version,
            test_type="regression"
        )
        
        self.matrix.register_test(test)
        
        # Save test to file
        test_file = self.test_directory / f"{name}.py"
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(f'"""\n{description}\n"""\n\n')
            f.write(test_function)
        
        self.logger.info(f"Created regression test: {name}")
    
    def run_api_compatibility_check(self, old_version: str, 
                                   new_version: str) -> Dict[str, Any]:
        """Run API compatibility check between versions."""
        
        # Compare API snapshots
        comparison = self.api_snapshot.compare_snapshots(old_version, new_version)
        
        if comparison is None:
            return {
                'error': f'Missing snapshots for versions {old_version} or {new_version}',
                'compatible': False
            }
        
        # Determine compatibility based on changes
        breaking_changes = [
            change for change in comparison.get('changes', [])
            if change.get('breaking', False)
        ]
        
        compatible = len(breaking_changes) == 0
        
        # Update compatibility matrix
        self.matrix.set_compatibility(old_version, new_version, compatible)
        
        return {
            'compatible': compatible,
            'breaking_changes': len(breaking_changes),
            'total_changes': comparison.get('total_changes', 0),
            'compatibility_level': comparison.get('compatibility_level', 'UNKNOWN'),
            'changes': comparison.get('changes', [])
        }
    
    def validate_version_compatibility(self, versions: List[str]) -> Dict[str, Any]:
        """Validate compatibility across multiple versions."""
        
        results = {}
        
        # Check each version pair
        for i, old_version in enumerate(versions[:-1]):
            for new_version in versions[i+1:]:
                key = f"{old_version}->{new_version}"
                results[key] = self.run_api_compatibility_check(old_version, new_version)
        
        # Generate summary
        compatible_pairs = sum(1 for r in results.values() if r.get('compatible', False))
        total_pairs = len(results)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'versions': versions,
            'total_comparisons': total_pairs,
            'compatible_pairs': compatible_pairs,
            'compatibility_rate': (compatible_pairs / total_pairs * 100) if total_pairs > 0 else 0,
            'results': results
        }
    
    def run_full_regression_suite(self) -> Dict[str, Any]:
        """Run complete regression test suite."""
        
        # Get available versions from snapshots
        snapshots = self.api_snapshot.list_snapshots()
        versions = [s['version'] for s in snapshots]
        
        if len(versions) < 2:
            return {
                'error': 'Need at least 2 version snapshots for regression testing',
                'available_versions': versions
            }
        
        # Run compatibility validation
        version_compatibility = self.validate_version_compatibility(versions)
        
        # Run functional tests
        functional_tests = self.matrix.generate_compatibility_report(versions)
        
        # Combine results
        return {
            'timestamp': datetime.now().isoformat(),
            'api_compatibility': version_compatibility,
            'functional_compatibility': functional_tests,
            'overall_status': self._determine_overall_status(
                version_compatibility, functional_tests
            )
        }
    
    def _determine_overall_status(self, api_results: Dict[str, Any], 
                                func_results: Dict[str, Any]) -> str:
        """Determine overall compatibility status."""
        
        api_rate = api_results.get('compatibility_rate', 0)
        func_rate = func_results.get('overall_success_rate', 0)
        
        overall_rate = (api_rate + func_rate) / 2
        
        if overall_rate >= 95:
            return "EXCELLENT"
        elif overall_rate >= 90:
            return "GOOD"
        elif overall_rate >= 80:
            return "ACCEPTABLE"
        elif overall_rate >= 70:
            return "CONCERNING"
        else:
            return "POOR"


# Global instances
_compatibility_matrix: Optional[CompatibilityMatrix] = None
_regression_runner: Optional[RegressionTestRunner] = None


def get_compatibility_matrix() -> CompatibilityMatrix:
    """Get the global compatibility matrix instance."""
    global _compatibility_matrix
    if _compatibility_matrix is None:
        _compatibility_matrix = CompatibilityMatrix()
    return _compatibility_matrix


def get_regression_runner() -> RegressionTestRunner:
    """Get the global regression test runner instance."""
    global _regression_runner
    if _regression_runner is None:
        _regression_runner = RegressionTestRunner()
    return _regression_runner


def register_compatibility_test(name: str, description: str, test_code: str,
                               expected_result: Any, minimum_version: str) -> None:
    """Register a new compatibility test."""
    test = CompatibilityTest(
        name=name,
        description=description,
        test_code=test_code,
        expected_result=expected_result,
        minimum_version=minimum_version
    )
    get_compatibility_matrix().register_test(test)


def check_version_compatibility(from_version: str, to_version: str) -> Optional[bool]:
    """Check compatibility between two versions."""
    return get_compatibility_matrix().is_compatible(from_version, to_version)
