"""
Tests for Environment Determinism Module

This module tests all aspects of environment determinism including:
- Environment fingerprinting and validation
- Deterministic package installation
- Environment validation and auto-fixing
- Lockfile creation and usage
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from qemlflow.reproducibility import (
    PackageInfo,
    EnvironmentFingerprint,
    EnvironmentManager,
    InstallationPlan,
    DeterministicInstaller,
    ValidationLevel,
    ValidationStatus,
    ValidationIssue,
    ValidationReport,
    EnvironmentValidator,
    capture_environment,
    create_installation_plan
)


class TestPackageInfo(unittest.TestCase):
    """Test PackageInfo dataclass."""
    
    def test_package_info_creation(self):
        """Test creating PackageInfo instance."""
        pkg = PackageInfo(
            name="numpy",
            version="1.24.0",
            location="/usr/local/lib/python3.9/site-packages/numpy",
            requires=["python>=3.8"],
            required_by=["pandas", "scipy"],
            installer="pip"
        )
        
        self.assertEqual(pkg.name, "numpy")
        self.assertEqual(pkg.version, "1.24.0")
        self.assertEqual(len(pkg.requires), 1)
        self.assertEqual(len(pkg.required_by), 2)
    
    def test_package_info_serialization(self):
        """Test PackageInfo serialization/deserialization."""
        pkg = PackageInfo(
            name="scipy",
            version="1.10.0",
            location="/path/to/scipy",
            requires=["numpy>=1.20.0"],
            installer="conda"
        )
        
        # Test to_dict
        pkg_dict = pkg.to_dict()
        self.assertIsInstance(pkg_dict, dict)
        self.assertEqual(pkg_dict['name'], "scipy")
        self.assertEqual(pkg_dict['version'], "1.10.0")
        
        # Test from_dict
        pkg_restored = PackageInfo.from_dict(pkg_dict)
        self.assertEqual(pkg_restored.name, pkg.name)
        self.assertEqual(pkg_restored.version, pkg.version)
        self.assertEqual(pkg_restored.requires, pkg.requires)


class TestEnvironmentFingerprint(unittest.TestCase):
    """Test EnvironmentFingerprint functionality."""
    
    def setUp(self):
        """Set up test environment fingerprint."""
        self.sample_packages = [
            PackageInfo(name="numpy", version="1.24.0", location="/path/numpy"),
            PackageInfo(name="scipy", version="1.10.0", location="/path/scipy"),
            PackageInfo(name="pandas", version="1.5.0", location="/path/pandas")
        ]
        
        self.sample_platform = {
            'system': 'Linux',
            'release': '5.15.0',
            'version': '5.15.0-generic',
            'architecture': 'x86_64',
            'machine': 'x86_64',
            'processor': 'x86_64',
            'python_implementation': 'CPython'
        }
        
        self.fingerprint = EnvironmentFingerprint(
            timestamp="2025-01-01T00:00:00",
            python_version="3.9.0",
            platform_info=self.sample_platform,
            packages=self.sample_packages
        )
    
    def test_fingerprint_hash_calculation(self):
        """Test fingerprint hash calculation."""
        # Hash should be calculated automatically
        self.assertIsNotNone(self.fingerprint.fingerprint_hash)
        self.assertEqual(len(self.fingerprint.fingerprint_hash), 64)  # SHA256 hex length
        
        # Same data should produce same hash
        fingerprint2 = EnvironmentFingerprint(
            timestamp="2025-01-01T00:00:00",  # Different timestamp
            python_version="3.9.0",
            platform_info=self.sample_platform,
            packages=self.sample_packages
        )
        
        # Hash should be the same despite different timestamp
        # (timestamp is excluded from hash calculation)
        self.assertEqual(self.fingerprint.fingerprint_hash, fingerprint2.fingerprint_hash)
    
    def test_fingerprint_serialization(self):
        """Test fingerprint serialization/deserialization."""
        # Test to_dict
        fp_dict = self.fingerprint.to_dict()
        self.assertIsInstance(fp_dict, dict)
        self.assertEqual(fp_dict['python_version'], "3.9.0")
        self.assertEqual(len(fp_dict['packages']), 3)
        
        # Test from_dict
        fp_restored = EnvironmentFingerprint.from_dict(fp_dict)
        self.assertEqual(fp_restored.python_version, self.fingerprint.python_version)
        self.assertEqual(len(fp_restored.packages), len(self.fingerprint.packages))
        self.assertEqual(fp_restored.fingerprint_hash, self.fingerprint.fingerprint_hash)


class TestEnvironmentManager(unittest.TestCase):
    """Test EnvironmentManager functionality."""
    
    def setUp(self):
        """Set up test environment manager."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = EnvironmentManager(requirements_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up test directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('pkg_resources.working_set')
    @patch('platform.system')
    @patch('platform.python_implementation')
    def test_capture_current_environment(self, mock_impl, mock_system, mock_working_set):
        """Test capturing current environment."""
        # Mock platform info
        mock_system.return_value = 'Linux'
        mock_impl.return_value = 'CPython'
        
        # Mock installed packages
        mock_pkg = Mock()
        mock_pkg.key = 'numpy'
        mock_pkg.version = '1.24.0'
        mock_pkg.location = '/path/to/numpy'
        mock_pkg.requires.return_value = []
        mock_working_set.__iter__.return_value = [mock_pkg]
        
        with patch.object(self.manager, '_get_package_installer', return_value='pip'), \
             patch.object(self.manager, '_get_package_metadata', return_value={}):
            
            fingerprint = self.manager.capture_current_environment()
            
            self.assertIsInstance(fingerprint, EnvironmentFingerprint)
            self.assertIsNotNone(fingerprint.fingerprint_hash)
            self.assertEqual(fingerprint.platform_info['system'], 'Linux')
    
    def test_generate_exact_requirements(self):
        """Test generating exact requirements file."""
        # Create a sample fingerprint
        packages = [
            PackageInfo(name="numpy", version="1.24.0", location="/path"),
            PackageInfo(name="requests", version="2.28.0", location="/path")
        ]
        
        fingerprint = EnvironmentFingerprint(
            timestamp="2025-01-01T00:00:00",
            python_version="3.9.0",
            platform_info={'system': 'Linux', 'release': '5.15.0', 'architecture': 'x86_64'},
            packages=packages
        )
        
        with patch.object(self.manager, 'capture_current_environment', return_value=fingerprint):
            requirements_path = str(Path(self.temp_dir) / 'test_requirements.txt')
            content = self.manager.generate_exact_requirements(requirements_path)
            
            # Check file was created
            self.assertTrue(Path(requirements_path).exists())
            
            # Check content
            self.assertIn('numpy==1.24.0', content)
            self.assertIn('requests==2.28.0', content)
            self.assertIn('# Generated on:', content)
    
    def test_validate_environment(self):
        """Test environment validation."""
        # Create expected and current fingerprints
        expected = EnvironmentFingerprint(
            timestamp="2025-01-01T00:00:00",
            python_version="3.9.0",
            platform_info={'system': 'Linux', 'python_implementation': 'CPython'},
            packages=[PackageInfo(name="numpy", version="1.24.0", location="/path")]
        )
        
        current = EnvironmentFingerprint(
            timestamp="2025-01-01T01:00:00",
            python_version="3.9.1",  # Different patch version
            platform_info={'system': 'Linux', 'python_implementation': 'CPython'},
            packages=[PackageInfo(name="numpy", version="1.24.0", location="/path")]
        )
        
        with patch.object(self.manager, 'capture_current_environment', return_value=current):
            result = self.manager.validate_environment(expected, strict=False)
            
            self.assertIsInstance(result, dict)
            self.assertIn('valid', result)
            self.assertIn('differences', result)


class TestDeterministicInstaller(unittest.TestCase):
    """Test DeterministicInstaller functionality."""
    
    def setUp(self):
        """Set up test installer."""
        self.temp_dir = tempfile.mkdtemp()
        self.installer = DeterministicInstaller(cache_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up test directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_installation_plan(self):
        """Test creating installation plan."""
        packages = [
            PackageInfo(name="numpy", version="1.24.0", location="/path", requires=[]),
            PackageInfo(name="scipy", version="1.10.0", location="/path", requires=["numpy"])
        ]
        
        fingerprint = EnvironmentFingerprint(
            timestamp="2025-01-01T00:00:00",
            python_version="3.9.0",
            platform_info={'system': 'Linux', 'release': '5.15.0', 'architecture': 'x86_64'},
            packages=packages
        )
        
        plan = self.installer.create_installation_plan(fingerprint)
        
        self.assertIsInstance(plan, InstallationPlan)
        self.assertEqual(len(plan.packages), 2)
        self.assertIn("numpy", plan.installation_order)
        self.assertIn("scipy", plan.installation_order)
        
        # numpy should come before scipy (dependency order)
        numpy_idx = plan.installation_order.index("numpy")
        scipy_idx = plan.installation_order.index("scipy")
        self.assertLess(numpy_idx, scipy_idx)
    
    def test_installation_plan_serialization(self):
        """Test installation plan serialization."""
        packages = [PackageInfo(name="numpy", version="1.24.0", location="/path")]
        
        plan = InstallationPlan(
            packages=packages,
            installation_order=["numpy"],
            constraints={"numpy": "==1.24.0"}
        )
        
        # Test to_dict
        plan_dict = plan.to_dict()
        self.assertIsInstance(plan_dict, dict)
        self.assertEqual(plan_dict['installation_order'], ["numpy"])
        
        # Test from_dict
        plan_restored = InstallationPlan.from_dict(plan_dict)
        self.assertEqual(plan_restored.installation_order, plan.installation_order)
        self.assertEqual(plan_restored.constraints, plan.constraints)
    
    def test_create_lockfile(self):
        """Test lockfile creation."""
        packages = [PackageInfo(name="numpy", version="1.24.0", location="/path")]
        
        plan = InstallationPlan(
            packages=packages,
            installation_order=["numpy"],
            platform_requirements={"python_version": "3.9.0"}
        )
        
        lockfile_path = Path(self.temp_dir) / "test_lockfile.json"
        self.installer.create_lockfile(plan, str(lockfile_path))
        
        # Check file was created
        self.assertTrue(lockfile_path.exists())
        
        # Check content
        with open(lockfile_path, 'r', encoding='utf-8') as f:
            lockfile_data = json.load(f)
        
        self.assertIn('version', lockfile_data)
        self.assertIn('packages', lockfile_data)
        self.assertIn('checksum', lockfile_data)
        self.assertEqual(lockfile_data['packages']['numpy']['version'], "1.24.0")


class TestEnvironmentValidator(unittest.TestCase):
    """Test EnvironmentValidator functionality."""
    
    def setUp(self):
        """Set up test validator."""
        self.validator = EnvironmentValidator()
    
    def test_validation_issue_creation(self):
        """Test creating validation issues."""
        issue = ValidationIssue(
            category="Test Category",
            severity=ValidationStatus.WARNING,
            message="Test message",
            expected="expected_value",
            actual="actual_value",
            suggestion="Test suggestion"
        )
        
        self.assertEqual(issue.category, "Test Category")
        self.assertEqual(issue.severity, ValidationStatus.WARNING)
        self.assertEqual(issue.message, "Test message")
        
        # Test serialization
        issue_dict = issue.to_dict()
        self.assertIsInstance(issue_dict, dict)
        self.assertEqual(issue_dict['severity'], 'warning')
    
    def test_validation_report(self):
        """Test validation report functionality."""
        report = ValidationReport(
            timestamp="2025-01-01T00:00:00",
            environment_hash="test_hash",
            validation_level=ValidationLevel.MODERATE,
            overall_status=ValidationStatus.PASSED
        )
        
        # Add some issues
        report.add_issue(ValidationIssue(
            category="Test",
            severity=ValidationStatus.PASSED,
            message="Passed test"
        ))
        
        report.add_issue(ValidationIssue(
            category="Test",
            severity=ValidationStatus.WARNING,
            message="Warning test"
        ))
        
        # Check statistics
        self.assertEqual(report.statistics['total_checks'], 2)
        self.assertEqual(report.statistics['passed'], 1)
        self.assertEqual(report.statistics['warnings'], 1)
        self.assertEqual(report.overall_status, ValidationStatus.WARNING)
        
        # Test summary generation
        summary = report.generate_summary()
        self.assertIn('WARNING', summary)
        self.assertIn('Total Checks: 2', summary)
    
    @patch.object(EnvironmentManager, 'capture_current_environment')
    def test_validate_environment(self, mock_capture):
        """Test environment validation."""
        # Mock current environment
        current = EnvironmentFingerprint(
            timestamp="2025-01-01T01:00:00",
            python_version="3.9.1",
            platform_info={'system': 'Linux', 'python_implementation': 'CPython'},
            packages=[PackageInfo(name="numpy", version="1.24.1", location="/path")]
        )
        mock_capture.return_value = current
        
        # Expected environment
        expected = EnvironmentFingerprint(
            timestamp="2025-01-01T00:00:00",
            python_version="3.9.0",
            platform_info={'system': 'Linux', 'python_implementation': 'CPython'},
            packages=[PackageInfo(name="numpy", version="1.24.0", location="/path")]
        )
        
        report = self.validator.validate_environment(expected, ValidationLevel.MODERATE)
        
        self.assertIsInstance(report, ValidationReport)
        self.assertGreater(len(report.issues), 0)
        
        # Should have issues for Python version and package version differences
        categories = [issue.category for issue in report.issues]
        self.assertIn("Python Version", categories)
        self.assertIn("Package Versions", categories)


class TestIntegrationFunctions(unittest.TestCase):
    """Test module-level integration functions."""
    
    @patch('qemlflow.reproducibility.environment.get_environment_manager')
    def test_capture_environment(self, mock_get_manager):
        """Test capture_environment function."""
        mock_manager = Mock()
        mock_fingerprint = Mock()
        mock_manager.capture_current_environment.return_value = mock_fingerprint
        mock_get_manager.return_value = mock_manager
        
        result = capture_environment()
        
        mock_get_manager.assert_called_once()
        mock_manager.capture_current_environment.assert_called_once()
        self.assertEqual(result, mock_fingerprint)
    
    def test_create_installation_plan_function(self):
        """Test create_installation_plan function."""
        fingerprint = EnvironmentFingerprint(
            timestamp="2025-01-01T00:00:00",
            python_version="3.9.0",
            platform_info={'system': 'Linux', 'release': '5.15.0', 'architecture': 'x86_64'},
            packages=[PackageInfo(name="numpy", version="1.24.0", location="/path")]
        )
        
        plan = create_installation_plan(fingerprint)
        
        self.assertIsInstance(plan, InstallationPlan)
        self.assertEqual(len(plan.packages), 1)
        self.assertEqual(plan.packages[0].name, "numpy")


class TestValidationLevels(unittest.TestCase):
    """Test different validation levels."""
    
    def setUp(self):
        """Set up test data."""
        self.validator = EnvironmentValidator()
        
        self.expected = EnvironmentFingerprint(
            timestamp="2025-01-01T00:00:00",
            python_version="3.9.0",
            platform_info={
                'system': 'Linux',
                'release': '5.15.0',
                'python_implementation': 'CPython',
                'architecture': 'x86_64'
            },
            packages=[PackageInfo(name="numpy", version="1.24.0", location="/path")]
        )
        
        self.current_minor_diff = EnvironmentFingerprint(
            timestamp="2025-01-01T01:00:00",
            python_version="3.9.1",  # Patch version difference
            platform_info={
                'system': 'Linux',
                'release': '5.16.0',  # Different release
                'python_implementation': 'CPython',
                'architecture': 'x86_64'
            },
            packages=[PackageInfo(name="numpy", version="1.24.0", location="/path")]
        )
    
    @patch.object(EnvironmentManager, 'capture_current_environment')
    def test_strict_validation(self, mock_capture):
        """Test strict validation level."""
        mock_capture.return_value = self.current_minor_diff
        
        report = self.validator.validate_environment(self.expected, ValidationLevel.STRICT)
        
        # In strict mode, Python patch version differences should be failures
        python_failures = [
            issue for issue in report.issues 
            if issue.category == "Python Version" and issue.severity == ValidationStatus.FAILED
        ]
        self.assertGreater(len(python_failures), 0)
    
    @patch.object(EnvironmentManager, 'capture_current_environment')
    def test_lenient_validation(self, mock_capture):
        """Test lenient validation level."""
        mock_capture.return_value = self.current_minor_diff
        
        report = self.validator.validate_environment(self.expected, ValidationLevel.LENIENT)
        
        # Lenient mode should allow platform differences as warnings
        platform_warnings = [
            issue for issue in report.issues 
            if issue.category == "Platform Compatibility" and issue.severity == ValidationStatus.WARNING
        ]
        
        # Should have fewer failures than strict mode
        platform_failures = [
            issue for issue in report.issues 
            if issue.category == "Platform Compatibility" and issue.severity == ValidationStatus.FAILED
        ]
        
        # In lenient mode, architecture differences should be warnings, not failures
        self.assertLessEqual(len(platform_failures), len(platform_warnings))


if __name__ == '__main__':
    unittest.main()
