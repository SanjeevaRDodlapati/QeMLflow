"""
Tests for Production Readiness Checklist - Enhanced Robust Version
"""

from unittest.mock import Mock, patch

import pytest

from qemlflow.production_readiness import (
    ProductionReadinessValidator,
    ReadinessCheck,
    validate_production_readiness,
    generate_readiness_report
)


class TestReadinessCheck:
    """Test readiness check data structure."""
    
    def test_readiness_check_creation(self):
        """Test creating a readiness check."""
        def dummy_check():
            return True
        
        check = ReadinessCheck(
            name="test_check",
            description="Test check description",
            category="test",
            check_function=dummy_check,
            fix_suggestion="Fix suggestion",
            critical=True
        )
        
        assert check.name == "test_check"
        assert check.description == "Test check description"
        assert check.category == "test"
        assert check.check_function == dummy_check
        assert check.fix_suggestion == "Fix suggestion"
        assert check.critical is True
    
    def test_readiness_check_defaults(self):
        """Test readiness check default values."""
        def dummy_check():
            return False
        
        check = ReadinessCheck(
            name="test_check",
            description="Test description",
            category="test",
            check_function=dummy_check,
            fix_suggestion="Fix it"
        )
        
        # Default critical should be True
        assert check.critical is True
    
    def test_readiness_check_callable_validation(self):
        """Test that check_function must be callable."""
        def dummy_check():
            return True
        
        # This should work
        check = ReadinessCheck(
            name="test_check",
            description="Test description",
            category="test",
            check_function=dummy_check,
            fix_suggestion="Fix it"
        )
        
        assert callable(check.check_function)
        assert check.check_function() is True


class TestProductionReadinessValidator:
    """Test production readiness validator with comprehensive scenarios."""
    
    @pytest.fixture
    def minimal_config(self):
        """Minimal configuration for testing."""
        return {}
    
    @pytest.fixture
    def complete_config(self):
        """Complete configuration for testing."""
        return {
            'security': {
                'enable_ssl': True,
                'enable_authentication': True
            },
            'performance': {
                'cpu': {'max_cpu_usage': 0.85},
                'memory': {'gc_threshold': 0.8},
                'caching': {'enabled': True},
                'limits': {'cpu': '2000m', 'memory': '4Gi'}
            },
            'backup': {
                'enabled': True
            },
            'reliability': {
                'health_checks': {'enabled': True}
            },
            'monitoring': {
                'metrics': {'enabled': True},
                'logging': {'level': 'INFO', 'format': 'structured'}
            },
            'deployment': {
                'limits': {'cpu': '2000m', 'memory': '4Gi'}
            }
        }
    
    @pytest.fixture
    def partial_config(self):
        """Partial configuration missing some settings."""
        return {
            'security': {
                'enable_ssl': False
            },
            'performance': {
                'caching': {'enabled': False}
            },
            'monitoring': {
                'logging': {'level': 'DEBUG'}  # Missing format
            }
        }
    
    def test_initialization_empty_config(self):
        """Test validator initialization with empty config."""
        validator = ProductionReadinessValidator({})
        assert len(validator.checks) > 0
        assert validator.results == {}
        assert validator.config == {}
    
    def test_initialization_none_config(self):
        """Test validator initialization with None config."""
        validator = ProductionReadinessValidator(None)
        assert len(validator.checks) > 0
        assert validator.results == {}
        assert validator.config == {}
    
    def test_initialization_complete_config(self, complete_config):
        """Test validator initialization with complete config."""
        validator = ProductionReadinessValidator(complete_config)
        assert len(validator.checks) > 0
        assert validator.config == complete_config
        
        # Check that all expected categories are registered
        categories = {check.category for check in validator.checks}
        expected_categories = {
            'security', 'performance', 'high_availability',
            'monitoring', 'configuration', 'deployment'
        }
        assert categories == expected_categories
    
    def test_check_registration_completeness(self, complete_config):
        """Test that all check types are properly registered."""
        validator = ProductionReadinessValidator(complete_config)
        
        # Count checks by category
        category_counts = {}
        for check in validator.checks:
            category_counts[check.category] = category_counts.get(check.category, 0) + 1
        
        # Verify minimum number of checks per category
        assert category_counts['security'] >= 4
        assert category_counts['performance'] >= 3
        assert category_counts['high_availability'] >= 3
        assert category_counts['monitoring'] >= 3
        assert category_counts['configuration'] >= 3
        assert category_counts['deployment'] >= 3
    
    def test_security_checks_comprehensive(self, complete_config):
        """Test security checks comprehensively."""
        validator = ProductionReadinessValidator(complete_config)
        security_checks = [check for check in validator.checks if check.category == 'security']
        
        # Verify all expected security checks
        check_names = {check.name for check in security_checks}
        required_checks = {
            'security_module_available',
            'ssl_configuration',
            'authentication_enabled',
            'vulnerability_scanning'
        }
        assert required_checks.issubset(check_names)
        
        # Test that all security checks are marked as critical or properly classified
        for check in security_checks:
            assert hasattr(check, 'critical')
            assert isinstance(check.critical, bool)
    
    def test_performance_checks_comprehensive(self, complete_config):
        """Test performance checks comprehensively."""
        validator = ProductionReadinessValidator(complete_config)
        performance_checks = [check for check in validator.checks if check.category == 'performance']
        
        check_names = {check.name for check in performance_checks}
        required_checks = {
            'performance_tuning_available',
            'resource_limits_configured',
            'caching_enabled'
        }
        assert required_checks.issubset(check_names)
    
    def test_individual_check_functions(self, complete_config):
        """Test individual check functions work correctly."""
        validator = ProductionReadinessValidator(complete_config)
        
        # Test SSL config check
        assert validator._check_ssl_config() is True
        
        # Test authentication check
        assert validator._check_authentication() is True
        
        # Test caching check
        assert validator._check_caching() is True
        
        # Test backup config check
        assert validator._check_backup_config() is True
        
        # Test health checks
        assert validator._check_health_checks() is True
        
        # Test monitoring system
        assert validator._check_monitoring_system() is True
        
        # Test logging config
        assert validator._check_logging_config() is True
        
        # Test resource limits
        assert validator._check_resource_limits() is True
    
    def test_individual_check_functions_with_partial_config(self, partial_config):
        """Test individual check functions with partial config."""
        validator = ProductionReadinessValidator(partial_config)
        
        # These should fail with partial config
        assert validator._check_ssl_config() is False
        assert validator._check_authentication() is False
        assert validator._check_caching() is False
        assert validator._check_backup_config() is False
        assert validator._check_health_checks() is False
        assert validator._check_monitoring_system() is False
        assert validator._check_resource_limits() is False
        
        # Logging should fail due to missing format
        assert validator._check_logging_config() is False
    
    @patch('qemlflow.production_readiness.Path')
    def test_production_config_check_file_exists(self, mock_path_class, complete_config):
        """Test production config check when file exists."""
        validator = ProductionReadinessValidator(complete_config)
        
        # Mock path object
        mock_path = Mock()
        mock_path.exists.return_value = True
        mock_path.stat.return_value.st_size = 1000
        mock_path_class.return_value = mock_path
        
        result = validator._check_production_config()
        assert result is True
        mock_path_class.assert_called_once_with('config/production.yml')
    
    @patch('qemlflow.production_readiness.Path')
    def test_production_config_check_file_missing(self, mock_path_class, complete_config):
        """Test production config check when file is missing."""
        validator = ProductionReadinessValidator(complete_config)
        
        mock_path = Mock()
        mock_path.exists.return_value = False
        mock_path_class.return_value = mock_path
        
        result = validator._check_production_config()
        assert result is False
    
    @patch('qemlflow.production_readiness.Path')
    def test_production_config_check_file_empty(self, mock_path_class, complete_config):
        """Test production config check when file is empty."""
        validator = ProductionReadinessValidator(complete_config)
        
        mock_path = Mock()
        mock_path.exists.return_value = True
        mock_path.stat.return_value.st_size = 0
        mock_path_class.return_value = mock_path
        
        result = validator._check_production_config()
        assert result is False
    
    @patch('qemlflow.production_readiness.os.getenv')
    def test_environment_variables_all_present(self, mock_getenv, complete_config):
        """Test environment variables check when all are present."""
        validator = ProductionReadinessValidator(complete_config)
        
        # Mock all required environment variables
        env_vars = {
            'QEMLFLOW_ENV': 'production',
            'QEMLFLOW_CONFIG_PATH': '/etc/qemlflow/production.yml'
        }
        mock_getenv.side_effect = lambda var: env_vars.get(var)
        
        result = validator._check_environment_variables()
        assert result is True
    
    @patch('qemlflow.production_readiness.os.getenv')
    def test_environment_variables_missing(self, mock_getenv, complete_config):
        """Test environment variables check when some are missing."""
        validator = ProductionReadinessValidator(complete_config)
        
        # Mock only some environment variables
        env_vars = {
            'QEMLFLOW_ENV': 'production',
            # Missing QEMLFLOW_CONFIG_PATH
        }
        mock_getenv.side_effect = lambda var: env_vars.get(var)
        
        result = validator._check_environment_variables()
        assert result is False
    
    @patch('qemlflow.production_readiness.os.getenv')
    def test_environment_variables_empty_values(self, mock_getenv, complete_config):
        """Test environment variables check with empty values."""
        validator = ProductionReadinessValidator(complete_config)
        
        # Mock empty environment variables
        env_vars = {
            'QEMLFLOW_ENV': '',
            'QEMLFLOW_CONFIG_PATH': '/etc/qemlflow/production.yml'
        }
        mock_getenv.side_effect = lambda var: env_vars.get(var)
        
        result = validator._check_environment_variables()
        assert result is False
    
    def test_secrets_management_safe_config(self, complete_config):
        """Test secrets management with safe config."""
        validator = ProductionReadinessValidator(complete_config)
        result = validator._check_secrets_management()
        assert result is True
    
    def test_secrets_management_unsafe_config(self):
        """Test secrets management with unsafe config."""
        unsafe_config = {
            'database': {
                'password': 'plaintext_password',
                'secret': 'my_secret_key'
            }
        }
        validator = ProductionReadinessValidator(unsafe_config)
        result = validator._check_secrets_management()
        assert result is False
    
    def test_secrets_management_edge_cases(self):
        """Test secrets management with edge cases."""
        # Test with various suspicious patterns
        test_configs = [
            {'db': {'PASSWORD': 'test'}},  # Uppercase
            {'api': {'Secret': 'value'}},  # Mixed case
            {'auth': {'token': 'abc123'}},  # Token
            {'service': {'key': 'mykey'}},  # Key
        ]
        
        for config in test_configs:
            validator = ProductionReadinessValidator(config)
            result = validator._check_secrets_management()
            assert result is False, f"Should fail for config: {config}"
    
    @patch('qemlflow.production_readiness.psutil')
    def test_system_requirements_adequate(self, mock_psutil, complete_config):
        """Test system requirements check with adequate resources."""
        validator = ProductionReadinessValidator(complete_config)
        
        # Mock adequate system resources
        mock_psutil.cpu_count.return_value = 8
        mock_psutil.virtual_memory.return_value.total = 16 * 1024 * 1024 * 1024  # 16 GB
        mock_psutil.disk_usage.return_value.free = 100 * 1024 * 1024 * 1024  # 100 GB
        
        result = validator._check_system_requirements()
        assert result is True
    
    @patch('qemlflow.production_readiness.psutil')
    def test_system_requirements_cpu_insufficient(self, mock_psutil, complete_config):
        """Test system requirements check with insufficient CPU."""
        validator = ProductionReadinessValidator(complete_config)
        
        mock_psutil.cpu_count.return_value = 2  # Too few cores
        mock_psutil.virtual_memory.return_value.total = 16 * 1024 * 1024 * 1024
        mock_psutil.disk_usage.return_value.free = 100 * 1024 * 1024 * 1024
        
        result = validator._check_system_requirements()
        assert result is False
    
    @patch('qemlflow.production_readiness.psutil')
    def test_system_requirements_memory_insufficient(self, mock_psutil, complete_config):
        """Test system requirements check with insufficient memory."""
        validator = ProductionReadinessValidator(complete_config)
        
        mock_psutil.cpu_count.return_value = 8
        mock_psutil.virtual_memory.return_value.total = 4 * 1024 * 1024 * 1024  # 4 GB (too little)
        mock_psutil.disk_usage.return_value.free = 100 * 1024 * 1024 * 1024
        
        result = validator._check_system_requirements()
        assert result is False
    
    @patch('qemlflow.production_readiness.psutil')
    def test_system_requirements_disk_insufficient(self, mock_psutil, complete_config):
        """Test system requirements check with insufficient disk space."""
        validator = ProductionReadinessValidator(complete_config)
        
        mock_psutil.cpu_count.return_value = 8
        mock_psutil.virtual_memory.return_value.total = 16 * 1024 * 1024 * 1024
        mock_psutil.disk_usage.return_value.free = 10 * 1024 * 1024 * 1024  # 10 GB (too little)
        
        result = validator._check_system_requirements()
        assert result is False
    
    @patch('qemlflow.production_readiness.psutil')
    def test_system_requirements_exception_handling(self, mock_psutil, complete_config):
        """Test system requirements check with exception."""
        validator = ProductionReadinessValidator(complete_config)
        
        mock_psutil.cpu_count.side_effect = Exception("System error")
        
        result = validator._check_system_requirements()
        assert result is False
    
    def test_dependencies_check_success(self, complete_config):
        """Test dependencies check when all dependencies are available."""
        validator = ProductionReadinessValidator(complete_config)
        
        # This should pass in the test environment since we have these dependencies
        result = validator._check_dependencies()
        assert result is True
    
    @patch('builtins.__import__')
    def test_dependencies_check_import_error(self, mock_import, complete_config):
        """Test dependencies check with import errors."""
        validator = ProductionReadinessValidator(complete_config)
        
        def mock_import_side_effect(name, *args, **kwargs):
            if name in ['yaml', 'psutil', 'requests']:
                raise ImportError(f"No module named '{name}'")
            return __import__(name, *args, **kwargs)
        
        mock_import.side_effect = mock_import_side_effect
        
        result = validator._check_dependencies()
        assert result is False
    
    @patch('requests.get')
    def test_network_connectivity_success(self, mock_get, complete_config):
        """Test network connectivity check success."""
        validator = ProductionReadinessValidator(complete_config)
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        result = validator._check_network_connectivity()
        assert result is True
        mock_get.assert_called_once_with('https://www.google.com', timeout=5)
    
    @patch('requests.get')
    def test_network_connectivity_timeout(self, mock_get, complete_config):
        """Test network connectivity check with timeout."""
        validator = ProductionReadinessValidator(complete_config)
        
        mock_get.side_effect = Exception("Timeout")
        
        result = validator._check_network_connectivity()
        assert result is False
    
    @patch('requests.get')
    def test_network_connectivity_status_error(self, mock_get, complete_config):
        """Test network connectivity check with HTTP error."""
        validator = ProductionReadinessValidator(complete_config)
        
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        result = validator._check_network_connectivity()
        assert result is False
    
    @patch('qeMLflow.production_readiness.subprocess.run')
    def test_vulnerability_scanning_available(self, mock_run, complete_config):
        """Test vulnerability scanning when tools are available."""
        validator = ProductionReadinessValidator(complete_config)
        
        mock_run.return_value = Mock(returncode=0)
        
        result = validator._check_vulnerability_scanning()
        assert result is True
        
        # Verify both tools were checked
        assert mock_run.call_count == 2
    
    @patch('qemlflow.production_readiness.subprocess.run')
    def test_vulnerability_scanning_unavailable(self, mock_run, complete_config):
        """Test vulnerability scanning when tools are unavailable."""
        validator = ProductionReadinessValidator(complete_config)
        
        mock_run.side_effect = FileNotFoundError("Command not found")
        
        result = validator._check_vulnerability_scanning()
        assert result is False
    
    @patch('qemlflow.production_readiness.subprocess.run')
    def test_vulnerability_scanning_error(self, mock_run, complete_config):
        """Test vulnerability scanning with subprocess error."""
        validator = ProductionReadinessValidator(complete_config)
        
        from subprocess import CalledProcessError
        mock_run.side_effect = CalledProcessError(1, 'safety')
        
        result = validator._check_vulnerability_scanning()
        assert result is False
        expected_checks = {
            'production_config_exists',
            'environment_variables',
            'secrets_management'
        }
        assert expected_checks.issubset(check_names)
    
    def test_deployment_checks(self, validator):
        """Test deployment checks."""
        deployment_checks = [check for check in validator.checks if check.category == 'deployment']
        assert len(deployment_checks) >= 3
        
        check_names = {check.name for check in deployment_checks}
        expected_checks = {
            'system_requirements',
            'dependencies_installed',
            'network_connectivity'
        }
        assert expected_checks.issubset(check_names)
    
    def test_ssl_config_check(self, validator):
        """Test SSL configuration check."""
        result = validator._check_ssl_config()
        assert result is True  # Based on mock_config
    
    def test_authentication_check(self, validator):
        """Test authentication check."""
        result = validator._check_authentication()
        assert result is True  # Based on mock_config
    
    def test_caching_check(self, validator):
        """Test caching check."""
        result = validator._check_caching()
        assert result is True  # Based on mock_config
    
    def test_backup_config_check(self, validator):
        """Test backup configuration check."""
        result = validator._check_backup_config()
        assert result is True  # Based on mock_config
    
    def test_health_checks_check(self, validator):
        """Test health checks check."""
        result = validator._check_health_checks()
        assert result is True  # Based on mock_config
    
    def test_monitoring_system_check(self, validator):
        """Test monitoring system check."""
        result = validator._check_monitoring_system()
        assert result is True  # Based on mock_config
    
    def test_logging_config_check(self, validator):
        """Test logging configuration check."""
        result = validator._check_logging_config()
        assert result is True  # Based on mock_config
    
    def test_resource_limits_check(self, validator):
        """Test resource limits check."""
        result = validator._check_resource_limits()
        assert result is True  # Based on mock_config
    
    @patch('qemlflow.production_readiness.Path')
    def test_production_config_check(self, mock_path, validator):
        """Test production config check."""
        mock_path_obj = Mock()
        mock_path_obj.exists.return_value = True
        mock_path_obj.stat.return_value.st_size = 1000
        mock_path.return_value = mock_path_obj
        
        result = validator._check_production_config()
        assert result is True
    
    @patch('qemlflow.production_readiness.os.getenv')
    def test_environment_variables_check(self, mock_getenv, validator):
        """Test environment variables check."""
        mock_getenv.side_effect = lambda var: {
            'QEMLFLOW_ENV': 'production',
            'QEMLFLOW_CONFIG_PATH': '/etc/qemlflow/production.yml'
        }.get(var)
        
        result = validator._check_environment_variables()
        assert result is True
    
    def test_secrets_management_check(self, validator):
        """Test secrets management check."""
        result = validator._check_secrets_management()
        assert result is True  # Config doesn't contain plain text secrets
    
    @patch('qemlflow.production_readiness.psutil')
    def test_system_requirements_check(self, mock_psutil, validator):
        """Test system requirements check."""
        mock_psutil.cpu_count.return_value = 8
        mock_psutil.virtual_memory.return_value.total = 16 * 1024 * 1024 * 1024  # 16 GB
        mock_psutil.disk_usage.return_value.free = 100 * 1024 * 1024 * 1024  # 100 GB
        
        result = validator._check_system_requirements()
        assert result is True
    
    @patch('qemlflow.production_readiness.psutil')
    def test_system_requirements_insufficient(self, mock_psutil, validator):
        """Test system requirements check with insufficient resources."""
        mock_psutil.cpu_count.return_value = 2  # Too few cores
        mock_psutil.virtual_memory.return_value.total = 4 * 1024 * 1024 * 1024  # 4 GB (too little)
        mock_psutil.disk_usage.return_value.free = 10 * 1024 * 1024 * 1024  # 10 GB (too little)
        
        result = validator._check_system_requirements()
        assert result is False
    
    def test_dependencies_check(self, validator):
        """Test dependencies check."""
        # This should pass in the test environment since we have these dependencies
        result = validator._check_dependencies()
        assert result is True
    
    @patch('requests.get')
    def test_network_connectivity_check(self, mock_get, validator):
        """Test network connectivity check."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        result = validator._check_network_connectivity()
        assert result is True
    
    @patch('requests.get')
    def test_network_connectivity_failure(self, mock_get, validator):
        """Test network connectivity check failure."""
        mock_get.side_effect = Exception("Network error")
        
        result = validator._check_network_connectivity()
        assert result is False
    
    @patch('qemlflow.production_readiness.subprocess.run')
    def test_vulnerability_scanning_check(self, mock_run, validator):
        """Test vulnerability scanning check."""
        mock_run.return_value = Mock(returncode=0)
        
        result = validator._check_vulnerability_scanning()
        assert result is True
    
    @patch('qemlflow.production_readiness.subprocess.run')
    def test_vulnerability_scanning_not_available(self, mock_run, validator):
        """Test vulnerability scanning when tools not available."""
        mock_run.side_effect = FileNotFoundError("Command not found")
        
        result = validator._check_vulnerability_scanning()
        assert result is False
    
    def test_run_all_checks_success(self, validator):
        """Test running all checks with successful results."""
        # Mock all check functions to return True
        for check in validator.checks:
            check.check_function = Mock(return_value=True)
        
        results = validator.run_all_checks()
        
        assert results['total_checks'] == len(validator.checks)
        assert results['passed_checks'] == len(validator.checks)
        assert results['failed_checks'] == 0
        assert results['critical_failures'] == 0
        assert results['overall_status'] == 'ready'
        assert results['readiness_score'] == 1.0
        assert 'timestamp' in results
        assert 'categories' in results
        assert 'checks' in results
    
    def test_run_all_checks_with_failures(self, validator):
        """Test running all checks with some failures."""
        # Mock some check functions to fail
        critical_failed = 0
        non_critical_failed = 0
        
        for i, check in enumerate(validator.checks):
            if i % 3 == 0:  # Fail every third check
                check.check_function = Mock(return_value=False)
                if check.critical:
                    critical_failed += 1
                else:
                    non_critical_failed += 1
            else:
                check.check_function = Mock(return_value=True)
        
        results = validator.run_all_checks()
        
        total_failed = critical_failed + non_critical_failed
        assert results['failed_checks'] == total_failed
        assert results['critical_failures'] == critical_failed
        
        if critical_failed > 0:
            assert results['overall_status'] == 'not_ready'
        elif total_failed > 0:
            assert results['overall_status'] == 'ready_with_warnings'
        else:
            assert results['overall_status'] == 'ready'
    
    def test_run_all_checks_with_errors(self, validator):
        """Test running all checks with errors."""
        # Mock first check to raise an exception
        validator.checks[0].check_function = Mock(side_effect=Exception("Test error"))
        
        # Mock other checks to succeed
        for check in validator.checks[1:]:
            check.check_function = Mock(return_value=True)
        
        results = validator.run_all_checks()
        
        assert results['failed_checks'] >= 1
        error_check = next(check for check in results['checks'] if 'error' in check)
        assert error_check['error'] == "Test error"
    
    def test_get_readiness_report_no_results(self, validator):
        """Test getting readiness report without running checks."""
        report = validator.get_readiness_report()
        assert "No readiness check results available" in report
    
    def test_get_readiness_report_with_results(self, validator):
        """Test getting readiness report with results."""
        # Mock all checks to succeed
        for check in validator.checks:
            check.check_function = Mock(return_value=True)
        
        validator.run_all_checks()
        report = validator.get_readiness_report()
        
        assert "QEMLFLOW PRODUCTION READINESS REPORT" in report
        assert "Overall Status: READY" in report
        assert "Readiness Score: 100.0%" in report
        assert "SUMMARY:" in report
        assert "CATEGORY BREAKDOWN:" in report
        assert "System is production ready!" in report
    
    def test_get_readiness_report_with_failures(self, validator):
        """Test getting readiness report with failures."""
        # Mock some checks to fail
        for i, check in enumerate(validator.checks):
            check.check_function = Mock(return_value=(i % 2 == 0))  # Fail every other check
        
        validator.run_all_checks()
        report = validator.get_readiness_report()
        
        assert "FAILED CHECKS:" in report
        assert "CRITICAL" in report or "WARNING" in report


class TestUtilityFunctions:
    """Test utility functions."""
    
    @patch('qemlflow.production_readiness.ProductionReadinessValidator')
    def test_validate_production_readiness(self, mock_validator_class):
        """Test production readiness validation function."""
        mock_validator = Mock()
        mock_results = {'status': 'ready'}
        mock_validator.run_all_checks.return_value = mock_results
        mock_validator_class.return_value = mock_validator
        
        config = {'test': 'config'}
        result = validate_production_readiness(config)
        
        mock_validator_class.assert_called_once_with(config)
        mock_validator.run_all_checks.assert_called_once()
        assert result == mock_results
    
    @patch('qemlflow.production_readiness.ProductionReadinessValidator')
    def test_generate_readiness_report(self, mock_validator_class):
        """Test readiness report generation function."""
        mock_validator = Mock()
        mock_report = "Test readiness report"
        mock_validator.get_readiness_report.return_value = mock_report
        mock_validator_class.return_value = mock_validator
        
        config = {'test': 'config'}
        result = generate_readiness_report(config)
        
        mock_validator_class.assert_called_once_with(config)
        mock_validator.run_all_checks.assert_called_once()
        mock_validator.get_readiness_report.assert_called_once()
        assert result == mock_report


if __name__ == '__main__':
    pytest.main([__file__])
