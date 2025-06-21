"""
Tests for Security Hardening system.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from qemlflow.security import (
    SecurityScanner,
    AuthenticationManager,
    SSLManager,
    SecurityMonitor,
    ComplianceValidator,
    SecurityHardening,
    initialize_security,
    get_security_status,
    run_security_audit
)


class TestSecurityScanner:
    """Test security vulnerability scanning."""
    
    @pytest.fixture
    def config(self):
        return {
            'scanning': {
                'enabled': True,
                'tools': {
                    'safety': True,
                    'bandit': True
                }
            }
        }
    
    @pytest.fixture
    def security_scanner(self, config):
        return SecurityScanner(config)
    
    def test_initialization(self, security_scanner):
        """Test security scanner initialization."""
        assert security_scanner.config is not None
        assert len(security_scanner.scan_results) == 0
    
    def test_scan_dependencies(self, security_scanner):
        """Test dependency vulnerability scanning."""
        with patch.object(security_scanner, '_run_safety_check') as mock_safety, \
             patch.object(security_scanner, '_run_bandit_scan') as mock_bandit:
            
            mock_safety.return_value = {'vulnerabilities': []}
            mock_bandit.return_value = {'vulnerabilities': []}
            
            result = security_scanner.scan_dependencies()
            
            assert result['scan_type'] == 'dependencies'
            assert result['status'] == 'completed'
            assert 'summary' in result
            assert len(security_scanner.scan_results) == 1
    
    def test_safety_check_with_vulnerabilities(self, security_scanner):
        """Test safety check with mock vulnerabilities."""
        mock_subprocess_result = Mock()
        mock_subprocess_result.returncode = 1
        mock_subprocess_result.stdout = '''[
            {
                "package": "test-package",
                "installed_version": "1.0.0",
                "vulnerability_id": "12345",
                "advisory": "Test vulnerability"
            }
        ]'''
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = mock_subprocess_result
            
            result = security_scanner._run_safety_check()
            
            assert len(result['vulnerabilities']) == 1
            assert result['vulnerabilities'][0]['package'] == 'test-package'
            assert result['vulnerabilities'][0]['type'] == 'dependency'
    
    def test_bandit_scan_with_issues(self, security_scanner):
        """Test Bandit scan with mock security issues."""
        mock_subprocess_result = Mock()
        mock_subprocess_result.stdout = '''
        {
            "results": [
                {
                    "filename": "test.py",
                    "line_number": 10,
                    "test_id": "B123",
                    "issue_severity": "HIGH",
                    "issue_confidence": "HIGH",
                    "issue_text": "Test security issue"
                }
            ]
        }'''
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = mock_subprocess_result
            
            result = security_scanner._run_bandit_scan()
            
            assert len(result['vulnerabilities']) == 1
            assert result['vulnerabilities'][0]['file'] == 'test.py'
            assert result['vulnerabilities'][0]['type'] == 'code'
            assert result['vulnerabilities'][0]['severity'] == 'high'
    
    def test_summarize_vulnerabilities(self, security_scanner):
        """Test vulnerability summarization."""
        vulnerabilities = [
            {'severity': 'high', 'type': 'dependency'},
            {'severity': 'medium', 'type': 'code'},
            {'severity': 'low', 'type': 'dependency'}
        ]
        
        summary = security_scanner._summarize_vulnerabilities(vulnerabilities)
        
        assert summary['total'] == 3
        assert summary['by_severity']['high'] == 1
        assert summary['by_severity']['medium'] == 1
        assert summary['by_severity']['low'] == 1
        assert summary['by_type']['dependency'] == 2
        assert summary['by_type']['code'] == 1


class TestAuthenticationManager:
    """Test authentication and authorization."""
    
    @pytest.fixture
    def config_enabled(self):
        return {
            'authentication': {
                'enabled': True,
                'type': 'token',
                'token_length': 32
            }
        }
    
    @pytest.fixture
    def config_disabled(self):
        return {
            'authentication': {
                'enabled': False
            }
        }
    
    @pytest.fixture
    def auth_manager_enabled(self, config_enabled):
        return AuthenticationManager(config_enabled)
    
    @pytest.fixture
    def auth_manager_disabled(self, config_disabled):
        return AuthenticationManager(config_disabled)
    
    def test_initialization_enabled(self, auth_manager_enabled):
        """Test auth manager initialization when enabled."""
        assert auth_manager_enabled.auth_enabled is True
    
    def test_initialization_disabled(self, auth_manager_disabled):
        """Test auth manager initialization when disabled."""
        assert auth_manager_disabled.auth_enabled is False
    
    def test_setup_authentication_disabled(self, auth_manager_disabled):
        """Test authentication setup when disabled."""
        result = auth_manager_disabled.setup_authentication()
        
        assert result['status'] == 'disabled'
        assert 'Authentication is disabled' in result['message']
    
    def test_setup_token_authentication(self, auth_manager_enabled):
        """Test token-based authentication setup."""
        with patch('secrets.token_urlsafe') as mock_token:
            mock_token.return_value = 'test_token_12345'
            
            result = auth_manager_enabled.setup_authentication()
            
            assert result['status'] == 'configured'
            assert result['auth_type'] == 'token'
            assert 'token_file' in result
    
    def test_validate_credentials_disabled(self, auth_manager_disabled):
        """Test credential validation when auth is disabled."""
        result = auth_manager_disabled.validate_credentials({'token': 'any_token'})
        
        assert result is True
    
    def test_validate_token_success(self, auth_manager_enabled):
        """Test successful token validation."""
        # Create temporary token file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.token') as f:
            f.write('valid_token')
            token_file = f.name
        
        try:
            with patch('qemlflow.security.Path') as mock_path:
                mock_path_instance = Mock()
                mock_path_instance.exists.return_value = True
                mock_path_instance.read_text.return_value = 'valid_token'
                mock_path.return_value = mock_path_instance
                
                result = auth_manager_enabled._validate_token('valid_token')
                assert result is True
                
        finally:
            Path(token_file).unlink(missing_ok=True)
    
    def test_validate_token_failure(self, auth_manager_enabled):
        """Test failed token validation."""
        with patch('qemlflow.security.Path') as mock_path:
            mock_path_instance = Mock()
            mock_path_instance.exists.return_value = True
            mock_path_instance.read_text.return_value = 'valid_token'
            mock_path.return_value = mock_path_instance
            
            result = auth_manager_enabled._validate_token('invalid_token')
            assert result is False


class TestSSLManager:
    """Test SSL/TLS configuration."""
    
    @pytest.fixture
    def config_enabled(self):
        return {
            'ssl': {
                'enabled': True,
                'cert_file': 'test.crt',
                'key_file': 'test.key',
                'ca_file': 'ca.crt',
                'check_hostname': True
            }
        }
    
    @pytest.fixture
    def config_disabled(self):
        return {
            'ssl': {
                'enabled': False
            }
        }
    
    @pytest.fixture
    def ssl_manager_enabled(self, config_enabled):
        return SSLManager(config_enabled)
    
    @pytest.fixture
    def ssl_manager_disabled(self, config_disabled):
        return SSLManager(config_disabled)
    
    def test_initialization_enabled(self, ssl_manager_enabled):
        """Test SSL manager initialization when enabled."""
        assert ssl_manager_enabled.ssl_enabled is True
    
    def test_initialization_disabled(self, ssl_manager_disabled):
        """Test SSL manager initialization when disabled."""
        assert ssl_manager_disabled.ssl_enabled is False
    
    def test_setup_ssl_disabled(self, ssl_manager_disabled):
        """Test SSL setup when disabled."""
        result = ssl_manager_disabled.setup_ssl()
        
        assert result['status'] == 'disabled'
        assert 'SSL is disabled' in result['message']
    
    def test_setup_ssl_enabled(self, ssl_manager_enabled):
        """Test SSL setup when enabled."""
        with patch('ssl.create_default_context') as mock_context:
            mock_ssl_context = Mock()
            mock_context.return_value = mock_ssl_context
            
            result = ssl_manager_enabled.setup_ssl()
            
            assert result['status'] == 'configured'
            assert 'ssl_context' in result
            assert 'certificates' in result
    
    def test_load_certificates_missing_files(self, ssl_manager_enabled):
        """Test certificate loading with missing files."""
        config = {
            'cert_file': 'nonexistent.crt',
            'key_file': 'nonexistent.key',
            'ca_file': 'nonexistent_ca.crt'
        }
        
        result = ssl_manager_enabled._load_certificates(config)
        
        assert len(result['errors']) == 3
        assert len(result['loaded']) == 0
    
    def test_validate_certificates_disabled(self, ssl_manager_disabled):
        """Test certificate validation when SSL is disabled."""
        result = ssl_manager_disabled.validate_certificates()
        
        assert result['status'] == 'disabled'
        assert 'SSL is disabled' in result['message']


class TestSecurityMonitor:
    """Test security monitoring and intrusion detection."""
    
    @pytest.fixture
    def config(self):
        return {
            'monitoring': {
                'enabled': True,
                'alert_thresholds': {
                    'failed_auth_attempts': 5,
                    'unusual_access_patterns': 10
                }
            }
        }
    
    @pytest.fixture
    def security_monitor(self, config):
        return SecurityMonitor(config)
    
    def test_initialization(self, security_monitor):
        """Test security monitor initialization."""
        assert security_monitor.monitoring_enabled is True
        assert len(security_monitor.security_events) == 0
    
    def test_start_monitoring_enabled(self, security_monitor):
        """Test starting security monitoring when enabled."""
        result = security_monitor.start_monitoring()
        
        assert result['status'] == 'started'
        assert 'monitoring_types' in result
        assert len(result['monitoring_types']) > 0
    
    def test_start_monitoring_disabled(self):
        """Test starting security monitoring when disabled."""
        config = {'monitoring': {'enabled': False}}
        monitor = SecurityMonitor(config)
        
        result = monitor.start_monitoring()
        
        assert result['status'] == 'disabled'
        assert 'Security monitoring is disabled' in result['message']
    
    def test_log_security_event(self, security_monitor):
        """Test logging security events."""
        event_details = {
            'user': 'test_user',
            'action': 'failed_login',
            'severity': 'high'
        }
        
        security_monitor.log_security_event('authentication_failure', event_details)
        
        assert len(security_monitor.security_events) == 1
        event = security_monitor.security_events[0]
        assert event['event_type'] == 'authentication_failure'
        assert event['details'] == event_details
        assert event['severity'] == 'high'
    
    def test_get_security_events(self, security_monitor):
        """Test retrieving security events."""
        # Add some test events
        security_monitor.log_security_event('test_event_1', {'severity': 'low'})
        security_monitor.log_security_event('test_event_2', {'severity': 'medium'})
        
        events = security_monitor.get_security_events(24)
        
        assert len(events) == 2
        assert events[0]['event_type'] == 'test_event_1'
        assert events[1]['event_type'] == 'test_event_2'


class TestComplianceValidator:
    """Test compliance validation."""
    
    @pytest.fixture
    def config(self):
        return {
            'compliance': {
                'standards': ['general', 'pci', 'hipaa']
            },
            'authentication': {'enabled': True},
            'ssl': {'enabled': True},
            'monitoring': {'enabled': True}
        }
    
    @pytest.fixture
    def compliance_validator(self, config):
        return ComplianceValidator(config)
    
    def test_initialization(self, compliance_validator):
        """Test compliance validator initialization."""
        assert compliance_validator.config is not None
    
    def test_validate_compliance_general(self, compliance_validator):
        """Test general compliance validation."""
        with patch.object(compliance_validator, '_check_secure_configs', return_value=True), \
             patch.object(compliance_validator, '_check_dependency_security', return_value=True), \
             patch.object(compliance_validator, '_check_authentication_enabled', return_value=True), \
             patch.object(compliance_validator, '_check_ssl_enabled', return_value=True), \
             patch.object(compliance_validator, '_check_security_monitoring', return_value=True):
            
            result = compliance_validator.validate_compliance(['general'])
            
            assert 'validation_id' in result
            assert result['standards'] == ['general']
            assert 'general' in result['results']
            assert result['results']['general']['score'] == 100.0
            assert result['overall_score'] == 100.0
    
    def test_validate_general_security_partial(self, compliance_validator):
        """Test general security validation with partial compliance."""
        with patch.object(compliance_validator, '_check_secure_configs', return_value=True), \
             patch.object(compliance_validator, '_check_dependency_security', return_value=False), \
             patch.object(compliance_validator, '_check_authentication_enabled', return_value=True), \
             patch.object(compliance_validator, '_check_ssl_enabled', return_value=False), \
             patch.object(compliance_validator, '_check_security_monitoring', return_value=True):
            
            result = compliance_validator._validate_general_security()
            
            assert result['score'] == 60.0  # 3 out of 5 checks passed
            assert len(result['recommendations']) == 2
    
    def test_check_authentication_enabled(self, compliance_validator):
        """Test authentication check."""
        result = compliance_validator._check_authentication_enabled()
        assert result is True
    
    def test_check_ssl_enabled(self, compliance_validator):
        """Test SSL check."""
        result = compliance_validator._check_ssl_enabled()
        assert result is True


class TestSecurityHardening:
    """Test main security hardening system."""
    
    @pytest.fixture
    def config(self):
        return {
            'scanning': {'enabled': True},
            'authentication': {'enabled': False, 'type': 'token'},
            'ssl': {'enabled': False},
            'monitoring': {'enabled': True},
            'compliance': {'standards': ['general']}
        }
    
    @pytest.fixture
    def security_hardening(self, config):
        with patch.object(SecurityHardening, '_load_config', return_value=config):
            return SecurityHardening()
    
    def test_initialization(self, security_hardening):
        """Test security hardening initialization."""
        assert security_hardening.scanner is not None
        assert security_hardening.auth_manager is not None
        assert security_hardening.ssl_manager is not None
        assert security_hardening.security_monitor is not None
        assert security_hardening.compliance_validator is not None
    
    def test_run_full_security_audit(self, security_hardening):
        """Test comprehensive security audit."""
        with patch.object(security_hardening.scanner, 'scan_dependencies') as mock_scan, \
             patch.object(security_hardening.compliance_validator, 'validate_compliance') as mock_compliance, \
             patch.object(security_hardening.security_monitor, 'get_security_events') as mock_events:
            
            mock_scan.return_value = {'status': 'completed', 'summary': {'total': 0}}
            mock_compliance.return_value = {'overall_score': 85.0}
            mock_events.return_value = []
            
            result = security_hardening.run_full_security_audit()
            
            assert 'audit_id' in result
            assert result['overall_status'] in ['passed', 'warning', 'failed']
            assert 'components' in result
            assert 'vulnerability_scan' in result['components']
            assert 'compliance' in result['components']
    
    def test_harden_system(self, security_hardening):
        """Test system hardening."""
        with patch.object(security_hardening.auth_manager, 'setup_authentication') as mock_auth, \
             patch.object(security_hardening.ssl_manager, 'setup_ssl') as mock_ssl, \
             patch.object(security_hardening.security_monitor, 'start_monitoring') as mock_monitor:
            
            mock_auth.return_value = {'status': 'disabled'}
            mock_ssl.return_value = {'status': 'disabled'}
            mock_monitor.return_value = {'status': 'started'}
            
            result = security_hardening.harden_system()
            
            assert 'hardening_id' in result
            assert result['status'] == 'completed'
            assert 'applied_measures' in result
            assert len(result['applied_measures']) == 1  # Only monitoring enabled
    
    def test_determine_audit_status_passed(self, security_hardening):
        """Test audit status determination - passed."""
        components = {
            'scan': {'status': 'completed'},
            'compliance': {'overall_score': 95.0}
        }
        
        status = security_hardening._determine_audit_status(components)
        assert status == 'passed'
    
    def test_determine_audit_status_warning(self, security_hardening):
        """Test audit status determination - warning."""
        components = {
            'scan': {'status': 'warning'},
            'compliance': {'overall_score': 75.0}
        }
        
        status = security_hardening._determine_audit_status(components)
        assert status == 'warning'
    
    def test_determine_audit_status_failed(self, security_hardening):
        """Test audit status determination - failed."""
        components = {
            'scan': {'status': 'failed'},
            'compliance': {'overall_score': 35.0}
        }
        
        status = security_hardening._determine_audit_status(components)
        assert status == 'failed'


class TestSecurityIntegration:
    """Test security system integration functions."""
    
    def test_initialize_security(self):
        """Test security system initialization."""
        with patch.object(SecurityHardening, '__init__', return_value=None):
            security_manager = initialize_security('config/security.yml')
            assert security_manager is not None
    
    def test_get_security_status_not_initialized(self):
        """Test getting security status when not initialized."""
        # Reset global state
        import qemlflow.security
        qemlflow.security._security_manager = None
        
        status = get_security_status()
        assert status['status'] == 'not_initialized'
    
    def test_get_security_status_initialized(self):
        """Test getting security status when initialized."""
        with patch.object(SecurityHardening, '__init__', return_value=None):
            initialize_security('config/security.yml')
            status = get_security_status()
            assert status['status'] == 'initialized'
            assert 'components' in status
    
    def test_run_security_audit_function(self):
        """Test standalone security audit function."""
        mock_audit_result = {
            'audit_id': 'test_audit',
            'overall_status': 'passed',
            'components': {}
        }
        
        with patch.object(SecurityHardening, '__init__', return_value=None), \
             patch.object(SecurityHardening, 'run_full_security_audit', return_value=mock_audit_result):
            
            result = run_security_audit('config/security.yml')
            assert result['audit_id'] == 'test_audit'
            assert result['overall_status'] == 'passed'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
