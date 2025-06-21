"""
Production Readiness Checklist for QeMLflow

This module provides a comprehensive production readiness validation system
that checks all aspects of the system for production deployment.
"""

import os
import logging
import subprocess
import psutil
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ReadinessCheck:
    """Production readiness check definition."""
    name: str
    description: str
    category: str
    check_function: Callable[[], bool]
    fix_suggestion: str
    critical: bool = True


class ProductionReadinessValidator:
    """Production readiness validation system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize production readiness validator."""
        self.config = config or {}
        self.checks: List[ReadinessCheck] = []
        self.results: Dict[str, Any] = {}
        
        # Initialize all checks
        self._register_security_checks()
        self._register_performance_checks()
        self._register_high_availability_checks()
        self._register_monitoring_checks()
        self._register_configuration_checks()
        self._register_deployment_checks()
        
        logger.info("Production readiness validator initialized with {} checks".format(len(self.checks)))
    
    def _register_security_checks(self):
        """Register security-related checks."""
        self.checks.extend([
            ReadinessCheck(
                name="security_module_available",
                description="Security module is available and functional",
                category="security",
                check_function=self._check_security_module,
                fix_suggestion="Ensure security module is properly installed and configured",
                critical=True
            ),
            ReadinessCheck(
                name="ssl_configuration",
                description="SSL/TLS is properly configured",
                category="security",
                check_function=self._check_ssl_config,
                fix_suggestion="Configure SSL certificates and enable SSL in production config",
                critical=True
            ),
            ReadinessCheck(
                name="authentication_enabled",
                description="Authentication is enabled for production",
                category="security",
                check_function=self._check_authentication,
                fix_suggestion="Enable authentication in production configuration",
                critical=True
            ),
            ReadinessCheck(
                name="vulnerability_scanning",
                description="Vulnerability scanning is configured",
                category="security",
                check_function=self._check_vulnerability_scanning,
                fix_suggestion="Install security scanning tools (safety, bandit)",
                critical=False
            )
        ])
    
    def _register_performance_checks(self):
        """Register performance-related checks."""
        self.checks.extend([
            ReadinessCheck(
                name="performance_tuning_available",
                description="Performance tuning module is available",
                category="performance",
                check_function=self._check_performance_tuning,
                fix_suggestion="Ensure production tuning module is properly installed",
                critical=True
            ),
            ReadinessCheck(
                name="resource_limits_configured",
                description="Resource limits are properly configured",
                category="performance",
                check_function=self._check_resource_limits,
                fix_suggestion="Configure CPU and memory limits in production config",
                critical=True
            ),
            ReadinessCheck(
                name="caching_enabled",
                description="Caching is enabled and configured",
                category="performance",
                check_function=self._check_caching,
                fix_suggestion="Enable and configure caching in production settings",
                critical=False
            )
        ])
    
    def _register_high_availability_checks(self):
        """Register high availability checks."""
        self.checks.extend([
            ReadinessCheck(
                name="high_availability_module",
                description="High availability module is available",
                category="high_availability",
                check_function=self._check_ha_module,
                fix_suggestion="Ensure high availability module is properly installed",
                critical=False
            ),
            ReadinessCheck(
                name="backup_configuration",
                description="Backup and restore is configured",
                category="high_availability",
                check_function=self._check_backup_config,
                fix_suggestion="Configure backup schedules and retention policies",
                critical=True
            ),
            ReadinessCheck(
                name="health_checks_enabled",
                description="Health checks are enabled and functional",
                category="high_availability",
                check_function=self._check_health_checks,
                fix_suggestion="Enable and configure health check endpoints",
                critical=True
            )
        ])
    
    def _register_monitoring_checks(self):
        """Register monitoring and observability checks."""
        self.checks.extend([
            ReadinessCheck(
                name="monitoring_system",
                description="Monitoring system is configured",
                category="monitoring",
                check_function=self._check_monitoring_system,
                fix_suggestion="Configure monitoring, metrics, and alerting",
                critical=True
            ),
            ReadinessCheck(
                name="logging_configuration",
                description="Production logging is properly configured",
                category="monitoring",
                check_function=self._check_logging_config,
                fix_suggestion="Configure structured logging with appropriate levels",
                critical=True
            ),
            ReadinessCheck(
                name="metrics_collection",
                description="Metrics collection is enabled",
                category="monitoring",
                check_function=self._check_metrics_collection,
                fix_suggestion="Enable metrics collection and export",
                critical=False
            )
        ])
    
    def _register_configuration_checks(self):
        """Register configuration checks."""
        self.checks.extend([
            ReadinessCheck(
                name="production_config_exists",
                description="Production configuration file exists",
                category="configuration",
                check_function=self._check_production_config,
                fix_suggestion="Create production.yml configuration file",
                critical=True
            ),
            ReadinessCheck(
                name="environment_variables",
                description="Required environment variables are set",
                category="configuration",
                check_function=self._check_environment_variables,
                fix_suggestion="Set all required production environment variables",
                critical=True
            ),
            ReadinessCheck(
                name="secrets_management",
                description="Secrets are properly managed",
                category="configuration",
                check_function=self._check_secrets_management,
                fix_suggestion="Use secure secrets management (not plain text)",
                critical=True
            )
        ])
    
    def _register_deployment_checks(self):
        """Register deployment-related checks."""
        self.checks.extend([
            ReadinessCheck(
                name="system_requirements",
                description="System meets minimum requirements",
                category="deployment",
                check_function=self._check_system_requirements,
                fix_suggestion="Ensure system meets minimum CPU, memory, and disk requirements",
                critical=True
            ),
            ReadinessCheck(
                name="dependencies_installed",
                description="All production dependencies are installed",
                category="deployment",
                check_function=self._check_dependencies,
                fix_suggestion="Install all required dependencies using requirements.txt",
                critical=True
            ),
            ReadinessCheck(
                name="network_connectivity",
                description="Network connectivity is available",
                category="deployment",
                check_function=self._check_network_connectivity,
                fix_suggestion="Ensure network connectivity for external services",
                critical=True
            )
        ])
    
    # Security check implementations
    def _check_security_module(self) -> bool:
        """Check if security module is available."""
        try:
            from qemlflow.security import SecurityScanner
            return True
        except ImportError:
            return False
    
    def _check_ssl_config(self) -> bool:
        """Check SSL configuration."""
        return self.config.get('security', {}).get('enable_ssl', False)
    
    def _check_authentication(self) -> bool:
        """Check authentication configuration."""
        return self.config.get('security', {}).get('enable_authentication', False)
    
    def _check_vulnerability_scanning(self) -> bool:
        """Check vulnerability scanning tools."""
        try:
            subprocess.run(['safety', '--version'], capture_output=True, check=True)
            subprocess.run(['bandit', '--version'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    # Performance check implementations
    def _check_performance_tuning(self) -> bool:
        """Check performance tuning module."""
        try:
            from qemlflow.production_tuning import ProductionPerformanceTuner
            return True
        except ImportError:
            return False
    
    def _check_resource_limits(self) -> bool:
        """Check resource limits configuration."""
        perf_config = self.config.get('performance', {})
        return (
            'cpu' in perf_config and
            'memory' in perf_config and
            'limits' in self.config.get('deployment', {})
        )
    
    def _check_caching(self) -> bool:
        """Check caching configuration."""
        return self.config.get('performance', {}).get('caching', {}).get('enabled', False)
    
    # High availability check implementations
    def _check_ha_module(self) -> bool:
        """Check high availability module."""
        try:
            from qemlflow.high_availability import HighAvailabilityManager
            return True
        except ImportError:
            return False
    
    def _check_backup_config(self) -> bool:
        """Check backup configuration."""
        return self.config.get('backup', {}).get('enabled', False)
    
    def _check_health_checks(self) -> bool:
        """Check health check configuration."""
        return self.config.get('reliability', {}).get('health_checks', {}).get('enabled', False)
    
    # Monitoring check implementations
    def _check_monitoring_system(self) -> bool:
        """Check monitoring system configuration."""
        return self.config.get('monitoring', {}).get('metrics', {}).get('enabled', False)
    
    def _check_logging_config(self) -> bool:
        """Check logging configuration."""
        logging_config = self.config.get('monitoring', {}).get('logging', {})
        return (
            logging_config.get('level') in ['INFO', 'WARNING', 'ERROR'] and
            logging_config.get('format') == 'structured'
        )
    
    def _check_metrics_collection(self) -> bool:
        """Check metrics collection."""
        return self.config.get('monitoring', {}).get('metrics', {}).get('enabled', False)
    
    # Configuration check implementations
    def _check_production_config(self) -> bool:
        """Check production configuration file."""
        config_path = Path('config/production.yml')
        return config_path.exists() and config_path.stat().st_size > 0
    
    def _check_environment_variables(self) -> bool:
        """Check required environment variables."""
        required_vars = [
            'QEMLFLOW_ENV',
            'QEMLFLOW_CONFIG_PATH'
        ]
        return all(os.getenv(var) for var in required_vars)
    
    def _check_secrets_management(self) -> bool:
        """Check secrets management."""
        # Check that sensitive values are not in plain text
        config_str = str(self.config)
        suspicious_patterns = ['password=', 'secret=', 'key=', 'token=']
        return not any(pattern in config_str.lower() for pattern in suspicious_patterns)
    
    # Deployment check implementations
    def _check_system_requirements(self) -> bool:
        """Check system requirements."""
        try:
            # Check CPU cores
            cpu_count = psutil.cpu_count()
            if cpu_count < 4:
                return False
            
            # Check memory
            memory = psutil.virtual_memory()
            if memory.total < 8 * 1024 * 1024 * 1024:  # 8 GB
                return False
            
            # Check disk space
            disk = psutil.disk_usage('/')
            if disk.free < 50 * 1024 * 1024 * 1024:  # 50 GB
                return False
            
            return True
        except Exception:
            return False
    
    def _check_dependencies(self) -> bool:
        """Check if all dependencies are installed."""
        try:
            import yaml
            import psutil
            import requests
            return True
        except ImportError:
            return False
    
    def _check_network_connectivity(self) -> bool:
        """Check network connectivity."""
        try:
            import requests
            response = requests.get('https://www.google.com', timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all production readiness checks."""
        logger.info("Starting production readiness validation")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'total_checks': len(self.checks),
            'passed_checks': 0,
            'failed_checks': 0,
            'critical_failures': 0,
            'categories': {},
            'checks': [],
            'overall_status': 'unknown',
            'readiness_score': 0.0
        }
        
        for check in self.checks:
            logger.debug(f"Running check: {check.name}")
            
            try:
                passed = check.check_function()
                
                check_result = {
                    'name': check.name,
                    'description': check.description,
                    'category': check.category,
                    'passed': passed,
                    'critical': check.critical,
                    'fix_suggestion': check.fix_suggestion if not passed else None
                }
                
                results['checks'].append(check_result)
                
                if passed:
                    results['passed_checks'] += 1
                else:
                    results['failed_checks'] += 1
                    if check.critical:
                        results['critical_failures'] += 1
                
                # Update category statistics
                category = check.category
                if category not in results['categories']:
                    results['categories'][category] = {
                        'total': 0,
                        'passed': 0,
                        'failed': 0,
                        'critical_failures': 0
                    }
                
                results['categories'][category]['total'] += 1
                if passed:
                    results['categories'][category]['passed'] += 1
                else:
                    results['categories'][category]['failed'] += 1
                    if check.critical:
                        results['categories'][category]['critical_failures'] += 1
                
            except Exception as e:
                logger.error(f"Error running check {check.name}: {e}")
                check_result = {
                    'name': check.name,
                    'description': check.description,
                    'category': check.category,
                    'passed': False,
                    'critical': check.critical,
                    'error': str(e),
                    'fix_suggestion': check.fix_suggestion
                }
                results['checks'].append(check_result)
                results['failed_checks'] += 1
                if check.critical:
                    results['critical_failures'] += 1
        
        # Determine overall status
        if results['critical_failures'] == 0:
            if results['failed_checks'] == 0:
                results['overall_status'] = 'ready'
            else:
                results['overall_status'] = 'ready_with_warnings'
        else:
            results['overall_status'] = 'not_ready'
        
        # Calculate readiness score
        if results['total_checks'] == 0:
            results['readiness_score'] = 0.0
        else:
            results['readiness_score'] = results['passed_checks'] / results['total_checks']  # type: ignore
        
        self.results = results
        logger.info(f"Production readiness validation completed: {results['overall_status']}")
        
        return results
    
    def get_readiness_report(self) -> str:
        """Get a formatted readiness report."""
        if not self.results:
            return "No readiness check results available. Run run_all_checks() first."
        
        report = []
        report.append("=" * 80)
        report.append("QEMLFLOW PRODUCTION READINESS REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {self.results['timestamp']}")
        report.append(f"Overall Status: {self.results['overall_status'].upper()}")
        report.append(f"Readiness Score: {self.results['readiness_score']:.1%}")
        report.append("")
        
        report.append("SUMMARY:")
        report.append(f"  Total Checks: {self.results['total_checks']}")
        report.append(f"  Passed: {self.results['passed_checks']}")
        report.append(f"  Failed: {self.results['failed_checks']}")
        report.append(f"  Critical Failures: {self.results['critical_failures']}")
        report.append("")
        
        # Category breakdown
        report.append("CATEGORY BREAKDOWN:")
        for category, stats in self.results['categories'].items():
            report.append(f"  {category.title()}:")
            report.append(f"    Passed: {stats['passed']}/{stats['total']}")
            if stats['failed'] > 0:
                report.append(f"    Failed: {stats['failed']} (Critical: {stats['critical_failures']})")
        report.append("")
        
        # Failed checks
        failed_checks = [check for check in self.results['checks'] if not check['passed']]
        if failed_checks:
            report.append("FAILED CHECKS:")
            for check in failed_checks:
                status = "CRITICAL" if check['critical'] else "WARNING"
                report.append(f"  [{status}] {check['name']}")
                report.append(f"    Description: {check['description']}")
                if 'error' in check:
                    report.append(f"    Error: {check['error']}")
                if check['fix_suggestion']:
                    report.append(f"    Fix: {check['fix_suggestion']}")
                report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        if self.results['overall_status'] == 'ready':
            report.append("  ✅ System is production ready!")
        elif self.results['overall_status'] == 'ready_with_warnings':
            report.append("  ⚠️  System is ready but has non-critical issues to address")
        else:
            report.append("  ❌ System is NOT production ready - fix critical issues first")
        
        report.append("=" * 80)
        
        return "\n".join(report)


# Utility functions
def validate_production_readiness(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Validate production readiness with given configuration."""
    validator = ProductionReadinessValidator(config)
    return validator.run_all_checks()


def generate_readiness_report(config: Optional[Dict[str, Any]] = None) -> str:
    """Generate a formatted production readiness report."""
    validator = ProductionReadinessValidator(config)
    validator.run_all_checks()
    return validator.get_readiness_report()


# Export public API
__all__ = [
    'ProductionReadinessValidator',
    'ReadinessCheck',
    'validate_production_readiness',
    'generate_readiness_report'
]
