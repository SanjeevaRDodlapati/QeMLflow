"""
Environment Validation Module

This module provides comprehensive environment validation capabilities to ensure
reproducible scientific computing environments with detailed validation reports
and automated fixing suggestions.
"""

import logging
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from .environment import EnvironmentFingerprint, EnvironmentManager


class ValidationLevel(Enum):
    """Validation strictness levels."""
    STRICT = "strict"
    MODERATE = "moderate"
    LENIENT = "lenient"


class ValidationStatus(Enum):
    """Validation result status."""
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"
    ERROR = "error"


@dataclass
class ValidationIssue:
    """Individual validation issue."""
    
    category: str
    severity: ValidationStatus
    message: str
    expected: Any = None
    actual: Any = None
    suggestion: str = ""
    fixable: bool = False
    fix_command: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'category': self.category,
            'severity': self.severity.value,
            'message': self.message,
            'expected': self.expected,
            'actual': self.actual,
            'suggestion': self.suggestion,
            'fixable': self.fixable,
            'fix_command': self.fix_command
        }


@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    
    timestamp: str
    environment_hash: str
    validation_level: ValidationLevel
    overall_status: ValidationStatus
    issues: List[ValidationIssue] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Calculate statistics after initialization."""
        self._calculate_statistics()
    
    def _calculate_statistics(self):
        """Calculate validation statistics."""
        self.statistics = {
            'total_checks': len(self.issues),
            'passed': len([i for i in self.issues if i.severity == ValidationStatus.PASSED]),
            'warnings': len([i for i in self.issues if i.severity == ValidationStatus.WARNING]),
            'failed': len([i for i in self.issues if i.severity == ValidationStatus.FAILED]),
            'errors': len([i for i in self.issues if i.severity == ValidationStatus.ERROR]),
            'fixable_issues': len([i for i in self.issues if i.fixable])
        }
    
    def add_issue(self, issue: ValidationIssue):
        """Add validation issue and recalculate statistics."""
        self.issues.append(issue)
        self._calculate_statistics()
        
        # Update overall status based on worst issue
        if issue.severity == ValidationStatus.ERROR:
            self.overall_status = ValidationStatus.ERROR
        elif issue.severity == ValidationStatus.FAILED and self.overall_status != ValidationStatus.ERROR:
            self.overall_status = ValidationStatus.FAILED
        elif issue.severity == ValidationStatus.WARNING and self.overall_status not in [ValidationStatus.ERROR, ValidationStatus.FAILED]:
            self.overall_status = ValidationStatus.WARNING
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp,
            'environment_hash': self.environment_hash,
            'validation_level': self.validation_level.value,
            'overall_status': self.overall_status.value,
            'issues': [issue.to_dict() for issue in self.issues],
            'statistics': self.statistics,
            'recommendations': self.recommendations
        }
    
    def get_fixable_issues(self) -> List[ValidationIssue]:
        """Get list of automatically fixable issues."""
        return [issue for issue in self.issues if issue.fixable]
    
    def generate_summary(self) -> str:
        """Generate human-readable summary."""
        stats = self.statistics
        
        summary = [
            f"Environment Validation Report - {self.overall_status.value.upper()}",
            f"Generated: {self.timestamp}",
            f"Validation Level: {self.validation_level.value}",
            "",
            "Summary:",
            f"  Total Checks: {stats['total_checks']}",
            f"  Passed: {stats['passed']}",
            f"  Warnings: {stats['warnings']}",
            f"  Failed: {stats['failed']}",
            f"  Errors: {stats['errors']}",
            f"  Fixable Issues: {stats['fixable_issues']}",
            ""
        ]
        
        if self.issues:
            summary.append("Issues Found:")
            for issue in self.issues:
                if issue.severity != ValidationStatus.PASSED:
                    summary.append(f"  [{issue.severity.value.upper()}] {issue.category}: {issue.message}")
                    if issue.suggestion:
                        summary.append(f"    Suggestion: {issue.suggestion}")
        
        if self.recommendations:
            summary.extend(["", "Recommendations:"])
            for rec in self.recommendations:
                summary.append(f"  - {rec}")
        
        return "\n".join(summary)


class EnvironmentValidator:
    """
    Comprehensive environment validator for reproducible scientific computing.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.env_manager = EnvironmentManager()
    
    def validate_environment(self, expected: EnvironmentFingerprint,
                           level: ValidationLevel = ValidationLevel.MODERATE) -> ValidationReport:
        """Perform comprehensive environment validation."""
        
        current = self.env_manager.capture_current_environment()
        
        report = ValidationReport(
            timestamp=datetime.now().isoformat(),
            environment_hash=current.fingerprint_hash,
            validation_level=level,
            overall_status=ValidationStatus.PASSED
        )
        
        # Validate Python version
        self._validate_python_version(expected, current, report, level)
        
        # Validate platform compatibility
        self._validate_platform(expected, current, report, level)
        
        # Validate packages
        self._validate_packages(expected, current, report, level)
        
        # Validate environment variables
        self._validate_environment_variables(expected, current, report, level)
        
        # Validate system requirements
        self._validate_system_requirements(expected, current, report, level)
        
        # Generate recommendations
        self._generate_recommendations(report)
        
        return report
    
    def _validate_python_version(self, expected: EnvironmentFingerprint,
                                current: EnvironmentFingerprint,
                                report: ValidationReport,
                                level: ValidationLevel):
        """Validate Python version compatibility."""
        
        if expected.python_version == current.python_version:
            report.add_issue(ValidationIssue(
                category="Python Version",
                severity=ValidationStatus.PASSED,
                message=f"Python version matches: {current.python_version}"
            ))
        else:
            # Parse version numbers for compatibility check
            expected_parts = [int(x) for x in expected.python_version.split('.')]
            current_parts = [int(x) for x in current.python_version.split('.')]
            
            # Check major.minor compatibility
            if expected_parts[:2] == current_parts[:2]:
                # Same major.minor, different patch
                severity = ValidationStatus.WARNING if level != ValidationLevel.STRICT else ValidationStatus.FAILED
                message = f"Python patch version differs: expected {expected.python_version}, got {current.python_version}"
                suggestion = "Consider updating to exact Python version for strict reproducibility"
            else:
                # Different major or minor version
                severity = ValidationStatus.FAILED
                message = f"Python version mismatch: expected {expected.python_version}, got {current.python_version}"
                suggestion = f"Install Python {expected.python_version} using pyenv or conda"
            
            report.add_issue(ValidationIssue(
                category="Python Version",
                severity=severity,
                message=message,
                expected=expected.python_version,
                actual=current.python_version,
                suggestion=suggestion,
                fixable=True,
                fix_command=f"pyenv install {expected.python_version} && pyenv global {expected.python_version}"
            ))
    
    def _validate_platform(self, expected: EnvironmentFingerprint,
                          current: EnvironmentFingerprint,
                          report: ValidationReport,
                          level: ValidationLevel):
        """Validate platform compatibility."""
        
        critical_attrs = ['system', 'python_implementation']
        important_attrs = ['architecture', 'machine']
        info_attrs = ['release', 'version']
        
        # Check critical attributes
        for attr in critical_attrs:
            expected_val = expected.platform_info.get(attr)
            current_val = current.platform_info.get(attr)
            
            if expected_val != current_val:
                report.add_issue(ValidationIssue(
                    category="Platform Compatibility",
                    severity=ValidationStatus.FAILED,
                    message=f"Critical platform difference in {attr}: expected {expected_val}, got {current_val}",
                    expected=expected_val,
                    actual=current_val,
                    suggestion=f"Environment requires {attr}={expected_val}",
                    fixable=False
                ))
            else:
                report.add_issue(ValidationIssue(
                    category="Platform Compatibility",
                    severity=ValidationStatus.PASSED,
                    message=f"Platform {attr} matches: {current_val}"
                ))
        
        # Check important attributes (architecture compatibility)
        for attr in important_attrs:
            expected_val = expected.platform_info.get(attr)
            current_val = current.platform_info.get(attr)
            
            if expected_val != current_val:
                severity = ValidationStatus.WARNING if level == ValidationLevel.LENIENT else ValidationStatus.FAILED
                report.add_issue(ValidationIssue(
                    category="Platform Compatibility",
                    severity=severity,
                    message=f"Platform {attr} differs: expected {expected_val}, got {current_val}",
                    expected=expected_val,
                    actual=current_val,
                    suggestion="Binary packages may not be compatible across architectures"
                ))
        
        # Check informational attributes (in strict mode only)
        if level == ValidationLevel.STRICT:
            for attr in info_attrs:
                expected_val = expected.platform_info.get(attr)
                current_val = current.platform_info.get(attr)
                
                if expected_val != current_val:
                    report.add_issue(ValidationIssue(
                        category="Platform Compatibility",
                        severity=ValidationStatus.WARNING,
                        message=f"Platform {attr} differs: expected {expected_val}, got {current_val}",
                        expected=expected_val,
                        actual=current_val,
                        suggestion="May affect build reproducibility"
                    ))
    
    def _validate_packages(self, expected: EnvironmentFingerprint,
                          current: EnvironmentFingerprint,
                          report: ValidationReport,
                          level: ValidationLevel):
        """Validate package installation and versions."""
        
        expected_packages = {pkg.name: pkg for pkg in expected.packages}
        current_packages = {pkg.name: pkg for pkg in current.packages}
        
        # Check for missing packages
        missing_packages = set(expected_packages.keys()) - set(current_packages.keys())
        for pkg_name in missing_packages:
            expected_pkg = expected_packages[pkg_name]
            report.add_issue(ValidationIssue(
                category="Package Installation",
                severity=ValidationStatus.FAILED,
                message=f"Missing package: {pkg_name}=={expected_pkg.version}",
                expected=f"{pkg_name}=={expected_pkg.version}",
                actual="not installed",
                suggestion="Install missing package",
                fixable=True,
                fix_command=f"pip install {pkg_name}=={expected_pkg.version}"
            ))
        
        # Check for extra packages (informational only)
        extra_packages = set(current_packages.keys()) - set(expected_packages.keys())
        if extra_packages and level == ValidationLevel.STRICT:
            for pkg_name in extra_packages:
                current_pkg = current_packages[pkg_name]
                report.add_issue(ValidationIssue(
                    category="Package Installation",
                    severity=ValidationStatus.WARNING,
                    message=f"Extra package installed: {pkg_name}=={current_pkg.version}",
                    expected="not installed",
                    actual=f"{pkg_name}=={current_pkg.version}",
                    suggestion="Remove extra packages for exact reproducibility"
                ))
        
        # Check version compatibility for common packages
        for pkg_name in expected_packages:
            if pkg_name in current_packages:
                expected_pkg = expected_packages[pkg_name]
                current_pkg = current_packages[pkg_name]
                
                if expected_pkg.version == current_pkg.version:
                    report.add_issue(ValidationIssue(
                        category="Package Versions",
                        severity=ValidationStatus.PASSED,
                        message=f"Package version matches: {pkg_name}=={current_pkg.version}"
                    ))
                else:
                    # Version mismatch - severity depends on validation level
                    if level == ValidationLevel.STRICT:
                        severity = ValidationStatus.FAILED
                    else:
                        # Check if versions are compatible (same major version)
                        if self._are_versions_compatible(expected_pkg.version, current_pkg.version):
                            severity = ValidationStatus.WARNING
                        else:
                            severity = ValidationStatus.FAILED
                    
                    report.add_issue(ValidationIssue(
                        category="Package Versions",
                        severity=severity,
                        message=f"Version mismatch for {pkg_name}: expected {expected_pkg.version}, got {current_pkg.version}",
                        expected=expected_pkg.version,
                        actual=current_pkg.version,
                        suggestion="Update to exact version for reproducibility",
                        fixable=True,
                        fix_command=f"pip install {pkg_name}=={expected_pkg.version}"
                    ))
    
    def _validate_environment_variables(self, expected: EnvironmentFingerprint,
                                      current: EnvironmentFingerprint,
                                      report: ValidationReport,
                                      level: ValidationLevel):
        """Validate critical environment variables."""
        
        critical_vars = ['PYTHONPATH', 'VIRTUAL_ENV', 'CONDA_DEFAULT_ENV']
        
        for var in critical_vars:
            expected_val = expected.environment_variables.get(var)
            current_val = current.environment_variables.get(var)
            
            if expected_val != current_val and expected_val is not None:
                severity = ValidationStatus.WARNING if level == ValidationLevel.LENIENT else ValidationStatus.FAILED
                report.add_issue(ValidationIssue(
                    category="Environment Variables",
                    severity=severity,
                    message=f"Environment variable {var} differs: expected {expected_val}, got {current_val}",
                    expected=expected_val,
                    actual=current_val,
                    suggestion=f"Set {var}={expected_val}",
                    fixable=True,
                    fix_command=f"export {var}={expected_val}"
                ))
    
    def _validate_system_requirements(self, expected: EnvironmentFingerprint,
                                    current: EnvironmentFingerprint,
                                    report: ValidationReport,
                                    level: ValidationLevel):
        """Validate system-level requirements."""
        
        # Check available memory (if specified)
        expected_memory = expected.system_info.get('total_memory')
        current_memory = current.system_info.get('total_memory')
        
        if expected_memory and current_memory:
            if current_memory < expected_memory * 0.9:  # Allow 10% tolerance
                report.add_issue(ValidationIssue(
                    category="System Requirements",
                    severity=ValidationStatus.WARNING,
                    message=f"Insufficient memory: expected {expected_memory // (1024**3)}GB, got {current_memory // (1024**3)}GB",
                    expected=f"{expected_memory // (1024**3)}GB",
                    actual=f"{current_memory // (1024**3)}GB",
                    suggestion="Ensure sufficient system memory for reproducible results"
                ))
    
    def _are_versions_compatible(self, expected: str, actual: str) -> bool:
        """Check if two versions are compatible (same major version)."""
        try:
            expected_parts = [int(x) for x in expected.split('.')]
            actual_parts = [int(x) for x in actual.split('.')]
            
            # Compatible if major version is the same
            return expected_parts[0] == actual_parts[0]
        except (ValueError, IndexError):
            return False
    
    def _generate_recommendations(self, report: ValidationReport):
        """Generate actionable recommendations based on validation results."""
        
        failed_issues = [i for i in report.issues if i.severity == ValidationStatus.FAILED]
        warning_issues = [i for i in report.issues if i.severity == ValidationStatus.WARNING]
        
        if failed_issues:
            report.recommendations.append("Address all failed validation checks before proceeding")
            
            # Group by category for better recommendations
            package_failures = [i for i in failed_issues if i.category in ["Package Installation", "Package Versions"]]
            if package_failures:
                report.recommendations.append("Consider using 'pip install -r requirements-exact.txt' for exact package versions")
        
        if warning_issues:
            report.recommendations.append("Review warnings to ensure consistent results")
        
        fixable_count = len(report.get_fixable_issues())
        if fixable_count > 0:
            report.recommendations.append(f"{fixable_count} issues can be automatically fixed")
    
    def auto_fix_issues(self, report: ValidationReport, 
                       dry_run: bool = True) -> Dict[str, Any]:
        """Automatically fix validation issues where possible."""
        
        fix_results = {
            'attempted_fixes': 0,
            'successful_fixes': 0,
            'failed_fixes': 0,
            'fix_log': [],
            'dry_run': dry_run
        }
        
        fixable_issues = report.get_fixable_issues()
        fix_results['attempted_fixes'] = len(fixable_issues)
        
        for issue in fixable_issues:
            if issue.fix_command:
                try:
                    if dry_run:
                        fix_results['fix_log'].append(f"[DRY RUN] Would execute: {issue.fix_command}")
                        fix_results['successful_fixes'] += 1
                    else:
                        # Execute fix command
                        result = subprocess.run(
                            issue.fix_command,
                            shell=True,
                            capture_output=True,
                            text=True,
                            timeout=60
                        )
                        
                        if result.returncode == 0:
                            fix_results['fix_log'].append(f"✓ Fixed: {issue.message}")
                            fix_results['successful_fixes'] += 1
                        else:
                            fix_results['fix_log'].append(f"✗ Failed to fix: {issue.message} - {result.stderr}")
                            fix_results['failed_fixes'] += 1
                
                except Exception as e:
                    fix_results['fix_log'].append(f"✗ Error fixing {issue.message}: {e}")
                    fix_results['failed_fixes'] += 1
        
        return fix_results


# Global validator instance
_environment_validator: Optional[EnvironmentValidator] = None


def get_environment_validator() -> EnvironmentValidator:
    """Get the global environment validator instance."""
    global _environment_validator
    if _environment_validator is None:
        _environment_validator = EnvironmentValidator()
    return _environment_validator


def validate_current_environment(expected_file: str, 
                                level: ValidationLevel = ValidationLevel.MODERATE) -> ValidationReport:
    """Validate current environment against expected fingerprint file."""
    from .environment import EnvironmentManager
    
    manager = EnvironmentManager()
    expected = manager.load_fingerprint(expected_file)
    
    validator = get_environment_validator()
    return validator.validate_environment(expected, level)


def validate_fingerprint(expected: EnvironmentFingerprint,
                        level: ValidationLevel = ValidationLevel.MODERATE) -> ValidationReport:
    """Validate current environment against expected fingerprint."""
    validator = get_environment_validator()
    return validator.validate_environment(expected, level)


def auto_fix_environment(report: ValidationReport, dry_run: bool = True) -> Dict[str, Any]:
    """Automatically fix environment validation issues."""
    validator = get_environment_validator()
    return validator.auto_fix_issues(report, dry_run)
