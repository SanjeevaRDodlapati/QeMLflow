"""
QeMLflow Reproducibility Package

This package provides comprehensive reproducibility infrastructure including:
- Environment determinism and exact dependency management
- Deterministic package installation and lockfile management
- Environment validation and automated fixing
- Experiment tracking and versioning
- Audit trail and data lineage tracking
- Validation frameworks for scientific reproducibility
"""

from .environment import (
    PackageInfo,
    EnvironmentFingerprint,
    EnvironmentManager,
    get_environment_manager,
    capture_environment,
    validate_environment,
    generate_requirements
)

from .deterministic_install import (
    InstallationPlan,
    DeterministicInstaller,
    get_deterministic_installer,
    create_installation_plan,
    install_deterministically,
    create_lockfile,
    install_from_lockfile
)

from .validation import (
    ValidationLevel,
    ValidationStatus,
    ValidationIssue,
    ValidationReport,
    EnvironmentValidator,
    get_environment_validator,
    validate_current_environment,
    validate_fingerprint,
    auto_fix_environment
)

__all__ = [
    # Environment Determinism
    'PackageInfo',
    'EnvironmentFingerprint', 
    'EnvironmentManager',
    'get_environment_manager',
    'capture_environment',
    'validate_environment',
    'generate_requirements',
    
    # Deterministic Installation
    'InstallationPlan',
    'DeterministicInstaller',
    'get_deterministic_installer',
    'create_installation_plan',
    'install_deterministically',
    'create_lockfile',
    'install_from_lockfile',
    
    # Environment Validation
    'ValidationLevel',
    'ValidationStatus',
    'ValidationIssue',
    'ValidationReport',
    'EnvironmentValidator',
    'get_environment_validator',
    'validate_current_environment',
    'validate_fingerprint',
    'auto_fix_environment'
]

# Version information
__version__ = "1.0.0"
