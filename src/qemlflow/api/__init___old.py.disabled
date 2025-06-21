"""
QeMLflow API Management Package

This package provides comprehensive API management including:
- Semantic versioning and version management
- API compatibility testing and validation
- Deprecation policy and management
- Backward compatibility testing
"""

from .versioning import (
    SemanticVersion,
    VersionType, 
    CompatibilityLevel,
    VersionManager,
    get_version_manager,
    get_version,
    version_info,
    check_compatibility,
    parse_version,
    compare_versions
)

from .compatibility import (
    APISignature,
    APIChange,
    APIAnalyzer,
    APICompatibilityChecker,
    APISnapshot,
    get_api_snapshot
)

from .deprecation import (
    DeprecationLevel,
    DeprecationInfo,
    DeprecationManager,
    deprecated,
    deprecate_parameter,
    get_deprecation_manager,
    register_deprecation,
    warn_deprecated,
    get_deprecation_status
)

from .backward_compatibility import (
    CompatibilityTest,
    CompatibilityTestResult,
    CompatibilityMatrix,
    RegressionTestRunner,
    get_compatibility_matrix,
    get_regression_runner,
    register_compatibility_test,
    check_version_compatibility
)

from .deprecation import (
    DeprecationPolicy,
    deprecated,
    deprecate_function,
    deprecate_class
)

from .testing import (
    BackwardCompatibilityTester,
    CompatibilityMatrix,
    RegressionTester,
    APITestSuite
)

__all__ = [
    # Versioning
    'SemanticVersion',
    'VersionManager', 
    'VersionCompatibility',
    'version_info',
    'get_version',
    'check_compatibility',
    
    # Compatibility
    'APICompatibilityChecker',
    'CompatibilityResult',
    'BreakingChange',
    'APISnapshot',
    'compatibility_test',
    
    # Deprecation
    'DeprecationManager',
    'QeMLflowDeprecationWarning',
    'DeprecationPolicy',
    'deprecated',
    'deprecate_function',
    'deprecate_class',
    
    # Testing
    'BackwardCompatibilityTester',
    'CompatibilityMatrix',
    'RegressionTester',
    'APITestSuite'
]

__version__ = '1.0.0'
