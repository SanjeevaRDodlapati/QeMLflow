"""
🔧 QeMLflow Configuration Management

Enterprise-grade configuration management system providing:
- Multi-environment configuration loading
- Schema-based validation  
- Template-based configuration generation
- Configuration versioning and migration
- Secure credential management

Unified configuration management for QeMLflow.
"""

# Import existing configuration components
from .unified_config import (
    ConfigContext,
    ExperimentTrackingConfig,
    ModelConfig,
    NotebookConfig,
    PreprocessingConfig,
    QeMLflowConfig,
    QuantumConfig,
    VisualizationConfig,
    create_default_config_file,
    get_config,
    load_config as legacy_load_config,
    set_config,
)

# Import new enterprise configuration engine
from .engine import (
    ConfigurationEngine,
    EnvironmentType,
    ConfigurationMetadata,
    load_config,
    get_config_value,
    validate_environment
)

__all__ = [
    # Legacy configuration
    "QeMLflowConfig",
    "ExperimentTrackingConfig",
    "ModelConfig",
    "PreprocessingConfig",
    "VisualizationConfig",
    "QuantumConfig",
    "NotebookConfig",
    "get_config",
    "set_config",
    "legacy_load_config",
    "create_default_config_file",
    "ConfigContext",
    # Enterprise configuration engine
    "ConfigurationEngine",
    "EnvironmentType", 
    "ConfigurationMetadata",
    "load_config",
    "get_config_value",
    "validate_environment"
]

# Version information
__version__ = "2.0.0"
