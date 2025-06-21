"""
🔧 QeMLflow Configuration Management Engine

Enterprise-grade configuration management with environment-aware loading,
validation, templating, and versioning capabilities.

This module provides:
- Multi-environment configuration loading
- Schema-based validation
- Template-based configuration generation
- Configuration versioning and migration
- Secure credential management
"""

import os
import json
import yaml
import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import jsonschema
from jinja2 import Environment as Jinja2Environment, FileSystemLoader
import hashlib
from datetime import datetime

logger = logging.getLogger(__name__)


class EnvironmentType(Enum):
    """Supported deployment environments"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    LOCAL = "local"


@dataclass
class ConfigurationMetadata:
    """Configuration metadata for versioning and tracking"""
    version: str
    environment: str
    created_at: datetime
    checksum: str
    source_files: List[str] = field(default_factory=list)
    template_version: Optional[str] = None
    migration_applied: Optional[str] = None


class ConfigurationEngine:
    """
    Enterprise-grade configuration management engine
    
    Features:
    - Multi-source configuration loading with priority
    - Environment-aware configuration resolution
    - Schema-based validation
    - Template rendering with inheritance
    - Configuration versioning and migration
    """
    
    def __init__(
        self,
        config_root: Optional[Union[str, Path]] = None,
        environment: Optional[str] = None,
        schema_path: Optional[Union[str, Path]] = None
    ):
        """
        Initialize configuration engine
        
        Args:
            config_root: Root directory for configuration files
            environment: Target environment (auto-detected if None)
            schema_path: Path to configuration schema files
        """
        self.config_root = Path(config_root or self._get_default_config_root())
        self.schema_path = Path(schema_path or self.config_root / "schemas")
        self.templates_path = self.config_root / "templates"
        
        # Environment detection and resolution
        self.environment = self._detect_environment(environment)
        
        # Configuration state
        self._config_cache: Dict[str, Any] = {}
        self._metadata: Optional[ConfigurationMetadata] = None
        self._validation_schema: Optional[Dict] = None
        
        # Template engine
        self._template_env = Jinja2Environment(
            loader=FileSystemLoader(str(self.templates_path)),
            autoescape=True,
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        logger.info(f"Configuration engine initialized for environment: {self.environment}")
    
    def _get_default_config_root(self) -> Path:
        """Get default configuration root directory"""
        return Path(__file__).parent.parent.parent.parent / "config"
    
    def _detect_environment(self, environment: Optional[str] = None) -> str:
        """
        Detect current environment with priority:
        1. Explicit parameter
        2. QEMLFLOW_ENV environment variable
        3. Environment-specific indicators
        4. Default to development
        """
        if environment:
            return environment
            
        # Check environment variable
        env_var = os.getenv("QEMLFLOW_ENV")
        if env_var:
            return env_var
            
        # Check environment indicators
        if os.getenv("CI"):
            return EnvironmentType.TESTING.value
        if os.getenv("PRODUCTION"):
            return EnvironmentType.PRODUCTION.value
        if os.getenv("STAGING"):
            return EnvironmentType.STAGING.value
            
        # Check for Docker environment
        if os.path.exists("/.dockerenv"):
            return EnvironmentType.PRODUCTION.value
            
        # Default to development
        return EnvironmentType.DEVELOPMENT.value
    
    def load_configuration(self, config_name: str = "qemlflow") -> Dict[str, Any]:
        """
        Load configuration with multi-source priority:
        1. Environment variables
        2. Local configuration files
        3. Environment-specific configuration
        4. Base configuration
        5. Default values
        """
        logger.info(f"Loading configuration '{config_name}' for environment '{self.environment}'")
        
        # Configuration sources in priority order
        config_sources = [
            self._load_base_config(config_name),
            self._load_environment_config(config_name),
            self._load_local_config(config_name),
            self._load_environment_variables(),
        ]
        
        # Merge configurations with priority
        merged_config = {}
        source_files = []
        
        for config, source_file in config_sources:
            if config:
                merged_config = self._deep_merge(merged_config, config)
                if source_file:
                    source_files.append(source_file)
        
        # Generate metadata
        self._metadata = ConfigurationMetadata(
            version=self._generate_version(),
            environment=self.environment,
            created_at=datetime.now(),
            checksum=self._calculate_checksum(merged_config),
            source_files=source_files
        )
        
        # Validate configuration
        if self._validation_schema:
            self._validate_configuration(merged_config)
        
        # Cache configuration
        self._config_cache[config_name] = merged_config
        
        logger.info(f"Configuration loaded successfully. Version: {self._metadata.version}")
        return merged_config
    
    def _load_base_config(self, config_name: str) -> tuple[Optional[Dict], Optional[str]]:
        """Load base configuration file"""
        config_file = self.config_root / f"{config_name}.yml"
        return self._load_yaml_file(config_file)
    
    def _load_environment_config(self, config_name: str) -> tuple[Optional[Dict], Optional[str]]:
        """Load environment-specific configuration"""
        config_file = self.config_root / self.environment / f"{config_name}.yml"
        if not config_file.exists():
            config_file = self.config_root / f"{config_name}-{self.environment}.yml"
        return self._load_yaml_file(config_file)
    
    def _load_local_config(self, config_name: str) -> tuple[Optional[Dict], Optional[str]]:
        """Load local configuration overrides"""
        config_file = self.config_root / f"{config_name}.local.yml"
        return self._load_yaml_file(config_file)
    
    def _load_environment_variables(self) -> tuple[Optional[Dict], Optional[str]]:
        """Load configuration from environment variables"""
        env_config = {}
        prefix = "QEMLFLOW_"
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower().replace("_", ".")
                
                # Try to parse as JSON for complex values
                try:
                    parsed_value = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    parsed_value = value
                
                # Set nested configuration keys
                self._set_nested_key(env_config, config_key, parsed_value)
        
        return (env_config, "environment_variables") if env_config else (None, None)
    
    def _load_yaml_file(self, file_path: Path) -> tuple[Optional[Dict], Optional[str]]:
        """Load YAML configuration file"""
        if not file_path.exists():
            return None, None
            
        try:
            with open(file_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.debug(f"Loaded configuration from: {file_path}")
            return config, str(file_path)
        except Exception as e:
            logger.warning(f"Failed to load configuration from {file_path}: {e}")
            return None, None
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries with override taking priority"""
        result = base.copy()
        
        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _set_nested_key(self, config: Dict, key_path: str, value: Any) -> None:
        """Set nested configuration key using dot notation"""
        keys = key_path.split(".")
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def _calculate_checksum(self, config: Dict) -> str:
        """Calculate configuration checksum for change detection"""
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def _generate_version(self) -> str:
        """Generate configuration version"""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"v{timestamp}-{self.environment}"
    
    def load_schema(self, schema_name: str = "qemlflow") -> Optional[Dict]:
        """Load configuration validation schema"""
        schema_file = self.schema_path / f"{schema_name}.schema.json"
        
        if not schema_file.exists():
            logger.warning(f"Schema file not found: {schema_file}")
            return None
        
        try:
            with open(schema_file, 'r') as f:
                schema = json.load(f)
            
            self._validation_schema = schema
            logger.info(f"Loaded validation schema: {schema_name}")
            return schema
        except Exception as e:
            logger.error(f"Failed to load schema {schema_file}: {e}")
            return None
    
    def _validate_configuration(self, config: Dict) -> None:
        """Validate configuration against schema"""
        if not self._validation_schema:
            logger.warning("No validation schema loaded")
            return
        
        try:
            jsonschema.validate(config, self._validation_schema)
            logger.info("Configuration validation passed")
        except jsonschema.ValidationError as e:
            logger.error(f"Configuration validation failed: {e.message}")
            raise ValueError(f"Invalid configuration: {e.message}")
    
    def render_template(
        self,
        template_name: str,
        context: Optional[Dict] = None
    ) -> str:
        """
        Render configuration template with context
        
        Args:
            template_name: Name of template file
            context: Template rendering context
            
        Returns:
            Rendered configuration as string
        """
        template_file = f"{template_name}.j2"
        
        try:
            template = self._template_env.get_template(template_file)
            
            # Build rendering context
            render_context = {
                "environment": self.environment,
                "config_root": str(self.config_root),
                "timestamp": datetime.now().isoformat(),
                **(context or {})
            }
            
            rendered = template.render(**render_context)
            logger.info(f"Template '{template_name}' rendered successfully")
            return rendered
            
        except Exception as e:
            logger.error(f"Failed to render template {template_name}: {e}")
            raise
    
    def save_rendered_config(
        self,
        template_name: str,
        output_file: Union[str, Path],
        context: Optional[Dict] = None
    ) -> None:
        """Render template and save to file"""
        rendered_config = self.render_template(template_name, context)
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(rendered_config)
        
        logger.info(f"Rendered configuration saved to: {output_path}")
    
    def get_config_metadata(self) -> Optional[ConfigurationMetadata]:
        """Get configuration metadata"""
        return self._metadata
    
    def get_config_value(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        if not self._config_cache:
            raise ValueError("No configuration loaded")
        
        config = list(self._config_cache.values())[0]  # Get first loaded config
        keys = key_path.split(".")
        
        current = config
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        
        return current
    
    def validate_environment(self) -> Dict[str, Any]:
        """Validate current environment configuration"""
        validation_results = {
            "environment": self.environment,
            "config_root_exists": self.config_root.exists(),
            "templates_available": self.templates_path.exists(),
            "schemas_available": self.schema_path.exists(),
            "environment_variables": {
                key: value for key, value in os.environ.items()
                if key.startswith("QEMLFLOW_")
            },
            "detected_indicators": {
                "ci": bool(os.getenv("CI")),
                "production": bool(os.getenv("PRODUCTION")),
                "staging": bool(os.getenv("STAGING")),
                "docker": os.path.exists("/.dockerenv")
            }
        }
        
        return validation_results


# Convenience functions for common operations
def load_config(
    config_name: str = "qemlflow",
    environment: Optional[str] = None
) -> Dict[str, Any]:
    """Convenience function to load configuration"""
    engine = ConfigurationEngine(environment=environment)
    return engine.load_configuration(config_name)


def get_config_value(key_path: str, default: Any = None) -> Any:
    """Convenience function to get configuration value"""
    engine = ConfigurationEngine()
    engine.load_configuration()
    return engine.get_config_value(key_path, default)


def validate_environment() -> Dict[str, Any]:
    """Convenience function to validate environment"""
    engine = ConfigurationEngine()
    return engine.validate_environment()
