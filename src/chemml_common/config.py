"""
Configuration management for ChemML using Pydantic for type safety.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

try:
    from pydantic import BaseSettings, Field, validator

    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

    # Create fallback classes
    class BaseSettings:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    def Field(default=None, **kwargs):
        return default

    def validator(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


@dataclass
class ChemMLPaths:
    """Standard paths for ChemML data and outputs."""

    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")
    cache_dir: Path = Path("data/cache")
    models_dir: Path = Path("models")
    logs_dir: Path = Path("logs")

    def ensure_exists(self):
        """Create directories if they don't exist."""
        for path in [
            self.data_dir,
            self.output_dir,
            self.cache_dir,
            self.models_dir,
            self.logs_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)


class ChemMLConfig(BaseSettings):
    """
    Central configuration for ChemML with environment variable support.
    """

    # Student/User identification
    student_id: str = Field("student", env="CHEMML_STUDENT_ID")

    # Paths
    data_dir: Path = Field(Path("data"), env="CHEMML_DATA_DIR")
    output_dir: Path = Field(Path("outputs"), env="CHEMML_OUTPUT_DIR")
    cache_dir: Path = Field(Path("data/cache"), env="CHEMML_CACHE_DIR")
    models_dir: Path = Field(Path("models"), env="CHEMML_MODELS_DIR")
    logs_dir: Path = Field(Path("logs"), env="CHEMML_LOGS_DIR")

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        "INFO", env="CHEMML_LOG_LEVEL"
    )
    log_to_file: bool = Field(True, env="CHEMML_LOG_TO_FILE")

    # Performance
    max_workers: int = Field(4, env="CHEMML_MAX_WORKERS")
    memory_limit: float = Field(8.0, env="CHEMML_MEMORY_LIMIT")  # GB
    cache_enabled: bool = Field(True, env="CHEMML_CACHE_ENABLED")

    # External Services
    chembl_base_url: str = Field(
        "https://www.ebi.ac.uk/chembl/api/data", env="CHEMML_CHEMBL_URL"
    )

    # Model settings
    random_seed: int = Field(42, env="CHEMML_RANDOM_SEED")
    default_cv_folds: int = Field(5, env="CHEMML_CV_FOLDS")

    if PYDANTIC_AVAILABLE:

        class Config:
            env_file = ".env"
            env_file_encoding = "utf-8"
            case_sensitive = False

        @validator("data_dir", "output_dir", "cache_dir", "models_dir", "logs_dir")
        def path_validator(cls, v):
            if isinstance(v, str):
                return Path(v)
            return v

        @validator("memory_limit")
        def memory_limit_validator(cls, v):
            if v <= 0:
                raise ValueError("Memory limit must be positive")
            return v

        @validator("max_workers")
        def max_workers_validator(cls, v):
            if v <= 0:
                raise ValueError("Max workers must be positive")
            return min(v, os.cpu_count() or 4)

    def get_paths(self) -> ChemMLPaths:
        """Get structured paths object."""
        return ChemMLPaths(
            data_dir=self.data_dir,
            output_dir=self.output_dir,
            cache_dir=self.cache_dir,
            models_dir=self.models_dir,
            logs_dir=self.logs_dir,
        )

    def ensure_directories(self):
        """Create all required directories."""
        self.get_paths().ensure_exists()


# Global configuration instance
_config: Optional[ChemMLConfig] = None


def get_config() -> ChemMLConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = ChemMLConfig()
        _config.ensure_directories()
    return _config


def update_config(**kwargs):
    """Update the global configuration."""
    global _config
    _config = ChemMLConfig(**kwargs)
    _config.ensure_directories()


def reset_config():
    """Reset the global configuration to defaults."""
    global _config
    _config = None
