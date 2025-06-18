from typing import Any

"""
Configuration management for QeMLflow using Pydantic for type safety.
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
        def __init__(self, **kwargs) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)

    def Field(default=None, **kwargs) -> Any:
        return default

    def validator(*args, **kwargs) -> Any:
        def decorator(func) -> Any:
            return func

        return decorator


@dataclass
class WandBConfig:
    """Configuration for Weights & Biases experiment tracking."""

    enabled: bool = True
    project_name: str = "qemlflow-experiments"
    entity: Optional[str] = None
    api_key: Optional[str] = "b4f102d87161194b68baa7395d5862aa3f93b2b7"
    auto_login: bool = True
    log_artifacts: bool = True
    log_models: bool = True
    log_code: bool = True
    save_checkpoints: bool = True


@dataclass
class QeMLflowPaths:
    """Standard paths for QeMLflow data and outputs."""

    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")
    cache_dir: Path = Path("data/cache")
    models_dir: Path = Path("models")
    logs_dir: Path = Path("logs")

    def ensure_exists(self) -> None:
        """Create directories if they don't exist."""
        for path in [
            self.data_dir,
            self.output_dir,
            self.cache_dir,
            self.models_dir,
            self.logs_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)


class QeMLflowConfig(BaseSettings):
    """
    Central configuration for QeMLflow with environment variable support.
    """

    # Student/User identification
    student_id: str = Field("student", env="QEMLFLOW_STUDENT_ID")

    # Paths
    data_dir: Path = Field(Path("data"), env="QEMLFLOW_DATA_DIR")
    output_dir: Path = Field(Path("outputs"), env="QEMLFLOW_OUTPUT_DIR")
    cache_dir: Path = Field(Path("data/cache"), env="QEMLFLOW_CACHE_DIR")
    models_dir: Path = Field(Path("models"), env="QEMLFLOW_MODELS_DIR")
    logs_dir: Path = Field(Path("logs"), env="QEMLFLOW_LOGS_DIR")

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        "INFO", env="QEMLFLOW_LOG_LEVEL"
    )
    log_to_file: bool = Field(True, env="QEMLFLOW_LOG_TO_FILE")

    # Performance
    max_workers: int = Field(4, env="QEMLFLOW_MAX_WORKERS")
    memory_limit: float = Field(8.0, env="QEMLFLOW_MEMORY_LIMIT")  # GB
    cache_enabled: bool = Field(True, env="QEMLFLOW_CACHE_ENABLED")

    # External Services
    chembl_base_url: str = Field(
        "https://www.ebi.ac.uk/chembl/api/data", env="QEMLFLOW_CHEMBL_URL"
    )

    # Model settings
    random_seed: int = Field(42, env="QEMLFLOW_RANDOM_SEED")
    default_cv_folds: int = Field(5, env="QEMLFLOW_CV_FOLDS")

    # WandB settings
    wandb_enabled: bool = Field(True, env="QEMLFLOW_WANDB_ENABLED")
    wandb_project_name: str = Field(
        "qemlflow-experiments", env="QEMLFLOW_WANDB_PROJECT"
    )
    wandb_entity: Optional[str] = Field(None, env="QEMLFLOW_WANDB_ENTITY")
    wandb_api_key: Optional[str] = Field(None, env="QEMLFLOW_WANDB_API_KEY")
    wandb_auto_login: bool = Field(True, env="QEMLFLOW_WANDB_AUTO_LOGIN")
    wandb_log_artifacts: bool = Field(True, env="QEMLFLOW_WANDB_LOG_ARTIFACTS")
    wandb_log_models: bool = Field(True, env="QEMLFLOW_WANDB_LOG_MODELS")
    wandb_log_code: bool = Field(True, env="QEMLFLOW_WANDB_LOG_CODE")
    wandb_save_checkpoints: bool = Field(True, env="QEMLFLOW_WANDB_SAVE_CHECKPOINTS")

    if PYDANTIC_AVAILABLE:

        class Config:
            env_file = ".env"
            env_file_encoding = "utf-8"
            case_sensitive = False

        @validator("data_dir", "output_dir", "cache_dir", "models_dir", "logs_dir")
        def path_validator(cls, v) -> Any:
            if isinstance(v, str):
                return Path(v)
            return v

        @validator("memory_limit")
        def memory_limit_validator(cls, v) -> Any:
            if v <= 0:
                raise ValueError("Memory limit must be positive")
            return v

        @validator("max_workers")
        def max_workers_validator(cls, v) -> Any:
            if v <= 0:
                raise ValueError("Max workers must be positive")
            return min(v, os.cpu_count() or 4)

    def get_paths(self) -> QeMLflowPaths:
        """Get structured paths object."""
        return QeMLflowPaths(
            data_dir=self.data_dir,
            output_dir=self.output_dir,
            cache_dir=self.cache_dir,
            models_dir=self.models_dir,
            logs_dir=self.logs_dir,
        )

    def ensure_directories(self) -> None:
        """Create all required directories."""
        self.get_paths().ensure_exists()


# Global configuration instance
_config: Optional[QeMLflowConfig] = None


def get_config() -> QeMLflowConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = QeMLflowConfig()
        _config.ensure_directories()
    return _config


def update_config(**kwargs) -> None:
    """Update the global configuration."""
    global _config
    _config = QeMLflowConfig(**kwargs)
    _config.ensure_directories()


def reset_config() -> None:
    """Reset the global configuration to defaults."""
    global _config
    _config = None
