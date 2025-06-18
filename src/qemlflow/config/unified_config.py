"""
QeMLflow Unified Configuration System
==================================

Centralized configuration management for all QeMLflow components.
Provides environment-based settings, feature flags, and integration controls.
"""

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ExperimentTrackingConfig:
    """Configuration for experiment tracking."""

    enabled: bool = True
    default_project: str = "qemlflow-experiments"
    auto_log_metrics: bool = True
    auto_log_artifacts: bool = True
    tags: List[str] = field(default_factory=list)
    wandb_api_key: Optional[str] = None


@dataclass
class ModelConfig:
    """Configuration for model training and inference."""

    default_model_type: str = "random_forest"
    auto_feature_scaling: bool = True
    validation_split: float = 0.2
    random_state: int = 42
    n_jobs: int = -1
    memory_limit_gb: Optional[float] = None


@dataclass
class PreprocessingConfig:
    """Configuration for molecular preprocessing."""

    default_descriptor_type: str = "morgan"
    fingerprint_radius: int = 2
    fingerprint_bits: int = 2048
    include_3d_descriptors: bool = False
    normalize_features: bool = True
    handle_missing: str = "impute"


@dataclass
class VisualizationConfig:
    """Configuration for visualization components."""

    default_backend: str = "matplotlib"
    figure_size: List[int] = field(default_factory=lambda: [10, 6])
    dpi: int = 100
    style: str = "seaborn-v0_8"
    color_palette: str = "viridis"
    save_plots: bool = False
    plot_directory: str = "plots"


@dataclass
class QuantumConfig:
    """Configuration for quantum computing features."""

    enabled: bool = True
    default_backend: str = "qiskit_aer"
    max_qubits: int = 16
    optimization_level: int = 1
    shots: int = 1024


@dataclass
class NotebookConfig:
    """Configuration for notebook integration."""

    auto_setup: bool = True
    display_warnings: bool = False
    progress_tracking: bool = True
    auto_save_outputs: bool = True
    template_directory: str = "notebooks/templates"


@dataclass
class QeMLflowConfig:
    """Main QeMLflow configuration container."""

    environment: str = "development"
    debug_mode: bool = False
    log_level: str = "INFO"
    data_directory: str = "data"
    cache_directory: str = "cache"
    experiment_tracking: ExperimentTrackingConfig = field(
        default_factory=ExperimentTrackingConfig
    )
    models: ModelConfig = field(default_factory=ModelConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    quantum: QuantumConfig = field(default_factory=QuantumConfig)
    notebooks: NotebookConfig = field(default_factory=NotebookConfig)
    enable_gpu_acceleration: bool = True
    enable_distributed_computing: bool = False
    enable_auto_optimization: bool = True
    enable_telemetry: bool = True

    def __post_init__(self):
        """Post-initialization to apply environment variables."""
        self._apply_environment_overrides()
        self._validate_configuration()

    def _apply_environment_overrides(self):
        """Apply environment variable overrides."""
        self.environment = os.getenv("QEMLFLOW_ENV") or self.environment
        self.debug_mode = (
            os.getenv("QEMLFLOW_DEBUG", str(self.debug_mode)).lower() == "true"
        )
        self.log_level = os.getenv("QEMLFLOW_LOG_LEVEL") or self.log_level
        if os.getenv("WANDB_API_KEY"):
            self.experiment_tracking.wandb_api_key = os.getenv("WANDB_API_KEY")
        if os.getenv("QEMLFLOW_MODEL_TYPE"):
            model_type = os.getenv("QEMLFLOW_MODEL_TYPE")
            if model_type:
                self.models.default_model_type = model_type
        self.data_directory = os.getenv("QEMLFLOW_DATA_DIR") or self.data_directory
        self.cache_directory = os.getenv("QEMLFLOW_CACHE_DIR") or self.cache_directory
        self.enable_gpu_acceleration = (
            os.getenv("QEMLFLOW_GPU", str(self.enable_gpu_acceleration)).lower()
            == "true"
        )
        self.enable_distributed_computing = (
            os.getenv(
                "QEMLFLOW_DISTRIBUTED", str(self.enable_distributed_computing)
            ).lower()
            == "true"
        )

    def _validate_configuration(self):
        """Validate configuration values."""
        valid_environments = ["development", "testing", "production"]
        if self.environment not in valid_environments:
            raise ValueError(
                f"Invalid environment: {self.environment}. Must be one of {valid_environments}"
            )
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level not in valid_log_levels:
            raise ValueError(
                f"Invalid log level: {self.log_level}. Must be one of {valid_log_levels}"
            )
        if not 0 < self.models.validation_split < 1:
            raise ValueError("Validation split must be between 0 and 1")
        for directory in [
            self.data_directory,
            self.cache_directory,
            self.visualization.plot_directory,
        ]:
            Path(directory).mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_file(cls, config_path: str) -> "QeMLflowConfig":
        """Load configuration from YAML file."""
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "QeMLflowConfig":
        """Create configuration from dictionary."""
        exp_tracking_dict = config_dict.pop("experiment_tracking", {})
        models_dict = config_dict.pop("models", {})
        preprocessing_dict = config_dict.pop("preprocessing", {})
        visualization_dict = config_dict.pop("visualization", {})
        quantum_dict = config_dict.pop("quantum", {})
        notebooks_dict = config_dict.pop("notebooks", {})
        exp_tracking = ExperimentTrackingConfig(**exp_tracking_dict)
        models = ModelConfig(**models_dict)
        preprocessing = PreprocessingConfig(**preprocessing_dict)
        visualization = VisualizationConfig(**visualization_dict)
        quantum = QuantumConfig(**quantum_dict)
        notebooks = NotebookConfig(**notebooks_dict)
        return cls(
            experiment_tracking=exp_tracking,
            models=models,
            preprocessing=preprocessing,
            visualization=visualization,
            quantum=quantum,
            notebooks=notebooks,
            **config_dict,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        import dataclasses

        def convert_value(value):
            """Convert a value to YAML-safe format."""
            if isinstance(value, tuple):
                return list(value)
            elif dataclasses.is_dataclass(value) and not isinstance(value, type):
                return {
                    k: convert_value(v) for k, v in dataclasses.asdict(value).items()
                }
            return value

        base_dict = dataclasses.asdict(self)
        return {k: convert_value(v) for k, v in base_dict.items()}

    def save(self, config_path: str):
        """Save configuration to YAML file."""
        config_dict = self.to_dict()
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

    def get_environment_summary(self) -> Dict[str, Any]:
        """Get summary of current environment and configuration."""
        import platform
        import sys

        try:
            import qemlflow

            qemlflow_version = qemlflow.__version__
        except Exception:
            qemlflow_version = "unknown"
        return {
            "timestamp": datetime.now().isoformat(),
            "environment": self.environment,
            "debug_mode": self.debug_mode,
            "python_version": sys.version,
            "platform": platform.platform(),
            "qemlflow_version": qemlflow_version,
            "experiment_tracking_enabled": self.experiment_tracking.enabled,
            "gpu_acceleration_enabled": self.enable_gpu_acceleration,
            "quantum_enabled": self.quantum.enabled,
            "data_directory": os.path.abspath(self.data_directory),
            "cache_directory": os.path.abspath(self.cache_directory),
        }


_global_config: Optional[QeMLflowConfig] = None


def get_config() -> QeMLflowConfig:
    """Get the global QeMLflow configuration."""
    global _global_config
    if _global_config is None:
        _global_config = load_config()
    return _global_config


def set_config(config: QeMLflowConfig):
    """Set the global QeMLflow configuration."""
    global _global_config
    _global_config = config


def load_config() -> QeMLflowConfig:
    """Load configuration from default locations."""
    config_paths = [
        os.getenv("QEMLFLOW_CONFIG"),
        "qemlflow_config.yaml",
        "config/qemlflow_config.yaml",
        os.path.expanduser("~/.qemlflow/config.yaml"),
    ]
    for config_path in config_paths:
        if config_path and Path(config_path).exists():
            try:
                return QeMLflowConfig.from_file(config_path)
            except Exception as e:
                print(f"Warning: Failed to load config from {config_path}: {e}")
    return QeMLflowConfig()


def create_default_config_file(config_path: str = "config/qemlflow_config.yaml"):
    """Create a default configuration file."""
    config = QeMLflowConfig()
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    config.save(config_path)
    print(f"✅ Created default configuration at {config_path}")


class ConfigContext:
    """Context manager for temporary configuration changes."""

    def __init__(self, **overrides):
        self.overrides = overrides
        self.original_values = {}

    def __enter__(self):
        config = get_config()
        for key, value in self.overrides.items():
            self.original_values[key] = getattr(config, key, None)
            setattr(config, key, value)
        return config

    def __exit__(self, exc_type, exc_val, exc_tb):
        config = get_config()
        for key, original_value in self.original_values.items():
            setattr(config, key, original_value)


if __name__ == "__main__":
    config = get_config()
    print("Current QeMLflow Configuration:")
    print("=" * 40)
    summary = config.get_environment_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
    create_default_config_file()
