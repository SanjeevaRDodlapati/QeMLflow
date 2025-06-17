"""
External Model Integration for ChemML
====================================

Integration wrapper for external GitHub models from research publications.
Provides unified interface for models from different repositories.

Key Features:
- Automatic dependency management
- Unified ChemML API compatibility
- Error handling and fallbacks
- Model versioning and caching
"""

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
import numpy as np
from pathlib import Path
try:
    import git

    HAS_GITPYTHON = True
except ImportError:
    HAS_GITPYTHON = False
    warnings.warn("GitPython not available. Install with: pip install GitPython")

# Import ChemML base classes
try:
    from ..core.models import BaseModel
except ImportError:
    from chemml.core.models import BaseModel


class ExternalModelWrapper(BaseModel):
    """
    Base wrapper class for external models from GitHub repositories.

    This class provides a unified interface for integrating models from
    research publications and external repositories into ChemML.
    """

    def __init__(
        self,
        repo_url: str,
        model_class_name: str,
        repo_path: Optional[str] = None,
        branch: str = "main",
        install_requirements: bool = True,
        **model_kwargs,
    ):
        """
        Initialize external model wrapper.

        Args:
            repo_url: GitHub repository URL
            model_class_name: Name of the model class to import
            repo_path: Local path to clone repository (auto-generated if None)
            branch: Git branch to use
            install_requirements: Whether to install requirements.txt
            **model_kwargs: Arguments to pass to the external model
        """
        super().__init__(**model_kwargs)

        self.repo_url = repo_url
        self.model_class_name = model_class_name
        self.branch = branch
        self.install_requirements = install_requirements
        self.model_kwargs = model_kwargs

        # Setup repository
        if repo_path is None:
            self.repo_path = self._setup_temp_repo()
        else:
            self.repo_path = Path(repo_path)

        # Clone and setup external model
        self._clone_repository()
        self._setup_dependencies()
        self._import_external_model()

    def _setup_temp_repo(self) -> Path:
        """Create temporary directory for repository."""
        temp_dir = tempfile.mkdtemp(prefix="chemml_external_")
        repo_name = self.repo_url.split("/")[-1].replace(".git", "")
        return Path(temp_dir) / repo_name

    def _clone_repository(self):
        """Clone the external repository."""
        if not HAS_GITPYTHON:
            raise ImportError("GitPython required for automatic repository cloning")

        if self.repo_path.exists():
            print(f"Repository already exists at {self.repo_path}")
            return

        print(f"Cloning {self.repo_url} to {self.repo_path}")
        try:
            git.Repo.clone_from(self.repo_url, self.repo_path, branch=self.branch)
            print("✅ Repository cloned successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to clone repository: {e}")

    def _setup_dependencies(self):
        """Install repository requirements."""
        if not self.install_requirements:
            return

        requirements_file = self.repo_path / "requirements.txt"
        if requirements_file.exists():
            print("Installing repository requirements...")
            try:
                subprocess.check_call(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "-r",
                        str(requirements_file),
                    ]
                )
                print("✅ Requirements installed")
            except subprocess.CalledProcessError as e:
                warnings.warn(f"Failed to install requirements: {e}")

        # Also try setup.py if available
        setup_file = self.repo_path / "setup.py"
        if setup_file.exists():
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "-e", str(self.repo_path)]
                )
                print("✅ Package installed in development mode")
            except subprocess.CalledProcessError as e:
                warnings.warn(f"Failed to install package: {e}")

    def _import_external_model(self):
        """Import the external model class."""
        # Add repository to Python path
        if str(self.repo_path) not in sys.path:
            sys.path.insert(0, str(self.repo_path))

        # Try to find and import the model class
        self.external_model_class = None
        self.external_model = None

        # Common patterns for finding model classes
        search_patterns = [
            f"{self.model_class_name}",
            f"models.{self.model_class_name}",
            f"src.models.{self.model_class_name}",
            f"model.{self.model_class_name}",
        ]

        for pattern in search_patterns:
            try:
                module_parts = pattern.split(".")
                if len(module_parts) == 1:
                    # Try to find in main files
                    for main_file in ["main.py", "model.py", "models.py"]:
                        if (self.repo_path / main_file).exists():
                            spec = importlib.util.spec_from_file_location(
                                "external_module", self.repo_path / main_file
                            )
                            module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(module)
                            if hasattr(module, self.model_class_name):
                                self.external_model_class = getattr(
                                    module, self.model_class_name
                                )
                                break
                else:
                    # Try importing as module
                    module = importlib.import_module(".".join(module_parts[:-1]))
                    self.external_model_class = getattr(module, module_parts[-1])

                if self.external_model_class is not None:
                    print(f"✅ Found model class: {pattern}")
                    break

            except Exception as e:
                continue

        if self.external_model_class is None:
            raise ImportError(
                f"Could not find model class '{self.model_class_name}' in repository"
            )

        # Initialize the external model
        try:
            self.external_model = self.external_model_class(**self.model_kwargs)
            print(f"✅ External model initialized: {self.model_class_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize external model: {e}")

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, float]:
        """Fit the external model to training data."""
        if not hasattr(self.external_model, "fit") and not hasattr(
            self.external_model, "train"
        ):
            raise NotImplementedError("External model does not have fit/train method")

        try:
            # Try common training method names
            if hasattr(self.external_model, "fit"):
                result = self.external_model.fit(X, y, **kwargs)
            elif hasattr(self.external_model, "train"):
                result = self.external_model.train(X, y, **kwargs)
            else:
                raise AttributeError("No training method found")

            self.is_fitted = True

            # Try to extract metrics
            if isinstance(result, dict):
                return result
            else:
                return {"training_completed": True}

        except Exception as e:
            raise RuntimeError(f"Training failed: {e}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the external model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        try:
            # Try common prediction method names
            if hasattr(self.external_model, "predict"):
                return self.external_model.predict(X)
            elif hasattr(self.external_model, "forward"):
                return self.external_model.forward(X)
            elif hasattr(self.external_model, "__call__"):
                return self.external_model(X)
            else:
                raise AttributeError("No prediction method found")

        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")

    def cleanup(self):
        """Clean up temporary files."""
        if hasattr(self, "repo_path") and "temp" in str(self.repo_path):
            try:
                shutil.rmtree(self.repo_path.parent)
                print("✅ Temporary files cleaned up")
            except Exception as e:
                warnings.warn(f"Failed to cleanup: {e}")


class PublicationModelRegistry:
    """Registry for commonly used models from research publications."""

    KNOWN_MODELS = {
        # Example entries - replace with actual repositories
        "chemprop": {
            "repo_url": "https://github.com/chemprop/chemprop.git",
            "model_class": "MoleculeModel",
            "description": "Message Passing Neural Networks for Molecular Property Prediction",
        },
        "schnet": {
            "repo_url": "https://github.com/atomistic-machine-learning/schnetpack.git",
            "model_class": "SchNet",
            "description": "SchNet: A Deep Learning Architecture for Molecules and Materials",
        },
        "megnet": {
            "repo_url": "https://github.com/materialsvirtuallab/megnet.git",
            "model_class": "MEGNetModel",
            "description": "Graph Networks as a Universal Machine Learning Framework",
        },
        "dmpnn": {
            "repo_url": "https://github.com/swansonk14/chemprop.git",
            "model_class": "DMPNN",
            "description": "Analyzing Learned Molecular Representations for Property Prediction",
        },
    }

    @classmethod
    def get_model(cls, model_name: str, **kwargs) -> ExternalModelWrapper:
        """Get a pre-configured model from the registry."""
        if model_name not in cls.KNOWN_MODELS:
            raise ValueError(
                f"Model '{model_name}' not found in registry. Available: {list(cls.KNOWN_MODELS.keys())}"
            )

        model_info = cls.KNOWN_MODELS[model_name]
        return ExternalModelWrapper(
            repo_url=model_info["repo_url"],
            model_class_name=model_info["model_class"],
            **kwargs,
        )

    @classmethod
    def list_models(cls) -> Dict[str, str]:
        """List available models in the registry."""
        return {name: info["description"] for name, info in cls.KNOWN_MODELS.items()}


def integrate_external_model(
    repo_url: str, model_class_name: str, example_usage: Optional[str] = None, **kwargs
) -> ExternalModelWrapper:
    """
    Convenience function to integrate an external model.

    Args:
        repo_url: GitHub repository URL
        model_class_name: Name of the model class
        example_usage: Optional example of how to use the model
        **kwargs: Additional arguments for the wrapper

    Returns:
        Configured external model wrapper
    """
    wrapper = ExternalModelWrapper(
        repo_url=repo_url, model_class_name=model_class_name, **kwargs
    )

    if example_usage:
        print(f"📖 Example usage:\n{example_usage}")

    return wrapper


# Factory functions for common use cases
def create_publication_model(paper_repo_url: str, model_class: str, **kwargs):
    """Create model from research publication repository."""
    return integrate_external_model(paper_repo_url, model_class, **kwargs)


def create_huggingface_model(model_name: str, **kwargs):
    """Create model from Hugging Face repository."""
    repo_url = f"https://github.com/huggingface/{model_name}.git"
    return integrate_external_model(repo_url, "AutoModel", **kwargs)


# Example usage patterns
INTEGRATION_EXAMPLES = {
    "basic": """
# Basic integration
model = integrate_external_model(
    repo_url="https://github.com/author/paper-model.git",
    model_class_name="MyModel"
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
""",
    "with_config": """
# With custom configuration
model = integrate_external_model(
    repo_url="https://github.com/author/paper-model.git",
    model_class_name="MyModel",
    branch="experimental",
    hidden_dim=256,
    num_layers=4
)
""",
    "registry": """
# Using registry for known models
model = PublicationModelRegistry.get_model("chemprop")
model.fit(X_train, y_train)
predictions = model.predict(X_test)
""",
}


__all__ = [
    "ExternalModelWrapper",
    "PublicationModelRegistry",
    "integrate_external_model",
    "create_publication_model",
    "create_huggingface_model",
    "INTEGRATION_EXAMPLES",
]
