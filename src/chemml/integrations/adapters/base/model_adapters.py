"""
Model Adapters for External Repository Integration
=================================================

Specialized adapters for different types of external models commonly
found in research publications and GitHub repositories.
"""

import warnings
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ....integrations.core.external_models import ExternalModelWrapper


class TorchModelAdapter(ExternalModelWrapper):
    """Adapter specifically for PyTorch models from external repositories."""

    def __init__(self, device: str = "auto", **kwargs):
        """Initialize PyTorch model adapter."""
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        super().__init__(**kwargs)

        # Move model to device if it's a PyTorch model
        if hasattr(self.external_model, "to"):
            self.external_model = self.external_model.to(self.device)

    def _prepare_torch_data(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Convert numpy arrays to PyTorch tensors."""
        X_tensor = torch.FloatTensor(X).to(self.device)

        if y is not None:
            y_tensor = torch.FloatTensor(y).to(self.device)
            return X_tensor, y_tensor

        return X_tensor

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        **kwargs,
    ) -> Dict[str, float]:
        """Train PyTorch model with standard training loop."""

        X_tensor, y_tensor = self._prepare_torch_data(X, y)

        # Setup optimizer if model doesn't have one
        if not hasattr(self.external_model, "optimizer"):
            if hasattr(self.external_model, "parameters"):
                optimizer = torch.optim.Adam(
                    self.external_model.parameters(), lr=learning_rate
                )
            else:
                raise AttributeError("Cannot create optimizer - no parameters found")
        else:
            optimizer = self.external_model.optimizer

        # Setup loss function
        if self.task_type == "regression":
            criterion = nn.MSELoss()
        else:
            criterion = nn.CrossEntropyLoss()

        # Training loop
        self.external_model.train()
        train_losses = []

        for epoch in range(epochs):
            # Simple batch processing
            for i in range(0, len(X_tensor), batch_size):
                batch_X = X_tensor[i : i + batch_size]
                batch_y = y_tensor[i : i + batch_size]

                optimizer.zero_grad()

                # Forward pass
                if hasattr(self.external_model, "forward"):
                    outputs = self.external_model.forward(batch_X)
                else:
                    outputs = self.external_model(batch_X)

                loss = criterion(outputs.squeeze(), batch_y.squeeze())
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

        self.is_fitted = True
        return {"final_loss": train_losses[-1], "avg_loss": np.mean(train_losses)}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with PyTorch model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        X_tensor = self._prepare_torch_data(X)

        self.external_model.eval()
        with torch.no_grad():
            if hasattr(self.external_model, "forward"):
                outputs = self.external_model.forward(X_tensor)
            else:
                outputs = self.external_model(X_tensor)

            return outputs.cpu().numpy()


class SklearnModelAdapter(ExternalModelWrapper):
    """Adapter for scikit-learn style models from external repositories."""

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, float]:
        """Fit sklearn-style model."""
        try:
            # Try sklearn-style fit
            self.external_model.fit(X, y, **kwargs)
            self.is_fitted = True

            # Try to get score if available
            score = {}
            if hasattr(self.external_model, "score"):
                try:
                    score["train_score"] = self.external_model.score(X, y)
                except Exception:
                    pass

            return score

        except Exception as e:
            raise RuntimeError(f"Sklearn-style training failed: {e}")


class HuggingFaceModelAdapter(ExternalModelWrapper):
    """Adapter for Hugging Face transformers models."""

    def __init__(self, model_name: str, **kwargs):
        """Initialize Hugging Face model adapter."""
        self.model_name = model_name

        # Don't clone repository for HF models
        kwargs["repo_url"] = None
        super().__init__(**kwargs)

    def _import_external_model(self):
        """Import Hugging Face model."""
        try:
            from transformers import AutoModel, AutoTokenizer

            self.external_model = AutoModel.from_pretrained(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            print(f"✅ Loaded Hugging Face model: {self.model_name}")

        except ImportError:
            raise ImportError("transformers library required for Hugging Face models")
        except Exception as e:
            raise RuntimeError(f"Failed to load HF model: {e}")

    def _clone_repository(self):
        """Skip repository cloning for HF models."""
        pass

    def _setup_dependencies(self):
        """Install transformers if needed."""
        try:
            import transformers
        except ImportError:
            import subprocess
            import sys

            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "transformers"]
            )


class PaperReproductionAdapter(ExternalModelWrapper):
    """
    Specialized adapter for reproducing models from research papers.

    This adapter includes common patterns and workarounds for research code.
    """

    def __init__(
        self,
        paper_title: str,
        authors: List[str],
        config_file: Optional[str] = None,
        weights_url: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize paper reproduction adapter.

        Args:
            paper_title: Title of the research paper
            authors: List of paper authors
            config_file: Path to configuration file in repository
            weights_url: URL to download pre-trained weights
        """
        self.paper_title = paper_title
        self.authors = authors
        self.config_file = config_file
        self.weights_url = weights_url

        super().__init__(**kwargs)

        # Load configuration if provided
        if self.config_file:
            self._load_config()

        # Download weights if provided
        if self.weights_url:
            self._download_weights()

    def _load_config(self):
        """Load configuration from paper repository."""
        config_path = self.repo_path / self.config_file
        if config_path.exists():
            try:
                if config_path.suffix == ".json":
                    import json

                    with open(config_path) as f:
                        self.config = json.load(f)
                elif config_path.suffix in [".yml", ".yaml"]:
                    import yaml

                    with open(config_path) as f:
                        self.config = yaml.safe_load(f)
                else:
                    warnings.warn(f"Unknown config format: {config_path.suffix}")

                print(f"✅ Loaded configuration from {self.config_file}")

            except Exception as e:
                warnings.warn(f"Failed to load config: {e}")

    def _download_weights(self):
        """Download pre-trained weights."""
        try:
            import requests

            weights_path = self.repo_path / "pretrained_weights.pth"

            print(f"Downloading weights from {self.weights_url}")
            response = requests.get(self.weights_url)
            response.raise_for_status()

            with open(weights_path, "wb") as f:
                f.write(response.content)

            print("✅ Pre-trained weights downloaded")

            # Try to load weights into model
            if hasattr(self.external_model, "load_state_dict"):
                try:
                    state_dict = torch.load(weights_path, map_location="cpu")
                    self.external_model.load_state_dict(state_dict)
                    print("✅ Weights loaded into model")
                except Exception as e:
                    warnings.warn(f"Failed to load weights: {e}")

        except Exception as e:
            warnings.warn(f"Failed to download weights: {e}")


class ModelZooAdapter:
    """
    Adapter for accessing collections of models from research groups.

    Useful for repositories that contain multiple models or model families.
    """

    def __init__(self, repo_url: str, model_zoo_config: Dict[str, Any]):
        """
        Initialize model zoo adapter.

        Args:
            repo_url: Repository URL
            model_zoo_config: Configuration mapping model names to classes
        """
        self.repo_url = repo_url
        self.model_zoo_config = model_zoo_config
        self.available_models = list(model_zoo_config.keys())

    def get_model(self, model_name: str, **kwargs) -> ExternalModelWrapper:
        """Get a specific model from the zoo."""
        if model_name not in self.available_models:
            raise ValueError(
                f"Model '{model_name}' not available. "
                f"Available: {self.available_models}"
            )

        model_config = self.model_zoo_config[model_name]

        return ExternalModelWrapper(
            repo_url=self.repo_url,
            model_class_name=model_config["class_name"],
            **model_config.get("default_params", {}),
            **kwargs,
        )

    def list_models(self) -> List[str]:
        """List available models in the zoo."""
        return self.available_models


# Convenience functions for specific use cases
def integrate_pytorch_model(
    repo_url: str, model_class: str, **kwargs
) -> TorchModelAdapter:
    """Integrate a PyTorch model from external repository."""
    return TorchModelAdapter(repo_url=repo_url, model_class_name=model_class, **kwargs)


def integrate_sklearn_model(
    repo_url: str, model_class: str, **kwargs
) -> SklearnModelAdapter:
    """Integrate a scikit-learn style model from external repository."""
    return SklearnModelAdapter(
        repo_url=repo_url, model_class_name=model_class, **kwargs
    )


def integrate_huggingface_model(model_name: str, **kwargs) -> HuggingFaceModelAdapter:
    """Integrate a Hugging Face model."""
    return HuggingFaceModelAdapter(model_name=model_name, **kwargs)


def reproduce_paper_model(
    repo_url: str, model_class: str, paper_title: str, authors: List[str], **kwargs
) -> PaperReproductionAdapter:
    """Reproduce a model from a research paper."""
    return PaperReproductionAdapter(
        repo_url=repo_url,
        model_class_name=model_class,
        paper_title=paper_title,
        authors=authors,
        **kwargs,
    )


__all__ = [
    "TorchModelAdapter",
    "SklearnModelAdapter",
    "HuggingFaceModelAdapter",
    "PaperReproductionAdapter",
    "ModelZooAdapter",
    "integrate_pytorch_model",
    "integrate_sklearn_model",
    "integrate_huggingface_model",
    "reproduce_paper_model",
]
