"""
Base Model Adapters
==================

Base adapter classes for different types of external models.
These provide the foundation for specific model integrations.
"""

from .model_adapters import (
    HuggingFaceModelAdapter,
    ModelZooAdapter,
    PaperReproductionAdapter,
    SklearnModelAdapter,
    TorchModelAdapter,
    integrate_huggingface_model,
    integrate_pytorch_model,
    integrate_sklearn_model,
    reproduce_paper_model,
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
