"""
External Model Integration Manager
=================================

High-level interface for integrating external models from research papers
and GitHub repositories into QeMLflow workflows.

Features:
- One-line model integration
- Automatic adapter selection
- Common research patterns
- Dependency management
- Advanced registry and monitoring
"""

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

# Lazy import to break circular dependency


def _get_model_adapters():
    from ..adapters.base.model_adapters import (
        HuggingFaceModelAdapter,
        ModelZooAdapter,
        PaperReproductionAdapter,
        SklearnModelAdapter,
        TorchModelAdapter,
    )

    return {
        "HuggingFaceModelAdapter": HuggingFaceModelAdapter,
        "ModelZooAdapter": ModelZooAdapter,
        "PaperReproductionAdapter": PaperReproductionAdapter,
        "SklearnModelAdapter": SklearnModelAdapter,
        "TorchModelAdapter": TorchModelAdapter,
    }


# Import new advanced features
from .advanced_registry import get_advanced_registry
from .automated_testing import create_adapter_test_suite, quick_adapter_test
from .external_models import ExternalModelWrapper, PublicationModelRegistry
from .performance_monitoring import get_metrics, track_integration

try:
    from ..adapters.molecular.boltz_adapter import BoltzAdapter, BoltzModel

    HAS_BOLTZ_ADAPTER = True
except ImportError:
    HAS_BOLTZ_ADAPTER = False


class ExternalModelManager:
    """
    Central manager for integrating external models into QeMLflow.

    This class provides a simple interface for integrating models from
    research publications, GitHub repositories, and model hubs with
    advanced registry, monitoring, and testing capabilities.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the external model manager.

        Args:
            cache_dir: Directory to cache downloaded repositories
        """
        self.cache_dir = (
            Path(cache_dir)
            if cache_dir
            else Path.home() / ".qemlflow" / "external_models"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Registry of integrated models
        self.integrated_models = {}

        # Load cached model information
        self._load_cache_info()

        # Registry of specialized adapters (lazy loaded)
        adapters = _get_model_adapters()
        self.specialized_adapters = {
            "boltz": BoltzAdapter if HAS_BOLTZ_ADAPTER else None,
            "pytorch": adapters["TorchModelAdapter"],
            "scikit-learn": adapters["SklearnModelAdapter"],
            "huggingface": adapters["HuggingFaceModelAdapter"],
            "paper_reproduction": adapters["PaperReproductionAdapter"],
            "model_zoo": adapters["ModelZooAdapter"],
        }

        # Advanced features
        self.advanced_registry = get_advanced_registry()
        self.metrics = get_metrics()
        self.test_suite = create_adapter_test_suite()

    def _load_cache_info(self):
        """Load information about cached models."""
        cache_info_file = self.cache_dir / "cache_info.json"
        if cache_info_file.exists():
            try:
                with open(cache_info_file) as f:
                    self.cache_info = json.load(f)
            except Exception:
                self.cache_info = {}
        else:
            self.cache_info = {}

    def _save_cache_info(self):
        """Save cache information."""
        cache_info_file = self.cache_dir / "cache_info.json"
        try:
            with open(cache_info_file, "w") as f:
                json.dump(self.cache_info, f, indent=2)
        except Exception as e:
            warnings.warn(f"Failed to save cache info: {e}")

    @track_integration("github_model")
    def integrate_from_github(
        self,
        repo_url: str,
        model_class: str,
        model_name: Optional[str] = None,
        adapter_type: str = "auto",
        validate_adapter: bool = True,
        **kwargs,
    ) -> ExternalModelWrapper:
        """
        Integrate a model from a GitHub repository.

        Args:
            repo_url: GitHub repository URL
            model_class: Name of the model class to import
            model_name: Optional name to register the model
            adapter_type: Type of adapter ('auto', 'torch', 'sklearn', 'paper')
            validate_adapter: Whether to validate the adapter compatibility
            **kwargs: Additional arguments for the model

        Returns:
            Integrated external model wrapper
        """
        if model_name is None:
            model_name = f"{repo_url.split('/')[-1]}_{model_class}"

        # Select appropriate adapter
        if adapter_type == "auto":
            adapter_class = self._auto_select_adapter(repo_url, model_class)
        else:
            adapters = _get_model_adapters()
            adapter_map = {
                "torch": adapters["TorchModelAdapter"],
                "sklearn": adapters["SklearnModelAdapter"],
                "paper": adapters["PaperReproductionAdapter"],
                "basic": ExternalModelWrapper,
            }
            adapter_class = adapter_map.get(adapter_type, ExternalModelWrapper)

        # Use cached repository if available
        repo_path = self._get_cached_repo_path(repo_url)

        # Create model wrapper
        model = adapter_class(
            repo_url=repo_url,
            model_class_name=model_class,
            repo_path=str(repo_path) if repo_path.exists() else None,
            **kwargs,
        )

        # Register the model
        self.integrated_models[model_name] = model

        # Update cache info
        self.cache_info[model_name] = {
            "repo_url": repo_url,
            "model_class": model_class,
            "adapter_type": adapter_class.__name__,
            "cached_path": str(repo_path),
        }
        self._save_cache_info()

        print(f"✅ Successfully integrated model '{model_name}' from {repo_url}")
        return model

    def integrate_from_huggingface(self, model_name: str, **kwargs) -> Any:
        """
        Integrate a model from Hugging Face.

        Args:
            model_name: Hugging Face model identifier
            **kwargs: Additional arguments

        Returns:
            Hugging Face model adapter
        """
        adapters = _get_model_adapters()
        model = adapters["HuggingFaceModelAdapter"](model_name=model_name, **kwargs)

        # Register the model
        self.integrated_models[model_name] = model

        print(f"✅ Successfully integrated Hugging Face model '{model_name}'")
        return model

    def integrate_from_paper(
        self,
        repo_url: str,
        model_class: str,
        paper_title: str,
        authors: List[str],
        model_name: Optional[str] = None,
        **kwargs,
    ) -> Any:
        """
        Integrate a model from a research paper repository.

        Args:
            repo_url: Repository URL
            model_class: Model class name
            paper_title: Title of the research paper
            authors: List of authors
            model_name: Optional name for the model
            **kwargs: Additional arguments

        Returns:
            Paper reproduction adapter
        """
        if model_name is None:
            model_name = f"paper_{paper_title.replace(' ', '_').lower()}"

        repo_path = self._get_cached_repo_path(repo_url)

        adapters = _get_model_adapters()
        model = adapters["PaperReproductionAdapter"](
            repo_url=repo_url,
            model_class_name=model_class,
            paper_title=paper_title,
            authors=authors,
            repo_path=str(repo_path) if repo_path.exists() else None,
            **kwargs,
        )

        # Register the model
        self.integrated_models[model_name] = model

        print(f"✅ Successfully integrated paper model '{model_name}'")
        return model

    def integrate_from_registry(
        self, model_name: str, **kwargs
    ) -> ExternalModelWrapper:
        """
        Integrate a model from the built-in registry.

        Args:
            model_name: Name of the model in the registry
            **kwargs: Additional arguments

        Returns:
            External model wrapper
        """
        model = PublicationModelRegistry.get_model(model_name, **kwargs)

        # Register the model
        self.integrated_models[model_name] = model

        print(f"✅ Successfully integrated registry model '{model_name}'")
        return model

    def integrate_boltz(self, **kwargs) -> "BoltzModel":
        """
        Integrate Boltz biomolecular interaction model.

        Args:
            **kwargs: Additional configuration for Boltz

        Returns:
            Integrated Boltz model instance
        """
        if not HAS_BOLTZ_ADAPTER:
            raise ImportError("Boltz adapter not available. Please check installation.")

        print("Integrating Boltz biomolecular interaction model...")

        # Configure Boltz with QeMLflow settings
        boltz_config = {
            "cache_dir": str(self.cache_dir / "boltz"),
            "use_msa_server": kwargs.get("use_msa_server", True),
            "device": kwargs.get("device", "auto"),
            **kwargs,
        }

        # Create Boltz model instance
        boltz_model = BoltzModel(**boltz_config)

        # Register in integrated models
        model_key = f"boltz_{kwargs.get('model_version', 'latest')}"
        self.integrated_models[model_key] = boltz_model

        print(f"✓ Boltz model integrated successfully as '{model_key}'")

        return boltz_model

    def predict_protein_structure(
        self, sequence: str, method: str = "boltz", **kwargs
    ) -> Dict[str, Any]:
        """
        Predict protein structure using specified method.

        Args:
            sequence: Protein amino acid sequence
            method: Prediction method ('boltz', etc.)
            **kwargs: Additional parameters

        Returns:
            Structure prediction results
        """
        if method == "boltz":
            if not HAS_BOLTZ_ADAPTER:
                raise ImportError("Boltz adapter not available")

            boltz_model = self.integrate_boltz(**kwargs)
            return boltz_model.adapter.predict_structure(sequence, **kwargs)
        else:
            raise ValueError(f"Unknown structure prediction method: {method}")

    def predict_binding_affinity(
        self, protein_sequence: str, ligand_smiles: str, method: str = "boltz", **kwargs
    ) -> Dict[str, Any]:
        """
        Predict protein-ligand binding affinity.

        Args:
            protein_sequence: Protein amino acid sequence
            ligand_smiles: Ligand SMILES string
            method: Prediction method
            **kwargs: Additional parameters

        Returns:
            Affinity prediction results
        """
        if method == "boltz":
            if not HAS_BOLTZ_ADAPTER:
                raise ImportError("Boltz adapter not available")

            boltz_model = self.integrate_boltz(**kwargs)
            return boltz_model.adapter.predict_affinity_only(
                protein_sequence, ligand_smiles, **kwargs
            )
        else:
            raise ValueError(f"Unknown affinity prediction method: {method}")

    def _auto_select_adapter(
        self, repo_url: str, model_class: str
    ) -> Type[ExternalModelWrapper]:
        """Automatically select the best adapter type."""
        # Simple heuristics for adapter selection

        # Check for PyTorch indicators
        pytorch_indicators = [
            "torch",
            "pytorch",
            "neural",
            "deep",
            "transformer",
            "gnn",
        ]
        if any(indicator in repo_url.lower() for indicator in pytorch_indicators):
            adapters = _get_model_adapters()
            return adapters["TorchModelAdapter"]

        # Check for sklearn indicators
        sklearn_indicators = ["sklearn", "scikit", "ensemble", "forest", "svm"]
        if any(indicator in repo_url.lower() for indicator in sklearn_indicators):
            adapters = _get_model_adapters()
            return adapters["SklearnModelAdapter"]

        # Check for research paper indicators
        paper_indicators = [
            "paper",
            "reproduction",
            "replication",
            "neurips",
            "icml",
            "iclr",
        ]
        if any(indicator in repo_url.lower() for indicator in paper_indicators):
            adapters = _get_model_adapters()
            return adapters["PaperReproductionAdapter"]

        # Default to basic wrapper
        return ExternalModelWrapper

    def _get_cached_repo_path(self, repo_url: str) -> Path:
        """Get the cached path for a repository."""
        repo_name = repo_url.split("/")[-1].replace(".git", "")
        return self.cache_dir / repo_name

    def get_model(self, model_name: str) -> Optional[ExternalModelWrapper]:
        """Get a previously integrated model."""
        return self.integrated_models.get(model_name)

    def list_integrated_models(self) -> List[str]:
        """List all integrated models."""
        return list(self.integrated_models.keys())

    def list_registry_models(self) -> Dict[str, str]:
        """List models available in the registry."""
        return PublicationModelRegistry.list_models()

    def remove_model(self, model_name: str):
        """Remove an integrated model."""
        if model_name in self.integrated_models:
            model = self.integrated_models.pop(model_name)
            if hasattr(model, "cleanup"):
                model.cleanup()

            if model_name in self.cache_info:
                del self.cache_info[model_name]
                self._save_cache_info()

            print(f"✅ Removed model '{model_name}'")
        else:
            print(f"❌ Model '{model_name}' not found")

    def clear_cache(self):
        """Clear all cached repositories."""
        import shutil

        try:
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache_info = {}
            self._save_cache_info()
            print("✅ Cache cleared")
        except Exception as e:
            print(f"❌ Failed to clear cache: {e}")

    def integrate_from_github_advanced(
        self,
        repo_url: str,
        model_class: str,
        model_name: Optional[str] = None,
        adapter_type: str = "auto",
        validate_adapter: bool = True,
        **kwargs,
    ) -> ExternalModelWrapper:
        """
        Integrate a model from a GitHub repository with advanced features.

        Args:
            repo_url: GitHub repository URL
            model_class: Name of the model class
            model_name: Optional name for the model
            adapter_type: Type of adapter ("auto", "pytorch", "sklearn", etc.)
            validate_adapter: Whether to run automated tests
            **kwargs: Additional arguments for the adapter

        Returns:
            Configured external model wrapper
        """
        if model_name is None:
            model_name = f"github_{repo_url.split('/')[-1].replace('.git', '')}"

        # Check if model exists in advanced registry
        registry_metadata = self.advanced_registry.get_model_metadata(model_name)
        if registry_metadata:
            print(f"📋 Found model in registry: {registry_metadata.description}")

        # Auto-select adapter type
        if adapter_type == "auto":
            adapter_class = self._auto_select_adapter(repo_url, model_class)
        else:
            adapter_class = self.specialized_adapters.get(
                adapter_type, ExternalModelWrapper
            )

        print(f"🔧 Using adapter: {adapter_class.__name__}")

        # Get cached repository path
        repo_path = self._get_cached_repo_path(repo_url)

        # Create adapter with performance tracking
        try:
            with self.metrics.track_operation(model_name, "integration"):
                adapter = adapter_class(
                    repo_url=repo_url,
                    model_class_name=model_class,
                    repo_path=str(repo_path) if repo_path.exists() else None,
                    **kwargs,
                )

                # Validate adapter if requested
                if validate_adapter:
                    print("🧪 Running adapter validation...")
                    test_result = quick_adapter_test(type(adapter))
                    print(f"   {test_result}")

                # Register the model
                self.integrated_models[model_name] = adapter

                # Update usage statistics
                self.advanced_registry.update_usage_stats(model_name)

                print(f"✅ Successfully integrated GitHub model '{model_name}'")
                return adapter

        except Exception as e:
            print(f"❌ Integration failed: {e}")
            raise

    def get_model_recommendations(
        self,
        task_description: str,
        complexity: str = "moderate",
        gpu_available: bool = False,
        max_memory_gb: float = 8.0,
        max_runtime_minutes: float = 30.0,
    ) -> List[str]:
        """
        Get AI-powered model recommendations based on task requirements.

        Args:
            task_description: Description of the task
            complexity: Task complexity ("simple", "moderate", "complex")
            gpu_available: Whether GPU is available
            max_memory_gb: Maximum memory available
            max_runtime_minutes: Maximum acceptable runtime

        Returns:
            List of recommended model names
        """
        from .advanced_registry import TaskComplexity

        complexity_map = {
            "simple": TaskComplexity.SIMPLE,
            "moderate": TaskComplexity.MODERATE,
            "complex": TaskComplexity.COMPLEX,
        }

        recommendations = self.advanced_registry.suggest_models(
            task_type=task_description,
            complexity=complexity_map.get(complexity),
            gpu_available=gpu_available,
            max_memory_gb=max_memory_gb,
            max_runtime_minutes=max_runtime_minutes,
        )

        return [name for name, score in recommendations[:5]]  # Top 5 recommendations

    def get_workflow_suggestions(self, goal: str) -> List[List[str]]:
        """Get suggested model workflows for achieving a goal."""
        return self.advanced_registry.get_workflow_suggestions(goal)

    def generate_performance_report(self, days: int = 7) -> str:
        """Generate performance report for integrated models."""
        return self.metrics.generate_performance_report(days=days)

    def validate_all_adapters(self) -> Dict[str, str]:
        """Run automated tests on all integrated models."""
        results = {}
        for model_name, adapter in self.integrated_models.items():
            try:
                result = quick_adapter_test(type(adapter))
                results[model_name] = result
            except Exception as e:
                results[model_name] = f"Test failed: {e}"
        return results

    def search_registry(self, query: str) -> List[str]:
        """Search the model registry."""
        return self.advanced_registry.search_models(query)

    def get_model_info(self, model_name: str) -> str:
        """Get detailed information about a model."""
        return self.advanced_registry.generate_model_report(model_name)


# Global manager instance
# _global_manager = None


def get_manager() -> ExternalModelManager:
    """Get the global external model manager."""
    global _global_manager
    if _global_manager is None:
        _global_manager = ExternalModelManager()
    return _global_manager


# Convenience functions for easy integration
def integrate_github_model(repo_url: str, model_class: str, **kwargs):
    """Integrate a model from GitHub repository."""
    return get_manager().integrate_from_github(repo_url, model_class, **kwargs)


def integrate_paper_model(
    repo_url: str, model_class: str, paper_title: str, authors: List[str], **kwargs
):
    """Integrate a model from research paper repository."""
    return get_manager().integrate_from_paper(
        repo_url, model_class, paper_title, authors, **kwargs
    )


def integrate_huggingface_model(model_name: str, **kwargs):
    """Integrate a Hugging Face model."""
    return get_manager().integrate_from_huggingface(model_name, **kwargs)


def integrate_registry_model(model_name: str, **kwargs):
    """Integrate a model from the built-in registry."""
    return get_manager().integrate_from_registry(model_name, **kwargs)


# Example usage patterns
USAGE_EXAMPLES = {
    "basic_github": """
# Integrate a PyTorch model from GitHub
model = integrate_github_model(
    repo_url="https://github.com/author/molecular-model.git",
    model_class="MolecularGNN"
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
""",
    "paper_reproduction": """
# Reproduce a model from a research paper
model = integrate_paper_model(
    repo_url="https://github.com/paper-authors/neurips2024-model.git",
    model_class="NovelArchitecture",
    paper_title="Novel Graph Neural Network for Molecular Property Prediction",
    authors=["Smith, J.", "Doe, A."],
    config_file="config.yaml",
    weights_url="https://zenodo.org/record/123456/pretrained.pth"
)
""",
    "huggingface": """
# Use a Hugging Face model
model = integrate_huggingface_model("microsoft/DialoGPT-medium")
""",
    "registry": """
# Use a model from the built-in registry
model = integrate_registry_model("chemprop")
""",
    "manager_usage": """
# Using the manager directly
manager = get_manager()

# List available models
print("Registry models:", manager.list_registry_models())
print("Integrated models:", manager.list_integrated_models())

# Get a previously integrated model
model = manager.get_model("my_model")
""",
}


__all__ = [
    "ExternalModelManager",
    "get_manager",
    "integrate_github_model",
    "integrate_paper_model",
    "integrate_huggingface_model",
    "integrate_registry_model",
    "USAGE_EXAMPLES",
]
