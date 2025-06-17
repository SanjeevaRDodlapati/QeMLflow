"""
Advanced Model Registry with AI-Powered Recommendations
=====================================================

Enhanced registry system with categorization, compatibility checking,
and intelligent model recommendations for users.
"""

import json
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple


class ModelCategory(Enum):
    """Categories for organizing models."""

    STRUCTURE_PREDICTION = "structure_prediction"
    MOLECULAR_DOCKING = "molecular_docking"
    PROPERTY_PREDICTION = "property_prediction"
    DRUG_DISCOVERY = "drug_discovery"
    QUANTUM_CHEMISTRY = "quantum_chemistry"
    FEATURIZATION = "featurization"
    GENERATIVE = "generative"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    OTHER = "other"


class TaskComplexity(Enum):
    """Task complexity levels."""

    SIMPLE = "simple"  # Single prediction, standard inputs
    MODERATE = "moderate"  # Multiple steps, some configuration
    COMPLEX = "complex"  # Advanced configuration, expert knowledge needed


@dataclass
class ModelMetadata:
    """Enhanced metadata for registered models."""

    repo_url: str
    model_class: str
    description: str
    category: ModelCategory
    complexity: TaskComplexity
    requirements: List[str] = field(default_factory=list)
    gpu_required: bool = False
    memory_gb: float = 1.0
    typical_runtime_minutes: float = 1.0
    input_types: List[str] = field(default_factory=list)
    output_types: List[str] = field(default_factory=list)
    paper_title: Optional[str] = None
    paper_authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    citations: int = 0
    license: str = "Unknown"
    maintenance_status: str = "Active"  # Active, Deprecated, Archived
    compatibility_tags: Set[str] = field(default_factory=set)
    user_rating: float = 0.0
    usage_count: int = 0


class AdvancedModelRegistry:
    """
    Enhanced model registry with categorization, recommendations,
    and compatibility checking.
    """

    def __init__(self, registry_path: Optional[str] = None):
        """Initialize the advanced registry."""
        if registry_path:
            self.registry_path = Path(registry_path)
        else:
            self.registry_path = Path.home() / ".qemlflow" / "model_registry.json"

        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

        # Enhanced model database
        self.models: Dict[str, ModelMetadata] = {}
        self.categories: Dict[ModelCategory, List[str]] = {}
        self.compatibility_matrix: Dict[str, Set[str]] = {}
        self.popularity_scores: Dict[str, float] = {}

        # Load existing registry
        self._load_registry()

        # Initialize with enhanced known models
        self._initialize_enhanced_models()

    def _load_registry(self):
        """Load registry from disk."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path) as f:
                    data = json.load(f)

                # Reconstruct ModelMetadata objects
                for name, model_data in data.get("models", {}).items():
                    self.models[name] = ModelMetadata(
                        repo_url=model_data["repo_url"],
                        model_class=model_data["model_class"],
                        description=model_data["description"],
                        category=ModelCategory(model_data["category"]),
                        complexity=TaskComplexity(model_data["complexity"]),
                        requirements=model_data.get("requirements", []),
                        gpu_required=model_data.get("gpu_required", False),
                        memory_gb=model_data.get("memory_gb", 1.0),
                        typical_runtime_minutes=model_data.get(
                            "typical_runtime_minutes", 1.0
                        ),
                        input_types=model_data.get("input_types", []),
                        output_types=model_data.get("output_types", []),
                        paper_title=model_data.get("paper_title"),
                        paper_authors=model_data.get("paper_authors", []),
                        year=model_data.get("year"),
                        citations=model_data.get("citations", 0),
                        license=model_data.get("license", "Unknown"),
                        maintenance_status=model_data.get(
                            "maintenance_status", "Active"
                        ),
                        compatibility_tags=set(
                            model_data.get("compatibility_tags", [])
                        ),
                        user_rating=model_data.get("user_rating", 0.0),
                        usage_count=model_data.get("usage_count", 0),
                    )

                self.compatibility_matrix = {
                    k: set(v) if isinstance(v, list) else v
                    for k, v in data.get("compatibility_matrix", {}).items()
                }
                self.popularity_scores = data.get("popularity_scores", {})

            except Exception as e:
                warnings.warn(f"Failed to load registry: {e}")

    def _save_registry(self):
        """Save registry to disk."""
        try:
            # Convert to serializable format
            data = {
                "models": {},
                "compatibility_matrix": {
                    k: list(v) for k, v in self.compatibility_matrix.items()
                },
                "popularity_scores": self.popularity_scores,
            }

            for name, metadata in self.models.items():
                data["models"][name] = {
                    "repo_url": metadata.repo_url,
                    "model_class": metadata.model_class,
                    "description": metadata.description,
                    "category": metadata.category.value,
                    "complexity": metadata.complexity.value,
                    "requirements": metadata.requirements,
                    "gpu_required": metadata.gpu_required,
                    "memory_gb": metadata.memory_gb,
                    "typical_runtime_minutes": metadata.typical_runtime_minutes,
                    "input_types": metadata.input_types,
                    "output_types": metadata.output_types,
                    "paper_title": metadata.paper_title,
                    "paper_authors": metadata.paper_authors,
                    "year": metadata.year,
                    "citations": metadata.citations,
                    "license": metadata.license,
                    "maintenance_status": metadata.maintenance_status,
                    "compatibility_tags": list(metadata.compatibility_tags),
                    "user_rating": metadata.user_rating,
                    "usage_count": metadata.usage_count,
                }

            with open(self.registry_path, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            warnings.warn(f"Failed to save registry: {e}")

    def _initialize_enhanced_models(self):
        """Initialize registry with enhanced model metadata."""
        enhanced_models = {
            "boltz": ModelMetadata(
                repo_url="https://github.com/jwohlwend/boltz.git",
                model_class="BoltzModel",
                description="Biomolecular interaction prediction model approaching AlphaFold3 accuracy",
                category=ModelCategory.STRUCTURE_PREDICTION,
                complexity=TaskComplexity.MODERATE,
                requirements=["torch", "numpy", "biotite"],
                gpu_required=True,
                memory_gb=8.0,
                typical_runtime_minutes=5.0,
                input_types=["protein_sequence", "ligand_smiles", "yaml_config"],
                output_types=["pdb_structure", "affinity_score", "confidence_score"],
                paper_title="Boltz: Biomolecular Interaction Prediction",
                paper_authors=["Wohlwend, J.", "et al."],
                year=2024,
                citations=150,
                license="MIT",
                maintenance_status="Active",
                compatibility_tags={"pytorch", "gpu", "protein", "drug_discovery"},
                user_rating=4.5,
                usage_count=89,
            ),
            "chemprop": ModelMetadata(
                repo_url="https://github.com/chemprop/chemprop.git",
                model_class="MoleculeModel",
                description="Message Passing Neural Networks for Molecular Property Prediction",
                category=ModelCategory.PROPERTY_PREDICTION,
                complexity=TaskComplexity.SIMPLE,
                requirements=["torch", "rdkit", "scikit-learn"],
                gpu_required=False,
                memory_gb=2.0,
                typical_runtime_minutes=2.0,
                input_types=["smiles", "molecular_graph"],
                output_types=["property_prediction", "uncertainty"],
                paper_title="Analyzing Learned Molecular Representations for Property Prediction",
                paper_authors=["Yang, K.", "Swanson, K.", "Jin, W."],
                year=2019,
                citations=800,
                license="MIT",
                maintenance_status="Active",
                compatibility_tags={"pytorch", "molecules", "property_prediction"},
                user_rating=4.8,
                usage_count=234,
            ),
            "alphafold": ModelMetadata(
                repo_url="https://github.com/deepmind/alphafold.git",
                model_class="AlphaFold",
                description="Highly accurate protein structure prediction using AI",
                category=ModelCategory.STRUCTURE_PREDICTION,
                complexity=TaskComplexity.COMPLEX,
                requirements=["jax", "haiku", "tensorflow"],
                gpu_required=True,
                memory_gb=16.0,
                typical_runtime_minutes=30.0,
                input_types=["protein_sequence", "msa"],
                output_types=["pdb_structure", "confidence_scores"],
                paper_title="Highly accurate protein structure prediction with AlphaFold",
                paper_authors=["Jumper, J.", "Evans, R.", "Pritzel, A."],
                year=2021,
                citations=15000,
                license="Apache-2.0",
                maintenance_status="Active",
                compatibility_tags={"jax", "gpu", "protein", "structure"},
                user_rating=4.9,
                usage_count=456,
            ),
            "autodock_vina": ModelMetadata(
                repo_url="https://github.com/ccsb-scripps/AutoDock-Vina.git",
                model_class="AutoDockVina",
                description="Fast and accurate molecular docking software",
                category=ModelCategory.MOLECULAR_DOCKING,
                complexity=TaskComplexity.MODERATE,
                requirements=["openbabel", "pymol"],
                gpu_required=False,
                memory_gb=4.0,
                typical_runtime_minutes=10.0,
                input_types=["protein_pdb", "ligand_sdf", "search_space"],
                output_types=["docked_poses", "binding_affinity", "rmsd"],
                paper_title="AutoDock Vina: Improving the speed and accuracy of docking",
                paper_authors=["Trott, O.", "Olson, A.J."],
                year=2010,
                citations=12000,
                license="Apache-2.0",
                maintenance_status="Active",
                compatibility_tags={"docking", "drug_discovery", "protein"},
                user_rating=4.6,
                usage_count=567,
            ),
        }

        # Add models that don't exist yet
        for name, metadata in enhanced_models.items():
            if name not in self.models:
                self.models[name] = metadata

        # Update categories
        self._update_categories()

        # Initialize compatibility matrix
        self._initialize_compatibility()

        # Save updated registry
        self._save_registry()

    def _update_categories(self):
        """Update category mappings."""
        self.categories.clear()
        for name, metadata in self.models.items():
            if metadata.category not in self.categories:
                self.categories[metadata.category] = []
            self.categories[metadata.category].append(name)

    def _initialize_compatibility(self):
        """Initialize model compatibility matrix."""
        # Define compatibility rules
        compatibility_rules = {
            "boltz": {"alphafold", "autodock_vina"},  # Structure prediction + docking
            "alphafold": {
                "boltz",
                "autodock_vina",
            },  # Can provide structures for docking
            "chemprop": {"autodock_vina"},  # Property prediction + docking
            "autodock_vina": {
                "boltz",
                "alphafold",
                "chemprop",
            },  # Can use structures and properties
        }

        for model, compatible_models in compatibility_rules.items():
            if model not in self.compatibility_matrix:
                self.compatibility_matrix[model] = set()
            self.compatibility_matrix[model].update(compatible_models)

    def register_model(self, name: str, metadata: ModelMetadata):
        """Register a new model."""
        self.models[name] = metadata
        self._update_categories()
        self._save_registry()
        print(f"✅ Registered model '{name}'")

    def get_model_metadata(self, name: str) -> Optional[ModelMetadata]:
        """Get metadata for a specific model."""
        return self.models.get(name)

    def list_models_by_category(self, category: ModelCategory) -> List[str]:
        """List models in a specific category."""
        return self.categories.get(category, [])

    def suggest_models(
        self,
        task_type: str,
        complexity: Optional[TaskComplexity] = None,
        gpu_available: bool = False,
        max_memory_gb: float = 8.0,
        max_runtime_minutes: float = 30.0,
        input_type: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        """
        AI-powered model recommendations based on requirements.

        Args:
            task_type: Type of task (e.g., "structure_prediction", "docking")
            complexity: Maximum complexity level
            gpu_available: Whether GPU is available
            max_memory_gb: Maximum memory available
            max_runtime_minutes: Maximum acceptable runtime
            input_type: Type of input data

        Returns:
            List of (model_name, recommendation_score) tuples, sorted by score
        """
        recommendations = []

        for name, metadata in self.models.items():
            score = 0.0

            # Category matching
            if task_type.lower() in metadata.category.value.lower():
                score += 3.0
            elif any(tag in task_type.lower() for tag in metadata.compatibility_tags):
                score += 2.0

            # Complexity filter
            if complexity and metadata.complexity.value > complexity.value:
                continue  # Skip if too complex

            # Resource constraints
            if metadata.gpu_required and not gpu_available:
                continue  # Skip if GPU required but not available

            if metadata.memory_gb > max_memory_gb:
                continue  # Skip if memory requirement too high

            if metadata.typical_runtime_minutes > max_runtime_minutes:
                score -= 1.0  # Penalize longer runtime

            # Input type matching
            if input_type and input_type in metadata.input_types:
                score += 2.0

            # Popularity and quality factors
            score += metadata.user_rating / 5.0  # 0-1 range
            score += min(metadata.citations / 1000, 2.0)  # Up to 2 points for citations
            score += min(metadata.usage_count / 100, 1.0)  # Up to 1 point for usage

            # Maintenance status
            if metadata.maintenance_status == "Active":
                score += 1.0
            elif metadata.maintenance_status == "Deprecated":
                score -= 2.0

            if score > 0:
                recommendations.append((name, score))

        # Sort by recommendation score (descending)
        recommendations.sort(key=lambda x: x[1], reverse=True)

        return recommendations

    def check_compatibility(self, model_a: str, model_b: str) -> bool:
        """Check if two models are compatible for use together."""
        if model_a not in self.models or model_b not in self.models:
            return False

        # Check explicit compatibility matrix
        if model_a in self.compatibility_matrix:
            if model_b in self.compatibility_matrix[model_a]:
                return True

        # Check reverse compatibility
        if model_b in self.compatibility_matrix:
            if model_a in self.compatibility_matrix[model_b]:
                return True

        # Check tag-based compatibility
        tags_a = self.models[model_a].compatibility_tags
        tags_b = self.models[model_b].compatibility_tags

        # Compatible if they share common tags
        if tags_a.intersection(tags_b):
            return True

        return False

    def get_workflow_suggestions(self, goal: str) -> List[List[str]]:
        """
        Suggest multi-model workflows for achieving a goal.

        Args:
            goal: Description of the desired outcome

        Returns:
            List of workflow paths (each path is a list of model names)
        """
        goal_lower = goal.lower()
        workflows = []

        # Drug discovery workflow
        if any(
            term in goal_lower for term in ["drug", "discovery", "screening", "binding"]
        ):
            workflows.extend(
                [
                    ["boltz", "autodock_vina"],  # Structure + docking
                    ["alphafold", "autodock_vina"],  # Structure prediction + docking
                    ["chemprop", "autodock_vina"],  # Property + docking
                    ["alphafold", "boltz", "autodock_vina"],  # Full pipeline
                ]
            )

        # Structure analysis workflow
        if any(term in goal_lower for term in ["structure", "protein", "folding"]):
            workflows.extend(
                [
                    ["alphafold"],  # Single structure prediction
                    ["boltz"],  # Biomolecular interactions
                    ["alphafold", "boltz"],  # Structure + interactions
                ]
            )

        # Property prediction workflow
        if any(term in goal_lower for term in ["property", "prediction", "molecular"]):
            workflows.extend(
                [
                    ["chemprop"],  # Molecular properties
                    ["chemprop", "autodock_vina"],  # Properties + docking
                ]
            )

        # Filter workflows to ensure compatibility
        valid_workflows = []
        for workflow in workflows:
            valid = True
            for i in range(len(workflow) - 1):
                if not self.check_compatibility(workflow[i], workflow[i + 1]):
                    valid = False
                    break
            if valid:
                valid_workflows.append(workflow)

        return valid_workflows

    def update_usage_stats(self, model_name: str):
        """Update usage statistics for a model."""
        if model_name in self.models:
            self.models[model_name].usage_count += 1
            self._save_registry()

    def rate_model(self, model_name: str, rating: float, user_id: Optional[str] = None):
        """Add a user rating for a model."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")

        if not 0 <= rating <= 5:
            raise ValueError("Rating must be between 0 and 5")

        # Simple average for now (could be enhanced with user tracking)
        current_rating = self.models[model_name].user_rating
        current_count = self.models[model_name].usage_count

        # Weighted average
        new_rating = (current_rating * current_count + rating) / (current_count + 1)
        self.models[model_name].user_rating = new_rating

        self._save_registry()
        print(f"✅ Updated rating for '{model_name}': {new_rating:.2f}")

    def search_models(self, query: str) -> List[str]:
        """Search models by description, tags, or name."""
        query_lower = query.lower()
        matches = []

        for name, metadata in self.models.items():
            # Search in name
            if query_lower in name.lower():
                matches.append(name)
                continue

            # Search in description
            if query_lower in metadata.description.lower():
                matches.append(name)
                continue

            # Search in tags
            if any(query_lower in tag.lower() for tag in metadata.compatibility_tags):
                matches.append(name)
                continue

            # Search in paper title
            if metadata.paper_title and query_lower in metadata.paper_title.lower():
                matches.append(name)
                continue

        return matches

    def generate_model_report(self, model_name: str) -> str:
        """Generate a comprehensive report for a model."""
        if model_name not in self.models:
            return f"Model '{model_name}' not found"

        metadata = self.models[model_name]

        report = f"""
# Model Report: {model_name}

## Overview
- **Description**: {metadata.description}
- **Category**: {metadata.category.value.replace('_', ' ').title()}
- **Complexity**: {metadata.complexity.value.title()}
- **User Rating**: {metadata.user_rating:.1f}/5.0 ⭐
- **Usage Count**: {metadata.usage_count}

## Technical Details
- **Repository**: {metadata.repo_url}
- **Model Class**: {metadata.model_class}
- **GPU Required**: {'Yes' if metadata.gpu_required else 'No'}
- **Memory**: {metadata.memory_gb:.1f} GB
- **Typical Runtime**: {metadata.typical_runtime_minutes:.1f} minutes

## Publication Info
- **Paper**: {metadata.paper_title or 'Not specified'}
- **Authors**: {', '.join(metadata.paper_authors) if metadata.paper_authors else 'Not specified'}
- **Year**: {metadata.year or 'Not specified'}
- **Citations**: {metadata.citations}

## Compatibility
- **Input Types**: {', '.join(metadata.input_types) or 'Not specified'}
- **Output Types**: {', '.join(metadata.output_types) or 'Not specified'}
- **Tags**: {', '.join(metadata.compatibility_tags) if metadata.compatibility_tags else 'None'}

## Requirements
{chr(10).join(f'- {req}' for req in metadata.requirements) if metadata.requirements else '- None specified'}

## Status
- **License**: {metadata.license}
- **Maintenance**: {metadata.maintenance_status}
"""

        # Add compatible models
        compatible = [
            model
            for model in self.models.keys()
            if model != model_name and self.check_compatibility(model_name, model)
        ]
        if compatible:
            report += f"\n## Compatible Models\n{chr(10).join(f'- {model}' for model in compatible)}\n"

        return report.strip()


# Global instance
_advanced_registry = None


def get_advanced_registry() -> AdvancedModelRegistry:
    """Get the global advanced registry instance."""
    global _advanced_registry
    if _advanced_registry is None:
        _advanced_registry = AdvancedModelRegistry()
    return _advanced_registry
