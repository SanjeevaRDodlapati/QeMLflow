"""
QeMLflow Intelligent Model Recommendation System
=============================================

AI-powered system that recommends optimal models based on data characteristics,
task requirements, and computational constraints.
"""

import warnings

import numpy as np
import pandas as pd


class ModelRecommendationEngine:
    """AI-powered model recommendation for molecular ML tasks."""

    def __init__(self) -> None:
        self.model_database = self._build_model_database()
        self.recommendation_history = []

    def recommend_best_model(
        self,
        molecules: Union[List[str], np.ndarray, pd.DataFrame],
        target_property: str,
        task_type: str = "auto",
        computational_budget: str = "medium",
        accuracy_priority: float = 0.7,
        speed_priority: float = 0.3,
    ) -> Dict[str, Any]:
        """
        Recommend optimal model based on data characteristics and requirements.

        Parameters:
        -----------
        molecular_data : array-like
            Molecular data (SMILES strings, descriptors, or features)
        target_property : str
            Property to predict (e.g., 'logP', 'toxicity', 'solubility')
        task_type : str
            'classification', 'regression', or 'auto' for auto-detection
        computational_budget : str
            'low', 'medium', 'high' - available computational resources
        accuracy_priority : float
            Weight for accuracy vs speed (0.0 to 1.0)
        speed_priority : float
            Weight for speed vs accuracy (0.0 to 1.0)

        Returns:
        --------
        Dict with recommended model, rationale, and configuration
        """
        data_profile = self._analyze_data_characteristics(molecules, target_property)
        if task_type == "auto":
            task_type = self._detect_task_type(molecules, target_property)
        candidates = self._get_model_candidates(data_profile, task_type)
        scored_models = self._score_models(
            candidates,
            data_profile,
            computational_budget,
            accuracy_priority,
            speed_priority,
        )
        best_model = scored_models[0]
        config = self._generate_model_config(best_model, data_profile)
        recommendation = {
            "recommended_model": best_model["name"],
            "model_type": best_model["type"],
            "confidence": best_model["score"],
            "rationale": self._generate_rationale(best_model, data_profile),
            "configuration": config,
            "alternatives": [model["name"] for model in scored_models[1:4]],
            "data_profile": data_profile,
            "expected_performance": self._estimate_performance(
                best_model, data_profile
            ),
        }
        self.recommendation_history.append(recommendation)
        return recommendation

    def _analyze_data_characteristics(
        self, molecules: Any, target_property: str
    ) -> Dict[str, Any]:
        """Analyze molecular data to extract characteristics."""
        profile = {
            "dataset_size": 0,
            "feature_dimension": 0,
            "has_missing_values": False,
            "data_type": "unknown",
            "molecular_complexity": "medium",
            "target_property": target_property,
            "property_category": self._categorize_property(target_property),
            "has_graph_structure": False,
            "is_sequence_data": False,
            "sparsity": 0.0,
            "class_balance": None,
        }
        try:
            if isinstance(molecules, list):
                profile["dataset_size"] = len(molecules)
                profile["data_type"] = (
                    "smiles"
                    if all(isinstance(x, str) for x in molecules[:10])
                    else "mixed"
                )
                profile["is_sequence_data"] = profile["data_type"] == "smiles"
                if profile["data_type"] == "smiles":
                    profile["has_graph_structure"] = True
                    profile["molecular_complexity"] = self._assess_molecular_complexity(
                        molecules[:100]
                    )
            elif isinstance(molecules, np.ndarray):
                profile["dataset_size"] = len(molecules)
                profile["feature_dimension"] = (
                    molecules.shape[1] if len(molecules.shape) > 1 else 1
                )
                profile["has_missing_values"] = np.isnan(molecules).any()
                profile["sparsity"] = (
                    np.mean(molecules == 0)
                    if molecules.dtype in [np.float32, np.float64]
                    else 0
                )
                profile["data_type"] = "features"
            elif isinstance(molecules, pd.DataFrame):
                profile["dataset_size"] = len(molecules)
                profile["feature_dimension"] = len(molecules.columns)
                profile["has_missing_values"] = molecules.isnull().any().any()
                profile["data_type"] = "dataframe"
                smiles_cols = [
                    col for col in molecules.columns if "smiles" in col.lower()
                ]
                if smiles_cols:
                    profile["is_sequence_data"] = True
                    profile["has_graph_structure"] = True
        except Exception as e:
            warnings.warn(f"Could not fully analyze data: {e}")
        return profile

    def _categorize_property(self, target_property: str) -> str:
        """Categorize the target property type."""
        property_lower = target_property.lower()
        if any(
            term in property_lower
            for term in [
                "admet",
                "absorption",
                "distribution",
                "metabolism",
                "excretion",
                "toxicity",
            ]
        ):
            return "admet"
        if any(
            term in property_lower
            for term in ["logp", "solubility", "mw", "molecular_weight", "pka", "polar"]
        ):
            return "physicochemical"
        if any(
            term in property_lower
            for term in ["activity", "ic50", "ki", "binding", "inhibition", "potency"]
        ):
            return "biological_activity"
        if any(
            term in property_lower
            for term in [
                "homo",
                "lumo",
                "bandgap",
                "dipole",
                "polarizability",
                "energy",
            ]
        ):
            return "quantum"
        if any(
            term in property_lower
            for term in ["toxic", "mutagenic", "carcinogenic", "ames"]
        ):
            return "toxicity"
        return "general"

    def _assess_molecular_complexity(self, smiles_list: List[str]) -> str:
        """Assess average molecular complexity from SMILES."""
        try:
            avg_length = np.mean(
                [len(smiles) for smiles in smiles_list[:50] if isinstance(smiles, str)]
            )
            ring_indicators = np.mean(
                [
                    (smiles.count("c") + smiles.count("C"))
                    for smiles in smiles_list[:50]
                    if isinstance(smiles, str)
                ]
            )
            if avg_length > 50 or ring_indicators > 20:
                return "high"
            elif avg_length > 25 or ring_indicators > 10:
                return "medium"
            else:
                return "low"
        except Exception:
            return "medium"

    def _detect_task_type(self, molecules: Any, target_property: str) -> str:
        """Auto-detect whether task is classification or regression."""
        property_lower = target_property.lower()
        if any(
            term in property_lower
            for term in ["toxic", "active", "mutagenic", "class", "category", "binary"]
        ):
            return "classification"
        if any(
            term in property_lower
            for term in ["logp", "ic50", "ki", "solubility", "mw", "energy", "value"]
        ):
            return "regression"
        return "regression"

    def _build_model_database(self) -> List[Dict[str, Any]]:
        """Build database of available models with their characteristics."""
        return [
            {
                "name": "Random Forest",
                "type": "ensemble",
                "good_for": ["small_datasets", "interpretability", "mixed_features"],
                "computational_cost": "low",
                "accuracy_tier": "medium",
                "training_speed": "fast",
                "prediction_speed": "fast",
                "handles_missing": True,
                "handles_categorical": True,
                "min_dataset_size": 10,
                "max_dataset_size": 100000,
                "suitable_properties": ["physicochemical", "general", "admet"],
            },
            {
                "name": "Gradient Boosting (XGBoost)",
                "type": "ensemble",
                "good_for": ["medium_datasets", "high_accuracy", "structured_data"],
                "computational_cost": "medium",
                "accuracy_tier": "high",
                "training_speed": "medium",
                "prediction_speed": "fast",
                "handles_missing": True,
                "handles_categorical": True,
                "min_dataset_size": 100,
                "max_dataset_size": 500000,
                "suitable_properties": [
                    "physicochemical",
                    "biological_activity",
                    "admet",
                ],
            },
            {
                "name": "Graph Neural Network (GCN)",
                "type": "deep_learning",
                "good_for": ["molecular_graphs", "large_datasets", "high_accuracy"],
                "computational_cost": "high",
                "accuracy_tier": "very_high",
                "training_speed": "slow",
                "prediction_speed": "medium",
                "handles_missing": False,
                "handles_categorical": False,
                "min_dataset_size": 1000,
                "max_dataset_size": 1000000,
                "suitable_properties": [
                    "quantum",
                    "biological_activity",
                    "physicochemical",
                ],
                "requires_graph": True,
            },
            {
                "name": "Graph Attention Network (GAT)",
                "type": "deep_learning",
                "good_for": [
                    "complex_molecules",
                    "attention_mechanisms",
                    "interpretability",
                ],
                "computational_cost": "high",
                "accuracy_tier": "very_high",
                "training_speed": "slow",
                "prediction_speed": "medium",
                "handles_missing": False,
                "handles_categorical": False,
                "min_dataset_size": 1000,
                "max_dataset_size": 1000000,
                "suitable_properties": ["quantum", "biological_activity"],
                "requires_graph": True,
            },
            {
                "name": "Molecular Transformer",
                "type": "deep_learning",
                "good_for": ["sequence_data", "smiles", "transfer_learning"],
                "computational_cost": "high",
                "accuracy_tier": "very_high",
                "training_speed": "slow",
                "prediction_speed": "medium",
                "handles_missing": False,
                "handles_categorical": False,
                "min_dataset_size": 500,
                "max_dataset_size": 1000000,
                "suitable_properties": ["general", "physicochemical", "toxicity"],
                "requires_sequence": True,
            },
            {
                "name": "Support Vector Machine",
                "type": "classical",
                "good_for": ["small_datasets", "high_dimensional", "robust"],
                "computational_cost": "medium",
                "accuracy_tier": "medium",
                "training_speed": "medium",
                "prediction_speed": "fast",
                "handles_missing": False,
                "handles_categorical": False,
                "min_dataset_size": 50,
                "max_dataset_size": 10000,
                "suitable_properties": ["general", "physicochemical"],
            },
            {
                "name": "Neural Network (MLP)",
                "type": "deep_learning",
                "good_for": ["medium_datasets", "feature_learning", "non_linear"],
                "computational_cost": "medium",
                "accuracy_tier": "high",
                "training_speed": "medium",
                "prediction_speed": "fast",
                "handles_missing": False,
                "handles_categorical": False,
                "min_dataset_size": 200,
                "max_dataset_size": 100000,
                "suitable_properties": ["general", "admet", "physicochemical"],
            },
            {
                "name": "Linear Regression",
                "type": "classical",
                "good_for": ["interpretability", "baseline", "linear_relationships"],
                "computational_cost": "very_low",
                "accuracy_tier": "low",
                "training_speed": "very_fast",
                "prediction_speed": "very_fast",
                "handles_missing": False,
                "handles_categorical": False,
                "min_dataset_size": 10,
                "max_dataset_size": 1000000,
                "suitable_properties": ["general"],
            },
        ]

    def _get_model_candidates(
        self, data_profile: Dict[str, Any], task_type: str
    ) -> List[Dict[str, Any]]:
        """Get candidate models based on data characteristics."""
        candidates = []
        for model in self.model_database:
            if (
                not model["min_dataset_size"]
                <= data_profile["dataset_size"]
                <= model["max_dataset_size"]
            ):
                continue
            if (
                model.get("requires_graph", False)
                and not data_profile["has_graph_structure"]
            ):
                continue
            if (
                model.get("requires_sequence", False)
                and not data_profile["is_sequence_data"]
            ):
                continue
            if (
                data_profile["property_category"] in model["suitable_properties"]
                or "general" in model["suitable_properties"]
            ):
                candidates.append(model)
        if not candidates:
            general_models = [
                m for m in self.model_database if "general" in m["suitable_properties"]
            ]
            candidates = sorted(general_models, key=lambda x: x["min_dataset_size"])[:3]
        return candidates

    def _score_models(
        self,
        candidates: List[Dict[str, Any]],
        data_profile: Dict[str, Any],
        computational_budget: str,
        accuracy_priority: float,
        speed_priority: float,
    ) -> List[Dict[str, Any]]:
        """Score and rank model candidates."""
        budget_scores = {
            "low": {"very_low": 1.0, "low": 0.8, "medium": 0.3, "high": 0.1},
            "medium": {"very_low": 0.8, "low": 1.0, "medium": 1.0, "high": 0.6},
            "high": {"very_low": 0.6, "low": 0.8, "medium": 1.0, "high": 1.0},
        }
        accuracy_scores = {"low": 0.3, "medium": 0.6, "high": 0.8, "very_high": 1.0}
        speed_scores = {
            "very_slow": 0.2,
            "slow": 0.4,
            "medium": 0.6,
            "fast": 0.8,
            "very_fast": 1.0,
        }
        scored_candidates = []
        for model in candidates:
            accuracy_score = accuracy_scores.get(model["accuracy_tier"], 0.5)
            speed_score = speed_scores.get(model["training_speed"], 0.5)
            budget_score = budget_scores[computational_budget].get(
                model["computational_cost"], 0.5
            )
            dataset_size = data_profile["dataset_size"]
            size_bonus = 1.0
            if dataset_size < 1000 and model["type"] == "classical":
                size_bonus = 1.2
            elif dataset_size > 10000 and model["type"] == "deep_learning":
                size_bonus = 1.2
            property_bonus = 1.0
            if data_profile["property_category"] in model["suitable_properties"]:
                property_bonus = 1.1
            final_score = (
                (
                    accuracy_priority * accuracy_score
                    + speed_priority * speed_score
                    + 0.3 * budget_score
                )
                * size_bonus
                * property_bonus
            )
            scored_candidate = model.copy()
            scored_candidate["score"] = final_score
            scored_candidates.append(scored_candidate)
        return sorted(scored_candidates, key=lambda x: x["score"], reverse=True)

    def _generate_model_config(
        self, model: Dict[str, Any], data_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate optimal configuration for the recommended model."""
        dataset_size = data_profile["dataset_size"]
        feature_dim = data_profile["feature_dimension"]
        if model["name"] == "Random Forest":
            return {
                "n_estimators": min(200, max(50, dataset_size // 10)),
                "max_depth": min(20, max(5, int(np.log2(dataset_size)))),
                "min_samples_split": max(2, dataset_size // 1000),
                "random_state": 42,
            }
        elif model["name"] == "Gradient Boosting (XGBoost)":
            return {
                "n_estimators": min(500, max(100, dataset_size // 5)),
                "max_depth": min(10, max(3, int(np.log2(dataset_size)) - 2)),
                "learning_rate": 0.1 if dataset_size < 10000 else 0.05,
                "random_state": 42,
            }
        elif "Neural Network" in model["name"]:
            hidden_size = min(512, max(64, feature_dim * 2))
            return {
                "hidden_layers": [hidden_size, hidden_size // 2],
                "batch_size": min(256, max(32, dataset_size // 100)),
                "learning_rate": 0.001,
                "epochs": 100 if dataset_size < 10000 else 50,
                "dropout": 0.2,
            }
        elif "Graph" in model["name"]:
            return {
                "hidden_dim": 128,
                "num_layers": 3,
                "num_heads": 8 if "Attention" in model["name"] else None,
                "dropout": 0.1,
                "batch_size": min(128, max(16, dataset_size // 200)),
                "learning_rate": 0.001,
                "epochs": 200,
            }
        else:
            return {"random_state": 42}

    def _generate_rationale(
        self, model: Dict[str, Any], data_profile: Dict[str, Any]
    ) -> str:
        """Generate human-readable rationale for the recommendation."""
        reasons = []
        size = data_profile["dataset_size"]
        if size < 1000:
            reasons.append(f"Small dataset ({size} samples) favors classical methods")
        elif size > 10000:
            reasons.append(f"Large dataset ({size} samples) can leverage deep learning")
        if data_profile["has_graph_structure"] and "Graph" in model["name"]:
            reasons.append(
                "Molecular graph structure detected - graph neural networks excel here"
            )
        if data_profile["is_sequence_data"] and "Transformer" in model["name"]:
            reasons.append("Sequence data (SMILES) detected - transformers are optimal")
        prop_cat = data_profile["property_category"]
        if prop_cat == "quantum" and model["accuracy_tier"] == "very_high":
            reasons.append("Quantum properties require high-accuracy models")
        reasons.append(
            f"Model offers {model['accuracy_tier']} accuracy with {model['training_speed']} training"
        )
        return " • ".join(reasons)

    def _estimate_performance(
        self, model: Dict[str, Any], data_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Estimate expected model performance."""
        base_accuracy = {"low": 0.6, "medium": 0.75, "high": 0.85, "very_high": 0.92}[
            model["accuracy_tier"]
        ]
        size_factor = min(1.1, 0.8 + data_profile["dataset_size"] / 10000 * 0.3)
        complexity_factor = {"low": 1.0, "medium": 0.95, "high": 0.9}[
            data_profile["molecular_complexity"]
        ]
        estimated_accuracy = base_accuracy * size_factor * complexity_factor
        time_estimates = {
            "very_fast": 1,
            "fast": 5,
            "medium": 30,
            "slow": 120,
            "very_slow": 600,
        }
        base_time = time_estimates[model["training_speed"]]
        size_time_factor = max(1.0, np.log10(data_profile["dataset_size"]) / 3)
        estimated_time = base_time * size_time_factor
        return {
            "estimated_accuracy": round(estimated_accuracy, 3),
            "confidence_interval": [
                round(estimated_accuracy - 0.05, 3),
                round(estimated_accuracy + 0.05, 3),
            ],
            "estimated_training_time_minutes": round(estimated_time, 1),
            "memory_requirements": self._estimate_memory_requirements(
                model, data_profile
            ),
        }

    def _estimate_memory_requirements(
        self, model: Dict[str, Any], data_profile: Dict[str, Any]
    ) -> str:
        """Estimate memory requirements."""
        size = data_profile["dataset_size"]
        if model["type"] == "classical":
            if size < 10000:
                return "Low (< 1GB)"
            else:
                return "Medium (1-4GB)"
        elif model["type"] == "deep_learning":
            if size < 5000:
                return "Medium (2-4GB)"
            elif size < 50000:
                return "High (4-8GB)"
            else:
                return "Very High (8GB+)"
        return "Medium (1-4GB)"

    def get_recommendation_history(self) -> List[Dict[str, Any]]:
        """Get history of all recommendations made."""
        return self.recommendation_history

    def compare_models(
        self,
        molecules: Any,
        target_property: str,
        models_to_compare: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Compare multiple models for the same dataset."""
        if models_to_compare is None:
            recommendation = self.recommend_best_model(molecules, target_property)
            models_to_compare = [recommendation["recommended_model"]] + recommendation[
                "alternatives"
            ][:2]
        comparisons = {}
        for model_name in models_to_compare:
            model_info = next(
                (m for m in self.model_database if m["name"] == model_name), None
            )
            if model_info:
                data_profile = self._analyze_data_characteristics(
                    molecules, target_property
                )
                config = self._generate_model_config(model_info, data_profile)
                performance = self._estimate_performance(model_info, data_profile)
                comparisons[model_name] = {
                    "configuration": config,
                    "estimated_performance": performance,
                    "pros": self._get_model_pros(model_info, data_profile),
                    "cons": self._get_model_cons(model_info, data_profile),
                }
        return {
            "comparison": comparisons,
            "recommendation": "Choose based on your priorities: accuracy vs speed vs interpretability",
        }

    def _get_model_pros(
        self, model: Dict[str, Any], data_profile: Dict[str, Any]
    ) -> List[str]:
        """Get advantages of a model for the given data."""
        pros = []
        if model["accuracy_tier"] in ["high", "very_high"]:
            pros.append("High accuracy potential")
        if model["training_speed"] in ["fast", "very_fast"]:
            pros.append("Fast training")
        if model.get("handles_missing", False):
            pros.append("Handles missing values")
        if model["type"] == "classical":
            pros.append("Good interpretability")
        if model["type"] == "deep_learning" and data_profile["dataset_size"] > 10000:
            pros.append("Can learn complex patterns")
        return pros

    def _get_model_cons(
        self, model: Dict[str, Any], data_profile: Dict[str, Any]
    ) -> List[str]:
        """Get disadvantages of a model for the given data."""
        cons = []
        if model["computational_cost"] == "high":
            cons.append("High computational requirements")
        if model["training_speed"] in ["slow", "very_slow"]:
            cons.append("Slow training")
        if not model.get("handles_missing", True):
            cons.append("Requires clean data (no missing values)")
        if model["type"] == "deep_learning" and data_profile["dataset_size"] < 1000:
            cons.append("May overfit on small datasets")
        if model["accuracy_tier"] == "low":
            cons.append("Limited accuracy potential")
        return cons


def recommend_model(molecules, target_property, **kwargs) -> Any:
    """Quick model recommendation function."""
    engine = ModelRecommendationEngine()
    return engine.recommend_best_model(molecules, target_property, **kwargs)


def compare_models(molecules, target_property, models=None):
    """Quick model comparison function."""
    engine = ModelRecommendationEngine()
    return engine.compare_models(molecules, target_property, models)
