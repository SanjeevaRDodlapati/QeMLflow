"""
QeMLflow Intelligent Model Recommender
=====================================

AI-powered model selection based on data characteristics and task requirements.
Implements philosophy: "AI-Enhanced framework with intelligent model selection"
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelRecommendation:
    """Model recommendation with confidence and reasoning."""

    model_name: str
    confidence: float
    reasoning: str
    estimated_performance: float
    computational_cost: str
    recommended_params: Dict[str, Any]


class IntelligentModelRecommender:
    """
    AI-powered model recommendation system.

    Analyzes dataset characteristics and suggests optimal models
    based on QeMLflow's extensive model collection.
    """

    def __init__(self):
        self.model_performance_db = self._load_model_performance_database()
        self.dataset_analyzers = {
            "size": self._analyze_dataset_size,
            "dimensionality": self._analyze_dimensionality,
            "target_distribution": self._analyze_target_distribution,
            "feature_correlations": self._analyze_feature_correlations,
            "noise_level": self._estimate_noise_level,
        }

    def recommend_models(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task_type: str = "auto",
        computational_budget: str = "medium",
        top_k: int = 3,
    ) -> List[ModelRecommendation]:
        """
        Recommend optimal models based on data characteristics.

        Args:
            X: Feature matrix
            y: Target values
            task_type: "classification", "regression", or "auto"
            computational_budget: "low", "medium", "high"
            top_k: Number of recommendations to return

        Returns:
            List of model recommendations with reasoning
        """
        # Analyze dataset characteristics
        dataset_profile = self._analyze_dataset(X, y, task_type)

        # Score all available models
        model_scores = self._score_models(dataset_profile, computational_budget)

        # Generate top recommendations
        top_models = sorted(
            model_scores.items(), key=lambda x: x[1]["score"], reverse=True
        )[:top_k]

        recommendations = []
        for model_name, score_info in top_models:
            rec = ModelRecommendation(
                model_name=model_name,
                confidence=score_info["score"],
                reasoning=score_info["reasoning"],
                estimated_performance=score_info["estimated_performance"],
                computational_cost=score_info["computational_cost"],
                recommended_params=score_info["params"],
            )
            recommendations.append(rec)

        return recommendations

    def _analyze_dataset(
        self, X: np.ndarray, y: np.ndarray, task_type: str
    ) -> Dict[str, Any]:
        """Comprehensive dataset analysis."""
        n_samples, n_features = X.shape

        # Auto-detect task type if not specified
        if task_type == "auto":
            unique_targets = len(np.unique(y))
            task_type = "classification" if unique_targets < 10 else "regression"

        profile = {
            "task_type": task_type,
            "n_samples": n_samples,
            "n_features": n_features,
            "sample_to_feature_ratio": n_samples / n_features,
        }

        # Run all analyzers
        for analyzer_name, analyzer_func in self.dataset_analyzers.items():
            try:
                profile[analyzer_name] = analyzer_func(X, y)
            except Exception as e:
                profile[analyzer_name] = f"Analysis failed: {e}"

        return profile

    def _analyze_dataset_size(self, X: np.ndarray, y: np.ndarray) -> str:
        """Categorize dataset size."""
        n_samples = X.shape[0]
        if n_samples < 100:
            return "very_small"
        elif n_samples < 1000:
            return "small"
        elif n_samples < 10000:
            return "medium"
        elif n_samples < 100000:
            return "large"
        else:
            return "very_large"

    def _analyze_dimensionality(self, X: np.ndarray, y: np.ndarray) -> str:
        """Analyze feature dimensionality characteristics."""
        n_features = X.shape[1]
        if n_features < 10:
            return "low_dimensional"
        elif n_features < 100:
            return "medium_dimensional"
        elif n_features < 1000:
            return "high_dimensional"
        else:
            return "very_high_dimensional"

    def _analyze_target_distribution(
        self, X: np.ndarray, y: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze target variable distribution."""
        if len(np.unique(y)) < 10:  # Classification
            unique, counts = np.unique(y, return_counts=True)
            balance_ratio = min(counts) / max(counts)
            return {
                "type": "classification",
                "n_classes": len(unique),
                "balance_ratio": balance_ratio,
                "is_balanced": balance_ratio > 0.7,
            }
        else:  # Regression
            return {
                "type": "regression",
                "mean": float(np.mean(y)),
                "std": float(np.std(y)),
                "skewness": float(
                    np.abs(np.mean((y - np.mean(y)) ** 3) / np.std(y) ** 3)
                ),
                "is_normal": np.abs(np.mean((y - np.mean(y)) ** 3) / np.std(y) ** 3)
                < 1.0,
            }

    def _analyze_feature_correlations(
        self, X: np.ndarray, y: np.ndarray
    ) -> Dict[str, float]:
        """Analyze feature correlation patterns."""
        # Sample for efficiency on large datasets
        if X.shape[0] > 5000:
            indices = np.random.choice(X.shape[0], 5000, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X

        corr_matrix = np.corrcoef(X_sample.T)
        corr_matrix = np.nan_to_num(corr_matrix)  # Handle NaN values

        # Remove diagonal (self-correlations)
        np.fill_diagonal(corr_matrix, 0)

        high_corr_pairs = np.sum(np.abs(corr_matrix) > 0.8)
        total_pairs = corr_matrix.shape[0] * (corr_matrix.shape[0] - 1) / 2

        return {
            "max_correlation": float(np.max(np.abs(corr_matrix))),
            "mean_correlation": float(np.mean(np.abs(corr_matrix))),
            "high_correlation_ratio": high_corr_pairs / total_pairs
            if total_pairs > 0
            else 0,
        }

    def _estimate_noise_level(self, X: np.ndarray, y: np.ndarray) -> str:
        """Estimate dataset noise level."""
        try:
            # Use subset for efficiency
            if X.shape[0] > 1000:
                indices = np.random.choice(X.shape[0], 1000, replace=False)
                X_subset = X[indices]
            else:
                X_subset = X

            # For high-dimensional data, use clustering-based noise estimation
            if X_subset.shape[1] > 50:
                # Use PCA to reduce dimensionality for noise estimation
                from sklearn.decomposition import PCA

                pca = PCA(n_components=min(10, X_subset.shape[1]))
                X_reduced = pca.fit_transform(StandardScaler().fit_transform(X_subset))
            else:
                X_reduced = StandardScaler().fit_transform(X_subset)

            # Estimate noise using silhouette score
            if X_reduced.shape[0] > 10:
                from sklearn.cluster import KMeans

                n_clusters = min(3, X_reduced.shape[0] // 3)
                if n_clusters >= 2:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(X_reduced)
                    silhouette = silhouette_score(X_reduced, labels)

                    if silhouette > 0.7:
                        return "low_noise"
                    elif silhouette > 0.3:
                        return "medium_noise"
                    else:
                        return "high_noise"

            return "unknown_noise"
        except Exception:
            return "unknown_noise"

    def _load_model_performance_database(self) -> Dict[str, Dict]:
        """Load model performance database from QeMLflow's benchmarks."""
        return {
            "RandomForestModel": {
                "strengths": ["robust", "handles_missing", "feature_importance"],
                "weaknesses": ["memory_intensive", "slow_prediction"],
                "best_for": ["medium_datasets", "mixed_features", "interpretability"],
                "computational_cost": "medium",
                "typical_performance": 0.85,
            },
            "LinearModel": {
                "strengths": ["fast", "interpretable", "low_memory"],
                "weaknesses": ["assumes_linearity", "sensitive_to_outliers"],
                "best_for": ["small_datasets", "linear_relationships", "baseline"],
                "computational_cost": "low",
                "typical_performance": 0.75,
            },
            "SVMModel": {
                "strengths": ["kernel_trick", "robust_outliers", "high_dimensional"],
                "weaknesses": ["slow_large_datasets", "parameter_sensitive"],
                "best_for": ["small_to_medium", "high_dimensional", "non_linear"],
                "computational_cost": "medium",
                "typical_performance": 0.82,
            },
            "NeuralNetwork": {
                "strengths": ["flexible", "non_linear", "feature_learning"],
                "weaknesses": ["overfitting", "hyperparameter_sensitive", "black_box"],
                "best_for": [
                    "large_datasets",
                    "complex_patterns",
                    "representation_learning",
                ],
                "computational_cost": "high",
                "typical_performance": 0.88,
            },
            "EnsembleModel": {
                "strengths": ["robust", "high_performance", "reduces_overfitting"],
                "weaknesses": ["computational_expensive", "less_interpretable"],
                "best_for": [
                    "medium_to_large",
                    "high_performance_needed",
                    "diverse_features",
                ],
                "computational_cost": "high",
                "typical_performance": 0.90,
            },
        }

    def _score_models(
        self, dataset_profile: Dict[str, Any], computational_budget: str
    ) -> Dict[str, Dict]:
        """Score all models based on dataset characteristics."""
        model_scores = {}

        for model_name, model_info in self.model_performance_db.items():
            score = self._calculate_model_score(
                model_name, model_info, dataset_profile, computational_budget
            )
            model_scores[model_name] = score

        return model_scores

    def _calculate_model_score(
        self,
        model_name: str,
        model_info: Dict,
        dataset_profile: Dict[str, Any],
        computational_budget: str,
    ) -> Dict[str, Any]:
        """Calculate compatibility score for a specific model."""
        score = 0.5  # Base score
        reasoning_parts = []

        # Dataset size compatibility
        dataset_size = dataset_profile.get("size", "unknown")
        if dataset_size in model_info["best_for"]:
            score += 0.2
            reasoning_parts.append(f"Good for {dataset_size} datasets")

        # Computational budget compatibility
        model_cost = model_info["computational_cost"]
        budget_compatibility = {
            "low": {"low": 0.3, "medium": 0.1, "high": -0.2},
            "medium": {"low": 0.2, "medium": 0.3, "high": 0.1},
            "high": {"low": 0.1, "medium": 0.2, "high": 0.3},
        }
        score += budget_compatibility[computational_budget][model_cost]
        reasoning_parts.append(f"Computational cost: {model_cost}")

        # Special case adjustments
        if (
            dataset_profile.get("dimensionality") == "high_dimensional"
            and "high_dimensional" in model_info["best_for"]
        ):
            score += 0.15
            reasoning_parts.append("Optimized for high-dimensional data")

        if (
            dataset_profile.get("target_distribution", {}).get("is_balanced", True)
            is False
        ):
            if model_name in ["RandomForestModel", "EnsembleModel"]:
                score += 0.1
                reasoning_parts.append("Handles imbalanced data well")

        # Noise level considerations
        noise_level = dataset_profile.get("noise_level", "unknown")
        if noise_level == "high_noise" and "robust" in model_info["strengths"]:
            score += 0.1
            reasoning_parts.append("Robust to noise")

        # Generate recommended parameters
        recommended_params = self._generate_recommended_params(
            model_name, dataset_profile
        )

        return {
            "score": min(max(score, 0.0), 1.0),  # Clamp to [0, 1]
            "reasoning": "; ".join(reasoning_parts),
            "estimated_performance": model_info["typical_performance"] * score,
            "computational_cost": model_cost,
            "params": recommended_params,
        }

    def _generate_recommended_params(
        self, model_name: str, dataset_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate recommended parameters based on dataset characteristics."""
        n_samples = dataset_profile.get("n_samples", 1000)
        n_features = dataset_profile.get("n_features", 10)

        if model_name == "RandomForestModel":
            return {
                "n_estimators": min(200, max(50, n_samples // 10)),
                "max_depth": min(20, max(3, n_features // 2)),
                "min_samples_split": max(2, n_samples // 1000),
            }
        elif model_name == "LinearModel":
            return {"regularization": "ridge" if n_features > n_samples else "none"}
        elif model_name == "SVMModel":
            return {
                "kernel": "linear" if n_features > n_samples else "rbf",
                "C": 0.1 if n_samples < 1000 else 1.0,
            }
        elif model_name == "NeuralNetwork":
            return {
                "hidden_layers": [min(100, max(10, n_features * 2))],
                "dropout_rate": 0.2 if n_samples < 1000 else 0.1,
            }
        else:
            return {}


def quick_model_recommendation(
    X: np.ndarray, y: np.ndarray, task_type: str = "auto"
) -> None:
    """Quick model recommendation for interactive use."""
    recommender = IntelligentModelRecommender()
    recommendations = recommender.recommend_models(X, y, task_type)

    print("🤖 QeMLflow Intelligent Model Recommendations")
    print("=" * 50)

    for i, rec in enumerate(recommendations, 1):
        print(f"\n#{i}. {rec.model_name}")
        print(f"   Confidence: {rec.confidence:.1%}")
        print(f"   Reasoning: {rec.reasoning}")
        print(f"   Est. Performance: {rec.estimated_performance:.3f}")
        print(f"   Computational Cost: {rec.computational_cost}")
        print(f"   Recommended Params: {rec.recommended_params}")


if __name__ == "__main__":
    # Demo with synthetic data
    np.random.seed(42)
    X = np.random.randn(1000, 20)
    y = np.random.randn(1000)

    quick_model_recommendation(X, y)
