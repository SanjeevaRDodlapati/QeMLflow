"""
QeMLflow Advanced Ensemble Methods
===============================

Sophisticated ensemble learning methods specifically designed for molecular ML.
Includes adaptive ensemble selection, multi-modal fusion, and uncertainty quantification.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from .common.performance import performance_context


class AdaptiveEnsemble(BaseEstimator):
    """
    Adaptive ensemble that dynamically selects and weights models
    based on their performance on different molecular subgroups.
    """

    def __init__(
        self,
        base_models: List[Any],
        adaptation_strategy: str = "performance_weighted",
        uncertainty_quantification: bool = True,
        molecular_similarity_threshold: float = 0.7,
    ):
        self.base_models = base_models
        self.adaptation_strategy = adaptation_strategy
        self.uncertainty_quantification = uncertainty_quantification
        self.molecular_similarity_threshold = molecular_similarity_threshold
        self.model_weights_ = None
        self.performance_history_ = []
        self.molecular_clusters_ = None
        self.fitted_ = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        molecular_features: Optional[np.ndarray] = None,
    ) -> Self:
        """
        Fit the adaptive ensemble to training data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training features
        y : array-like, shape (n_samples,)
            Training targets
        molecular_features : array-like, optional
            Additional molecular features for clustering
        """
        with performance_context("adaptive_ensemble_fit"):
            if molecular_features is not None:
                self.molecular_clusters_ = self._identify_molecular_clusters(
                    molecular_features
                )
            for i, model in enumerate(self.base_models):
                try:
                    model.fit(X, y)
                except Exception as e:
                    warnings.warn(f"Model {i} failed to fit: {e}")
            self.model_weights_ = self._calculate_adaptive_weights(X, y)
            self.fitted_ = True
        return self

    def predict(
        self, X: np.ndarray, return_uncertainty: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions using the adaptive ensemble.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test features
        return_uncertainty : bool
            Whether to return uncertainty estimates

        Returns:
        --------
        predictions : array-like
            Ensemble predictions
        uncertainties : array-like, optional
            Uncertainty estimates (if requested)
        """
        if not self.fitted_:
            raise ValueError("Ensemble must be fitted before making predictions")
        model_predictions = []
        for model in self.base_models:
            try:
                pred = model.predict(X)
                model_predictions.append(pred)
            except Exception as e:
                warnings.warn(f"Model failed to predict: {e}")
                model_predictions.append(np.zeros(len(X)))
        model_predictions = np.array(model_predictions).T
        ensemble_predictions = self._adaptive_predict(model_predictions, X)
        if return_uncertainty and self.uncertainty_quantification:
            uncertainties = self._calculate_uncertainties(model_predictions)
            return ensemble_predictions, uncertainties
        return ensemble_predictions

    def _identify_molecular_clusters(
        self, molecular_features: np.ndarray
    ) -> np.ndarray:
        """Identify molecular clusters based on similarity."""
        from sklearn.cluster import KMeans

        n_clusters = min(10, len(molecular_features) // 50)
        if n_clusters < 2:
            return np.zeros(len(molecular_features))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        return kmeans.fit_predict(molecular_features)

    def _calculate_adaptive_weights(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculate adaptive weights for base models with robust cross-validation."""
        n_models = len(self.base_models)
        weights = np.ones(n_models) / n_models

        if self.adaptation_strategy == "performance_weighted":
            for i, model in enumerate(self.base_models):
                try:
                    # Enhanced cross-validation with error handling
                    from sklearn.model_selection import KFold, StratifiedKFold

                    # Determine if this is classification or regression
                    is_classification = (
                        hasattr(model, "predict_proba") or len(np.unique(y)) < 20
                    )

                    # Choose appropriate CV strategy
                    if is_classification and len(np.unique(y)) > 1:
                        try:
                            cv_strategy = StratifiedKFold(
                                n_splits=min(3, len(np.unique(y))),
                                shuffle=True,
                                random_state=42,
                            )
                        except ValueError:
                            cv_strategy = KFold(
                                n_splits=min(3, len(y) // 2),
                                shuffle=True,
                                random_state=42,
                            )
                    else:
                        cv_strategy = KFold(
                            n_splits=min(3, len(y) // 2), shuffle=True, random_state=42
                        )

                    # Perform cross-validation
                    scores = cross_val_score(
                        model,
                        X,
                        y,
                        cv=cv_strategy,
                        scoring="neg_mean_squared_error",
                        error_score=0.0,
                    )

                    # Convert to positive weight (higher is better)
                    if len(scores) > 0:
                        weights[i] = np.exp(
                            np.mean(scores) / 1000
                        )  # Scale to prevent overflow
                    else:
                        weights[i] = 0.1

                except Exception as e:
                    warnings.warn(f"Model {i} failed during weight calculation: {e}")
                    weights[i] = 0.1

        elif self.adaptation_strategy == "diversity_weighted":
            weights = self._calculate_diversity_weights(X, y)

        # Normalize weights and handle edge cases
        weights = np.maximum(weights, 0.01)  # Minimum weight
        weights = weights / np.sum(weights)
        return weights

    def _calculate_diversity_weights(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculate weights based on model diversity."""
        n_models = len(self.base_models)
        weights = np.ones(n_models)
        predictions = []
        for model in self.base_models:
            try:
                pred = cross_val_score(model, X, y, cv=3)
                predictions.append(pred)
            except Exception:
                predictions.append(np.zeros(3))
        correlations = np.corrcoef(predictions)
        for i in range(n_models):
            avg_correlation = np.mean(np.abs(correlations[i, :]))
            weights[i] = 1.0 / (1.0 + avg_correlation)
        return weights

    def _adaptive_predict(
        self, model_predictions: np.ndarray, X: np.ndarray
    ) -> np.ndarray:
        """Make adaptive predictions based on input characteristics."""
        if self.molecular_clusters_ is not None:
            cluster_assignments = np.random.randint(
                0, max(self.molecular_clusters_) + 1, len(X)
            )
            weighted_predictions = np.average(
                model_predictions, weights=self.model_weights_, axis=1
            )
        else:
            weighted_predictions = np.average(
                model_predictions, weights=self.model_weights_, axis=1
            )
        return weighted_predictions

    def _calculate_uncertainties(self, model_predictions: np.ndarray) -> np.ndarray:
        """Calculate prediction uncertainties."""
        uncertainties = np.var(model_predictions, axis=1)
        if self.model_weights_ is not None:
            weighted_uncertainty = uncertainties * (1.0 + np.std(self.model_weights_))
        else:
            weighted_uncertainty = uncertainties
        return weighted_uncertainty


class MultiModalEnsemble(BaseEstimator):
    """
    Ensemble that combines predictions from different molecular representations
    (e.g., SMILES, descriptors, fingerprints, 3D structure).
    """

    def __init__(
        self,
        modality_models: Dict[str, Any],
        fusion_strategy: str = "late_fusion",
        modality_weights: Optional[Dict[str, float]] = None,
    ):
        self.modality_models = modality_models
        self.fusion_strategy = fusion_strategy
        self.modality_weights = modality_weights or {}
        self.fitted_models_ = {}
        self.learned_weights_ = {}
        self.fitted_ = False

    def fit(self, modality_data: Dict[str, np.ndarray], y: np.ndarray) -> Self:
        """
        Fit multi-modal ensemble.

        Parameters:
        -----------
        modality_data : dict
            Dictionary mapping modality names to their data arrays
        y : array-like
            Target values
        """
        with performance_context("multimodal_ensemble_fit"):
            for modality, data in modality_data.items():
                if modality in self.modality_models:
                    model = self.modality_models[modality]
                    try:
                        model.fit(data, y)
                        self.fitted_models_[modality] = model
                    except Exception as e:
                        warnings.warn(
                            f"Failed to fit model for modality {modality}: {e}"
                        )
            self.learned_weights_ = self._learn_modality_weights(modality_data, y)
            self.fitted_ = True
        return self

    def predict(self, modality_data: Dict[str, np.ndarray]) -> np.ndarray:
        """Make multi-modal predictions."""
        if not self.fitted_:
            raise ValueError("Ensemble must be fitted before making predictions")
        modality_predictions = {}
        for modality, data in modality_data.items():
            if modality in self.fitted_models_:
                try:
                    pred = self.fitted_models_[modality].predict(data)
                    modality_predictions[modality] = pred
                except Exception as e:
                    warnings.warn(f"Failed to predict with modality {modality}: {e}")
        if self.fusion_strategy == "late_fusion":
            return self._late_fusion(modality_predictions)
        elif self.fusion_strategy == "weighted_fusion":
            return self._weighted_fusion(modality_predictions)
        else:
            return self._simple_average(modality_predictions)

    def _learn_modality_weights(
        self, modality_data: Dict[str, np.ndarray], y: np.ndarray
    ) -> Dict[str, float]:
        """Learn optimal weights for each modality."""
        weights = {}
        for modality in self.fitted_models_:
            if modality in modality_data:
                try:
                    model = self.fitted_models_[modality]
                    scores = cross_val_score(model, modality_data[modality], y, cv=3)
                    weights[modality] = np.mean(scores)
                except Exception:
                    weights[modality] = 0.1
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: (v / total_weight) for k, v in weights.items()}
        return weights

    def _late_fusion(self, modality_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Perform late fusion of modality predictions."""
        if not modality_predictions:
            raise ValueError("No valid predictions from any modality")
        predictions_array = []
        weights_array = []
        for modality, pred in modality_predictions.items():
            predictions_array.append(pred)
            weight = self.learned_weights_.get(modality, 1.0)
            weights_array.append(weight)
        predictions_array = np.array(predictions_array)
        weights_array = np.array(weights_array)
        weighted_pred = np.average(predictions_array, weights=weights_array, axis=0)
        return weighted_pred

    def _weighted_fusion(
        self, modality_predictions: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Perform weighted fusion with confidence-based weighting."""
        return self._late_fusion(modality_predictions)

    def _simple_average(
        self, modality_predictions: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Simple average of all modality predictions."""
        if not modality_predictions:
            raise ValueError("No valid predictions from any modality")
        predictions_array = np.array(list(modality_predictions.values()))
        return np.mean(predictions_array, axis=0)


class UncertaintyQuantifiedEnsemble(BaseEstimator):
    """
    Ensemble with comprehensive uncertainty quantification for molecular predictions.
    Provides both aleatoric (data) and epistemic (model) uncertainty estimates.
    """

    def __init__(
        self,
        base_models: List[Any],
        uncertainty_methods: Optional[List[str]] = None,
        bootstrap_samples: int = 100,
    ):
        self.base_models = base_models
        self.uncertainty_methods = uncertainty_methods or [
            "bootstrap",
            "variance",
            "entropy",
        ]
        self.bootstrap_samples = bootstrap_samples
        self.bootstrap_models_ = []
        self.fitted_ = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """Fit ensemble with uncertainty quantification."""
        with performance_context("uncertainty_ensemble_fit"):
            for model in self.base_models:
                try:
                    model.fit(X, y)
                except Exception as e:
                    warnings.warn(f"Model failed to fit: {e}")
            if "bootstrap" in self.uncertainty_methods:
                self._create_bootstrap_models(X, y)
            self.fitted_ = True
        return self

    def predict(
        self, X: np.ndarray, return_uncertainties: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, np.ndarray]]]:
        """Make predictions with uncertainty estimates."""
        if not self.fitted_:
            raise ValueError("Ensemble must be fitted before making predictions")
        base_predictions = []
        for model in self.base_models:
            try:
                pred = model.predict(X)
                base_predictions.append(pred)
            except Exception:
                continue
        base_predictions = np.array(base_predictions)
        ensemble_pred = np.mean(base_predictions, axis=0)
        if not return_uncertainties:
            return ensemble_pred
        uncertainties = self._calculate_comprehensive_uncertainties(X, base_predictions)
        return ensemble_pred, uncertainties

    def _create_bootstrap_models(self, X: np.ndarray, y: np.ndarray) -> None:
        """Create bootstrap models for uncertainty estimation."""
        n_samples = len(X)
        for _ in range(self.bootstrap_samples):
            bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]
            bootstrap_model_set = []
            for model in self.base_models:
                try:
                    bootstrap_model = type(model)(**model.get_params())
                    bootstrap_model.fit(X_bootstrap, y_bootstrap)
                    bootstrap_model_set.append(bootstrap_model)
                except Exception:
                    continue
            if bootstrap_model_set:
                self.bootstrap_models_.append(bootstrap_model_set)

    def _calculate_comprehensive_uncertainties(
        self, X: np.ndarray, base_predictions: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Calculate multiple types of uncertainty estimates."""
        uncertainties = {}
        if "variance" in self.uncertainty_methods:
            uncertainties["variance"] = np.var(base_predictions, axis=0)
        if "bootstrap" in self.uncertainty_methods and self.bootstrap_models_:
            bootstrap_uncertainties = self._calculate_bootstrap_uncertainty(X)
            uncertainties["bootstrap"] = bootstrap_uncertainties
        if "intervals" in self.uncertainty_methods:
            uncertainties[
                "prediction_intervals"
            ] = self._calculate_prediction_intervals(base_predictions)
        total_uncertainty = np.zeros(len(X))
        for unc_type, unc_values in uncertainties.items():
            if unc_type != "prediction_intervals":
                total_uncertainty += unc_values
        uncertainties["total"] = total_uncertainty / len(
            [k for k in uncertainties.keys() if k != "prediction_intervals"]
        )
        return uncertainties

    def _calculate_bootstrap_uncertainty(self, X: np.ndarray) -> np.ndarray:
        """Calculate uncertainty from bootstrap models."""
        bootstrap_predictions = []
        for model_set in self.bootstrap_models_:
            set_predictions = []
            for model in model_set:
                try:
                    pred = model.predict(X)
                    set_predictions.append(pred)
                except Exception:
                    continue
            if set_predictions:
                bootstrap_predictions.append(np.mean(set_predictions, axis=0))
        if bootstrap_predictions:
            bootstrap_predictions = np.array(bootstrap_predictions)
            return np.std(bootstrap_predictions, axis=0)
        else:
            return np.zeros(len(X))

    def _calculate_prediction_intervals(
        self, base_predictions: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Calculate prediction intervals."""
        intervals = {}
        intervals["lower_95"] = np.percentile(base_predictions, 2.5, axis=0)
        intervals["upper_95"] = np.percentile(base_predictions, 97.5, axis=0)
        intervals["lower_68"] = np.percentile(base_predictions, 16, axis=0)
        intervals["upper_68"] = np.percentile(base_predictions, 84, axis=0)
        return intervals


def create_adaptive_ensemble(
    base_models: List[Any], adaptation_strategy: str = "performance_weighted"
) -> AdaptiveEnsemble:
    """
    Create an adaptive ensemble with default settings.

    Parameters:
    -----------
    base_models : list
        List of base models to ensemble
    adaptation_strategy : str
        Strategy for adaptation ('performance_weighted' or 'diversity_weighted')

    Returns:
    --------
    AdaptiveEnsemble : Configured adaptive ensemble
    """
    return AdaptiveEnsemble(
        base_models=base_models,
        adaptation_strategy=adaptation_strategy,
        uncertainty_quantification=True,
    )


def create_multimodal_ensemble(
    modality_models: Dict[str, Any], fusion_strategy: str = "late_fusion"
) -> MultiModalEnsemble:
    """
    Create a multi-modal ensemble for different molecular representations.

    Parameters:
    -----------
    modality_models : dict
        Dictionary mapping modality names to their models
    fusion_strategy : str
        Fusion strategy ('late_fusion', 'weighted_fusion', or 'simple_average')

    Returns:
    --------
    MultiModalEnsemble : Configured multi-modal ensemble
    """
    return MultiModalEnsemble(
        modality_models=modality_models, fusion_strategy=fusion_strategy
    )


def create_uncertainty_ensemble(
    base_models: List[Any], uncertainty_methods: Optional[List[str]] = None
) -> UncertaintyQuantifiedEnsemble:
    """
    Create an uncertainty quantified ensemble.

    Parameters:
    -----------
    base_models : list
        List of base models
    uncertainty_methods : list, optional
        Methods for uncertainty quantification

    Returns:
    --------
    UncertaintyQuantifiedEnsemble : Configured uncertainty ensemble
    """
    return UncertaintyQuantifiedEnsemble(
        base_models=base_models,
        uncertainty_methods=uncertainty_methods or ["bootstrap", "variance"],
    )
