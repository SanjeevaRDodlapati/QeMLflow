# Enhanced Ensemble Methods for Day 2 Deep Learning Notebook
# This code enhances the existing EnsemblePredictor with robust error handling,
# multiple model type support, weighted averaging, and uncertainty quantification.

import logging
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


class EnhancedEnsemblePredictor:
    """Advanced ensemble predictor with robust error handling and multiple model type support"""

    def __init__(
        self,
        models_info: List[Dict],
        performance_weights: bool = True,
        fallback_strategy: str = "average",
        uncertainty_quantification: bool = True,
    ):
        """
        Enhanced ensemble predictor initialization

        Args:
            models_info: List of dicts with 'model', 'type', 'weight', 'performance' keys
            performance_weights: Whether to use performance-based weighting
            fallback_strategy: Strategy for failed models ('average', 'best', 'skip')
            uncertainty_quantification: Whether to compute prediction uncertainties
        """
        self.models_info = models_info
        self.performance_weights = performance_weights
        self.fallback_strategy = fallback_strategy
        self.uncertainty_quantification = uncertainty_quantification

        # Model performance tracking
        self.model_performances = {}
        self.prediction_history = defaultdict(list)
        self.failure_counts = defaultdict(int)

        # Initialize performance weights if provided
        self._initialize_performance_weights()

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _initialize_performance_weights(self):
        """Initialize performance-based weights"""
        for model_info in self.models_info:
            model_id = id(model_info["model"])
            performance = model_info.get("performance", 0.8)  # Default performance
            self.model_performances[model_id] = performance

    def _get_dynamic_weights(self) -> np.ndarray:
        """Calculate dynamic weights based on model performance"""
        if not self.performance_weights:
            return np.array([info["weight"] for info in self.models_info])

        weights = []
        for model_info in self.models_info:
            model_id = id(model_info["model"])
            base_weight = model_info["weight"]
            performance = self.model_performances.get(model_id, 0.8)
            failure_penalty = max(0.1, 1.0 - (self.failure_counts[model_id] * 0.1))

            dynamic_weight = base_weight * performance * failure_penalty
            weights.append(dynamic_weight)

        # Normalize weights
        weights = np.array(weights)
        return weights / weights.sum() if weights.sum() > 0 else weights

    def _predict_single_model(
        self, model_info: Dict, graph_data, transformer_data
    ) -> Optional[np.ndarray]:
        """Predict with a single model with comprehensive error handling"""
        model = model_info["model"]
        model_type = model_info["type"]
        model_id = id(model)

        try:
            model.eval()
            with torch.no_grad():
                if model_type in ["graph", "gcn", "gat"]:
                    pred = self._predict_graph_model(model, graph_data)
                elif model_type == "transformer":
                    pred = self._predict_transformer_model(model, transformer_data)
                else:
                    self.logger.warning(f"Unknown model type: {model_type}")
                    return None

                # Validate prediction
                if self._validate_prediction(pred):
                    self.prediction_history[model_id].append(pred)
                    return pred
                else:
                    self.logger.warning(f"Invalid prediction from {model_type} model")
                    return None

        except Exception as e:
            self.failure_counts[model_id] += 1
            self.logger.error(f"Model {model_type} failed: {str(e)}")
            return None

    def _predict_graph_model(self, model, graph_data) -> np.ndarray:
        """Predict with graph-based models (GCN, GAT, etc.)"""
        try:
            # Try standard graph model signature
            out = model(graph_data.x, graph_data.edge_index, graph_data.batch)
        except (TypeError, AttributeError):
            try:
                # Fallback for models without batch parameter
                out = model(graph_data)
            except Exception:
                # Final fallback for direct data input
                out = model(graph_data.x, graph_data.edge_index)

        # Handle different output formats and apply appropriate activation
        if hasattr(model, "classifier") and hasattr(model.classifier, "__getitem__"):
            # Model already has activation in classifier
            pred = out.squeeze().cpu().numpy()
        else:
            # Apply sigmoid for probability outputs
            pred = torch.sigmoid(out.squeeze()).cpu().numpy()

        return pred

    def _predict_transformer_model(self, model, transformer_data) -> np.ndarray:
        """Predict with transformer models"""
        try:
            # Handle padding mask creation
            if "char_to_idx" in globals() and "<PAD>" in char_to_idx:
                padding_mask = transformer_data == char_to_idx["<PAD>"]
                out = model(transformer_data, padding_mask)
            else:
                # Fallback without padding mask
                out = model(transformer_data)

            # Transformer typically has activation in classifier
            pred = out.squeeze().cpu().numpy()
            return pred

        except Exception as e:
            # Try alternative transformer interfaces
            try:
                out = model(transformer_data)
                pred = torch.sigmoid(out.squeeze()).cpu().numpy()
                return pred
            except Exception:
                raise e

    def _validate_prediction(self, pred: np.ndarray) -> bool:
        """Validate prediction output"""
        if pred is None:
            return False
        if np.any(np.isnan(pred)) or np.any(np.isinf(pred)):
            return False
        if np.any(pred < 0) or np.any(pred > 1):
            # Clip values if slightly out of bounds
            if np.all(pred >= -0.1) and np.all(pred <= 1.1):
                np.clip(pred, 0, 1, out=pred)
                return True
            return False
        return True

    def _apply_fallback_strategy(
        self, successful_predictions: List[np.ndarray], successful_weights: List[float]
    ) -> np.ndarray:
        """Apply fallback strategy when some models fail"""
        if not successful_predictions:
            # All models failed - return default prediction
            self.logger.error("All models failed - returning default prediction")
            return np.array([0.5])  # Neutral prediction

        if self.fallback_strategy == "average":
            return np.mean(successful_predictions, axis=0)
        elif self.fallback_strategy == "weighted":
            if len(successful_weights) > 0:
                weights = np.array(successful_weights)
                weights = weights / weights.sum()
                return np.average(successful_predictions, axis=0, weights=weights)
            else:
                return np.mean(successful_predictions, axis=0)
        elif self.fallback_strategy == "best":
            # Return prediction from model with highest weight
            best_idx = np.argmax(successful_weights)
            return successful_predictions[best_idx]
        else:
            return np.mean(successful_predictions, axis=0)

    def predict(
        self, graph_data, transformer_data, return_uncertainty: bool = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """Make ensemble predictions with advanced error handling"""
        if return_uncertainty is None:
            return_uncertainty = self.uncertainty_quantification

        predictions = []
        weights = []
        successful_models = []

        # Get dynamic weights
        dynamic_weights = self._get_dynamic_weights()

        # Collect predictions from all models
        for i, model_info in enumerate(self.models_info):
            pred = self._predict_single_model(model_info, graph_data, transformer_data)

            if pred is not None:
                predictions.append(pred)
                weights.append(dynamic_weights[i])
                successful_models.append(model_info["type"])

        # Apply fallback strategy if needed
        if len(predictions) < len(self.models_info):
            failed_count = len(self.models_info) - len(predictions)
            self.logger.warning(
                f"{failed_count} models failed, using fallback strategy"
            )

        # Compute ensemble prediction
        ensemble_pred = self._apply_fallback_strategy(predictions, weights)

        if not return_uncertainty:
            return ensemble_pred

        # Compute uncertainty metrics
        uncertainty_info = self._compute_uncertainty(
            predictions, weights, successful_models
        )

        return ensemble_pred, uncertainty_info

    def _compute_uncertainty(
        self,
        predictions: List[np.ndarray],
        weights: List[float],
        successful_models: List[str],
    ) -> Dict:
        """Compute prediction uncertainty metrics"""
        if len(predictions) <= 1:
            return {
                "std": 0.0,
                "variance": 0.0,
                "confidence": 0.5,
                "model_agreement": 0.0,
                "successful_models": successful_models,
            }

        predictions_array = np.array(predictions)

        # Calculate basic uncertainty metrics
        std = np.std(predictions_array, axis=0)
        variance = np.var(predictions_array, axis=0)

        # Model agreement (inverse of coefficient of variation)
        mean_pred = np.mean(predictions_array, axis=0)
        cv = std / (mean_pred + 1e-8)
        agreement = 1.0 / (1.0 + cv)

        # Confidence based on weight distribution and agreement
        weight_entropy = -np.sum(weights * np.log(weights + 1e-8))
        confidence = agreement * (1.0 - weight_entropy / np.log(len(weights)))

        return {
            "std": float(np.mean(std)),
            "variance": float(np.mean(variance)),
            "confidence": float(np.mean(confidence)),
            "model_agreement": float(np.mean(agreement)),
            "successful_models": successful_models,
            "weight_distribution": weights,
        }

    def update_performance(self, model_idx: int, performance_score: float):
        """Update model performance for dynamic weighting"""
        if 0 <= model_idx < len(self.models_info):
            model_id = id(self.models_info[model_idx]["model"])
            self.model_performances[model_id] = performance_score

    def get_model_statistics(self) -> Dict:
        """Get comprehensive model performance statistics"""
        stats = {}
        for i, model_info in enumerate(self.models_info):
            model_id = id(model_info["model"])
            stats[f"{model_info['type']}_model_{i}"] = {
                "performance": self.model_performances.get(model_id, 0.8),
                "failure_count": self.failure_counts[model_id],
                "prediction_count": len(self.prediction_history[model_id]),
                "reliability": max(0.0, 1.0 - (self.failure_counts[model_id] * 0.1)),
            }
        return stats


# Enhanced backward compatible ensemble predictor
class EnsemblePredictor(EnhancedEnsemblePredictor):
    """Backward compatible ensemble predictor with enhanced features"""

    def __init__(self, models_info):
        # Convert old format to new format if needed
        if isinstance(models_info, list) and len(models_info) > 0:
            if "performance" not in models_info[0]:
                for model_info in models_info:
                    model_info["performance"] = 0.8  # Default performance

        super().__init__(
            models_info,
            performance_weights=True,
            fallback_strategy="weighted",
            uncertainty_quantification=False,
        )


# Demo usage and integration example
def demonstrate_enhanced_ensemble():
    """Demonstrate the enhanced ensemble capabilities"""
    print("🚀 Enhanced Ensemble Methods Integration")
    print("=" * 50)

    # This would be called with the actual models from the notebook:
    # enhanced_ensemble_models = [
    #     {'model': model_gcn, 'type': 'graph', 'weight': 0.4, 'performance': 0.85},
    #     {'model': model_gat, 'type': 'graph', 'weight': 0.4, 'performance': 0.87},
    #     {'model': model_transformer, 'type': 'transformer', 'weight': 0.2, 'performance': 0.82}
    # ]
    #
    # enhanced_ensemble = EnhancedEnsemblePredictor(
    #     enhanced_ensemble_models,
    #     performance_weights=True,
    #     fallback_strategy='weighted',
    #     uncertainty_quantification=True
    # )
    #
    # # Make predictions with uncertainty
    # pred, uncertainty = enhanced_ensemble.predict(graph_data, transformer_data, return_uncertainty=True)
    #
    # print(f"✅ Ensemble prediction: {pred}")
    # print(f"✅ Prediction confidence: {uncertainty['confidence']:.3f}")
    # print(f"✅ Model agreement: {uncertainty['model_agreement']:.3f}")
    # print(f"✅ Successful models: {uncertainty['successful_models']}")
    #
    # # Update model performance
    # enhanced_ensemble.update_performance(0, 0.90)  # Update GCN performance
    #
    # # Get model statistics
    # stats = enhanced_ensemble.get_model_statistics()
    # print(f"✅ Model statistics: {stats}")

    print("✅ Enhanced ensemble methods ready for integration!")
    print(
        "📝 To integrate: Replace the existing EnsemblePredictor class with EnhancedEnsemblePredictor"
    )
    print("📝 The EnsemblePredictor class provides backward compatibility")


if __name__ == "__main__":
    demonstrate_enhanced_ensemble()
