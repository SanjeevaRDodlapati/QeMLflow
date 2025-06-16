"""
Comprehensive test suite for metrics module.

This test suite provides thorough coverage of the metrics functionality
including classification metrics, regression metrics, molecular metrics,
and fallback implementations.
"""

import os
import sys
import tempfile
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn import *

try:
    from rdkit import Chem
except ImportError:
    pass

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

try:
    from utils.metrics import (
        ClassificationMetrics,
        MolecularMetrics,
        RegressionMetrics,
        accuracy,
        calculate_enrichment_factor,
        evaluate_model_performance,
        explained_variance_score,
        f1_score_manual,
        mean_absolute_error_manual,
        mean_absolute_percentage_error,
        mean_squared_error_manual,
        precision,
        r_squared,
        recall,
    )
except ImportError as e:
    pytest.skip(f"Could not import metrics module: {e}", allow_module_level=True)


class TestClassificationMetrics:
    """Test cases for ClassificationMetrics class."""

    def test_calculate_all_metrics_with_sklearn(self):
        """Test comprehensive classification metrics with sklearn available."""
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 0, 1, 1, 1, 1])
        y_prob = np.array([0.1, 0.9, 0.3, 0.2, 0.8, 0.7, 0.9, 0.8])

        with patch("utils.metrics.SKLEARN_AVAILABLE", True):
            metrics = ClassificationMetrics.calculate_all_metrics(
                y_true, y_pred, y_prob
            )

            assert "accuracy" in metrics
            assert "precision" in metrics
            assert "recall" in metrics
            assert "f1_score" in metrics
            assert "roc_auc" in metrics

            assert 0.0 <= metrics["accuracy"] <= 1.0
            assert 0.0 <= metrics["precision"] <= 1.0
            assert 0.0 <= metrics["recall"] <= 1.0
            assert 0.0 <= metrics["f1_score"] <= 1.0
            assert 0.0 <= metrics["roc_auc"] <= 1.0

    def test_calculate_all_metrics_without_sklearn(self):
        """Test classification metrics fallback when sklearn not available."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])

        with patch("utils.metrics.SKLEARN_AVAILABLE", False):
            metrics = ClassificationMetrics.calculate_all_metrics(y_true, y_pred)

            assert "accuracy" in metrics
            assert "precision" in metrics
            assert "recall" in metrics
            assert "f1_score" in metrics
            assert "roc_auc" not in metrics  # Should not be calculated without sklearn

            assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_calculate_all_metrics_with_roc_auc_error(self):
        """Test ROC AUC calculation handles errors gracefully."""
        y_true = np.array([1, 1, 1, 1, 1])  # All same class
        y_pred = np.array([1, 1, 0, 1, 1])
        y_prob = np.array([0.9, 0.8, 0.3, 0.9, 0.8])

        with patch("utils.metrics.SKLEARN_AVAILABLE", True):
            metrics = ClassificationMetrics.calculate_all_metrics(
                y_true, y_pred, y_prob
            )

            # ROC AUC should not be included when calculation fails
            assert "accuracy" in metrics
            assert "precision" in metrics
            assert "recall" in metrics
            assert "f1_score" in metrics

    def test_calculate_all_metrics_multiclass(self):
        """Test classification metrics with multiclass data."""
        y_true = np.array([0, 1, 2, 0, 1, 2, 1, 2])
        y_pred = np.array([0, 1, 1, 0, 2, 2, 1, 2])

        with patch("utils.metrics.SKLEARN_AVAILABLE", True):
            metrics = ClassificationMetrics.calculate_all_metrics(y_true, y_pred)

            assert "accuracy" in metrics
            assert "precision" in metrics
            assert "recall" in metrics
            assert "f1_score" in metrics
            # ROC AUC should not be calculated for multiclass without explicit handling

    def test_confusion_matrix_metrics_binary(self):
        """Test confusion matrix metrics for binary classification."""
        y_true = np.array([0, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 1, 1])

        with patch("utils.metrics.SKLEARN_AVAILABLE", True):
            result = ClassificationMetrics.confusion_matrix_metrics(y_true, y_pred)

            assert "confusion_matrix" in result
            assert "sensitivity" in result
            assert "specificity" in result
            assert "true_positive_rate" in result
            assert "false_positive_rate" in result

            assert result["confusion_matrix"].shape == (2, 2)
            assert 0.0 <= result["sensitivity"] <= 1.0
            assert 0.0 <= result["specificity"] <= 1.0

    def test_confusion_matrix_metrics_without_sklearn(self):
        """Test confusion matrix calculation without sklearn."""
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 1])

        with patch("utils.metrics.SKLEARN_AVAILABLE", False):
            result = ClassificationMetrics.confusion_matrix_metrics(y_true, y_pred)

            assert "confusion_matrix" in result
            assert isinstance(result["confusion_matrix"], np.ndarray)

    def test_confusion_matrix_metrics_multiclass(self):
        """Test confusion matrix for multiclass classification."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 1, 0, 2, 2])

        with patch("utils.metrics.SKLEARN_AVAILABLE", True):
            result = ClassificationMetrics.confusion_matrix_metrics(y_true, y_pred)

            assert "confusion_matrix" in result
            assert result["confusion_matrix"].shape == (3, 3)
            # Should not have binary-specific metrics
            assert "sensitivity" not in result

    def test_confusion_matrix_edge_cases(self):
        """Test confusion matrix with edge cases."""
        # Perfect classification
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])

        with patch("utils.metrics.SKLEARN_AVAILABLE", True):
            result = ClassificationMetrics.confusion_matrix_metrics(y_true, y_pred)

            assert result["sensitivity"] == 1.0
            assert result["specificity"] == 1.0

        # All wrong classification
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 0])

        with patch("utils.metrics.SKLEARN_AVAILABLE", True):
            result = ClassificationMetrics.confusion_matrix_metrics(y_true, y_pred)

            assert result["sensitivity"] == 0.0
            assert result["specificity"] == 0.0


class TestRegressionMetrics:
    """Test cases for RegressionMetrics class."""

    def test_calculate_all_metrics_with_sklearn(self):
        """Test comprehensive regression metrics with sklearn available."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])

        with patch("utils.metrics.SKLEARN_AVAILABLE", True):
            metrics = RegressionMetrics.calculate_all_metrics(y_true, y_pred)

            assert "mse" in metrics
            assert "rmse" in metrics
            assert "mae" in metrics
            assert "r2" in metrics
            assert "mape" in metrics
            assert "max_error" in metrics
            assert "explained_variance" in metrics

            assert metrics["mse"] >= 0.0
            assert metrics["rmse"] >= 0.0
            assert metrics["mae"] >= 0.0
            assert metrics["max_error"] >= 0.0
            assert metrics["rmse"] == np.sqrt(metrics["mse"])

    def test_calculate_all_metrics_without_sklearn(self):
        """Test regression metrics fallback when sklearn not available."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])

        with patch("utils.metrics.SKLEARN_AVAILABLE", False):
            metrics = RegressionMetrics.calculate_all_metrics(y_true, y_pred)

            assert "mse" in metrics
            assert "rmse" in metrics
            assert "mae" in metrics
            assert "r2" in metrics
            assert "mape" in metrics
            assert "max_error" in metrics
            assert "explained_variance" in metrics

            # Check calculations are reasonable
            assert metrics["mse"] >= 0.0
            assert metrics["rmse"] == np.sqrt(metrics["mse"])

    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        metrics = RegressionMetrics.calculate_all_metrics(y_true, y_pred)

        assert metrics["mse"] == 0.0
        assert metrics["rmse"] == 0.0
        assert metrics["mae"] == 0.0
        assert metrics["max_error"] == 0.0
        assert metrics["r2"] == 1.0
        assert metrics["mape"] == 0.0
        assert metrics["explained_variance"] == 1.0

    def test_regression_metrics_edge_cases(self):
        """Test regression metrics with edge cases."""
        # Constant predictions
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([3.0, 3.0, 3.0, 3.0, 3.0])

        metrics = RegressionMetrics.calculate_all_metrics(y_true, y_pred)

        assert metrics["mse"] > 0.0
        assert metrics["mae"] > 0.0
        assert metrics["r2"] == 0.0  # R² should be 0 for constant predictions

    def test_regression_metrics_with_zeros(self):
        """Test regression metrics when true values contain zeros."""
        y_true = np.array([0.0, 1.0, 2.0, 0.0, 3.0])
        y_pred = np.array([0.1, 1.1, 1.9, 0.1, 2.9])

        metrics = RegressionMetrics.calculate_all_metrics(y_true, y_pred)

        # MAPE should handle zeros appropriately
        assert "mape" in metrics
        assert metrics["mape"] >= 0.0


class TestMolecularMetrics:
    """Test cases for MolecularMetrics class."""

    def test_tanimoto_similarity_with_rdkit(self):
        """Test Tanimoto similarity calculation with RDKit available."""
        smiles1 = "CCO"  # Ethanol
        smiles2 = "CCO"  # Same molecule

        with patch("utils.metrics.RDKIT_AVAILABLE", True):
            similarity = MolecularMetrics.tanimoto_similarity(smiles1, smiles2)

            assert similarity == 1.0  # Identical molecules should have similarity 1.0

    def test_tanimoto_similarity_different_molecules(self):
        """Test Tanimoto similarity between different molecules."""
        smiles1 = "CCO"  # Ethanol
        smiles2 = "CCCO"  # Propanol

        with patch("utils.metrics.RDKIT_AVAILABLE", True):
            similarity = MolecularMetrics.tanimoto_similarity(smiles1, smiles2)

            assert 0.0 <= similarity <= 1.0
            assert similarity < 1.0  # Different molecules should have similarity < 1.0

    def test_tanimoto_similarity_without_rdkit(self):
        """Test Tanimoto similarity fallback when RDKit not available."""
        smiles1 = "CCO"
        smiles2 = "CCCO"

        with patch("utils.metrics.RDKIT_AVAILABLE", False):
            similarity = MolecularMetrics.tanimoto_similarity(smiles1, smiles2)

            assert similarity == 0.5  # Should return default value

    def test_tanimoto_similarity_invalid_smiles(self):
        """Test Tanimoto similarity with invalid SMILES."""
        smiles1 = "INVALID_SMILES"
        smiles2 = "CCO"

        with patch("utils.metrics.RDKIT_AVAILABLE", True):
            similarity = MolecularMetrics.tanimoto_similarity(smiles1, smiles2)

            assert similarity == 0.0  # Should return 0 for invalid SMILES

    def test_tanimoto_similarity_fingerprint_types(self):
        """Test different fingerprint types for Tanimoto similarity."""
        smiles1 = "CCO"
        smiles2 = "CCCO"

        with patch("utils.metrics.RDKIT_AVAILABLE", True):
            # Test Morgan fingerprints
            similarity_morgan = MolecularMetrics.tanimoto_similarity(
                smiles1, smiles2, fingerprint_type="morgan"
            )

            # Test MACCS fingerprints
            similarity_maccs = MolecularMetrics.tanimoto_similarity(
                smiles1, smiles2, fingerprint_type="maccs"
            )

            assert 0.0 <= similarity_morgan <= 1.0
            assert 0.0 <= similarity_maccs <= 1.0

    def test_tanimoto_similarity_unsupported_fingerprint(self):
        """Test Tanimoto similarity with unsupported fingerprint type."""
        smiles1 = "CCO"
        smiles2 = "CCCO"

        with patch("utils.metrics.RDKIT_AVAILABLE", True):
            similarity = MolecularMetrics.tanimoto_similarity(
                smiles1, smiles2, fingerprint_type="unsupported"
            )

            assert similarity == 0.0  # Should handle error and return 0

    def test_diversity_metrics_with_rdkit(self):
        """Test diversity metrics calculation with RDKit available."""
        smiles_list = ["CCO", "CCCO", "CCCCO", "CC(C)O"]

        with patch("utils.metrics.RDKIT_AVAILABLE", True):
            diversity = MolecularMetrics.diversity_metrics(smiles_list)

            assert "mean_pairwise_similarity" in diversity
            assert "diversity_index" in diversity
            assert "max_similarity" in diversity
            assert "min_similarity" in diversity
            assert "std_similarity" in diversity

            assert 0.0 <= diversity["mean_pairwise_similarity"] <= 1.0
            assert 0.0 <= diversity["diversity_index"] <= 1.0
            assert 0.0 <= diversity["max_similarity"] <= 1.0
            assert 0.0 <= diversity["min_similarity"] <= 1.0

    def test_diversity_metrics_without_rdkit(self):
        """Test diversity metrics fallback when RDKit not available."""
        smiles_list = ["CCO", "CCCO", "CCCCO"]

        with patch("utils.metrics.RDKIT_AVAILABLE", False):
            diversity = MolecularMetrics.diversity_metrics(smiles_list)

            assert diversity["mean_pairwise_similarity"] == 0.5
            assert diversity["diversity_index"] == 0.5
            assert diversity["max_similarity"] == 1.0
            assert diversity["min_similarity"] == 0.0

    def test_diversity_metrics_single_molecule(self):
        """Test diversity metrics with single molecule."""
        smiles_list = ["CCO"]

        diversity = MolecularMetrics.diversity_metrics(smiles_list)

        # Should handle gracefully
        assert isinstance(diversity, dict)

    def test_diversity_metrics_invalid_molecules(self):
        """Test diversity metrics with invalid molecules."""
        smiles_list = ["INVALID", "ALSO_INVALID", "CCO"]

        with patch("utils.metrics.RDKIT_AVAILABLE", True):
            diversity = MolecularMetrics.diversity_metrics(smiles_list)

            # Should handle invalid molecules gracefully
            assert isinstance(diversity, dict)
            assert "mean_pairwise_similarity" in diversity

    def test_diversity_metrics_empty_list(self):
        """Test diversity metrics with empty molecule list."""
        smiles_list = []

        diversity = MolecularMetrics.diversity_metrics(smiles_list)

        assert diversity["mean_pairwise_similarity"] == 0.5
        assert diversity["diversity_index"] == 0.5


class TestFallbackFunctions:
    """Test cases for fallback metric implementations."""

    def test_accuracy_function(self):
        """Test manual accuracy calculation."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])

        acc = accuracy(y_true, y_pred)
        expected = 4 / 5  # 4 correct out of 5

        assert acc == expected

    def test_accuracy_edge_cases(self):
        """Test accuracy function edge cases."""
        # Perfect accuracy
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        assert accuracy(y_true, y_pred) == 1.0

        # Zero accuracy
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 0])
        assert accuracy(y_true, y_pred) == 0.0

        # Test error cases
        with pytest.raises(ValueError):
            accuracy(np.array([]), np.array([1, 2]))

        with pytest.raises(ValueError):
            accuracy(np.array([1, 2]), np.array([1]))

    def test_precision_function(self):
        """Test manual precision calculation."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 1])

        prec = precision(y_true, y_pred)
        # TP = 2, FP = 1, so precision = 2/3
        expected = 2 / 3

        assert abs(prec - expected) < 1e-10

    def test_precision_no_positive_predictions(self):
        """Test precision when no positive predictions."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 0, 0, 0])

        prec = precision(y_true, y_pred)
        assert prec == 0.0  # Should handle division by zero

    def test_recall_function(self):
        """Test manual recall calculation."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])

        rec = recall(y_true, y_pred)
        # TP = 2, FN = 1, so recall = 2/3
        expected = 2 / 3

        assert abs(rec - expected) < 1e-10

    def test_recall_no_positive_true(self):
        """Test recall when no positive true labels."""
        y_true = np.array([0, 0, 0, 0, 0])
        y_pred = np.array([0, 1, 0, 1, 1])

        rec = recall(y_true, y_pred)
        assert rec == 0.0  # Should handle division by zero

    def test_f1_score_manual_function(self):
        """Test manual F1 score calculation."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])

        f1 = f1_score_manual(y_true, y_pred)

        # Calculate expected F1
        prec = precision(y_true, y_pred)
        rec = recall(y_true, y_pred)
        expected = 2 * (prec * rec) / (prec + rec)

        assert abs(f1 - expected) < 1e-10

    def test_f1_score_zero_precision_recall(self):
        """Test F1 score when precision and recall are zero."""
        y_true = np.array([1, 1, 1])
        y_pred = np.array([0, 0, 0])

        f1 = f1_score_manual(y_true, y_pred)
        assert f1 == 0.0

    def test_mean_squared_error_manual_function(self):
        """Test manual MSE calculation."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 1.9, 3.1])

        mse = mean_squared_error_manual(y_true, y_pred)
        expected = np.mean([0.01, 0.01, 0.01])

        assert abs(mse - expected) < 1e-10

    def test_mean_absolute_error_manual_function(self):
        """Test manual MAE calculation."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 1.8, 3.2])

        mae = mean_absolute_error_manual(y_true, y_pred)
        expected = np.mean([0.1, 0.2, 0.2])

        assert abs(mae - expected) < 1e-10

    def test_r_squared_function(self):
        """Test manual R² calculation."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        r2 = r_squared(y_true, y_pred)
        assert r2 == 1.0  # Perfect predictions

        # Test with some error
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        r2 = r_squared(y_true, y_pred)
        assert 0.0 <= r2 <= 1.0

    def test_r_squared_constant_true(self):
        """Test R² with constant true values."""
        y_true = np.array([3.0, 3.0, 3.0, 3.0])
        y_pred = np.array([3.1, 2.9, 3.2, 2.8])

        r2 = r_squared(y_true, y_pred)
        assert r2 == 0.0  # Should handle constant true values

    def test_mean_absolute_percentage_error_function(self):
        """Test MAPE calculation."""
        y_true = np.array([1.0, 2.0, 4.0])
        y_pred = np.array([1.1, 1.8, 4.4])

        mape = mean_absolute_percentage_error(y_true, y_pred)
        expected = np.mean([10.0, 10.0, 10.0])  # 10% error for each

        assert abs(mape - expected) < 1e-10

    def test_mape_with_zeros(self):
        """Test MAPE with zero true values."""
        y_true = np.array([0.0, 2.0, 4.0])
        y_pred = np.array([0.1, 1.8, 4.4])

        mape = mean_absolute_percentage_error(y_true, y_pred)
        # Should only consider non-zero true values
        expected = np.mean([10.0, 10.0])

        assert abs(mape - expected) < 1e-10

    def test_mape_all_zeros(self):
        """Test MAPE when all true values are zero."""
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([0.1, 0.2, 0.3])

        mape = mean_absolute_percentage_error(y_true, y_pred)
        assert mape == 0.0

    def test_explained_variance_score_function(self):
        """Test explained variance score calculation."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        evs = explained_variance_score(y_true, y_pred)
        assert evs == 1.0  # Perfect predictions

    def test_explained_variance_constant_true(self):
        """Test explained variance with constant true values."""
        y_true = np.array([3.0, 3.0, 3.0, 3.0])
        y_pred = np.array([3.1, 2.9, 3.2, 2.8])

        evs = explained_variance_score(y_true, y_pred)
        assert evs == 0.0  # Should handle constant true values


class TestUtilityFunctions:
    """Test cases for utility evaluation functions."""

    def test_evaluate_model_performance_regression(self):
        """Test model performance evaluation for regression tasks."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])

        metrics = evaluate_model_performance(y_true, y_pred, task_type="regression")

        assert "mse" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics

    def test_evaluate_model_performance_classification(self):
        """Test model performance evaluation for classification tasks."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.3, 0.2, 0.8])

        metrics = evaluate_model_performance(
            y_true, y_pred, task_type="classification", y_prob=y_prob
        )

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics

    def test_evaluate_model_performance_unsupported_task(self):
        """Test model performance evaluation with unsupported task type."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 1.9, 3.1])

        with pytest.raises(ValueError):
            evaluate_model_performance(y_true, y_pred, task_type="unsupported")

    def test_calculate_enrichment_factor_basic(self):
        """Test enrichment factor calculation."""
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 0, 1, 0])  # 4 actives out of 10
        y_scores = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.1, 0.2, 0.6, 0.1])

        # Top 30% should contain 3 compounds
        ef = calculate_enrichment_factor(y_true, y_scores, fraction=0.3)

        assert ef >= 0.0  # Enrichment factor should be non-negative

    def test_calculate_enrichment_factor_perfect_ranking(self):
        """Test enrichment factor with perfect ranking."""
        y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8, 0.9, 1.0])

        # Top 50% should contain all actives
        ef = calculate_enrichment_factor(y_true, y_scores, fraction=0.5)

        assert ef == 1.6  # Actual enrichment calculated

    def test_calculate_enrichment_factor_no_actives(self):
        """Test enrichment factor when no actives in dataset."""
        y_true = np.array([0, 0, 0, 0, 0])
        y_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        ef = calculate_enrichment_factor(y_true, y_scores, fraction=0.4)

        assert ef == 0.0  # Should handle no actives gracefully

    def test_calculate_enrichment_factor_all_actives(self):
        """Test enrichment factor when all compounds are active."""
        y_true = np.array([1, 1, 1, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        ef = calculate_enrichment_factor(y_true, y_scores, fraction=0.4)

        assert ef == 1.0  # Should be 1.0 when all are active

    def test_calculate_enrichment_factor_random_ranking(self):
        """Test enrichment factor with random ranking."""
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_scores = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])  # All same score

        ef = calculate_enrichment_factor(y_true, y_scores, fraction=0.5)

        # With random ranking, enrichment should be around 1.0
        assert 0.8 <= ef <= 1.2


class TestIntegrationScenarios:
    """Integration test scenarios for metrics functionality."""

    def test_comprehensive_classification_evaluation(self):
        """Test complete classification evaluation workflow."""
        # Simulated binary classification results
        np.random.seed(42)
        n_samples = 100
        y_true = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
        y_prob = np.random.random(n_samples)
        y_pred = (y_prob > 0.5).astype(int)

        # Test comprehensive metrics
        metrics = ClassificationMetrics.calculate_all_metrics(y_true, y_pred, y_prob)

        assert all(
            key in metrics for key in ["accuracy", "precision", "recall", "f1_score"]
        )
        assert all(0.0 <= value <= 1.0 for value in metrics.values())

        # Test confusion matrix
        cm_metrics = ClassificationMetrics.confusion_matrix_metrics(y_true, y_pred)
        assert "confusion_matrix" in cm_metrics

        # Test utility function
        util_metrics = evaluate_model_performance(
            y_true, y_pred, task_type="classification", y_prob=y_prob
        )
        assert len(util_metrics) > 0

    def test_comprehensive_regression_evaluation(self):
        """Test complete regression evaluation workflow."""
        # Simulated regression results
        np.random.seed(42)
        y_true = np.random.normal(5, 2, 50)
        noise = np.random.normal(0, 0.5, 50)
        y_pred = y_true + noise

        # Test comprehensive metrics
        metrics = RegressionMetrics.calculate_all_metrics(y_true, y_pred)

        expected_keys = [
            "mse",
            "rmse",
            "mae",
            "r2",
            "mape",
            "max_error",
            "explained_variance",
        ]
        assert all(key in metrics for key in expected_keys)

        # Test utility function
        util_metrics = evaluate_model_performance(
            y_true, y_pred, task_type="regression"
        )
        assert len(util_metrics) > 0

    def test_molecular_similarity_analysis(self):
        """Test molecular similarity and diversity analysis."""
        # Common molecules for testing
        molecules = [
            "CCO",  # Ethanol
            "CCCO",  # Propanol
            "CCCCO",  # Butanol
            "CC(C)O",  # Isopropanol
            "CC(O)C",  # Acetone
        ]

        # Test pairwise similarities
        similarities = []
        for i in range(len(molecules)):
            for j in range(i + 1, len(molecules)):
                sim = MolecularMetrics.tanimoto_similarity(molecules[i], molecules[j])
                similarities.append(sim)
                assert 0.0 <= sim <= 1.0

        # Test diversity metrics
        diversity = MolecularMetrics.diversity_metrics(molecules)

        assert "mean_pairwise_similarity" in diversity
        assert "diversity_index" in diversity
        assert 0.0 <= diversity["mean_pairwise_similarity"] <= 1.0
        assert 0.0 <= diversity["diversity_index"] <= 1.0

    def test_virtual_screening_evaluation(self):
        """Test virtual screening evaluation workflow."""
        # Simulate virtual screening results
        np.random.seed(42)
        n_compounds = 1000
        n_actives = 50

        # Create ground truth (50 actives, 950 inactives)
        y_true = np.zeros(n_compounds)
        active_indices = np.random.choice(n_compounds, n_actives, replace=False)
        y_true[active_indices] = 1

        # Simulate scoring with some enrichment for actives
        y_scores = np.random.random(n_compounds)
        y_scores[active_indices] += np.random.normal(0.3, 0.1, n_actives)
        y_scores = np.clip(y_scores, 0, 1)

        # Test enrichment factors at different fractions
        ef_1 = calculate_enrichment_factor(y_true, y_scores, fraction=0.01)  # Top 1%
        ef_5 = calculate_enrichment_factor(y_true, y_scores, fraction=0.05)  # Top 5%
        ef_10 = calculate_enrichment_factor(y_true, y_scores, fraction=0.10)  # Top 10%

        assert ef_1 >= 0.0
        assert ef_5 >= 0.0
        assert ef_10 >= 0.0

        # Generally, enrichment should decrease as fraction increases (for good model)
        # But we don't enforce this strictly due to randomness in test data

    def test_cross_platform_compatibility(self):
        """Test metrics work with and without optional dependencies."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 1])

        # Test with sklearn available
        with patch("utils.metrics.SKLEARN_AVAILABLE", True):
            metrics_sklearn = ClassificationMetrics.calculate_all_metrics(
                y_true, y_pred
            )

        # Test without sklearn
        with patch("utils.metrics.SKLEARN_AVAILABLE", False):
            metrics_fallback = ClassificationMetrics.calculate_all_metrics(
                y_true, y_pred
            )

        # Both should return metrics, potentially different but valid
        assert len(metrics_sklearn) > 0
        assert len(metrics_fallback) > 0
        assert "accuracy" in metrics_sklearn
        assert "accuracy" in metrics_fallback

    def test_performance_with_large_datasets(self):
        """Test metrics performance with larger datasets."""
        # Large classification dataset
        np.random.seed(42)
        n_samples = 10000
        y_true = np.random.choice([0, 1], size=n_samples)
        y_pred = np.random.choice([0, 1], size=n_samples)

        import time

        start_time = time.time()
        metrics = ClassificationMetrics.calculate_all_metrics(y_true, y_pred)
        end_time = time.time()

        # Should complete in reasonable time (< 1 second)
        assert end_time - start_time < 1.0
        assert len(metrics) > 0

        # Large regression dataset
        y_true_reg = np.random.normal(0, 1, n_samples)
        y_pred_reg = y_true_reg + np.random.normal(0, 0.1, n_samples)

        start_time = time.time()
        metrics_reg = RegressionMetrics.calculate_all_metrics(y_true_reg, y_pred_reg)
        end_time = time.time()

        assert end_time - start_time < 1.0
        assert len(metrics_reg) > 0


if __name__ == "__main__":
    pytest.main([__file__])
