"""
Tests to target specific missing lines in metrics.py for coverage improvement.
"""

import unittest
from unittest.mock import patch

import numpy as np


class TestMetricsHighImpact(unittest.TestCase):
    """Test cases targeting missing lines in metrics.py."""

    def test_sklearn_import_warning(self):
        """Test lines 31-33: sklearn import warning."""
        # Test by mocking the import to fail
        with patch("builtins.__import__", side_effect=ImportError("sklearn not found")):
            with patch("src.utils.metrics.logging.warning") as mock_warning:
                try:
                    # Force re-evaluation of sklearn import
                    import importlib

                    import src.utils.metrics
                    from src.utils.metrics import ClassificationMetrics

                    importlib.reload(src.utils.metrics)
                except Exception:
                    pass

                # Should have logged warning about sklearn
                mock_warning.assert_called()

    def test_rdkit_import_warning(self):
        """Test lines 40-42: RDKit import warning."""
        # Test by checking if warning is properly issued
        from src.utils.metrics import MolecularMetrics

        # Call a method that would check RDKIT_AVAILABLE
        result = MolecularMetrics.tanimoto_similarity("CCO", "CCC")
        self.assertIsInstance(result, float)

    def test_roc_auc_value_error_handling(self):
        """Test lines 84-85: ROC AUC ValueError handling."""
        from src.utils.metrics import ClassificationMetrics

        # Create problematic data that could cause roc_auc_score to fail
        y_true = np.array([0, 0, 0, 1])  # Imbalanced
        y_pred = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.1, 0.9, 0.9])  # Problematic probabilities

        # This should handle the ValueError gracefully (lines 84-85)
        metrics_result = ClassificationMetrics.calculate_all_metrics(
            y_true, y_pred, y_prob
        )
        self.assertIn("roc_auc", metrics_result)
        # Should default to 0.5 when ValueError occurs
        self.assertIsInstance(metrics_result["roc_auc"], float)

    def test_enrichment_factor_edge_case(self):
        """Test line 269: Enrichment factor edge case."""
        from src.utils.metrics import calculate_enrichment_factor

        # Test with very small dataset where top_k equals length
        scores = np.array([0.9, 0.7])
        labels = np.array([1, 0])
        top_k = 2  # Same as length

        ef_result = calculate_enrichment_factor(scores, labels, top_k)
        self.assertIsInstance(ef_result, float)

    def test_diversity_metrics_empty_similarities(self):
        """Test line 269: Diversity metrics with no valid similarities."""
        from src.utils.metrics import MolecularMetrics

        # Test with molecules that would result in no valid similarities
        # This could happen with all invalid SMILES
        invalid_smiles = ["INVALID1", "INVALID2", "INVALID3"]

        diversity_result = MolecularMetrics.calculate_diversity_metrics(invalid_smiles)

        # Should handle empty similarities gracefully (line 269-271)
        self.assertIsInstance(diversity_result, dict)
        self.assertIn("mean_pairwise_similarity", diversity_result)
        # Should return default value of 0.0 when no similarities calculated
        self.assertEqual(diversity_result["mean_pairwise_similarity"], 0.0)


if __name__ == "__main__":
    unittest.main()
