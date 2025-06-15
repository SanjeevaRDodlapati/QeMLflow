"""
High-impact tests for property_prediction.py to target specific missing lines.
"""
import unittest
from unittest.mock import Mock, patch

import pandas as pd


class TestPropertyPredictionHighImpact(unittest.TestCase):
    """Test cases targeting missing lines in property_prediction.py."""

    def test_predict_properties_type_error_line_349(self):
        """Test line 349: TypeError for invalid molecular_data type."""
        from chemml.research.drug_discovery.property_prediction import (
            predict_properties,
        )

        # Create a mock model
        mock_model = Mock()
        mock_model.predict_multiple_properties.return_value = {"property": [1.0]}

        # Test with invalid data type (not DataFrame or list) - should hit line 349
        with self.assertRaises(TypeError):
            predict_properties(mock_model, "invalid_string_type")

    def test_rdkit_import_warning_lines_24_26(self):
        """Test lines 24-26: RDKit import warning."""
        # Test that the import warning is properly structured
        with patch("builtins.__import__", side_effect=ImportError("RDKit not found")):
            with patch("logging.warning") as mock_warning:
                try:
                    # Force re-import to trigger the warning
                    import importlib

                    import src.drug_design.property_prediction

                    importlib.reload(src.drug_design.property_prediction)
                except Exception:
                    pass

                # Should have called the warning about RDKit
                mock_warning.assert_called()


if __name__ == "__main__":
    unittest.main()
