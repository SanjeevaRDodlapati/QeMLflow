"""
Surgical tests to hit specific missing lines in feature_extraction.py for maximum coverage gains.
"""

import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd


class TestFeatureExtractionSurgical(unittest.TestCase):
    """Surgical tests targeting highest-impact missing lines."""

    def test_line_86_mol_object_handling(self):
        """Test line 86: mol object handling in calculate_properties."""
        from src.data_processing.feature_extraction import calculate_properties

        # Create a mock that behaves like a Mol object
        mock_mol = Mock()
        mock_mol.GetNumAtoms.return_value = 3  # This makes hasattr return True

        with patch("src.data_processing.feature_extraction.RDKIT_AVAILABLE", True):
            with patch(
                "src.data_processing.feature_extraction.Chem.MolFromSmiles"
            ) as mock_from_smiles:
                with patch(
                    "src.data_processing.feature_extraction.Descriptors"
                ) as mock_desc:
                    # Configure mocks
                    mock_desc.MolWt.return_value = 46.07
                    mock_desc.MolLogP.return_value = -0.31
                    mock_desc.NumRotatableBonds.return_value = 0
                    mock_desc.NumHDonors.return_value = 1
                    mock_desc.NumHAcceptors.return_value = 1
                    mock_desc.TPSA.return_value = 20.23

                    # Test with mol object - this should hit line 86
                    result = calculate_properties([mock_mol])

                    self.assertIsInstance(result, pd.DataFrame)
                    self.assertEqual(len(result), 1)

    def test_line_164_basic_descriptors(self):
        """Test line 164: basic descriptors implementation."""
        from src.data_processing.feature_extraction import _extract_basic_descriptors

        # This function should always work and hit line 164
        result = _extract_basic_descriptors(["CCO", "CC"])

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        self.assertIn("num_atoms", result.columns)

    def test_lines_236_245_basic_fingerprints(self):
        """Test lines 236-245: basic fingerprints implementation."""
        from src.data_processing.feature_extraction import _extract_basic_fingerprints

        # This should hit the basic fingerprints implementation
        result = _extract_basic_fingerprints(["CCO"], n_bits=32)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape[1], 32)

    def test_line_420_unknown_property(self):
        """Test line 420: unknown property estimation."""
        from src.data_processing.feature_extraction import _estimate_property

        # Test with unknown property - should hit line 420
        result = _estimate_property("CCO", "unknown_property")

        self.assertIsInstance(result, (int, float))

    def test_lines_563_571_581_legacy_empty(self):
        """Test lines 563, 571, 581: legacy functions with empty data."""
        from src.data_processing.feature_extraction import (
            calculate_logP,
            calculate_molecular_weight,
            calculate_num_rotatable_bonds,
        )

        # Mock empty results to hit the else branches
        with patch(
            "src.data_processing.feature_extraction.calculate_properties"
        ) as mock_calc:
            # Empty molecular_weight - hits line 563
            mock_calc.return_value = pd.DataFrame(
                {"molecular_weight": pd.Series([], dtype=float)}
            )
            result_mw = calculate_molecular_weight("CCO")
            self.assertEqual(result_mw, 0.0)

            # Empty logp - hits line 571
            mock_calc.return_value = pd.DataFrame({"logp": pd.Series([], dtype=float)})
            result_logp = calculate_logP("CCO")
            self.assertEqual(result_logp, 0.0)

            # Empty rotatable bonds - hits line 581
            mock_calc.return_value = pd.DataFrame(
                {"num_rotatable_bonds": pd.Series([], dtype=int)}
            )
            result_rot = calculate_num_rotatable_bonds("CCO")
            self.assertEqual(result_rot, 0)


if __name__ == "__main__":
    unittest.main()
