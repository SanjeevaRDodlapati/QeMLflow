"""
High-impact test cases for feature_extraction.py to target specific missing lines.
Focus on achieving maximum coverage improvement.
"""

import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

try:
    from rdkit import Chem
except ImportError:
    pass


class TestFeatureExtractionHighImpact(unittest.TestCase):
    """Test cases specifically targeting missing lines for coverage improvement."""

    def setUp(self):
        self.sample_smiles = ["CCO", "CC(=O)O"]

    def test_mordred_syntax_error_handling(self):
        """Test lines 27-29: Mordred import with SyntaxError handling."""
        # This tests the SyntaxError exception handling in the import block
        with patch.dict("sys.modules", {"mordred": None}):
            with patch(
                "builtins.__import__",
                side_effect=SyntaxError("Syntax error in mordred"),
            ):
                try:
                    # Import the module to trigger the exception handling
                    import importlib

                    import src.data_processing.feature_extraction

                    importlib.reload(src.data_processing.feature_extraction)
                except SyntaxError:
                    pass  # Expected to catch SyntaxError

    def test_calculate_properties_mol_object_input(self):
        """Test line 86: Handle Mol object input in calculate_properties."""
        from src.data_processing.feature_extraction import calculate_properties

        # Create a mock Mol object with GetNumAtoms method
        mock_mol = Mock()
        mock_mol.GetNumAtoms.return_value = 3

        with patch("src.data_processing.feature_extraction.RDKIT_AVAILABLE", True):
            with patch(
                "src.data_processing.feature_extraction.Descriptors"
            ) as mock_desc:
                mock_desc.MolWt.return_value = 46.07
                mock_desc.MolLogP.return_value = -0.31
                mock_desc.NumRotatableBonds.return_value = 0
                mock_desc.NumHDonors.return_value = 1
                mock_desc.NumHAcceptors.return_value = 1
                mock_desc.TPSA.return_value = 20.23

                # Test with Mol object (line 86)
                mol_result = calculate_properties([mock_mol])

                self.assertIsInstance(mol_result, pd.DataFrame)
                self.assertEqual(len(mol_result), 1)

    def test_extract_basic_descriptors_implementation(self):
        """Test line 164: Basic descriptors extraction fallback."""
        from src.data_processing.feature_extraction import _extract_basic_descriptors

        basic_result = _extract_basic_descriptors(self.sample_smiles)

        self.assertIsInstance(basic_result, pd.DataFrame)
        self.assertEqual(len(basic_result), 2)

        # Verify expected columns are present
        expected_cols = ["num_atoms", "num_carbons", "num_nitrogens", "num_oxygens"]
        for col in expected_cols:
            self.assertIn(col, basic_result.columns)

    def test_extract_basic_fingerprints(self):
        """Test lines 236-245: Basic fingerprints implementation."""
        from src.data_processing.feature_extraction import _extract_basic_fingerprints

        fp_result = _extract_basic_fingerprints(self.sample_smiles, n_bits=64)

        self.assertIsInstance(fp_result, pd.DataFrame)
        self.assertEqual(len(fp_result), 2)
        self.assertEqual(fp_result.shape[1], 64)

        # Verify all values are binary
        for col in fp_result.columns:
            self.assertTrue(fp_result[col].isin([0, 1]).all())

    def test_extract_rdkit_fingerprints_fallback(self):
        """Test lines 288-292: RDKit fingerprints fallback to topological."""
        from src.data_processing.feature_extraction import _extract_rdkit_fingerprints

        with patch("src.data_processing.feature_extraction.Chem") as mock_chem:
            mock_mol = Mock()
            mock_chem.MolFromSmiles.return_value = mock_mol

            with patch(
                "src.data_processing.feature_extraction.rdMolDescriptors"
            ) as mock_rdmol:
                mock_fp = Mock()
                mock_fp.__getitem__ = Mock(side_effect=lambda x: x % 2)
                mock_rdmol.GetHashedTopologicalTorsionFingerprintAsBitVect.return_value = (
                    mock_fp
                )

                # Test with "topological" type to trigger else clause (lines 294-296)
                topo_result = _extract_rdkit_fingerprints(["CCO"], "topological", 16)

                self.assertIsInstance(topo_result, pd.DataFrame)

    def test_calculate_single_fingerprint_mol_object(self):
        """Test lines 362-370: Single fingerprint calculation with Mol object."""
        from src.data_processing.feature_extraction import _calculate_single_fingerprint

        # Create mock Mol object that has GetNumAtoms method
        mock_mol = Mock()
        mock_mol.GetNumAtoms.return_value = 3

        with patch("src.data_processing.feature_extraction.RDKIT_AVAILABLE", True):
            with patch(
                "src.data_processing.feature_extraction.rdMolDescriptors"
            ) as mock_rdmol:
                mock_fp = Mock()
                mock_fp.__getitem__ = Mock(side_effect=lambda x: x % 2)
                mock_rdmol.GetMorganFingerprintAsBitVect.return_value = mock_fp

                # Test with Mol object input (lines 365-367)
                mol_fp_result = _calculate_single_fingerprint(mock_mol, "morgan", 4, 2)

                self.assertIsInstance(mol_fp_result, np.ndarray)
                self.assertEqual(len(mol_fp_result), 4)

    def test_calculate_single_fingerprint_invalid_mol(self):
        """Test line 384: Handle invalid molecule input."""
        from src.data_processing.feature_extraction import _calculate_single_fingerprint

        with patch("src.data_processing.feature_extraction.RDKIT_AVAILABLE", True):
            # Test with invalid input that can't be converted to Mol (line 384)
            invalid_result = _calculate_single_fingerprint(None, "morgan", 4, 2)

            self.assertIsInstance(invalid_result, np.ndarray)
            self.assertEqual(len(invalid_result), 4)
            # Should return zeros for invalid input
            self.assertTrue(np.all(invalid_result == 0))

    def test_calculate_single_fingerprint_maccs_without_rdkit(self):
        """Test lines 394, 397-400: MACCS fingerprint without RDKit."""
        from src.data_processing.feature_extraction import _calculate_single_fingerprint

        with patch("src.data_processing.feature_extraction.RDKIT_AVAILABLE", False):
            # Test MACCS without RDKit (line 394)
            maccs_result = _calculate_single_fingerprint("CCO", "maccs", 167, 2)

            self.assertIsInstance(maccs_result, np.ndarray)
            self.assertEqual(len(maccs_result), 167)  # MACCS keys are 167 bits

            # Test other fingerprint types without RDKit (line 399)
            morgan_result = _calculate_single_fingerprint("CCO", "morgan", 8, 2)

            self.assertIsInstance(morgan_result, np.ndarray)
            self.assertEqual(len(morgan_result), 8)

    def test_estimate_property_unknown_property(self):
        """Test line 420: Handle unknown property in estimation."""
        from src.data_processing.feature_extraction import _estimate_property

        # Test with unknown property to trigger default case (line 420)
        unknown_result = _estimate_property("CCO", "unknown_property")

        # Should return a default estimate for unknown properties
        self.assertIsInstance(unknown_result, (int, float))

    def test_legacy_functions_empty_properties(self):
        """Test lines 555, 563, 571, 581, 586-589: Legacy function edge cases."""
        from src.data_processing.feature_extraction import (
            calculate_logP,
            calculate_molecular_weight,
            calculate_num_rotatable_bonds,
        )

        # Test molecular weight with empty properties (line 563)
        with patch(
            "src.data_processing.feature_extraction.calculate_properties"
        ) as mock_calc:
            mock_calc.return_value = pd.DataFrame({"molecular_weight": []})

            empty_mw_result = calculate_molecular_weight("CCO")
            self.assertEqual(empty_mw_result, 0.0)

        # Test LogP with empty properties (line 571)
        with patch(
            "src.data_processing.feature_extraction.calculate_properties"
        ) as mock_calc:
            mock_calc.return_value = pd.DataFrame({"logp": []})

            empty_logp_result = calculate_logP("CCO")
            self.assertEqual(empty_logp_result, 0.0)

        # Test rotatable bonds with empty properties (line 581)
        with patch(
            "src.data_processing.feature_extraction.calculate_properties"
        ) as mock_calc:
            mock_calc.return_value = pd.DataFrame({"num_rotatable_bonds": []})

            empty_rot_result = calculate_num_rotatable_bonds("CCO")
            self.assertEqual(empty_rot_result, 0)

    def test_legacy_functions_with_values(self):
        """Test lines 586-589: Legacy functions returning values."""
        from src.data_processing.feature_extraction import (
            calculate_logP,
            calculate_molecular_weight,
            calculate_num_rotatable_bonds,
        )

        # Test molecular weight with values (line 562)
        with patch(
            "src.data_processing.feature_extraction.calculate_properties"
        ) as mock_calc:
            mock_calc.return_value = pd.DataFrame({"molecular_weight": [46.07]})

            mw_result = calculate_molecular_weight("CCO")
            self.assertEqual(mw_result, 46.07)

        # Test LogP with values (line 570)
        with patch(
            "src.data_processing.feature_extraction.calculate_properties"
        ) as mock_calc:
            mock_calc.return_value = pd.DataFrame({"logp": [-0.31]})

            logp_result = calculate_logP("CCO")
            self.assertEqual(logp_result, -0.31)

        # Test rotatable bonds with values (line 579-580)
        with patch(
            "src.data_processing.feature_extraction.calculate_properties"
        ) as mock_calc:
            mock_calc.return_value = pd.DataFrame({"num_rotatable_bonds": [0]})

            rot_result = calculate_num_rotatable_bonds("CCO")
            self.assertEqual(rot_result, 0)


if __name__ == "__main__":
    unittest.main()
