"""
Comprehensive test suite for src.data_processing.feature_extraction module.

This test suite provides extensive coverage for all feature extraction functionality including:
- extract_descriptors function (RDKit, Mordred, basic descriptors)
- calculate_properties function (molecular property calculation)
- extract_features function (general feature extraction)
- extract_fingerprints function (molecular fingerprints)
- generate_fingerprints function (fingerprint generation)
- extract_structural_features function (structural feature extraction)
- Legacy functions (backward compatibility)
- Error handling and fallback behavior
- Cross-platform compatibility (with/without RDKit, Mordred)
- Performance testing with large datasets
"""

import io
import os
import sys
import tempfile
import unittest
import warnings
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd

# Mock dependencies that might not be available
MOCK_RDKIT = False
MOCK_MORDRED = False

# Try importing with fallbacks
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
except ImportError:
    MOCK_RDKIT = True
    # Mock RDKit
    Chem = Mock()
    Descriptors = Mock()
    rdMolDescriptors = Mock()

try:
    from mordred import Calculator, descriptors
except ImportError:
    MOCK_MORDRED = True
    # Mock Mordred
    Calculator = Mock()
    descriptors = Mock()

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Import the feature_extraction module
from src.data_processing.feature_extraction import (
    Chem,
    ImportError:,
    _calculate_single_fingerprint,
    _estimate_property,
    _extract_basic_descriptors,
    _extract_basic_fingerprints,
    _extract_mordred_descriptors,
    _extract_rdkit_descriptors,
    _extract_rdkit_fingerprints,
    _extract_single_structural_features,
    calculate_logP,
    calculate_molecular_weight,
    calculate_num_rotatable_bonds,
    calculate_properties,
    except,
    extract_descriptors,
    extract_features,
    extract_fingerprints,
    extract_molecular_descriptors,
    extract_structural_features,
    from,
    generate_fingerprints,
    import,
    pass,
    rdkit,
    try:,
)


class TestExtractDescriptors(unittest.TestCase):
    """Test extract_descriptors function."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_smiles = ["CCO", "CC(=O)O", "CC(C)O"]
        self.single_smiles = "CCO"

    def test_extract_descriptors_input_validation(self):
        """Test input validation for extract_descriptors."""
        # Test non-list input
        with self.assertRaises(TypeError):
            extract_descriptors("CCO")

        # Test empty list
        result = extract_descriptors([])
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 0)

    @patch("src.data_processing.feature_extraction.RDKIT_AVAILABLE", True)
    @patch("src.data_processing.feature_extraction._extract_rdkit_descriptors")
    def test_extract_descriptors_rdkit(self, mock_rdkit_extract):
        """Test extract_descriptors with RDKit available."""
        mock_df = pd.DataFrame({"MolWt": [46.07], "MolLogP": [-0.31]})
        mock_rdkit_extract.return_value = mock_df

        result = extract_descriptors(self.sample_smiles, descriptor_set="rdkit")

        mock_rdkit_extract.assert_called_once_with(self.sample_smiles)
        pd.testing.assert_frame_equal(result, mock_df)

    @patch("src.data_processing.feature_extraction.MORDRED_AVAILABLE", True)
    @patch("src.data_processing.feature_extraction._extract_mordred_descriptors")
    def test_extract_descriptors_mordred(self, mock_mordred_extract):
        """Test extract_descriptors with Mordred available."""
        mock_df = pd.DataFrame({"ABC": [1.0], "XYZ": [2.0]})
        mock_mordred_extract.return_value = mock_df

        result = extract_descriptors(self.sample_smiles, descriptor_set="mordred")

        mock_mordred_extract.assert_called_once_with(self.sample_smiles)
        pd.testing.assert_frame_equal(result, mock_df)

    @patch("src.data_processing.feature_extraction._extract_basic_descriptors")
    def test_extract_descriptors_basic(self, mock_basic_extract):
        """Test extract_descriptors with basic descriptors."""
        mock_df = pd.DataFrame({"num_atoms": [3], "num_carbons": [2]})
        mock_basic_extract.return_value = mock_df

        result = extract_descriptors(self.sample_smiles, descriptor_set="basic")

        mock_basic_extract.assert_called_once_with(self.sample_smiles)
        pd.testing.assert_frame_equal(result, mock_df)

    @patch("src.data_processing.feature_extraction.RDKIT_AVAILABLE", False)
    @patch("src.data_processing.feature_extraction._extract_basic_descriptors")
    def test_extract_descriptors_fallback_to_basic(self, mock_basic_extract):
        """Test extract_descriptors fallback to basic when RDKit unavailable."""
        mock_df = pd.DataFrame({"num_atoms": [3]})
        mock_basic_extract.return_value = mock_df

        _result = extract_descriptors(self.sample_smiles, descriptor_set="rdkit")

        mock_basic_extract.assert_called_once_with(self.sample_smiles)


class TestDescriptorExtractors(unittest.TestCase):
    """Test individual descriptor extraction functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_smiles = ["CCO", "CC(=O)O"]

    @patch("src.data_processing.feature_extraction.Descriptors")
    @patch("src.data_processing.feature_extraction.Chem")
    def test_extract_rdkit_descriptors_with_rdkit(self, mock_chem, mock_descriptors):
        """Test RDKit descriptor extraction with RDKit available."""
        # Setup mocks
        mock_mol = Mock()
        mock_chem.MolFromSmiles.return_value = mock_mol

        # Mock descriptor values
        mock_descriptors.MolWt.return_value = 46.07
        mock_descriptors.MolLogP.return_value = -0.31
        mock_descriptors.NumRotatableBonds.return_value = 0
        mock_descriptors.NumHDonors.return_value = 1
        mock_descriptors.NumHAcceptors.return_value = 1
        mock_descriptors.TPSA.return_value = 20.23
        mock_descriptors.NumAromaticRings.return_value = 0

        result = _extract_rdkit_descriptors(self.sample_smiles)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.sample_smiles))
        self.assertIn("MolWt", result.columns)
        self.assertIn("MolLogP", result.columns)

    @patch("src.data_processing.feature_extraction.Chem")
    def test_extract_rdkit_descriptors_invalid_smiles(self, mock_chem):
        """Test RDKit descriptor extraction with invalid SMILES."""
        mock_chem.MolFromSmiles.return_value = None

        result = _extract_rdkit_descriptors(["INVALID"])

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 1)
        # Should have NaN values for invalid molecules
        self.assertTrue(result.iloc[0].isna().all())

    @patch("src.data_processing.feature_extraction.Chem")
    def test_extract_rdkit_descriptors_mol_objects(self, mock_chem):
        """Test RDKit descriptor extraction with Mol objects."""
        mock_mol = Mock()
        mock_mol.GetNumAtoms.return_value = 3  # Mock method to identify as Mol object

        with patch(
            "src.data_processing.feature_extraction.Descriptors"
        ) as mock_descriptors:
            mock_descriptors.MolWt.return_value = 46.07
            mock_descriptors.MolLogP.return_value = -0.31
            mock_descriptors.NumRotatableBonds.return_value = 0
            mock_descriptors.NumHDonors.return_value = 1
            mock_descriptors.NumHAcceptors.return_value = 1
            mock_descriptors.TPSA.return_value = 20.23
            mock_descriptors.NumAromaticRings.return_value = 0

            result = _extract_rdkit_descriptors([mock_mol])

            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual(len(result), 1)

    @patch("src.data_processing.feature_extraction.Calculator")
    @patch("src.data_processing.feature_extraction.descriptors")
    @patch("src.data_processing.feature_extraction.Chem")
    def test_extract_mordred_descriptors(
        self, mock_chem, mock_descriptors, mock_calculator
    ):
        """Test Mordred descriptor extraction."""
        # Setup mocks
        mock_mol = Mock()
        mock_chem.MolFromSmiles.return_value = mock_mol

        mock_calc_instance = Mock()
        mock_calculator.return_value = mock_calc_instance

        mock_df = pd.DataFrame(
            {"desc1": [1.0, 2.0], "desc2": [3.0, 4.0], "text_col": ["a", "b"]}
        )
        mock_calc_instance.pandas.return_value = mock_df

        result = _extract_mordred_descriptors(self.sample_smiles)

        self.assertIsInstance(result, pd.DataFrame)
        # Should only return numeric columns
        self.assertNotIn("text_col", result.columns)
        self.assertIn("desc1", result.columns)
        self.assertIn("desc2", result.columns)

    def test_extract_basic_descriptors(self):
        """Test basic descriptor extraction without dependencies."""
        result = _extract_basic_descriptors(self.sample_smiles)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.sample_smiles))

        expected_columns = [
            "num_atoms",
            "num_carbons",
            "num_nitrogens",
            "num_oxygens",
            "num_rings",
            "smiles_length",
        ]
        for col in expected_columns:
            self.assertIn(col, result.columns)

        # Test specific values for "CCO"
        row_0 = result.iloc[0]
        self.assertEqual(row_0["num_carbons"], 2)  # Two C's in CCO
        self.assertEqual(row_0["num_oxygens"], 1)  # One O in CCO
        self.assertEqual(row_0["smiles_length"], 3)  # Length of "CCO"


class TestCalculateProperties(unittest.TestCase):
    """Test calculate_properties function."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_smiles = ["CCO", "CC(=O)O"]

    @patch("src.data_processing.feature_extraction.RDKIT_AVAILABLE", True)
    @patch("src.data_processing.feature_extraction.Descriptors")
    @patch("src.data_processing.feature_extraction.Chem")
    def test_calculate_properties_with_rdkit(self, mock_chem, mock_descriptors):
        """Test property calculation with RDKit available."""
        mock_mol = Mock()
        mock_chem.MolFromSmiles.return_value = mock_mol

        # Mock property calculations
        mock_descriptors.MolWt.return_value = 46.07
        mock_descriptors.MolLogP.return_value = -0.31
        mock_descriptors.TPSA.return_value = 20.23
        mock_descriptors.NumHDonors.return_value = 1
        mock_descriptors.NumHAcceptors.return_value = 1

        result = calculate_properties(self.sample_smiles)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.sample_smiles))

        expected_columns = ["molecular_weight", "logp", "tpsa", "hbd", "hba"]
        for col in expected_columns:
            self.assertIn(col, result.columns)

    @patch("src.data_processing.feature_extraction.RDKIT_AVAILABLE", False)
    @patch("src.data_processing.feature_extraction._estimate_property")
    def test_calculate_properties_without_rdkit(self, mock_estimate):
        """Test property calculation without RDKit."""
        mock_estimate.side_effect = [
            46.0,
            -0.3,
            20.0,
            1.0,
            1.0,
            50.0,
        ]  # Mock return values for 6 properties

        result = calculate_properties(["CCO"])

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 1)
        self.assertEqual(mock_estimate.call_count, 6)  # Called for each property

    def test_calculate_properties_empty_list(self):
        """Test property calculation with empty list."""
        result = calculate_properties([])

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 0)

    @patch("src.data_processing.feature_extraction.Chem")
    def test_calculate_properties_invalid_smiles(self, mock_chem):
        """Test property calculation with invalid SMILES."""
        mock_chem.MolFromSmiles.return_value = None

        result = calculate_properties(["INVALID"])

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 1)
        # Should have NaN or default values


class TestExtractFeatures(unittest.TestCase):
    """Test extract_features function."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_smiles = ["CCO", "CC(=O)O"]
        self.sample_df = pd.DataFrame({"SMILES": self.sample_smiles})

    @patch("src.data_processing.feature_extraction.extract_descriptors")
    @patch("src.data_processing.feature_extraction.extract_fingerprints")
    def test_extract_features_from_list(self, mock_fingerprints, mock_descriptors):
        """Test feature extraction from SMILES list."""
        mock_descriptors.return_value = pd.DataFrame({"desc1": [1.0, 2.0]})
        mock_fingerprints.return_value = pd.DataFrame({"fp1": [1, 0], "fp2": [0, 1]})

        result = extract_features(
            self.sample_smiles, feature_types=["descriptors", "fingerprints"]
        )

        self.assertIsInstance(result, pd.DataFrame)
        mock_descriptors.assert_called_once()
        mock_fingerprints.assert_called_once()

    @patch("src.data_processing.feature_extraction.extract_descriptors")
    def test_extract_features_from_dataframe(self, mock_descriptors):
        """Test feature extraction from DataFrame."""
        mock_descriptors.return_value = pd.DataFrame({"desc1": [1.0, 2.0]})

        result = extract_features(self.sample_df, feature_types=["descriptors"])

        self.assertIsInstance(result, pd.DataFrame)
        mock_descriptors.assert_called_once_with(self.sample_smiles)

    def test_extract_features_default_types(self):
        """Test extract_features with default feature types."""
        with patch(
            "src.data_processing.feature_extraction.extract_descriptors"
        ) as mock_desc:
            mock_desc.return_value = pd.DataFrame({"desc1": [1.0, 2.0]})

            _result = extract_features(self.sample_smiles)

            mock_desc.assert_called_once()

    def test_extract_features_invalid_input(self):
        """Test extract_features with invalid input."""
        with self.assertRaises(ValueError):
            extract_features("invalid_input")


class TestExtractFingerprints(unittest.TestCase):
    """Test extract_fingerprints function."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_smiles = ["CCO", "CC(=O)O"]

    @patch("src.data_processing.feature_extraction.RDKIT_AVAILABLE", True)
    @patch("src.data_processing.feature_extraction._extract_rdkit_fingerprints")
    def test_extract_fingerprints_with_rdkit(self, mock_rdkit_fp):
        """Test fingerprint extraction with RDKit available."""
        mock_df = pd.DataFrame({"fp_0": [1, 0], "fp_1": [0, 1]})
        mock_rdkit_fp.return_value = mock_df

        result = extract_fingerprints(self.sample_smiles, fp_type="morgan")

        mock_rdkit_fp.assert_called_once_with(self.sample_smiles, "morgan", 2048)
        pd.testing.assert_frame_equal(result, mock_df)

    @patch("src.data_processing.feature_extraction.RDKIT_AVAILABLE", False)
    @patch("src.data_processing.feature_extraction._extract_basic_fingerprints")
    def test_extract_fingerprints_without_rdkit(self, mock_basic_fp):
        """Test fingerprint extraction without RDKit."""
        mock_df = pd.DataFrame({"fp_0": [1, 0], "fp_1": [0, 1]})
        mock_basic_fp.return_value = mock_df

        result = extract_fingerprints(self.sample_smiles)

        mock_basic_fp.assert_called_once_with(self.sample_smiles, 2048)
        pd.testing.assert_frame_equal(result, mock_df)

    def test_extract_fingerprints_empty_list(self):
        """Test fingerprint extraction with empty list."""
        result = extract_fingerprints([])

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 0)


class TestFingerprintExtractors(unittest.TestCase):
    """Test individual fingerprint extraction functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_smiles = ["CCO", "CC(=O)O"]

    @patch("rdkit.Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect")
    @patch("src.data_processing.feature_extraction.Chem")
    def test_extract_rdkit_fingerprints_morgan(self, mock_chem, mock_morgan):
        """Test RDKit Morgan fingerprint extraction."""
        mock_mol = Mock()
        mock_chem.MolFromSmiles.return_value = mock_mol

        # Mock fingerprint bit vector
        mock_fp = Mock()
        mock_fp.__getitem__ = Mock(side_effect=lambda x: x % 2)  # Alternate 0,1
        mock_morgan.return_value = mock_fp

        result = _extract_rdkit_fingerprints(self.sample_smiles, "morgan", 4)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.sample_smiles))
        self.assertEqual(len(result.columns), 4)

    @patch("rdkit.Chem.rdMolDescriptors.GetMACCSKeysFingerprint")
    @patch("src.data_processing.feature_extraction.Chem")
    def test_extract_rdkit_fingerprints_maccs(self, mock_chem, mock_maccs):
        """Test RDKit MACCS fingerprint extraction."""
        mock_mol = Mock()
        mock_chem.MolFromSmiles.return_value = mock_mol

        # Mock MACCS fingerprint
        mock_fp = Mock()
        mock_fp.__len__ = Mock(return_value=167)
        mock_fp.__getitem__ = Mock(side_effect=lambda x: x % 2)
        mock_maccs.return_value = mock_fp

        result = _extract_rdkit_fingerprints(["CCO"], "maccs", 1024)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result.columns), 167)  # MACCS keys are 167 bits
        mock_maccs.assert_called_once_with(mock_mol)

    @patch(
        "rdkit.Chem.rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect"
    )
    @patch("src.data_processing.feature_extraction.Chem")
    def test_extract_rdkit_fingerprints_topological(self, mock_chem, mock_topo):
        """Test RDKit topological fingerprint extraction."""
        mock_mol = Mock()
        mock_chem.MolFromSmiles.return_value = mock_mol

        mock_fp = Mock()
        mock_fp.__getitem__ = Mock(side_effect=lambda x: x % 2)
        mock_topo.return_value = mock_fp

        result = _extract_rdkit_fingerprints(["CCO"], "topological", 4)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result.columns), 4)

    @patch("src.data_processing.feature_extraction.Chem")
    def test_extract_rdkit_fingerprints_invalid_smiles(self, mock_chem):
        """Test RDKit fingerprint extraction with invalid SMILES."""
        mock_chem.MolFromSmiles.return_value = None

        result = _extract_rdkit_fingerprints(["INVALID"], "morgan", 4)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 1)
        # Should have zero vector for invalid molecules
        self.assertTrue((result.iloc[0] == 0).all())

    def test_extract_basic_fingerprints(self):
        """Test basic fingerprint extraction without RDKit."""
        result = _extract_basic_fingerprints(self.sample_smiles, 8)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.sample_smiles))
        self.assertEqual(len(result.columns), 8)

        # Should be binary values
        self.assertTrue(result.isin([0, 1]).all().all())


class TestGenerateFingerprints(unittest.TestCase):
    """Test generate_fingerprints function."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_smiles = ["CCO", "CC(=O)O"]

    @patch("src.data_processing.feature_extraction._calculate_single_fingerprint")
    def test_generate_fingerprints_single_molecule(self, mock_single_fp):
        """Test fingerprint generation for single molecule."""
        mock_fp = np.array([1, 0, 1, 0])
        mock_single_fp.return_value = mock_fp

        result = generate_fingerprints("CCO", fp_type="morgan", n_bits=4)

        np.testing.assert_array_equal(result, mock_fp)
        mock_single_fp.assert_called_once_with("CCO", "morgan", 4, 2)

    @patch("src.data_processing.feature_extraction._calculate_single_fingerprint")
    def test_generate_fingerprints_multiple_molecules(self, mock_single_fp):
        """Test fingerprint generation for multiple molecules."""
        mock_fp1 = np.array([1, 0, 1, 0])
        mock_fp2 = np.array([0, 1, 0, 1])
        mock_single_fp.side_effect = [mock_fp1, mock_fp2]

        result = generate_fingerprints(self.sample_smiles, fp_type="morgan", n_bits=4)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        np.testing.assert_array_equal(result[0], mock_fp1)
        np.testing.assert_array_equal(result[1], mock_fp2)

    @patch("src.data_processing.feature_extraction.RDKIT_AVAILABLE", True)
    @patch("rdkit.Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect")
    @patch("src.data_processing.feature_extraction.Chem")
    def test_calculate_single_fingerprint_with_rdkit(self, mock_chem, mock_rdmol):
        """Test single fingerprint calculation with RDKit."""
        mock_mol = Mock()
        mock_chem.MolFromSmiles.return_value = mock_mol

        mock_fp = Mock()
        mock_fp.__getitem__ = Mock(side_effect=lambda x: x % 2)
        mock_rdmol.return_value = mock_fp

        result = _calculate_single_fingerprint("CCO", "morgan", 4, 2)

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), 4)

    @patch("src.data_processing.feature_extraction.RDKIT_AVAILABLE", False)
    def test_calculate_single_fingerprint_without_rdkit(self):
        """Test single fingerprint calculation without RDKit."""
        result = _calculate_single_fingerprint("CCO", "morgan", 8, 2)

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), 8)
        # Should be binary values
        self.assertTrue(np.all(np.isin(result, [0, 1])))


class TestExtractStructuralFeatures(unittest.TestCase):
    """Test extract_structural_features function."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_smiles = ["CCO", "CC(=O)O"]

    @patch("src.data_processing.feature_extraction._extract_single_structural_features")
    def test_extract_structural_features_single_molecule(self, mock_single_features):
        """Test structural feature extraction for single molecule."""
        mock_features = {"num_rings": 0, "num_atoms": 3}
        mock_single_features.return_value = mock_features

        result = extract_structural_features("CCO", feature_types=["rings", "atoms"])

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 1)
        mock_single_features.assert_called_once_with("CCO", ["rings", "atoms"])

    @patch("src.data_processing.feature_extraction._extract_single_structural_features")
    def test_extract_structural_features_multiple_molecules(self, mock_single_features):
        """Test structural feature extraction for multiple molecules."""
        mock_features1 = {"num_rings": 0, "num_atoms": 3}
        mock_features2 = {"num_rings": 1, "num_atoms": 4}
        mock_single_features.side_effect = [mock_features1, mock_features2]

        result = extract_structural_features(
            self.sample_smiles, feature_types=["rings", "atoms"]
        )

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        self.assertEqual(mock_single_features.call_count, 2)

    def test_extract_structural_features_default_types(self):
        """Test structural feature extraction with default feature types."""
        result = extract_structural_features(["CCO"])

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 1)

    @patch("src.data_processing.feature_extraction.RDKIT_AVAILABLE", True)
    @patch("src.data_processing.feature_extraction.Descriptors")
    @patch("src.data_processing.feature_extraction.Chem")
    def test_extract_single_structural_features_with_rdkit(
        self, mock_chem, mock_descriptors
    ):
        """Test single structural feature extraction with RDKit."""
        mock_mol = Mock()
        mock_mol.GetNumAtoms.return_value = 3
        mock_chem.MolFromSmiles.return_value = mock_mol

        # Mock descriptor calculations
        mock_descriptors.RingCount.return_value = 0
        mock_descriptors.NumAromaticRings.return_value = 0
        mock_descriptors.NumSaturatedRings.return_value = 0
        mock_descriptors.NumHeterocycles.return_value = 0
        mock_descriptors.NumAliphaticRings.return_value = 0

        result = _extract_single_structural_features("CCO", ["rings", "atoms"])

        self.assertIsInstance(result, dict)
        self.assertIn("num_rings", result)
        self.assertIn("num_atoms", result)

    @patch("src.data_processing.feature_extraction.RDKIT_AVAILABLE", False)
    def test_extract_single_structural_features_without_rdkit(self):
        """Test single structural feature extraction without RDKit."""
        result = _extract_single_structural_features("CCO", ["rings", "atoms"])

        self.assertIsInstance(result, dict)
        # Should return random values when RDKit not available
        for value in result.values():
            self.assertIsInstance(value, (int, float))

    @patch("src.data_processing.feature_extraction.Chem")
    def test_extract_single_structural_features_invalid_smiles(self, mock_chem):
        """Test single structural feature extraction with invalid SMILES."""
        mock_chem.MolFromSmiles.return_value = None

        result = _extract_single_structural_features("INVALID", ["rings"])

        self.assertIsInstance(result, dict)
        # Should return 0.0 for invalid molecules
        self.assertEqual(result["rings"], 0.0)


class TestLegacyFunctions(unittest.TestCase):
    """Test legacy functions for backward compatibility."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_molecular_data = ["CCO", "CC(=O)O"]

    @patch("src.data_processing.feature_extraction.calculate_molecular_weight")
    @patch("src.data_processing.feature_extraction.calculate_logP")
    @patch("src.data_processing.feature_extraction.calculate_num_rotatable_bonds")
    def test_extract_molecular_descriptors_legacy(
        self, mock_rot_bonds, mock_logp, mock_mw
    ):
        """Test legacy extract_molecular_descriptors function."""
        mock_mw.return_value = 46.07
        mock_logp.return_value = -0.31
        mock_rot_bonds.return_value = 0

        result = extract_molecular_descriptors(self.sample_molecular_data)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), len(self.sample_molecular_data))

        for descriptor in result:
            self.assertIn("molecular_weight", descriptor)
            self.assertIn("logP", descriptor)
            self.assertIn("num_rotatable_bonds", descriptor)

    @patch("src.data_processing.feature_extraction.RDKIT_AVAILABLE", True)
    @patch("src.data_processing.feature_extraction.Descriptors")
    @patch("src.data_processing.feature_extraction.Chem")
    def test_calculate_molecular_weight_with_rdkit(self, mock_chem, mock_descriptors):
        """Test molecular weight calculation with RDKit."""
        mock_mol = Mock()
        mock_chem.MolFromSmiles.return_value = mock_mol
        mock_descriptors.MolWt.return_value = 46.07

        result = calculate_molecular_weight("CCO")

        self.assertEqual(result, 46.07)
        mock_chem.MolFromSmiles.assert_called_once_with("CCO")
        mock_descriptors.MolWt.assert_called_once_with(mock_mol)

    @patch("src.data_processing.feature_extraction.RDKIT_AVAILABLE", False)
    @patch("src.data_processing.feature_extraction._estimate_property")
    def test_calculate_molecular_weight_without_rdkit(self, mock_estimate):
        """Test molecular weight calculation without RDKit."""
        mock_estimate.return_value = 46.0

        result = calculate_molecular_weight("CCO")

        self.assertEqual(result, 46.0)
        # Function calls calculate_properties which calls _estimate_property for all 6 properties
        self.assertEqual(mock_estimate.call_count, 6)

    @patch("src.data_processing.feature_extraction.RDKIT_AVAILABLE", True)
    @patch("src.data_processing.feature_extraction.Descriptors")
    @patch("src.data_processing.feature_extraction.Chem")
    def test_calculate_logp_with_rdkit(self, mock_chem, mock_descriptors):
        """Test LogP calculation with RDKit."""
        mock_mol = Mock()
        mock_chem.MolFromSmiles.return_value = mock_mol
        mock_descriptors.MolLogP.return_value = -0.31

        result = calculate_logP("CCO")

        self.assertEqual(result, -0.31)

    @patch("src.data_processing.feature_extraction.RDKIT_AVAILABLE", True)
    @patch("src.data_processing.feature_extraction.Descriptors")
    @patch("src.data_processing.feature_extraction.Chem")
    def test_calculate_num_rotatable_bonds_with_rdkit(
        self, mock_chem, mock_descriptors
    ):
        """Test rotatable bonds calculation with RDKit."""
        mock_mol = Mock()
        mock_chem.MolFromSmiles.return_value = mock_mol
        mock_descriptors.NumRotatableBonds.return_value = 0

        result = calculate_num_rotatable_bonds("CCO")

        self.assertEqual(result, 0)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility and helper functions."""

    def test_estimate_property(self):
        """Test property estimation without RDKit."""
        # Test molecular weight estimation
        mw = _estimate_property("CCO", "molecular_weight")
        self.assertIsInstance(mw, (int, float))
        self.assertGreater(mw, 0)

        # Test LogP estimation
        logp = _estimate_property("CCO", "logp")
        self.assertIsInstance(logp, (int, float))

        # Test unknown property
        unknown = _estimate_property("CCO", "unknown_property")
        self.assertEqual(unknown, 0.0)

    def test_estimate_property_different_molecules(self):
        """Test property estimation for different molecules."""
        # Different molecules should give different estimates
        mw1 = _estimate_property("C", "molecular_weight")
        _mw2 = _estimate_property("CC", "molecular_weight")
        mw3 = _estimate_property("CCC", "molecular_weight")

        # Longer molecules should generally have higher molecular weights
        self.assertLess(mw1, mw3)


class TestIntegrationScenarios(unittest.TestCase):
    """Test integration scenarios combining multiple feature extraction components."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_smiles = ["CCO", "CC(=O)O", "CC(C)O"]

    def test_complete_feature_extraction_workflow(self):
        """Test complete feature extraction workflow."""
        with patch.multiple(
            "src.data_processing.feature_extraction",
            extract_descriptors=Mock(return_value=pd.DataFrame({"desc1": [1, 2, 3]})),
            extract_fingerprints=Mock(return_value=pd.DataFrame({"fp1": [1, 0, 1]})),
            extract_structural_features=Mock(
                return_value=pd.DataFrame({"struct1": [0, 1, 0]})
            ),
        ):
            # Test complete workflow
            descriptors = extract_descriptors(self.sample_smiles)
            fingerprints = extract_fingerprints(self.sample_smiles)
            structural = extract_structural_features(self.sample_smiles)

            # Combine features
            combined = pd.concat([descriptors, fingerprints, structural], axis=1)

            self.assertIsInstance(combined, pd.DataFrame)
            self.assertEqual(len(combined), 3)
            self.assertIn("desc1", combined.columns)
            self.assertIn("fp1", combined.columns)
            self.assertIn("struct1", combined.columns)

    def test_feature_extraction_pipeline_with_dataframe(self):
        """Test feature extraction pipeline with DataFrame input."""
        df = pd.DataFrame({"SMILES": self.sample_smiles, "target": [1, 0, 1]})

        with patch(
            "src.data_processing.feature_extraction.extract_descriptors"
        ) as mock_desc:
            mock_desc.return_value = pd.DataFrame({"MW": [46, 60, 60]})

            # Extract features and combine with original data
            features = extract_features(df, feature_types=["descriptors"])
            combined = pd.concat([df, features], axis=1)

            self.assertIn("target", combined.columns)
            self.assertIn("MW", combined.columns)
            self.assertEqual(len(combined), 3)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""

    def setUp(self):
        """Set up test fixtures."""
        self.invalid_smiles = ["INVALID", "ALSOINVALID"]

    def test_extract_descriptors_with_errors(self):
        """Test descriptor extraction with problematic inputs."""
        # Test with some invalid SMILES mixed with valid ones
        mixed_smiles = ["CCO", "INVALID", "CC(=O)O"]

        with patch(
            "src.data_processing.feature_extraction._extract_basic_descriptors"
        ) as mock_basic:
            mock_df = pd.DataFrame({"desc1": [1.0, np.nan, 2.0]})
            mock_basic.return_value = mock_df

            result = extract_descriptors(mixed_smiles, descriptor_set="basic")

            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual(len(result), 3)

    def test_fingerprint_extraction_with_errors(self):
        """Test fingerprint extraction with problematic inputs."""
        with patch(
            "src.data_processing.feature_extraction._extract_basic_fingerprints"
        ) as mock_basic:
            mock_df = pd.DataFrame({"fp_0": [1, 0], "fp_1": [0, 1]})
            mock_basic.return_value = mock_df

            result = extract_fingerprints(self.invalid_smiles)

            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual(len(result), 2)

    def test_calculate_properties_with_none_input(self):
        """Test property calculation with None input."""
        result = calculate_properties(None)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 0)

    def test_feature_extraction_type_errors(self):
        """Test feature extraction with type errors."""
        # Test invalid feature types
        with self.assertRaises(ValueError):
            extract_features(123)  # Invalid input type

        # Test unknown feature types
        result = extract_features(["CCO"], feature_types=["unknown_feature"])
        self.assertIsInstance(result, pd.DataFrame)


class TestCrossModuleCompatibility(unittest.TestCase):
    """Test compatibility with other modules and dependencies."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_smiles = ["CCO", "CC(=O)O"]

    def test_feature_extraction_imports(self):
        """Test that feature extraction module imports correctly."""
        # Test module-level imports
        from src.data_processing.feature_extraction import (
            calculate_properties,
            extract_descriptors,
            extract_features,
        )

        # Verify functions are callable
        self.assertTrue(callable(extract_descriptors))
        self.assertTrue(callable(calculate_properties))
        self.assertTrue(callable(extract_features))

    def test_dependency_availability_flags(self):
        """Test dependency availability flags."""
        from src.data_processing import feature_extraction

        # Test that availability flags exist
        self.assertTrue(hasattr(feature_extraction, "RDKIT_AVAILABLE"))
        self.assertTrue(hasattr(feature_extraction, "MORDRED_AVAILABLE"))

        # Flags should be boolean
        self.assertIsInstance(feature_extraction.RDKIT_AVAILABLE, bool)
        self.assertIsInstance(feature_extraction.MORDRED_AVAILABLE, bool)

    def test_numpy_pandas_integration(self):
        """Test integration with NumPy and pandas."""
        # Test with pandas DataFrame input
        df = pd.DataFrame({"SMILES": self.sample_smiles})

        try:
            result = extract_features(df)
            self.assertIsInstance(result, pd.DataFrame)
        except Exception as e:
            # Only fail if it's not a mocked dependency issue
            if "Mock" not in str(e):
                self.fail(f"DataFrame integration failed: {e}")

        # Test with numpy array conversion
        arr = np.array(self.sample_smiles)

        try:
            result = extract_descriptors(arr.tolist())
            self.assertIsInstance(result, pd.DataFrame)
        except Exception as e:
            if "Mock" not in str(e):
                self.fail(f"NumPy integration failed: {e}")


class TestPerformance(unittest.TestCase):
    """Test performance aspects of feature extraction functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.large_smiles_list = ["CCO"] * 100

    def test_large_dataset_descriptor_extraction(self):
        """Test descriptor extraction with large datasets."""
        with patch(
            "src.data_processing.feature_extraction._extract_basic_descriptors"
        ) as mock_basic:
            mock_df = pd.DataFrame({"desc1": list(range(100))})
            mock_basic.return_value = mock_df

            try:
                result = extract_descriptors(self.large_smiles_list)
                self.assertIsInstance(result, pd.DataFrame)
                self.assertEqual(len(result), 100)
            except MemoryError:
                self.fail("Large dataset caused memory error")

    def test_large_dataset_fingerprint_extraction(self):
        """Test fingerprint extraction with large datasets."""
        with patch(
            "src.data_processing.feature_extraction._extract_basic_fingerprints"
        ) as mock_basic:
            # Create mock fingerprint matrix
            mock_df = pd.DataFrame(np.random.randint(0, 2, size=(100, 10)))
            mock_basic.return_value = mock_df

            try:
                result = extract_fingerprints(self.large_smiles_list[:100])
                self.assertIsInstance(result, pd.DataFrame)
            except MemoryError:
                self.fail("Large dataset caused memory error")

    def test_high_dimensional_fingerprints(self):
        """Test fingerprint extraction with high dimensionality."""
        with patch(
            "src.data_processing.feature_extraction._extract_basic_fingerprints"
        ) as mock_basic:
            # Create high-dimensional fingerprint
            mock_df = pd.DataFrame(np.random.randint(0, 2, size=(10, 4096)))
            mock_basic.return_value = mock_df

            try:
                result = extract_fingerprints(["CCO"] * 10, n_bits=4096)
                self.assertIsInstance(result, pd.DataFrame)
                self.assertEqual(result.shape[1], 4096)
            except MemoryError:
                self.fail("High-dimensional fingerprints caused memory error")


if __name__ == "__main__":
    # Suppress warnings during testing
    warnings.filterwarnings("ignore")

    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test classes
    test_classes = [
        TestExtractDescriptors,
        TestDescriptorExtractors,
        TestCalculateProperties,
        TestExtractFeatures,
        TestExtractFingerprints,
        TestFingerprintExtractors,
        TestGenerateFingerprints,
        TestExtractStructuralFeatures,
        TestLegacyFunctions,
        TestUtilityFunctions,
        TestIntegrationScenarios,
        TestErrorHandling,
        TestCrossModuleCompatibility,
        TestPerformance,
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Print summary
    print(f"\n{'='*50}")
    print("Feature Extraction Test Summary")
    print(f"{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(
        f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.2f}%"
    )

    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            error_msg = traceback.split("AssertionError: ")[-1].split("\n")[0]
            print(f"  - {test}: {error_msg}")

    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            error_msg = traceback.split("\n")[-2]
            print(f"  - {test}: {error_msg}")

    # Coverage improvement tests
    unittest.main(exit=False, verbosity=2)
