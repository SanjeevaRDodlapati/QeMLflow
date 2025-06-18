#!/usr/bin/env python3
"""
Additional comprehensive tests for molecular_utils functionality.

Tests cover descriptor calculation, fingerprint generation, and molecular operations.
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from qemlflow.core.utils.molecular_utils import (
    Chem,
    LipinskiFilter,
    MolecularDescriptors,
    MolecularVisualization,
    SimilarityCalculator,
    StructuralAlerts,
    calculate_molecular_properties,
    generate_conformers,
    rdkit,
    standardize_smiles,
    validate_molecule,
)


class TestMolecularDescriptors(unittest.TestCase):
    """Test cases for MolecularDescriptors class."""

    def setUp(self):
        try:
            self.descriptors = MolecularDescriptors()
            self.available = True
        except ImportError:
            self.available = False

    @unittest.skipIf(
        not hasattr(MolecularDescriptors, "__init__"),
        "MolecularDescriptors not available",
    )
    def test_calculate_basic_descriptors(self):
        """Test basic descriptor calculation."""
        if not self.available:
            self.skipTest("RDKit not available")

        # Use mock if RDKit not available
        with patch("qemlflow.core.utils.molecular_utils.RDKIT_AVAILABLE", True):
            with patch("qemlflow.core.utils.molecular_utils.Chem.MolFromSmiles") as mock_mol:
                with patch("qemlflow.core.utils.molecular_utils.Descriptors") as mock_desc:
                    # Setup mock
                    mock_mol.return_value = MagicMock()
                    mock_desc.MolWt.return_value = 180.16
                    mock_desc.MolLogP.return_value = -0.74
                    mock_desc.NumHDonors.return_value = 5
                    mock_desc.NumHAcceptors.return_value = 6
                    mock_desc.TPSA.return_value = 110.38
                    mock_desc.NumRotatableBonds.return_value = 5
                    mock_desc.NumAromaticRings.return_value = 0
                    mock_desc.HeavyAtomCount.return_value = 12

                    # Test
                    mol = mock_mol.return_value
                    descriptors = MolecularDescriptors.calculate_basic_descriptors(mol)

                    self.assertIsInstance(descriptors, dict)
                    expected_keys = [
                        "molecular_weight",
                        "logp",
                        "hbd",
                        "hba",
                        "tpsa",
                        "rotatable_bonds",
                        "aromatic_rings",
                        "heavy_atoms",
                    ]
                    for key in expected_keys:
                        self.assertIn(key, descriptors)

    @unittest.skipIf(
        not hasattr(MolecularDescriptors, "__init__"),
        "MolecularDescriptors not available",
    )
    def test_calculate_lipinski_descriptors(self):
        """Test Lipinski descriptor calculation."""
        if not self.available:
            self.skipTest("RDKit not available")

        with patch("qemlflow.core.utils.molecular_utils.RDKIT_AVAILABLE", True):
            with patch("qemlflow.core.utils.molecular_utils.Chem.MolFromSmiles") as mock_mol:
                with patch("qemlflow.core.utils.molecular_utils.Descriptors") as mock_desc:
                    with patch("qemlflow.core.utils.molecular_utils.Lipinski") as mock_lipinski:
                        # Setup mock
                        mock_mol.return_value = MagicMock()
                        mock_desc.MolWt.return_value = 180.16
                        mock_desc.MolLogP.return_value = -0.74
                        mock_lipinski.NumHDonors.return_value = 5
                        mock_lipinski.NumHAcceptors.return_value = 6

                        # Test
                        mol = mock_mol.return_value
                        descriptors = (
                            MolecularDescriptors.calculate_lipinski_descriptors(mol)
                        )

                        self.assertIsInstance(descriptors, dict)
                        expected_keys = ["mw", "logp", "hbd", "hba"]
                        for key in expected_keys:
                            self.assertIn(key, descriptors)

    @unittest.skipIf(
        not hasattr(MolecularDescriptors, "__init__"),
        "MolecularDescriptors not available",
    )
    def test_calculate_morgan_fingerprint(self):
        """Test Morgan fingerprint calculation."""
        if not self.available:
            self.skipTest("RDKit not available")

        with patch("qemlflow.core.utils.molecular_utils.RDKIT_AVAILABLE", True):
            with patch(
                "qemlflow.core.utils.molecular_utils.rdFingerprintGenerator"
            ) as mock_fp_gen:
                # Setup mock
                mock_gen = MagicMock()
                mock_fp = MagicMock()
                mock_fp.__array__ = MagicMock(return_value=np.zeros(2048))
                mock_gen.GetFingerprint.return_value = mock_fp
                mock_fp_gen.GetMorganGenerator.return_value = mock_gen

                # Test
                mol = MagicMock()
                fp = MolecularDescriptors.calculate_morgan_fingerprint(mol)

                self.assertIsInstance(fp, np.ndarray)
                self.assertEqual(len(fp), 2048)


class TestLipinskiFilter(unittest.TestCase):
    """Test cases for LipinskiFilter class."""

    def setUp(self):
        try:
            self.filter = LipinskiFilter()
            self.available = True
        except ImportError:
            self.available = False

    @unittest.skipIf(
        not hasattr(LipinskiFilter, "__init__"), "LipinskiFilter not available"
    )
    def test_passes_lipinski_good_molecule(self):
        """Test Lipinski filter with drug-like molecule."""
        if not self.available:
            self.skipTest("RDKit not available")

        with patch("qemlflow.core.utils.molecular_utils.RDKIT_AVAILABLE", True):
            with patch(
                "qemlflow.core.utils.molecular_utils.MolecularDescriptors.calculate_lipinski_descriptors"
            ) as mock_calc:
                # Setup mock for drug-like molecule
                mock_calc.return_value = {"mw": 250.0, "logp": 2.5, "hbd": 2, "hba": 4}

                mol = MagicMock()
                result = self.filter.passes_lipinski(mol)
                self.assertTrue(result)

    @unittest.skipIf(
        not hasattr(LipinskiFilter, "__init__"), "LipinskiFilter not available"
    )
    def test_passes_lipinski_bad_molecule(self):
        """Test Lipinski filter with non-drug-like molecule."""
        if not self.available:
            self.skipTest("RDKit not available")

        with patch("qemlflow.core.utils.molecular_utils.RDKIT_AVAILABLE", True):
            with patch(
                "qemlflow.core.utils.molecular_utils.MolecularDescriptors.calculate_lipinski_descriptors"
            ) as mock_calc:
                # Setup mock for non-drug-like molecule
                mock_calc.return_value = {
                    "mw": 700.0,  # Too high
                    "logp": 8.0,  # Too high
                    "hbd": 8,  # Too high
                    "hba": 15,  # Too high
                }

                mol = MagicMock()
                result = self.filter.passes_lipinski(mol)
                self.assertFalse(result)


class TestStructuralAlerts(unittest.TestCase):
    """Test cases for StructuralAlerts class."""

    def setUp(self):
        try:
            from qemlflow.core.utils.molecular_utils import StructuralAlerts

            self.alerts = StructuralAlerts()
            self.available = True
        except (ImportError, AttributeError):
            self.available = False

    @unittest.skipIf(
        not hasattr(StructuralAlerts, "__init__"), "StructuralAlerts not available"
    )
    def test_check_pains_alerts(self):
        """Test PAINS alerts checking."""
        if not self.available:
            self.skipTest("StructuralAlerts not available")

        with patch("qemlflow.core.utils.molecular_utils.RDKIT_AVAILABLE", True):
            # Test with mock molecule
            mol = MagicMock()
            alerts = self.alerts.check_pains_alerts(mol)
            self.assertIsInstance(alerts, list)

    @unittest.skipIf(
        not hasattr(StructuralAlerts, "__init__"), "StructuralAlerts not available"
    )
    def test_check_brenk_alerts(self):
        """Test Brenk alerts checking."""
        if not self.available:
            self.skipTest("StructuralAlerts not available")

        with patch("qemlflow.core.utils.molecular_utils.RDKIT_AVAILABLE", True):
            # Test with mock molecule
            mol = MagicMock()
            alerts = self.alerts.check_brenk_alerts(mol)
            self.assertIsInstance(alerts, list)


class TestSimilarityCalculator(unittest.TestCase):
    """Test cases for SimilarityCalculator class."""

    def setUp(self):
        try:
            from qemlflow.core.utils.molecular_utils import SimilarityCalculator

            self.calculator = SimilarityCalculator()
            self.available = True
        except (ImportError, AttributeError):
            self.available = False

    @unittest.skipIf(
        not hasattr(SimilarityCalculator, "__init__"),
        "SimilarityCalculator not available",
    )
    def test_tanimoto_similarity(self):
        """Test Tanimoto similarity calculation."""
        if not self.available:
            self.skipTest("SimilarityCalculator not available")

        # Test with mock fingerprints
        fp1 = np.array([1, 0, 1, 0, 1])
        fp2 = np.array([1, 1, 0, 0, 1])

        similarity = self.calculator.tanimoto_similarity(fp1, fp2)
        self.assertIsInstance(similarity, float)
        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)

    @unittest.skipIf(
        not hasattr(SimilarityCalculator, "__init__"),
        "SimilarityCalculator not available",
    )
    def test_dice_similarity(self):
        """Test Dice similarity calculation."""
        if not self.available:
            self.skipTest("SimilarityCalculator not available")

        # Test with mock fingerprints
        fp1 = np.array([1, 0, 1, 0, 1])
        fp2 = np.array([1, 1, 0, 0, 1])

        similarity = self.calculator.dice_similarity(fp1, fp2)
        self.assertIsInstance(similarity, float)
        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)


class TestMolecularVisualization(unittest.TestCase):
    """Test cases for MolecularVisualization class."""

    def setUp(self):
        try:
            from qemlflow.core.utils.molecular_utils import MolecularVisualization

            self.viz = MolecularVisualization()
            self.available = True
        except (ImportError, AttributeError):
            self.available = False

    @unittest.skipIf(
        not hasattr(MolecularVisualization, "__init__"),
        "MolecularVisualization not available",
    )
    def test_draw_molecule_2d(self):
        """Test 2D molecule drawing."""
        if not self.available:
            self.skipTest("MolecularVisualization not available")

        with patch("qemlflow.core.utils.molecular_utils.RDKIT_AVAILABLE", True):
            mol = MagicMock()
            # Should not raise an exception
            self.viz.draw_molecule_2d(mol)

    @unittest.skipIf(
        not hasattr(MolecularVisualization, "__init__"),
        "MolecularVisualization not available",
    )
    def test_draw_molecule_3d(self):
        """Test 3D molecule drawing."""
        if not self.available:
            self.skipTest("MolecularVisualization not available")

        with patch("qemlflow.core.utils.molecular_utils.PY3DMOL_AVAILABLE", True):
            mol = MagicMock()
            # Should not raise an exception
            self.viz.draw_molecule_3d(mol)


class TestStandaloneFunctions(unittest.TestCase):
    """Test cases for standalone utility functions."""

    def test_standardize_smiles_basic(self):
        """Test SMILES standardization."""
        # Test with mock
        with patch("qemlflow.core.utils.molecular_utils.RDKIT_AVAILABLE", True):
            with patch("qemlflow.core.utils.molecular_utils.Chem.MolFromSmiles") as mock_mol_from:
                with patch("qemlflow.core.utils.molecular_utils.Chem.MolToSmiles") as mock_mol_to:
                    mock_mol_from.return_value = MagicMock()
                    mock_mol_to.return_value = "CCO"

                    result = standardize_smiles("CCO")
                    self.assertEqual(result, "CCO")

    def test_standardize_smiles_invalid(self):
        """Test SMILES standardization with invalid input."""
        with patch("qemlflow.core.utils.molecular_utils.RDKIT_AVAILABLE", True):
            with patch("qemlflow.core.utils.molecular_utils.Chem.MolFromSmiles") as mock_mol_from:
                mock_mol_from.return_value = None

                result = standardize_smiles("invalid_smiles")
                self.assertIsNone(result)

    def test_calculate_molecular_properties_basic(self):
        """Test molecular properties calculation."""
        with patch("qemlflow.core.utils.molecular_utils.RDKIT_AVAILABLE", True):
            with patch("qemlflow.core.utils.molecular_utils.Chem.MolFromSmiles") as mock_mol_from:
                with patch(
                    "qemlflow.core.utils.molecular_utils.MolecularDescriptors.calculate_basic_descriptors"
                ) as mock_calc:
                    mock_mol_from.return_value = MagicMock()
                    mock_calc.return_value = {"mw": 46.07, "logp": -0.31}

                    result = calculate_molecular_properties("CCO")
                    self.assertIsInstance(result, dict)
                    self.assertIn("mw", result)

    def test_calculate_molecular_properties_invalid(self):
        """Test molecular properties calculation with invalid input."""
        with patch("qemlflow.core.utils.molecular_utils.RDKIT_AVAILABLE", True):
            with patch("qemlflow.core.utils.molecular_utils.Chem.MolFromSmiles") as mock_mol_from:
                mock_mol_from.return_value = None

                result = calculate_molecular_properties("invalid_smiles")
                self.assertIsNone(result)

    def test_generate_conformers_basic(self):
        """Test conformer generation."""
        with patch("qemlflow.core.utils.molecular_utils.RDKIT_AVAILABLE", True):
            with patch("qemlflow.core.utils.molecular_utils.Chem.MolFromSmiles") as mock_mol_from:
                with patch("qemlflow.core.utils.molecular_utils.Chem.AddHs") as mock_add_hs:
                    mock_mol = MagicMock()
                    mock_mol_from.return_value = mock_mol
                    mock_add_hs.return_value = mock_mol

                    result = generate_conformers("CCO", num_conformers=5)
                    self.assertEqual(result, mock_mol)

    def test_generate_conformers_invalid(self):
        """Test conformer generation with invalid input."""
        with patch("qemlflow.core.utils.molecular_utils.RDKIT_AVAILABLE", True):
            with patch("qemlflow.core.utils.molecular_utils.Chem.MolFromSmiles") as mock_mol_from:
                mock_mol_from.return_value = None

                result = generate_conformers("invalid_smiles")
                self.assertIsNone(result)

    def test_validate_molecule_valid(self):
        """Test molecule validation with valid SMILES."""
        with patch("qemlflow.core.utils.molecular_utils.RDKIT_AVAILABLE", True):
            with patch("qemlflow.core.utils.molecular_utils.Chem.MolFromSmiles") as mock_mol_from:
                mock_mol_from.return_value = MagicMock()

                result = validate_molecule("CCO")
                self.assertTrue(result)

    def test_validate_molecule_invalid(self):
        """Test molecule validation with invalid SMILES."""
        with patch("qemlflow.core.utils.molecular_utils.RDKIT_AVAILABLE", True):
            with patch("qemlflow.core.utils.molecular_utils.Chem.MolFromSmiles") as mock_mol_from:
                mock_mol_from.return_value = None

                result = validate_molecule("invalid_smiles")
                self.assertFalse(result)

    @patch("qemlflow.core.utils.molecular_utils.RDKIT_AVAILABLE", False)
    def test_functions_without_rdkit(self):
        """Test functions when RDKit is not available."""
        result = standardize_smiles("CCO")
        self.assertIsNone(result)

        result = calculate_molecular_properties("CCO")
        self.assertIsNone(result)

        result = generate_conformers("CCO")
        self.assertIsNone(result)

        result = validate_molecule("CCO")
        self.assertFalse(result)


class TestErrorHandling(unittest.TestCase):
    """Test error handling in molecular utilities."""

    def test_none_input_handling(self):
        """Test handling of None inputs."""
        result = standardize_smiles(None)
        self.assertIsNone(result)

        result = calculate_molecular_properties(None)
        self.assertIsNone(result)

        result = validate_molecule(None)
        self.assertFalse(result)

    def test_empty_string_handling(self):
        """Test handling of empty string inputs."""
        result = standardize_smiles("")
        self.assertIsNone(result)

        result = calculate_molecular_properties("")
        self.assertIsNone(result)

        result = validate_molecule("")
        self.assertFalse(result)

    def test_malformed_smiles_handling(self):
        """Test handling of various malformed SMILES."""
        malformed_smiles = [
            "C[C@H](C)C",  # Stereochemistry
            "C1=CC=CC=C1",  # Benzene
            "[Na+].[Cl-]",  # Salt
            "CC(=O)O",  # Acetic acid
            "invalid",  # Completely invalid
        ]

        for smiles in malformed_smiles:
            # Should not raise exceptions
            try:
                standardize_smiles(smiles)
                calculate_molecular_properties(smiles)
                validate_molecule(smiles)
            except Exception as e:
                self.fail(f"Exception raised for SMILES {smiles}: {e}")


class TestPerformance(unittest.TestCase):
    """Test performance with larger datasets."""

    def test_batch_smiles_processing(self):
        """Test processing of multiple SMILES."""
        smiles_list = ["CCO", "CCC", "CCCC", "CCCCC"] * 25  # 100 SMILES

        with patch("qemlflow.core.utils.molecular_utils.RDKIT_AVAILABLE", True):
            with patch("qemlflow.core.utils.molecular_utils.Chem.MolFromSmiles") as mock_mol_from:
                mock_mol_from.return_value = MagicMock()

                # Should process efficiently
                results = []
                for smiles in smiles_list:
                    result = validate_molecule(smiles)
                    results.append(result)

                self.assertEqual(len(results), 100)
                self.assertTrue(all(results))  # All should be valid with mock


if __name__ == "__main__":
    unittest.main()
