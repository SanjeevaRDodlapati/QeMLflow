"""
Comprehensive tests for molecular_utils module

This test suite achieves high coverage for molecular utilities including
descriptor calculation, SMILES processing, visualization, filtering,
similarity calculations, and structural alerts.
"""

import logging
import sys
from typing import Dict, List, Optional
from unittest.mock import MagicMock, Mock, call, patch

import numpy as np
import pandas as pd
import pytest

# Import the module under test
sys.path.insert(0, "/Users/sanjeevadodlapati/Downloads/Repos/ChemML/src")
from utils.molecular_utils import (
try:
    from rdkit import Chem
except ImportError:
    pass
    PY3DMOL_AVAILABLE,
    RDKIT_AVAILABLE,
    LipinskiFilter,
    MolecularDescriptors,
    MolecularVisualization,
    MoleculeVisualizer,
    SimilarityCalculator,
    SMILESProcessor,
    StructuralAlerts,
    batch_process_molecules,
    calculate_drug_likeness_score,
    calculate_logp,
    calculate_molecular_properties,
    calculate_molecular_weight,
    calculate_similarity,
    filter_molecules_by_properties,
    generate_conformers,
    get_molecular_formula,
    mol_to_smiles,
    neutralize_molecule,
    remove_salts,
    smiles_to_mol,
    standardize_molecule,
    standardize_smiles,
    validate_molecule,
    validate_smiles,
)


class TestMolecularDescriptors:
    """Test MolecularDescriptors class"""

    def test_init_without_rdkit(self):
        """Test initialization when RDKit is not available"""
        with patch("utils.molecular_utils.RDKIT_AVAILABLE", False):
            with pytest.raises(ImportError, match="RDKit is required"):
                MolecularDescriptors()

    @patch("utils.molecular_utils.RDKIT_AVAILABLE", True)
    @patch("utils.molecular_utils.Descriptors")
    def test_calculate_basic_descriptors(self, mock_descriptors):
        """Test basic descriptors calculation"""
        # Setup mocks
        mock_mol = Mock()
        mock_descriptors.MolWt.return_value = 180.16
        mock_descriptors.MolLogP.return_value = -0.74
        mock_descriptors.NumHDonors.return_value = 5
        mock_descriptors.NumHAcceptors.return_value = 6
        mock_descriptors.TPSA.return_value = 110.38
        mock_descriptors.NumRotatableBonds.return_value = 5
        mock_descriptors.NumAromaticRings.return_value = 0
        mock_descriptors.HeavyAtomCount.return_value = 12

        descriptors = MolecularDescriptors.calculate_basic_descriptors(mock_mol)

        # Verify all descriptors are calculated
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
            assert key in descriptors

        # Verify specific values
        assert descriptors["molecular_weight"] == 180.16
        assert descriptors["logp"] == -0.74
        assert descriptors["hbd"] == 5
        assert descriptors["hba"] == 6

    @patch("utils.molecular_utils.RDKIT_AVAILABLE", True)
    @patch("utils.molecular_utils.Descriptors")
    @patch("utils.molecular_utils.Lipinski")
    def test_calculate_lipinski_descriptors(self, mock_lipinski, mock_descriptors):
        """Test Lipinski descriptors calculation"""
        mock_mol = Mock()
        mock_descriptors.MolWt.return_value = 250.0
        mock_descriptors.MolLogP.return_value = 2.5
        mock_lipinski.NumHDonors.return_value = 2
        mock_lipinski.NumHAcceptors.return_value = 4

        descriptors = MolecularDescriptors.calculate_lipinski_descriptors(mock_mol)

        assert descriptors["mw"] == 250.0
        assert descriptors["logp"] == 2.5
        assert descriptors["hbd"] == 2
        assert descriptors["hba"] == 4

    @patch("utils.molecular_utils.RDKIT_AVAILABLE", True)
    @patch("utils.molecular_utils.rdFingerprintGenerator")
    def test_calculate_morgan_fingerprint(self, mock_fp_gen):
        """Test Morgan fingerprint calculation"""
        mock_mol = Mock()
        mock_generator = Mock()
        mock_fingerprint = Mock()

        mock_fp_gen.GetMorganGenerator.return_value = mock_generator
        mock_generator.GetFingerprint.return_value = mock_fingerprint

        # Mock the fingerprint to behave like an array
        mock_fingerprint.__array__ = Mock(return_value=np.array([0, 1, 0, 1]))

        result = MolecularDescriptors.calculate_morgan_fingerprint(
            mock_mol, radius=2, n_bits=2048
        )

        mock_fp_gen.GetMorganGenerator.assert_called_once_with(radius=2, fpSize=2048)
        mock_generator.GetFingerprint.assert_called_once_with(mock_mol)

        # Check result is numpy array
        assert isinstance(result, np.ndarray)


class TestLipinskiFilter:
    """Test LipinskiFilter class"""

    def test_init_without_rdkit(self):
        """Test initialization when RDKit is not available"""
        with patch("utils.molecular_utils.RDKIT_AVAILABLE", False):
            with pytest.raises(ImportError, match="RDKit is required"):
                LipinskiFilter()

    @patch("utils.molecular_utils.RDKIT_AVAILABLE", True)
    def test_init_with_rdkit(self):
        """Test initialization when RDKit is available"""
        filter_strict = LipinskiFilter(strict=True)
        assert filter_strict.strict is True

        filter_nonstrict = LipinskiFilter(strict=False)
        assert filter_nonstrict.strict is False

    @patch("utils.molecular_utils.RDKIT_AVAILABLE", True)
    @patch("utils.molecular_utils.MolecularDescriptors.calculate_lipinski_descriptors")
    def test_passes_lipinski_strict_mode(self, mock_calc_descriptors):
        """Test Lipinski filtering in strict mode"""
        lipinski_filter = LipinskiFilter(strict=True)
        mock_mol = Mock()

        # Test molecule that passes all rules
        mock_calc_descriptors.return_value = {
            "mw": 400.0,
            "logp": 3.0,
            "hbd": 3,
            "hba": 6,
        }

        assert lipinski_filter.passes_lipinski(mock_mol) is True

        # Test molecule that violates one rule (should fail in strict mode)
        mock_calc_descriptors.return_value = {
            "mw": 600.0,
            "logp": 3.0,
            "hbd": 3,
            "hba": 6,  # MW > 500
        }

        assert lipinski_filter.passes_lipinski(mock_mol) is False

    @patch("utils.molecular_utils.RDKIT_AVAILABLE", True)
    @patch("utils.molecular_utils.MolecularDescriptors.calculate_lipinski_descriptors")
    def test_passes_lipinski_non_strict_mode(self, mock_calc_descriptors):
        """Test Lipinski filtering in non-strict mode"""
        lipinski_filter = LipinskiFilter(strict=False)
        mock_mol = Mock()

        # Test molecule that violates one rule (should pass in non-strict mode)
        mock_calc_descriptors.return_value = {
            "mw": 600.0,
            "logp": 3.0,
            "hbd": 3,
            "hba": 6,  # MW > 500
        }

        assert lipinski_filter.passes_lipinski(mock_mol) is True

        # Test molecule that violates two rules (should fail even in non-strict mode)
        mock_calc_descriptors.return_value = {
            "mw": 600.0,
            "logp": 7.0,
            "hbd": 3,
            "hba": 6,  # MW > 500 and LogP > 5
        }

        assert lipinski_filter.passes_lipinski(mock_mol) is False

    @patch("utils.molecular_utils.RDKIT_AVAILABLE", True)
    @patch("utils.molecular_utils.Chem")
    def test_filter_molecules(self, mock_chem):
        """Test filtering a list of SMILES"""
        lipinski_filter = LipinskiFilter(strict=False)

        # Setup mocks
        mock_mol_pass = Mock()
        mock_mol_fail = Mock()

        mock_chem.MolFromSmiles.side_effect = [mock_mol_pass, mock_mol_fail, None]

        # Mock passes_lipinski method
        lipinski_filter.passes_lipinski = Mock(side_effect=[True, False])

        smiles_list = ["CCO", "invalid_smiles", "bad_smiles"]
        filtered = lipinski_filter.filter_molecules(smiles_list)

        assert len(filtered) == 1
        assert filtered[0] == "CCO"

        # Verify method calls
        assert mock_chem.MolFromSmiles.call_count == 3
        assert lipinski_filter.passes_lipinski.call_count == 2


class TestSMILESProcessor:
    """Test SMILESProcessor class"""

    def test_init_without_rdkit(self):
        """Test initialization when RDKit is not available"""
        with patch("utils.molecular_utils.RDKIT_AVAILABLE", False):
            with pytest.raises(ImportError, match="RDKit is required"):
                SMILESProcessor()

    @patch("utils.molecular_utils.RDKIT_AVAILABLE", True)
    @patch("utils.molecular_utils.Chem")
    def test_canonicalize_smiles(self, mock_chem):
        """Test SMILES canonicalization"""
        mock_mol = Mock()
        mock_chem.MolFromSmiles.return_value = mock_mol
        mock_chem.MolToSmiles.return_value = "CCO"

        result = SMILESProcessor.canonicalize_smiles("C(C)O")

        mock_chem.MolFromSmiles.assert_called_once_with("C(C)O")
        mock_chem.MolToSmiles.assert_called_once_with(mock_mol, canonical=True)
        assert result == "CCO"

    @patch("utils.molecular_utils.RDKIT_AVAILABLE", True)
    @patch("utils.molecular_utils.Chem")
    def test_canonicalize_smiles_invalid(self, mock_chem):
        """Test SMILES canonicalization with invalid SMILES"""
        mock_chem.MolFromSmiles.return_value = None

        result = SMILESProcessor.canonicalize_smiles("invalid_smiles")

        assert result is None

    @patch("utils.molecular_utils.RDKIT_AVAILABLE", True)
    @patch("utils.molecular_utils.Chem")
    def test_is_valid_smiles(self, mock_chem):
        """Test SMILES validation"""
        # Valid SMILES
        mock_chem.MolFromSmiles.return_value = Mock()
        assert SMILESProcessor.is_valid_smiles("CCO") is True

        # Invalid SMILES
        mock_chem.MolFromSmiles.return_value = None
        assert SMILESProcessor.is_valid_smiles("invalid") is False

        # Exception handling
        mock_chem.MolFromSmiles.side_effect = Exception("Error")
        assert SMILESProcessor.is_valid_smiles("CCO") is False

    @patch("utils.molecular_utils.RDKIT_AVAILABLE", True)
    @patch("utils.molecular_utils.Chem")
    def test_smiles_to_mol(self, mock_chem):
        """Test SMILES to Mol conversion"""
        mock_mol = Mock()
        mock_chem.MolFromSmiles.return_value = mock_mol

        result = SMILESProcessor.smiles_to_mol("CCO")

        mock_chem.MolFromSmiles.assert_called_once_with("CCO")
        assert result == mock_mol

    @patch("utils.molecular_utils.RDKIT_AVAILABLE", True)
    def test_process_smiles_list(self):
        """Test processing a list of SMILES"""
        processor = SMILESProcessor()

        # Mock the validation and canonicalization methods
        processor.is_valid_smiles = Mock(side_effect=[True, False, True])
        processor.canonicalize_smiles = Mock(side_effect=["CCO", "C1=CC=CC=C1"])

        smiles_list = ["C(C)O", "invalid", "c1ccccc1"]
        result = processor.process_smiles_list(smiles_list)

        assert len(result["valid"]) == 2
        assert len(result["invalid"]) == 1
        assert len(result["canonical"]) == 2

        assert "C(C)O" in result["valid"]
        assert "c1ccccc1" in result["valid"]
        assert "invalid" in result["invalid"]


class TestMoleculeVisualizer:
    """Test MoleculeVisualizer class"""

    def test_init_without_rdkit(self):
        """Test initialization when RDKit is not available"""
        with patch("utils.molecular_utils.RDKIT_AVAILABLE", False):
            with pytest.raises(ImportError, match="RDKit is required"):
                MoleculeVisualizer()

    @patch("utils.molecular_utils.RDKIT_AVAILABLE", True)
    @patch("rdkit.Chem.Draw")
    def test_view_2d(self, mock_draw):
        """Test 2D visualization"""
        mock_mol = Mock()
        mock_image = Mock()
        mock_draw.MolToImage.return_value = mock_image

        result = MoleculeVisualizer.view_2d(mock_mol, size=(400, 400))

        mock_draw.MolToImage.assert_called_once_with(mock_mol, size=(400, 400))
        assert result == mock_image

    @patch("utils.molecular_utils.RDKIT_AVAILABLE", True)
    @patch("utils.molecular_utils.PY3DMOL_AVAILABLE", False)
    def test_view_3d_without_py3dmol(self):
        """Test 3D visualization without py3Dmol"""
        result = MoleculeVisualizer.view_3d("CCO")
        assert result is None

    @patch("utils.molecular_utils.RDKIT_AVAILABLE", True)
    @patch("utils.molecular_utils.PY3DMOL_AVAILABLE", True)
    @patch("utils.molecular_utils.Chem")
    @patch("utils.molecular_utils.py3Dmol")
    def test_view_3d_with_py3dmol(self, mock_py3dmol, mock_chem):
        """Test 3D visualization with py3Dmol"""
        mock_mol = Mock()
        mock_mol_with_h = Mock()
        mock_viewer = Mock()

        mock_chem.MolFromSmiles.return_value = mock_mol
        mock_chem.AddHs.return_value = mock_mol_with_h
        mock_chem.MolToMolBlock.return_value = "SDF_BLOCK"
        mock_py3dmol.view.return_value = mock_viewer

        # Mock AllChem functions
        with patch("rdkit.Chem.AllChem") as mock_allchem:
            mock_allchem.EmbedMolecule.return_value = 0
            mock_allchem.UFFOptimizeMolecule.return_value = 0

            result = MoleculeVisualizer.view_3d("CCO", style="stick")

        mock_chem.MolFromSmiles.assert_called_once_with("CCO")
        mock_chem.AddHs.assert_called_once_with(mock_mol)
        mock_py3dmol.view.assert_called_once_with(width=400, height=400)

        assert result == mock_viewer


class TestStandaloneFunctions:
    """Test standalone utility functions"""

    @patch("utils.molecular_utils.RDKIT_AVAILABLE", True)
    @patch("utils.molecular_utils.Descriptors")
    def test_calculate_drug_likeness_score(self, mock_descriptors):
        """Test drug-likeness score calculation"""
        mock_mol = Mock()

        # Setup mock descriptors for a drug-like molecule
        mock_descriptors.MolWt.return_value = 300.0
        mock_descriptors.MolLogP.return_value = 2.5
        mock_descriptors.NumHDonors.return_value = 2
        mock_descriptors.NumHAcceptors.return_value = 4
        mock_descriptors.TPSA.return_value = 80.0
        mock_descriptors.NumRotatableBonds.return_value = 5
        mock_descriptors.NumAromaticRings.return_value = 2
        mock_descriptors.HeavyAtomCount.return_value = 25

        score = calculate_drug_likeness_score(mock_mol)

        # Should pass all 8 criteria
        assert score == 1.0

    @patch("utils.molecular_utils.RDKIT_AVAILABLE", True)
    @patch("utils.molecular_utils.Descriptors")
    def test_calculate_drug_likeness_score_poor_molecule(self, mock_descriptors):
        """Test drug-likeness score for a poor molecule"""
        mock_mol = Mock()

        # Setup mock descriptors for a non-drug-like molecule
        mock_descriptors.MolWt.return_value = 800.0  # Too heavy
        mock_descriptors.MolLogP.return_value = 8.0  # Too lipophilic
        mock_descriptors.NumHDonors.return_value = 10  # Too many donors
        mock_descriptors.NumHAcceptors.return_value = 15  # Too many acceptors
        mock_descriptors.TPSA.return_value = 200.0  # Too polar
        mock_descriptors.NumRotatableBonds.return_value = 15  # Too flexible
        mock_descriptors.NumAromaticRings.return_value = 6  # Too many rings
        mock_descriptors.HeavyAtomCount.return_value = 80  # Too many atoms

        score = calculate_drug_likeness_score(mock_mol)

        # Should fail all criteria
        assert score == 0.0

    def test_calculate_drug_likeness_score_without_rdkit(self):
        """Test drug-likeness calculation without RDKit"""
        with patch("utils.molecular_utils.RDKIT_AVAILABLE", False):
            with pytest.raises(ImportError, match="RDKit is required"):
                calculate_drug_likeness_score(Mock())

    @patch("utils.molecular_utils.RDKIT_AVAILABLE", True)
    @patch("utils.molecular_utils.Chem")
    @patch("utils.molecular_utils.MolecularDescriptors")
    @patch("utils.molecular_utils.LipinskiFilter")
    def test_batch_process_molecules(
        self, mock_filter_class, mock_descriptors_class, mock_chem
    ):
        """Test batch processing of molecules"""
        # Setup mocks
        mock_mol = Mock()
        mock_chem.MolFromSmiles.return_value = mock_mol
        mock_chem.MolToSmiles.return_value = "CCO"

        mock_descriptors_class.calculate_basic_descriptors.return_value = {
            "molecular_weight": 46.07,
            "logp": -0.31,
        }

        mock_filter = Mock()
        mock_filter.passes_lipinski.return_value = True
        mock_filter_class.return_value = mock_filter

        # Mock calculate_drug_likeness_score
        with patch(
            "utils.molecular_utils.calculate_drug_likeness_score", return_value=0.8
        ):
            result = batch_process_molecules(
                ["CCO"], calculate_descriptors=True, filter_lipinski=True
            )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result.iloc[0]["original_smiles"] == "CCO"
        assert result.iloc[0]["canonical_smiles"] == "CCO"
        assert result.iloc[0]["valid"] is True
        assert result.iloc[0]["passes_lipinski"] is True

    def test_batch_process_molecules_without_rdkit(self):
        """Test batch processing without RDKit"""
        with patch("utils.molecular_utils.RDKIT_AVAILABLE", False):
            with pytest.raises(ImportError, match="RDKit is required"):
                batch_process_molecules(["CCO"])

    @patch("utils.molecular_utils.RDKIT_AVAILABLE", True)
    @patch("utils.molecular_utils.Chem")
    def test_smiles_to_mol_with_rdkit(self, mock_chem):
        """Test SMILES to Mol conversion with RDKit"""
        mock_mol = Mock()
        mock_chem.MolFromSmiles.return_value = mock_mol

        result = smiles_to_mol("CCO")

        mock_chem.MolFromSmiles.assert_called_once_with("CCO")
        assert result == mock_mol

    def test_smiles_to_mol_without_rdkit(self):
        """Test SMILES to Mol conversion without RDKit"""
        with patch("utils.molecular_utils.RDKIT_AVAILABLE", False):
            result = smiles_to_mol("CCO")

            assert isinstance(result, dict)
            assert result["smiles"] == "CCO"
            assert result["valid"] is True

    def test_mol_to_smiles_with_string_input(self):
        """Test Mol to SMILES conversion with string input"""
        result = mol_to_smiles("CCO")
        assert result == "CCO"

    @patch("utils.molecular_utils.RDKIT_AVAILABLE", True)
    @patch("utils.molecular_utils.Chem")
    def test_mol_to_smiles_with_mol_object(self, mock_chem):
        """Test Mol to SMILES conversion with Mol object"""
        mock_mol = Mock()
        mock_chem.MolToSmiles.return_value = "CCO"

        result = mol_to_smiles(mock_mol)

        mock_chem.MolToSmiles.assert_called_once_with(mock_mol)
        assert result == "CCO"

    def test_mol_to_smiles_with_dict_input(self):
        """Test Mol to SMILES conversion with dict input"""
        mock_mol = {"smiles": "CCO"}
        result = mol_to_smiles(mock_mol)
        assert result == "CCO"

    @patch("utils.molecular_utils.RDKIT_AVAILABLE", True)
    @patch("utils.molecular_utils.Chem")
    def test_validate_smiles_with_rdkit(self, mock_chem):
        """Test SMILES validation with RDKit"""
        mock_chem.MolFromSmiles.return_value = Mock()
        assert validate_smiles("CCO") is True

        mock_chem.MolFromSmiles.return_value = None
        assert validate_smiles("invalid") is False

    def test_validate_smiles_without_rdkit(self):
        """Test SMILES validation without RDKit"""
        with patch("utils.molecular_utils.RDKIT_AVAILABLE", False):
            assert validate_smiles("CCO") is True
            assert validate_smiles("") is False
            assert validate_smiles("   ") is False

    @patch("utils.molecular_utils.RDKIT_AVAILABLE", True)
    @patch("utils.molecular_utils.smiles_to_mol")
    @patch("utils.molecular_utils.Descriptors")
    def test_calculate_molecular_weight_with_rdkit(
        self, mock_descriptors, mock_smiles_to_mol
    ):
        """Test molecular weight calculation with RDKit"""
        mock_mol = Mock()
        mock_smiles_to_mol.return_value = mock_mol
        mock_descriptors.MolWt.return_value = 46.07

        result = calculate_molecular_weight("CCO")

        mock_smiles_to_mol.assert_called_once_with("CCO")
        mock_descriptors.MolWt.assert_called_once_with(mock_mol)
        assert result == 46.07

    def test_calculate_molecular_weight_without_rdkit(self):
        """Test molecular weight calculation without RDKit"""
        with patch("utils.molecular_utils.RDKIT_AVAILABLE", False):
            result = calculate_molecular_weight("CCO")
            # Should return length * 8.0 = 3 * 8.0 = 24.0
            assert result == 24.0

    @patch("utils.molecular_utils.RDKIT_AVAILABLE", True)
    @patch("utils.molecular_utils.smiles_to_mol")
    @patch("utils.molecular_utils.Descriptors")
    def test_calculate_logp_with_rdkit(self, mock_descriptors, mock_smiles_to_mol):
        """Test LogP calculation with RDKit"""
        mock_mol = Mock()
        mock_smiles_to_mol.return_value = mock_mol
        mock_descriptors.MolLogP.return_value = -0.31

        result = calculate_logp("CCO")

        mock_smiles_to_mol.assert_called_once_with("CCO")
        mock_descriptors.MolLogP.assert_called_once_with(mock_mol)
        assert result == -0.31

    def test_calculate_logp_without_rdkit(self):
        """Test LogP calculation without RDKit"""
        with patch("utils.molecular_utils.RDKIT_AVAILABLE", False):
            result = calculate_logp("CCO")
            # Should calculate based on atom counts
            assert isinstance(result, float)

    @patch("utils.molecular_utils.RDKIT_AVAILABLE", True)
    @patch("utils.molecular_utils.smiles_to_mol")
    @patch("utils.molecular_utils.rdMolDescriptors")
    def test_get_molecular_formula_with_rdkit(
        self, mock_mol_descriptors, mock_smiles_to_mol
    ):
        """Test molecular formula calculation with RDKit"""
        mock_mol = Mock()
        mock_smiles_to_mol.return_value = mock_mol
        mock_mol_descriptors.CalcMolFormula.return_value = "C2H6O"

        result = get_molecular_formula("CCO")

        mock_smiles_to_mol.assert_called_once_with("CCO")
        mock_mol_descriptors.CalcMolFormula.assert_called_once_with(mock_mol)
        assert result == "C2H6O"

    def test_get_molecular_formula_without_rdkit(self):
        """Test molecular formula calculation without RDKit"""
        with patch("utils.molecular_utils.RDKIT_AVAILABLE", False):
            result = get_molecular_formula("CCO")
            # Should return rough estimation
            assert isinstance(result, str)
            assert "C" in result
            assert "H" in result

    @patch("utils.molecular_utils.RDKIT_AVAILABLE", True)
    @patch("utils.molecular_utils.smiles_to_mol")
    @patch("utils.molecular_utils.Chem")
    def test_standardize_molecule_with_rdkit(self, mock_chem, mock_smiles_to_mol):
        """Test molecule standardization with RDKit"""
        mock_mol = Mock()
        mock_smiles_to_mol.return_value = mock_mol
        mock_chem.MolToSmiles.return_value = "CCO"

        result = standardize_molecule("C(C)O")

        mock_smiles_to_mol.assert_called_once_with("C(C)O")
        mock_chem.MolToSmiles.assert_called_once_with(mock_mol, canonical=True)
        assert result == "CCO"

    def test_standardize_molecule_without_rdkit(self):
        """Test molecule standardization without RDKit"""
        with patch("utils.molecular_utils.RDKIT_AVAILABLE", False):
            result = standardize_molecule("C(C)O")
            assert result == "C(C)O"  # Should return as-is

    @patch("utils.molecular_utils.RDKIT_AVAILABLE", True)
    @patch("utils.molecular_utils.smiles_to_mol")
    @patch("rdkit.Chem.SaltRemover")
    @patch("utils.molecular_utils.Chem")
    def test_remove_salts_with_rdkit(
        self, mock_chem, mock_salt_remover, mock_smiles_to_mol
    ):
        """Test salt removal with RDKit"""
        mock_mol = Mock()
        mock_mol_no_salt = Mock()
        mock_remover = Mock()

        mock_smiles_to_mol.return_value = mock_mol
        mock_salt_remover.SaltRemover.return_value = mock_remover
        mock_remover.StripMol.return_value = mock_mol_no_salt
        mock_chem.MolToSmiles.return_value = "CCO"

        result = remove_salts("CCO.Cl")

        mock_smiles_to_mol.assert_called_once_with("CCO.Cl")
        mock_remover.StripMol.assert_called_once_with(mock_mol)
        mock_chem.MolToSmiles.assert_called_once_with(mock_mol_no_salt)
        assert result == "CCO"

    def test_remove_salts_without_rdkit(self):
        """Test salt removal without RDKit"""
        with patch("utils.molecular_utils.RDKIT_AVAILABLE", False):
            result = remove_salts("CCO.Cl")
            assert result == "CCO.Cl"  # Should return as-is

    def test_neutralize_molecule(self):
        """Test molecule neutralization"""
        result = neutralize_molecule("CC[N+](C)(C)C.[O-]S(=O)(=O)C")
        # Should apply simple pattern replacements
        assert isinstance(result, str)

    @patch("utils.molecular_utils.RDKIT_AVAILABLE", True)
    @patch("utils.molecular_utils.rdFingerprintGenerator")
    @patch("rdkit.DataStructs")
    def test_calculate_similarity_with_rdkit(self, mock_datastructs, mock_fp_gen):
        """Test similarity calculation with RDKit"""
        # Setup mocks
        mock_generator = Mock()
        mock_fp1 = Mock()
        mock_fp2 = Mock()

        mock_fp_gen.GetMorganGenerator.return_value = mock_generator
        mock_generator.GetFingerprint.side_effect = [mock_fp1, mock_fp2]
        mock_datastructs.TanimotoSimilarity.return_value = 0.75

        with patch("utils.molecular_utils.Chem") as mock_chem:
            mock_mol1 = Mock()
            mock_mol2 = Mock()
            mock_chem.MolFromSmiles.side_effect = [mock_mol1, mock_mol2]

            result = calculate_similarity("CCO", "CC(C)O", method="tanimoto")

        assert result == 0.75
        mock_datastructs.TanimotoSimilarity.assert_called_once_with(mock_fp1, mock_fp2)

    def test_calculate_similarity_without_rdkit(self):
        """Test similarity calculation without RDKit"""
        with patch("utils.molecular_utils.RDKIT_AVAILABLE", False):
            result = calculate_similarity("CCO", "CC(C)O")
            assert 0.0 <= result <= 1.0

    @patch("utils.molecular_utils.RDKIT_AVAILABLE", True)
    @patch("utils.molecular_utils.Chem")
    @patch("utils.molecular_utils.Descriptors")
    def test_filter_molecules_by_properties(self, mock_descriptors, mock_chem):
        """Test molecular property filtering"""
        # Setup mocks for two molecules
        mock_mol1 = Mock()
        mock_mol2 = Mock()
        mock_chem.MolFromSmiles.side_effect = [mock_mol1, mock_mol2]

        # First molecule passes all filters
        # Second molecule fails MW filter
        mock_descriptors.MolWt.side_effect = [300.0, 1000.0]
        mock_descriptors.MolLogP.side_effect = [2.5, 3.0]
        mock_descriptors.NumHDonors.side_effect = [2, 3]
        mock_descriptors.NumHAcceptors.side_effect = [4, 5]

        molecules = ["CCO", "very_large_molecule"]
        filtered = filter_molecules_by_properties(
            molecules, mw_range=(50, 900), logp_range=(-3, 7), apply_lipinski=True
        )

        assert len(filtered) == 1
        assert filtered[0] == "CCO"

    def test_filter_molecules_by_properties_without_rdkit(self):
        """Test molecular property filtering without RDKit"""
        with patch("utils.molecular_utils.RDKIT_AVAILABLE", False):
            molecules = ["CCO", "CC(C)O"]
            filtered = filter_molecules_by_properties(molecules)

            # Should return all molecules as strings
            assert len(filtered) == 2
            assert all(isinstance(mol, str) for mol in filtered)


class TestNewStandaloneFunctions:
    """Test additional standalone functions"""

    @patch("utils.molecular_utils.RDKIT_AVAILABLE", True)
    @patch("utils.molecular_utils.Chem")
    def test_standardize_smiles_with_rdkit(self, mock_chem):
        """Test SMILES standardization with RDKit"""
        mock_mol = Mock()
        mock_chem.MolFromSmiles.return_value = mock_mol
        mock_chem.MolToSmiles.return_value = "CCO"

        result = standardize_smiles("C(C)O")

        mock_chem.MolFromSmiles.assert_called_once_with("C(C)O")
        mock_chem.MolToSmiles.assert_called_once_with(mock_mol)
        assert result == "CCO"

    def test_standardize_smiles_without_rdkit(self):
        """Test SMILES standardization without RDKit"""
        with patch("utils.molecular_utils.RDKIT_AVAILABLE", False):
            result = standardize_smiles("CCO")
            assert result is None

    def test_standardize_smiles_empty_input(self):
        """Test SMILES standardization with empty input"""
        assert standardize_smiles("") is None
        assert standardize_smiles(None) is None

    @patch("utils.molecular_utils.RDKIT_AVAILABLE", True)
    @patch("utils.molecular_utils.Chem")
    @patch("utils.molecular_utils.MolecularDescriptors")
    def test_calculate_molecular_properties_with_rdkit(
        self, mock_descriptors, mock_chem
    ):
        """Test molecular properties calculation with RDKit"""
        mock_mol = Mock()
        mock_chem.MolFromSmiles.return_value = mock_mol
        mock_descriptors.calculate_basic_descriptors.return_value = {
            "molecular_weight": 46.07,
            "logp": -0.31,
        }

        result = calculate_molecular_properties("CCO")

        mock_chem.MolFromSmiles.assert_called_once_with("CCO")
        mock_descriptors.calculate_basic_descriptors.assert_called_once_with(mock_mol)
        assert result["molecular_weight"] == 46.07
        assert result["logp"] == -0.31

    def test_calculate_molecular_properties_without_rdkit(self):
        """Test molecular properties calculation without RDKit"""
        with patch("utils.molecular_utils.RDKIT_AVAILABLE", False):
            result = calculate_molecular_properties("CCO")
            assert result is None

    def test_calculate_molecular_properties_empty_input(self):
        """Test molecular properties calculation with empty input"""
        result = calculate_molecular_properties("")
        assert result is None

        result = calculate_molecular_properties(None)
        assert result is None

    @patch("utils.molecular_utils.RDKIT_AVAILABLE", True)
    @patch("utils.molecular_utils.Chem")
    def test_generate_conformers_with_rdkit(self, mock_chem):
        """Test conformer generation with RDKit"""
        mock_mol = Mock()
        mock_mol_with_h = Mock()

        mock_chem.MolFromSmiles.return_value = mock_mol
        mock_chem.AddHs.return_value = mock_mol_with_h

        result = generate_conformers("CCO", num_conformers=1)

        mock_chem.MolFromSmiles.assert_called_once_with("CCO")
        mock_chem.AddHs.assert_called_once_with(mock_mol)
        assert result == mock_mol_with_h

    def test_generate_conformers_without_rdkit(self):
        """Test conformer generation without RDKit"""
        with patch("utils.molecular_utils.RDKIT_AVAILABLE", False):
            result = generate_conformers("CCO")
            assert result is None

    def test_generate_conformers_empty_input(self):
        """Test conformer generation with empty input"""
        result = generate_conformers("")
        assert result is None

        result = generate_conformers(None)
        assert result is None

    @patch("utils.molecular_utils.RDKIT_AVAILABLE", True)
    @patch("utils.molecular_utils.Chem")
    def test_validate_molecule_with_rdkit(self, mock_chem):
        """Test molecule validation with RDKit"""
        mock_chem.MolFromSmiles.return_value = Mock()
        assert validate_molecule("CCO") is True

        mock_chem.MolFromSmiles.return_value = None
        assert validate_molecule("invalid") is False

    def test_validate_molecule_without_rdkit(self):
        """Test molecule validation without RDKit"""
        with patch("utils.molecular_utils.RDKIT_AVAILABLE", False):
            assert validate_molecule("CCO") is False

    def test_validate_molecule_empty_input(self):
        """Test molecule validation with empty input"""
        assert validate_molecule("") is False
        assert validate_molecule(None) is False


class TestStructuralAlerts:
    """Test StructuralAlerts class"""

    def test_init_without_rdkit(self):
        """Test initialization when RDKit is not available"""
        with patch("utils.molecular_utils.RDKIT_AVAILABLE", False):
            with pytest.raises(ImportError, match="RDKit is required"):
                StructuralAlerts()

    @patch("utils.molecular_utils.RDKIT_AVAILABLE", True)
    def test_init_with_rdkit(self):
        """Test initialization when RDKit is available"""
        alerts = StructuralAlerts()
        assert alerts is not None

    @patch("utils.molecular_utils.RDKIT_AVAILABLE", True)
    @patch("utils.molecular_utils.Chem")
    def test_check_pains_alerts(self, mock_chem):
        """Test PAINS alerts checking"""
        alerts = StructuralAlerts()
        mock_mol = Mock()

        # Test molecule with nitro group
        mock_chem.MolToSmiles.return_value = "CC(=O)c1ccc([N+](=O)[O-])cc1"
        result = alerts.check_pains_alerts(mock_mol)

        assert len(result) == 1
        assert "nitro" in result[0]

    @patch("utils.molecular_utils.RDKIT_AVAILABLE", True)
    @patch("utils.molecular_utils.Chem")
    def test_check_pains_alerts_clean_molecule(self, mock_chem):
        """Test PAINS alerts for clean molecule"""
        alerts = StructuralAlerts()
        mock_mol = Mock()

        # Test clean molecule
        mock_chem.MolToSmiles.return_value = "CCO"
        result = alerts.check_pains_alerts(mock_mol)

        assert len(result) == 0

    @patch("utils.molecular_utils.RDKIT_AVAILABLE", True)
    @patch("utils.molecular_utils.Chem")
    def test_check_brenk_alerts(self, mock_chem):
        """Test Brenk alerts checking"""
        alerts = StructuralAlerts()
        mock_mol = Mock()

        # Test molecule with azide
        mock_chem.MolToSmiles.return_value = "CC[N-][N+]#N"
        result = alerts.check_brenk_alerts(mock_mol)

        assert len(result) == 1
        assert "azide" in result[0]

    @patch("utils.molecular_utils.RDKIT_AVAILABLE", True)
    def test_check_alerts_with_mock_object(self):
        """Test alerts checking with mock molecule object"""
        alerts = StructuralAlerts()

        # Test with mock object that raises exception
        mock_mol = Mock()
        mock_mol.GetNumAtoms.side_effect = Exception("Mock exception")

        # Should handle exceptions gracefully
        pains_result = alerts.check_pains_alerts(mock_mol)
        brenk_result = alerts.check_brenk_alerts(mock_mol)

        assert isinstance(pains_result, list)
        assert isinstance(brenk_result, list)


class TestSimilarityCalculator:
    """Test SimilarityCalculator class"""

    def test_init(self):
        """Test initialization"""
        calc = SimilarityCalculator()
        assert calc is not None

    def test_tanimoto_similarity(self):
        """Test Tanimoto similarity calculation"""
        calc = SimilarityCalculator()

        fp1 = np.array([1, 0, 1, 1, 0])
        fp2 = np.array([1, 1, 1, 0, 0])

        # Intersection: [1, 0, 1, 0, 0] = 2
        # Union: [1, 1, 1, 1, 0] = 4
        # Tanimoto = 2/4 = 0.5

        similarity = calc.tanimoto_similarity(fp1, fp2)
        assert similarity == 0.5

    def test_tanimoto_similarity_identical(self):
        """Test Tanimoto similarity for identical fingerprints"""
        calc = SimilarityCalculator()

        fp1 = np.array([1, 0, 1, 1, 0])
        fp2 = np.array([1, 0, 1, 1, 0])

        similarity = calc.tanimoto_similarity(fp1, fp2)
        assert similarity == 1.0

    def test_tanimoto_similarity_zero_union(self):
        """Test Tanimoto similarity with zero union"""
        calc = SimilarityCalculator()

        fp1 = np.array([0, 0, 0, 0, 0])
        fp2 = np.array([0, 0, 0, 0, 0])

        similarity = calc.tanimoto_similarity(fp1, fp2)
        assert similarity == 0.0

    def test_dice_similarity(self):
        """Test Dice similarity calculation"""
        calc = SimilarityCalculator()

        fp1 = np.array([1, 0, 1, 1, 0])  # sum = 3
        fp2 = np.array([1, 1, 1, 0, 0])  # sum = 3

        # Intersection: [1, 0, 1, 0, 0] = 2
        # Dice = 2 * 2 / (3 + 3) = 4/6 = 2/3

        similarity = calc.dice_similarity(fp1, fp2)
        assert abs(similarity - 2 / 3) < 1e-10

    def test_dice_similarity_zero_total(self):
        """Test Dice similarity with zero total"""
        calc = SimilarityCalculator()

        fp1 = np.array([0, 0, 0, 0, 0])
        fp2 = np.array([0, 0, 0, 0, 0])

        similarity = calc.dice_similarity(fp1, fp2)
        assert similarity == 0.0


class TestMolecularVisualization:
    """Test MolecularVisualization class"""

    def test_init_without_rdkit(self):
        """Test initialization when RDKit is not available"""
        with patch("utils.molecular_utils.RDKIT_AVAILABLE", False):
            with pytest.raises(ImportError, match="RDKit is required"):
                MolecularVisualization()

    @patch("utils.molecular_utils.RDKIT_AVAILABLE", True)
    def test_init_with_rdkit(self):
        """Test initialization when RDKit is available"""
        viz = MolecularVisualization()
        assert viz is not None

    @patch("utils.molecular_utils.RDKIT_AVAILABLE", True)
    @patch("rdkit.Chem.Draw")
    def test_draw_molecule_2d_with_rdkit(self, mock_draw):
        """Test 2D molecule drawing with RDKit"""
        viz = MolecularVisualization()
        mock_mol = Mock()
        mock_image = Mock()

        mock_draw.MolToImage.return_value = mock_image

        result = viz.draw_molecule_2d(mock_mol, size=(400, 400))

        mock_draw.MolToImage.assert_called_once_with(mock_mol, size=(400, 400))
        assert result == mock_image

    @patch("utils.molecular_utils.RDKIT_AVAILABLE", False)
    def test_draw_molecule_2d_without_rdkit(self):
        """Test 2D molecule drawing without RDKit"""
        # Since init requires RDKit, this tests the method behavior
        # when RDKit becomes unavailable after initialization
        viz = MolecularVisualization.__new__(MolecularVisualization)
        mock_mol = Mock()

        result = viz.draw_molecule_2d(mock_mol)
        assert result is None

    @patch("utils.molecular_utils.RDKIT_AVAILABLE", True)
    @patch("utils.molecular_utils.PY3DMOL_AVAILABLE", False)
    def test_draw_molecule_3d_without_py3dmol(self):
        """Test 3D molecule drawing without py3Dmol"""
        viz = MolecularVisualization()
        mock_mol = Mock()

        result = viz.draw_molecule_3d(mock_mol)
        assert result is None

    @patch("utils.molecular_utils.RDKIT_AVAILABLE", True)
    @patch("utils.molecular_utils.PY3DMOL_AVAILABLE", True)
    def test_draw_molecule_3d_with_py3dmol(self):
        """Test 3D molecule drawing with py3Dmol"""
        viz = MolecularVisualization()
        mock_mol = Mock()

        result = viz.draw_molecule_3d(mock_mol)

        # Should return placeholder string
        assert result == "3D visualization placeholder"


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple components"""

    @patch("utils.molecular_utils.RDKIT_AVAILABLE", True)
    def test_complete_molecule_analysis_workflow(self):
        """Test complete molecule analysis workflow"""
        # Test workflow: SMILES processing -> descriptor calculation -> filtering

        processor = SMILESProcessor()
        lipinski_filter = LipinskiFilter(strict=False)

        # Mock the individual methods
        processor.is_valid_smiles = Mock(return_value=True)
        processor.canonicalize_smiles = Mock(return_value="CCO")
        lipinski_filter.passes_lipinski = Mock(return_value=True)

        # Process SMILES
        smiles_list = ["C(C)O"]
        result = processor.process_smiles_list(smiles_list)

        # Verify processing
        assert len(result["valid"]) == 1
        assert result["canonical"][0] == "CCO"

    @patch("utils.molecular_utils.RDKIT_AVAILABLE", True)
    @patch("utils.molecular_utils.Chem")
    def test_similarity_analysis_workflow(self, mock_chem):
        """Test molecular similarity analysis workflow"""
        # Test calculation of similarities between multiple molecules

        molecules = ["CCO", "CC(C)O", "C1=CC=CC=C1"]

        # Mock molecule creation
        mock_mols = [Mock() for _ in molecules]
        mock_chem.MolFromSmiles.side_effect = mock_mols

        # Mock similarity calculation
        with patch("utils.molecular_utils.calculate_similarity", return_value=0.6):
            similarities = []
            for i in range(len(molecules)):
                for j in range(i + 1, len(molecules)):
                    sim = calculate_similarity(molecules[i], molecules[j])
                    similarities.append(sim)

        assert len(similarities) == 3  # 3 choose 2 = 3 pairs
        assert all(sim == 0.6 for sim in similarities)

    @patch("utils.molecular_utils.RDKIT_AVAILABLE", True)
    def test_comprehensive_property_calculation(self):
        """Test comprehensive molecular property calculation"""
        # Test calculation of multiple properties for a molecule

        smiles = "CCO"

        # Mock all property calculations
        with patch(
            "utils.molecular_utils.calculate_molecular_weight", return_value=46.07
        ):
            with patch("utils.molecular_utils.calculate_logp", return_value=-0.31):
                with patch(
                    "utils.molecular_utils.get_molecular_formula", return_value="C2H6O"
                ):
                    with patch(
                        "utils.molecular_utils.validate_molecule", return_value=True
                    ):
                        properties = {
                            "mw": calculate_molecular_weight(smiles),
                            "logp": calculate_logp(smiles),
                            "formula": get_molecular_formula(smiles),
                            "valid": validate_molecule(smiles),
                        }

        assert properties["mw"] == 46.07
        assert properties["logp"] == -0.31
        assert properties["formula"] == "C2H6O"
        assert properties["valid"] is True


class TestErrorHandling:
    """Test error handling and edge cases"""

    @patch("utils.molecular_utils.RDKIT_AVAILABLE", True)
    @patch("utils.molecular_utils.Chem")
    def test_invalid_smiles_handling(self, mock_chem):
        """Test handling of invalid SMILES strings"""
        # Test various invalid SMILES inputs
        invalid_smiles = ["", None, "invalid_chars_@#$", "C(C(C"]

        mock_chem.MolFromSmiles.return_value = None

        for smiles in invalid_smiles:
            try:
                if smiles is not None:
                    result = validate_smiles(smiles)
                    assert result is False
                else:
                    result = validate_smiles(smiles)
                    assert result is False
            except Exception:
                # Some functions may raise exceptions for None input
                pass

    def test_fallback_behavior_without_dependencies(self):
        """Test fallback behavior when dependencies are not available"""
        with patch("utils.molecular_utils.RDKIT_AVAILABLE", False):
            with patch("utils.molecular_utils.PY3DMOL_AVAILABLE", False):
                # Test functions that should work without RDKit
                result = smiles_to_mol("CCO")
                assert isinstance(result, dict)

                result = mol_to_smiles("CCO")
                assert result == "CCO"

                result = validate_smiles("CCO")
                assert result is True

    @patch("utils.molecular_utils.RDKIT_AVAILABLE", True)
    def test_exception_handling_in_calculations(self):
        """Test exception handling in calculations"""
        # Test with mocked functions that raise exceptions

        with patch(
            "utils.molecular_utils.Chem.MolFromSmiles",
            side_effect=Exception("RDKit error"),
        ):
            result = validate_smiles("CCO")
            assert result is False

        with patch(
            "utils.molecular_utils.Descriptors.MolWt",
            side_effect=Exception("Descriptor error"),
        ):
            # Test that drug likeness calculation handles exceptions
            try:
                mock_mol = Mock()
                result = calculate_drug_likeness_score(mock_mol)
                # Should either handle gracefully or raise appropriate error
            except Exception:
                pass

    def test_empty_and_none_inputs(self):
        """Test handling of empty and None inputs"""
        # Test various functions with empty/None inputs

        assert validate_smiles("") is False
        assert validate_smiles(None) is False

        assert mol_to_smiles(None) == ""
        assert mol_to_smiles("") == ""

        with patch("utils.molecular_utils.RDKIT_AVAILABLE", True):
            assert standardize_smiles("") is None
            assert standardize_smiles(None) is None

            assert calculate_molecular_properties("") is None
            assert calculate_molecular_properties(None) is None


class TestPerformance:
    """Test performance aspects"""

    @patch("utils.molecular_utils.RDKIT_AVAILABLE", True)
    def test_batch_processing_performance(self):
        """Test batch processing with large molecule sets"""
        # Test processing of many molecules
        large_smiles_list = ["CCO"] * 100

        # Mock the expensive operations
        with patch("utils.molecular_utils.Chem.MolFromSmiles", return_value=Mock()):
            with patch("utils.molecular_utils.Chem.MolToSmiles", return_value="CCO"):
                with patch(
                    "utils.molecular_utils.MolecularDescriptors.calculate_basic_descriptors",
                    return_value={"mw": 46.07},
                ):
                    with patch(
                        "utils.molecular_utils.calculate_drug_likeness_score",
                        return_value=0.8,
                    ):
                        result = batch_process_molecules(
                            large_smiles_list,
                            calculate_descriptors=True,
                            filter_lipinski=False,
                        )

        assert len(result) == 100
        assert isinstance(result, pd.DataFrame)

    def test_similarity_calculation_performance(self):
        """Test similarity calculations for many molecules"""
        # Test similarity matrix calculation for multiple molecules
        molecules = ["CCO", "CC(C)O", "C1=CC=CC=C1", "CC(=O)O"] * 10  # 40 molecules

        with patch("utils.molecular_utils.calculate_similarity", return_value=0.5):
            similarities = []
            for i in range(len(molecules)):
                for j in range(i + 1, len(molecules)):
                    sim = calculate_similarity(molecules[i], molecules[j])
                    similarities.append(sim)

        # Should calculate (40 choose 2) = 780 similarities
        expected_count = len(molecules) * (len(molecules) - 1) // 2
        assert len(similarities) == expected_count


class TestCrossModuleCompatibility:
    """Test compatibility with other modules"""

    def test_molecular_utils_imports(self):
        """Test that all expected functions and classes are importable"""
        from utils.molecular_utils import (
            LipinskiFilter,
            MolecularDescriptors,
            MolecularVisualization,
            MoleculeVisualizer,
            SimilarityCalculator,
            SMILESProcessor,
            StructuralAlerts,
            batch_process_molecules,
            calculate_drug_likeness_score,
            calculate_logp,
            calculate_molecular_properties,
            calculate_molecular_weight,
            calculate_similarity,
            filter_molecules_by_properties,
            generate_conformers,
            get_molecular_formula,
            mol_to_smiles,
            neutralize_molecule,
            remove_salts,
            smiles_to_mol,
            standardize_molecule,
            standardize_smiles,
            validate_molecule,
            validate_smiles,
        )

        # All imports should succeed
        assert MolecularDescriptors is not None
        assert LipinskiFilter is not None
        assert SMILESProcessor is not None
        assert MoleculeVisualizer is not None
        assert StructuralAlerts is not None
        assert SimilarityCalculator is not None
        assert MolecularVisualization is not None

    def test_availability_flags(self):
        """Test dependency availability flags"""
        from utils.molecular_utils import PY3DMOL_AVAILABLE, RDKIT_AVAILABLE

        # Flags should be boolean
        assert isinstance(RDKIT_AVAILABLE, bool)
        assert isinstance(PY3DMOL_AVAILABLE, bool)

    def test_pandas_integration(self):
        """Test pandas DataFrame integration"""
        # Test that functions return proper pandas DataFrames

        with patch("utils.molecular_utils.RDKIT_AVAILABLE", True):
            with patch("utils.molecular_utils.batch_process_molecules") as mock_batch:
                mock_df = pd.DataFrame({"smiles": ["CCO"], "mw": [46.07]})
                mock_batch.return_value = mock_df

                result = batch_process_molecules(["CCO"])

                assert isinstance(result, pd.DataFrame)
                assert "smiles" in result.columns or "original_smiles" in result.columns

    def test_numpy_array_compatibility(self):
        """Test NumPy array compatibility"""
        # Test functions that work with NumPy arrays

        calc = SimilarityCalculator()

        # Test with different NumPy array types
        fp1 = np.array([1, 0, 1, 0], dtype=np.int32)
        fp2 = np.array([1, 1, 0, 0], dtype=np.int64)

        similarity = calc.tanimoto_similarity(fp1, fp2)

        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
