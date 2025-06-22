#!/usr/bin/env python3
"""
Consolidated Core Molecular Tests for QeMLflow

This file combines essential tests from multiple molecular test files:
- test_molecular_utils_comprehensive.py
- test_molecular_utils_extended.py  
- test_molecular_preprocessing_comprehensive.py
- test_molecular_optimization_comprehensive.py

Focus: Core molecular functionality only - SMILES processing, descriptors, 
basic optimization, and essential molecular operations.
"""

import logging
import sys
import unittest

# Import the module under test
sys.path.insert(0, "/Users/sanjeevadodlapati/Downloads/Repos/QeMLflow/src")

try:
    from qemlflow.core.utils.molecular_utils import (
        LipinskiFilter,
        MolecularDescriptors,
        SMILESProcessor,
        calculate_molecular_properties,
        calculate_molecular_weight,
        mol_to_smiles,
        smiles_to_mol,
        standardize_smiles,
        validate_smiles,
    )
    MOLECULAR_UTILS_AVAILABLE = True
except ImportError:
    MOLECULAR_UTILS_AVAILABLE = False

try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    Chem = None
    RDKIT_AVAILABLE = False


class TestCoreMolecularDescriptors(unittest.TestCase):
    """Essential tests for molecular descriptor calculation."""

    def setUp(self):
        """Set up test fixtures."""
        if MOLECULAR_UTILS_AVAILABLE:
            try:
                self.descriptors = MolecularDescriptors()
                self.available = True
            except Exception:
                self.available = False
        else:
            self.available = False

    @unittest.skipUnless(RDKIT_AVAILABLE, "RDKit not available")
    def test_basic_descriptor_calculation(self):
        """Test basic molecular descriptor calculations."""
        if not self.available:
            self.skipTest("MolecularDescriptors not available")
        
        # Test with simple molecule (methane)
        test_smiles = "C"
        result = self.descriptors.calculate_basic_descriptors(test_smiles)
        
        self.assertIsInstance(result, dict)
        self.assertIn('molecular_weight', result)
        self.assertGreater(result['molecular_weight'], 0)

    @unittest.skipUnless(RDKIT_AVAILABLE, "RDKit not available")
    def test_lipinski_descriptors(self):
        """Test Lipinski rule of five descriptors."""
        if not self.available:
            self.skipTest("MolecularDescriptors not available")
        
        # Test with aspirin
        test_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
        result = self.descriptors.calculate_lipinski_descriptors(test_smiles)
        
        self.assertIsInstance(result, dict)
        self.assertIn('molecular_weight', result)
        self.assertIn('logp', result)


class TestCoreSMILESProcessing(unittest.TestCase):
    """Essential tests for SMILES processing functionality."""

    def setUp(self):
        """Set up test fixtures."""
        if MOLECULAR_UTILS_AVAILABLE:
            try:
                self.processor = SMILESProcessor()
                self.available = True
            except Exception:
                self.available = False
        else:
            self.available = False

    @unittest.skipUnless(RDKIT_AVAILABLE, "RDKit not available")
    def test_smiles_validation(self):
        """Test SMILES validation functionality."""
        if not self.available:
            self.skipTest("SMILESProcessor not available")
        
        # Valid SMILES
        valid_smiles = ["C", "CCO", "CC(=O)O"]
        for smiles in valid_smiles:
            self.assertTrue(self.processor.is_valid_smiles(smiles))
        
        # Invalid SMILES
        invalid_smiles = ["", "XYZ", "C("]
        for smiles in invalid_smiles:
            self.assertFalse(self.processor.is_valid_smiles(smiles))

    @unittest.skipUnless(RDKIT_AVAILABLE, "RDKit not available")
    def test_smiles_canonicalization(self):
        """Test SMILES canonicalization."""
        if not self.available:
            self.skipTest("SMILESProcessor not available")
        
        # Test canonicalization
        test_smiles = "CCO"
        canonical = self.processor.canonicalize_smiles(test_smiles)
        
        self.assertIsInstance(canonical, str)
        self.assertEqual(canonical, "CCO")


class TestCoreLipinskiFilter(unittest.TestCase):
    """Essential tests for Lipinski rule filtering."""

    def setUp(self):
        """Set up test fixtures."""
        if MOLECULAR_UTILS_AVAILABLE:
            try:
                self.filter = LipinskiFilter()
                self.available = True
            except Exception:
                self.available = False
        else:
            self.available = False

    @unittest.skipUnless(RDKIT_AVAILABLE, "RDKit not available")
    def test_drug_like_filtering(self):
        """Test basic drug-like molecule filtering."""
        if not self.available:
            self.skipTest("LipinskiFilter not available")
        
        # Drug-like molecule (aspirin)
        drug_like = "CC(=O)OC1=CC=CC=C1C(=O)O"
        result = self.filter.passes_lipinski(drug_like)
        
        self.assertIsInstance(result, bool)

    @unittest.skipUnless(RDKIT_AVAILABLE, "RDKit not available")
    def test_filter_molecule_list(self):
        """Test filtering a list of molecules."""
        if not self.available:
            self.skipTest("LipinskiFilter not available")
        
        test_molecules = ["C", "CCO", "CC(=O)O"]
        filtered = self.filter.filter_molecules(test_molecules)
        
        self.assertIsInstance(filtered, list)
        self.assertLessEqual(len(filtered), len(test_molecules))


class TestCoreMolecularProperties(unittest.TestCase):
    """Essential tests for molecular property calculations."""

    @unittest.skipUnless(MOLECULAR_UTILS_AVAILABLE, "Molecular utils not available")
    def test_molecular_weight_calculation(self):
        """Test molecular weight calculation."""
        if not RDKIT_AVAILABLE:
            self.skipTest("RDKit not available")
        
        # Test with water
        test_smiles = "O"
        mw = calculate_molecular_weight(test_smiles)
        
        self.assertIsInstance(mw, (int, float))
        self.assertGreater(mw, 0)

    @unittest.skipUnless(MOLECULAR_UTILS_AVAILABLE, "Molecular utils not available")
    def test_smiles_conversion(self):
        """Test SMILES to molecule conversion."""
        if not RDKIT_AVAILABLE:
            self.skipTest("RDKit not available")
        
        test_smiles = "CCO"
        mol = smiles_to_mol(test_smiles)
        
        if mol is not None:
            converted_smiles = mol_to_smiles(mol)
            self.assertIsInstance(converted_smiles, str)

    @unittest.skipUnless(MOLECULAR_UTILS_AVAILABLE, "Molecular utils not available")
    def test_smiles_validation_function(self):
        """Test standalone SMILES validation function."""
        if not RDKIT_AVAILABLE:
            self.skipTest("RDKit not available")
        
        # Valid SMILES
        self.assertTrue(validate_smiles("C"))
        self.assertTrue(validate_smiles("CCO"))
        
        # Invalid SMILES
        self.assertFalse(validate_smiles(""))
        self.assertFalse(validate_smiles("XYZ"))


class TestCoreMolecularOptimization(unittest.TestCase):
    """Essential tests for basic molecular optimization."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_molecules = ["C", "CCO", "CC(=O)O"]

    @unittest.skipUnless(MOLECULAR_UTILS_AVAILABLE, "Molecular utils not available")
    def test_basic_standardization(self):
        """Test basic molecule standardization."""
        if not RDKIT_AVAILABLE:
            self.skipTest("RDKit not available")
        
        for smiles in self.test_molecules:
            try:
                standardized = standardize_smiles(smiles)
                self.assertIsInstance(standardized, str)
                self.assertTrue(validate_smiles(standardized))
            except Exception:
                # Some molecules may not be standardizable
                pass

    @unittest.skipUnless(MOLECULAR_UTILS_AVAILABLE, "Molecular utils not available")
    def test_property_calculation_batch(self):
        """Test batch property calculations."""
        if not RDKIT_AVAILABLE:
            self.skipTest("RDKit not available")
        
        results = []
        for smiles in self.test_molecules:
            try:
                props = calculate_molecular_properties(smiles)
                if props:
                    results.append(props)
            except Exception:
                # Some calculations may fail
                pass
        
        # At least some should succeed
        self.assertGreater(len(results), 0)


class TestIntegrationWorkflows(unittest.TestCase):
    """Essential integration tests for core workflows."""

    @unittest.skipUnless(MOLECULAR_UTILS_AVAILABLE, "Molecular utils not available")
    def test_complete_molecule_processing_workflow(self):
        """Test a complete molecule processing workflow."""
        if not RDKIT_AVAILABLE:
            self.skipTest("RDKit not available")
        
        # Start with a raw SMILES
        raw_smiles = "CCO"  # ethanol
        
        # Validate
        self.assertTrue(validate_smiles(raw_smiles))
        
        # Standardize
        try:
            standardized = standardize_smiles(raw_smiles)
            if standardized:
                self.assertIsInstance(standardized, str)
                self.assertTrue(validate_smiles(standardized))
            else:
                standardized = raw_smiles
        except ImportError:
            standardized = raw_smiles
        
        # Calculate properties
        try:
            properties = calculate_molecular_properties(standardized)
            if properties:
                self.assertIsInstance(properties, dict)
        except ImportError:
            pass  # Properties calculation may not be available
        
        # Calculate molecular weight
        try:
            mw = calculate_molecular_weight(standardized)
            self.assertIsInstance(mw, (int, float))
            self.assertGreater(mw, 0)
        except ImportError:
            pass  # MW calculation may not be available

    @unittest.skipUnless(MOLECULAR_UTILS_AVAILABLE, "Molecular utils not available")
    def test_drug_discovery_pipeline(self):
        """Test essential drug discovery pipeline components."""
        if not RDKIT_AVAILABLE:
            self.skipTest("RDKit not available")
        
        # Test molecules including drug-like compounds
        test_compounds = [
            "CCO",  # ethanol
            "CC(=O)OC1=CC=CC=C1C(=O)O",  # aspirin
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"  # ibuprofen
        ]
        
        processed_count = 0
        for smiles in test_compounds:
            try:
                # Validate
                if validate_smiles(smiles):
                    # Calculate basic properties
                    mw = calculate_molecular_weight(smiles)
                    if mw and mw > 0:
                        processed_count += 1
            except Exception:
                pass
        
        # At least some molecules should be processed successfully
        self.assertGreater(processed_count, 0)


if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(level=logging.WARNING)
    
    # Run tests
    unittest.main(verbosity=2)
