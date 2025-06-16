#!/usr/bin/env python3
"""
Unit tests for ADMET prediction functionality.

Tests cover prediction methods, drug-likeness assessment, and filtering functions.
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from chemml.research.drug_discovery.admet import (
try:
    from rdkit import Chem
except ImportError:
    pass
    ADMETPredictor,
    apply_admet_filters,
    assess_drug_likeness,
    predict_admet_properties,
)


class TestADMETPredictor(unittest.TestCase):
    """Test cases for ADMETPredictor class."""

    def setUp(self):
        self.predictor = ADMETPredictor()
        self.test_smiles = [
            "CCO",  # Ethanol - simple molecule
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
            "invalid_smiles",  # Invalid SMILES for error testing
        ]

    def test_predictor_initialization(self):
        """Test ADMETPredictor initialization."""
        self.assertIsInstance(self.predictor.models, dict)
        self.assertIsInstance(self.predictor.scalers, dict)
        self.assertEqual(len(self.predictor.models), 0)
        self.assertEqual(len(self.predictor.scalers), 0)

    def test_predict_admet_properties_basic(self):
        """Test basic ADMET properties prediction."""
        results = self.predictor.predict_admet_properties(self.test_smiles[:2])

        self.assertIsInstance(results, pd.DataFrame)
        self.assertEqual(len(results), 2)

        # Check required columns
        expected_columns = [
            "SMILES",
            "absorption",
            "bioavailability",
            "bbb_permeability",
            "cyp_inhibition",
            "hepatotoxicity",
            "mutagenicity",
            "drug_likeness",
        ]
        for col in expected_columns:
            self.assertIn(col, results.columns)

        # Check data types
        for col in expected_columns[1:]:  # Skip SMILES
            if col == "cyp_inhibition":
                continue  # This returns a dict
            self.assertTrue(all(isinstance(val, (int, float)) for val in results[col]))

    def test_predict_absorption(self):
        """Test absorption prediction."""
        # Test valid SMILES
        absorption = self.predictor.predict_absorption("CCO")
        self.assertIsInstance(absorption, float)
        self.assertGreaterEqual(absorption, 0.0)
        self.assertLessEqual(absorption, 1.0)

        # Test invalid SMILES
        absorption_invalid = self.predictor.predict_absorption("invalid_smiles")
        self.assertEqual(absorption_invalid, 0.0)

    def test_predict_bioavailability(self):
        """Test bioavailability prediction."""
        bioavailability = self.predictor.predict_bioavailability("CCO")
        self.assertIsInstance(bioavailability, float)
        self.assertGreaterEqual(bioavailability, 0.0)
        self.assertLessEqual(bioavailability, 1.0)

        # Test invalid SMILES
        bio_invalid = self.predictor.predict_bioavailability("invalid_smiles")
        self.assertEqual(bio_invalid, 0.0)

    def test_predict_bbb_permeability(self):
        """Test blood-brain barrier permeability prediction."""
        bbb = self.predictor.predict_bbb_permeability("CCO")
        self.assertIsInstance(bbb, float)
        self.assertGreaterEqual(bbb, 0.0)
        self.assertLessEqual(bbb, 1.0)

        # Test invalid SMILES
        bbb_invalid = self.predictor.predict_bbb_permeability("invalid_smiles")
        self.assertEqual(bbb_invalid, 0.0)

    def test_predict_cyp_inhibition(self):
        """Test CYP enzyme inhibition prediction."""
        cyp_result = self.predictor.predict_cyp_inhibition("CCO")
        self.assertIsInstance(cyp_result, dict)

        # Check expected CYP enzymes
        expected_cyps = ["CYP1A2", "CYP2C9", "CYP2C19", "CYP2D6", "CYP3A4"]
        for cyp in expected_cyps:
            self.assertIn(cyp, cyp_result)
            self.assertIsInstance(cyp_result[cyp], float)
            self.assertGreaterEqual(cyp_result[cyp], 0.0)
            self.assertLessEqual(cyp_result[cyp], 1.0)

    def test_predict_hepatotoxicity(self):
        """Test hepatotoxicity prediction."""
        hepatotox = self.predictor.predict_hepatotoxicity("CCO")
        self.assertIsInstance(hepatotox, float)
        self.assertGreaterEqual(hepatotox, 0.0)
        self.assertLessEqual(hepatotox, 1.0)

    def test_predict_mutagenicity(self):
        """Test mutagenicity prediction."""
        mutagen = self.predictor.predict_mutagenicity("CCO")
        self.assertIsInstance(mutagen, float)
        self.assertGreaterEqual(mutagen, 0.0)
        self.assertLessEqual(mutagen, 1.0)

    def test_calculate_drug_likeness_score(self):
        """Test drug-likeness score calculation."""
        score = self.predictor.calculate_drug_likeness_score("CCO")
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

        # Test invalid SMILES
        score_invalid = self.predictor.calculate_drug_likeness_score("invalid_smiles")
        self.assertEqual(score_invalid, 0.0)

    @patch("src.drug_design.admet_prediction.RDKIT_AVAILABLE", False)
    def test_rdkit_unavailable_fallback(self):
        """Test fallback behavior when RDKit is not available."""
        predictor = ADMETPredictor()

        # Test absorption fallback
        absorption = predictor.predict_absorption("CCO")
        self.assertEqual(absorption, 0.5)

        # Test bioavailability fallback
        bioavailability = predictor.predict_bioavailability("CCO")
        self.assertEqual(bioavailability, 0.5)

        # Test BBB permeability fallback
        bbb = predictor.predict_bbb_permeability("CCO")
        self.assertEqual(bbb, 0.3)

        # Test CYP inhibition fallback
        cyp = predictor.predict_cyp_inhibition("CCO")
        expected_cyps = ["CYP1A2", "CYP2C9", "CYP2C19", "CYP2D6", "CYP3A4"]
        for enzyme in expected_cyps:
            self.assertIn(enzyme, cyp)
            self.assertEqual(cyp[enzyme], 0.2)


class TestToxicityPredictor(unittest.TestCase):
    """Test cases for ToxicityPredictor class."""

    def setUp(self):
        from chemml.research.drug_discovery.admet import ToxicityPredictor

        self.predictor = ToxicityPredictor()
        self.test_smiles = ["CCO", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"]

    def test_predictor_initialization(self):
        """Test ToxicityPredictor initialization."""
        self.assertIsInstance(self.predictor.endpoints, list)
        self.assertGreater(len(self.predictor.endpoints), 0)

    def test_predict_toxicity(self):
        """Test toxicity prediction."""
        result = self.predictor.predict_toxicity("CCO")
        self.assertIsInstance(result, dict)

        for endpoint in self.predictor.endpoints:
            self.assertIn(endpoint, result)
            self.assertIsInstance(result[endpoint], float)
            self.assertGreaterEqual(result[endpoint], 0.0)
            self.assertLessEqual(result[endpoint], 1.0)

    def test_predict_multiple_toxicity(self):
        """Test toxicity prediction for multiple molecules."""
        results = self.predictor.predict_toxicity(self.test_smiles)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), len(self.test_smiles))

        for result in results:
            self.assertIsInstance(result, dict)
            for endpoint in self.predictor.endpoints:
                self.assertIn(endpoint, result)

    def test_invalid_smiles_toxicity(self):
        """Test toxicity prediction with invalid SMILES."""
        result = self.predictor.predict_toxicity("invalid_smiles")
        self.assertIsInstance(result, dict)

        # Should return high toxicity for invalid SMILES
        for endpoint in self.predictor.endpoints:
            self.assertEqual(result[endpoint], 1.0)


class TestDrugLikenessAssessment(unittest.TestCase):
    """Test cases for drug-likeness assessment functions."""

    def setUp(self):
        self.test_molecules = [
            "CCO",  # Simple molecule
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        ]

    def test_assess_drug_likeness_basic(self):
        """Test basic drug-likeness assessment."""
        result = assess_drug_likeness(self.test_molecules)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.test_molecules))

        # Check required columns
        expected_columns = [
            "SMILES",
            "lipinski_violations",
            "veber_violations",
            "muegge_violations",
            "drug_like_score",
        ]
        for col in expected_columns:
            self.assertIn(col, result.columns)

    def test_assess_drug_likeness_empty_input(self):
        """Test drug-likeness assessment with empty input."""
        result = assess_drug_likeness([])
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 0)

    def test_assess_drug_likeness_invalid_smiles(self):
        """Test drug-likeness assessment with invalid SMILES."""
        result = assess_drug_likeness(["invalid_smiles"])
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 1)

        # Should handle invalid SMILES gracefully
        self.assertEqual(result.iloc[0]["SMILES"], "invalid_smiles")

    @patch("src.drug_design.admet_prediction.RDKIT_AVAILABLE", False)
    def test_assess_drug_likeness_no_rdkit(self):
        """Test drug-likeness assessment when RDKit is not available."""
        result = assess_drug_likeness(self.test_molecules)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.test_molecules))

        # Should return default values when RDKit is not available
        for col in ["lipinski_violations", "veber_violations", "muegge_violations"]:
            self.assertTrue(all(result[col] == 0))
        self.assertTrue(all(result["drug_like_score"] == 0.5))


class TestADMETFilters(unittest.TestCase):
    """Test cases for ADMET-based filtering functions."""

    def setUp(self):
        self.test_df = pd.DataFrame(
            {
                "SMILES": [
                    "CCO",
                    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
                    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
                    "invalid_smiles",
                ],
                "activity": [0.8, 0.6, 0.7, 0.1],
            }
        )

    def test_apply_admet_filters_basic(self):
        """Test basic ADMET filtering."""
        filtered_df = apply_admet_filters(self.test_df)

        self.assertIsInstance(filtered_df, pd.DataFrame)
        self.assertLessEqual(len(filtered_df), len(self.test_df))

        # Should have ADMET prediction columns added
        admet_columns = [
            "absorption",
            "bioavailability",
            "bbb_permeability",
            "hepatotoxicity",
            "mutagenicity",
            "drug_likeness",
        ]
        for col in admet_columns:
            self.assertIn(col, filtered_df.columns)

    def test_apply_admet_filters_strict(self):
        """Test strict ADMET filtering."""
        filtered_df = apply_admet_filters(self.test_df, strict=True)

        self.assertIsInstance(filtered_df, pd.DataFrame)
        self.assertLessEqual(len(filtered_df), len(self.test_df))

    def test_apply_admet_filters_no_smiles_column(self):
        """Test ADMET filtering when no SMILES column exists."""
        df_no_smiles = self.test_df.drop("SMILES", axis=1)
        filtered_df = apply_admet_filters(df_no_smiles)

        # Should return original dataframe if no SMILES column
        pd.testing.assert_frame_equal(filtered_df, df_no_smiles)

    def test_apply_admet_filters_empty_input(self):
        """Test ADMET filtering with empty input."""
        empty_df = pd.DataFrame({"SMILES": []})
        filtered_df = apply_admet_filters(empty_df)

        self.assertIsInstance(filtered_df, pd.DataFrame)
        self.assertEqual(len(filtered_df), 0)


class TestStandaloneFunctions(unittest.TestCase):
    """Test cases for standalone ADMET prediction functions."""

    def setUp(self):
        self.test_smiles = [
            "CCO",
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        ]

    def test_predict_admet_properties_function(self):
        """Test standalone predict_admet_properties function."""
        # Test with single SMILES
        result_single = predict_admet_properties("CCO")
        self.assertIsInstance(result_single, dict)

        # Test with list of SMILES
        result_list = predict_admet_properties(self.test_smiles)
        self.assertIsInstance(result_list, list)
        self.assertEqual(len(result_list), len(self.test_smiles))

        for result in result_list:
            self.assertIsInstance(result, dict)
            expected_keys = [
                "absorption",
                "bioavailability",
                "bbb_permeability",
                "cyp_inhibition",
                "hepatotoxicity",
                "mutagenicity",
                "drug_likeness",
            ]
            for key in expected_keys:
                self.assertIn(key, result)

    def test_predict_admet_properties_invalid_input(self):
        """Test standalone function with invalid input."""
        # Test with invalid SMILES
        result = predict_admet_properties("invalid_smiles")
        self.assertIsInstance(result, dict)

        # Test with empty input
        result_empty = predict_admet_properties([])
        self.assertIsInstance(result_empty, list)
        self.assertEqual(len(result_empty), 0)


class TestPerformance(unittest.TestCase):
    """Test performance with larger datasets."""

    def test_large_dataset_admet_prediction(self):
        """Test ADMET prediction with larger dataset."""
        # Generate larger test dataset
        large_smiles = ["CCO"] * 100  # 100 copies of ethanol

        predictor = ADMETPredictor()
        result = predictor.predict_admet_properties(large_smiles)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 100)

        # Check that all predictions are consistent
        first_row = result.iloc[0]
        for i in range(1, len(result)):
            for col in result.columns:
                if col != "SMILES":
                    if isinstance(first_row[col], dict):
                        continue  # Skip dict comparisons for CYP inhibition
                    self.assertAlmostEqual(
                        result.iloc[i][col], first_row[col], places=5
                    )

    def test_drug_likeness_assessment_performance(self):
        """Test drug-likeness assessment performance."""
        large_smiles = [
            "CCO",
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        ] * 50  # 150 molecules

        result = assess_drug_likeness(large_smiles)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 150)


class TestErrorHandling(unittest.TestCase):
    """Test error handling in ADMET prediction."""

    def test_prediction_with_none_input(self):
        """Test predictions with None input."""
        predictor = ADMETPredictor()

        # These should handle None gracefully
        absorption = predictor.predict_absorption(None)
        self.assertEqual(absorption, 0.0)

        bioavail = predictor.predict_bioavailability(None)
        self.assertEqual(bioavail, 0.0)

    def test_prediction_with_empty_string(self):
        """Test predictions with empty string."""
        predictor = ADMETPredictor()

        absorption = predictor.predict_absorption("")
        self.assertEqual(absorption, 0.0)

        drug_like = predictor.calculate_drug_likeness_score("")
        self.assertEqual(drug_like, 0.0)

    def test_malformed_smiles_handling(self):
        """Test handling of various malformed SMILES."""
        predictor = ADMETPredictor()
        malformed_smiles = [
            "C[C@H](C)C",  # Stereochemistry that might cause issues
            "C1=CC=CC=C1",  # Benzene
            "[Na+].[Cl-]",  # Salt
            "CC(=O)O",  # Acetic acid
        ]

        # Should not raise exceptions
        for smiles in malformed_smiles:
            try:
                result = predictor.predict_admet_properties([smiles])
                self.assertIsInstance(result, pd.DataFrame)
                self.assertEqual(len(result), 1)
            except Exception as e:
                self.fail(f"Exception raised for SMILES {smiles}: {e}")


if __name__ == "__main__":
    unittest.main()
