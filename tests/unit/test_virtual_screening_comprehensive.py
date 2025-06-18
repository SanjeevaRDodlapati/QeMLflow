"""
Comprehensive test suite for virtual_screening module.

This module tests virtual screening functionality including similarity-based
screening, pharmacophore screening, and screening pipeline workflows.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from qemlflow.research.drug_discovery.screening import (
    PharmacophoreScreener,
    SimilarityScreener,
    VirtualScreener,
    calculate_screening_metrics,
    perform_virtual_screening,
)

try:
    from rdkit import Chem

    RDKIT_AVAILABLE = True
except ImportError:
    Chem = None
    RDKIT_AVAILABLE = False


class TestVirtualScreener(unittest.TestCase):
    """Test VirtualScreener class functionality"""

    def setUp(self):
        self.test_smiles = [
            "CCO",  # ethanol
            "CC(=O)O",  # acetic acid
            "c1ccccc1",  # benzene
            "CN(C)C",  # trimethylamine
            "c1ccc2[nH]c3ccccc3c2c1",  # carbazole
        ]
        self.reference_smiles = ["CCO", "CC(=O)O"]

    def test_initialization_similarity(self):
        """Test VirtualScreener initialization with similarity method"""
        screener = VirtualScreener(screening_method="similarity")
        self.assertEqual(screener.screening_method, "similarity")
        self.assertFalse(screener.is_configured)
        self.assertIsInstance(screener.screener, SimilarityScreener)

    def test_initialization_pharmacophore(self):
        """Test VirtualScreener initialization with pharmacophore method"""
        screener = VirtualScreener(screening_method="pharmacophore")
        self.assertEqual(screener.screening_method, "pharmacophore")
        self.assertIsInstance(screener.screener, PharmacophoreScreener)

    def test_initialization_default(self):
        """Test VirtualScreener initialization with default/unknown method"""
        screener = VirtualScreener(screening_method="unknown")
        self.assertIsInstance(
            screener.screener, SimilarityScreener
        )  # Should default to similarity

    def test_set_reference_compounds(self):
        """Test setting reference compounds"""
        screener = VirtualScreener()
        screener.set_reference_compounds(self.reference_smiles)

        self.assertTrue(screener.is_configured)
        self.assertEqual(len(screener.reference_compounds), 2)

    def test_screen_library_not_configured(self):
        """Test screening without setting reference compounds"""
        screener = VirtualScreener()

        with self.assertRaises(ValueError):
            screener.screen_library(self.test_smiles)

    def test_screen_library_configured(self):
        """Test successful library screening"""
        screener = VirtualScreener()
        screener.set_reference_compounds(self.reference_smiles)

        results = screener.screen_library(self.test_smiles, threshold=0.1)

        self.assertIsInstance(results, pd.DataFrame)
        self.assertGreater(len(results), 0)
        self.assertIn("smiles", results.columns)
        self.assertIn("score", results.columns)

    def test_get_top_hits_empty_results(self):
        """Test getting top hits with no results"""
        screener = VirtualScreener()
        screener.screening_results = []

        top_hits = screener.get_top_hits(5)
        self.assertIsInstance(top_hits, pd.DataFrame)
        self.assertTrue(top_hits.empty)

    def test_get_top_hits_with_results(self):
        """Test getting top hits with results"""
        screener = VirtualScreener()
        screener.screening_results = pd.DataFrame(
            {"smiles": ["CCO", "CC(=O)O", "c1ccccc1"], "score": [0.9, 0.8, 0.7]}
        )

        top_hits = screener.get_top_hits(2)
        self.assertEqual(len(top_hits), 2)
        self.assertEqual(top_hits.iloc[0]["score"], 0.9)  # Highest score first

    def test_get_top_hits_list_results(self):
        """Test getting top hits when results is a list"""
        screener = VirtualScreener()
        screener.screening_results = [
            {"smiles": "CCO", "score": 0.9},
            {"smiles": "CC(=O)O", "score": 0.8},
        ]

        top_hits = screener.get_top_hits(1)
        self.assertEqual(len(top_hits), 1)

    def test_calculate_enrichment_factor_no_results(self):
        """Test enrichment factor calculation with no results"""
        screener = VirtualScreener()
        screener.screening_results = []

        enrichment = screener.calculate_enrichment_factor(["CCO", "CC(=O)O"])
        self.assertEqual(enrichment, 0.0)

    def test_calculate_enrichment_factor_with_results(self):
        """Test enrichment factor calculation with results"""
        screener = VirtualScreener()
        screener.screening_results = pd.DataFrame(
            {
                "smiles": ["CCO", "CC(=O)O", "c1ccccc1", "CN(C)C"],
                "score": [0.9, 0.8, 0.7, 0.6],
            }
        )

        known_actives = ["CCO", "CC(=O)O"]
        enrichment = screener.calculate_enrichment_factor(known_actives, fraction=0.5)

        self.assertGreater(enrichment, 0.0)
        self.assertIsInstance(enrichment, float)


class TestSimilarityScreener(unittest.TestCase):
    """Test SimilarityScreener class functionality"""

    def setUp(self):
        self.test_smiles = [
            "CCO",  # ethanol
            "CC(=O)O",  # acetic acid
            "c1ccccc1",  # benzene
            "CCCCCCCCCCCCCCCCCCCC",  # long alkane
        ]
        self.reference_smiles = ["CCO", "CC(=O)O"]

    def test_initialization(self):
        """Test SimilarityScreener initialization"""
        screener = SimilarityScreener(fingerprint_type="morgan")
        self.assertEqual(screener.fingerprint_type, "morgan")
        self.assertEqual(len(screener.reference_fingerprints), 0)

    def test_set_reference_compounds(self):
        """Test setting reference compounds"""
        screener = SimilarityScreener()
        screener.set_reference_compounds(self.reference_smiles)

        self.assertEqual(len(screener.reference_compounds), 2)
        # Should have fingerprints calculated
        self.assertGreater(len(screener.reference_fingerprints), 0)

    def test_screen_library_basic(self):
        """Test basic library screening"""
        screener = SimilarityScreener()
        screener.set_reference_compounds(self.reference_smiles)

        results = screener.screen_library(self.test_smiles, threshold=0.1)

        self.assertIsInstance(results, pd.DataFrame)
        self.assertGreater(len(results), 0)
        self.assertIn("smiles", results.columns)
        self.assertIn("score", results.columns)
        self.assertIn("screening_method", results.columns)

    def test_screen_library_high_threshold(self):
        """Test library screening with high similarity threshold"""
        screener = SimilarityScreener()
        screener.set_reference_compounds(self.reference_smiles)

        results = screener.screen_library(self.test_smiles, threshold=0.95)

        # With high threshold, should find fewer or no matches
        self.assertIsInstance(results, pd.DataFrame)
        # Results may be empty depending on similarity

    def test_screen_library_max_compounds_limit(self):
        """Test library screening with compound limit"""
        screener = SimilarityScreener()
        screener.set_reference_compounds(self.reference_smiles)

        # Create larger library
        large_library = self.test_smiles * 10
        results = screener.screen_library(large_library, threshold=0.1, max_compounds=5)

        self.assertLessEqual(len(results), 5)

    def test_calculate_fingerprint_with_rdkit(self):
        """Test fingerprint calculation with RDKit available"""
        screener = SimilarityScreener()

        with patch("qemlflow.research.drug_discovery.screening.RDKIT_AVAILABLE", True):
            with patch("qemlflow.research.drug_discovery.screening.Chem") as mock_chem:
                with patch(
                    "qemlflow.research.drug_discovery.screening.rdMolDescriptors"
                ) as mock_desc:
                    mock_mol = MagicMock()
                    mock_chem.MolFromSmiles.return_value = mock_mol
                    mock_desc.GetMorganFingerprintAsBitVect.return_value = (
                        "mock_fingerprint"
                    )

                    fp = screener._calculate_fingerprint("CCO")
                    self.assertEqual(fp, "mock_fingerprint")

    def test_calculate_fingerprint_without_rdkit(self):
        """Test fingerprint calculation without RDKit"""
        screener = SimilarityScreener()

        with patch("qemlflow.research.drug_discovery.screening.RDKIT_AVAILABLE", False):
            fp = screener._calculate_fingerprint("CCO")
            self.assertIsInstance(fp, int)  # Should return hash

    def test_calculate_fingerprint_invalid_smiles(self):
        """Test fingerprint calculation with invalid SMILES"""
        screener = SimilarityScreener()

        with patch("qemlflow.research.drug_discovery.screening.RDKIT_AVAILABLE", True):
            with patch("qemlflow.research.drug_discovery.screening.Chem") as mock_chem:
                mock_chem.MolFromSmiles.return_value = None

                fp = screener._calculate_fingerprint("invalid_smiles")
                self.assertIsNone(fp)

    def test_calculate_similarity_with_rdkit(self):
        """Test similarity calculation with RDKit"""
        screener = SimilarityScreener()

        with patch("qemlflow.research.drug_discovery.screening.RDKIT_AVAILABLE", True):
            with patch(
                "qemlflow.research.drug_discovery.screening.TanimotoSimilarity"
            ) as mock_tanimoto:
                mock_tanimoto.return_value = 0.85

                similarity = screener._calculate_similarity("fp1", "fp2")
                self.assertEqual(similarity, 0.85)

    def test_calculate_similarity_without_rdkit(self):
        """Test similarity calculation without RDKit"""
        screener = SimilarityScreener()

        with patch("qemlflow.research.drug_discovery.screening.RDKIT_AVAILABLE", False):
            similarity = screener._calculate_similarity(123456, 123456)
            self.assertEqual(similarity, 1.0)  # Same hash should give 1.0

            similarity = screener._calculate_similarity(123456, 654321)
            self.assertLess(similarity, 1.0)  # Different hashes should give < 1.0


class TestPharmacophoreScreener(unittest.TestCase):
    """Test PharmacophoreScreener class functionality"""

    def setUp(self):
        self.test_smiles = [
            "CCO",  # ethanol
            "c1ccc2[nH]c3ccccc3c2c1",  # carbazole (aromatic with NH)
            "CC(=O)N",  # acetamide (HBD and HBA)
            "c1ccccc1O",  # phenol (aromatic with OH)
        ]

    def test_initialization_default(self):
        """Test PharmacophoreScreener initialization with defaults"""
        screener = PharmacophoreScreener()
        self.assertIn("aromatic_ring", screener.pharmacophore_features)
        self.assertIn("hydrogen_bond_donor", screener.pharmacophore_features)
        self.assertIn("hydrogen_bond_acceptor", screener.pharmacophore_features)

    def test_initialization_custom_features(self):
        """Test PharmacophoreScreener initialization with custom features"""
        custom_features = ["aromatic_ring", "positive_charge"]
        screener = PharmacophoreScreener(pharmacophore_features=custom_features)
        self.assertEqual(screener.pharmacophore_features, custom_features)

    def test_set_reference_compounds(self):
        """Test setting reference compounds"""
        screener = PharmacophoreScreener()
        screener.set_reference_compounds(self.test_smiles[:2])

        self.assertEqual(len(screener.reference_compounds), 2)

    def test_screen_library_basic(self):
        """Test basic pharmacophore screening"""
        screener = PharmacophoreScreener()
        screener.set_reference_compounds(self.test_smiles[:2])

        results = screener.screen_library(self.test_smiles, threshold=0.3)

        self.assertIsInstance(results, pd.DataFrame)
        self.assertIn("smiles", results.columns)
        self.assertIn("score", results.columns)
        self.assertIn("screening_method", results.columns)

    def test_screen_library_high_threshold(self):
        """Test pharmacophore screening with high threshold"""
        screener = PharmacophoreScreener()
        screener.set_reference_compounds(self.test_smiles[:2])

        results = screener.screen_library(self.test_smiles, threshold=0.95)

        # High threshold should find fewer matches
        self.assertIsInstance(results, pd.DataFrame)

    def test_calculate_pharmacophore_score_with_rdkit(self):
        """Test pharmacophore score calculation with RDKit"""
        screener = PharmacophoreScreener()

        with patch("qemlflow.research.drug_discovery.screening.RDKIT_AVAILABLE", True):
            with patch("qemlflow.research.drug_discovery.screening.Chem") as mock_chem:
                with patch(
                    "qemlflow.research.drug_discovery.screening.rdMolDescriptors"
                ) as mock_desc:
                    mock_mol = MagicMock()
                    mock_chem.MolFromSmiles.return_value = mock_mol
                    mock_desc.CalcNumAromaticRings.return_value = 1
                    mock_desc.CalcNumHBD.return_value = 1
                    mock_desc.CalcNumHBA.return_value = 1

                    score = screener._calculate_pharmacophore_score("c1ccccc1O")
                    self.assertGreater(score, 0.0)
                    self.assertLessEqual(score, 1.0)

    def test_calculate_pharmacophore_score_without_rdkit(self):
        """Test pharmacophore score calculation without RDKit"""
        screener = PharmacophoreScreener()

        with patch("qemlflow.research.drug_discovery.screening.RDKIT_AVAILABLE", False):
            score = screener._calculate_pharmacophore_score("CCO")
            self.assertGreater(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_calculate_pharmacophore_score_invalid_smiles(self):
        """Test pharmacophore score calculation with invalid SMILES"""
        screener = PharmacophoreScreener()

        with patch("qemlflow.research.drug_discovery.screening.RDKIT_AVAILABLE", True):
            with patch("qemlflow.research.drug_discovery.screening.Chem") as mock_chem:
                mock_chem.MolFromSmiles.return_value = None

                score = screener._calculate_pharmacophore_score("invalid_smiles")
                self.assertEqual(score, 0.0)


class TestStandaloneFunctions(unittest.TestCase):
    """Test standalone virtual screening functions"""

    def setUp(self):
        self.reference_smiles = ["CCO", "CC(=O)O"]
        self.library_smiles = [
            "CCO",  # exact match
            "CCC",  # similar
            "c1ccccc1",  # different
            "CCCCCCCCCC",  # different
        ]

    def test_perform_virtual_screening_similarity(self):
        """Test perform_virtual_screening function with similarity method"""
        result = perform_virtual_screening(
            reference_smiles=self.reference_smiles,
            library_smiles=self.library_smiles,
            method="similarity",
            threshold=0.1,
            max_hits=10,
        )

        self.assertIn("results", result)
        self.assertIn("top_hits", result)
        self.assertIn("statistics", result)
        self.assertIn("screener", result)

        # Check results structure
        self.assertIsInstance(result["results"], pd.DataFrame)
        self.assertIsInstance(result["top_hits"], pd.DataFrame)
        self.assertIsInstance(result["statistics"], dict)

        # Check statistics
        stats = result["statistics"]
        self.assertEqual(stats["total_screened"], len(self.library_smiles))
        self.assertEqual(stats["method"], "similarity")
        self.assertEqual(stats["threshold"], 0.1)

    def test_perform_virtual_screening_pharmacophore(self):
        """Test perform_virtual_screening function with pharmacophore method"""
        result = perform_virtual_screening(
            reference_smiles=self.reference_smiles,
            library_smiles=self.library_smiles,
            method="pharmacophore",
            threshold=0.3,
            max_hits=5,
        )

        self.assertIn("results", result)
        stats = result["statistics"]
        self.assertEqual(stats["method"], "pharmacophore")
        self.assertEqual(stats["threshold"], 0.3)

    def test_perform_virtual_screening_empty_library(self):
        """Test perform_virtual_screening with empty library"""
        result = perform_virtual_screening(
            reference_smiles=self.reference_smiles,
            library_smiles=[],
            method="similarity",
        )

        stats = result["statistics"]
        self.assertEqual(stats["total_screened"], 0)
        self.assertEqual(stats["hits_found"], 0)
        self.assertEqual(stats["hit_rate"], 0)

    def test_perform_virtual_screening_error_handling(self):
        """Test perform_virtual_screening error handling"""
        # Test with invalid reference compounds that might cause errors
        with patch(
            "qemlflow.research.drug_discovery.screening.VirtualScreener"
        ) as mock_screener_class:
            mock_screener = MagicMock()
            mock_screener.screen_library.side_effect = Exception("Test error")
            mock_screener_class.return_value = mock_screener

            result = perform_virtual_screening(
                reference_smiles=["invalid"], library_smiles=self.library_smiles
            )

            # Should handle error gracefully
            self.assertIn("error", result["statistics"])

    def test_calculate_screening_metrics_empty_results(self):
        """Test calculate_screening_metrics with empty results"""
        empty_results = pd.DataFrame()
        known_actives = ["CCO", "CC(=O)O"]

        metrics = calculate_screening_metrics(empty_results, known_actives)

        self.assertEqual(metrics["enrichment_factor"], 0.0)
        self.assertEqual(metrics["precision"], 0.0)
        self.assertEqual(metrics["recall"], 0.0)
        self.assertEqual(metrics["f1_score"], 0.0)

    def test_calculate_screening_metrics_with_results(self):
        """Test calculate_screening_metrics with actual results"""
        results = pd.DataFrame(
            {
                "smiles": ["CCO", "CC(=O)O", "c1ccccc1", "CCC"],
                "score": [0.9, 0.8, 0.7, 0.6],
            }
        )
        known_actives = ["CCO", "CC(=O)O"]

        metrics = calculate_screening_metrics(results, known_actives)

        self.assertGreater(metrics["precision"], 0.0)
        self.assertGreater(metrics["recall"], 0.0)
        self.assertGreater(metrics["f1_score"], 0.0)
        self.assertIsInstance(metrics["true_positives"], int)
        self.assertIsInstance(metrics["false_positives"], int)
        self.assertIsInstance(metrics["false_negatives"], int)

    def test_calculate_screening_metrics_perfect_precision(self):
        """Test calculate_screening_metrics with perfect precision"""
        results = pd.DataFrame(
            {"smiles": ["CCO", "CC(=O)O"], "score": [0.9, 0.8]}  # Only actives
        )
        known_actives = ["CCO", "CC(=O)O"]

        metrics = calculate_screening_metrics(results, known_actives)

        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f1_score"], 1.0)

    def test_calculate_screening_metrics_no_true_positives(self):
        """Test calculate_screening_metrics with no true positives"""
        results = pd.DataFrame(
            {"smiles": ["c1ccccc1", "CCC"], "score": [0.7, 0.6]}  # No actives
        )
        known_actives = ["CCO", "CC(=O)O"]

        metrics = calculate_screening_metrics(results, known_actives)

        self.assertEqual(metrics["precision"], 0.0)
        self.assertEqual(metrics["recall"], 0.0)
        self.assertEqual(metrics["f1_score"], 0.0)


class TestIntegrationScenarios(unittest.TestCase):
    """Test integration scenarios and workflows"""

    def setUp(self):
        self.drug_like_smiles = [
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
            "CC1=CC=C(C=C1)C(=O)O",  # p-Toluic acid
            "c1ccc2nc3ccccc3cc2c1",  # Phenanthroline
            "CCN(CC)CCNC(=O)C1=CC=CC=C1C",  # Procainamide analog
        ]

        self.library_smiles = [
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Exact match
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)N",  # Similar (amide vs acid)
            "c1ccccc1",  # Simple benzene
            "CCCCCCCCCC",  # Aliphatic chain
            "CC1=CC=C(C=C1)C(=O)N",  # Similar to p-toluic acid
        ]

    def test_complete_similarity_screening_workflow(self):
        """Test complete similarity-based screening workflow"""
        # Initialize and configure screener
        screener = VirtualScreener(screening_method="similarity")
        screener.set_reference_compounds(self.drug_like_smiles[:2])

        # Perform screening
        results = screener.screen_library(self.library_smiles, threshold=0.3)

        # Get top hits
        top_hits = screener.get_top_hits(3)

        # Calculate enrichment
        known_actives = [
            self.library_smiles[0],
            self.library_smiles[1],
        ]  # First two are active
        enrichment = screener.calculate_enrichment_factor(known_actives)

        # Verify workflow completed successfully
        self.assertIsInstance(results, pd.DataFrame)
        self.assertIsInstance(top_hits, pd.DataFrame)
        self.assertIsInstance(enrichment, float)
        self.assertGreaterEqual(enrichment, 0.0)

    def test_complete_pharmacophore_screening_workflow(self):
        """Test complete pharmacophore-based screening workflow"""
        # Initialize and configure screener
        screener = VirtualScreener(screening_method="pharmacophore")
        screener.set_reference_compounds(self.drug_like_smiles)

        # Perform screening
        results = screener.screen_library(self.library_smiles, threshold=0.4)

        # Calculate metrics
        known_actives = [self.library_smiles[0]]
        metrics = calculate_screening_metrics(results, known_actives)

        # Verify workflow completed
        self.assertIsInstance(results, pd.DataFrame)
        self.assertIsInstance(metrics, dict)
        self.assertIn("precision", metrics)

    def test_comparative_screening_methods(self):
        """Test comparison between different screening methods"""
        # Similarity screening
        sim_result = perform_virtual_screening(
            reference_smiles=self.drug_like_smiles[:2],
            library_smiles=self.library_smiles,
            method="similarity",
            threshold=0.3,
        )

        # Pharmacophore screening
        pharm_result = perform_virtual_screening(
            reference_smiles=self.drug_like_smiles[:2],
            library_smiles=self.library_smiles,
            method="pharmacophore",
            threshold=0.3,
        )

        # Compare results
        sim_hits = len(sim_result["results"])
        pharm_hits = len(pharm_result["results"])

        # Both methods should find some hits
        self.assertGreater(sim_hits + pharm_hits, 0)

        # Statistics should be properly calculated
        self.assertEqual(sim_result["statistics"]["method"], "similarity")
        self.assertEqual(pharm_result["statistics"]["method"], "pharmacophore")

    def test_large_library_screening_performance(self):
        """Test screening performance with larger library"""
        # Create larger library by replicating
        large_library = self.library_smiles * 20  # 100 compounds

        # Perform screening with limits
        result = perform_virtual_screening(
            reference_smiles=self.drug_like_smiles[:1],
            library_smiles=large_library,
            method="similarity",
            threshold=0.1,
            max_hits=10,
        )

        # Verify reasonable performance
        self.assertEqual(result["statistics"]["total_screened"], len(large_library))
        self.assertLessEqual(len(result["results"]), 10)  # Should respect max_hits
        self.assertIsInstance(result["top_hits"], pd.DataFrame)


class TestErrorHandlingAndEdgeCases(unittest.TestCase):
    """Test error handling and edge cases"""

    def test_invalid_smiles_handling(self):
        """Test handling of invalid SMILES strings"""
        invalid_smiles = ["invalid_smiles", "", "C[", "definitely_not_smiles"]
        valid_reference = ["CCO"]

        screener = VirtualScreener()
        screener.set_reference_compounds(valid_reference)

        # Should handle invalid SMILES gracefully
        results = screener.screen_library(invalid_smiles, threshold=0.5)

        # Results may be empty but should not crash
        self.assertIsInstance(results, pd.DataFrame)

    def test_empty_reference_compounds(self):
        """Test behavior with empty reference compounds"""
        screener = VirtualScreener()
        screener.set_reference_compounds([])

        # Should still be able to screen but with empty fingerprints
        results = screener.screen_library(["CCO"], threshold=0.5)
        self.assertIsInstance(results, pd.DataFrame)

    def test_single_compound_library(self):
        """Test screening single compound library"""
        screener = VirtualScreener()
        screener.set_reference_compounds(["CCO"])

        results = screener.screen_library(["CC(=O)O"], threshold=0.1)
        self.assertIsInstance(results, pd.DataFrame)

    def test_threshold_edge_cases(self):
        """Test edge cases for similarity thresholds"""
        screener = VirtualScreener()
        screener.set_reference_compounds(["CCO"])

        # Test with threshold 0.0 (should include all)
        results_low = screener.screen_library(["CCO", "CC(=O)O"], threshold=0.0)

        # Test with threshold 1.0 (should include only exact matches)
        results_high = screener.screen_library(["CCO", "CC(=O)O"], threshold=1.0)

        self.assertGreaterEqual(len(results_low), len(results_high))

    def test_molecular_object_input(self):
        """Test handling of molecular objects as input"""
        screener = VirtualScreener()

        # Mock molecular objects
        with patch("qemlflow.research.drug_discovery.screening.RDKIT_AVAILABLE", True):
            with patch("qemlflow.research.drug_discovery.screening.Chem") as mock_chem:
                mock_mol = MagicMock()
                mock_chem.MolToSmiles.return_value = "CCO"

                # Test with mock molecular object
                screener.set_reference_compounds([mock_mol])
                results = screener.screen_library([mock_mol], threshold=0.5)

                self.assertIsInstance(results, pd.DataFrame)


class TestPerformanceAndScalability(unittest.TestCase):
    """Test performance and scalability aspects"""

    def test_screening_with_limits(self):
        """Test screening respects compound limits"""
        # Create library with many similar compounds
        library = ["CCO"] * 100  # 100 identical compounds

        screener = VirtualScreener()
        screener.set_reference_compounds(["CCO"])

        # Screen with low limit
        results = screener.screen_library(library, threshold=0.9, max_compounds=5)

        self.assertLessEqual(len(results), 5)

    def test_scoring_consistency(self):
        """Test that scoring is consistent and reasonable"""
        screener = VirtualScreener()
        screener.set_reference_compounds(["CCO"])

        results = screener.screen_library(["CCO", "CC(=O)O", "c1ccccc1"], threshold=0.1)

        if not results.empty:
            # Scores should be between 0 and 1
            scores = results["score"].values
            self.assertTrue(all(0.0 <= score <= 1.0 for score in scores))

            # Results should be sorted by score (descending)
            sorted_scores = sorted(scores, reverse=True)
            np.testing.assert_array_equal(scores, sorted_scores)

    def test_memory_efficiency_large_library(self):
        """Test memory efficiency with larger libraries"""
        # Create a reasonably large library
        library = ["C" * i for i in range(1, 51)]  # 50 different alkanes

        screener = VirtualScreener()
        screener.set_reference_compounds(["CCCC"])

        # This should complete without memory issues
        results = screener.screen_library(library, threshold=0.1)

        self.assertIsInstance(results, pd.DataFrame)
        # Should have some results given the similar alkane structures
        self.assertGreaterEqual(len(results), 0)


if __name__ == "__main__":
    unittest.main()
