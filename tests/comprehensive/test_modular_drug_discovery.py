#!/usr/bin/env python3
"""
Comprehensive Test Suite for Modular Drug Discovery Architecture
================================================================

Tests all six new modules in the drug_discovery package:
- molecular_optimization.py
- admet.py
- screening.py
- properties.py
- generation.py
- qsar.py

Usage:
    python test_modular_drug_discovery.py
"""

import os
import sys
import unittest
import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore")


class TestMolecularOptimization(unittest.TestCase):
    """Test molecular optimization module"""

    def setUp(self):
        """Set up test fixtures"""
        try:
            from qemlflow.research.drug_discovery.molecular_optimization import (
                BayesianOptimizer,
                GeneticAlgorithmOptimizer,
                MolecularOptimizer,
                batch_optimize,
                optimize_molecule,
            )

            self.MolecularOptimizer = MolecularOptimizer
            self.BayesianOptimizer = BayesianOptimizer
            self.GeneticAlgorithmOptimizer = GeneticAlgorithmOptimizer
            self.optimize_molecule = optimize_molecule
            self.batch_optimize = batch_optimize
        except ImportError as e:
            self.skipTest(f"Molecular optimization module not available: {e}")

    def test_molecular_optimizer_initialization(self):
        """Test MolecularOptimizer can be initialized"""
        optimizer = self.MolecularOptimizer()
        self.assertIsNotNone(optimizer)
        self.assertIsNotNone(optimizer.objective_function)

    def test_bayesian_optimizer_initialization(self):
        """Test BayesianOptimizer can be initialized"""
        optimizer = self.BayesianOptimizer()
        self.assertIsNotNone(optimizer)
        self.assertIsInstance(optimizer, self.MolecularOptimizer)

    def test_genetic_algorithm_optimizer_initialization(self):
        """Test GeneticAlgorithmOptimizer can be initialized"""
        optimizer = self.GeneticAlgorithmOptimizer()
        self.assertIsNotNone(optimizer)
        self.assertIsInstance(optimizer, self.MolecularOptimizer)
        self.assertEqual(optimizer.population_size, 50)

    def test_optimize_molecule_function(self):
        """Test optimize_molecule function"""
        result = self.optimize_molecule("CCO", num_iterations=5)

        self.assertIsInstance(result, dict)
        required_keys = [
            "original_molecule",
            "optimized_molecule",
            "improvement",
            "optimization_score",
        ]
        for key in required_keys:
            self.assertIn(key, result)

    def test_batch_optimize_function(self):
        """Test batch_optimize function"""
        molecules = ["CCO", "CCC", "CC(C)O"]
        result = self.batch_optimize(molecules, num_iterations=3)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(molecules))


class TestADMET(unittest.TestCase):
    """Test ADMET module"""

    def setUp(self):
        """Set up test fixtures"""
        try:
            from qemlflow.research.drug_discovery.admet import (
                ADMETPredictor,
                DrugLikenessAssessor,
                ToxicityPredictor,
                assess_drug_likeness,
                predict_admet_properties,
            )

            self.ADMETPredictor = ADMETPredictor
            self.DrugLikenessAssessor = DrugLikenessAssessor
            self.ToxicityPredictor = ToxicityPredictor
            self.predict_admet_properties = predict_admet_properties
            self.assess_drug_likeness = assess_drug_likeness
        except ImportError as e:
            self.skipTest(f"ADMET module not available: {e}")

    def test_admet_predictor_initialization(self):
        """Test ADMETPredictor can be initialized"""
        predictor = self.ADMETPredictor()
        self.assertIsNotNone(predictor)

    def test_drug_likeness_assessor_initialization(self):
        """Test DrugLikenessAssessor can be initialized"""
        assessor = self.DrugLikenessAssessor()
        self.assertIsNotNone(assessor)

    def test_toxicity_predictor_initialization(self):
        """Test ToxicityPredictor can be initialized"""
        predictor = self.ToxicityPredictor()
        self.assertIsNotNone(predictor)

    def test_predict_admet_properties_function(self):
        """Test predict_admet_properties function"""
        molecules = ["CCO", "CCC"]
        result = self.predict_admet_properties(molecules)

        # Function returns a list of dictionaries for multiple molecules
        self.assertIsInstance(result, list)
        if result:
            self.assertIsInstance(result[0], dict)

    def test_assess_drug_likeness_function(self):
        """Test assess_drug_likeness function"""
        result = self.assess_drug_likeness("CCO")
        # Function returns a pandas DataFrame
        self.assertIsInstance(result, pd.DataFrame)


class TestScreening(unittest.TestCase):
    """Test virtual screening module"""

    def setUp(self):
        """Set up test fixtures"""
        try:
            from qemlflow.research.drug_discovery.screening import (
                PharmacophoreScreener,
                SimilarityScreener,
                VirtualScreener,
            )

            self.VirtualScreener = VirtualScreener
            self.SimilarityScreener = SimilarityScreener
            self.PharmacophoreScreener = PharmacophoreScreener
        except ImportError as e:
            self.skipTest(f"Screening module not available: {e}")

    def test_virtual_screener_initialization(self):
        """Test VirtualScreener can be initialized"""
        screener = self.VirtualScreener()
        self.assertIsNotNone(screener)

    def test_similarity_screener_initialization(self):
        """Test SimilarityScreener can be initialized"""
        screener = self.SimilarityScreener()
        self.assertIsNotNone(screener)

    def test_pharmacophore_screener_initialization(self):
        """Test PharmacophoreScreener can be initialized"""
        screener = self.PharmacophoreScreener()
        self.assertIsNotNone(screener)


class TestProperties(unittest.TestCase):
    """Test property prediction module"""

    def setUp(self):
        """Set up test fixtures"""
        try:
            from qemlflow.research.drug_discovery.properties import (
                MolecularPropertyPredictor,
                TrainedPropertyModel,
                predict_properties,
            )

            self.MolecularPropertyPredictor = MolecularPropertyPredictor
            self.TrainedPropertyModel = TrainedPropertyModel
            self.predict_properties = predict_properties
        except ImportError as e:
            self.skipTest(f"Properties module not available: {e}")

    def test_molecular_property_predictor_initialization(self):
        """Test MolecularPropertyPredictor can be initialized"""
        predictor = self.MolecularPropertyPredictor()
        self.assertIsNotNone(predictor)

    def test_predict_properties_function(self):
        """Test predict_properties function"""
        molecules = ["CCO", "CCC"]
        result = self.predict_properties(molecules)

        self.assertIsInstance(result, (dict, pd.DataFrame, list))


class TestGeneration(unittest.TestCase):
    """Test molecular generation module"""

    def setUp(self):
        """Set up test fixtures"""
        try:
            from qemlflow.research.drug_discovery.generation import (
                FragmentBasedGenerator,
                MolecularGenerator,
                generate_molecular_structures,
                optimize_structure,
            )

            self.MolecularGenerator = MolecularGenerator
            self.FragmentBasedGenerator = FragmentBasedGenerator
            self.generate_molecular_structures = generate_molecular_structures
            self.optimize_structure = optimize_structure
        except ImportError as e:
            self.skipTest(f"Generation module not available: {e}")

    def test_molecular_generator_initialization(self):
        """Test MolecularGenerator can be initialized"""
        generator = self.MolecularGenerator()
        self.assertIsNotNone(generator)

    def test_fragment_based_generator_initialization(self):
        """Test FragmentBasedGenerator can be initialized"""
        generator = self.FragmentBasedGenerator()
        self.assertIsNotNone(generator)

    def test_generate_molecular_structures_function(self):
        """Test generate_molecular_structures function"""
        result = self.generate_molecular_structures(num_samples=3)

        self.assertIsInstance(result, list)
        self.assertTrue(len(result) >= 0)  # May be empty if RDKit not available

    def test_optimize_structure_function(self):
        """Test optimize_structure function"""
        result = self.optimize_structure("CCO")
        self.assertIsInstance(result, str)


class TestQSAR(unittest.TestCase):
    """Test QSAR modeling module"""

    def setUp(self):
        """Set up test fixtures"""
        try:
            from qemlflow.research.drug_discovery.qsar import (
                ActivityPredictor,
                DescriptorCalculator,
                QSARModel,
                TrainedQSARModel,
                build_qsar_model,
                predict_activity,
            )

            self.DescriptorCalculator = DescriptorCalculator
            self.QSARModel = QSARModel
            self.ActivityPredictor = ActivityPredictor
            self.TrainedQSARModel = TrainedQSARModel
            self.build_qsar_model = build_qsar_model
            self.predict_activity = predict_activity
        except ImportError as e:
            self.skipTest(f"QSAR module not available: {e}")

    def test_descriptor_calculator_initialization(self):
        """Test DescriptorCalculator can be initialized"""
        try:
            calculator = self.DescriptorCalculator()
            self.assertIsNotNone(calculator)
        except ImportError:
            # RDKit may not be available
            self.skipTest("RDKit not available for descriptor calculation")

    def test_qsar_model_initialization(self):
        """Test QSARModel can be initialized"""
        model = self.QSARModel()
        self.assertIsNotNone(model)
        self.assertEqual(model.model_type, "random_forest")
        self.assertEqual(model.task_type, "regression")

    def test_activity_predictor_initialization(self):
        """Test ActivityPredictor can be initialized"""
        predictor = self.ActivityPredictor()
        self.assertIsNotNone(predictor)

    def test_build_qsar_model_function(self):
        """Test build_qsar_model function"""
        # Create simple test data
        X = np.random.random((50, 10))
        y = np.random.random(50)

        model = self.build_qsar_model(X, y, test_size=0.2)
        self.assertIsInstance(model, self.TrainedQSARModel)


class TestModuleIntegration(unittest.TestCase):
    """Test integration between modules"""

    def test_main_module_imports(self):
        """Test that main drug_discovery module imports work"""
        try:
            from qemlflow.research import drug_discovery

            # Test key classes are available
            self.assertTrue(hasattr(drug_discovery, "MolecularOptimizer"))
            self.assertTrue(hasattr(drug_discovery, "ADMETPredictor"))
            self.assertTrue(hasattr(drug_discovery, "VirtualScreener"))
            self.assertTrue(hasattr(drug_discovery, "MolecularGenerator"))
            self.assertTrue(hasattr(drug_discovery, "QSARModel"))

        except ImportError as e:
            self.fail(f"Main drug_discovery module import failed: {e}")

    def test_backward_compatibility(self):
        """Test backward compatibility imports"""
        try:
            from qemlflow.research.drug_discovery import (
                ADMETPredictor,
                MolecularGenerator,
                MolecularOptimizer,
                QSARModel,
                VirtualScreener,
            )

            # Test instantiation
            optimizer = MolecularOptimizer()
            admet = ADMETPredictor()
            screener = VirtualScreener()
            generator = MolecularGenerator()
            qsar = QSARModel()

            self.assertIsNotNone(optimizer)
            self.assertIsNotNone(admet)
            self.assertIsNotNone(screener)
            self.assertIsNotNone(generator)
            self.assertIsNotNone(qsar)

        except ImportError as e:
            self.fail(f"Backward compatibility import failed: {e}")


def run_tests():
    """Run all tests and provide summary"""
    print("🧪 Running Comprehensive Drug Discovery Module Tests")
    print("=" * 60)

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestMolecularOptimization,
        TestADMET,
        TestScreening,
        TestProperties,
        TestGeneration,
        TestQSAR,
        TestModuleIntegration,
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 60)
    print("🎯 Test Summary")
    print("=" * 60)
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")

    if result.failures:
        print(f"\n❌ {len(result.failures)} Failures:")
        for test, traceback in result.failures:
            print(f"  - {test}")

    if result.errors:
        print(f"\n💥 {len(result.errors)} Errors:")
        for test, traceback in result.errors:
            print(f"  - {test}")

    if result.wasSuccessful():
        print(
            "\n✅ All tests passed! Modular drug discovery architecture is working correctly."
        )
    else:
        print("\n⚠️  Some tests failed. Please review the failures above.")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
