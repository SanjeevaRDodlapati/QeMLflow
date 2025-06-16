#!/usr/bin/env python3
"""
Comprehensive test suite for QSAR modeling module - FIXED VERSION

Covers all classes and functions in qsar_modeling.py with focus on:
- DescriptorCalculator class and methods
- QSARModel class and training pipeline
- ActivityPredictor class with multiple models
- Standalone functions: build_qsar_dataset, evaluate_qsar_model, build_qsar_model, etc.
- TrainedQSARModel wrapper class
- Error handling and edge cases
- Performance and integration testing

Fixed issues:
1. RDKit mocking - patch RDKit modules correctly
2. Cross-validation - use appropriate cv values for small datasets
3. Task type detection - handle regression vs classification properly
4. Data conversion - prevent SMILES conversion to float
5. Column naming - match expected column names
"""

import os
import tempfile
import unittest
import warnings
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd

# Suppress sklearn warnings for cleaner test output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class TestDescriptorCalculator(unittest.TestCase):
    """Test DescriptorCalculator class"""

    def setUp(self):
        self.sample_smiles = ["CCO", "CC(=O)O", "c1ccccc1"]

    @patch("chemml.research.drug_discovery.qsar.RDKIT_AVAILABLE", True)
    @patch("chemml.research.drug_discovery.qsar.Chem")
    def test_descriptor_calculator_rdkit_init(self, mock_chem):
        """Test DescriptorCalculator initialization with RDKit"""
        from chemml.research.drug_discovery.qsar import DescriptorCalculator

        calculator = DescriptorCalculator("rdkit")
        self.assertEqual(calculator.descriptor_set, "rdkit")

    @patch("chemml.research.drug_discovery.qsar.RDKIT_AVAILABLE", False)
    def test_descriptor_calculator_rdkit_unavailable(self):
        """Test DescriptorCalculator when RDKit is unavailable"""
        from chemml.research.drug_discovery.qsar import DescriptorCalculator

        with self.assertRaises(ImportError):
            DescriptorCalculator("rdkit")

    @patch("chemml.research.drug_discovery.qsar.MORDRED_AVAILABLE", False)
    def test_descriptor_calculator_mordred_unavailable(self):
        """Test DescriptorCalculator when Mordred is unavailable"""
        from chemml.research.drug_discovery.qsar import DescriptorCalculator

        with self.assertRaises(ImportError):
            DescriptorCalculator("mordred")

    @patch("chemml.research.drug_discovery.qsar.RDKIT_AVAILABLE", True)
    @patch("chemml.research.drug_discovery.qsar.Descriptors")
    @patch("chemml.research.drug_discovery.qsar.Chem")
    def test_calculate_rdkit_descriptors(self, mock_chem, mock_descriptors):
        """Test RDKit descriptor calculation"""
        from chemml.research.drug_discovery.qsar import DescriptorCalculator

        # Mock RDKit descriptors
        mock_descriptors._descList = [
            ("MolWt", lambda x: 46.069),
            ("LogP", lambda x: -0.1),
            ("TPSA", lambda x: 20.23),
        ]

        mock_mol = Mock()
        calculator = DescriptorCalculator("rdkit")

        result = calculator.calculate_rdkit_descriptors(mock_mol)

        expected = {"MolWt": 46.069, "LogP": -0.1, "TPSA": 20.23}
        self.assertEqual(result, expected)

    @patch("chemml.research.drug_discovery.qsar.MORDRED_AVAILABLE", True)
    @patch("chemml.research.drug_discovery.qsar.Calculator")
    @patch("chemml.research.drug_discovery.qsar.descriptors")
    def test_calculate_mordred_descriptors(self, mock_descriptors, mock_calculator):
        """Test Mordred descriptor calculation"""
        from chemml.research.drug_discovery.qsar import DescriptorCalculator

        # Mock Mordred calculator
        mock_calc_instance = Mock()
        mock_calculator.return_value = mock_calc_instance

        mock_result = Mock()
        mock_result.asdict.return_value = {
            "ABC": 2.5,
            "DEF": float("inf"),
            "GHI": float("nan"),
            "JKL": 1.2,
        }
        mock_calc_instance.return_value = mock_result

        calculator = DescriptorCalculator("mordred")
        mock_mol = Mock()

        result = calculator.calculate_mordred_descriptors(mock_mol)

        # Should filter out inf and nan values
        expected = {"ABC": 2.5, "JKL": 1.2}
        self.assertEqual(result, expected)

    @patch("chemml.research.drug_discovery.qsar.RDKIT_AVAILABLE", True)
    def test_calculate_fingerprint_descriptors_morgan(self):
        """Test Morgan fingerprint calculation - FIXED"""
        from chemml.research.drug_discovery.qsar import DescriptorCalculator

        calculator = DescriptorCalculator("rdkit")
        mock_mol = Mock()

        # Mock the local import and function calls
        with patch("rdkit.Chem.rdFingerprintGenerator") as mock_fp_gen:
            mock_generator = Mock()
            mock_fp_gen.GetMorganGenerator.return_value = mock_generator
            mock_generator.GetFingerprint.return_value = [1, 0, 1, 0, 1]

            result = calculator.calculate_fingerprint_descriptors(mock_mol, "morgan", 5)

            np.testing.assert_array_equal(result, np.array([1, 0, 1, 0, 1]))
            mock_fp_gen.GetMorganGenerator.assert_called_once_with(radius=2, fpSize=5)

    @patch("chemml.research.drug_discovery.qsar.RDKIT_AVAILABLE", True)
    def test_calculate_fingerprint_descriptors_maccs(self):
        """Test MACCS fingerprint calculation - FIXED"""
        from chemml.research.drug_discovery.qsar import DescriptorCalculator

        calculator = DescriptorCalculator("rdkit")
        mock_mol = Mock()

        # Mock the local import and function call
        with patch("rdkit.Chem.MACCSkeys.GenMACCSKeys") as mock_maccs:
            mock_maccs.return_value = [0, 1, 1, 0]

            result = calculator.calculate_fingerprint_descriptors(mock_mol, "maccs")

            np.testing.assert_array_equal(result, np.array([0, 1, 1, 0]))

    @patch("chemml.research.drug_discovery.qsar.RDKIT_AVAILABLE", True)
    def test_calculate_fingerprint_descriptors_unsupported(self):
        """Test unsupported fingerprint type"""
        from chemml.research.drug_discovery.qsar import DescriptorCalculator

        calculator = DescriptorCalculator("rdkit")
        mock_mol = Mock()

        with self.assertRaises(ValueError):
            calculator.calculate_fingerprint_descriptors(mock_mol, "unsupported")

    @patch("chemml.research.drug_discovery.qsar.RDKIT_AVAILABLE", True)
    @patch("chemml.research.drug_discovery.qsar.Chem")
    def test_calculate_descriptors_from_smiles_valid(self, mock_chem):
        """Test descriptor calculation from SMILES"""
        from chemml.research.drug_discovery.qsar import DescriptorCalculator

        # Mock valid molecules
        mock_mol1 = Mock()
        mock_mol2 = Mock()
        mock_chem.MolFromSmiles.side_effect = [mock_mol1, mock_mol2]

        calculator = DescriptorCalculator("rdkit")

        # Mock descriptor calculation
        with patch.object(calculator, "calculate_rdkit_descriptors") as mock_calc:
            mock_calc.side_effect = [
                {"MolWt": 46.069, "LogP": -0.1},
                {"MolWt": 60.052, "LogP": -0.2},
            ]

            result = calculator.calculate_descriptors_from_smiles(["CCO", "CC(=O)O"])

            self.assertEqual(len(result), 2)
            self.assertIn("SMILES", result.columns)
            self.assertIn("MolWt", result.columns)
            self.assertIn("LogP", result.columns)

    @patch("chemml.research.drug_discovery.qsar.RDKIT_AVAILABLE", True)
    @patch("chemml.research.drug_discovery.qsar.Chem")
    def test_calculate_descriptors_from_smiles_invalid(self, mock_chem):
        """Test descriptor calculation with invalid SMILES"""
        from chemml.research.drug_discovery.qsar import DescriptorCalculator

        # Mock invalid molecule
        mock_chem.MolFromSmiles.return_value = None

        calculator = DescriptorCalculator("rdkit")

        result = calculator.calculate_descriptors_from_smiles(["invalid_smiles"])

        self.assertEqual(len(result), 0)

    @patch("chemml.research.drug_discovery.qsar.RDKIT_AVAILABLE", True)
    @patch("chemml.research.drug_discovery.qsar.Chem")
    def test_calculate_descriptors_from_smiles_exception(self, mock_chem):
        """Test descriptor calculation with exceptions"""
        from chemml.research.drug_discovery.qsar import DescriptorCalculator

        mock_mol = Mock()
        mock_chem.MolFromSmiles.return_value = mock_mol

        calculator = DescriptorCalculator("rdkit")

        # Mock descriptor calculation to raise exception
        with patch.object(
            calculator,
            "calculate_rdkit_descriptors",
            side_effect=Exception("Test error"),
        ):
            result = calculator.calculate_descriptors_from_smiles(["CCO"])

            self.assertEqual(len(result), 0)


class TestQSARModel(unittest.TestCase):
    """Test QSARModel class"""

    def setUp(self):
        # Create larger dataset to avoid CV issues
        self.sample_data = pd.DataFrame(
            {
                "feature1": np.random.rand(20),
                "feature2": np.random.rand(20),
                "feature3": np.random.rand(20),
            }
        )
        self.sample_y_regression = np.random.rand(20)
        # Create balanced classification data
        self.sample_y_classification = np.array([0, 1] * 10)

    def test_qsar_model_init_regression(self):
        """Test QSARModel initialization for regression"""
        from chemml.research.drug_discovery.qsar import QSARModel

        model = QSARModel("random_forest", "regression")

        self.assertEqual(model.model_type, "random_forest")
        self.assertEqual(model.task_type, "regression")
        self.assertIsNotNone(model.model)

    def test_qsar_model_init_classification(self):
        """Test QSARModel initialization for classification"""
        from chemml.research.drug_discovery.qsar import QSARModel

        model = QSARModel("linear", "classification")

        self.assertEqual(model.model_type, "linear")
        self.assertEqual(model.task_type, "classification")
        self.assertIsNotNone(model.model)

    def test_qsar_model_unsupported_model_type(self):
        """Test QSARModel with unsupported model type"""
        from chemml.research.drug_discovery.qsar import QSARModel

        with self.assertRaises(ValueError):
            QSARModel("unsupported", "regression")

    def test_qsar_model_unsupported_task_type(self):
        """Test QSARModel with unsupported task type"""
        from chemml.research.drug_discovery.qsar import QSARModel

        with self.assertRaises(ValueError):
            QSARModel("random_forest", "unsupported")

    def test_prepare_data_basic(self):
        """Test data preparation"""
        from chemml.research.drug_discovery.qsar import QSARModel

        model = QSARModel("random_forest", "regression")

        X_prepared, y_prepared = model.prepare_data(
            self.sample_data, self.sample_y_regression
        )

        self.assertEqual(X_prepared.shape[0], len(self.sample_y_regression))
        np.testing.assert_array_equal(y_prepared, self.sample_y_regression)

    def test_prepare_data_with_missing_values(self):
        """Test data preparation with missing values"""
        from chemml.research.drug_discovery.qsar import QSARModel

        data_with_nan = self.sample_data.copy()
        data_with_nan.iloc[0, 1] = np.nan

        model = QSARModel("random_forest", "regression")

        X_prepared, y_prepared = model.prepare_data(
            data_with_nan, self.sample_y_regression
        )

        # Should handle NaN values
        self.assertFalse(np.isnan(X_prepared).any())

    def test_prepare_data_with_constant_features(self):
        """Test data preparation with constant features"""
        from chemml.research.drug_discovery.qsar import QSARModel

        data_with_constant = self.sample_data.copy()
        data_with_constant["constant_feature"] = 5.0  # Constant value

        model = QSARModel("random_forest", "regression")

        X_prepared, y_prepared = model.prepare_data(
            data_with_constant, self.sample_y_regression
        )

        # Should remove constant features
        self.assertLess(X_prepared.shape[1], data_with_constant.shape[1])

    def test_prepare_data_with_scaling(self):
        """Test data preparation with feature scaling"""
        from chemml.research.drug_discovery.qsar import QSARModel

        model = QSARModel("svm", "regression")

        X_prepared, y_prepared = model.prepare_data(
            self.sample_data, self.sample_y_regression, scale_features=True
        )

        self.assertIsNotNone(model.scaler)

    def test_train_regression_model(self):
        """Test training regression model"""
        from chemml.research.drug_discovery.qsar import QSARModel

        model = QSARModel("random_forest", "regression")

        metrics = model.train(
            self.sample_data, self.sample_y_regression, validation_split=0.2
        )

        self.assertIn("r2_score", metrics)
        self.assertIn("rmse", metrics)
        self.assertIn("mae", metrics)

    def test_train_classification_model(self):
        """Test training classification model - FIXED CV"""
        from chemml.research.drug_discovery.qsar import QSARModel

        model = QSARModel("random_forest", "classification")

        # Use smaller CV for small dataset
        metrics = model.train(
            self.sample_data, self.sample_y_classification, validation_split=0.2
        )

        self.assertIn("accuracy", metrics)
        self.assertIn("cross_val_score", metrics)

    def test_predict_before_training(self):
        """Test prediction before training"""
        from chemml.research.drug_discovery.qsar import QSARModel

        model = QSARModel("random_forest", "regression")

        with self.assertRaises(ValueError):
            model.predict(self.sample_data)

    def test_predict_after_training(self):
        """Test prediction after training"""
        from chemml.research.drug_discovery.qsar import QSARModel

        model = QSARModel("random_forest", "regression")
        model.train(self.sample_data, self.sample_y_regression, validation_split=0.2)

        predictions = model.predict(self.sample_data)

        self.assertEqual(len(predictions), len(self.sample_data))

    def test_predict_with_missing_features(self):
        """Test prediction with missing features - FIXED"""
        from chemml.research.drug_discovery.qsar import QSARModel

        model = QSARModel("random_forest", "regression")
        model.train(self.sample_data, self.sample_y_regression, validation_split=0.2)

        # Create data with missing column but handle gracefully
        incomplete_data = self.sample_data[["feature1", "feature2"]].copy()

        # The function should handle this by adding missing features with zeros
        predictions = model.predict(incomplete_data)

        self.assertEqual(len(predictions), len(incomplete_data))

    def test_get_feature_importance(self):
        """Test feature importance extraction"""
        from chemml.research.drug_discovery.qsar import QSARModel

        model = QSARModel("random_forest", "regression")
        model.train(self.sample_data, self.sample_y_regression, validation_split=0.2)

        importance_df = model.get_feature_importance()

        self.assertIn("feature", importance_df.columns)
        self.assertIn("importance", importance_df.columns)
        self.assertTrue((importance_df["importance"] >= 0).all())

    def test_get_feature_importance_unsupported(self):
        """Test feature importance with unsupported model"""
        from chemml.research.drug_discovery.qsar import QSARModel

        model = QSARModel("linear", "regression")
        model.train(self.sample_data, self.sample_y_regression, validation_split=0.2)

        with self.assertRaises(ValueError):
            model.get_feature_importance()

    def test_save_and_load_model(self):
        """Test model saving and loading"""
        from chemml.research.drug_discovery.qsar import QSARModel

        model = QSARModel("random_forest", "regression")
        model.train(self.sample_data, self.sample_y_regression, validation_split=0.2)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            try:
                model.save_model(tmp.name)

                # Create new model and load
                new_model = QSARModel("random_forest", "regression")
                new_model.load_model(tmp.name)

                # Test that loaded model works
                predictions = new_model.predict(self.sample_data)
                self.assertEqual(len(predictions), len(self.sample_data))

            finally:
                os.unlink(tmp.name)


class TestActivityPredictor(unittest.TestCase):
    """Test ActivityPredictor class"""

    def setUp(self):
        self.sample_smiles = ["CCO", "CC(=O)O"]

    @patch("chemml.research.drug_discovery.qsar.DescriptorCalculator")
    def test_activity_predictor_init(self, mock_calc_class):
        """Test ActivityPredictor initialization"""
        from chemml.research.drug_discovery.qsar import ActivityPredictor

        predictor = ActivityPredictor()

        self.assertEqual(len(predictor.models), 0)
        mock_calc_class.assert_called_once_with("rdkit")

    def test_add_model(self):
        """Test adding model to predictor"""
        from chemml.research.drug_discovery.qsar import ActivityPredictor, QSARModel

        predictor = ActivityPredictor()
        mock_model = Mock(spec=QSARModel)

        predictor.add_model("toxicity", mock_model)

        self.assertIn("toxicity", predictor.models)
        self.assertEqual(predictor.models["toxicity"], mock_model)

    @patch("chemml.research.drug_discovery.qsar.DescriptorCalculator")
    def test_predict_activity_valid(self, mock_calc_class):
        """Test activity prediction"""
        from chemml.research.drug_discovery.qsar import ActivityPredictor

        # Mock descriptor calculator
        mock_calc = Mock()
        mock_calc_class.return_value = mock_calc
        mock_descriptors_df = pd.DataFrame({"desc1": [1.0, 2.0], "desc2": [3.0, 4.0]})
        mock_calc.calculate_descriptors_from_smiles.return_value = mock_descriptors_df

        # Mock model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.8, 0.3])

        predictor = ActivityPredictor()
        predictor.add_model("toxicity", mock_model)

        result = predictor.predict_activity(self.sample_smiles, "toxicity")

        self.assertIn("SMILES", result.columns)
        self.assertIn("predicted_toxicity", result.columns)
        self.assertEqual(len(result), 2)

    def test_predict_activity_missing_model(self):
        """Test activity prediction with missing model"""
        from chemml.research.drug_discovery.qsar import ActivityPredictor

        predictor = ActivityPredictor()

        with self.assertRaises(ValueError):
            predictor.predict_activity(self.sample_smiles, "nonexistent")

    @patch("chemml.research.drug_discovery.qsar.DescriptorCalculator")
    def test_predict_multiple_activities(self, mock_calc_class):
        """Test prediction of multiple activities"""
        from chemml.research.drug_discovery.qsar import ActivityPredictor

        # Mock descriptor calculator
        mock_calc = Mock()
        mock_calc_class.return_value = mock_calc
        mock_descriptors_df = pd.DataFrame({"desc1": [1.0, 2.0], "desc2": [3.0, 4.0]})
        mock_calc.calculate_descriptors_from_smiles.return_value = mock_descriptors_df

        # Mock models
        mock_model1 = Mock()
        mock_model1.predict.return_value = np.array([0.8, 0.3])
        mock_model2 = Mock()
        mock_model2.predict.return_value = np.array([0.6, 0.9])

        predictor = ActivityPredictor()
        predictor.add_model("toxicity", mock_model1)
        predictor.add_model("solubility", mock_model2)

        result = predictor.predict_multiple_activities(self.sample_smiles)

        self.assertIn("SMILES", result.columns)
        self.assertIn("predicted_toxicity", result.columns)
        self.assertIn("predicted_solubility", result.columns)
        self.assertEqual(len(result), 2)

    @patch("chemml.research.drug_discovery.qsar.DescriptorCalculator")
    def test_predict_multiple_activities_with_error(self, mock_calc_class):
        """Test prediction with model error"""
        from chemml.research.drug_discovery.qsar import ActivityPredictor

        # Mock descriptor calculator
        mock_calc = Mock()
        mock_calc_class.return_value = mock_calc
        mock_descriptors_df = pd.DataFrame({"desc1": [1.0, 2.0], "desc2": [3.0, 4.0]})
        mock_calc.calculate_descriptors_from_smiles.return_value = mock_descriptors_df

        # Mock model that raises exception
        mock_model = Mock()
        mock_model.predict.side_effect = Exception("Model error")

        predictor = ActivityPredictor()
        predictor.add_model("toxicity", mock_model)

        result = predictor.predict_multiple_activities(self.sample_smiles)

        self.assertIn("predicted_toxicity", result.columns)
        self.assertTrue(result["predicted_toxicity"].isna().all())


class TestStandaloneFunctions(unittest.TestCase):
    """Test standalone functions"""

    def setUp(self):
        self.sample_smiles_data = pd.DataFrame(
            {"SMILES": ["CCO", "CC(=O)O", "c1ccccc1"], "Activity": [1.5, 2.0, 0.5]}
        )
        self.sample_X = np.random.rand(20, 5)  # Larger dataset
        self.sample_y_regression = np.random.rand(20)
        self.sample_y_classification = np.array([0, 1] * 10)  # Balanced

    @patch("chemml.research.drug_discovery.qsar.DescriptorCalculator")
    def test_build_qsar_dataset(self, mock_calc_class):
        """Test building QSAR dataset"""
        from chemml.research.drug_discovery.qsar import build_qsar_dataset

        # Mock descriptor calculator
        mock_calc = Mock()
        mock_calc_class.return_value = mock_calc

        mock_descriptors_df = pd.DataFrame(
            {
                "SMILES": ["CCO", "CC(=O)O", "c1ccccc1"],
                "desc1": [1.0, 2.0, 3.0],
                "desc2": [4.0, 5.0, 6.0],
            }
        )
        mock_calc.calculate_descriptors_from_smiles.return_value = mock_descriptors_df

        X, y = build_qsar_dataset(self.sample_smiles_data)

        self.assertEqual(X.shape[0], 3)
        self.assertEqual(len(y), 3)
        self.assertIn("desc1", X.columns)
        self.assertIn("desc2", X.columns)
        np.testing.assert_array_equal(y, self.sample_smiles_data["Activity"].values)

    @patch("chemml.research.drug_discovery.qsar.DescriptorCalculator")
    def test_build_qsar_dataset_custom_columns(self, mock_calc_class):
        """Test building QSAR dataset with custom column names - FIXED"""
        from chemml.research.drug_discovery.qsar import build_qsar_dataset

        # Custom data with different column names
        custom_data = pd.DataFrame(
            {"smiles_col": ["CCO", "CC(=O)O"], "target_col": [1.5, 2.0]}
        )

        # Mock descriptor calculator
        mock_calc = Mock()
        mock_calc_class.return_value = mock_calc

        # Mock descriptors with correct SMILES column name
        mock_descriptors_df = pd.DataFrame(
            {
                "SMILES": ["CCO", "CC(=O)O"],  # This is what the merge expects
                "desc1": [1.0, 2.0],
            }
        )
        mock_calc.calculate_descriptors_from_smiles.return_value = mock_descriptors_df

        X, y = build_qsar_dataset(custom_data, "smiles_col", "target_col")

        self.assertEqual(X.shape[0], 2)
        self.assertEqual(len(y), 2)

    def test_evaluate_qsar_model_regression(self):
        """Test QSAR model evaluation for regression"""
        from chemml.research.drug_discovery.qsar import QSARModel, evaluate_qsar_model

        # Create and train a simple model
        model = QSARModel("linear", "regression")
        sample_data = pd.DataFrame(
            {"feature1": np.random.rand(20), "feature2": np.random.rand(20)}
        )
        sample_y = np.random.rand(20)

        model.train(sample_data, sample_y, validation_split=0.2)

        # Evaluate model
        test_data = pd.DataFrame({"feature1": [1.5, 2.5], "feature2": [3.0, 5.0]})
        test_y = np.array([2.25, 3.75])

        metrics = evaluate_qsar_model(model, test_data, test_y)

        self.assertIn("r2_score", metrics)
        self.assertIn("rmse", metrics)
        self.assertIn("mae", metrics)
        self.assertIn("mse", metrics)

    def test_evaluate_qsar_model_classification(self):
        """Test QSAR model evaluation for classification - FIXED CV"""
        from chemml.research.drug_discovery.qsar import QSARModel, evaluate_qsar_model

        # Create and train a simple model with larger dataset
        model = QSARModel("linear", "classification")
        sample_data = pd.DataFrame(
            {"feature1": np.random.rand(20), "feature2": np.random.rand(20)}
        )
        sample_y = np.array([0, 1] * 10)  # Balanced classification

        model.train(sample_data, sample_y, validation_split=0.2)

        # Evaluate model
        test_data = pd.DataFrame({"feature1": [1.5, 2.5], "feature2": [3.0, 5.0]})
        test_y = np.array([0, 1])

        metrics = evaluate_qsar_model(model, test_data, test_y)

        self.assertIn("accuracy", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1_score", metrics)

    def test_build_qsar_model_regression(self):
        """Test build_qsar_model function for regression"""
        from chemml.research.drug_discovery.qsar import build_qsar_model

        trained_model = build_qsar_model(
            self.sample_X,
            self.sample_y_regression,
            model_type="random_forest",
            task_type="regression",
        )

        self.assertIsNotNone(trained_model)
        self.assertTrue(hasattr(trained_model, "predict"))

        # Test prediction
        predictions = trained_model.predict(self.sample_X[:3])
        self.assertEqual(len(predictions), 3)

    def test_build_qsar_model_classification(self):
        """Test build_qsar_model function for classification"""
        from chemml.research.drug_discovery.qsar import build_qsar_model

        trained_model = build_qsar_model(
            self.sample_X,
            self.sample_y_classification,
            model_type="linear",
            task_type="classification",
        )

        self.assertIsNotNone(trained_model)
        self.assertTrue(hasattr(trained_model, "predict"))

        # Test prediction
        predictions = trained_model.predict(self.sample_X[:3])
        self.assertEqual(len(predictions), 3)

    def test_build_qsar_model_different_types(self):
        """Test build_qsar_model with different model types"""
        from chemml.research.drug_discovery.qsar import build_qsar_model

        model_types = ["random_forest", "linear", "svm", "neural_network"]

        for model_type in model_types:
            with self.subTest(model_type=model_type):
                trained_model = build_qsar_model(
                    self.sample_X,
                    self.sample_y_regression,
                    model_type=model_type,
                    task_type="regression",
                )

                self.assertIsNotNone(trained_model)
                predictions = trained_model.predict(self.sample_X[:3])
                self.assertEqual(len(predictions), 3)

    def test_build_qsar_model_unknown_type(self):
        """Test build_qsar_model with unknown model type"""
        from chemml.research.drug_discovery.qsar import build_qsar_model

        with self.assertRaises(ValueError):
            build_qsar_model(
                self.sample_X,
                self.sample_y_regression,
                model_type="unknown",
                task_type="regression",
            )

    def test_predict_activity_function(self):
        """Test predict_activity standalone function"""
        from chemml.research.drug_discovery.qsar import (
            build_qsar_model,
            predict_activity,
        )

        # Build a model first
        trained_model = build_qsar_model(
            self.sample_X,
            self.sample_y_regression,
            model_type="linear",
            task_type="regression",
        )

        predictions = predict_activity(trained_model, self.sample_X[:3])

        self.assertEqual(len(predictions), 3)

    def test_predict_activity_legacy_format(self):
        """Test predict_activity with legacy dictionary format"""
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler

        from chemml.research.drug_discovery.qsar import predict_activity

        # Create legacy model format
        model = LinearRegression()
        model.fit(self.sample_X, self.sample_y_regression)

        scaler = StandardScaler()
        _X_scaled = scaler.fit_transform(self.sample_X)

        legacy_model = {"model": model, "scaler": scaler}

        predictions = predict_activity(legacy_model, self.sample_X[:3])

        self.assertEqual(len(predictions), 3)

    def test_validate_qsar_model_regression(self):
        """Test validate_qsar_model for regression - FIXED"""
        from chemml.research.drug_discovery.qsar import (
            build_qsar_model,
            validate_qsar_model,
        )

        # Build a model
        trained_model = build_qsar_model(
            self.sample_X,
            self.sample_y_regression,
            model_type="linear",
            task_type="regression",
        )

        # Validate model - let the function detect task type automatically
        metrics = validate_qsar_model(
            trained_model, self.sample_X, self.sample_y_regression
        )

        self.assertIn("r2_score", metrics)
        self.assertIn("rmse", metrics)
        self.assertIn("cv_scores", metrics)
        self.assertIn("mean_score", metrics)

    def test_validate_qsar_model_classification(self):
        """Test validate_qsar_model for classification - FIXED"""
        from chemml.research.drug_discovery.qsar import (
            build_qsar_model,
            validate_qsar_model,
        )

        # Build a model
        trained_model = build_qsar_model(
            self.sample_X,
            self.sample_y_classification,
            model_type="linear",
            task_type="classification",
        )

        # Validate model - let the function detect task type automatically
        metrics = validate_qsar_model(
            trained_model, self.sample_X, self.sample_y_classification
        )

        # Should only have classification metrics, not accuracy for regression
        self.assertIn("precision", metrics)
        self.assertIn("cv_scores", metrics)
        self.assertIn("mean_score", metrics)

    def test_validate_qsar_model_legacy_format(self):
        """Test validate_qsar_model with legacy format - FIXED"""
        from sklearn.linear_model import LinearRegression

        from chemml.research.drug_discovery.qsar import validate_qsar_model

        # Create legacy model format
        model = LinearRegression()
        model.fit(self.sample_X, self.sample_y_regression)

        legacy_model = {"model": model}

        metrics = validate_qsar_model(
            legacy_model, self.sample_X, self.sample_y_regression
        )

        self.assertIn("r2_score", metrics)
        self.assertIn("cv_scores", metrics)


class TestTrainedQSARModel(unittest.TestCase):
    """Test TrainedQSARModel wrapper class"""

    def setUp(self):
        self.sample_X = np.random.rand(10, 5)
        self.sample_y = np.random.rand(10)

    def test_trained_qsar_model_init(self):
        """Test TrainedQSARModel initialization"""
        from sklearn.linear_model import LinearRegression

        from chemml.research.drug_discovery.qsar import TrainedQSARModel

        model = LinearRegression()
        model.fit(self.sample_X, self.sample_y)

        model_dict = {
            "model": model,
            "scaler": None,
            "metrics": {"r2_score": 0.8},
            "feature_importances": None,
        }

        trained_model = TrainedQSARModel(model_dict)

        self.assertEqual(trained_model.model, model)
        self.assertIsNone(trained_model.scaler)
        self.assertEqual(trained_model.metrics["r2_score"], 0.8)

    def test_trained_qsar_model_with_importances(self):
        """Test TrainedQSARModel with feature importances"""
        from sklearn.ensemble import RandomForestRegressor

        from chemml.research.drug_discovery.qsar import TrainedQSARModel

        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(self.sample_X, self.sample_y)

        raw_importances = np.array([0.2, 0.3, 0.1, 0.4, 0.0])

        model_dict = {
            "model": model,
            "scaler": None,
            "metrics": {"r2_score": 0.8},
            "feature_importances": raw_importances,
        }

        trained_model = TrainedQSARModel(model_dict)

        # Check that importances are normalized
        self.assertIsNotNone(trained_model.feature_importances_)
        np.testing.assert_almost_equal(
            np.sum(trained_model.feature_importances_), 1.0, decimal=6
        )

    def test_trained_qsar_model_zero_importances(self):
        """Test TrainedQSARModel with zero feature importances"""
        from sklearn.linear_model import LinearRegression

        from chemml.research.drug_discovery.qsar import TrainedQSARModel

        model = LinearRegression()
        model.fit(self.sample_X, self.sample_y)

        # All zero importances
        zero_importances = np.zeros(5)

        model_dict = {
            "model": model,
            "scaler": None,
            "metrics": {"r2_score": 0.8},
            "feature_importances": zero_importances,
        }

        trained_model = TrainedQSARModel(model_dict)

        # Should create uniform distribution
        self.assertIsNotNone(trained_model.feature_importances_)
        np.testing.assert_almost_equal(
            np.sum(trained_model.feature_importances_), 1.0, decimal=6
        )
        self.assertTrue(np.all(trained_model.feature_importances_ == 0.2))  # 1/5 = 0.2

    def test_trained_qsar_model_predict(self):
        """Test TrainedQSARModel prediction"""
        from sklearn.linear_model import LinearRegression

        from chemml.research.drug_discovery.qsar import TrainedQSARModel

        model = LinearRegression()
        model.fit(self.sample_X, self.sample_y)

        model_dict = {
            "model": model,
            "scaler": None,
            "metrics": {"r2_score": 0.8},
            "feature_importances": None,
        }

        trained_model = TrainedQSARModel(model_dict)

        predictions = trained_model.predict(self.sample_X[:3])

        self.assertEqual(len(predictions), 3)

    def test_trained_qsar_model_predict_with_scaler(self):
        """Test TrainedQSARModel prediction with scaler"""
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler

        from chemml.research.drug_discovery.qsar import TrainedQSARModel

        # Train with scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.sample_X)

        model = LinearRegression()
        model.fit(X_scaled, self.sample_y)

        model_dict = {
            "model": model,
            "scaler": scaler,
            "metrics": {"r2_score": 0.8},
            "feature_importances": None,
        }

        trained_model = TrainedQSARModel(model_dict)

        predictions = trained_model.predict(self.sample_X[:3])

        self.assertEqual(len(predictions), 3)

    def test_trained_qsar_model_get_metrics(self):
        """Test TrainedQSARModel get_metrics"""
        from sklearn.linear_model import LinearRegression

        from chemml.research.drug_discovery.qsar import TrainedQSARModel

        model = LinearRegression()
        model.fit(self.sample_X, self.sample_y)

        test_metrics = {"r2_score": 0.8, "rmse": 0.2}

        model_dict = {
            "model": model,
            "scaler": None,
            "metrics": test_metrics,
            "feature_importances": None,
        }

        trained_model = TrainedQSARModel(model_dict)

        metrics = trained_model.get_metrics()

        self.assertEqual(metrics, test_metrics)


class TestIntegrationScenarios(unittest.TestCase):
    """Test integration scenarios"""

    def setUp(self):
        self.sample_data = pd.DataFrame(
            {
                "SMILES": ["CCO", "CC(=O)O", "c1ccccc1", "CCN", "CCC"],
                "Activity": [1.5, 2.0, 0.5, 1.8, 1.2],
            }
        )

    @patch("chemml.research.drug_discovery.qsar.DescriptorCalculator")
    def test_complete_qsar_workflow(self, mock_calc_class):
        """Test complete QSAR modeling workflow - FIXED"""
        from chemml.research.drug_discovery.qsar import (
            build_qsar_dataset,
            build_qsar_model,
            predict_activity,
            validate_qsar_model,
        )

        # Mock descriptor calculator
        mock_calc = Mock()
        mock_calc_class.return_value = mock_calc

        mock_descriptors_df = pd.DataFrame(
            {
                "SMILES": ["CCO", "CC(=O)O", "c1ccccc1", "CCN", "CCC"],
                "desc1": [1.0, 2.0, 3.0, 1.5, 1.2],
                "desc2": [4.0, 5.0, 6.0, 4.5, 4.2],
                "desc3": [0.5, 1.5, 2.5, 1.0, 0.8],
            }
        )
        mock_calc.calculate_descriptors_from_smiles.return_value = mock_descriptors_df

        # Build dataset
        X, y = build_qsar_dataset(self.sample_data)

        # Build model
        trained_model = build_qsar_model(
            X.values, y, model_type="random_forest", task_type="regression"
        )

        # Make predictions
        predictions = predict_activity(trained_model, X.values[:3])

        # Validate model - this should detect it's regression automatically
        metrics = validate_qsar_model(trained_model, X.values, y)

        self.assertEqual(len(predictions), 3)
        self.assertIn("r2_score", metrics)
        self.assertIn("cv_scores", metrics)

    @patch("chemml.research.drug_discovery.qsar.DescriptorCalculator")
    def test_activity_predictor_workflow(self, mock_calc_class):
        """Test ActivityPredictor workflow"""
        from chemml.research.drug_discovery.qsar import (
            ActivityPredictor,
            build_qsar_model,
        )

        # Mock descriptor calculator for both ActivityPredictor and build_qsar_model
        mock_calc = Mock()
        mock_calc_class.return_value = mock_calc

        mock_descriptors_df = pd.DataFrame(
            {
                "SMILES": ["CCO", "CC(=O)O", "c1ccccc1", "CCN", "CCC"],
                "desc1": [1.0, 2.0, 3.0, 1.5, 1.2],
                "desc2": [4.0, 5.0, 6.0, 4.5, 4.2],
            }
        )
        mock_calc.calculate_descriptors_from_smiles.return_value = mock_descriptors_df

        # Build training data
        X = np.random.rand(20, 2)
        y_tox = np.random.rand(20)
        y_sol = np.random.rand(20)

        # Train models
        toxicity_model = build_qsar_model(
            X, y_tox, model_type="linear", task_type="regression"
        )
        solubility_model = build_qsar_model(
            X, y_sol, model_type="linear", task_type="regression"
        )

        # Create predictor
        predictor = ActivityPredictor()
        predictor.add_model("toxicity", toxicity_model)
        predictor.add_model("solubility", solubility_model)

        # Predict multiple activities
        test_smiles = ["CCO", "CC(=O)O"]
        results = predictor.predict_multiple_activities(test_smiles)

        self.assertIn("SMILES", results.columns)
        self.assertIn("predicted_toxicity", results.columns)
        self.assertIn("predicted_solubility", results.columns)
        self.assertEqual(len(results), 2)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases"""

    def test_descriptor_calculator_empty_smiles(self):
        """Test DescriptorCalculator with empty SMILES list"""
        from chemml.research.drug_discovery.qsar import DescriptorCalculator

        with patch("chemml.research.drug_discovery.qsar.RDKIT_AVAILABLE", True):
            calculator = DescriptorCalculator("rdkit")

            result = calculator.calculate_descriptors_from_smiles([])

            self.assertEqual(len(result), 0)

    def test_qsar_model_empty_data(self):
        """Test QSARModel with empty data"""
        from chemml.research.drug_discovery.qsar import QSARModel

        model = QSARModel("random_forest", "regression")

        empty_data = pd.DataFrame()
        empty_y = np.array([])

        with self.assertRaises((ValueError, IndexError)):
            model.train(empty_data, empty_y)

    def test_qsar_model_single_sample(self):
        """Test QSARModel with single sample"""
        from chemml.research.drug_discovery.qsar import QSARModel

        model = QSARModel("random_forest", "regression")

        single_data = pd.DataFrame({"feature1": [1.0]})
        single_y = np.array([2.0])

        # Should handle gracefully or raise appropriate error
        try:
            metrics = model.train(single_data, single_y, validation_split=0.0)
            self.assertIsInstance(metrics, dict)
        except (ValueError, IndexError) as e:
            # Expected for very small datasets
            pass

    def test_predict_activity_invalid_model(self):
        """Test predict_activity with invalid model format - FIXED"""
        from chemml.research.drug_discovery.qsar import predict_activity

        invalid_model = "not_a_model"
        X = np.random.rand(3, 5)

        with self.assertRaises((AttributeError, TypeError)):
            predict_activity(invalid_model, X)

    def test_build_qsar_dataset_missing_columns(self):
        """Test build_qsar_dataset with missing columns"""
        from chemml.research.drug_discovery.qsar import build_qsar_dataset

        incomplete_data = pd.DataFrame({"SMILES": ["CCO"]})  # Missing Activity column

        with self.assertRaises(KeyError):
            build_qsar_dataset(incomplete_data, activity_column="Activity")


class TestPerformance(unittest.TestCase):
    """Test performance with larger datasets"""

    def test_large_dataset_qsar_modeling(self):
        """Test QSAR modeling with larger dataset"""
        from chemml.research.drug_discovery.qsar import build_qsar_model

        # Create larger dataset
        large_X = np.random.rand(100, 10)
        large_y = np.random.rand(100)

        trained_model = build_qsar_model(
            large_X, large_y, model_type="random_forest", task_type="regression"
        )

        # Test prediction
        predictions = trained_model.predict(large_X[:20])

        self.assertEqual(len(predictions), 20)

    @patch("chemml.research.drug_discovery.qsar.DescriptorCalculator")
    def test_batch_activity_prediction(self, mock_calc_class):
        """Test batch activity prediction performance - FIXED"""
        from chemml.research.drug_discovery.qsar import (
            ActivityPredictor,
            build_qsar_model,
        )

        # Mock descriptor calculator
        mock_calc = Mock()
        mock_calc_class.return_value = mock_calc

        # Create mock descriptors for batch of compounds
        batch_size = 50
        # FIXED: Create proper descriptor data without SMILES in numeric columns
        mock_descriptors_df = pd.DataFrame(
            {
                "SMILES": [f"SMILES_{i}" for i in range(batch_size)],
                "desc1": np.random.rand(batch_size),
                "desc2": np.random.rand(batch_size),
            }
        )
        mock_calc.calculate_descriptors_from_smiles.return_value = mock_descriptors_df

        # Train model
        X = np.random.rand(100, 2)
        y = np.random.rand(100)
        trained_model = build_qsar_model(
            X, y, model_type="linear", task_type="regression"
        )

        # Create predictor
        predictor = ActivityPredictor()
        predictor.add_model("activity", trained_model)

        # Predict for batch
        test_smiles = [f"SMILES_{i}" for i in range(batch_size)]
        results = predictor.predict_activity(test_smiles, "activity")

        self.assertEqual(len(results), batch_size)
        self.assertIn("predicted_activity", results.columns)


class TestCrossModuleCompatibility(unittest.TestCase):
    """Test compatibility with other modules"""

    def test_qsar_modeling_imports(self):
        """Test that all QSAR modeling imports work"""
        try:
            from chemml.research.drug_discovery.qsar import (
                ActivityPredictor,
                DescriptorCalculator,
                QSARModel,
                TrainedQSARModel,
                build_qsar_dataset,
                build_qsar_model,
                evaluate_qsar_model,
                predict_activity,
                validate_qsar_model,
            )

            self.assertTrue(True)  # If we get here, imports worked
        except ImportError as e:
            self.fail(f"Import failed: {e}")

    def test_availability_flags(self):
        """Test availability flags are properly set"""
        from src.drug_design import qsar_modeling

        # These should be boolean values
        self.assertIsInstance(qsar_modeling.RDKIT_AVAILABLE, bool)
        self.assertIsInstance(qsar_modeling.MORDRED_AVAILABLE, bool)

    def test_numpy_pandas_integration(self):
        """Test integration with numpy and pandas"""
        from chemml.research.drug_discovery.qsar import build_qsar_model

        # Test with different input types
        X_numpy = np.random.rand(10, 5)
        y_numpy = np.random.rand(10)

        trained_model = build_qsar_model(X_numpy, y_numpy)

        predictions = trained_model.predict(X_numpy[:3])

        self.assertIsInstance(predictions, np.ndarray)
        self.assertEqual(len(predictions), 3)


if __name__ == "__main__":
    unittest.main()
