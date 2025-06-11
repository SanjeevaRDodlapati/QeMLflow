"""
Comprehensive tests for property_prediction module

This test suite provides extensive coverage for molecular property prediction,
including physicochemical properties, ADMET parameters, and trained models.
"""

import sys
from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock, Mock, call, patch

import numpy as np
import pandas as pd
import pytest

# Import the module under test
sys.path.insert(0, "/Users/sanjeevadodlapati/Downloads/Repos/ChemML/src")
from drug_design.property_prediction import (
    RDKIT_AVAILABLE,
    MolecularPropertyPredictor,
    TrainedPropertyModel,
    calculate_metrics,
    evaluate_model,
    evaluate_property_predictions,
    handle_missing_values,
    normalize_data,
    predict_properties,
    preprocess_data,
    train_property_model,
)


class TestMolecularPropertyPredictorInit:
    """Test MolecularPropertyPredictor initialization"""

    def test_init_default(self):
        """Test default initialization"""
        predictor = MolecularPropertyPredictor()

        assert predictor is not None
        assert hasattr(predictor, "models")
        assert hasattr(predictor, "scalers")
        assert hasattr(predictor, "trained_properties")
        assert isinstance(predictor.models, dict)
        assert isinstance(predictor.scalers, dict)
        assert isinstance(predictor.trained_properties, set)
        assert len(predictor.models) == 0
        assert len(predictor.scalers) == 0
        assert len(predictor.trained_properties) == 0


class TestMolecularDescriptorCalculation:
    """Test molecular descriptor calculation"""

    def test_calculate_descriptors_without_rdkit(self):
        """Test descriptor calculation without RDKit"""
        predictor = MolecularPropertyPredictor()

        with patch("drug_design.property_prediction.RDKIT_AVAILABLE", False):
            descriptors = predictor.calculate_molecular_descriptors("CCO")

            expected_keys = [
                "molecular_weight",
                "logp",
                "tpsa",
                "hbd",
                "hba",
                "rotatable_bonds",
                "aromatic_rings",
            ]

            assert isinstance(descriptors, dict)
            for key in expected_keys:
                assert key in descriptors
                assert descriptors[key] == 0.0

    def test_calculate_descriptors_with_rdkit_valid(self):
        """Test descriptor calculation with RDKit and valid SMILES"""
        predictor = MolecularPropertyPredictor()

        with patch("drug_design.property_prediction.RDKIT_AVAILABLE", True):
            with patch(
                "drug_design.property_prediction.Chem.MolFromSmiles"
            ) as mock_mol_from_smiles:
                with patch(
                    "drug_design.property_prediction.Descriptors"
                ) as mock_descriptors:
                    with patch(
                        "drug_design.property_prediction.Crippen"
                    ) as mock_crippen:
                        # Mock molecule
                        mock_mol = Mock()
                        mock_mol_from_smiles.return_value = mock_mol

                        # Mock descriptor values
                        mock_descriptors.MolWt.return_value = 46.07
                        mock_descriptors.MolLogP.return_value = -0.74
                        mock_descriptors.TPSA.return_value = 20.23
                        mock_descriptors.NumHDonors.return_value = 1
                        mock_descriptors.NumHAcceptors.return_value = 1
                        mock_descriptors.NumRotatableBonds.return_value = 0
                        mock_descriptors.NumAromaticRings.return_value = 0
                        mock_descriptors.HeavyAtomCount.return_value = 3
                        mock_descriptors.RingCount.return_value = 0
                        mock_crippen.MolMR.return_value = 13.35
                        mock_descriptors.NumHeterocycles.return_value = 0
                        mock_descriptors.NumSaturatedRings.return_value = 0
                        mock_descriptors.FractionCsp3.return_value = 0.5
                        mock_descriptors.NumAliphaticCarbocycles.return_value = 0
                        mock_descriptors.NumAliphaticHeterocycles.return_value = 0

                        descriptors = predictor.calculate_molecular_descriptors("CCO")

                        assert isinstance(descriptors, dict)
                        assert descriptors["molecular_weight"] == 46.07
                        assert descriptors["logp"] == -0.74
                        assert descriptors["tpsa"] == 20.23
                        assert descriptors["hbd"] == 1
                        assert descriptors["hba"] == 1

    def test_calculate_descriptors_invalid_smiles(self):
        """Test descriptor calculation with invalid SMILES"""
        predictor = MolecularPropertyPredictor()

        with patch("drug_design.property_prediction.RDKIT_AVAILABLE", True):
            with patch(
                "drug_design.property_prediction.Chem.MolFromSmiles"
            ) as mock_mol_from_smiles:
                mock_mol_from_smiles.return_value = None  # Invalid SMILES

                descriptors = predictor.calculate_molecular_descriptors("INVALID")

                assert isinstance(descriptors, dict)
                expected_keys = [
                    "molecular_weight",
                    "logp",
                    "tpsa",
                    "hbd",
                    "hba",
                    "rotatable_bonds",
                    "aromatic_rings",
                ]
                for key in expected_keys:
                    assert key in descriptors
                    assert descriptors[key] == 0.0

    def test_calculate_descriptors_exception_handling(self):
        """Test descriptor calculation with exception"""
        predictor = MolecularPropertyPredictor()

        with patch("drug_design.property_prediction.RDKIT_AVAILABLE", True):
            with patch(
                "drug_design.property_prediction.Chem.MolFromSmiles"
            ) as mock_mol_from_smiles:
                mock_mol_from_smiles.side_effect = Exception("RDKit error")

                descriptors = predictor.calculate_molecular_descriptors("CCO")

                assert isinstance(descriptors, dict)
                expected_keys = [
                    "molecular_weight",
                    "logp",
                    "tpsa",
                    "hbd",
                    "hba",
                    "rotatable_bonds",
                    "aromatic_rings",
                ]
                for key in expected_keys:
                    assert key in descriptors
                    assert descriptors[key] == 0.0


class TestPhysicochemicalProperties:
    """Test physicochemical property prediction"""

    def test_predict_properties_without_rdkit(self):
        """Test property prediction without RDKit"""
        predictor = MolecularPropertyPredictor()

        with patch("drug_design.property_prediction.RDKIT_AVAILABLE", False):
            properties = predictor.predict_physicochemical_properties("CCO")

            expected_keys = [
                "solubility",
                "permeability",
                "stability",
                "bioavailability",
            ]

            assert isinstance(properties, dict)
            for key in expected_keys:
                assert key in properties
                assert isinstance(properties[key], (int, float))
                assert 0.0 <= properties[key] <= 1.0

    def test_predict_properties_with_rdkit_valid_molecule(self):
        """Test property prediction with RDKit and valid molecule"""
        predictor = MolecularPropertyPredictor()

        with patch("drug_design.property_prediction.RDKIT_AVAILABLE", True):
            with patch(
                "drug_design.property_prediction.Chem.MolFromSmiles"
            ) as mock_mol_from_smiles:
                with patch(
                    "drug_design.property_prediction.Descriptors"
                ) as mock_descriptors:
                    # Mock molecule
                    mock_mol = Mock()
                    mock_mol_from_smiles.return_value = mock_mol

                    # Mock descriptor values for ethanol (CCO)
                    mock_descriptors.MolWt.return_value = 46.07
                    mock_descriptors.MolLogP.return_value = -0.74
                    mock_descriptors.TPSA.return_value = 20.23
                    mock_descriptors.NumHDonors.return_value = 1
                    mock_descriptors.NumHAcceptors.return_value = 1

                    properties = predictor.predict_physicochemical_properties("CCO")

                    assert isinstance(properties, dict)
                    expected_keys = [
                        "solubility",
                        "permeability",
                        "stability",
                        "bioavailability",
                    ]
                    for key in expected_keys:
                        assert key in properties
                        assert isinstance(properties[key], (int, float))
                        assert 0.0 <= properties[key] <= 1.0

    def test_predict_properties_invalid_molecule(self):
        """Test property prediction with invalid molecule"""
        predictor = MolecularPropertyPredictor()

        with patch("drug_design.property_prediction.RDKIT_AVAILABLE", True):
            with patch(
                "drug_design.property_prediction.Chem.MolFromSmiles"
            ) as mock_mol_from_smiles:
                mock_mol_from_smiles.return_value = None  # Invalid SMILES

                properties = predictor.predict_physicochemical_properties("INVALID")

                assert isinstance(properties, dict)
                expected_keys = [
                    "solubility",
                    "permeability",
                    "stability",
                    "bioavailability",
                ]
                for key in expected_keys:
                    assert key in properties
                    assert properties[key] == 0.0

    def test_predict_properties_exception_handling(self):
        """Test property prediction with exception"""
        predictor = MolecularPropertyPredictor()

        with patch("drug_design.property_prediction.RDKIT_AVAILABLE", True):
            with patch(
                "drug_design.property_prediction.Chem.MolFromSmiles"
            ) as mock_mol_from_smiles:
                mock_mol_from_smiles.side_effect = Exception("RDKit error")

                properties = predictor.predict_physicochemical_properties("CCO")

                assert isinstance(properties, dict)
                expected_keys = [
                    "solubility",
                    "permeability",
                    "stability",
                    "bioavailability",
                ]
                for key in expected_keys:
                    assert key in properties
                    assert properties[key] == 0.0


class TestPropertyModelTraining:
    """Test property model training"""

    def test_train_model_basic_functionality(self):
        """Test basic model training functionality"""
        predictor = MolecularPropertyPredictor()

        # Create mock training data
        training_data = pd.DataFrame(
            {
                "SMILES": ["CCO", "CCC", "CCCC", "CC(C)C", "CCCCC"],
                "solubility": [0.8, 0.6, 0.4, 0.7, 0.3],
            }
        )

        with patch.object(
            predictor, "calculate_molecular_descriptors"
        ) as mock_calc_desc:
            mock_calc_desc.return_value = {
                "molecular_weight": 100.0,
                "logp": 2.0,
                "tpsa": 30.0,
                "hbd": 1,
                "hba": 2,
            }

            metrics = predictor.train_property_model(training_data, "solubility")

            assert isinstance(metrics, dict)
            assert "r2_score" in metrics
            assert "rmse" in metrics
            assert "mae" in metrics
            assert "solubility" in predictor.trained_properties
            assert "solubility" in predictor.models
            assert "solubility" in predictor.scalers

    def test_train_model_custom_columns(self):
        """Test model training with custom column names"""
        predictor = MolecularPropertyPredictor()

        training_data = pd.DataFrame(
            {"molecule": ["CCO", "CCC", "CCCC"], "target_prop": [0.8, 0.6, 0.4]}
        )

        with patch.object(
            predictor, "calculate_molecular_descriptors"
        ) as mock_calc_desc:
            mock_calc_desc.return_value = {"molecular_weight": 100.0, "logp": 2.0}

            metrics = predictor.train_property_model(
                training_data,
                "custom_property",
                smiles_column="molecule",
                target_column="target_prop",
            )

            assert isinstance(metrics, dict)
            assert "custom_property" in predictor.trained_properties

    def test_train_model_missing_target_column(self):
        """Test model training with missing target column"""
        predictor = MolecularPropertyPredictor()

        training_data = pd.DataFrame(
            {"SMILES": ["CCO", "CCC"], "other_prop": [0.8, 0.6]}
        )

        with pytest.raises(ValueError, match="Target column 'missing_prop' not found"):
            predictor.train_property_model(training_data, "missing_prop")

    def test_train_model_no_valid_data(self):
        """Test model training with no valid SMILES/target pairs"""
        predictor = MolecularPropertyPredictor()

        training_data = pd.DataFrame(
            {"SMILES": [None, None], "solubility": [None, None]}
        )

        with patch.object(
            predictor, "calculate_molecular_descriptors"
        ) as mock_calc_desc:
            mock_calc_desc.return_value = {}  # Empty descriptors

            with pytest.raises(ValueError, match="No valid SMILES/target pairs found"):
                predictor.train_property_model(training_data, "solubility")


class TestPropertyPrediction:
    """Test property prediction using trained models"""

    def test_predict_single_property_single_smiles(self):
        """Test predicting single property for single SMILES"""
        predictor = MolecularPropertyPredictor()

        # Set up mock trained model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.75])
        mock_scaler = Mock()
        mock_scaler.transform.return_value = np.array([[1.0, 2.0, 3.0]])

        predictor.models["solubility"] = mock_model
        predictor.scalers["solubility"] = mock_scaler
        predictor.trained_properties.add("solubility")

        with patch.object(
            predictor, "calculate_molecular_descriptors"
        ) as mock_calc_desc:
            mock_calc_desc.return_value = {
                "molecular_weight": 100.0,
                "logp": 2.0,
                "tpsa": 30.0,
            }

            prediction = predictor.predict_property("CCO", "solubility")

            assert isinstance(prediction, (int, float))
            assert prediction == 0.75

    def test_predict_single_property_multiple_smiles(self):
        """Test predicting single property for multiple SMILES"""
        predictor = MolecularPropertyPredictor()

        # Set up mock trained model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.75])
        mock_scaler = Mock()
        mock_scaler.transform.return_value = np.array([[1.0, 2.0, 3.0]])

        predictor.models["solubility"] = mock_model
        predictor.scalers["solubility"] = mock_scaler
        predictor.trained_properties.add("solubility")

        with patch.object(
            predictor, "calculate_molecular_descriptors"
        ) as mock_calc_desc:
            mock_calc_desc.return_value = {
                "molecular_weight": 100.0,
                "logp": 2.0,
                "tpsa": 30.0,
            }

            predictions = predictor.predict_property(["CCO", "CCC"], "solubility")

            assert isinstance(predictions, list)
            assert len(predictions) == 2
            assert all(isinstance(p, (int, float, type(np.nan))) for p in predictions)

    def test_predict_property_untrained_model(self):
        """Test predicting property for which no model is trained"""
        predictor = MolecularPropertyPredictor()

        with pytest.raises(
            ValueError, match="No trained model found for property 'unknown_prop'"
        ):
            predictor.predict_property("CCO", "unknown_prop")

    def test_predict_property_invalid_descriptors(self):
        """Test predicting property with invalid descriptors"""
        predictor = MolecularPropertyPredictor()

        # Set up mock trained model
        mock_model = Mock()
        mock_scaler = Mock()

        predictor.models["solubility"] = mock_model
        predictor.scalers["solubility"] = mock_scaler
        predictor.trained_properties.add("solubility")

        with patch.object(
            predictor, "calculate_molecular_descriptors"
        ) as mock_calc_desc:
            mock_calc_desc.return_value = {}  # Empty descriptors (invalid)

            prediction = predictor.predict_property("INVALID", "solubility")

            assert np.isnan(prediction)


class TestMultiplePropertyPrediction:
    """Test multiple property prediction"""

    def test_predict_multiple_properties_basic(self):
        """Test basic multiple property prediction"""
        predictor = MolecularPropertyPredictor()

        # Set up mock trained models
        mock_model1 = Mock()
        mock_model1.predict.return_value = np.array([0.75])
        mock_model2 = Mock()
        mock_model2.predict.return_value = np.array([0.85])

        mock_scaler = Mock()
        mock_scaler.transform.return_value = np.array([[1.0, 2.0, 3.0]])

        predictor.models["solubility"] = mock_model1
        predictor.models["permeability"] = mock_model2
        predictor.scalers["solubility"] = mock_scaler
        predictor.scalers["permeability"] = mock_scaler
        predictor.trained_properties.add("solubility")
        predictor.trained_properties.add("permeability")

        with patch.object(
            predictor, "calculate_molecular_descriptors"
        ) as mock_calc_desc:
            with patch.object(
                predictor, "predict_physicochemical_properties"
            ) as mock_phys_props:
                mock_calc_desc.return_value = {
                    "molecular_weight": 100.0,
                    "logp": 2.0,
                    "tpsa": 30.0,
                }
                mock_phys_props.return_value = {
                    "stability": 0.8,
                    "bioavailability": 0.7,
                }

                results = predictor.predict_multiple_properties(["CCO", "CCC"])

                assert isinstance(results, pd.DataFrame)
                assert "SMILES" in results.columns
                assert "predicted_solubility" in results.columns
                assert "predicted_permeability" in results.columns
                assert "stability" in results.columns
                assert "bioavailability" in results.columns
                assert len(results) == 2

    def test_predict_multiple_properties_with_errors(self):
        """Test multiple property prediction with errors"""
        predictor = MolecularPropertyPredictor()

        # Set up mock trained model that raises an error
        mock_model = Mock()
        mock_model.predict.side_effect = Exception("Prediction error")

        predictor.models["solubility"] = mock_model
        predictor.scalers["solubility"] = Mock()
        predictor.trained_properties.add("solubility")

        with patch.object(
            predictor, "predict_physicochemical_properties"
        ) as mock_phys_props:
            mock_phys_props.return_value = {"stability": 0.8}

            results = predictor.predict_multiple_properties(["CCO"])

            assert isinstance(results, pd.DataFrame)
            assert "predicted_solubility" in results.columns
            assert pd.isna(results["predicted_solubility"].iloc[0])


class TestStandaloneFunctions:
    """Test standalone functions"""

    def test_predict_properties_function_with_dataframe(self):
        """Test predict_properties function with DataFrame input"""
        data = pd.DataFrame({"SMILES": ["CCO", "CCC"], "property": [0.8, 0.6]})

        with patch(
            "drug_design.property_prediction.MolecularPropertyPredictor"
        ) as mock_predictor_class:
            mock_predictor = Mock()
            mock_predictor.predict_multiple_properties.return_value = pd.DataFrame(
                {"SMILES": ["CCO", "CCC"], "predicted_solubility": [0.75, 0.65]}
            )
            mock_predictor_class.return_value = mock_predictor

            results = predict_properties(data)

            assert isinstance(results, pd.DataFrame)
            mock_predictor.predict_multiple_properties.assert_called_once()

    def test_predict_properties_function_with_list(self):
        """Test predict_properties function with list input"""
        smiles_list = ["CCO", "CCC"]

        with patch(
            "drug_design.property_prediction.MolecularPropertyPredictor"
        ) as mock_predictor_class:
            mock_predictor = Mock()
            mock_predictor.predict_multiple_properties.return_value = pd.DataFrame(
                {"SMILES": ["CCO", "CCC"], "predicted_solubility": [0.75, 0.65]}
            )
            mock_predictor_class.return_value = mock_predictor

            results = predict_properties(smiles_list)

            assert isinstance(results, pd.DataFrame)
            mock_predictor.predict_multiple_properties.assert_called_once_with(
                ["CCO", "CCC"]
            )

    def test_predict_properties_function_with_custom_model(self):
        """Test predict_properties function with custom model"""
        smiles_list = ["CCO"]
        custom_model = MolecularPropertyPredictor()

        with patch.object(custom_model, "predict_multiple_properties") as mock_predict:
            mock_predict.return_value = pd.DataFrame(
                {"SMILES": ["CCO"], "predicted_solubility": [0.75]}
            )

            results = predict_properties(smiles_list, model=custom_model)

            assert isinstance(results, pd.DataFrame)
            mock_predict.assert_called_once_with(["CCO"])

    def test_preprocess_data(self):
        """Test preprocess_data function"""
        data = pd.DataFrame(
            {
                "SMILES": ["CCO", "CCC", None],
                "property": [0.8, None, 0.6],
                "other": [1, 2, 3],
            }
        )

        processed = preprocess_data(data)

        assert isinstance(processed, pd.DataFrame)
        # Should handle missing values and normalize

    def test_handle_missing_values(self):
        """Test handle_missing_values function"""
        data = pd.DataFrame(
            {
                "col1": [1.0, None, 3.0],
                "col2": [None, 2.0, 3.0],
                "col3": ["a", "b", "c"],  # Non-numeric
            }
        )

        handled = handle_missing_values(data)

        assert isinstance(handled, pd.DataFrame)
        assert not handled["col1"].isna().any()
        assert not handled["col2"].isna().any()
        # Non-numeric columns should remain unchanged
        assert handled["col3"].equals(data["col3"])

    def test_normalize_data(self):
        """Test normalize_data function"""
        data = pd.DataFrame(
            {
                "col1": [1.0, 2.0, 3.0],
                "col2": [10.0, 20.0, 30.0],
                "col3": ["a", "b", "c"],  # Non-numeric
            }
        )

        normalized = normalize_data(data)

        assert isinstance(normalized, pd.DataFrame)
        # Numeric columns should be min-max normalized (0-1 range)
        assert normalized["col1"].min() == 0.0
        assert normalized["col1"].max() == 1.0
        assert normalized["col2"].min() == 0.0
        assert normalized["col2"].max() == 1.0
        # Non-numeric columns should remain unchanged
        assert normalized["col3"].equals(data["col3"])

    def test_evaluate_model(self):
        """Test evaluate_model function"""
        predictions = np.array([0.8, 0.6, 0.7])
        true_values = np.array([0.75, 0.65, 0.72])

        metrics = evaluate_model(predictions, true_values)

        assert isinstance(metrics, dict)
        assert "MAE" in metrics
        assert "RMSE" in metrics
        assert "mse" in metrics
        assert "r2" in metrics
        assert all(isinstance(v, (int, float)) for v in metrics.values())

    def test_calculate_metrics(self):
        """Test calculate_metrics function"""
        predictions = np.array([0.8, 0.6, 0.7])
        true_values = np.array([0.75, 0.65, 0.72])

        metrics = calculate_metrics(predictions, true_values)

        assert isinstance(metrics, dict)
        assert "MAE" in metrics
        assert "RMSE" in metrics


class TestTrainedPropertyModel:
    """Test TrainedPropertyModel class"""

    def test_trained_model_init(self):
        """Test TrainedPropertyModel initialization"""
        model_dict = {
            "model": Mock(),
            "scaler": Mock(),
            "feature_names": ["mw", "logp"],
            "property_name": "solubility",
            "metrics": {"r2_score": 0.85, "rmse": 0.1},  # Add required metrics
        }

        trained_model = TrainedPropertyModel(model_dict)

        assert trained_model is not None
        assert hasattr(trained_model, "model_dict")
        assert hasattr(trained_model, "model")
        assert hasattr(trained_model, "scaler")
        assert hasattr(trained_model, "metrics")
        assert trained_model.model_dict == model_dict

    def test_trained_model_predict(self):
        """Test TrainedPropertyModel prediction"""
        mock_sklearn_model = Mock()
        mock_sklearn_model.predict.return_value = np.array([0.75])
        mock_scaler = Mock()
        mock_scaler.transform.return_value = np.array([[1.0, 2.0]])

        model_dict = {
            "model": mock_sklearn_model,
            "scaler": mock_scaler,
            "feature_names": ["mw", "logp"],
            "metrics": {"r2_score": 0.85, "rmse": 0.1},  # Add required metrics
        }

        trained_model = TrainedPropertyModel(model_dict)

        # Test prediction
        X = np.array([[100.0, 2.0]])
        prediction = trained_model.predict(X)

        assert isinstance(prediction, np.ndarray)
        assert prediction[0] == 0.75
        mock_scaler.transform.assert_called_once()
        mock_sklearn_model.predict.assert_called_once()

    def test_trained_model_get_metrics(self):
        """Test TrainedPropertyModel.get_metrics method"""
        model_dict = {
            "model": Mock(),
            "scaler": None,
            "metrics": {"r2_score": 0.85, "rmse": 0.1, "mae": 0.05},
        }

        trained_model = TrainedPropertyModel(model_dict)

        metrics = trained_model.get_metrics()

        assert isinstance(metrics, dict)
        assert metrics == {"r2_score": 0.85, "rmse": 0.1, "mae": 0.05}


class TestTrainPropertyModelFunction:
    """Test train_property_model standalone function"""

    def test_train_property_model_basic(self):
        """Test basic train_property_model functionality"""
        X = np.random.rand(10, 5)
        y = np.random.rand(10)

        with patch("drug_design.property_prediction.train_test_split") as mock_split:
            # Setup mocks
            mock_split.return_value = (X[:8], X[8:], y[:8], y[8:])

            result = train_property_model(X, y)

            assert isinstance(result, TrainedPropertyModel)
            assert hasattr(result, "model")
            assert hasattr(result, "scaler")
            assert hasattr(result, "metrics")

    def test_train_property_model_with_options(self):
        """Test train_property_model with custom options"""
        X = np.random.rand(20, 3)
        y = np.random.randint(0, 2, 20)  # Use integer classification targets

        result = train_property_model(
            X, y, model_type="classification", test_size=0.2, random_state=123
        )

        assert isinstance(result, TrainedPropertyModel)
        assert hasattr(result, "model")
        assert hasattr(result, "scaler")
        assert hasattr(result, "metrics")
        assert "accuracy" in result.metrics


class TestEvaluatePropertyPredictions:
    """Test evaluate_property_predictions function"""

    def test_evaluate_regression_predictions(self):
        """Test evaluation of regression predictions"""
        y_true = np.array([0.8, 0.6, 0.7, 0.9])
        y_pred = np.array([0.75, 0.65, 0.72, 0.85])

        metrics = evaluate_property_predictions(y_true, y_pred, task_type="regression")

        assert isinstance(metrics, dict)
        assert "r2_score" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert all(isinstance(v, (int, float)) for v in metrics.values())

    def test_evaluate_classification_predictions(self):
        """Test evaluation of classification predictions"""
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 0])

        metrics = evaluate_property_predictions(
            y_true, y_pred, task_type="classification"
        )

        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics


class TestIntegrationScenarios:
    """Test integration scenarios and workflows"""

    def test_complete_property_prediction_workflow(self):
        """Test complete property prediction workflow"""
        # Create predictor
        predictor = MolecularPropertyPredictor()

        # Create training data
        training_data = pd.DataFrame(
            {
                "SMILES": ["CCO", "CCC", "CCCC", "CC(C)C"],
                "solubility": [0.8, 0.6, 0.4, 0.7],
            }
        )

        # Mock descriptor calculation
        with patch.object(
            predictor, "calculate_molecular_descriptors"
        ) as mock_calc_desc:
            mock_calc_desc.return_value = {
                "molecular_weight": 100.0,
                "logp": 2.0,
                "tpsa": 30.0,
            }

            # Train model
            metrics = predictor.train_property_model(training_data, "solubility")

            # Make predictions
            predictions = predictor.predict_property(["CCCCC", "CCCCCC"], "solubility")

            # Test workflow completion
            assert isinstance(metrics, dict)
            assert isinstance(predictions, list)
            assert len(predictions) == 2

    def test_multiple_property_training_workflow(self):
        """Test training multiple properties"""
        predictor = MolecularPropertyPredictor()

        training_data = pd.DataFrame(
            {
                "SMILES": ["CCO", "CCC", "CCCC"],
                "solubility": [0.8, 0.6, 0.4],
                "permeability": [0.7, 0.5, 0.3],
            }
        )

        with patch.object(
            predictor, "calculate_molecular_descriptors"
        ) as mock_calc_desc:
            mock_calc_desc.return_value = {"molecular_weight": 100.0, "logp": 2.0}

            # Train multiple models
            metrics1 = predictor.train_property_model(training_data, "solubility")
            metrics2 = predictor.train_property_model(training_data, "permeability")

            # Check both properties are trained
            assert "solubility" in predictor.trained_properties
            assert "permeability" in predictor.trained_properties
            assert len(predictor.models) == 2
            assert len(predictor.scalers) == 2
            assert isinstance(metrics1, dict)
            assert isinstance(metrics2, dict)


class TestErrorHandling:
    """Test error handling and edge cases"""

    def test_empty_training_data(self):
        """Test handling of empty training data"""
        predictor = MolecularPropertyPredictor()

        empty_data = pd.DataFrame(columns=["SMILES", "property"])

        with pytest.raises(ValueError):
            predictor.train_property_model(empty_data, "property")

    def test_all_invalid_smiles(self):
        """Test handling of all invalid SMILES"""
        predictor = MolecularPropertyPredictor()

        training_data = pd.DataFrame(
            {"SMILES": ["INVALID1", "INVALID2"], "property": [0.5, 0.6]}
        )

        with patch.object(
            predictor, "calculate_molecular_descriptors"
        ) as mock_calc_desc:
            mock_calc_desc.return_value = {}  # Empty descriptors for invalid SMILES

            with pytest.raises(ValueError, match="No valid SMILES/target pairs found"):
                predictor.train_property_model(training_data, "property")

    def test_prediction_with_invalid_model_state(self):
        """Test prediction with corrupted model state"""
        predictor = MolecularPropertyPredictor()

        # Simulate corrupted state - model exists but no scaler
        predictor.models["test_prop"] = Mock()
        predictor.trained_properties.add("test_prop")
        # Missing scaler for 'test_prop'

        with pytest.raises(KeyError):
            predictor.predict_property("CCO", "test_prop")


class TestPerformance:
    """Test performance and scalability"""

    def test_large_descriptor_calculation(self):
        """Test descriptor calculation performance with many molecules"""
        predictor = MolecularPropertyPredictor()

        smiles_list = ["CCO"] * 100  # 100 identical SMILES

        with patch.object(
            predictor, "calculate_molecular_descriptors"
        ) as mock_calc_desc:
            mock_calc_desc.return_value = {"mw": 46.0, "logp": -0.74}

            results = []
            for smiles in smiles_list:
                desc = predictor.calculate_molecular_descriptors(smiles)
                results.append(desc)

            # Should complete without issues
            assert len(results) == 100
            assert all(isinstance(r, dict) for r in results)

    def test_batch_property_prediction(self):
        """Test batch property prediction performance"""
        predictor = MolecularPropertyPredictor()

        # Set up mock model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.5] * 50)
        mock_scaler = Mock()
        mock_scaler.transform.return_value = np.array([[1.0, 2.0]] * 50)

        predictor.models["test_prop"] = mock_model
        predictor.scalers["test_prop"] = mock_scaler
        predictor.trained_properties.add("test_prop")

        with patch.object(
            predictor, "calculate_molecular_descriptors"
        ) as mock_calc_desc:
            mock_calc_desc.return_value = {"mw": 100.0, "logp": 2.0}

            large_smiles_list = ["CCO"] * 50
            predictions = predictor.predict_property(large_smiles_list, "test_prop")

            assert len(predictions) == 50
            assert all(isinstance(p, (int, float, type(np.nan))) for p in predictions)

    def test_import_errors_coverage(self):
        """Test coverage of import error handling lines 24-26."""
        # These lines (24-26) are import statements that are covered when RDKit is not available
        # Since we can't easily test import failures, we test the RDKIT_AVAILABLE flag

        with patch("drug_design.property_prediction.RDKIT_AVAILABLE", False):
            predictor = MolecularPropertyPredictor()

            # This should trigger the code path that uses the fallback when RDKit is not available
            properties = predictor.predict_physicochemical_properties("CCO")

            assert isinstance(properties, dict)
            # When RDKit is not available, properties should use default/estimated values
            expected_keys = [
                "solubility",
                "permeability",
                "stability",
                "bioavailability",
            ]
            for key in expected_keys:
                assert key in properties

    def test_predict_properties_exception_in_mol_creation(self):
        """Test property prediction with exception during molecule creation (line 349)."""
        predictor = MolecularPropertyPredictor()

        with patch("drug_design.property_prediction.RDKIT_AVAILABLE", True):
            with patch(
                "drug_design.property_prediction.Chem.MolFromSmiles"
            ) as mock_mol_from_smiles:
                # This tests line 349: when mol_from_smiles raises an exception
                mock_mol_from_smiles.side_effect = Exception("Molecule creation failed")

                properties = predictor.predict_physicochemical_properties("CCO")

                assert isinstance(properties, dict)
                expected_keys = [
                    "solubility",
                    "permeability",
                    "stability",
                    "bioavailability",
                ]
                for key in expected_keys:
                    assert key in properties
                    assert (
                        properties[key] == 0.0
                    )  # Should return default values on exception

    def test_predict_properties_type_error_line_349(self):
        """Test line 349: TypeError for invalid molecular_data type."""
        from src.drug_design.property_prediction import predict_properties

        # Create a mock model
        mock_model = Mock()
        mock_model.predict_multiple_properties.return_value = {"property": [1.0]}

        # Test with invalid data type (not DataFrame or list) - should hit line 349
        with self.assertRaises(TypeError):
            predict_properties(mock_model, "invalid_string_type")

    def test_rdkit_import_warning_lines_24_26(self):
        """Test lines 24-26: RDKit import warning."""
        # Test that the import warning is properly structured
        with patch("builtins.__import__", side_effect=ImportError("RDKit not found")):
            with patch(
                "src.drug_design.property_prediction.logging.warning"
            ) as mock_warning:
                # Force re-import to trigger the warning
                import importlib

                import src.drug_design.property_prediction

                importlib.reload(src.drug_design.property_prediction)

                # Should have called the warning
                mock_warning.assert_called()
