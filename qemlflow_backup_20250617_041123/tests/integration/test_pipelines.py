"""
Integration tests for ChemML pipelines and workflows.

Tests end-to-end functionality and component integration.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Import all modules for integration testing
try:
    from chemml.research.drug_discovery.properties import (
        predict_properties,
        train_property_model,
    )
    from src.data_processing.feature_extraction import (
        extract_descriptors,
        generate_fingerprints,
    )
    from src.data_processing.molecular_preprocessing import clean_data, normalize_data
    from src.models.classical_ml.regression_models import RegressionModels
    from src.utils.io_utils import load_molecular_data, save_molecular_data
    from src.utils.ml_utils import evaluate_model, normalize_features, split_data
except ImportError as e:
    pytest.skip(f"Integration modules not available: {e}", allow_module_level=True)

from tests.conftest import skip_if_no_deepchem, skip_if_no_rdkit

try:
    from rdkit import Chem
except ImportError:
    pass


class TestEndToEndPipelines:
    """Test complete end-to-end workflows."""

    @skip_if_no_rdkit
    def test_molecular_ml_pipeline(self, sample_molecular_data, tmp_path):
        """Test complete molecular ML pipeline from data to predictions."""
        # Step 1: Data preprocessing
        cleaned_data = clean_data(sample_molecular_data)
        assert len(cleaned_data) > 0

        # Step 2: Feature extraction
        smiles_list = cleaned_data["smiles"].tolist()

        try:
            # Convert SMILES to molecules
            from rdkit import Chem

            molecules = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
            molecules = [mol for mol in molecules if mol is not None]

            # Extract molecular descriptors
            descriptors = extract_descriptors(molecules)
            fingerprints = generate_fingerprints(molecules)

            # Step 3: Combine features
            X = np.hstack([descriptors.values, fingerprints])
            y = cleaned_data["activity"].values[: len(X)]  # Ensure matching length

            # Step 4: Train model
            model = RegressionModels(model_type="linear")
            mse, r2 = model.train(X, y)

            # Step 5: Make predictions
            predictions = model.predict(X)

            # Step 6: Evaluate results
            assert isinstance(predictions, np.ndarray)
            assert len(predictions) == len(y)
            assert isinstance(mse, float)
            assert isinstance(r2, float)

            # Step 7: Save results
            results_df = pd.DataFrame(
                {
                    "smiles": smiles_list[: len(predictions)],
                    "true_activity": y,
                    "predicted_activity": predictions,
                }
            )

            output_path = tmp_path / "pipeline_results.csv"
            save_molecular_data(results_df, str(output_path))

            # Verify saved results
            assert output_path.exists()
            loaded_results = load_molecular_data(str(output_path))
            assert len(loaded_results) == len(predictions)

        except NotImplementedError as e:
            pytest.skip(f"Pipeline component not implemented: {e}")

    def test_property_prediction_pipeline(self, sample_molecular_data):
        """Test property prediction pipeline."""
        try:
            # Prepare features and targets
            feature_cols = ["molecular_weight", "logp", "tpsa"]
            X = sample_molecular_data[feature_cols].values
            y = sample_molecular_data["solubility"].values

            # Split data
            X_train, X_test, y_train, y_test = split_data(
                X, y, test_size=0.3, random_state=42
            )

            # Normalize features
            X_train_norm = normalize_features(X_train)
            X_test_norm = normalize_features(X_test)

            # Train property prediction model
            model = train_property_model(X_train_norm, y_train, model_type="regression")

            # Make predictions
            y_pred = model.predict(X_test_norm)

            # Evaluate model
            metrics = evaluate_model(model, X_test_norm, y_test, task_type="regression")

            # Verify results
            assert len(y_pred) == len(y_test)
            assert isinstance(metrics, dict)
            assert "mse" in metrics or "r2" in metrics

        except NotImplementedError as e:
            pytest.skip(f"Property prediction pipeline not implemented: {e}")

    @skip_if_no_rdkit
    def test_virtual_screening_pipeline(self, sample_molecular_data):
        """Test virtual screening workflow."""
        try:
            # Step 1: Prepare training data
            feature_cols = ["molecular_weight", "logp", "tpsa"]
            X_train = sample_molecular_data[feature_cols].values
            y_train = sample_molecular_data["activity"].values

            # Step 2: Train QSAR model
            from chemml.research.drug_discovery.qsar import (
                build_qsar_model,
                predict_activity,
            )

            qsar_model = build_qsar_model(X_train, y_train, model_type="random_forest")

            # Step 3: Generate virtual compound library
            np.random.seed(42)
            virtual_compounds = np.random.randn(100, 3)  # 100 virtual compounds

            # Step 4: Screen virtual library
            activities = predict_activity(qsar_model, virtual_compounds)

            # Step 5: Filter active compounds
            active_indices = np.where(activities == 1)[0]
            active_compounds = virtual_compounds[active_indices]

            # Step 6: Rank by predicted probability (if available)
            if hasattr(qsar_model, "predict_proba"):
                probabilities = qsar_model.predict_proba(virtual_compounds)[:, 1]
                ranked_indices = np.argsort(probabilities)[::-1]  # Descending order
                top_compounds = virtual_compounds[ranked_indices[:10]]  # Top 10

                assert len(top_compounds) == 10

            # Verify screening results
            assert isinstance(activities, np.ndarray)
            assert len(activities) == 100
            assert len(active_compounds) <= 100

        except NotImplementedError as e:
            pytest.skip(f"Virtual screening pipeline not implemented: {e}")

    def test_quantum_ml_integration(self, sample_regression_data):
        """Test integration of quantum ML components."""
        try:
            from src.models.quantum_ml.quantum_circuits import QuantumCircuit

            X, y, _ = sample_regression_data

            # Create quantum circuit
            qc = QuantumCircuit(n_qubits=2)

            # Encode classical data (if implemented)
            for i, data_point in enumerate(X[:5]):  # Test with first 5 points
                _encoded_circuit = qc.encode_classical_data(
                    data_point[:2]
                )  # Use first 2 features
                result = qc.simulate()

                assert result is not None
                assert "counts" in result or "statevector" in result

                if i >= 4:  # Limit test to avoid long runtime
                    break

        except (NotImplementedError, ImportError) as e:
            pytest.skip(f"Quantum ML integration not available: {e}")


class TestDataFlowIntegration:
    """Test data flow between different components."""

    def test_data_preprocessing_to_modeling(self, sample_molecular_data):
        """Test data flow from preprocessing to modeling."""
        # Original data
        original_shape = sample_molecular_data.shape

        # Clean data
        cleaned_data = clean_data(sample_molecular_data)

        # Normalize numerical columns
        numerical_cols = cleaned_data.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            normalized_subset = normalize_data(cleaned_data[numerical_cols])

            # Replace normalized columns in cleaned data
            for col in numerical_cols:
                cleaned_data[col] = normalized_subset[col]

        # Verify data integrity
        assert cleaned_data.shape[1] == original_shape[1]  # Same number of columns
        assert (
            len(cleaned_data) <= original_shape[0]
        )  # May have fewer rows due to cleaning

        # Prepare for modeling
        if "activity" in cleaned_data.columns and len(numerical_cols) > 1:
            feature_cols = [col for col in numerical_cols if col != "activity"]
            if feature_cols:
                X = cleaned_data[feature_cols].values
                y = cleaned_data["activity"].values

                # Train model
                model = RegressionModels(model_type="linear")
                mse, r2 = model.train(X, y)

                # Verify model training
                assert isinstance(mse, float)
                assert isinstance(r2, float)

    @skip_if_no_rdkit
    def test_feature_extraction_to_modeling(self, sample_smiles):
        """Test data flow from feature extraction to modeling."""
        try:
            # Convert SMILES to molecules
            from rdkit import Chem

            molecules = [Chem.MolFromSmiles(smiles) for smiles in sample_smiles]
            molecules = [mol for mol in molecules if mol is not None]

            if len(molecules) == 0:
                pytest.skip("No valid molecules available")

            # Extract multiple feature types
            descriptors = extract_descriptors(molecules)
            fingerprints = generate_fingerprints(molecules)

            # Combine features
            combined_features = np.hstack([descriptors.values, fingerprints])

            # Create synthetic target
            np.random.seed(42)
            y = np.random.randint(0, 2, len(combined_features))

            # Train model with combined features
            model = RegressionModels(model_type="linear")
            mse, r2 = model.train(combined_features, y)

            # Verify successful training
            assert isinstance(mse, float)
            assert isinstance(r2, float)
            assert combined_features.shape[0] == len(molecules)
            assert (
                combined_features.shape[1]
                == descriptors.shape[1] + fingerprints.shape[1]
            )

        except NotImplementedError as e:
            pytest.skip(f"Feature extraction not implemented: {e}")

    def test_model_output_to_analysis(self, sample_regression_data):
        """Test data flow from model output to analysis."""
        X, y, _ = sample_regression_data

        # Train model
        model = RegressionModels(model_type="linear")
        mse, r2 = model.train(X, y)

        # Get predictions
        predictions = model.predict(X)

        # Analyze predictions
        from src.utils.metrics import mean_squared_error, r_squared

        analysis_mse = mean_squared_error(y, predictions)
        analysis_r2 = r_squared(y, predictions)

        # Verify consistency
        assert isinstance(analysis_mse, float)
        assert isinstance(analysis_r2, float)
        assert analysis_mse >= 0

        # Create analysis report
        analysis_report = {
            "model_type": "linear",
            "training_mse": mse,
            "training_r2": r2,
            "analysis_mse": analysis_mse,
            "analysis_r2": analysis_r2,
            "n_samples": len(X),
            "n_features": X.shape[1],
        }

        # Verify report structure
        assert all(isinstance(v, (int, float, str)) for v in analysis_report.values())


class TestErrorPropagation:
    """Test error handling across integrated components."""

    def test_invalid_data_propagation(self):
        """Test how invalid data is handled through the pipeline."""
        # Create problematic data
        bad_data = pd.DataFrame(
            {
                "smiles": ["CCO", "invalid_smiles", "", None],
                "activity": [1, np.nan, 2, 3],
                "molecular_weight": [46.07, np.inf, -100, None],
            }
        )

        # Clean data should handle problems
        try:
            cleaned_data = clean_data(bad_data)

            # Should remove or fix problematic entries
            assert not cleaned_data.isnull().any().any()
            assert np.isfinite(
                cleaned_data.select_dtypes(include=[np.number]).values
            ).all()

        except Exception as e:
            # If cleaning fails, it should fail gracefully
            assert isinstance(e, (ValueError, TypeError))

    def test_model_error_handling(self, sample_regression_data):
        """Test model error handling in pipeline context."""
        X, y, _ = sample_regression_data

        # Test with invalid model type
        with pytest.raises(ValueError):
            RegressionModels(model_type="invalid_model")

        # Test with mismatched data
        X_bad = X[:-1]  # Remove one sample
        model = RegressionModels(model_type="linear")

        with pytest.raises(ValueError):
            model.train(X_bad, y)  # Should fail due to length mismatch

    def test_file_io_error_propagation(self, tmp_path):
        """Test file I/O error handling in pipeline."""
        # Test with non-existent file
        non_existent_file = tmp_path / "does_not_exist.csv"

        try:
            with pytest.raises((FileNotFoundError, OSError)):
                load_molecular_data(str(non_existent_file))
        except NotImplementedError:
            pytest.skip("File loading not implemented")

        # Test with invalid save path
        try:
            data = pd.DataFrame({"smiles": ["CCO"], "activity": [1]})
            invalid_path = "/invalid/path/that/does/not/exist/file.csv"

            with pytest.raises((OSError, PermissionError, FileNotFoundError)):
                save_molecular_data(data, invalid_path)
        except NotImplementedError:
            pytest.skip("File saving not implemented")


class TestPerformanceIntegration:
    """Test performance of integrated workflows."""

    @pytest.mark.slow
    def test_large_dataset_pipeline_performance(self, performance_timer):
        """Test pipeline performance with larger datasets."""
        # Create larger synthetic dataset
        n_samples = 1000
        n_features = 50

        np.random.seed(42)
        large_data = pd.DataFrame(
            {"feature_" + str(i): np.random.randn(n_samples) for i in range(n_features)}
        )
        large_data["target"] = np.random.randn(n_samples)

        performance_timer.start()

        # Run pipeline steps
        cleaned_data = clean_data(large_data)

        feature_cols = [col for col in cleaned_data.columns if col != "target"]
        X = cleaned_data[feature_cols].values
        y = cleaned_data["target"].values

        # Train model
        model = RegressionModels(model_type="linear")
        mse, r2 = model.train(X, y)

        # Make predictions
        predictions = model.predict(X)

        performance_timer.stop()

        # Should complete in reasonable time
        assert performance_timer.elapsed < 30.0  # 30 seconds max
        assert len(predictions) == len(y)
        assert isinstance(mse, float)
        assert isinstance(r2, float)

    @pytest.mark.slow
    @skip_if_no_rdkit
    def test_molecular_feature_extraction_performance(self, performance_timer):
        """Test performance of molecular feature extraction pipeline."""
        # Create larger set of test molecules
        test_smiles = [
            "CCO",
            "CCC",
            "CCCC",
            "c1ccccc1",
            "CC(=O)O",
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        ] * 100  # 600 molecules total

        performance_timer.start()

        try:
            # Convert to molecules
            from rdkit import Chem

            molecules = []
            for smiles in test_smiles:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    molecules.append(mol)

            # Extract features
            if molecules:
                descriptors = extract_descriptors(molecules)
                fingerprints = generate_fingerprints(molecules)

                # Combine features
                combined_features = np.hstack([descriptors.values, fingerprints])

                performance_timer.stop()

                # Should complete in reasonable time
                assert performance_timer.elapsed < 60.0  # 60 seconds max
                assert combined_features.shape[0] == len(molecules)
                assert combined_features.shape[1] > 0
            else:
                pytest.skip("No valid molecules generated")

        except NotImplementedError as e:
            pytest.skip(f"Feature extraction not implemented: {e}")


class TestNotebookIntegration:
    """Test integration with bootcamp notebooks."""

    def test_bootcamp_day1_pipeline(self, sample_molecular_data):
        """Test pipeline similar to Day 1 bootcamp."""
        # Simulate Day 1: Basic ML and cheminformatics
        try:
            # Data preparation
            cleaned_data = clean_data(sample_molecular_data)

            # Feature selection
            feature_cols = ["molecular_weight", "logp", "tpsa"]
            available_cols = [
                col for col in feature_cols if col in cleaned_data.columns
            ]

            if len(available_cols) >= 2:
                X = cleaned_data[available_cols].values
                y = cleaned_data["activity"].values

                # Model training (Day 1 level)
                model = RegressionModels(model_type="linear")
                mse, r2 = model.train(X, y)

                # Basic evaluation
                predictions = model.predict(X)

                # Simple analysis
                from src.utils.metrics import accuracy, mean_squared_error

                if np.all(np.isin(y, [0, 1])):  # Binary classification
                    acc = accuracy(y, (predictions > 0.5).astype(int))
                    assert 0 <= acc <= 1
                else:  # Regression
                    eval_mse = mean_squared_error(y, predictions)
                    assert eval_mse >= 0

                # Verify Day 1 learning objectives met
                assert isinstance(predictions, np.ndarray)
                assert len(predictions) == len(y)

        except NotImplementedError as e:
            pytest.skip(f"Day 1 pipeline components not implemented: {e}")

    def test_bootcamp_assessment_integration(self, sample_molecular_data):
        """Test integration with bootcamp assessment framework."""
        try:
            # Load assessment framework
            from notebooks.quickstart_bootcamp.utils.assessment_framework import (
                BootcampAssessment,
                create_assessment,
            )

            # Create assessment instance
            assessment = create_assessment("integration_test", day=1)

            # Simulate completing a task
            task_result = {
                "task_type": "model_training",
                "model_type": "linear_regression",
                "performance": {"mse": 0.15, "r2": 0.85},
                "completion_time": 120,  # seconds
            }

            # Record progress
            assessment.record_activity("basic_ml", task_result)

            # Check assessment state
            progress = assessment.get_progress_summary()
            assert isinstance(progress, dict)
            # Just check that we can call the methods without error

        except ImportError:
            pytest.skip("Assessment framework not available")


class TestConfigurationIntegration:
    """Test integration with different configurations."""

    def test_different_model_configurations(self, sample_regression_data):
        """Test pipeline with different model configurations."""
        X, y, _ = sample_regression_data

        model_configs = [
            {"model_type": "linear"},
            {"model_type": "ridge"},
            {"model_type": "lasso"},
        ]

        results = {}

        for config in model_configs:
            try:
                model = RegressionModels(**config)
                mse, r2 = model.train(X, y)
                predictions = model.predict(X)

                results[config["model_type"]] = {
                    "mse": mse,
                    "r2": r2,
                    "predictions": predictions,
                }

            except ValueError:
                # Some configurations might not be valid
                continue

        # Should have results for at least one configuration
        assert len(results) > 0

        # All results should be valid
        for model_type, result in results.items():
            assert isinstance(result["mse"], float)
            assert isinstance(result["r2"], float)
            assert isinstance(result["predictions"], np.ndarray)

    def test_feature_extraction_configurations(self, sample_smiles):
        """Test different feature extraction configurations."""
        if not sample_smiles:
            pytest.skip("No sample SMILES available")

        try:
            from rdkit import Chem

            molecules = [Chem.MolFromSmiles(smiles) for smiles in sample_smiles]
            molecules = [mol for mol in molecules if mol is not None]

            if not molecules:
                pytest.skip("No valid molecules")

            # Test different fingerprint configurations
            fp_configs = [
                {"fp_type": "morgan", "n_bits": 1024},
                {"fp_type": "morgan", "n_bits": 2048},
                {"fp_type": "topological", "n_bits": 1024},
            ]

            for config in fp_configs:
                try:
                    fingerprints = generate_fingerprints(molecules, **config)
                    assert fingerprints.shape[0] == len(molecules)
                    assert fingerprints.shape[1] == config.get("n_bits", 1024)
                except (ValueError, NotImplementedError):
                    # Some configurations might not be supported
                    continue

        except NotImplementedError:
            pytest.skip("Feature extraction not implemented")
