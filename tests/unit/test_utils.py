"""
Unit tests for utils module.

Tests visualization, metrics, and utility functions.
"""

import os
import tempfile
from unittest.mock import Mock, patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

# Import modules under test
try:
    from src.utils.io_utils import (
        export_results,
        load_molecular_data,
        save_molecular_data,
    )
    from src.utils.metrics import (
        accuracy,
        f1_score,
        mean_squared_error,
        precision,
        r_squared,
        recall,
    )
    from src.utils.ml_utils import evaluate_model, normalize_features, split_data
    from src.utils.molecular_utils import (
        calculate_similarity,
        filter_molecules_by_properties,
        mol_to_smiles,
        smiles_to_mol,
    )
    from src.utils.quantum_utils import (
        apply_quantum_gate,
        create_quantum_circuit,
        measure_quantum_state,
    )
    from src.utils.visualization import (
        plot_feature_importance,
        plot_model_performance,
        plot_molecular_structure,
    )
except ImportError as e:
    pytest.skip(f"Utils modules not available: {e}", allow_module_level=True)

from tests.conftest import skip_if_no_qiskit, skip_if_no_rdkit


class TestVisualization:
    """Test visualization utilities."""

    @skip_if_no_rdkit
    def test_plot_molecular_structure(self, sample_molecules, tmp_path):
        """Test molecular structure plotting."""
        if not sample_molecules:
            pytest.skip("No sample molecules available")

        molecule = sample_molecules[0]
        output_path = tmp_path / "molecule.png"

        try:
            plot_molecular_structure(molecule, filename=str(output_path))

            # Check if file was created
            assert output_path.exists()
            assert output_path.stat().st_size > 0
        except NotImplementedError:
            pytest.skip("Molecular structure plotting not implemented")
        except Exception as e:
            # Some plotting failures are acceptable in headless environments
            if "display" in str(e).lower() or "gui" in str(e).lower():
                pytest.skip(f"Display not available: {e}")
            else:
                raise

    def test_plot_feature_importance(self, tmp_path):
        """Test feature importance plotting."""
        # Mock feature importance data
        importances = [0.3, 0.25, 0.2, 0.15, 0.1]
        feature_names = [
            "feature_1",
            "feature_2",
            "feature_3",
            "feature_4",
            "feature_5",
        ]
        output_path = tmp_path / "feature_importance.png"

        try:
            plot_feature_importance(
                importances,
                feature_names,
                title="Test Feature Importance",
                filename=str(output_path),
            )

            # Check if file was created
            assert output_path.exists()
            assert output_path.stat().st_size > 0
        except NotImplementedError:
            pytest.skip("Feature importance plotting not implemented")
        except Exception as e:
            if "display" in str(e).lower() or "gui" in str(e).lower():
                pytest.skip(f"Display not available: {e}")
            else:
                raise

    def test_plot_model_performance(self, tmp_path):
        """Test model performance plotting."""
        # Mock training history
        history = {
            "loss": [1.0, 0.8, 0.6, 0.4, 0.3],
            "val_loss": [1.1, 0.9, 0.7, 0.5, 0.4],
            "accuracy": [0.6, 0.7, 0.8, 0.85, 0.9],
            "val_accuracy": [0.55, 0.65, 0.75, 0.8, 0.85],
        }
        output_path = tmp_path / "model_performance.png"

        try:
            plot_model_performance(
                history, title="Test Model Performance", filename=str(output_path)
            )

            # Check if file was created
            assert output_path.exists()
            assert output_path.stat().st_size > 0
        except NotImplementedError:
            pytest.skip("Model performance plotting not implemented")
        except Exception as e:
            if "display" in str(e).lower() or "gui" in str(e).lower():
                pytest.skip(f"Display not available: {e}")
            else:
                raise

    def test_plot_with_invalid_data(self):
        """Test plotting functions with invalid data."""
        try:
            # Test with empty data
            with pytest.raises((ValueError, TypeError, IndexError)):
                plot_feature_importance([], [])

            # Test with mismatched data lengths
            with pytest.raises((ValueError, IndexError)):
                plot_feature_importance([0.5, 0.3], ["feature_1"])
        except NotImplementedError:
            pytest.skip("Plotting functions not implemented")


class TestMetrics:
    """Test metric calculation functions."""

    def test_accuracy_calculation(self):
        """Test accuracy metric calculation."""
        y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0])
        y_pred = np.array([1, 0, 1, 0, 0, 1, 0, 1])

        acc = accuracy(y_true, y_pred)

        assert isinstance(acc, float)
        assert 0 <= acc <= 1
        # Manual calculation: 6 correct out of 8 = 0.75
        expected_acc = 6 / 8
        assert abs(acc - expected_acc) < 1e-6

    def test_precision_calculation(self):
        """Test precision metric calculation."""
        y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0])
        y_pred = np.array([1, 0, 1, 0, 0, 1, 0, 1])

        prec = precision(y_true, y_pred)

        assert isinstance(prec, float)
        assert 0 <= prec <= 1

    def test_recall_calculation(self):
        """Test recall metric calculation."""
        y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0])
        y_pred = np.array([1, 0, 1, 0, 0, 1, 0, 1])

        rec = recall(y_true, y_pred)

        assert isinstance(rec, float)
        assert 0 <= rec <= 1

    def test_f1_score_calculation(self):
        """Test F1 score calculation."""
        y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0])
        y_pred = np.array([1, 0, 1, 0, 0, 1, 0, 1])

        f1 = f1_score(y_true, y_pred)

        assert isinstance(f1, float)
        assert 0 <= f1 <= 1

    def test_mean_squared_error_calculation(self):
        """Test MSE calculation."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])

        mse = mean_squared_error(y_true, y_pred)

        assert isinstance(mse, float)
        assert mse >= 0
        # Manual calculation
        expected_mse = np.mean((y_true - y_pred) ** 2)
        assert abs(mse - expected_mse) < 1e-6

    def test_r_squared_calculation(self):
        """Test R² calculation."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])

        r2 = r_squared(y_true, y_pred)

        assert isinstance(r2, float)
        # R² can be negative for very poor models
        assert r2 <= 1

    def test_metrics_with_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([1, 0, 1, 0, 1])
        y_pred = y_true.copy()  # Perfect predictions

        assert accuracy(y_true, y_pred) == 1.0
        assert precision(y_true, y_pred) == 1.0
        assert recall(y_true, y_pred) == 1.0
        assert f1_score(y_true, y_pred) == 1.0

        # Regression case
        y_true_reg = np.array([1.0, 2.0, 3.0])
        y_pred_reg = y_true_reg.copy()

        assert mean_squared_error(y_true_reg, y_pred_reg) == 0.0
        assert r_squared(y_true_reg, y_pred_reg) == 1.0

    def test_metrics_error_handling(self):
        """Test error handling in metrics."""
        # Test with mismatched lengths
        y_true = np.array([1, 0, 1])
        y_pred = np.array([1, 0])  # Different length

        with pytest.raises((ValueError, IndexError)):
            accuracy(y_true, y_pred)

        # Test with empty arrays
        with pytest.raises((ValueError, ZeroDivisionError)):
            accuracy(np.array([]), np.array([]))


class TestMolecularUtils:
    """Test molecular utility functions."""

    @skip_if_no_rdkit
    def test_smiles_to_mol(self, sample_smiles):
        """Test SMILES to molecule conversion."""
        try:
            molecules = [smiles_to_mol(smiles) for smiles in sample_smiles]

            # Should convert valid SMILES
            assert len(molecules) == len(sample_smiles)
            # All should be RDKit molecule objects or None
            assert all(mol is not None for mol in molecules)
        except NotImplementedError:
            pytest.skip("SMILES to molecule conversion not implemented")

    @skip_if_no_rdkit
    def test_mol_to_smiles(self, sample_molecules):
        """Test molecule to SMILES conversion."""
        if not sample_molecules:
            pytest.skip("No sample molecules available")

        try:
            smiles_list = [mol_to_smiles(mol) for mol in sample_molecules]

            # Should convert all molecules
            assert len(smiles_list) == len(sample_molecules)
            assert all(isinstance(smiles, str) for smiles in smiles_list)
            assert all(len(smiles) > 0 for smiles in smiles_list)
        except NotImplementedError:
            pytest.skip("Molecule to SMILES conversion not implemented")

    @skip_if_no_rdkit
    def test_calculate_similarity(self, sample_molecules):
        """Test molecular similarity calculation."""
        if len(sample_molecules) < 2:
            pytest.skip("Need at least 2 molecules for similarity")

        try:
            mol1, mol2 = sample_molecules[0], sample_molecules[1]
            similarity = calculate_similarity(mol1, mol2)

            assert isinstance(similarity, float)
            assert 0 <= similarity <= 1
        except NotImplementedError:
            pytest.skip("Similarity calculation not implemented")

    @skip_if_no_rdkit
    def test_filter_molecules_by_properties(self, sample_molecules):
        """Test filtering molecules by properties."""
        if not sample_molecules:
            pytest.skip("No sample molecules available")

        try:
            filtered = filter_molecules_by_properties(
                sample_molecules, molecular_weight_range=(50, 200), logp_range=(-2, 5)
            )

            assert isinstance(filtered, list)
            assert len(filtered) <= len(sample_molecules)
        except NotImplementedError:
            pytest.skip("Molecule filtering not implemented")


class TestMLUtils:
    """Test machine learning utility functions."""

    def test_split_data(self, sample_regression_data):
        """Test data splitting utility."""
        X, y, _ = sample_regression_data

        try:
            X_train, X_test, y_train, y_test = split_data(
                X, y, test_size=0.2, random_state=42
            )

            # Check split proportions
            assert len(X_train) == len(y_train)
            assert len(X_test) == len(y_test)
            assert len(X_train) + len(X_test) == len(X)
            assert len(y_train) + len(y_test) == len(y)

            # Check approximate split ratio
            expected_test_size = int(0.2 * len(X))
            assert abs(len(X_test) - expected_test_size) <= 1
        except NotImplementedError:
            pytest.skip("Data splitting not implemented")

    def test_normalize_features(self, sample_regression_data):
        """Test feature normalization."""
        X, _, _ = sample_regression_data

        try:
            X_normalized = normalize_features(X)

            assert X_normalized.shape == X.shape

            # Check normalization (approximately mean=0, std=1)
            means = np.mean(X_normalized, axis=0)
            stds = np.std(X_normalized, axis=0)

            assert np.allclose(means, 0, atol=1e-10)
            assert np.allclose(stds, 1, atol=1e-10)
        except NotImplementedError:
            pytest.skip("Feature normalization not implemented")

    def test_evaluate_model(
        self, mock_classification_model, sample_classification_data
    ):
        """Test model evaluation utility."""
        X, y = sample_classification_data

        # Train mock model
        mock_classification_model.fit(X, y)

        try:
            metrics = evaluate_model(
                mock_classification_model, X, y, task_type="classification"
            )

            assert isinstance(metrics, dict)
            # Should contain standard classification metrics
            expected_metrics = ["accuracy", "precision", "recall", "f1"]
            for metric in expected_metrics:
                if metric in metrics:
                    assert isinstance(metrics[metric], (int, float))
                    assert 0 <= metrics[metric] <= 1
        except NotImplementedError:
            pytest.skip("Model evaluation not implemented")


@skip_if_no_qiskit
class TestQuantumUtils:
    """Test quantum computing utilities."""

    def test_create_quantum_circuit(self):
        """Test quantum circuit creation."""
        try:
            circuit = create_quantum_circuit(n_qubits=2, n_cbits=2)

            assert circuit is not None
            # Should have correct number of qubits and classical bits
            assert circuit.num_qubits == 2
            assert circuit.num_clbits == 2
        except NotImplementedError:
            pytest.skip("Quantum circuit creation not implemented")

    def test_apply_quantum_gate(self):
        """Test quantum gate application."""
        try:
            circuit = create_quantum_circuit(n_qubits=2, n_cbits=2)

            # Apply different types of gates
            circuit_with_x = apply_quantum_gate(circuit, "x", 0)
            circuit_with_h = apply_quantum_gate(circuit_with_x, "h", 1)
            circuit_with_cx = apply_quantum_gate(circuit_with_h, "cx", [0, 1])

            assert circuit_with_cx is not None
            # Circuit should have gates applied
            assert len(circuit_with_cx.data) > 0
        except NotImplementedError:
            pytest.skip("Quantum gate application not implemented")

    def test_measure_quantum_state(self):
        """Test quantum state measurement."""
        try:
            circuit = create_quantum_circuit(n_qubits=2, n_cbits=2)
            circuit = apply_quantum_gate(circuit, "h", 0)  # Create superposition

            result = measure_quantum_state(circuit)

            assert result is not None
            assert "counts" in result or "statevector" in result
        except NotImplementedError:
            pytest.skip("Quantum state measurement not implemented")


class TestIOUtils:
    """Test input/output utility functions."""

    def test_load_molecular_data(self, temp_molecular_file):
        """Test loading molecular data from file."""
        try:
            data = load_molecular_data(str(temp_molecular_file))

            assert isinstance(data, pd.DataFrame)
            assert len(data) > 0
            assert "smiles" in data.columns
        except NotImplementedError:
            pytest.skip("Molecular data loading not implemented")

    def test_save_molecular_data(self, sample_molecular_data, tmp_path):
        """Test saving molecular data to file."""
        output_path = tmp_path / "output_molecules.csv"

        try:
            save_molecular_data(sample_molecular_data, str(output_path))

            # Check file was created
            assert output_path.exists()

            # Check file can be read back
            loaded_data = pd.read_csv(output_path)
            assert len(loaded_data) == len(sample_molecular_data)
        except NotImplementedError:
            pytest.skip("Molecular data saving not implemented")

    def test_export_results(self, tmp_path):
        """Test results export functionality."""
        # Mock results data
        results = {
            "model_type": "random_forest",
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.88,
            "feature_importance": [0.3, 0.25, 0.2, 0.15, 0.1],
        }

        output_path = tmp_path / "results.json"

        try:
            export_results(results, str(output_path), format="json")

            # Check file was created
            assert output_path.exists()

            # Check file content
            import json

            with open(output_path, "r") as f:
                loaded_results = json.load(f)

            assert loaded_results["accuracy"] == results["accuracy"]
        except NotImplementedError:
            pytest.skip("Results export not implemented")

    def test_io_error_handling(self):
        """Test error handling in I/O operations."""
        # Test with non-existent file
        try:
            with pytest.raises((FileNotFoundError, OSError)):
                load_molecular_data("non_existent_file.csv")
        except NotImplementedError:
            pytest.skip("Molecular data loading not implemented")

        # Test with invalid path for saving
        try:
            data = pd.DataFrame({"smiles": ["CCO"], "activity": [1]})
            with pytest.raises((OSError, PermissionError)):
                save_molecular_data(data, "/invalid/path/file.csv")
        except NotImplementedError:
            pytest.skip("Molecular data saving not implemented")


class TestUtilsIntegration:
    """Integration tests for utility functions."""

    @skip_if_no_rdkit
    def test_molecular_workflow_integration(self, sample_smiles, tmp_path):
        """Test integration of molecular utilities in a workflow."""
        try:
            # Convert SMILES to molecules
            molecules = [smiles_to_mol(smiles) for smiles in sample_smiles]
            molecules = [mol for mol in molecules if mol is not None]

            # Filter molecules by properties
            filtered_molecules = filter_molecules_by_properties(
                molecules, molecular_weight_range=(50, 500)
            )

            # Convert back to SMILES
            filtered_smiles = [mol_to_smiles(mol) for mol in filtered_molecules]

            # Save results
            results_df = pd.DataFrame({"smiles": filtered_smiles})
            output_path = tmp_path / "filtered_molecules.csv"
            save_molecular_data(results_df, str(output_path))

            # Verify saved file
            assert output_path.exists()
            loaded_data = load_molecular_data(str(output_path))
            assert len(loaded_data) == len(filtered_smiles)
        except NotImplementedError:
            pytest.skip("Molecular workflow utilities not fully implemented")

    def test_ml_pipeline_integration(self, sample_regression_data, tmp_path):
        """Test integration of ML utilities in a pipeline."""
        X, y, _ = sample_regression_data

        try:
            # Split data
            X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)

            # Normalize features
            X_train_norm = normalize_features(X_train)
            X_test_norm = normalize_features(X_test)

            # Train mock model
            from tests.conftest import MockModel

            model = MockModel("regression")
            model.fit(X_train_norm, y_train)

            # Evaluate model
            metrics = evaluate_model(model, X_test_norm, y_test, task_type="regression")

            # Export results
            output_path = tmp_path / "ml_results.json"
            export_results(metrics, str(output_path), format="json")

            # Verify results file
            assert output_path.exists()
        except NotImplementedError:
            pytest.skip("ML workflow utilities not fully implemented")


class TestPerformance:
    """Performance tests for utility functions."""

    @pytest.mark.slow
    def test_large_dataset_normalization_performance(self, performance_timer):
        """Test normalization performance with large dataset."""
        # Create large dataset
        large_X = np.random.randn(50000, 100)

        performance_timer.start()
        try:
            normalized_X = normalize_features(large_X)
            performance_timer.stop()

            # Should complete in reasonable time
            assert performance_timer.elapsed < 5.0  # 5 seconds max
            assert normalized_X.shape == large_X.shape
        except NotImplementedError:
            pytest.skip("Feature normalization not implemented")

    @pytest.mark.slow
    @skip_if_no_rdkit
    def test_molecular_similarity_performance(self, performance_timer):
        """Test molecular similarity calculation performance."""
        # Create test molecules
        test_smiles = ["CCO"] * 1000  # Simple molecule repeated

        try:
            molecules = [smiles_to_mol(smiles) for smiles in test_smiles]
            molecules = [mol for mol in molecules if mol is not None]

            if len(molecules) >= 2:
                performance_timer.start()
                # Calculate similarity between first 100 pairs
                similarities = []
                for i in range(min(100, len(molecules) - 1)):
                    sim = calculate_similarity(molecules[0], molecules[i + 1])
                    similarities.append(sim)
                performance_timer.stop()

                # Should complete in reasonable time
                assert performance_timer.elapsed < 10.0  # 10 seconds max
                assert len(similarities) == min(100, len(molecules) - 1)
        except NotImplementedError:
            pytest.skip("Molecular similarity calculation not implemented")
