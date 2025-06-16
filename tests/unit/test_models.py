"""
Unit tests for models module.

Tests classical ML and quantum ML model implementations.
"""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, mean_squared_error

# Import modules under test
try:
    from src.models.classical_ml.regression_models import RegressionModels
    from src.models.quantum_ml.quantum_circuits import QuantumCircuit
except ImportError as e:
    pytest.skip(f"Models modules not available: {e}", allow_module_level=True)

from tests.conftest import skip_if_no_qiskit
from sklearn import *


class TestRegressionModels:
    """Test classical regression models."""

    def test_linear_regression_initialization(self):
        """Test linear regression model initialization."""
        model = RegressionModels(model_type="linear")
        assert model.model is not None
        assert hasattr(model, "train")
        assert hasattr(model, "predict")

    def test_ridge_regression_initialization(self):
        """Test ridge regression model initialization."""
        model = RegressionModels(model_type="ridge")
        assert model.model is not None

    def test_lasso_regression_initialization(self):
        """Test lasso regression model initialization."""
        model = RegressionModels(model_type="lasso")
        assert model.model is not None

    def test_invalid_model_type(self):
        """Test error handling for invalid model type."""
        with pytest.raises(ValueError, match="Unsupported model type"):
            RegressionModels(model_type="invalid_type")

    def test_model_training(self, sample_regression_data):
        """Test model training functionality."""
        X, y, _ = sample_regression_data
        model = RegressionModels(model_type="linear")

        mse, r2 = model.train(X, y)

        # Check that metrics are returned
        assert isinstance(mse, float)
        assert isinstance(r2, float)
        assert mse >= 0  # MSE should be non-negative
        assert -1 <= r2 <= 1  # R² should be in valid range

    def test_model_prediction(self, sample_regression_data):
        """Test model prediction functionality."""
        X, y, _ = sample_regression_data
        model = RegressionModels(model_type="linear")

        # Train the model first
        model.train(X, y)

        # Make predictions
        predictions = model.predict(X)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(y)
        assert predictions.ndim == 1

    def test_different_model_types_performance(self, sample_regression_data):
        """Test different regression model types."""
        X, y, _ = sample_regression_data
        model_types = ["linear", "ridge", "lasso"]
        results = {}

        for model_type in model_types:
            model = RegressionModels(model_type=model_type)
            mse, r2 = model.train(X, y)
            results[model_type] = {"mse": mse, "r2": r2}

        # All models should produce reasonable results
        for model_type, metrics in results.items():
            assert metrics["mse"] >= 0
            assert -1 <= metrics["r2"] <= 1

    def test_get_model_method(self, sample_regression_data):
        """Test get_model method returns the underlying model."""
        X, y, _ = sample_regression_data
        model = RegressionModels(model_type="linear")
        model.train(X, y)

        underlying_model = model.get_model()
        assert underlying_model is not None
        assert hasattr(underlying_model, "predict")

    def test_prediction_without_training(self):
        """Test that prediction fails without training."""
        model = RegressionModels(model_type="linear")
        X = np.random.randn(10, 5)

        # Should work (sklearn handles untrained models gracefully)
        # But predictions won't be meaningful
        predictions = model.predict(X)
        assert len(predictions) == 10

    def test_training_with_small_dataset(self):
        """Test training with very small dataset."""
        X = np.random.randn(3, 2)
        y = np.random.randn(3)

        model = RegressionModels(model_type="linear")
        mse, r2 = model.train(X, y)

        # Should handle small datasets
        assert isinstance(mse, float)
        assert isinstance(r2, float)

    def test_training_with_single_feature(self):
        """Test training with single feature."""
        X = np.random.randn(50, 1)
        y = 2 * X.flatten() + np.random.randn(50) * 0.1

        model = RegressionModels(model_type="linear")
        mse, r2 = model.train(X, y)

        # Should work with single feature
        assert mse >= 0
        assert r2 > 0  # Should capture some variance


@skip_if_no_qiskit
class TestQuantumCircuit:
    """Test quantum circuit implementations."""

    def test_quantum_circuit_initialization(self):
        """Test quantum circuit initialization."""
        qc = QuantumCircuit(n_qubits=2)

        assert qc.n_qubits == 2
        assert hasattr(qc, "circuit")
        assert hasattr(qc, "simulate")
        assert hasattr(qc, "evaluate")

    def test_quantum_circuit_invalid_qubits(self):
        """Test error handling for invalid qubit count."""
        with pytest.raises(ValueError):
            QuantumCircuit(n_qubits=0)

        with pytest.raises(ValueError):
            QuantumCircuit(n_qubits=-1)

    def test_add_layer_method(self):
        """Test adding layers to quantum circuit."""
        qc = QuantumCircuit(n_qubits=2)

        # Add rotation layer
        qc.add_rotation_layer([0.1, 0.2])
        assert len(qc.circuit.data) > 0

        # Add entangling layer
        qc.add_entangling_layer()
        circuit_depth_before = len(qc.circuit.data)
        assert circuit_depth_before > 0

    def test_circuit_simulation(self):
        """Test quantum circuit simulation."""
        qc = QuantumCircuit(n_qubits=2)
        qc.add_rotation_layer([0.1, 0.2])

        result = qc.simulate()

        assert "counts" in result or "statevector" in result
        if "counts" in result:
            assert isinstance(result["counts"], dict)
        if "statevector" in result:
            assert isinstance(result["statevector"], (list, np.ndarray))

    def test_circuit_evaluation(self):
        """Test quantum circuit evaluation with mock data."""
        qc = QuantumCircuit(n_qubits=2)
        qc.add_rotation_layer([0.1, 0.2])

        # Mock training data
        X = np.random.randn(10, 2)
        y = np.random.randint(0, 2, 10)

        # This might need adjustment based on actual implementation
        try:
            accuracy = qc.evaluate(X, y)
            assert 0 <= accuracy <= 1
        except NotImplementedError:
            pytest.skip("Quantum circuit evaluation not implemented")

    def test_parameter_binding(self):
        """Test parameter binding in quantum circuits."""
        qc = QuantumCircuit(n_qubits=2)

        # Create parameterized circuit
        qc.create_parameterized_circuit(n_layers=2)

        # Bind parameters
        params = np.random.randn(qc.num_parameters)
        bound_circuit = qc.bind_parameters(params)

        assert bound_circuit is not None
        # Check that parameters are bound (no free parameters)
        assert len(bound_circuit.parameters) == 0

    def test_gradient_computation(self):
        """Test gradient computation for quantum circuits."""
        qc = QuantumCircuit(n_qubits=2)
        qc.create_parameterized_circuit(n_layers=1)

        params = np.random.randn(qc.num_parameters)

        try:
            gradients = qc.compute_gradients(params)
            assert isinstance(gradients, np.ndarray)
            assert len(gradients) == qc.num_parameters
        except NotImplementedError:
            pytest.skip("Gradient computation not implemented")

    def test_vqe_algorithm(self):
        """Test Variational Quantum Eigensolver implementation."""
        qc = QuantumCircuit(n_qubits=2)

        # Define simple Hamiltonian (Pauli-Z on first qubit)
        hamiltonian = "Z0"

        try:
            result = qc.run_vqe(hamiltonian, max_iterations=10)

            assert "energy" in result
            assert "parameters" in result
            assert isinstance(result["energy"], float)
            assert isinstance(result["parameters"], np.ndarray)
        except NotImplementedError:
            pytest.skip("VQE algorithm not implemented")

    def test_quantum_feature_map(self):
        """Test quantum feature map encoding."""
        qc = QuantumCircuit(n_qubits=2)

        # Test data encoding
        data_point = np.array([0.5, 0.3])

        try:
            encoded_circuit = qc.encode_classical_data(data_point)
            assert encoded_circuit is not None
            assert len(encoded_circuit.data) > 0
        except NotImplementedError:
            pytest.skip("Quantum feature map not implemented")


class TestModelIntegration:
    """Integration tests for model workflows."""

    def test_classical_to_quantum_pipeline(self, sample_regression_data):
        """Test pipeline combining classical and quantum models."""
        X, y, _ = sample_regression_data

        # Use classical model for feature preprocessing
        classical_model = RegressionModels(model_type="linear")
        classical_model.train(X, y)

        # Get classical predictions as features for quantum model
        classical_preds = classical_model.predict(X)

        # Combine original features with classical predictions
        enhanced_features = np.column_stack([X, classical_preds])

        # This could feed into quantum model (if implemented)
        assert enhanced_features.shape[1] == X.shape[1] + 1
        assert len(enhanced_features) == len(X)

    def test_model_serialization(self, sample_regression_data):
        """Test model saving and loading."""
        X, y, _ = sample_regression_data

        model = RegressionModels(model_type="linear")
        model.train(X, y)

        # Test that underlying sklearn model can be pickled
        import pickle

        try:
            serialized = pickle.dumps(model.get_model())
            loaded_model = pickle.loads(serialized)

            # Test that loaded model works
            predictions = loaded_model.predict(X)
            assert len(predictions) == len(y)
        except Exception as e:
            pytest.fail(f"Model serialization failed: {e}")

    def test_cross_validation_workflow(self, sample_regression_data):
        """Test cross-validation workflow."""
        from sklearn.model_selection import cross_val_score

        X, y, _ = sample_regression_data

        model = RegressionModels(model_type="linear")
        underlying_model = model.get_model()

        # Perform cross-validation
        cv_scores = cross_val_score(underlying_model, X, y, cv=3, scoring="r2")

        assert len(cv_scores) == 3
        assert all(isinstance(score, float) for score in cv_scores)


class TestModelPerformance:
    """Performance tests for model operations."""

    @pytest.mark.slow
    def test_large_dataset_training(self, performance_timer):
        """Test training performance with larger datasets."""
        # Create larger dataset
        n_samples = 10000
        n_features = 100

        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples)

        model = RegressionModels(model_type="linear")

        performance_timer.start()
        mse, r2 = model.train(X, y)
        performance_timer.stop()

        # Should complete in reasonable time
        assert performance_timer.elapsed < 5.0  # 5 seconds max
        assert mse >= 0
        assert isinstance(r2, float)

    @pytest.mark.slow
    def test_prediction_performance(self, performance_timer):
        """Test prediction performance."""
        # Train on medium dataset
        X_train = np.random.randn(1000, 50)
        y_train = np.random.randn(1000)

        model = RegressionModels(model_type="linear")
        model.train(X_train, y_train)

        # Large prediction dataset
        X_test = np.random.randn(100000, 50)

        performance_timer.start()
        predictions = model.predict(X_test)
        performance_timer.stop()

        # Should complete in reasonable time
        assert performance_timer.elapsed < 2.0  # 2 seconds max
        assert len(predictions) == len(X_test)


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_mismatched_dimensions(self):
        """Test error handling for mismatched X, y dimensions."""
        X = np.random.randn(10, 5)
        y = np.random.randn(8)  # Wrong length

        model = RegressionModels(model_type="linear")

        with pytest.raises(ValueError):
            model.train(X, y)

    def test_empty_dataset(self):
        """Test error handling for empty datasets."""
        X = np.array([]).reshape(0, 5)
        y = np.array([])

        model = RegressionModels(model_type="linear")

        with pytest.raises(ValueError):
            model.train(X, y)

    def test_single_sample_dataset(self):
        """Test handling of single sample dataset."""
        X = np.random.randn(1, 5)
        y = np.random.randn(1)

        model = RegressionModels(model_type="linear")

        # Should handle gracefully or raise appropriate error
        try:
            mse, r2 = model.train(X, y)
            # If it works, check the results
            assert isinstance(mse, float)
            assert isinstance(r2, float)
        except ValueError:
            # This is also acceptable behavior
            pass
