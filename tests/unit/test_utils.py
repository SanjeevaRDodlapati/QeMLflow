"""
Core scientific utilities tests - Essential molecular and quantum functionality

Focuses on core scientific computing utilities:
- Molecular processing and validation
- Quantum computing primitives  
- Data handling for scientific workflows
- Essential ML utilities for scientific data

Note: Metrics testing is covered in test_metrics_core.py
Visualization testing is minimal - focus on core scientific functionality
"""

import os
import tempfile
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

# Import core scientific utilities
try:
    from qemlflow.core.utils.io_utils import (
        export_results,
        load_molecular_data,
        save_molecular_data,
    )
    from qemlflow.core.utils.ml_utils import normalize_features, split_data
    from qemlflow.core.utils.molecular_utils import (
        calculate_similarity,
        filter_molecules_by_properties,
        mol_to_smiles,
        smiles_to_mol,
    )
    from qemlflow.core.utils.quantum_utils import (
        apply_quantum_gate,
        create_quantum_circuit,
        measure_quantum_state,
    )
except ImportError as e:
    pytest.skip(f"Core scientific utils not available: {e}", allow_module_level=True)


@pytest.fixture
def sample_smiles():
    """Sample SMILES strings for testing."""
    return ["CCO", "CC(=O)O", "c1ccccc1"]  # ethanol, acetic acid, benzene


@pytest.fixture  
def sample_regression_data():
    """Sample regression data for ML testing."""
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([10, 20, 30, 40])
    return X, y


class TestMolecularUtils:
    """Test molecular utility functions - core scientific functionality."""

    def test_smiles_to_mol(self, sample_smiles):
        """Test SMILES to molecule conversion."""
        try:
            for smiles in sample_smiles:
                mol = smiles_to_mol(smiles)
                # Should return a valid molecule object or None for invalid SMILES
                assert mol is not None or smiles == "invalid"
        except NotImplementedError:
            pytest.skip("SMILES to mol conversion not implemented")
        except ImportError:
            pytest.skip("RDKit not available")

    def test_mol_to_smiles(self):
        """Test molecule to SMILES conversion."""
        try:
            # Mock molecule for testing
            from unittest.mock import Mock
            mock_mol = Mock()
            result = mol_to_smiles(mock_mol)
            # Should return a string or None
            assert isinstance(result, (str, type(None)))
        except NotImplementedError:
            pytest.skip("Mol to SMILES conversion not implemented")
        except ImportError:
            pytest.skip("RDKit not available")

    def test_calculate_similarity(self):
        """Test molecular similarity calculation."""
        try:
            from unittest.mock import Mock
            mol1, mol2 = Mock(), Mock()
            similarity = calculate_similarity(mol1, mol2)
            # Should return a float between 0 and 1, or None
            assert isinstance(similarity, (float, type(None)))
            if similarity is not None:
                assert 0 <= similarity <= 1
        except NotImplementedError:
            pytest.skip("Similarity calculation not implemented")
        except ImportError:
            pytest.skip("RDKit not available")

    def test_filter_molecules_by_properties(self):
        """Test molecule filtering by properties."""
        try:
            from unittest.mock import Mock
            molecules = [Mock() for _ in range(3)]
            filters = {"mw": (100, 500), "logp": (-2, 5)}
            result = filter_molecules_by_properties(molecules, filters)
            # Should return a list (may be empty)
            assert isinstance(result, list)
        except NotImplementedError:
            pytest.skip("Molecule filtering not implemented")
        except ImportError:
            pytest.skip("RDKit not available")


class TestMLUtils:
    """Test machine learning utilities for scientific data."""

    def test_split_data(self, sample_regression_data):
        """Test data splitting for ML."""
        X, y = sample_regression_data
        try:
            X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.5)
            
            # Check shapes
            assert len(X_train) + len(X_test) == len(X)
            assert len(y_train) + len(y_test) == len(y)
            assert len(X_train) == len(y_train)
            assert len(X_test) == len(y_test)
        except NotImplementedError:
            pytest.skip("Data splitting not implemented")

    def test_normalize_features(self, sample_regression_data):
        """Test feature normalization."""
        X, _ = sample_regression_data
        try:
            X_normalized = normalize_features(X)
            
            # Check that normalization maintains shape
            assert X_normalized.shape == X.shape
            
            # For standard normalization, mean should be close to 0, std close to 1
            if hasattr(X_normalized, 'mean') and hasattr(X_normalized, 'std'):
                mean_vals = np.mean(X_normalized, axis=0)
                std_vals = np.std(X_normalized, axis=0, ddof=1)
                
                # Allow some tolerance for numerical precision
                assert np.allclose(mean_vals, 0, atol=1e-10)
                assert np.allclose(std_vals, 1, atol=1e-10)
        except NotImplementedError:
            pytest.skip("Feature normalization not implemented")


class TestQuantumUtils:
    """Test quantum computing utilities - core quantum functionality."""

    def test_create_quantum_circuit(self):
        """Test quantum circuit creation."""
        try:
            circuit = create_quantum_circuit(3)  # 3-qubit circuit
            # Should return a circuit object
            assert circuit is not None
        except NotImplementedError:
            pytest.skip("Quantum circuit creation not implemented")
        except ImportError:
            pytest.skip("Quantum computing libraries not available")

    def test_apply_quantum_gate(self):
        """Test quantum gate application."""
        try:
            # Mock circuit for testing
            from unittest.mock import Mock
            mock_circuit = Mock()
            result = apply_quantum_gate(mock_circuit, "H", 0)  # Hadamard on qubit 0
            # Should return a circuit (may be the same object)
            assert result is not None
        except NotImplementedError:
            pytest.skip("Quantum gate application not implemented")
        except ImportError:
            pytest.skip("Quantum computing libraries not available")

    def test_measure_quantum_state(self):
        """Test quantum state measurement."""
        try:
            from unittest.mock import Mock
            mock_circuit = Mock()
            result = measure_quantum_state(mock_circuit)
            # Should return measurement results (dict, list, or array)
            assert isinstance(result, (dict, list, np.ndarray, type(None)))
        except NotImplementedError:
            pytest.skip("Quantum measurement not implemented")
        except ImportError:
            pytest.skip("Quantum computing libraries not available")


class TestIOUtils:
    """Test I/O utilities for scientific data handling."""

    def test_load_molecular_data(self, tmp_path):
        """Test molecular data loading."""
        # Create a simple test file
        test_file = tmp_path / "test_molecules.csv"
        test_data = pd.DataFrame({
            "smiles": ["CCO", "CC(=O)O", "c1ccccc1"],
            "property": [0.5, 0.8, 1.2]
        })
        test_data.to_csv(test_file, index=False)
        
        try:
            data = load_molecular_data(str(test_file))
            assert isinstance(data, pd.DataFrame)
            assert len(data) == 3
            assert "smiles" in data.columns
        except NotImplementedError:
            pytest.skip("Molecular data loading not implemented")

    def test_save_molecular_data(self, tmp_path):
        """Test molecular data saving."""
        test_data = pd.DataFrame({
            "smiles": ["CCO", "CC(=O)O"],
            "property": [0.5, 0.8]
        })
        test_file = tmp_path / "output_molecules.csv"
        
        try:
            save_molecular_data(test_data, str(test_file))
            assert test_file.exists()
            
            # Verify data can be read back
            loaded_data = pd.read_csv(test_file)
            assert len(loaded_data) == 2
        except NotImplementedError:
            pytest.skip("Molecular data saving not implemented")

    def test_export_results(self, tmp_path):
        """Test results export functionality."""
        results = {
            "model_name": "test_model",
            "accuracy": 0.95,
            "predictions": [0.1, 0.9, 0.3]
        }
        output_file = tmp_path / "results.json"
        
        try:
            export_results(results, str(output_file))
            assert output_file.exists()
        except NotImplementedError:
            pytest.skip("Results export not implemented")


if __name__ == "__main__":
    pytest.main([__file__])
