import os
import sys
from pathlib import Path

# Add src directory to Python path
src_dir = Path(__file__).parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
"""
Pytest configuration and shared fixtures for QeMLflow tests.

This module provides common fixtures and configuration for the entire test suite,
including sample molecular data, mock models, and test utilities.
"""

import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

try:
    from rdkit import Chem
except ImportError:
    pass

# Try to import molecular libraries with graceful fallback
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolDescriptors

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

try:
    import deepchem as dc

    DEEPCHEM_AVAILABLE = True
except ImportError:
    DEEPCHEM_AVAILABLE = False


@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp(prefix="qemlflow_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_smiles() -> List[str]:
    """Provide sample SMILES strings for testing."""
    return [
        "CCO",  # Ethanol
        "CC(=O)O",  # Acetic acid
        "c1ccccc1",  # Benzene
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
        "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F",  # Celecoxib
    ]


@pytest.fixture
def sample_molecular_data(sample_smiles) -> pd.DataFrame:
    """Create sample molecular dataset for testing."""
    data = {
        "smiles": sample_smiles,
        "molecular_weight": [46.07, 60.05, 78.11, 194.19, 206.28, 381.37],
        "logp": [-0.31, -0.17, 2.13, -0.07, 3.97, 3.47],
        "tpsa": [20.23, 37.30, 0.00, 61.83, 37.30, 92.35],
        "activity": [0, 0, 1, 1, 1, 1],  # Binary classification target
        "solubility": [0.5, 1.2, -2.1, -0.8, -3.2, -4.1],  # Regression target
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_molecules(sample_smiles):
    """Create RDKit molecule objects if available."""
    if not RDKIT_AVAILABLE:
        pytest.skip("RDKit not available")

    molecules = []
    for smiles in sample_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            molecules.append(mol)
    return molecules


@pytest.fixture
def sample_molecular_descriptors(sample_molecules):
    """Calculate molecular descriptors for sample molecules."""
    if not RDKIT_AVAILABLE:
        pytest.skip("RDKit not available")

    descriptors = []
    for mol in sample_molecules:
        desc = {
            "mw": rdMolDescriptors.CalcExactMolWt(mol),
            "logp": rdMolDescriptors.CalcCrippenDescriptors(mol)[0],
            "tpsa": rdMolDescriptors.CalcTPSA(mol),
            "hbd": rdMolDescriptors.CalcNumHBD(mol),
            "hba": rdMolDescriptors.CalcNumHBA(mol),
            "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        }
        descriptors.append(desc)

    return pd.DataFrame(descriptors)


@pytest.fixture
def sample_fingerprints(sample_molecules):
    """Generate molecular fingerprints for sample molecules."""
    if not RDKIT_AVAILABLE:
        pytest.skip("RDKit not available")

    fingerprints = []
    for mol in sample_molecules:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        fingerprints.append(np.array(fp))

    return np.array(fingerprints)


@pytest.fixture
def sample_regression_data():
    """Create sample regression dataset."""
    np.random.seed(42)
    n_samples = 100
    n_features = 10

    X = np.random.randn(n_samples, n_features)
    # Create target with some linear relationship + noise
    true_coef = np.random.randn(n_features)
    y = X @ true_coef + 0.1 * np.random.randn(n_samples)

    return X, y, true_coef


@pytest.fixture
def sample_classification_data():
    """Create sample classification dataset."""
    np.random.seed(42)
    n_samples = 100
    n_features = 10

    X = np.random.randn(n_samples, n_features)
    # Create binary target
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    return X, y


@pytest.fixture
def mock_deepchem_dataset():
    """Create mock DeepChem dataset if available."""
    if not DEEPCHEM_AVAILABLE:
        pytest.skip("DeepChem not available")

    # Create simple mock dataset
    n_samples = 50
    n_features = 10

    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples, 1)

    dataset = dc.data.NumpyDataset(X, y)
    return dataset


@pytest.fixture
def quantum_simulator_mock():
    """Mock quantum simulator for testing without actual quantum hardware."""

    class MockQuantumSimulator:
        def __init__(self):
            self.circuits = []
            self.results = []

        def run_circuit(self, circuit):
            """Mock circuit execution."""
            # Return random results for testing
            return {
                "counts": {"00": 512, "11": 512},
                "statevector": [0.707, 0, 0, 0.707],
                "expectation_value": np.random.random(),
            }

        def get_backend_info(self):
            """Mock backend information."""
            return {
                "name": "mock_simulator",
                "qubits": 32,
                "gates": ["cx", "u3", "measure"],
            }

    return MockQuantumSimulator()


@pytest.fixture
def temp_molecular_file(test_data_dir, sample_smiles):
    """Create temporary molecular file for I/O testing."""
    file_path = test_data_dir / "test_molecules.csv"

    df = pd.DataFrame({"smiles": sample_smiles, "activity": [0, 0, 1, 1, 1, 1]})
    df.to_csv(file_path, index=False)

    return file_path


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility."""
    np.random.seed(42)


class MockModel:
    """Mock model class for testing model interfaces."""

    def __init__(self, model_type="regression"):
        self.model_type = model_type
        self.is_fitted = False
        self.feature_names = None

    def fit(self, X, y):
        """Mock fit method."""
        self.is_fitted = True
        self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        return self

    def predict(self, X):
        """Mock predict method."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        if self.model_type == "regression":
            return np.random.randn(X.shape[0])
        else:  # classification
            return np.random.randint(0, 2, X.shape[0])

    def score(self, X, y):
        """Mock score method."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring")
        return np.random.random()


@pytest.fixture
def mock_regression_model():
    """Provide mock regression model."""
    return MockModel("regression")


@pytest.fixture
def mock_classification_model():
    """Provide mock classification model."""
    return MockModel("classification")


# Performance testing utilities
@pytest.fixture
def performance_timer():
    """Timer fixture for performance testing."""
    import time

    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None

        def start(self):
            self.start_time = time.time()

        def stop(self):
            self.end_time = time.time()

        @property
        def elapsed(self):
            if self.start_time is None or self.end_time is None:
                return None
            return self.end_time - self.start_time

    return Timer()


# Skip conditions for optional dependencies
skip_if_no_rdkit = pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")

skip_if_no_deepchem = pytest.mark.skipif(
    not DEEPCHEM_AVAILABLE, reason="DeepChem not available"
)

# Quantum computing availability check
try:
    import qiskit

    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

skip_if_no_qiskit = pytest.mark.skipif(
    not QISKIT_AVAILABLE, reason="Qiskit not available"
)
