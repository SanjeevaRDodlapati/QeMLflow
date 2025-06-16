"""
Simple integration test to validate test infrastructure.
"""

import numpy as np
import pytest

from tests.conftest import MockModel

try:
    from rdkit import Chem
except ImportError:
    pass


def test_mock_model_functionality():
    """Test that mock model works as expected."""
    # Create regression model
    model = MockModel("regression")

    # Generate test data
    X = np.random.randn(10, 5)
    y = np.random.randn(10)

    # Test training
    model.fit(X, y)
    assert model.is_fitted

    # Test prediction
    predictions = model.predict(X)
    assert len(predictions) == len(y)
    assert isinstance(predictions, np.ndarray)

    # Test scoring
    score = model.score(X, y)
    assert isinstance(score, float)


def test_sample_data_generation():
    """Test that sample data generators work."""
    from tests.fixtures.sample_data import (
        TestDataGenerator,
        create_standard_molecular_dataset,
    )

    # Test molecular dataset generation
    dataset = TestDataGenerator.generate_molecular_dataset(n_samples=10)
    assert len(dataset) == 10
    assert "smiles" in dataset.columns
    assert "activity" in dataset.columns

    # Test standard dataset
    std_dataset = create_standard_molecular_dataset()
    assert len(std_dataset) > 0
    assert "smiles" in std_dataset.columns


def test_test_infrastructure_imports():
    """Test that test infrastructure imports work correctly."""
    # Test conftest imports
    from tests.conftest import skip_if_no_deepchem, skip_if_no_qiskit, skip_if_no_rdkit

    # These should be pytest markers
    assert hasattr(skip_if_no_rdkit, "mark")

    # Test fixtures import
    from tests.fixtures.sample_data import STANDARD_TEST_SMILES

    assert isinstance(STANDARD_TEST_SMILES, list)
    assert len(STANDARD_TEST_SMILES) > 0


if __name__ == "__main__":
    # Allow running this test directly
    test_mock_model_functionality()
    test_sample_data_generation()
    test_test_infrastructure_imports()
    print("✅ All infrastructure tests passed!")
