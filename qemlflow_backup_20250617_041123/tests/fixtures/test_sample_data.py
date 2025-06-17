"""
Sample data generators for ChemML tests.

Provides utilities to generate test datasets for molecular machine learning.
"""

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Standard test SMILES for consistent testing
STANDARD_TEST_SMILES = [
    "CCO",  # ethanol
    "CC(=O)O",  # acetic acid
    "C1=CC=CC=C1",  # benzene
    "CCN",  # ethylamine
    "C=C",  # ethene
    "CC",  # ethane
    "C#C",  # acetylene
    "CCC",  # propane
    "CO",  # methanol
    "C1=CC=CC=C1O",  # phenol
    "CN",  # methylamine
    "C1=CC=C(C=C1)O",  # para-cresol
    "C(C(=O)O)N",  # glycine
    "CC(C)O",  # isopropanol
    "C1=CC=C2C=CC=CC2=C1",  # naphthalene
]


class TestDataGenerator:
    """Generator for test molecular datasets."""

    @staticmethod
    def generate_molecular_dataset(
        n_samples: int = 100, seed: Optional[int] = 42
    ) -> pd.DataFrame:
        """Generate a synthetic molecular dataset with SMILES and activities.

        Args:
            n_samples: Number of samples to generate
            seed: Random seed for reproducibility

        Returns:
            DataFrame with 'smiles' and 'activity' columns
        """
        if seed is not None:
            np.random.seed(seed)

        # Generate simple SMILES strings (mostly valid basic molecules)
        simple_molecules = [
            "CCO",  # ethanol
            "CC(=O)O",  # acetic acid
            "C1=CC=CC=C1",  # benzene
            "CCN",  # ethylamine
            "C=C",  # ethene
            "CC",  # ethane
            "C#C",  # acetylene
            "CCC",  # propane
            "CO",  # methanol
            "C1=CC=CC=C1O",  # phenol
        ]

        # Generate SMILES by randomly selecting and potentially modifying
        smiles_list = []
        for _ in range(n_samples):
            base_mol = np.random.choice(simple_molecules)
            # Sometimes add simple modifications
            if np.random.random() < 0.3:
                modifications = ["C", "CC", "O", "N"]
                base_mol += np.random.choice(modifications)
            smiles_list.append(base_mol)

        # Generate synthetic activity values
        activities = np.random.normal(0.5, 0.2, n_samples)
        activities = np.clip(activities, 0, 1)  # Keep in [0,1] range

        return pd.DataFrame({"smiles": smiles_list, "activity": activities})

    @staticmethod
    def generate_regression_dataset(
        n_samples: int = 100,
        n_features: int = 10,
        noise: float = 0.1,
        seed: Optional[int] = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic regression dataset.

        Args:
            n_samples: Number of samples
            n_features: Number of features
            noise: Noise level
            seed: Random seed

        Returns:
            Tuple of (X, y) arrays
        """
        if seed is not None:
            np.random.seed(seed)

        X = np.random.randn(n_samples, n_features)
        # Create target with some signal
        true_coef = np.random.randn(n_features)
        y = X @ true_coef + noise * np.random.randn(n_samples)

        return X, y

    @staticmethod
    def generate_classification_dataset(
        n_samples: int = 100,
        n_features: int = 10,
        n_classes: int = 2,
        seed: Optional[int] = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic classification dataset.

        Args:
            n_samples: Number of samples
            n_features: Number of features
            n_classes: Number of classes
            seed: Random seed

        Returns:
            Tuple of (X, y) arrays
        """
        if seed is not None:
            np.random.seed(seed)

        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, n_classes, n_samples)

        return X, y


def create_standard_molecular_dataset(n_samples: int = 50) -> pd.DataFrame:
    """Create a standard molecular dataset for testing.

    Args:
        n_samples: Number of samples to include

    Returns:
        DataFrame with molecular data
    """
    return TestDataGenerator.generate_molecular_dataset(n_samples)


def create_test_fingerprints(n_samples: int = 50, n_bits: int = 1024) -> np.ndarray:
    """Create test molecular fingerprints.

    Args:
        n_samples: Number of samples
        n_bits: Number of bits in fingerprint

    Returns:
        Binary fingerprint array
    """
    np.random.seed(42)
    return np.random.randint(0, 2, size=(n_samples, n_bits))


def create_test_descriptors(
    n_samples: int = 50, n_descriptors: int = 20
) -> pd.DataFrame:
    """Create test molecular descriptors.

    Args:
        n_samples: Number of samples
        n_descriptors: Number of descriptors

    Returns:
        DataFrame with descriptor values
    """
    np.random.seed(42)
    descriptor_names = [f"descriptor_{i}" for i in range(n_descriptors)]
    data = np.random.randn(n_samples, n_descriptors)

    return pd.DataFrame(data, columns=descriptor_names)
