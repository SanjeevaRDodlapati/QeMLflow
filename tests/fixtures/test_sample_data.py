"""
Test data fixtures and generators for ChemML tests.

Provides standardized test data for molecular datasets, models, and results.
"""

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# Try to import molecular libraries
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolDescriptors

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


class TestDataGenerator:
    """Generator for standardized test data."""

    @staticmethod
    def generate_molecular_dataset(
        n_samples: int = 100, random_state: int = 42
    ) -> pd.DataFrame:
        """Generate synthetic molecular dataset."""
        np.random.seed(random_state)

        # Base SMILES patterns
        base_smiles = [
            "CCO",
            "CCC",
            "CCCC",
            "c1ccccc1",
            "CC(=O)O",
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
            "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F",
        ]

        # Generate dataset by sampling and modifying base SMILES
        smiles_list = []
        for i in range(n_samples):
            base_idx = i % len(base_smiles)
            smiles_list.append(base_smiles[base_idx])

        # Generate corresponding properties
        molecular_weights = np.random.normal(200, 50, n_samples)
        molecular_weights = np.clip(molecular_weights, 50, 800)  # Realistic range

        logp_values = np.random.normal(2, 1.5, n_samples)
        logp_values = np.clip(logp_values, -5, 8)  # Realistic range

        tpsa_values = np.random.normal(60, 30, n_samples)
        tpsa_values = np.clip(tpsa_values, 0, 200)  # Realistic range

        # Binary activity (based on simple rules for realism)
        activities = []
        for mw, lp, tpsa in zip(molecular_weights, logp_values, tpsa_values):
            # Simple rule: active if MW < 500, 0 < LogP < 5, TPSA < 140
            if mw < 500 and 0 < lp < 5 and tpsa < 140:
                activity = 1 if np.random.random() > 0.3 else 0  # 70% chance active
            else:
                activity = 1 if np.random.random() > 0.7 else 0  # 30% chance active
            activities.append(activity)

        # Continuous property (solubility-like)
        solubilities = -0.5 * logp_values + 0.1 * np.random.randn(n_samples)

        dataset = pd.DataFrame(
            {
                "smiles": smiles_list,
                "molecular_weight": molecular_weights,
                "logp": logp_values,
                "tpsa": tpsa_values,
                "activity": activities,
                "solubility": solubilities,
                "compound_id": [f"CMPD_{i:04d}" for i in range(n_samples)],
            }
        )

        return dataset

    @staticmethod
    def generate_regression_dataset(
        n_samples: int = 100,
        n_features: int = 10,
        noise_level: float = 0.1,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic regression dataset."""
        np.random.seed(random_state)

        # Generate features
        X = np.random.randn(n_samples, n_features)

        # Generate true coefficients
        true_coef = np.random.randn(n_features)
        true_coef[n_features // 2 :] = 0  # Make some coefficients zero (sparse)

        # Generate target with linear relationship + noise
        y = X @ true_coef + noise_level * np.random.randn(n_samples)

        return X, y

    @staticmethod
    def generate_classification_dataset(
        n_samples: int = 100,
        n_features: int = 10,
        n_classes: int = 2,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic classification dataset."""
        np.random.seed(random_state)

        # Generate features
        X = np.random.randn(n_samples, n_features)

        # Generate decision boundary
        weights = np.random.randn(n_features)
        decision_values = X @ weights

        if n_classes == 2:
            y = (decision_values > 0).astype(int)
        else:
            # Multi-class: divide decision values into bins
            thresholds = np.percentile(
                decision_values, np.linspace(0, 100, n_classes + 1)[1:-1]
            )
            y = np.digitize(decision_values, thresholds)

        return X, y

    @staticmethod
    def generate_molecular_fingerprints(
        n_molecules: int = 100, n_bits: int = 1024, random_state: int = 42
    ) -> np.ndarray:
        """Generate synthetic molecular fingerprints."""
        np.random.seed(random_state)

        # Generate sparse binary fingerprints (typical for molecules)
        fingerprints = np.zeros((n_molecules, n_bits))

        for i in range(n_molecules):
            # Each molecule has 50-200 bits set (typical range)
            n_bits_set = np.random.randint(50, 201)
            bit_indices = np.random.choice(n_bits, n_bits_set, replace=False)
            fingerprints[i, bit_indices] = 1

        return fingerprints

    @staticmethod
    def generate_quantum_circuit_data(
        n_qubits: int = 4, n_measurements: int = 1000, random_state: int = 42
    ) -> Dict[str, Any]:
        """Generate synthetic quantum circuit measurement data."""
        np.random.seed(random_state)

        # Generate measurement counts for all possible states
        n_states = 2**n_qubits
        state_names = [format(i, f"0{n_qubits}b") for i in range(n_states)]

        # Generate random counts that sum to n_measurements
        raw_counts = np.random.exponential(1, n_states)
        normalized_counts = raw_counts / raw_counts.sum() * n_measurements
        counts = {
            state: int(count) for state, count in zip(state_names, normalized_counts)
        }

        # Ensure total equals n_measurements
        total_counts = sum(counts.values())
        if total_counts != n_measurements:
            # Adjust the first state to make total correct
            counts[state_names[0]] += n_measurements - total_counts

        # Generate statevector (normalized)
        statevector = np.random.randn(n_states) + 1j * np.random.randn(n_states)
        statevector = statevector / np.linalg.norm(statevector)

        return {
            "counts": counts,
            "statevector": statevector.tolist(),
            "n_qubits": n_qubits,
            "n_measurements": n_measurements,
        }

    @staticmethod
    def create_mock_model_results(
        model_type: str = "classification", n_samples: int = 100, random_state: int = 42
    ) -> Dict[str, Any]:
        """Create mock model training/evaluation results."""
        np.random.seed(random_state)

        if model_type == "classification":
            # Generate classification metrics
            accuracy = np.random.uniform(0.7, 0.95)
            precision = np.random.uniform(0.65, 0.9)
            recall = np.random.uniform(0.6, 0.9)
            f1 = 2 * (precision * recall) / (precision + recall)

            # Generate confusion matrix
            true_positives = int(n_samples * 0.4 * recall)
            false_negatives = int(n_samples * 0.4) - true_positives
            false_positives = int(true_positives / precision) - true_positives
            true_negatives = (
                n_samples - true_positives - false_negatives - false_positives
            )

            return {
                "model_type": "classification",
                "metrics": {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                },
                "confusion_matrix": [
                    [true_negatives, false_positives],
                    [false_negatives, true_positives],
                ],
                "n_samples": n_samples,
                "training_time": np.random.uniform(10, 120),  # seconds
            }

        else:  # regression
            mse = np.random.uniform(0.1, 2.0)
            mae = mse * np.random.uniform(0.7, 0.9)  # MAE typically < MSE
            r2 = np.random.uniform(0.6, 0.95)

            return {
                "model_type": "regression",
                "metrics": {
                    "mse": mse,
                    "mae": mae,
                    "r2_score": r2,
                    "rmse": np.sqrt(mse),
                },
                "n_samples": n_samples,
                "training_time": np.random.uniform(5, 60),  # seconds
            }


class TestFileManager:
    """Manager for creating and cleaning up test files."""

    def __init__(self):
        self.temp_files = []
        self.temp_dirs = []

    def create_temp_molecular_file(
        self, data: pd.DataFrame, file_format: str = "csv"
    ) -> Path:
        """Create temporary molecular data file."""
        suffix = f".{file_format}"
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False)
        temp_path = Path(temp_file.name)
        temp_file.close()

        if file_format == "csv":
            data.to_csv(temp_path, index=False)
        elif file_format == "json":
            data.to_json(temp_path, orient="records", indent=2)
        elif file_format == "parquet":
            data.to_parquet(temp_path, index=False)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        self.temp_files.append(temp_path)
        return temp_path

    def create_temp_dir(self) -> Path:
        """Create temporary directory."""
        temp_dir = Path(tempfile.mkdtemp())
        self.temp_dirs.append(temp_dir)
        return temp_dir

    def cleanup(self):
        """Clean up all temporary files and directories."""
        for temp_file in self.temp_files:
            if temp_file.exists():
                temp_file.unlink()

        for temp_dir in self.temp_dirs:
            if temp_dir.exists():
                import shutil

                shutil.rmtree(temp_dir)

        self.temp_files.clear()
        self.temp_dirs.clear()


# Standard test datasets for consistent testing
STANDARD_TEST_SMILES = [
    "CCO",  # Ethanol
    "CC(=O)O",  # Acetic acid
    "c1ccccc1",  # Benzene
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
    "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F",  # Celecoxib
    "CN(C)CCOC(C1=CC=CC=C1)C2=CC=CC=C2",  # Diphenhydramine
    "CC(C)(C)NCC(C1=CC(=C(C=C1)O)CO)O",  # Salbutamol
]

STANDARD_TEST_PROPERTIES = {
    "molecular_weights": [46.07, 60.05, 78.11, 194.19, 206.28, 381.37, 255.35, 239.31],
    "logp_values": [-0.31, -0.17, 2.13, -0.07, 3.97, 3.47, 3.27, 0.64],
    "tpsa_values": [20.23, 37.30, 0.00, 61.83, 37.30, 92.35, 12.47, 72.72],
    "activities": [0, 0, 1, 1, 1, 1, 1, 1],
}


def create_standard_molecular_dataset() -> pd.DataFrame:
    """Create standard molecular dataset for consistent testing."""
    return pd.DataFrame(
        {
            "smiles": STANDARD_TEST_SMILES,
            "molecular_weight": STANDARD_TEST_PROPERTIES["molecular_weights"],
            "logp": STANDARD_TEST_PROPERTIES["logp_values"],
            "tpsa": STANDARD_TEST_PROPERTIES["tpsa_values"],
            "activity": STANDARD_TEST_PROPERTIES["activities"],
            "compound_id": [f"STD_{i:03d}" for i in range(len(STANDARD_TEST_SMILES))],
        }
    )


def create_performance_test_data(size: str = "medium") -> Dict[str, Any]:
    """Create datasets for performance testing."""
    sizes = {
        "small": {"n_samples": 100, "n_features": 10},
        "medium": {"n_samples": 1000, "n_features": 50},
        "large": {"n_samples": 10000, "n_features": 100},
        "xlarge": {"n_samples": 50000, "n_features": 200},
    }

    if size not in sizes:
        raise ValueError(f"Size must be one of {list(sizes.keys())}")

    params = sizes[size]
    generator = TestDataGenerator()

    # Generate different types of datasets
    X_reg, y_reg = generator.generate_regression_dataset(
        params["n_samples"], params["n_features"]
    )
    X_clf, y_clf = generator.generate_classification_dataset(
        params["n_samples"], params["n_features"]
    )

    molecular_data = generator.generate_molecular_dataset(
        min(params["n_samples"], 1000)  # Limit molecular dataset size
    )

    fingerprints = generator.generate_molecular_fingerprints(
        min(params["n_samples"], 1000), 1024
    )

    return {
        "regression": {"X": X_reg, "y": y_reg},
        "classification": {"X": X_clf, "y": y_clf},
        "molecular": molecular_data,
        "fingerprints": fingerprints,
        "size_info": params,
    }
