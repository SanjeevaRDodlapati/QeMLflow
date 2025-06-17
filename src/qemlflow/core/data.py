from typing import Dict\nfrom typing import List\nfrom typing import Union\n"""
QeMLflow Core Data Processing
==========================

Handles molecular datasets, preprocessing, and data quality assurance.

Key Features:
- Molecular data cleaning and validation
- SMILES processing and standardization
- Dataset splitting and sampling strategies
- Data quality metrics and reporting
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import MolStandardize, SaltRemover

    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
try:
    from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class MolecularDataProcessor:
    """
    Comprehensive molecular data processing pipeline.

    Handles SMILES validation, molecular standardization, and dataset preparation.
    """

    def __init__(
        self,
        standardize_molecules: bool = True,
        remove_salts: bool = True,
        validate_structures: bool = True,
    ):
        """
        Initialize molecular data processor.

        Args:
            standardize_molecules: Whether to standardize molecular structures
            remove_salts: Whether to remove salts from molecules
            validate_structures: Whether to validate molecular structures
        """
        self.standardize_molecules = standardize_molecules
        self.remove_salts = remove_salts
        self.validate_structures = validate_structures
        if HAS_RDKIT:
            if remove_salts:
                self.salt_remover = SaltRemover.SaltRemover()
            if standardize_molecules:
                self.standardizer = MolStandardize.StandardizeSmiles()
        elif any([standardize_molecules, remove_salts, validate_structures]):
            warnings.warn(
                "RDKit not available. Molecular processing features disabled."
            )

    def clean_smiles(self, smiles_list: List[str]) -> Tuple[List[str], List[bool]]:
        """
        Clean and validate SMILES strings.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            Tuple of (cleaned_smiles, valid_flags)
        """
        if not HAS_RDKIT:
            return smiles_list, [True] * len(smiles_list)
        cleaned_smiles = []
        valid_flags = []
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    cleaned_smiles.append(smiles)
                    valid_flags.append(False)
                    continue
                if self.remove_salts:
                    mol = self.salt_remover.StripMol(mol)
                if self.standardize_molecules:
                    standardized_smiles = self.standardizer(Chem.MolToSmiles(mol))
                    cleaned_smiles.append(standardized_smiles)
                else:
                    cleaned_smiles.append(Chem.MolToSmiles(mol))
                valid_flags.append(True)
            except Exception:
                cleaned_smiles.append(smiles)
                valid_flags.append(False)
        return cleaned_smiles, valid_flags

    def process_dataset(
        self,
        data: pd.DataFrame,
        smiles_column: str = "smiles",
        target_columns: Optional[List[str]] = None,
        remove_invalid: bool = True,
    ) -> pd.DataFrame:
        """
        Process a molecular dataset.

        Args:
            data: Input DataFrame
            smiles_column: Name of SMILES column
            target_columns: Names of target columns
            remove_invalid: Whether to remove invalid molecules

        Returns:
            Processed DataFrame
        """
        processed_data = data.copy()
        if smiles_column in processed_data.columns:
            smiles_list = processed_data[smiles_column].tolist()
            cleaned_smiles, valid_flags = self.clean_smiles(smiles_list)
            processed_data[smiles_column] = cleaned_smiles
            processed_data["is_valid"] = valid_flags
            if remove_invalid:
                processed_data = processed_data[processed_data["is_valid"]]
                processed_data = processed_data.drop("is_valid", axis=1)
        processed_data = processed_data.drop_duplicates()
        if target_columns:
            for col in target_columns:
                if col in processed_data.columns:
                    processed_data = processed_data.dropna(subset=[col])
        return processed_data.reset_index(drop=True)


class DataSplitter:
    """
    Advanced data splitting strategies for molecular datasets.

    Provides various splitting methods including scaffold-based splits
    for avoiding data leakage in drug discovery applications.
    """

    def __init__(self, random_state: int = 42):
        """Initialize data splitter."""
        self.random_state = random_state

    def random_split(
        self, data: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Perform random train/validation/test split.

        Args:
            data: Input DataFrame
            test_size: Fraction for test set
            val_size: Fraction for validation set

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        train_val, test = train_test_split(
            data, test_size=test_size, random_state=self.random_state
        )
        val_fraction = val_size / (1 - test_size)
        train, val = train_test_split(
            train_val, test_size=val_fraction, random_state=self.random_state
        )
        return train, val, test

    def stratified_split(
        self,
        data: pd.DataFrame,
        target_column: str,
        test_size: float = 0.2,
        val_size: float = 0.1,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Perform stratified split based on target distribution.

        Args:
            data: Input DataFrame
            target_column: Name of target column for stratification
            test_size: Fraction for test set
            val_size: Fraction for validation set

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if not HAS_SKLEARN:
            warnings.warn("sklearn not available. Using random split instead.")
            return self.random_split(data, test_size, val_size)
        y = data[target_column]
        if y.dtype in ["float64", "float32"]:
            y_binned = pd.cut(y, bins=5, labels=False)
        else:
            y_binned = y
        train_val, test = train_test_split(
            data, test_size=test_size, stratify=y_binned, random_state=self.random_state
        )
        val_fraction = val_size / (1 - test_size)
        y_train_val = train_val[target_column]
        if y_train_val.dtype in ["float64", "float32"]:
            y_train_val_binned = pd.cut(y_train_val, bins=5, labels=False)
        else:
            y_train_val_binned = y_train_val
        train, val = train_test_split(
            train_val,
            test_size=val_fraction,
            stratify=y_train_val_binned,
            random_state=self.random_state,
        )
        return train, val, test

    def scaffold_split(
        self,
        data: pd.DataFrame,
        smiles_column: str = "smiles",
        test_size: float = 0.2,
        val_size: float = 0.1,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Perform scaffold-based split to avoid data leakage.

        Groups molecules by their Bemis-Murcko scaffolds and assigns
        entire scaffold groups to train/val/test sets.

        Args:
            data: Input DataFrame
            smiles_column: Name of SMILES column
            test_size: Fraction for test set
            val_size: Fraction for validation set

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if not HAS_RDKIT:
            warnings.warn("RDKit not available. Using random split instead.")
            return self.random_split(data, test_size, val_size)
        try:
            from rdkit.Chem.Scaffolds import MurckoScaffold

            scaffolds = {}
            for idx, smiles in enumerate(data[smiles_column]):
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol)
                        if scaffold not in scaffolds:
                            scaffolds[scaffold] = []
                        scaffolds[scaffold].append(idx)
                except Exception:
                    scaffold = f"invalid_{idx}"
                    scaffolds[scaffold] = [idx]
            scaffold_list = sorted(
                scaffolds.items(), key=lambda x: len(x[1]), reverse=True
            )
            train_indices, val_indices, test_indices = [], [], []
            train_size, val_size_target, test_size_target = 0, 0, 0
            total_size = len(data)
            for scaffold, indices in scaffold_list:
                if test_size_target < test_size * total_size:
                    test_indices.extend(indices)
                    test_size_target += len(indices)
                elif val_size_target < val_size * total_size:
                    val_indices.extend(indices)
                    val_size_target += len(indices)
                else:
                    train_indices.extend(indices)
                    train_size += len(indices)
            train_df = data.iloc[train_indices].reset_index(drop=True)
            val_df = data.iloc[val_indices].reset_index(drop=True)
            test_df = data.iloc[test_indices].reset_index(drop=True)
            return train_df, val_df, test_df
        except ImportError:
            warnings.warn(
                "Scaffold splitting requires RDKit. Using random split instead."
            )
            return self.random_split(data, test_size, val_size)


class FeatureScaler:
    """
    Feature scaling utilities for molecular descriptors and fingerprints.
    """

    def __init__(self, method_type: str = "standard"):
        """
        Initialize feature scaler.

        Args:
            method: Scaling method ('standard', 'minmax', 'robust')
        """
        if not HAS_SKLEARN:
            raise ImportError("sklearn required for feature scaling")
        self.method = method_type
        if method_type == "standard":
            self.scaler = StandardScaler()
        elif method_type == "minmax":
            self.scaler = MinMaxScaler()
        elif method_type == "robust":
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method_type}")

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit scaler and transform features."""
        return self.scaler.fit_transform(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features using fitted scaler."""
        return self.scaler.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform scaled features."""
        return self.scaler.inverse_transform(X)


class DataProcessor:
    """
    General data processing utilities for QeMLflow workflows.

    Provides common data processing operations for molecular and chemical data.
    """

    def __init__(self):
        pass

    @staticmethod
    def normalize_features(
        data: Union[pd.DataFrame, np.ndarray], method_type: Any = "standardize"
    ) -> Any:
        """Normalize feature data using specified method."""

        if method_type == "standardize":
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            return (data - mean) / (std + 1e-08)
        elif method_type == "minmax":
            min_val = np.min(data, axis=0)
            max_val = np.max(data, axis=0)
            return (data - min_val) / (max_val - min_val + 1e-08)
        else:
            return data

    @staticmethod
    def split_data(
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        test_size: int = 0.2,
        random_state: int = 42,
    ) -> Any:
        """Split data into train/test sets."""
        from sklearn.model_selection import train_test_split

        return train_test_split(X, y, test_size=test_size, random_state=random_state)


def load_sample_data(dataset_name: str = "tox21") -> pd.DataFrame:
    """
    Load sample molecular datasets for testing and tutorials.

    Args:
        dataset_name: Name of dataset to load

    Returns:
        DataFrame with molecular data
    """
    if dataset_name == "tox21":
        sample_smiles = [
            "CCO",
            "CC(=O)OC1=CC=CC=C1C(=O)O",
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
            "CC1=CC=C(C=C1)C(=O)C2=CC=CC=C2",
        ]
        np.random.seed(42)
        n_compounds = len(sample_smiles)
        data = pd.DataFrame(
            {
                "smiles": sample_smiles,
                "tox_endpoint_1": np.random.randint(0, 2, n_compounds),
                "tox_endpoint_2": np.random.randint(0, 2, n_compounds),
                "molecular_weight": np.random.uniform(100, 500, n_compounds),
            }
        )
    elif dataset_name == "solubility":
        sample_smiles = ["CCO", "CCCCCCCC", "CC(=O)O", "O", "C1=CC=CC=C1"]
        np.random.seed(42)
        data = pd.DataFrame(
            {"smiles": sample_smiles, "logS": [-0.24, -5.15, -0.17, 0, -2.13]}
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return data


def calculate_data_quality_metrics(
    data: pd.DataFrame, smiles_column: str = "smiles"
) -> Dict[str, Any]:
    """
    Calculate data quality metrics for a molecular dataset.

    Args:
        data: Input DataFrame
        smiles_column: Name of SMILES column

    Returns:
        Dictionary with quality metrics
    """
    metrics = {
        "total_compounds": len(data),
        "missing_values": data.isnull().sum().to_dict(),
        "duplicate_rows": data.duplicated().sum(),
    }
    if HAS_RDKIT and smiles_column in data.columns:
        valid_count = 0
        for smiles in data[smiles_column]:
            try:
                mol = Chem.MolFromSmiles(str(smiles))
                if mol is not None:
                    valid_count += 1
            except Exception:
                pass
        metrics["valid_smiles"] = valid_count
        metrics["invalid_smiles"] = len(data) - valid_count
        metrics["smiles_validity_rate"] = (
            valid_count / len(data) if len(data) > 0 else 0
        )
    return metrics


def quick_clean(
    data: pd.DataFrame,
    smiles_column: str = "smiles",
    target_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Quickly clean a molecular dataset using default settings."""
    processor = MolecularDataProcessor()
    return processor.process_dataset(data, smiles_column, target_columns)


def quick_split(
    data: pd.DataFrame, method_type: str = "random", **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Quickly split data using specified method."""
    splitter = DataSplitter()
    if method_type == "random":
        return splitter.random_split(data, **kwargs)
    elif method_type == "stratified":
        return splitter.stratified_split(data, **kwargs)
    elif method_type == "scaffold":
        return splitter.scaffold_split(data, **kwargs)
    else:
        raise ValueError(f"Unknown split method: {method_type}")


from typing import Optional


def legacy_molecular_cleaning(
    df: pd.DataFrame,
    remove_duplicates: List[Any] = True,
    handle_missing: Any = "drop",
    validate_smiles: List[Any] = True,
    **kwargs,
) -> Any:
    """
    Clean molecular dataset using legacy cleaning functions.

    Wrapper around src.data_processing.molecular_preprocessing for backward compatibility
    and integration with new architecture.
    """
    try:
        import os
        import sys

        sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
        from src.data_processing.molecular_preprocessing import clean_data

        return clean_data(
            df,
            remove_duplicates=remove_duplicates,
            handle_missing=handle_missing,
            validate_smiles=validate_smiles,
        )
    except ImportError as e:
        warnings.warn(f"Legacy data processing module not available: {e}")
        return df


def enhanced_property_prediction() -> Union[pd.DataFrame, np.ndarray]:
    """
    Access to enhanced property prediction capabilities.

    Integrates legacy drug design functionality with new architecture.
    """
    try:
        from qemlflow.research.drug_discovery.properties import (
            MolecularPropertyPredictor,
        )

        return MolecularPropertyPredictor()
    except ImportError as e:
        warnings.warn(f"Drug discovery module not available: {e}")
        return None


class LegacyModuleWrapper:
    """
    Wrapper for legacy QeMLflow modules to ensure backward compatibility.

    Provides a compatibility layer for older QeMLflow modules while transitioning
    to the new hybrid architecture.
    """

    def __init__(self, module_name: str, fallback_message: Optional[str] = None):
        self.module_name = module_name
        self.fallback_message = (
            fallback_message or f"Legacy module {module_name} wrapped for compatibility"
        )

    def wrap_function(self, func_name: str, *args, **kwargs) -> Any:
        """Wrap a legacy function call with error handling."""
        try:
            print(f"🔧 Legacy wrapper: {self.module_name}.{func_name}")
            print(f"💡 {self.fallback_message}")
            return None
        except Exception as e:
            print(f"⚠️ Legacy function {func_name} not available: {e}")
            return None

    @staticmethod
    def create_compatibility_layer() -> Any:
        """Create compatibility wrappers for common legacy modules."""
        wrappers = {}
        legacy_modules = [
            "data_processing",
            "drug_design",
            "molecular_dynamics",
            "qsar_modeling",
            "descriptor_calculation",
        ]
        for module in legacy_modules:
            wrappers[module] = LegacyModuleWrapper(
                module, "Use modern qemlflow.core or qemlflow.research modules instead"
            )
        return wrappers


__all__ = [
    "MolecularDataProcessor",
    "DataSplitter",
    "FeatureScaler",
    "DataProcessor",
    "load_sample_data",
    "calculate_data_quality_metrics",
    "quick_clean",
    "quick_split",
    "legacy_molecular_cleaning",
    "enhanced_property_prediction",
]
