"""
Enhanced Data Loading and Processing for ChemML
Comprehensive data loaders for chemistry and drug discovery datasets.
"""

import gzip
import json
import os
import tarfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests

try:
    from rdkit import Chem, RDLogger

    # Suppress RDKit warnings
    RDLogger.DisableLog("rdApp.*")
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


class ChemMLDataLoader:
    """Comprehensive data loader for chemical datasets."""

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize data loader with optional caching."""
        self.cache_dir = (
            Path(cache_dir) if cache_dir else Path.home() / ".chemml" / "data"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Common dataset URLs
        self.dataset_urls = {
            "qm9": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv",
            "tox21": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz",
            "bace": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv",
            "bbbp": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
            "clintox": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz",
            "esol": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv",
            "freesolv": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv",
            "lipophilicity": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv",
        }

    def load_dataset(
        self,
        dataset_name: str,
        force_download: bool = False,
        smiles_column: str = "smiles",
        target_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Load a chemical dataset with automatic downloading and caching.

        Args:
            dataset_name: Name of dataset ('qm9', 'tox21', 'bace', etc.)
            force_download: Force re-download even if cached
            smiles_column: Name of column containing SMILES strings
            target_columns: Specific target columns to load

        Returns:
            DataFrame with chemical data
        """
        if dataset_name not in self.dataset_urls:
            raise ValueError(
                f"Dataset '{dataset_name}' not supported. "
                f"Available: {list(self.dataset_urls.keys())}"
            )

        cache_file = self.cache_dir / f"{dataset_name}.csv"

        # Load from cache if available
        if cache_file.exists() and not force_download:
            df = pd.read_csv(cache_file)
        else:
            # Download and cache
            df = self._download_dataset(dataset_name)
            df.to_csv(cache_file, index=False)

        # Validate SMILES if RDKit available
        if HAS_RDKIT and smiles_column in df.columns:
            df = self._validate_smiles(df, smiles_column)

        # Filter columns if specified
        if target_columns:
            columns_to_keep = [smiles_column] + target_columns
            available_columns = [col for col in columns_to_keep if col in df.columns]
            df = df[available_columns]

        return df

    def load_custom_dataset(
        self,
        file_path: Union[str, Path],
        smiles_column: str = "smiles",
        target_columns: Optional[List[str]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Load custom dataset from file.

        Args:
            file_path: Path to dataset file (.csv, .xlsx, .json, .sdf)
            smiles_column: Name of column containing SMILES strings
            target_columns: Specific target columns to load
            **kwargs: Additional arguments for pandas readers

        Returns:
            DataFrame with chemical data
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        # Load based on file extension
        if file_path.suffix.lower() == ".csv":
            df = pd.read_csv(file_path, **kwargs)
        elif file_path.suffix.lower() in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path, **kwargs)
        elif file_path.suffix.lower() == ".json":
            df = pd.read_json(file_path, **kwargs)
        elif file_path.suffix.lower() == ".sdf":
            df = self._load_sdf_file(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        # Validate SMILES if RDKit available
        if HAS_RDKIT and smiles_column in df.columns:
            df = self._validate_smiles(df, smiles_column)

        # Filter columns if specified
        if target_columns:
            columns_to_keep = [smiles_column] + target_columns
            available_columns = [col for col in columns_to_keep if col in df.columns]
            df = df[available_columns]

        return df

    def create_benchmark_dataset(
        self,
        dataset_names: List[str],
        sample_size: Optional[int] = None,
        random_state: int = 42,
    ) -> Dict[str, pd.DataFrame]:
        """
        Create benchmark dataset collection for evaluation.

        Args:
            dataset_names: List of dataset names to include
            sample_size: Optional sampling for large datasets
            random_state: Random seed for reproducibility

        Returns:
            Dictionary of dataset name to DataFrame
        """
        benchmark_data = {}

        for dataset_name in dataset_names:
            try:
                df = self.load_dataset(dataset_name)

                # Sample if requested
                if sample_size and len(df) > sample_size:
                    df = df.sample(n=sample_size, random_state=random_state)

                benchmark_data[dataset_name] = df
                print(f"✅ Loaded {dataset_name}: {len(df)} samples")

            except Exception as e:
                print(f"❌ Failed to load {dataset_name}: {e}")

        return benchmark_data

    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """Get information about a dataset."""
        if dataset_name not in self.dataset_urls:
            raise ValueError(f"Dataset '{dataset_name}' not supported")

        # Load dataset to get info
        df = self.load_dataset(dataset_name)

        return {
            "name": dataset_name,
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "url": self.dataset_urls[dataset_name],
        }

    def _download_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Download dataset from URL."""
        url = self.dataset_urls[dataset_name]
        print(f"Downloading {dataset_name} from {url}...")

        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Handle compressed files
        if url.endswith(".gz"):

            content = gzip.decompress(response.content)
            df = pd.read_csv(pd.io.common.StringIO(content.decode("utf-8")))
        else:
            df = pd.read_csv(pd.io.common.StringIO(response.text))

        print(f"✅ Downloaded {dataset_name}: {len(df)} samples")
        return df

    def _validate_smiles(self, df: pd.DataFrame, smiles_column: str) -> pd.DataFrame:
        """Validate SMILES strings and remove invalid ones."""
        if not HAS_RDKIT:
            return df

        initial_count = len(df)

        def is_valid_smiles(smiles):
            if pd.isna(smiles) or smiles == "":
                return False
            mol = Chem.MolFromSmiles(str(smiles))
            return mol is not None

        # Filter valid SMILES
        valid_mask = df[smiles_column].apply(is_valid_smiles)
        df_clean = df[valid_mask].copy()

        removed_count = initial_count - len(df_clean)
        if removed_count > 0:
            print(
                f"⚠️  Removed {removed_count} invalid SMILES ({removed_count/initial_count*100:.1f}%)"
            )

        return df_clean

    def _load_sdf_file(self, file_path: Path) -> pd.DataFrame:
        """Load SDF file using RDKit."""
        if not HAS_RDKIT:
            raise ImportError("RDKit required for SDF file loading")

        supplier = Chem.SDMolSupplier(str(file_path))
        molecules = []

        for mol in supplier:
            if mol is not None:
                mol_data = {"smiles": Chem.MolToSmiles(mol)}

                # Extract properties
                for prop_name in mol.GetPropNames():
                    mol_data[prop_name] = mol.GetProp(prop_name)

                molecules.append(mol_data)

        return pd.DataFrame(molecules)


class AdvancedDataPreprocessor:
    """Advanced preprocessing pipeline for chemical datasets."""

    def __init__(self):
        """Initialize preprocessor."""
        self.scaler_dict = {}
        self.encoder_dict = {}
        self.feature_names = {}

    def create_preprocessing_pipeline(
        self,
        df: pd.DataFrame,
        smiles_column: str = "smiles",
        target_columns: Optional[List[str]] = None,
        feature_types: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Create comprehensive preprocessing pipeline.

        Args:
            df: Input DataFrame
            smiles_column: Name of SMILES column
            target_columns: Target columns for prediction
            feature_types: Dictionary mapping column names to types ('numerical', 'categorical')

        Returns:
            Dictionary with processed data and metadata
        """
        # Separate features and targets
        feature_columns = [
            col
            for col in df.columns
            if col not in [smiles_column] + (target_columns or [])
        ]

        X_features = df[feature_columns].copy() if feature_columns else pd.DataFrame()
        X_smiles = df[smiles_column].copy()
        y = df[target_columns].copy() if target_columns else None

        # Process molecular features if available
        if not X_features.empty:
            X_features = self._process_molecular_features(X_features, feature_types)

        # Generate molecular fingerprints/descriptors
        X_molecular = self._generate_molecular_features(X_smiles)

        # Combine features
        if not X_features.empty:
            X_combined = pd.concat([X_features, X_molecular], axis=1)
        else:
            X_combined = X_molecular

        # Process targets
        if y is not None:
            y = self._process_targets(y)

        return {
            "X": X_combined,
            "y": y,
            "feature_names": list(X_combined.columns),
            "smiles": X_smiles,
            "preprocessing_info": {
                "scalers": self.scaler_dict,
                "encoders": self.encoder_dict,
                "feature_counts": {
                    "molecular": len(X_molecular.columns),
                    "additional": (
                        len(X_features.columns) if not X_features.empty else 0
                    ),
                    "total": len(X_combined.columns),
                },
            },
        }

    def _process_molecular_features(
        self, X: pd.DataFrame, feature_types: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """Process additional molecular features."""
        X_processed = X.copy()

        # Auto-detect feature types if not provided
        if feature_types is None:
            feature_types = {}
            for col in X.columns:
                if X[col].dtype in ["object", "category"]:
                    feature_types[col] = "categorical"
                else:
                    feature_types[col] = "numerical"

        # Process categorical features
        for col, ftype in feature_types.items():
            if col in X.columns:
                if ftype == "categorical":
                    X_processed = self._encode_categorical(X_processed, col)
                elif ftype == "numerical":
                    X_processed = self._scale_numerical(X_processed, col)

        return X_processed

    def _generate_molecular_features(self, smiles: pd.Series) -> pd.DataFrame:
        """Generate molecular fingerprints and descriptors."""
        if not HAS_RDKIT:
            # Fallback: create simple features based on SMILES strings
            features = pd.DataFrame(
                {
                    "smiles_length": smiles.str.len(),
                    "char_counts_C": smiles.str.count("C"),
                    "char_counts_N": smiles.str.count("N"),
                    "char_counts_O": smiles.str.count("O"),
                    "ring_count": smiles.str.count("c"),  # aromatic carbons as proxy
                }
            )
            return features

        # Use RDKit for proper molecular features with updated API
        from rdkit.Chem import Descriptors

        try:
            # Try new MorganGenerator API (RDKit 2022.03+)
            from rdkit.Chem.rdMolDescriptors import MorganGenerator

            use_new_api = True
        except ImportError:
            # Fallback to older API
            from rdkit.Chem import rdMolDescriptors

            use_new_api = False

        features_list = []

        # Initialize generators if using new API
        if use_new_api:
            morgan_gen_1 = MorganGenerator(radius=1)
            morgan_gen_2 = MorganGenerator(radius=2)

        for smi in smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                features = {
                    "mw": Descriptors.MolWt(mol),
                    "logp": Descriptors.MolLogP(mol),
                    "hbd": Descriptors.NumHDonors(mol),
                    "hba": Descriptors.NumHAcceptors(mol),
                    "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
                    "aromatic_rings": Descriptors.NumAromaticRings(mol),
                    "tpsa": Descriptors.TPSA(mol),
                }

                # Add Morgan fingerprint features with proper API
                if use_new_api:
                    # Use new MorganGenerator
                    fp1 = morgan_gen_1.GetSparseCountFingerprint(mol)
                    fp2 = morgan_gen_2.GetSparseCountFingerprint(mol)
                    features["morgan_bits_1"] = len(fp1.GetNonzeroElements())
                    features["morgan_bits_2"] = len(fp2.GetNonzeroElements())
                else:
                    # Use legacy API with suppressed warnings
                    import warnings

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        fp1 = rdMolDescriptors.GetMorganFingerprint(mol, 1)
                        fp2 = rdMolDescriptors.GetMorganFingerprint(mol, 2)
                        features["morgan_bits_1"] = len(fp1.GetNonzeroElements())
                        features["morgan_bits_2"] = len(fp2.GetNonzeroElements())
            else:
                features = {
                    key: np.nan
                    for key in [
                        "mw",
                        "logp",
                        "hbd",
                        "hba",
                        "rotatable_bonds",
                        "aromatic_rings",
                        "tpsa",
                        "morgan_bits_1",
                        "morgan_bits_2",
                    ]
                }

            features_list.append(features)

        return pd.DataFrame(features_list).fillna(0)  # Fill NaN values with 0

    def _encode_categorical(self, X: pd.DataFrame, column: str) -> pd.DataFrame:
        """Encode categorical variables."""
        from sklearn.preprocessing import LabelEncoder

        encoder = LabelEncoder()
        X[column] = encoder.fit_transform(X[column].astype(str))
        self.encoder_dict[column] = encoder

        return X

    def _scale_numerical(self, X: pd.DataFrame, column: str) -> pd.DataFrame:
        """Scale numerical variables."""
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X[column] = scaler.fit_transform(X[[column]])
        self.scaler_dict[column] = scaler

        return X

    def _process_targets(self, y: pd.DataFrame) -> pd.DataFrame:
        """Process target variables."""
        y_processed = y.copy()

        # Handle missing values
        if y_processed.isnull().any().any():
            print("⚠️  Found missing values in targets, filling with median/mode")
            for col in y_processed.columns:
                if y_processed[col].dtype in ["float64", "int64"]:
                    y_processed[col].fillna(y_processed[col].median(), inplace=True)
                else:
                    y_processed[col].fillna(y_processed[col].mode()[0], inplace=True)

        return y_processed


class IntelligentDataSplitter:
    """Intelligent data splitting with chemical awareness."""

    def __init__(self):
        """Initialize splitter."""
        pass

    def split_dataset(
        self,
        X: pd.DataFrame,
        y: Optional[pd.DataFrame] = None,
        smiles: Optional[pd.Series] = None,
        test_size: float = 0.2,
        val_size: float = 0.1,
        split_method: str = "random",
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """
        Intelligent dataset splitting with multiple strategies.

        Args:
            X: Feature matrix
            y: Target matrix
            smiles: SMILES strings for molecular-aware splitting
            test_size: Fraction for test set
            val_size: Fraction for validation set
            split_method: 'random', 'scaffold', 'stratified', 'temporal'
            random_state: Random seed

        Returns:
            Dictionary with train/val/test splits
        """
        _n_samples = len(X)

        if split_method == "random":
            return self._random_split(X, y, test_size, val_size, random_state)
        elif split_method == "scaffold" and smiles is not None:
            return self._scaffold_split(X, y, smiles, test_size, val_size, random_state)
        elif split_method == "stratified" and y is not None:
            return self._stratified_split(X, y, test_size, val_size, random_state)
        elif split_method == "temporal":
            return self._temporal_split(X, y, test_size, val_size)
        else:
            print(f"⚠️  Split method '{split_method}' not available, using random")
            return self._random_split(X, y, test_size, val_size, random_state)

    def _random_split(self, X, y, test_size, val_size, random_state):
        """Random dataset splitting."""
        from sklearn.model_selection import train_test_split

        # First split: train+val vs test
        if y is not None:
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
        else:
            X_temp, X_test = train_test_split(
                X, test_size=test_size, random_state=random_state
            )
            y_temp, y_test = None, None

        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        if y_temp is not None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
            )
        else:
            X_train, X_val = train_test_split(
                X_temp, test_size=val_size_adjusted, random_state=random_state
            )
            y_train, y_val = None, None

        return {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
            "split_info": {
                "method": "random",
                "train_size": len(X_train),
                "val_size": len(X_val),
                "test_size": len(X_test),
            },
        }

    def _scaffold_split(self, X, y, smiles, test_size, val_size, random_state):
        """Scaffold-based splitting for molecular data."""
        if not HAS_RDKIT:
            print("⚠️  RDKit not available, using random split")
            return self._random_split(X, y, test_size, val_size, random_state)

        # Generate scaffolds
        scaffolds = self._generate_scaffolds(smiles)

        # Group by scaffold
        scaffold_to_indices = {}
        for idx, scaffold in enumerate(scaffolds):
            if scaffold not in scaffold_to_indices:
                scaffold_to_indices[scaffold] = []
            scaffold_to_indices[scaffold].append(idx)

        # Sort scaffolds by size (largest first)
        scaffold_sizes = [
            (scaffold, len(indices))
            for scaffold, indices in scaffold_to_indices.items()
        ]
        scaffold_sizes.sort(key=lambda x: x[1], reverse=True)

        # Allocate to splits
        total_size = len(X)
        test_target = int(total_size * test_size)
        val_target = int(total_size * val_size)

        test_indices, val_indices, train_indices = [], [], []
        test_count, val_count = 0, 0

        for scaffold, size in scaffold_sizes:
            indices = scaffold_to_indices[scaffold]

            if test_count < test_target:
                test_indices.extend(indices)
                test_count += size
            elif val_count < val_target:
                val_indices.extend(indices)
                val_count += size
            else:
                train_indices.extend(indices)

        # Create splits
        X_train, X_val, X_test = (
            X.iloc[train_indices],
            X.iloc[val_indices],
            X.iloc[test_indices],
        )

        if y is not None:
            y_train, y_val, y_test = (
                y.iloc[train_indices],
                y.iloc[val_indices],
                y.iloc[test_indices],
            )
        else:
            y_train, y_val, y_test = None, None, None

        return {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
            "split_info": {
                "method": "scaffold",
                "train_size": len(X_train),
                "val_size": len(X_val),
                "test_size": len(X_test),
                "n_scaffolds": len(scaffold_to_indices),
            },
        }

    def _generate_scaffolds(self, smiles):
        """Generate molecular scaffolds."""
        from rdkit.Chem.Scaffolds import MurckoScaffold

        scaffolds = []
        for smi in smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                scaffold_smiles = Chem.MolToSmiles(scaffold)
                scaffolds.append(scaffold_smiles)
            else:
                scaffolds.append("invalid")

        return scaffolds

    def _stratified_split(self, X, y, test_size, val_size, random_state):
        """Stratified splitting for classification tasks."""

        # Use first target column for stratification
        _target_col = y.columns[0] if hasattr(y, "columns") else 0
        stratify_target = y.iloc[:, 0] if hasattr(y, "iloc") else y

        # Handle continuous targets by binning
        if stratify_target.dtype in ["float64", "float32"]:
            stratify_target = pd.cut(stratify_target, bins=5, labels=False)

        # First split
        X_temp, X_test, y_temp, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            stratify=stratify_target,
            random_state=random_state,
        )

        # Second split
        val_size_adjusted = val_size / (1 - test_size)
        stratify_temp = (
            stratify_target.iloc[X_temp.index]
            if hasattr(stratify_target, "iloc")
            else stratify_target[X_temp.index]
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=val_size_adjusted,
            stratify=stratify_temp,
            random_state=random_state,
        )

        return {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
            "split_info": {
                "method": "stratified",
                "train_size": len(X_train),
                "val_size": len(X_val),
                "test_size": len(X_test),
            },
        }

    def _temporal_split(self, X, y, test_size, val_size):
        """Temporal splitting (chronological order)."""
        n_total = len(X)
        n_test = int(n_total * test_size)
        n_val = int(n_total * val_size)
        n_train = n_total - n_test - n_val

        # Split chronologically
        X_train = X.iloc[:n_train]
        X_val = X.iloc[n_train : n_train + n_val]
        X_test = X.iloc[n_train + n_val :]

        if y is not None:
            y_train = y.iloc[:n_train]
            y_val = y.iloc[n_train : n_train + n_val]
            y_test = y.iloc[n_train + n_val :]
        else:
            y_train, y_val, y_test = None, None, None

        return {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
            "split_info": {
                "method": "temporal",
                "train_size": len(X_train),
                "val_size": len(X_val),
                "test_size": len(X_test),
            },
        }

    def scaffold_split(
        self, smiles: pd.Series, test_size: float = 0.2, random_state: int = 42
    ) -> Tuple[List[int], List[int]]:
        """
        Scaffold-based splitting to prevent data leakage.

        Args:
            smiles: Series of SMILES strings
            test_size: Fraction for test set
            random_state: Random seed

        Returns:
            Tuple of (train_indices, test_indices)
        """
        if not HAS_RDKIT:
            print("RDKit not available, falling back to random split")

            indices = list(range(len(smiles)))
            return train_test_split(
                indices, test_size=test_size, random_state=random_state
            )

        try:
            # Generate scaffolds
            scaffolds = self._generate_scaffolds(smiles)

            # Group by scaffold
            scaffold_to_indices = {}
            for i, scaffold in enumerate(scaffolds):
                if scaffold not in scaffold_to_indices:
                    scaffold_to_indices[scaffold] = []
                scaffold_to_indices[scaffold].append(i)

            # Sort scaffolds by size (largest first)
            scaffold_sizes = [
                (scaffold, len(indices))
                for scaffold, indices in scaffold_to_indices.items()
            ]
            scaffold_sizes.sort(key=lambda x: x[1], reverse=True)

            # Allocate to splits
            total_size = len(smiles)
            test_target = int(total_size * test_size)

            test_indices, train_indices = [], []
            test_count = 0

            for scaffold, size in scaffold_sizes:
                indices = scaffold_to_indices[scaffold]

                if test_count < test_target:
                    test_indices.extend(indices)
                    test_count += size
                else:
                    train_indices.extend(indices)

            return train_indices, test_indices

        except Exception as e:
            print(f"Scaffold split failed: {e}, using random split")

            indices = list(range(len(smiles)))
            return train_test_split(
                indices, test_size=test_size, random_state=random_state
            )

    def temporal_split(
        self, timestamps: pd.Series, test_size: float = 0.2
    ) -> Tuple[List[int], List[int]]:
        """
        Temporal splitting based on timestamps.

        Args:
            timestamps: Series of timestamps
            test_size: Fraction for test set

        Returns:
            Tuple of (train_indices, test_indices)
        """
        try:
            # Sort by timestamp
            sorted_indices = timestamps.sort_values().index.tolist()

            # Split chronologically
            n_total = len(sorted_indices)
            n_test = int(n_total * test_size)

            train_indices = sorted_indices[:-n_test]
            test_indices = sorted_indices[-n_test:]

            return train_indices, test_indices

        except Exception as e:
            print(f"Temporal split failed: {e}, using random split")

            indices = list(range(len(timestamps)))
            return train_test_split(indices, test_size=test_size, random_state=42)

    def stratified_split(
        self, targets: pd.Series, test_size: float = 0.2, random_state: int = 42
    ) -> Tuple[List[int], List[int]]:
        """
        Stratified splitting to maintain class balance.

        Args:
            targets: Series of target values
            test_size: Fraction for test set
            random_state: Random seed

        Returns:
            Tuple of (train_indices, test_indices)
        """
        try:

            # Handle continuous targets by binning
            stratify_target = targets
            if targets.dtype in ["float64", "float32"]:
                stratify_target = pd.cut(targets, bins=5, labels=False)

            indices = list(range(len(targets)))
            return train_test_split(
                indices,
                test_size=test_size,
                stratify=stratify_target,
                random_state=random_state,
            )

        except Exception as e:
            print(f"Stratified split failed: {e}, using random split")

            indices = list(range(len(targets)))
            return train_test_split(
                indices, test_size=test_size, random_state=random_state
            )


# Convenience functions for quick access
def load_chemical_dataset(dataset_name: str, **kwargs):
    """Load a chemical dataset quickly."""
    loader = ChemMLDataLoader()
    return loader.load_dataset(dataset_name, **kwargs)


def preprocess_chemical_data(df, smiles_column="smiles", target_columns=None):
    """Preprocess chemical data quickly."""
    preprocessor = AdvancedDataPreprocessor()
    return preprocessor.create_preprocessing_pipeline(df, smiles_column, target_columns)


def split_chemical_data(
    data,
    smiles_column="smiles",
    method="scaffold",
    test_size=0.2,
    val_size=0.1,
    **kwargs,
):
    """Split chemical data with different strategies."""
    splitter = IntelligentDataSplitter()

    # Extract features, targets, and SMILES
    X = data.drop(columns=[smiles_column])
    y = None
    smiles = data[smiles_column]

    return splitter.split_dataset(
        X=X,
        y=y,
        smiles=smiles,
        test_size=test_size,
        val_size=val_size,
        split_method=method,
        **kwargs,
    )
