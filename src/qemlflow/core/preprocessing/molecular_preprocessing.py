"""
Molecular data preprocessing utilities.

This module provides functions for cleaning, normalizing, and preprocessing
molecular datasets for machine learning applications.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional imports
try:
    from rdkit import Chem
    from rdkit.Chem import SaltRemover

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


def clean_data(
    data: pd.DataFrame,
    remove_duplicates: bool = True,
    handle_missing: str = "drop",
    validate_smiles: bool = True,
) -> pd.DataFrame:
    """
    Clean molecular dataset by removing duplicates, handling missing values, etc.

    Args:
        data: Input DataFrame
        remove_duplicates: Whether to remove duplicate rows
        handle_missing: How to handle missing values ('drop', 'fill', 'interpolate')
        validate_smiles: Whether to validate SMILES strings

    Returns:
        Cleaned DataFrame
    """
    cleaned_data = data.copy()

    # Remove duplicates
    if remove_duplicates:
        cleaned_data = cleaned_data.drop_duplicates()

    # Handle missing values
    if handle_missing == "drop":
        cleaned_data = cleaned_data.dropna()
    elif handle_missing == "fill":
        # Fill numeric columns with median, categorical with mode
        for col in cleaned_data.columns:
            if cleaned_data[col].dtype in ["float64", "int64"]:
                cleaned_data[col] = cleaned_data[col].fillna(cleaned_data[col].median())
            else:
                cleaned_data[col] = cleaned_data[col].fillna(
                    cleaned_data[col].mode()[0]
                    if not cleaned_data[col].mode().empty
                    else "unknown"
                )
    elif handle_missing == "interpolate":
        cleaned_data = cleaned_data.interpolate(method="linear")

    # Validate SMILES if column exists
    if validate_smiles and "SMILES" in cleaned_data.columns:
        cleaned_data = validate_smiles_column(cleaned_data)

    return cleaned_data


def validate_smiles_column(
    data: pd.DataFrame, smiles_col: str = "SMILES"
) -> pd.DataFrame:
    """
    Validate SMILES strings and remove invalid ones.

    Args:
        data: DataFrame with SMILES column
        smiles_col: Name of SMILES column

    Returns:
        DataFrame with only valid SMILES
    """
    if RDKIT_AVAILABLE:
        valid_mask = data[smiles_col].apply(
            lambda x: Chem.MolFromSmiles(str(x)) is not None
        )
        return data[valid_mask].reset_index(drop=True)
    else:
        # Basic validation without RDKit
        valid_mask = data[smiles_col].apply(
            lambda x: isinstance(x, str) and len(x) > 0 and not x.isspace()
        )
        return data[valid_mask].reset_index(drop=True)


def standardize_smiles(smiles_list: List[str]) -> List[str]:
    """
    Standardize SMILES strings by removing salts and canonicalizing.

    Args:
        smiles_list: List of SMILES strings

    Returns:
        List of standardized SMILES
    """
    if not RDKIT_AVAILABLE:
        return smiles_list  # Return as-is if RDKit not available

    remover = SaltRemover.SaltRemover()
    standardized = []

    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                # Remove salts
                mol = remover.StripMol(mol)
                # Canonicalize
                canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
                standardized.append(canonical_smiles)
            else:
                standardized.append(smiles)  # Keep original if invalid
        except Exception:
            standardized.append(smiles)  # Keep original if error

    return standardized


def filter_by_molecular_properties(
    data: pd.DataFrame,
    smiles_col: str = "smiles",  # Default to lowercase
    mw_range: Tuple[float, float] = (50, 900),
    logp_range: Tuple[float, float] = (-3, 7),
    molecular_weight_range: Tuple[float, float] = None,
    logp_range_filter: Tuple[float, float] = None,
    apply_lipinski: bool = True,
) -> pd.DataFrame:
    """
    Filter molecules based on drug-like properties.

    Args:
        data: DataFrame with molecular data
        smiles_col: Name of SMILES column
        mw_range: Molecular weight range (min, max)
        logp_range: LogP range (min, max)
        molecular_weight_range: Alternative parameter name for mw_range
        logp_range_filter: Alternative parameter name for logp_range
        apply_lipinski: Whether to apply Lipinski's rule of five

    Returns:
        Filtered DataFrame
    """
    if not RDKIT_AVAILABLE:
        return data  # Return as-is if RDKit not available

    # Handle alternative parameter names
    if molecular_weight_range is not None:
        mw_range = molecular_weight_range
    if logp_range_filter is not None:
        logp_range = logp_range_filter

    filtered_data = data.copy()
    valid_indices = []

    for idx, smiles in enumerate(filtered_data[smiles_col]):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        # Calculate properties
        mw = Chem.Descriptors.MolWt(mol)
        logp = Chem.Descriptors.MolLogP(mol)

        # Apply filters
        if not (mw_range[0] <= mw <= mw_range[1]):
            continue
        if not (logp_range[0] <= logp <= logp_range[1]):
            continue

        # Apply Lipinski's rule of five
        if apply_lipinski:
            hbd = Chem.Descriptors.NumHDonors(mol)
            hba = Chem.Descriptors.NumHAcceptors(mol)

            if not (mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10):
                continue

        valid_indices.append(idx)

    return filtered_data.iloc[valid_indices].reset_index(drop=True)


def clean_molecular_data(data) -> Any:
    """Clean molecular data by handling missing values and infinities."""
    # Create a copy to avoid modifying the original
    data = data.copy()

    # Handle infinity values first
    data = data.replace([np.inf, -np.inf], np.nan)

    # Handle missing values
    data = data.dropna()

    # Normalize data (example: min-max normalization)
    for column in data.select_dtypes(include=["float64", "int"]):
        if len(data[column].unique()) > 1:  # Only normalize if there's variation
            min_val = data[column].min()
            max_val = data[column].max()
            if max_val != min_val:  # Avoid division by zero
                data[column] = (data[column] - min_val) / (max_val - min_val)

    return data


def preprocess_molecular_data(data) -> Any:
    # Perform cleaning and any additional preprocessing steps
    cleaned_data = clean_molecular_data(data)

    # Additional preprocessing can be added here

    return cleaned_data


def handle_missing_values(data) -> Any:
    # Example function to handle missing values
    # This can be customized based on the specific requirements
    return data.fillna(data.mean())


def normalize_data(data) -> Any:
    """Normalize numerical features in the dataset."""
    # Create a copy to avoid modifying the original
    normalized_data = data.copy()

    # Normalize numerical features in the dataset
    for column in normalized_data.select_dtypes(include=["float64", "int"]):
        mean_val = normalized_data[column].mean()
        std_val = normalized_data[column].std()
        if std_val != 0:  # Avoid division by zero
            normalized_data.loc[:, column] = (
                normalized_data[column] - mean_val
            ) / std_val

    return normalized_data


# Alias for backward compatibility
filter_by_properties = filter_by_molecular_properties


# Note: clean_data function is defined above at line 23
# No alias needed to avoid redefinition warning


def standardize_molecules(smiles_list: List[str]) -> List[str]:
    """
    Standardize molecules by canonicalizing SMILES and removing salts.

    Args:
        smiles_list: List of SMILES strings

    Returns:
        List of standardized SMILES strings
    """
    return standardize_smiles(smiles_list)


def remove_invalid_molecules(smiles_list: List[str]) -> List[str]:
    """
    Remove invalid SMILES strings from a list.

    Args:
        smiles_list: List of SMILES strings

    Returns:
        List of valid SMILES strings
    """
    if not RDKIT_AVAILABLE:
        # Basic validation without RDKit - remove empty/invalid strings
        return [
            smiles
            for smiles in smiles_list
            if smiles and isinstance(smiles, str) and len(smiles.strip()) > 0
        ]

    valid_smiles = []
    for smiles in smiles_list:
        try:
            if smiles and isinstance(smiles, str):
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    valid_smiles.append(smiles)
        except Exception:
            continue  # Skip invalid SMILES

    return valid_smiles
