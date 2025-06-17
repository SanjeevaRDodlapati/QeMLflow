"""
Data Processing Module for QeMLflow

This module provides utilities for molecular data preprocessing, feature extraction,
and protein structure preparation for machine learning applications.

Modules:
    molecular_preprocessing: Molecular data cleaning and standardization
feature_extraction: Molecular descriptor and fingerprint generation
protein_preparation: Protein structure download and preparation for docking
"""

# Explicit imports from feature_extraction
from .feature_extraction import (
    calculate_properties,
    extract_basic_molecular_descriptors,
    extract_descriptors,
    extract_features,
    extract_fingerprints,
)

# Explicit imports from molecular_preprocessing
from .molecular_preprocessing import (
    clean_data,
    clean_molecular_data,
    filter_by_molecular_properties,
    handle_missing_values,
    normalize_data,
    preprocess_molecular_data,
    remove_invalid_molecules,
    standardize_molecules,
    standardize_smiles,
    validate_smiles_column,
)

# Protein preparation imports (already explicit)
from .protein_preparation import (
    ProteinPreparationPipeline,
    convert_to_pdbqt,
    download_pdb_file,
    fetch_metadata,
    prepare_proteins_simple,
)

__all__ = [
    # Feature extraction
    "calculate_properties",
    "extract_basic_molecular_descriptors",
    "extract_descriptors",
    "extract_features",
    "extract_fingerprints",
    # Molecular preprocessing
    "clean_data",
    "clean_molecular_data",
    "filter_by_molecular_properties",
    "handle_missing_values",
    "normalize_data",
    "preprocess_molecular_data",
    "remove_invalid_molecules",
    "standardize_molecules",
    "standardize_smiles",
    "validate_smiles_column",
    # Protein preparation
    "ProteinPreparationPipeline",
    "download_pdb_file",
    "convert_to_pdbqt",
    "fetch_metadata",
    "prepare_proteins_simple",
]
