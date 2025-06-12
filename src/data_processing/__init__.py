"""
Data Processing Module for ChemML

This module provides utilities for molecular data preprocessing, feature extraction,
and protein structure preparation for machine learning applications.

Modules:
    molecular_preprocessing: Molecular data cleaning and standardization
    feature_extraction: Molecular descriptor and fingerprint generation
    protein_preparation: Protein structure download and preparation for docking
"""

from .feature_extraction import *
from .molecular_preprocessing import *
from .protein_preparation import (
    ProteinPreparationPipeline,
    convert_to_pdbqt,
    download_pdb_file,
    fetch_metadata,
    prepare_proteins_simple,
)

__all__ = [
    # Molecular preprocessing
    "molecular_preprocessing",
    # Feature extraction
    "feature_extraction",
    # Protein preparation
    "ProteinPreparationPipeline",
    "download_pdb_file",
    "convert_to_pdbqt",
    "fetch_metadata",
    "prepare_proteins_simple",
]
