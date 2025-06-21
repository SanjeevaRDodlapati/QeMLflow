"""
Unit tests for data processing module.

Tests molecular preprocessing and feature extraction functionality.
"""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

# Import modules under test
try:
    from qemlflow.core.preprocessing.feature_extraction import (
        calculate_properties,
        extract_descriptors,
        extract_structural_features,
        generate_fingerprints,
    )
    from qemlflow.core.preprocessing.molecular_preprocessing import (
        clean_data,
        filter_by_properties,
        normalize_data,
        remove_invalid_molecules,
        standardize_molecules,
    )
except ImportError as e:
    pytest.skip(f"Data processing modules not available: {e}", allow_module_level=True)

from tests.conftest import skip_if_no_deepchem, skip_if_no_rdkit

try:
    from rdkit import Chem
except ImportError:
    pass


class TestMolecularPreprocessing:
    """Test molecular preprocessing functions."""

    def test_clean_data_basic(self, sample_molecular_data):
        """Test basic data cleaning functionality."""
        # Add some NaN values for testing
        dirty_data = sample_molecular_data.copy()
        dirty_data.loc[0, "molecular_weight"] = np.nan
        dirty_data.loc[1, "logp"] = np.inf

        cleaned_data = clean_data(dirty_data)

        # Should handle NaN and infinite values
        assert not cleaned_data.isnull().any().any()
        assert np.isfinite(cleaned_data.select_dtypes(include=[np.number]).values).all()

    def test_clean_data_empty_input(self):
        """Test clean_data with empty DataFrame."""
        empty_df = pd.DataFrame()
        result = clean_data(empty_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_normalize_data(self, sample_molecular_data):
        """Test data normalization."""
        numerical_cols = ["molecular_weight", "logp", "tpsa"]
        data_subset = sample_molecular_data[numerical_cols]

        normalized_data = normalize_data(data_subset)

        # Check normalization (should be roughly mean=0, std=1)
        assert abs(normalized_data.mean().mean()) < 1e-10
        assert abs(normalized_data.std().mean() - 1.0) < 1e-10

    @skip_if_no_rdkit
    def test_remove_invalid_molecules(self, sample_smiles):
        """Test removal of invalid SMILES."""
        # Add some invalid SMILES
        invalid_smiles = sample_smiles + ["invalid_smiles", "", "C(C)(C"]

        valid_smiles = remove_invalid_molecules(invalid_smiles)

        # Should only return valid SMILES
        assert len(valid_smiles) == len(sample_smiles)
        assert all(smiles in sample_smiles for smiles in valid_smiles)

    @skip_if_no_rdkit
    def test_standardize_molecules(self, sample_smiles):
        """Test molecule standardization."""
        standardized = standardize_molecules(sample_smiles)

        # Should return same number of molecules
        assert len(standardized) == len(sample_smiles)
        # All should be valid SMILES strings
        assert all(isinstance(smiles, str) for smiles in standardized)

    def test_filter_by_properties(self, sample_molecular_data):
        """Test filtering molecules by properties."""
        # Filter by molecular weight
        filtered = filter_by_properties(
            sample_molecular_data, molecular_weight_range=(50, 200)
        )

        # Should only contain molecules in range
        assert all(50 <= mw <= 200 for mw in filtered["molecular_weight"])

    def test_filter_by_properties_no_matches(self, sample_molecular_data):
        """Test filtering with criteria that match nothing."""
        filtered = filter_by_properties(
            sample_molecular_data,
            molecular_weight_range=(1000, 2000),  # No molecules in this range
        )

        assert len(filtered) == 0


class TestFeatureExtraction:
    """Test feature extraction functions."""

    @skip_if_no_rdkit
    def test_extract_descriptors(self, sample_molecules):
        """Test molecular descriptor extraction."""
        descriptors = extract_descriptors(sample_molecules)

        # Should return DataFrame with molecules as rows
        assert isinstance(descriptors, pd.DataFrame)
        assert len(descriptors) == len(sample_molecules)
        # Should have multiple descriptor columns
        assert descriptors.shape[1] > 5

    @skip_if_no_rdkit
    def test_extract_descriptors_empty_input(self):
        """Test descriptor extraction with empty input."""
        descriptors = extract_descriptors([])

        assert isinstance(descriptors, pd.DataFrame)
        assert len(descriptors) == 0

    @skip_if_no_rdkit
    def test_generate_fingerprints(self, sample_molecules):
        """Test molecular fingerprint generation."""
        fingerprints = generate_fingerprints(
            sample_molecules, fp_type="morgan", n_bits=1024
        )

        # Should return numpy array
        assert isinstance(fingerprints, np.ndarray)
        assert fingerprints.shape == (len(sample_molecules), 1024)
        # Fingerprints should be binary
        assert np.all((fingerprints == 0) | (fingerprints == 1))

    @skip_if_no_rdkit
    def test_generate_fingerprints_different_types(self, sample_molecules):
        """Test different fingerprint types."""
        fp_types = ["morgan", "topological", "maccs"]

        for fp_type in fp_types:
            fingerprints = generate_fingerprints(sample_molecules, fp_type=fp_type)
            assert isinstance(fingerprints, np.ndarray)
            assert fingerprints.shape[0] == len(sample_molecules)

    @skip_if_no_rdkit
    def test_calculate_properties(self, sample_molecules):
        """Test molecular property calculation."""
        properties = calculate_properties(sample_molecules)

        # Should return DataFrame with standard properties
        assert isinstance(properties, pd.DataFrame)
        assert len(properties) == len(sample_molecules)

        # Check for expected property columns
        expected_props = ["molecular_weight", "logp", "tpsa", "hbd", "hba"]
        for prop in expected_props:
            assert prop in properties.columns

    @skip_if_no_rdkit
    def test_extract_structural_features(self, sample_molecules):
        """Test structural feature extraction."""
        features = extract_structural_features(sample_molecules)

        # Should return meaningful structural information
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(sample_molecules)
        assert features.shape[1] > 0

    def test_feature_extraction_error_handling(self):
        """Test error handling in feature extraction."""
        # Test with invalid input
        with pytest.raises((ValueError, TypeError)):
            extract_descriptors("not_a_list")

        with pytest.raises((ValueError, TypeError)):
            generate_fingerprints("not_a_list")


class TestDataProcessingIntegration:
    """Integration tests for data processing pipeline."""

    @skip_if_no_rdkit
    def test_full_preprocessing_pipeline(self, sample_molecular_data):
        """Test complete preprocessing pipeline."""
        # Simulate full pipeline
        smiles_list = sample_molecular_data["smiles"].tolist()

        # Step 1: Remove invalid molecules
        valid_smiles = remove_invalid_molecules(smiles_list)

        # Step 2: Standardize molecules
        standardized_smiles = standardize_molecules(valid_smiles)

        # Step 3: Convert to molecules
        from rdkit import Chem

        molecules = [Chem.MolFromSmiles(smiles) for smiles in standardized_smiles]
        molecules = [mol for mol in molecules if mol is not None]

        # Step 4: Extract features
        descriptors = extract_descriptors(molecules)
        fingerprints = generate_fingerprints(molecules)

        # Verify pipeline results
        assert len(molecules) <= len(smiles_list)  # May filter out some
        assert len(descriptors) == len(molecules)
        assert fingerprints.shape[0] == len(molecules)

    def test_preprocessing_with_missing_data(self):
        """Test preprocessing handles missing data correctly."""
        # Create data with missing values
        data = pd.DataFrame(
            {
                "smiles": ["CCO", None, "invalid", "c1ccccc1"],
                "activity": [1, 0, np.nan, 1],
                "property": [1.5, np.inf, -1.0, 2.0],
            }
        )

        cleaned = clean_data(data)

        # Should handle missing values appropriately
        assert not cleaned.isnull().any().any()
        assert np.isfinite(cleaned.select_dtypes(include=[np.number]).values).all()


class TestPerformance:
    """Performance tests for data processing operations."""

    @pytest.mark.slow
    @skip_if_no_rdkit
    def test_large_dataset_processing(self, performance_timer):
        """Test processing performance with larger datasets."""
        # Create larger dataset
        large_smiles = ["CCO"] * 1000  # Simple molecule repeated

        performance_timer.start()
        # Test preprocessing performance
        valid_smiles = remove_invalid_molecules(large_smiles)
        performance_timer.stop()

        # Should complete in reasonable time (adjust threshold as needed)
        assert performance_timer.elapsed < 10.0  # 10 seconds max
        assert len(valid_smiles) == len(large_smiles)

    @pytest.mark.slow
    @skip_if_no_rdkit
    def test_fingerprint_generation_performance(self, performance_timer):
        """Test fingerprint generation performance."""
        from rdkit import Chem

        # Create test molecules
        molecules = [Chem.MolFromSmiles("CCO")] * 500

        performance_timer.start()
        fingerprints = generate_fingerprints(molecules)
        performance_timer.stop()

        # Should complete in reasonable time
        assert performance_timer.elapsed < 5.0  # 5 seconds max
        assert fingerprints.shape == (500, 1024)  # Default fingerprint size
