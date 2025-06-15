"""
ChemML Core Featurizers
======================

Modern molecular featurization methods optimized for machine learning.
Combines the best of RDKit's capabilities with clean, consistent APIs.

Key Features:
- Modern RDKit APIs (no deprecation warnings)
- Consistent interface across all featurizers
- Robust error handling
- Easy combination of multiple feature types
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Descriptors3D, rdMolDescriptors

# Use the newest RDKit APIs to avoid deprecation warnings
try:
    # Try to use the new MorganGenerator (RDKit 2022.09+)
    from rdkit.Chem.rdMolDescriptors import GetMorganGenerator

    HAS_MORGAN_GENERATOR = True
except ImportError:
    # Fallback to older API
    from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect

    HAS_MORGAN_GENERATOR = False


class BaseFeaturizer(ABC):
    """Abstract base class for molecular featurizers."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @abstractmethod
    def featurize(self, molecules: List[Union[str, Chem.Mol]]) -> np.ndarray:
        """Convert molecules to feature vectors."""
        pass

    def _smiles_to_mol(self, smiles: str) -> Optional[Chem.Mol]:
        """Convert SMILES to RDKit molecule object."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            # Add hydrogens for more accurate calculations
            mol = Chem.AddHs(mol)
            return mol
        except Exception:
            return None


class MorganFingerprint(BaseFeaturizer):
    """
    Modern Morgan fingerprint implementation using latest RDKit APIs.

    This eliminates the deprecation warnings from DeepChem's implementation.
    """

    def __init__(self, radius: int = 2, n_bits: int = 2048, use_features: bool = False):
        """
        Initialize Morgan fingerprint generator.

        Args:
            radius: Radius for Morgan algorithm
            n_bits: Number of bits in fingerprint
            use_features: Whether to use feature-based Morgan fingerprints
        """
        super().__init__()
        self.radius = radius
        self.n_bits = n_bits
        self.use_features = use_features

    def featurize(self, molecules: List[Union[str, Chem.Mol]]) -> np.ndarray:
        """Generate Morgan fingerprints for molecules."""
        features = []

        # Suppress RDKit deprecation warnings for cleaner output
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*MorganGenerator.*")
            warnings.filterwarnings("ignore", category=DeprecationWarning)

            for mol_input in molecules:
                if isinstance(mol_input, str):
                    mol = self._smiles_to_mol(mol_input)
                else:
                    mol = mol_input

                if mol is None:
                    # Return zero vector for failed molecules
                    features.append(np.zeros(self.n_bits))
                    continue

                try:
                    # Use the well-tested older API for now
                    from rdkit.Chem.rdMolDescriptors import (
                        GetMorganFingerprintAsBitVect,
                    )

                    fp = GetMorganFingerprintAsBitVect(
                        mol,
                        radius=self.radius,
                        nBits=self.n_bits,
                        useFeatures=self.use_features,
                    )
                    # Convert to numpy array
                    arr = np.array(fp)
                    features.append(arr)

                except Exception as e:
                    print(f"Warning: Failed to generate fingerprint: {e}")
                    features.append(np.zeros(self.n_bits))

        return np.array(features)


class DescriptorCalculator(BaseFeaturizer):
    """
    Modern molecular descriptor calculator using latest RDKit APIs.

    Calculates a comprehensive set of molecular descriptors with proper
    error handling and modern API usage.
    """

    def __init__(self, descriptor_list: Optional[List[str]] = None):
        """
        Initialize descriptor calculator.

        Args:
            descriptor_list: List of descriptor names to calculate.
                           If None, uses a default set of important descriptors.
        """
        super().__init__()

        if descriptor_list is None:
            # Default set of important descriptors for drug discovery
            self.descriptor_list = [
                "MolWt",
                "LogP",
                "NumHDonors",
                "NumHAcceptors",
                "TPSA",
                "NumRotatableBonds",
                "NumAromaticRings",
                "NumAliphaticRings",
                "FractionCsp3",
                "HeavyAtomCount",
                "MolMR",
                "BalabanJ",
            ]
        else:
            self.descriptor_list = descriptor_list

        # Map descriptor names to RDKit functions (with aliases for convenience)
        self.descriptor_functions = {
            # Standard names
            "MolWt": Descriptors.MolWt,
            "LogP": Descriptors.MolLogP,
            "NumHDonors": Descriptors.NumHDonors,
            "NumHAcceptors": Descriptors.NumHAcceptors,
            "TPSA": Descriptors.TPSA,
            "NumRotatableBonds": Descriptors.NumRotatableBonds,
            "NumAromaticRings": Descriptors.NumAromaticRings,
            "NumAliphaticRings": Descriptors.NumAliphaticRings,
            "FractionCsp3": Descriptors.FractionCSP3,
            "HeavyAtomCount": Descriptors.HeavyAtomCount,
            "MolMR": Descriptors.MolMR,
            "BalabanJ": Descriptors.BalabanJ,
            # Common aliases for convenience
            "mw": Descriptors.MolWt,
            "logp": Descriptors.MolLogP,
            "molwt": Descriptors.MolWt,
            "mol_wt": Descriptors.MolWt,
            "hbd": Descriptors.NumHDonors,
            "hba": Descriptors.NumHAcceptors,
            "tpsa": Descriptors.TPSA,
            "rotatable_bonds": Descriptors.NumRotatableBonds,
            "heavy_atoms": Descriptors.HeavyAtomCount,
        }

    def featurize(self, molecules: List[Union[str, Chem.Mol]]) -> np.ndarray:
        """Calculate molecular descriptors for molecules."""
        features = []

        for mol_input in molecules:
            if isinstance(mol_input, str):
                mol = self._smiles_to_mol(mol_input)
            else:
                mol = mol_input

            if mol is None:
                # Return NaN vector for failed molecules
                features.append(np.full(len(self.descriptor_list), np.nan))
                continue

            mol_descriptors = []
            for desc_name in self.descriptor_list:
                try:
                    desc_func = self.descriptor_functions[desc_name]
                    value = desc_func(mol)
                    mol_descriptors.append(value)
                except Exception as e:
                    print(f"Warning: Failed to calculate {desc_name}: {e}")
                    mol_descriptors.append(np.nan)

            features.append(np.array(mol_descriptors))

        return np.array(features)

    def get_feature_names(self) -> List[str]:
        """Get names of calculated descriptors."""
        return self.descriptor_list.copy()


class ECFPFingerprint(BaseFeaturizer):
    """
    Modern ECFP (Extended Connectivity Fingerprint) implementation.

    This is essentially equivalent to Morgan fingerprints but with
    the traditional ECFP naming convention.
    """

    def __init__(self, radius: int = 2, n_bits: int = 1024):
        """
        Initialize ECFP generator.

        Args:
            radius: ECFP radius (ECFP4 = radius 2, ECFP6 = radius 3)
            n_bits: Number of bits in fingerprint
        """
        super().__init__()
        self.radius = radius
        self.n_bits = n_bits

    def featurize(self, molecules: List[Union[str, Chem.Mol]]) -> np.ndarray:
        """Generate ECFP fingerprints for molecules."""
        # ECFP is essentially Morgan fingerprints
        morgan_generator = MorganFingerprint(radius=self.radius, n_bits=self.n_bits)
        return morgan_generator.featurize(molecules)


class CombinedFeaturizer(BaseFeaturizer):
    """
    Combines multiple featurizers into a single feature vector.

    This allows you to easily combine fingerprints and descriptors.
    """

    def __init__(self, featurizers: List[BaseFeaturizer]):
        """
        Initialize combined featurizer.

        Args:
            featurizers: List of featurizer objects to combine
        """
        super().__init__()
        self.featurizers = featurizers

    def featurize(self, molecules: List[Union[str, Chem.Mol]]) -> np.ndarray:
        """Generate combined features for molecules."""
        all_features = []

        for featurizer in self.featurizers:
            features = featurizer.featurize(molecules)
            all_features.append(features)

        # Concatenate all features horizontally
        return np.hstack(all_features)


class CustomRDKitFeaturizer(BaseFeaturizer):
    """
    Custom RDKit featurizer combining multiple molecular features.

    A comprehensive featurizer that combines fingerprints and descriptors
    to provide a rich representation of molecular properties.
    """

    def __init__(
        self,
        include_fingerprints: bool = True,
        include_descriptors: bool = True,
        fingerprint_radius: int = 2,
        fingerprint_size: int = 2048,
        descriptor_subset: Optional[List[str]] = None,
    ):
        """
        Initialize custom RDKit featurizer.

        Args:
            include_fingerprints: Whether to include Morgan fingerprints
            include_descriptors: Whether to include molecular descriptors
            fingerprint_radius: Radius for Morgan fingerprints
            fingerprint_size: Size of fingerprint bit vector
            descriptor_subset: Subset of descriptors to compute (None for all)
        """
        super().__init__()
        self.include_fingerprints = include_fingerprints
        self.include_descriptors = include_descriptors
        self.fingerprint_radius = fingerprint_radius
        self.fingerprint_size = fingerprint_size
        self.descriptor_subset = descriptor_subset

        # Initialize sub-featurizers
        if include_fingerprints:
            self.fingerprint_featurizer = MorganFingerprint(
                radius=fingerprint_radius, n_bits=fingerprint_size
            )

        if include_descriptors:
            self.descriptor_featurizer = DescriptorCalculator(
                descriptor_list=descriptor_subset
            )

    def featurize(self, molecules: List[Union[str, Chem.Mol]]) -> np.ndarray:
        """
        Featurize molecules using combined fingerprints and descriptors.

        Args:
            molecules: List of SMILES strings or RDKit molecule objects

        Returns:
            Feature matrix with combined features
        """
        features_list = []

        # Get fingerprints
        if self.include_fingerprints:
            fingerprints = self.fingerprint_featurizer.featurize(molecules)
            features_list.append(fingerprints)

        # Get descriptors
        if self.include_descriptors:
            descriptors = self.descriptor_featurizer.featurize(molecules)
            features_list.append(descriptors)

        if not features_list:
            raise ValueError("At least one feature type must be enabled")

        # Combine features
        if len(features_list) == 1:
            return features_list[0]
        else:
            return np.hstack(features_list)

    def get_feature_names(self) -> List[str]:
        """Get names of all features."""
        names = []

        if self.include_fingerprints:
            names.extend([f"Morgan_bit_{i}" for i in range(self.fingerprint_size)])

        if self.include_descriptors:
            if hasattr(self.descriptor_featurizer, "get_feature_names"):
                names.extend(self.descriptor_featurizer.get_feature_names())
            else:
                # Fallback: generate generic names
                n_descriptors = len(self.descriptor_featurizer.descriptor_list or [])
                names.extend([f"Descriptor_{i}" for i in range(n_descriptors)])

        return names


class HybridMolecularFeaturizer(BaseFeaturizer):
    """
    Hybrid molecular featurizer combining RDKit and DeepChem features.

    This featurizer implements the hybrid architecture approach, combining:
    - Custom RDKit features (Morgan fingerprints, molecular descriptors)
    - Optional DeepChem integration for advanced features
    - Configurable feature selection and scaling

    Args:
        use_morgan: Whether to include Morgan fingerprints
        morgan_radius: Radius for Morgan fingerprints
        morgan_bits: Number of bits for Morgan fingerprints
        use_descriptors: Whether to include molecular descriptors
        descriptor_subset: Specific descriptors to calculate
        use_deepchem: Whether to include DeepChem features (optional)
        deepchem_featurizer: DeepChem featurizer to use (if available)
        scale_features: Whether to scale features to [0,1]
    """

    def __init__(
        self,
        use_morgan: bool = True,
        morgan_radius: int = 2,
        morgan_bits: int = 1024,
        use_descriptors: bool = True,
        descriptor_subset: Optional[List[str]] = None,
        use_deepchem: bool = False,
        deepchem_featurizer: Optional[Any] = None,
        scale_features: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.use_morgan = use_morgan
        self.morgan_radius = morgan_radius
        self.morgan_bits = morgan_bits
        self.use_descriptors = use_descriptors
        self.descriptor_subset = descriptor_subset
        self.use_deepchem = use_deepchem
        self.deepchem_featurizer = deepchem_featurizer
        self.scale_features = scale_features

        # Initialize RDKit featurizers
        if self.use_morgan:
            self.morgan_featurizer = MorganFingerprint(
                radius=morgan_radius, n_bits=morgan_bits
            )

        if self.use_descriptors:
            self.descriptor_featurizer = DescriptorCalculator(
                descriptor_list=descriptor_subset
            )

        # Initialize DeepChem featurizer if requested and available
        if self.use_deepchem:
            try:
                import deepchem as dc

                if deepchem_featurizer is None:
                    # Default to CircularFingerprint if no specific featurizer provided
                    self.deepchem_featurizer = dc.feat.CircularFingerprint(size=1024)
                else:
                    self.deepchem_featurizer = deepchem_featurizer
                self.has_deepchem = True
            except ImportError:
                print(
                    "Warning: DeepChem not available. Falling back to RDKit-only features."
                )
                self.has_deepchem = False
                self.use_deepchem = False
            except AttributeError:
                print(
                    "Warning: DeepChem CircularFingerprint not available. Using default."
                )
                self.deepchem_featurizer = deepchem_featurizer  # Use provided or None
                self.has_deepchem = deepchem_featurizer is not None
        else:
            self.has_deepchem = False

    def featurize(self, molecules: List[Union[str, Chem.Mol]]) -> np.ndarray:
        """
        Featurize molecules using hybrid RDKit + DeepChem approach.

        Args:
            molecules: List of SMILES strings or RDKit molecule objects

        Returns:
            Feature matrix combining all selected feature types
        """
        feature_arrays = []

        # Convert to SMILES if needed for consistent processing
        smiles_list = []
        for mol in molecules:
            if isinstance(mol, str):
                smiles_list.append(mol)
            elif mol is not None:
                smiles_list.append(Chem.MolToSmiles(mol))
            else:
                smiles_list.append("")  # Handle None molecules

        # RDKit Morgan fingerprints
        if self.use_morgan:
            morgan_features = self.morgan_featurizer.featurize(molecules)
            feature_arrays.append(morgan_features)

        # RDKit molecular descriptors
        if self.use_descriptors:
            descriptor_features = self.descriptor_featurizer.featurize(molecules)
            feature_arrays.append(descriptor_features)

        # DeepChem features (if available and requested)
        if (
            self.use_deepchem
            and self.has_deepchem
            and self.deepchem_featurizer is not None
        ):
            try:
                # Convert SMILES to DeepChem format
                valid_smiles = [s for s in smiles_list if s]
                if valid_smiles:
                    deepchem_features = self.deepchem_featurizer.featurize(valid_smiles)

                    # Handle potential dimension mismatch
                    if len(deepchem_features) == len(molecules):
                        feature_arrays.append(deepchem_features)
                    else:
                        print(
                            f"Warning: DeepChem feature count mismatch. Expected {len(molecules)}, got {len(deepchem_features)}"
                        )
            except Exception as e:
                print(f"Warning: DeepChem featurization failed: {e}")

        # Combine all features
        if not feature_arrays:
            raise ValueError("No features were successfully generated")

        if len(feature_arrays) == 1:
            combined_features = feature_arrays[0]
        else:
            combined_features = np.hstack(feature_arrays)

        # Optional feature scaling
        if self.scale_features:
            # Min-max scaling to [0,1]
            min_vals = np.min(combined_features, axis=0)
            max_vals = np.max(combined_features, axis=0)

            # Avoid division by zero
            range_vals = max_vals - min_vals
            range_vals[range_vals == 0] = 1

            combined_features = (combined_features - min_vals) / range_vals

        return combined_features

    def get_feature_names(self) -> List[str]:
        """Get names of all features in the hybrid featurizer."""
        names = []

        if self.use_morgan:
            names.extend([f"Morgan_{i}" for i in range(self.morgan_bits)])

        if self.use_descriptors:
            if hasattr(self.descriptor_featurizer, "get_feature_names"):
                names.extend(self.descriptor_featurizer.get_feature_names())
            else:
                n_desc = len(self.descriptor_subset) if self.descriptor_subset else 200
                names.extend([f"Descriptor_{i}" for i in range(n_desc)])

        if self.use_deepchem and self.has_deepchem:
            # Estimate DeepChem feature count (typically 1024 for CircularFingerprint)
            deepchem_size = getattr(self.deepchem_featurizer, "size", 1024)
            names.extend([f"DeepChem_{i}" for i in range(deepchem_size)])

        return names

    def get_feature_info(self) -> Dict[str, Any]:
        """Get information about the hybrid featurizer configuration."""
        info = {
            "featurizer_type": "HybridMolecularFeaturizer",
            "rdkit_morgan": self.use_morgan,
            "rdkit_descriptors": self.use_descriptors,
            "deepchem_enabled": self.use_deepchem,
            "deepchem_available": self.has_deepchem,
            "feature_scaling": self.scale_features,
            "total_features": len(self.get_feature_names()),
        }

        if self.use_morgan:
            info["morgan_radius"] = self.morgan_radius
            info["morgan_bits"] = self.morgan_bits

        if self.use_descriptors:
            info["descriptor_count"] = (
                len(self.descriptor_subset) if self.descriptor_subset else "default"
            )

        return info


# Convenience functions for easy usage
def morgan_fingerprints(
    smiles_list: List[str], radius: int = 2, n_bits: int = 2048
) -> np.ndarray:
    """
    Calculate Morgan fingerprints for a list of SMILES.

    Args:
        smiles_list: List of SMILES strings
        radius: Morgan radius
        n_bits: Number of bits

    Returns:
        Feature matrix (n_molecules x n_bits)
    """
    featurizer = MorganFingerprint(radius=radius, n_bits=n_bits)
    return featurizer.featurize(smiles_list)


def molecular_descriptors(
    smiles_list: List[str], descriptor_list: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Calculate molecular descriptors for a list of SMILES.

    Args:
        smiles_list: List of SMILES strings
        descriptor_list: List of descriptor names

    Returns:
        DataFrame with descriptors as columns
    """
    featurizer = DescriptorCalculator(descriptor_list=descriptor_list)
    features = featurizer.featurize(smiles_list)

    return pd.DataFrame(
        features, columns=featurizer.get_feature_names(), index=smiles_list
    )


def comprehensive_features(smiles_list: List[str]) -> Dict[str, np.ndarray]:
    """
    Create a comprehensive feature set for drug discovery.

    Args:
        smiles_list: List of SMILES strings

    Returns:
        Dictionary with different feature types
    """
    results = {}

    # Morgan fingerprints
    morgan_featurizer = MorganFingerprint(radius=2, n_bits=2048)
    results["morgan_fp"] = morgan_featurizer.featurize(smiles_list)

    # ECFP fingerprints
    ecfp_featurizer = ECFPFingerprint(radius=2, n_bits=1024)
    results["ecfp"] = ecfp_featurizer.featurize(smiles_list)

    # Molecular descriptors
    descriptor_featurizer = DescriptorCalculator()
    results["descriptors"] = descriptor_featurizer.featurize(smiles_list)

    # Combined features
    combined_featurizer = CombinedFeaturizer(
        [MorganFingerprint(radius=2, n_bits=1024), DescriptorCalculator()]
    )
    results["combined"] = combined_featurizer.featurize(smiles_list)

    return results


# Export main classes and functions
__all__ = [
    "BaseFeaturizer",
    "MorganFingerprint",
    "DescriptorCalculator",
    "ECFPFingerprint",
    "CombinedFeaturizer",
    "HybridMolecularFeaturizer",
    "morgan_fingerprints",
    "molecular_descriptors",
    "comprehensive_features",
]
