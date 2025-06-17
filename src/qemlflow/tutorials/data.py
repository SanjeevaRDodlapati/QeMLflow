from typing import Dict\nfrom typing import List\nfrom typing import Optional\n"""
Educational Data Management for QeMLflow Tutorials
===============================================

This module provides curated datasets and data generation utilities specifically
designed for educational purposes in computational chemistry and machine learning.

Key Features:
- Pre-loaded molecular datasets for tutorials
- Synthetic data generation for ddef load_quantum_molecules(
    molecules: Optional[List[str]] = None,
    basis_set: str = 'STO-3G',
    include_hamiltonians: bool = True,
    difficulty_level: str = 'beginner'
) -> Any:rations
- Educational molecule collections
- Data validation and preprocessing for learning
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

# Sample molecular datasets for educational purposes
EDUCATIONAL_MOLECULES = {
    # Common drugs and their SMILES
    "drugs": {
        "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        "acetaminophen": "CC(=O)NC1=CC=C(C=C1)O",
        "morphine": "CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O",
        "penicillin": "CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C",
    },
    # Simple organic molecules for basic tutorials
    "simple_organics": {
        "methane": "C",
        "ethane": "CC",
        "propane": "CCC",
        "butane": "CCCC",
        "benzene": "c1ccccc1",
        "toluene": "Cc1ccccc1",
        "ethanol": "CCO",
        "acetone": "CC(=O)C",
    },
    # Molecules with different functional groups
    "functional_groups": {
        "carboxylic_acid": "CC(=O)O",
        "amine": "CCN",
        "alcohol": "CCO",
        "ketone": "CC(=O)C",
        "aldehyde": "CC=O",
        "ester": "CC(=O)OC",
        "ether": "COC",
        "amide": "CC(=O)N",
    },
}

# Molecular properties for educational examples
MOLECULAR_PROPERTIES = {
    "physical": ["molecular_weight", "logp", "tpsa", "hbd", "hba"],
    "descriptors": ["num_atoms", "num_bonds", "num_rings", "aromatic_rings"],
    "druglike": ["lipinski_violations", "ro5_violations", "bioavailability_score"],
}


class EducationalDatasets:
    """
    Manager for educational datasets used in QeMLflow tutorials.

    This class provides easy access to curated molecular datasets,
    synthetic data generation, and educational examples.
    """

    def __init__(self):
        self.molecules = EDUCATIONAL_MOLECULES
        self.properties = MOLECULAR_PROPERTIES
        self._cache = {}

    def load_drug_molecules(self) -> Dict[str, str]:
        """
        Load a curated collection of common drug molecules.

        Returns:
            Dict[str, str]: Dictionary mapping drug names to SMILES strings
        """
        return self.molecules["drugs"].copy()

    def load_simple_organics(self) -> Dict[str, str]:
        """
        Load simple organic molecules for basic tutorials.

        Returns:
            Dict[str, str]: Dictionary mapping molecule names to SMILES strings
        """
        return self.molecules["simple_organics"].copy()

    def load_functional_groups(self) -> Dict[str, str]:
        """
        Load molecules representing different functional groups.

        Returns:
            Dict[str, str]: Dictionary mapping functional group names to SMILES strings
        """
        return self.molecules["functional_groups"].copy()

    def get_molecule_dataset(self, dataset_name: str) -> pd.DataFrame:
        """
        Get a complete molecular dataset with computed properties.

        Args:
            dataset_name (str): Name of the dataset ('drugs', 'simple_organics', 'functional_groups')

        Returns:
            pd.DataFrame: DataFrame with molecules and their computed properties
        """
        if dataset_name not in self.molecules:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. Available: {list(self.molecules.keys())}"
            )

        cache_key = f"dataset_{dataset_name}"
        if cache_key in self._cache:
            return self._cache[cache_key].copy()

        molecules = self.molecules[dataset_name]
        data = []

        for name, smiles in molecules.items():
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    properties = self._calculate_properties(mol)
                    data.append({"name": name, "smiles": smiles, **properties})
            except Exception as e:
                warnings.warn(f"Error processing {name}: {e}")
                continue

        df = pd.DataFrame(data)
        self._cache[cache_key] = df
        return df.copy()

    def create_synthetic_examples(
        self, num_molecules: int = 50, complexity: str = "simple"
    ) -> pd.DataFrame:
        """
        Generate synthetic molecular examples for demonstrations.

        Args:
            num_molecules (int): Number of molecules to generate
            complexity (str): Complexity level ('simple', 'medium', 'complex')

        Returns:
            pd.DataFrame: DataFrame with synthetic molecules and properties
        """
        if complexity == "simple":
            return self._generate_simple_molecules(num_molecules)
        elif complexity == "medium":
            return self._generate_medium_molecules(num_molecules)
        elif complexity == "complex":
            return self._generate_complex_molecules(num_molecules)
        else:
            raise ValueError("Complexity must be 'simple', 'medium', or 'complex'")

    def get_property_ranges(self, dataset_name: str) -> Dict[str, Tuple[float, float]]:
        """
        Get the range of molecular properties for a dataset.

        Args:
            dataset_name (str): Name of the dataset

        Returns:
            Dict[str, Tuple[float, float]]: Property ranges (min, max)
        """
        df = self.get_molecule_dataset(dataset_name)
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        ranges = {}
        for col in numeric_cols:
            if col != "name":
                ranges[col] = (df[col].min(), df[col].max())

        return ranges

    def _calculate_properties(self, mol) -> Dict[str, Any]:
        """Calculate molecular properties for educational purposes."""
        try:
            return {
                "molecular_weight": Descriptors.MolWt(mol),
                "logp": Descriptors.MolLogP(mol),
                "tpsa": Descriptors.TPSA(mol),
                "hbd": Descriptors.NumHDonors(mol),
                "hba": Descriptors.NumHAcceptors(mol),
                "num_atoms": mol.GetNumAtoms(),
                "num_bonds": mol.GetNumBonds(),
                "num_rings": Descriptors.RingCount(mol),
                "aromatic_rings": Descriptors.NumAromaticRings(mol),
                "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
            }
        except Exception as e:
            warnings.warn(f"Error calculating properties: {e}")
            return {}

    def _generate_simple_molecules(self, num_molecules: int) -> pd.DataFrame:
        """Generate simple linear alkanes and alcohols."""
        data = []
        np.random.seed(42)  # For reproducibility in tutorials

        for i in range(num_molecules):
            if i % 2 == 0:
                # Generate alkane
                length = np.random.randint(1, 8)
                smiles = "C" * length if length == 1 else "C" + "C" * (length - 1)
                name = f"alkane_{length}"
            else:
                # Generate alcohol
                length = np.random.randint(1, 6)
                smiles = (
                    "C" * length + "O" if length == 1 else "C" * (length - 1) + "CO"
                )
                name = f"alcohol_{length}"

            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                properties = self._calculate_properties(mol)
                data.append({"name": name, "smiles": smiles, **properties})

        return pd.DataFrame(data)

    def _generate_medium_molecules(self, num_molecules: int) -> pd.DataFrame:
        """Generate molecules with functional groups and rings."""
        # This would include more complex synthetic generation
        # For now, return a subset of real molecules with variations
        base_molecules = {
            **self.molecules["drugs"],
            **self.molecules["functional_groups"],
        }
        data = []

        for i, (name, smiles) in enumerate(
            list(base_molecules.items())[:num_molecules]
        ):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                properties = self._calculate_properties(mol)
                data.append(
                    {"name": f"synthetic_{name}_{i}", "smiles": smiles, **properties}
                )

        return pd.DataFrame(data)

    def _generate_complex_molecules(self, num_molecules: int) -> pd.DataFrame:
        """Generate complex drug-like molecules."""
        # This would include sophisticated molecular generation
        # For now, return drug molecules with property variations
        return self.get_molecule_dataset("drugs").head(num_molecules)


def get_sample_datasets() -> Dict[str, pd.DataFrame]:
    """
    Get all available sample datasets for tutorials.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary of dataset names to DataFrames
    """
    datasets = EducationalDatasets()
    return {
        "drugs": datasets.get_molecule_dataset("drugs"),
        "simple_organics": datasets.get_molecule_dataset("simple_organics"),
        "functional_groups": datasets.get_molecule_dataset("functional_groups"),
    }


def load_educational_molecules(category: str = "all") -> Dict[str, str]:
    """
    Load educational molecules by category.

    Args:
        category (str): Category of molecules ('drugs', 'simple_organics',
                       'functional_groups', or 'all')

    Returns:
        Dict[str, str]: Dictionary mapping molecule names to SMILES strings
    """
    datasets = EducationalDatasets()

    if category == "all":
        result = {}
        for cat in datasets.molecules:
            result.update(datasets.molecules[cat])
        return result
    elif category in datasets.molecules:
        return datasets.molecules[category].copy()
    else:
        raise ValueError(
            f"Unknown category: {category}. Available: {list(datasets.molecules.keys())} or 'all'"
        )


def create_synthetic_examples(
    num_examples: int = 20, property_target: Optional[str] = None
) -> pd.DataFrame:
    """
    Create synthetic molecular examples for demonstrations.

    Args:
        num_examples (int): Number of examples to create
        property_target (str, optional): Target property to optimize for

    Returns:
        pd.DataFrame: DataFrame with synthetic examples
    """
    datasets = EducationalDatasets()

    if property_target is None:
        return datasets.create_synthetic_examples(num_examples, complexity="simple")
    else:
        # Create examples optimized for a specific property
        df = datasets.create_synthetic_examples(num_examples * 2, complexity="medium")

        if property_target in df.columns:
            # Sort by target property and take diverse examples
            df_sorted = df.sort_values(property_target)
            indices = np.linspace(0, len(df_sorted) - 1, num_examples, dtype=int)
            return df_sorted.iloc[indices].reset_index(drop=True)
        else:
            return df.head(num_examples)


# For backward compatibility and convenience
def load_tutorial_molecules():
    """Load all tutorial molecules (backward compatibility)."""
    return load_educational_molecules("all")


def get_molecule_properties(smiles_list: List[str]) -> pd.DataFrame:
    """
    Calculate properties for a list of SMILES strings.

    Args:
        smiles_list (List[str]): List of SMILES strings

    Returns:
        pd.DataFrame: DataFrame with molecules and their properties
    """
    datasets = EducationalDatasets()
    data = []

    for i, smiles in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                properties = datasets._calculate_properties(mol)
                data.append({"molecule_id": f"mol_{i}", "smiles": smiles, **properties})
        except Exception as e:
            warnings.warn(f"Error processing SMILES {smiles}: {e}")
            continue

    return pd.DataFrame(data)


def load_quantum_molecules(
    molecules: List[str] = None,
    basis_set: str = "STO-3G",
    include_hamiltonians: bool = True,
    difficulty_level: str = "beginner",
) -> Any:
    """
    Load quantum molecular systems for tutorials.

    Args:
        molecules: List of molecule names
        basis_set: Quantum chemistry basis set
        include_hamiltonians: Whether to include molecular Hamiltonians
        difficulty_level: Tutorial difficulty level

    Returns:
        Quantum molecular dataset object
    """
    if molecules is None:
        molecules = ["H2", "LiH"]

    class QuantumMolecularDataset:
        def __init__(self, mols, metadata):
            self.molecules = mols
            self.metadata = metadata

    # Create mock quantum molecular data
    mock_molecules = {mol: f"Quantum_system_{mol}" for mol in molecules}

    metadata = {
        "learning_focus": "Molecular Hamiltonians and VQE",
        "basis_set": basis_set,
        "difficulty": difficulty_level,
        "include_hamiltonians": include_hamiltonians,
    }

    print("🧬 Quantum molecular dataset loaded:")
    print(f"   • Molecules: {', '.join(molecules)}")
    print(f"   • Basis set: {basis_set}")
    print(f"   • Difficulty: {difficulty_level}")

    return QuantumMolecularDataset(mock_molecules, metadata)
