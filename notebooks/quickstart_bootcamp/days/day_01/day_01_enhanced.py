#!/usr/bin/env python3
"""
Day 1: ML & Cheminformatics - Enhanced Production-Ready Script
=============================================================

Clean, modular implementation of the Day 1 ChemML bootcamp using the new
common infrastructure. Demonstrates best practices for maintainable code.

Author: ChemML Enhancement System
Version: 2.0.0

Features:
- Modular architecture with clear separation of concerns
- Unified configuration and library management
- Comprehensive error handling with graceful fallbacks
- Standardized assessment and reporting
- Type safety and modern Python practices
"""

import sys
from pathlib import Path

# Add the project root to Python path for imports
project_root = Path(
    __file__
).parent.parent.parent  # Go up to quickstart_bootcamp directory
sys.path.insert(0, str(project_root))

import logging
from typing import Any, Dict, List, Optional

from chemml_common import (
    AssessmentFramework,
    BaseRunner,
    LibraryManager,
    get_config,
    print_banner,
)
from chemml_common.core.base_runner import SectionResult, SectionRunner


class MolecularRepresentationSection(SectionRunner):
    """Section 1: Molecular Representations and Basic Cheminformatics."""

    def __init__(self, config):
        super().__init__(config, "molecular_representation")
        self.library_manager = LibraryManager()

    def execute(self) -> SectionResult:
        """Execute molecular representation demonstrations."""
        self.logger.info("Starting molecular representation section")

        try:
            outputs = {}

            # Setup libraries
            rdkit_available = self._setup_libraries()

            # Generate sample molecules
            sample_molecules = self._get_sample_molecules()
            outputs["sample_molecules"] = sample_molecules

            # Demonstrate different representations
            if rdkit_available:
                outputs["smiles_parsing"] = self._demonstrate_smiles_parsing(
                    sample_molecules
                )
                outputs["molecular_fingerprints"] = self._demonstrate_fingerprints(
                    sample_molecules
                )
                outputs["molecular_descriptors"] = self._demonstrate_descriptors(
                    sample_molecules
                )
            else:
                outputs[
                    "fallback_representations"
                ] = self._demonstrate_fallback_representations(sample_molecules)

            # Create assessment
            self._create_assessment()

            return self._create_result(
                success=True,
                outputs=outputs,
                metadata={
                    "molecules_processed": len(sample_molecules),
                    "rdkit_available": rdkit_available,
                },
            )

        except Exception as e:
            self.logger.error("Molecular representation section failed: %s", str(e))
            return self._create_result(
                success=False, errors=[f"Molecular representation failed: {str(e)}"]
            )

    def _setup_libraries(self) -> bool:
        """Setup required libraries with fallbacks."""
        success, _ = self.library_manager.import_library("rdkit")
        if not success:
            self.logger.warning("RDKit not available, using fallback implementations")
        return success

    def _get_sample_molecules(self) -> List[str]:
        """Get sample SMILES strings for demonstration."""
        return [
            "CCO",  # Ethanol
            "CC(=O)O",  # Acetic acid
            "c1ccccc1",  # Benzene
            "CCN(CC)CC",  # Triethylamine
            "CC(C)C",  # Isobutane
        ]

    def _demonstrate_smiles_parsing(self, smiles_list: List[str]) -> Dict[str, Any]:
        """Demonstrate SMILES parsing with RDKit."""
        rdkit = self.library_manager.get_module("rdkit")

        results = {}
        for smiles in smiles_list:
            try:
                mol = rdkit.MolFromSmiles(smiles)
                if mol:
                    results[smiles] = {
                        "valid": True,
                        "num_atoms": mol.GetNumAtoms(),
                        "num_bonds": mol.GetNumBonds(),
                        "molecular_formula": rdkit.MolToFormula(mol),
                    }
                else:
                    results[smiles] = {"valid": False}
            except Exception as e:
                results[smiles] = {"valid": False, "error": str(e)}

        self.logger.info("Parsed %d SMILES strings", len(smiles_list))
        return results

    def _demonstrate_fingerprints(self, smiles_list: List[str]) -> Dict[str, Any]:
        """Demonstrate molecular fingerprint generation."""
        rdkit = self.library_manager.get_module("rdkit")

        fingerprints = {}
        for smiles in smiles_list:
            try:
                mol = rdkit.MolFromSmiles(smiles)
                if mol:
                    # Generate Morgan fingerprint
                    fp = rdkit.GetMorganFingerprint(mol, 2)
                    fingerprints[smiles] = {
                        "fingerprint_length": len(fp.GetNonzeroElements()),
                        "available": True,
                    }
                else:
                    fingerprints[smiles] = {"available": False}
            except Exception as e:
                fingerprints[smiles] = {"available": False, "error": str(e)}

        self.logger.info("Generated fingerprints for %d molecules", len(smiles_list))
        return fingerprints

    def _demonstrate_descriptors(self, smiles_list: List[str]) -> Dict[str, Any]:
        """Demonstrate molecular descriptor calculation."""
        rdkit = self.library_manager.get_module("rdkit")

        descriptors = {}
        for smiles in smiles_list:
            try:
                mol = rdkit.MolFromSmiles(smiles)
                if mol:
                    descriptors[smiles] = {
                        "molecular_weight": rdkit.Descriptors.MolWt(mol),
                        "logp": rdkit.Descriptors.MolLogP(mol),
                        "num_rotatable_bonds": rdkit.Descriptors.NumRotatableBonds(mol),
                        "available": True,
                    }
                else:
                    descriptors[smiles] = {"available": False}
            except Exception as e:
                descriptors[smiles] = {"available": False, "error": str(e)}

        self.logger.info("Calculated descriptors for %d molecules", len(smiles_list))
        return descriptors

    def _demonstrate_fallback_representations(
        self, smiles_list: List[str]
    ) -> Dict[str, Any]:
        """Demonstrate basic representations without RDKit."""
        representations = {}
        for smiles in smiles_list:
            representations[smiles] = {
                "smiles": smiles,
                "length": len(smiles),
                "contains_aromatic": "c" in smiles.lower(),
                "contains_nitrogen": "n" in smiles.lower(),
                "contains_oxygen": "o" in smiles.lower(),
            }

        self.logger.info(
            "Generated fallback representations for %d molecules", len(smiles_list)
        )
        return representations

    def _create_assessment(self) -> None:
        """Create assessment questions for this section."""
        # This would be handled by the assessment framework
        pass


class PropertyPredictionSection(SectionRunner):
    """Section 2: Property Prediction and QSAR Modeling."""

    def __init__(self, config):
        super().__init__(config, "property_prediction")
        self.library_manager = LibraryManager()

    def execute(self) -> SectionResult:
        """Execute property prediction demonstrations."""
        self.logger.info("Starting property prediction section")

        try:
            outputs = {}

            # Setup libraries
            sklearn_available = self._setup_libraries()

            # Generate sample data
            dataset = self._create_sample_dataset()
            outputs["dataset"] = dataset

            # Demonstrate property prediction
            if sklearn_available:
                outputs["predictions"] = self._demonstrate_ml_prediction(dataset)
            else:
                outputs["fallback_predictions"] = self._demonstrate_simple_prediction(
                    dataset
                )

            return self._create_result(
                success=True,
                outputs=outputs,
                metadata={
                    "samples_processed": len(dataset["features"]),
                    "sklearn_available": sklearn_available,
                },
            )

        except Exception as e:
            self.logger.error("Property prediction section failed: %s", str(e))
            return self._create_result(
                success=False, errors=[f"Property prediction failed: {str(e)}"]
            )

    def _setup_libraries(self) -> bool:
        """Setup required libraries."""
        success, _ = self.library_manager.import_library("sklearn")
        return success

    def _create_sample_dataset(self) -> Dict[str, Any]:
        """Create a simple sample dataset for demonstration."""
        import random

        random.seed(42)  # For reproducibility

        # Simple features: molecular weight, logP, number of atoms
        features = []
        targets = []

        for i in range(50):
            mw = random.uniform(50, 500)
            logp = random.uniform(-2, 5)
            atoms = random.randint(5, 30)

            features.append([mw, logp, atoms])

            # Synthetic target: solubility based on features
            target = (
                10 - 0.01 * mw - 0.5 * abs(logp) + 0.1 * atoms + random.gauss(0, 0.5)
            )
            targets.append(target)

        return {
            "features": features,
            "targets": targets,
            "feature_names": ["molecular_weight", "logp", "num_atoms"],
            "target_name": "solubility",
        }

    def _demonstrate_ml_prediction(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Demonstrate ML-based property prediction."""
        sklearn = self.library_manager.get_module("sklearn")

        # Split data
        X, y = dataset["features"], dataset["targets"]
        split_idx = int(0.8 * len(X))

        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Train simple model
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error, r2_score

        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        results = {
            "model_type": "RandomForestRegressor",
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "mse": mean_squared_error(y_test, y_pred),
            "r2_score": r2_score(y_test, y_pred),
            "feature_importance": model.feature_importances_.tolist(),
        }

        self.logger.info("ML prediction completed with R² = %.3f", results["r2_score"])
        return results

    def _demonstrate_simple_prediction(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Demonstrate simple prediction without sklearn."""
        # Simple linear model based on molecular weight
        features = dataset["features"]
        targets = dataset["targets"]

        # Calculate correlation with molecular weight
        mw_values = [f[0] for f in features]
        correlation = self._calculate_correlation(mw_values, targets)

        # Simple prediction: use average target value
        avg_target = sum(targets) / len(targets)

        results = {
            "model_type": "simple_average",
            "prediction_value": avg_target,
            "mw_correlation": correlation,
            "samples_used": len(targets),
        }

        self.logger.info("Simple prediction completed")
        return results

    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient."""
        n = len(x)
        if n == 0:
            return 0.0

        mean_x = sum(x) / n
        mean_y = sum(y) / n

        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        denominator_x = sum((x[i] - mean_x) ** 2 for i in range(n))
        denominator_y = sum((y[i] - mean_y) ** 2 for i in range(n))

        if denominator_x == 0 or denominator_y == 0:
            return 0.0

        return numerator / (denominator_x * denominator_y) ** 0.5


class Day01Runner(BaseRunner):
    """Enhanced runner for Day 1 ChemML activities."""

    def setup_sections(self):
        """Register all sections for Day 1."""
        self.register_section(MolecularRepresentationSection(self.config))
        self.register_section(PropertyPredictionSection(self.config))

        # Additional sections can be added here
        self.logger.info("Registered %d sections for Day 1", len(self.section_runners))


def main():
    """Main entry point - clean and simple."""
    try:
        # Get configuration
        config = get_config(day=1, script_name="ml_cheminformatics_enhanced")

        # Print banner
        print_banner(
            config, "Machine Learning & Cheminformatics Foundations - Enhanced"
        )

        # Create and run
        runner = Day01Runner(config)
        runner.setup_sections()
        summary = runner.execute_all_sections()

        # Final status
        if summary.overall_success:
            print("🎉 Day 1 completed successfully!")
        else:
            print("⚠️ Day 1 completed with some issues. Check logs for details.")

    except Exception as e:
        logging.error("Fatal error in Day 1 script: %s", str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
