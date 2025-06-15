#!/usr/bin/env python3
"""
Comprehensive Bootcamp Notebook Validation
==========================================

This script validates the core functionality that the bootcamp notebooks depend on,
without requiring notebook execution. It tests:

1. Core library imports
2. ChemML module availability
3. Dataset loading capabilities
4. Model training workflows
5. Quantum computing features

Usage:
    python validate_bootcamp_notebooks.py
    python validate_bootcamp_notebooks.py --verbose
"""

import argparse
import importlib
import sys
import traceback
import warnings
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class BootcampValidator:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.results = {
            "core_libraries": {},
            "chemml_modules": {},
            "quantum_libraries": {},
            "ml_workflows": {},
            "data_processing": {},
        }

    def log(self, message, level="INFO"):
        if self.verbose or level in ["ERROR", "SUCCESS"]:
            prefix = {"INFO": "🔍", "SUCCESS": "✅", "ERROR": "❌", "WARNING": "⚠️"}.get(
                level, "📝"
            )
            print(f"{prefix} {message}")

    def test_import(self, module_name, test_name=None):
        """Test if a module can be imported"""
        try:
            module = importlib.import_module(module_name)
            self.log(f"Successfully imported {module_name}", "SUCCESS")
            return True, module
        except ImportError as e:
            self.log(f"Failed to import {module_name}: {e}", "ERROR")
            return False, None
        except Exception as e:
            self.log(f"Error importing {module_name}: {e}", "ERROR")
            return False, None

    def validate_core_libraries(self):
        """Test core scientific libraries used across all days"""
        self.log("Testing Core Scientific Libraries...")

        core_libs = [
            "numpy",
            "pandas",
            "matplotlib",
            "seaborn",
            "sklearn",
            "rdkit",
            "deepchem",
            "scipy",
            "networkx",
            "plotly",
        ]

        for lib in core_libs:
            success, module = self.test_import(lib)
            self.results["core_libraries"][lib] = success

        return self.results["core_libraries"]

    def validate_chemml_modules(self):
        """Test ChemML module structure and imports"""
        self.log("Testing ChemML Module Structure...")

        # Test new hybrid architecture
        chemml_modules = [
            "chemml.core.featurizers",
            "chemml.core.data",
            "chemml.research.quantum",
            "chemml.integrations.deepchem_integration",
        ]

        for module in chemml_modules:
            success, mod = self.test_import(module)
            self.results["chemml_modules"][module] = success

        # Test legacy module integration
        try:
            from chemml.core.data import (
                enhanced_property_prediction,
                legacy_molecular_cleaning,
            )

            self.log("Legacy integration functions available", "SUCCESS")
            self.results["chemml_modules"]["legacy_integration"] = True
        except ImportError:
            self.log("Legacy integration not available", "ERROR")
            self.results["chemml_modules"]["legacy_integration"] = False

        return self.results["chemml_modules"]

    def validate_quantum_libraries(self):
        """Test quantum computing libraries for Days 6-7"""
        self.log("Testing Quantum Computing Libraries...")

        quantum_libs = [
            "qiskit",
            "qiskit_aer",
            "qiskit_nature",
            "qiskit.algorithms",
            "qiskit.primitives",
        ]

        for lib in quantum_libs:
            success, module = self.test_import(lib)
            self.results["quantum_libraries"][lib] = success

        # Test specific quantum functionality
        try:
            from qiskit import QuantumCircuit
            from qiskit.algorithms import VQE
            from qiskit_nature.second_q.drivers import PySCFDriver

            self.log("Core quantum classes available", "SUCCESS")
            self.results["quantum_libraries"]["core_functionality"] = True
        except ImportError as e:
            self.log(f"Quantum functionality issue: {e}", "ERROR")
            self.results["quantum_libraries"]["core_functionality"] = False

        return self.results["quantum_libraries"]

    def validate_ml_workflows(self):
        """Test ML workflow capabilities"""
        self.log("Testing ML Workflow Components...")

        try:
            # Test scikit-learn workflow
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.metrics import mean_squared_error
            from sklearn.model_selection import train_test_split

            self.log("Scikit-learn workflow ready", "SUCCESS")
            self.results["ml_workflows"]["sklearn"] = True
        except ImportError:
            self.results["ml_workflows"]["sklearn"] = False

        try:
            # Test DeepChem workflow
            import deepchem as dc
            from deepchem.models import GraphConvModel

            self.log("DeepChem workflow ready", "SUCCESS")
            self.results["ml_workflows"]["deepchem"] = True
        except ImportError:
            self.results["ml_workflows"]["deepchem"] = False

        try:
            # Test RDKit molecular processing
            from rdkit import Chem
            from rdkit.Chem import Descriptors, rdMolDescriptors

            # Test basic molecule processing
            mol = Chem.MolFromSmiles("CCO")
            if mol is not None:
                mw = Descriptors.MolWt(mol)
                self.log("RDKit molecular processing working", "SUCCESS")
                self.results["ml_workflows"]["rdkit_processing"] = True
            else:
                self.results["ml_workflows"]["rdkit_processing"] = False
        except Exception as e:
            self.log(f"RDKit processing error: {e}", "ERROR")
            self.results["ml_workflows"]["rdkit_processing"] = False

        return self.results["ml_workflows"]

    def validate_data_processing(self):
        """Test data loading and processing capabilities"""
        self.log("Testing Data Processing Capabilities...")

        try:
            # Test if we can access datasets
            import deepchem as dc

            # Try to load a small dataset
            tasks, datasets, transformers = dc.molnet.load_delaney(featurizer="ECFP")
            train, valid, test = datasets

            if len(train) > 0:
                self.log("Dataset loading functional", "SUCCESS")
                self.results["data_processing"]["dataset_loading"] = True
            else:
                self.results["data_processing"]["dataset_loading"] = False
        except Exception as e:
            self.log(f"Dataset loading error: {e}", "ERROR")
            self.results["data_processing"]["dataset_loading"] = False

        return self.results["data_processing"]

    def run_validation(self):
        """Run complete validation"""
        self.log("🧪 Starting Comprehensive Bootcamp Validation\n" + "=" * 50)

        # Run all validation tests
        self.validate_core_libraries()
        self.validate_chemml_modules()
        self.validate_quantum_libraries()
        self.validate_ml_workflows()
        self.validate_data_processing()

        # Generate summary report
        self.generate_report()

    def generate_report(self):
        """Generate comprehensive validation report"""
        print("\n" + "=" * 50)
        print("📊 BOOTCAMP VALIDATION SUMMARY")
        print("=" * 50)

        total_tests = 0
        passed_tests = 0

        for category, tests in self.results.items():
            print(f"\n📋 {category.replace('_', ' ').title()}:")
            print("-" * 30)

            for test_name, result in tests.items():
                total_tests += 1
                if result:
                    passed_tests += 1
                    print(f"  ✅ {test_name}")
                else:
                    print(f"  ❌ {test_name}")

        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0

        print(f"\n🎯 OVERALL RESULTS:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Passed: {passed_tests}")
        print(f"   Failed: {total_tests - passed_tests}")
        print(f"   Success Rate: {success_rate:.1f}%")

        if success_rate >= 80:
            print("\n🎉 BOOTCAMP READY! Most functionality is working.")
        elif success_rate >= 60:
            print("\n⚠️ MOSTLY READY: Some issues need attention.")
        else:
            print("\n🚨 NEEDS WORK: Significant issues detected.")

        return success_rate


def main():
    parser = argparse.ArgumentParser(
        description="Validate bootcamp notebook dependencies"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Add ChemML to path if needed
    chemml_path = Path(__file__).parent / "src"
    if chemml_path.exists():
        sys.path.insert(0, str(chemml_path))

    validator = BootcampValidator(verbose=args.verbose)
    success_rate = validator.run_validation()

    # Exit with appropriate code
    if success_rate is not None:
        sys.exit(0 if success_rate >= 80 else 1)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
