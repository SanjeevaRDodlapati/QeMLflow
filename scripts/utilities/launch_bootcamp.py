#!/usr/bin/env python3
"""
QeMLflow Bootcamp Launcher
=======================

Simple launcher script that validates the environment and guides users
to the appropriate notebooks based on what's working.

Usage:
    python launch_bootcamp.py
    python launch_bootcamp.py --fix-quantum
    python launch_bootcamp.py --quick-test

import argparse
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")


class BootcampLauncher:
    def __init__(self):
        self.base_path = Path(__file__).parent
        self.notebooks_path = self.base_path / "notebooks"
        self.environment_status = {}

    def check_core_environment(self):
        """Check if core libraries are available"""
        print("🔍 Checking core environment...")

        core_imports = {
            "numpy": "import numpy",
            "pandas": "import pandas",
            "matplotlib": "import matplotlib.pyplot",
            "sklearn": "import sklearn",
            "rdkit": "import rdkit",
            "deepchem": "import deepchem",
        }

        all_good = True
        for name, import_cmd in core_imports.items():
            try:
                exec(import_cmd)
                print(f"   ✅ {name}")
                self.environment_status[name] = True
            except ImportError:
                print(f"   ❌ {name} - MISSING!")
                self.environment_status[name] = False
                all_good = False

        return all_good

    def check_qemlflow_modules(self):
        """Check QeMLflow modules"""
        print("\n🔍 Checking QeMLflow modules...")

        try:
            # Add src to path
            src_path = self.base_path / "src"
            if src_path.exists():
                sys.path.insert(0, str(src_path))

            # Test imports
            from qemlflow.core.data import legacy_molecular_cleaning
            from qemlflow.core.featurizers import MorganFingerprint

            print("   ✅ QeMLflow hybrid architecture")
            self.environment_status["qemlflow"] = True
            return True
        except ImportError as e:
            print(f"   ❌ QeMLflow modules - {e}")
            self.environment_status["qemlflow"] = False
            return False

    def check_quantum_environment(self):
        """Check quantum computing setup"""
        print("\n🔍 Checking quantum environment...")

        quantum_status = {}

        try:
            import qiskit

            print(f"   ✅ Qiskit {qiskit.__version__}")
            quantum_status["qiskit"] = True
        except ImportError:
            print("   ❌ Qiskit - MISSING!")
            quantum_status["qiskit"] = False

        try:
            import qiskit_aer

            print("   ✅ Qiskit Aer")
            quantum_status["aer"] = True
        except ImportError:
            print("   ❌ Qiskit Aer - MISSING!")
            quantum_status["aer"] = False

        try:
            import qiskit_algorithms

            print("   ✅ Qiskit Algorithms")
            quantum_status["algorithms"] = True
        except ImportError:
            print("   ⚠️ Qiskit Algorithms - using fallbacks")
            quantum_status["algorithms"] = False

        try:
            import qiskit_nature

            print("   ✅ Qiskit Nature")
            quantum_status["nature"] = True
        except ImportError:
            print("   ⚠️ Qiskit Nature - compatibility issues")
            quantum_status["nature"] = False

        self.environment_status["quantum"] = quantum_status
        return any(quantum_status.values())

    def recommend_notebooks(self):
        """Recommend which notebooks to use based on environment"""
        print("\n" + "=" * 60)
        print("📚 RECOMMENDED LEARNING PATH")
        print("=" * 60)

        # Always available tutorials
        print("\n✅ CORE TUTORIALS (Always Available):")
        if self.environment_status.get("qemlflow", False):
            print("   📖 notebooks/tutorials/01_basic_cheminformatics.ipynb")
            print("      → Enhanced with hybrid architecture demo")
            print("   📖 notebooks/tutorials/03_deepchem_drug_discovery.ipynb")
            print("      → Complete drug discovery workflow")

        # Bootcamp days assessment
        print("\n📅 BOOTCAMP PROGRESSION:")

        if self.environment_status.get("rdkit", False) and self.environment_status.get(
            "sklearn", False
        ):
            print("   ✅ Day 1: ML & Cheminformatics")
            print("   ✅ Day 2: Data Processing")
            print("   ✅ Day 3: Advanced ML")
            print("   ✅ Day 4: Molecular Design")
            print("   ✅ Day 5: Integration")

        quantum = self.environment_status.get("quantum", {})
        if quantum.get("qiskit", False):
            if quantum.get("algorithms", False) and quantum.get("nature", False):
                print("   ✅ Day 6: Quantum Foundations")
                print("   ✅ Day 7: Advanced Quantum")
            else:
                print("   ⚠️ Day 6: Quantum Foundations (with compatibility fixes)")
                print("   ⚠️ Day 7: Advanced Quantum (simplified version)")
        else:
            print("   ❌ Day 6-7: Quantum sections unavailable")

        # Quick start recommendations
        print("\n🚀 QUICK START OPTIONS:")
        print("   1. Complete bootcamp simulation:")
        print("      python test_complete_workflow.py")
        print("   2. Interactive tutorials:")
        print("      jupyter notebook notebooks/tutorials/")
        print("   3. Validate specific day:")
        print("      python validate_bootcamp_notebooks.py")

    def fix_quantum_environment(self):
        """Attempt to fix quantum environment issues"""
        print("\n🔧 Attempting to fix quantum environment...")

        try:
            # Try to install compatible versions
            print("Installing compatible Qiskit versions...")
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "qiskit>=2.0.0",
                    "qiskit-aer>=0.17.0",
                    "qiskit-algorithms>=0.3.0",
                ],
                check=False,
            )

            print("✅ Quantum packages installation attempted")
            print("   Please restart and run the launcher again")

        except Exception as e:
            print(f"❌ Could not fix quantum environment: {e}")
            print("   Manual intervention may be required")

    def launch(self, fix_quantum=False, quick_test=False):
        """Main launcher logic"""
        print("🎯 QeMLflow Bootcamp Environment Launcher")
        print("=" * 60)

        if quick_test:
            print("Running quick environment test...")
            subprocess.run([sys.executable, "validate_bootcamp_notebooks.py"])
            return

        if fix_quantum:
            self.fix_quantum_environment()
            return

        # Check environment components
        core_ok = self.check_core_environment()
        qemlflow_ok = self.check_qemlflow_modules()
        quantum_ok = self.check_quantum_environment()

        # Summary
        print("\n🎯 ENVIRONMENT STATUS:")
        print(f"   Core Libraries: {'✅ Ready' if core_ok else '❌ Issues'}")
        print(f"   QeMLflow Modules: {'✅ Ready' if qemlflow_ok else '❌ Issues'}")
        print(f"   Quantum Setup:  {'✅ Ready' if quantum_ok else '⚠️ Partial'}")

        overall_score = sum([core_ok, qemlflow_ok, quantum_ok]) / 3 * 100
        print(f"   Overall: {overall_score:.0f}% Ready")

        # Provide recommendations
        self.recommend_notebooks()

        if not core_ok:
            print("\n❌ CRITICAL: Core libraries missing!")
            print("   Run: pip install -r requirements.txt")
        elif overall_score >= 67:
            print("\n🎉 Environment ready for bootcamp!")
            print("   Start with: notebooks/tutorials/01_basic_cheminformatics.ipynb")
        else:
            print("\n⚠️ Environment partially ready")
            print("   Consider running: python launch_bootcamp.py --fix-quantum")


def main():
    parser = argparse.ArgumentParser(description="Launch QeMLflow Bootcamp")
    parser.add_argument(
        "--fix-quantum", action="store_true", help="Attempt to fix quantum environment"
    )
    parser.add_argument(
        "--quick-test", action="store_true", help="Run quick validation test"
    )

    args = parser.parse_args()

    launcher = BootcampLauncher()
    launcher.launch(fix_quantum=args.fix_quantum, quick_test=args.quick_test)


if __name__ == "__main__":
    main()
