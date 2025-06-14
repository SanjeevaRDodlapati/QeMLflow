#!/usr/bin/env python
"""
Complete Dependency Check for Day 04 Quantum Chemistry Notebook

This script performs a comprehensive check of all required libraries for the
Day 04 Quantum Chemistry notebook, verifies their functionality, and provides
detailed recommendations for missing components.
"""

import importlib
import os
import platform
import subprocess
import sys
from importlib.util import find_spec

# ANSI colors for better terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
BOLD = "\033[1m"
ENDC = "\033[0m"


def print_header(title):
    """Print a formatted header."""
    width = 60
    print(f"\n{BLUE}{BOLD}{'=' * width}{ENDC}")
    print(f"{BLUE}{BOLD}{title.center(width)}{ENDC}")
    print(f"{BLUE}{BOLD}{'=' * width}{ENDC}")


def check_library(
    name, description, required=True, alternatives=None, fallback_exists=False
):
    """Check if a library is installed and print its status."""
    status = "REQUIRED" if required else "OPTIONAL"

    try:
        # Handling special case for RDKit
        if name == "rdkit":
            if find_spec(name):
                import rdkit
                from rdkit import rdBase

                version = rdBase.rdkitVersion
            else:
                raise ImportError("Not found")
        else:
            if find_spec(name):
                # Try to import and get version
                module = importlib.import_module(name)
                if hasattr(module, "__version__"):
                    version = module.__version__
                elif hasattr(module, "version"):
                    version = module.version
                else:
                    version = "Unknown version"
            else:
                raise ImportError("Not found")

        print(f"{GREEN}✅ {name} {version} - {description}{ENDC}")
        return True

    except ImportError:
        if required:
            message = f"{RED}❌ {name} NOT FOUND - {description} [{status}]"
            if alternatives:
                message += f" - Alternative: {alternatives}"
            if fallback_exists:
                message += f" - Fallback available"
            print(message + ENDC)
        else:
            print(f"{YELLOW}⚠️ {name} missing - {description} [{status}]{ENDC}")
        return False
    except Exception as e:
        print(f"{RED}❌ Error checking {name}: {str(e)}{ENDC}")
        return False


def test_numpy_compatibility():
    """Test NumPy compatibility with other libraries."""
    print_header("NUMPY COMPATIBILITY TEST")

    try:
        import numpy as np

        print(f"NumPy version: {np.__version__}")

        # Create and manipulate a simple array to verify functionality
        test_array = np.array([1, 2, 3])
        print(f"Array creation test: {test_array}")

        # Check if NumPy version is 2.x (potential compatibility issues)
        if np.__version__.startswith("2."):
            print(
                f"{YELLOW}⚠️ NumPy 2.x detected - may have compatibility issues with some libraries{ENDC}"
            )

            # Test DeepChem specifically
            try:
                import deepchem

                print("Testing DeepChem with NumPy 2.x...")

                # Simple test: create a featurizer
                featurizer = deepchem.feat.CircularFingerprint(size=1024)
                print(f"{GREEN}✅ DeepChem works with current NumPy version{ENDC}")
            except Exception as e:
                print(f"{RED}❌ DeepChem has issues with NumPy 2.x: {str(e)}{ENDC}")
                print(f"{YELLOW}Recommended fix: pip install numpy==1.24.0{ENDC}")
        else:
            print(
                f"{GREEN}✅ NumPy version should be compatible with all libraries{ENDC}"
            )

    except ImportError:
        print(f"{RED}❌ NumPy not installed!{ENDC}")
    except Exception as e:
        print(f"{RED}❌ Error testing NumPy: {str(e)}{ENDC}")


def check_psi4_fallbacks():
    """Check if fallback mechanisms for Psi4 will work properly."""
    print_header("PSI4 FALLBACK MECHANISM CHECK")

    psi4_available = find_spec("psi4") is not None

    if psi4_available:
        print(f"{GREEN}✅ Psi4 is installed - no fallbacks needed{ENDC}")
        return True

    print(f"{YELLOW}⚠️ Psi4 is not installed - checking fallback mechanisms{ENDC}")

    fallbacks_ok = True

    # Check for PySCF (alternative quantum chemistry package)
    pyscf_available = find_spec("pyscf") is not None
    if pyscf_available:
        print(
            f"{GREEN}✅ PySCF is available as an alternative quantum chemistry package{ENDC}"
        )
    else:
        print(
            f"{RED}❌ PySCF not available - quantum chemistry features will be limited{ENDC}"
        )
        fallbacks_ok = False

    # Check for NumPy/SciPy (needed for mock implementations)
    numpy_available = find_spec("numpy") is not None
    scipy_available = find_spec("scipy") is not None

    if numpy_available and scipy_available:
        print(f"{GREEN}✅ NumPy and SciPy available for mock implementations{ENDC}")
    else:
        missing = []
        if not numpy_available:
            missing.append("NumPy")
        if not scipy_available:
            missing.append("SciPy")
        print(f"{RED}❌ Missing {', '.join(missing)} for mock calculations{ENDC}")
        fallbacks_ok = False

    # Overall status
    if fallbacks_ok:
        print(f"\n{GREEN}✅ Fallback mechanisms should work properly{ENDC}")
    else:
        print(f"\n{RED}❌ Some fallback mechanisms may not work properly{ENDC}")

    return fallbacks_ok


def test_deepchem_functionality():
    """Test basic DeepChem functionality."""
    print_header("DEEPCHEM FUNCTIONALITY TEST")

    try:
        import deepchem

        print(f"DeepChem version: {deepchem.__version__}")

        # Try to create a simple featurizer to test functionality
        try:
            print("Testing feature creation...")
            featurizer = deepchem.feat.CircularFingerprint(size=1024)
            print(f"{GREEN}✅ DeepChem feature creation successful{ENDC}")
        except Exception as e:
            print(f"{RED}❌ DeepChem feature creation failed: {str(e)}{ENDC}")
            return False

    except ImportError:
        print(f"{RED}❌ DeepChem not installed{ENDC}")
        return False
    except Exception as e:
        print(f"{RED}❌ Error testing DeepChem: {str(e)}{ENDC}")
        return False

    return True


def analyze_notebook_impact():
    """Analyze the impact of missing libraries on notebook functionality."""
    print_header("NOTEBOOK IMPACT ANALYSIS")

    # Define notebook sections and their dependencies
    notebook_sections = [
        {
            "name": "Setup and Introduction",
            "dependencies": ["numpy", "pandas", "matplotlib"],
            "criticality": "high",
            "status": None,
        },
        {
            "name": "Molecule Representation",
            "dependencies": ["rdkit", "ase"],
            "criticality": "high",
            "status": None,
        },
        {
            "name": "Basic Quantum Chemistry",
            "dependencies": ["numpy", "scipy", "pyscf"],
            "criticality": "high",
            "status": None,
        },
        {
            "name": "Advanced QM with Psi4",
            "dependencies": ["psi4"],
            "alternative_dependencies": ["numpy", "scipy", "pyscf"],
            "criticality": "medium",
            "status": None,
            "has_fallback": True,
        },
        {
            "name": "ML with QM Data",
            "dependencies": ["numpy", "scikit-learn", "torch"],
            "criticality": "medium",
            "status": None,
        },
        {
            "name": "Interactive Visualization",
            "dependencies": ["matplotlib", "plotly"],
            "criticality": "low",
            "status": None,
        },
    ]

    # Check status of each section
    for section in notebook_sections:
        # Check primary dependencies
        all_deps_available = all(
            find_spec(dep) is not None for dep in section["dependencies"]
        )

        # If section has fallbacks and primary dependencies aren't available, check alternatives
        if (
            not all_deps_available
            and section.get("has_fallback")
            and section.get("alternative_dependencies")
        ):
            alt_deps_available = all(
                find_spec(dep) is not None
                for dep in section["alternative_dependencies"]
            )
            if alt_deps_available:
                section["status"] = "fallback"
            else:
                section["status"] = "unavailable"
        else:
            section["status"] = "available" if all_deps_available else "unavailable"

    # Calculate overall functionality percentage
    total_sections = len(notebook_sections)
    available_sections = sum(
        1 for s in notebook_sections if s["status"] in ["available", "fallback"]
    )
    functionality_percent = (available_sections / total_sections) * 100

    # Print section status
    print(f"\n{'Section':<30} {'Status':<15} {'Impact'}")
    print("-" * 60)

    for section in notebook_sections:
        status_str = ""
        if section["status"] == "available":
            status_str = f"{GREEN}✅ Available{ENDC}"
        elif section["status"] == "fallback":
            status_str = f"{YELLOW}⚠️ Fallback{ENDC}"
        else:
            status_str = f"{RED}❌ Unavailable{ENDC}"

        impact = section["criticality"].upper()
        impact_color = RED if impact == "HIGH" else YELLOW if impact == "MEDIUM" else ""
        colored_impact = f"{impact_color}{impact}{ENDC}" if impact_color else impact

        print(f"{section['name']:<30} {status_str:<15} {colored_impact}")

    # Print overall functionality
    print(f"\nOverall notebook functionality: {functionality_percent:.1f}%")

    if functionality_percent < 70:
        print(
            f"\n{RED}⚠️ Critical features missing. Consider using Docker for a complete environment.{ENDC}"
        )
    elif functionality_percent < 90:
        print(f"\n{YELLOW}⚠️ Some features will use fallbacks or be limited.{ENDC}")
    else:
        print(f"\n{GREEN}✅ Most notebook features should work correctly.{ENDC}")


def main():
    """Run all checks and provide a comprehensive report."""
    print_header("DAY 04 QUANTUM CHEMISTRY NOTEBOOK DEPENDENCY CHECK")

    print(f"Python version: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")

    print_header("CORE SCIENTIFIC LIBRARIES")
    numpy_ok = check_library("numpy", "Scientific computing foundation", required=True)
    pandas_ok = check_library("pandas", "Data manipulation", required=True)
    scipy_ok = check_library("scipy", "Scientific algorithms", required=True)
    matplotlib_ok = check_library("matplotlib", "Visualization", required=True)

    print_header("CHEMISTRY & QUANTUM LIBRARIES")
    rdkit_ok = check_library("rdkit", "Chemical informatics toolkit", required=True)
    ase_ok = check_library("ase", "Atomic Simulation Environment", required=True)
    pyscf_ok = check_library("pyscf", "Python-based quantum chemistry", required=True)
    psi4_ok = check_library(
        "psi4",
        "Quantum chemistry package",
        required=False,
        alternatives="PySCF",
        fallback_exists=True,
    )

    print_header("MACHINE LEARNING LIBRARIES")
    sklearn_ok = check_library("sklearn", "Machine learning", required=True)
    torch_ok = check_library("torch", "Deep learning", required=True)
    deepchem_ok = check_library(
        "deepchem", "Molecular machine learning", required=False
    )

    # Run special tests
    if numpy_ok:
        test_numpy_compatibility()

    if not psi4_ok:
        check_psi4_fallbacks()

    if deepchem_ok:
        test_deepchem_functionality()

    # Analyze notebook impact
    analyze_notebook_impact()

    print_header("INSTALLATION RECOMMENDATIONS")

    # Recommend Docker if multiple key components are missing
    critical_missing = not all([numpy_ok, pandas_ok, scipy_ok, rdkit_ok, pyscf_ok])
    if critical_missing:
        print(
            f"{RED}⚠️ Critical libraries are missing. Docker is strongly recommended.{ENDC}"
        )
        print(
            """
# Build and run Docker container (recommended solution)
docker build -t chemml .
docker run -p 8888:8888 -v $(pwd):/app chemml
        """
        )

    # Specific recommendations for missing components
    if not psi4_ok:
        print(f"\n{YELLOW}📦 Psi4 Installation Options:{ENDC}")
        print(
            """
# Option 1: Miniforge (recommended)
curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh
bash Miniforge3-MacOSX-x86_64.sh -b -p $HOME/miniforge3
echo 'export PATH="$HOME/miniforge3/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
mamba create -n psi4env python=3.8 -y
mamba activate psi4env
mamba install -c conda-forge psi4=1.9.1 numpy pandas matplotlib rdkit ase pyscf -y

# Option 2: Docker Psi4-specific container
docker pull psi4/psi4:latest
docker run -it -v "$(pwd)":/work -w /work psi4/psi4:latest
        """
        )

    if numpy_ok and deepchem_ok:
        import numpy

        if numpy.__version__.startswith("2."):
            print(f"\n{YELLOW}📦 NumPy/DeepChem Compatibility Fix:{ENDC}")
            print("pip install numpy==1.24.0 --force-reinstall")

    # Note about fallbacks
    if not psi4_ok and pyscf_ok:
        print(
            f"\n{GREEN}ℹ️ Note:{ENDC} The notebook has fallback mechanisms for missing libraries."
        )
        print("It will automatically use PySCF when Psi4 is unavailable.")


if __name__ == "__main__":
    main()
