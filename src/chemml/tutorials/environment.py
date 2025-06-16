"""
Environment Management for ChemML Tutorials
==========================================

This module provides robust environment setup, dependency checking, and fallback
mechanisms for ChemML tutorials to ensure smooth learning experiences across
different computational environments.

Key Features:
- Dependency validation and installation
- Fallback mechanisms for missing libraries
- Environment configuration for tutorials
- Resource availability checking
- Error handling and user guidance
"""

import importlib
import logging
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Configure logging for environment management
logger = logging.getLogger(__name__)


class EnvironmentManager:
    """
    Comprehensive environment management for ChemML tutorials.

    This class handles dependency checking, environment setup, and provides
    fallback mechanisms to ensure tutorials work across different environments.
    """

    def __init__(self, tutorial_name: str = "general"):
        self.tutorial_name = tutorial_name
        self.dependencies = {}
        self.fallbacks = {}
        self.environment_status = {}

        # Core dependencies required for all tutorials
        self.core_dependencies = {
            "numpy": {"required": True, "min_version": "1.19.0"},
            "pandas": {"required": True, "min_version": "1.3.0"},
            "matplotlib": {"required": True, "min_version": "3.3.0"},
            "rdkit": {
                "required": True,
                "min_version": "2021.03.0",
                "fallback": "rdkit_fallback",
            },
            "sklearn": {"required": True, "min_version": "1.0.0"},
        }

        # Optional dependencies with fallbacks
        self.optional_dependencies = {
            "deepchem": {
                "required": False,
                "min_version": "2.6.0",
                "fallback": "deepchem_fallback",
            },
            "torch": {
                "required": False,
                "min_version": "1.9.0",
                "fallback": "torch_fallback",
            },
            "tensorflow": {
                "required": False,
                "min_version": "2.6.0",
                "fallback": "tf_fallback",
            },
            "qiskit": {
                "required": False,
                "min_version": "0.40.0",
                "fallback": "quantum_fallback",
            },
            "psi4": {
                "required": False,
                "min_version": "1.6.0",
                "fallback": "quantum_fallback",
            },
            "openmm": {
                "required": False,
                "min_version": "7.6.0",
                "fallback": "md_fallback",
            },
            "mdtraj": {
                "required": False,
                "min_version": "1.9.0",
                "fallback": "md_fallback",
            },
            "py3Dmol": {
                "required": False,
                "min_version": "1.8.0",
                "fallback": "visualization_fallback",
            },
            "ipywidgets": {
                "required": False,
                "min_version": "7.6.0",
                "fallback": "widget_fallback",
            },
        }

    def check_environment(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Comprehensive environment check for tutorial requirements.

        Args:
            verbose (bool): Whether to print detailed status information

        Returns:
            Dict[str, Any]: Environment status report
        """
        status = {
            "python_version": sys.version,
            "python_executable": sys.executable,
            "dependencies": {},
            "missing_required": [],
            "missing_optional": [],
            "recommendations": [],
            "overall_status": "unknown",
        }

        # Check Python version
        python_version = sys.version_info
        if python_version < (3, 8):
            status["recommendations"].append(
                "Python 3.8+ recommended for best compatibility"
            )

        # Check core dependencies
        for dep, config in self.core_dependencies.items():
            dep_status = self._check_dependency(dep, config)
            status["dependencies"][dep] = dep_status

            if not dep_status["available"] and config["required"]:
                status["missing_required"].append(dep)

        # Check optional dependencies
        for dep, config in self.optional_dependencies.items():
            dep_status = self._check_dependency(dep, config)
            status["dependencies"][dep] = dep_status

            if not dep_status["available"]:
                status["missing_optional"].append(dep)

        # Determine overall status
        if not status["missing_required"]:
            if not status["missing_optional"]:
                status["overall_status"] = "excellent"
            elif len(status["missing_optional"]) <= 3:
                status["overall_status"] = "good"
            else:
                status["overall_status"] = "adequate"
        else:
            status["overall_status"] = "insufficient"

        # Generate recommendations
        self._generate_recommendations(status)

        if verbose:
            self._print_status_report(status)

        self.environment_status = status
        return status

    def setup_tutorial_environment(
        self, tutorial_type: str = "basic", auto_install: bool = False
    ) -> bool:
        """
        Setup environment for specific tutorial type.

        Args:
            tutorial_type (str): Type of tutorial ('basic', 'advanced', 'quantum', 'ml')
            auto_install (bool): Whether to automatically install missing dependencies

        Returns:
            bool: True if environment is ready, False otherwise
        """
        required_deps = self._get_tutorial_dependencies(tutorial_type)

        # Check current status
        status = self.check_environment(verbose=False)

        missing_deps = []
        for dep in required_deps:
            if (
                dep in status["dependencies"]
                and not status["dependencies"][dep]["available"]
            ):
                missing_deps.append(dep)

        if not missing_deps:
            logger.info(f"Environment ready for {tutorial_type} tutorials")
            return True

        if auto_install:
            return self._install_dependencies(missing_deps)
        else:
            logger.warning(f"Missing dependencies for {tutorial_type}: {missing_deps}")
            self._suggest_installation(missing_deps)
            return False

    def setup_fallbacks(self) -> Dict[str, Any]:
        """
        Setup fallback mechanisms for missing dependencies.

        Returns:
            Dict[str, Any]: Configured fallbacks
        """
        fallbacks = {}

        status = self.environment_status or self.check_environment(verbose=False)

        for dep, dep_status in status["dependencies"].items():
            if not dep_status["available"]:
                fallback_name = self.optional_dependencies.get(dep, {}).get("fallback")
                if fallback_name:
                    fallbacks[dep] = self._create_fallback(fallback_name, dep)

        self.fallbacks = fallbacks
        return fallbacks

    def validate_tutorial_requirements(
        self, requirements: List[str]
    ) -> Tuple[bool, List[str]]:
        """
        Validate that specific tutorial requirements are met.

        Args:
            requirements (List[str]): List of required capabilities/dependencies

        Returns:
            Tuple[bool, List[str]]: (all_met, missing_requirements)
        """
        missing = []

        for req in requirements:
            if req in self.core_dependencies or req in self.optional_dependencies:
                # Check dependency
                if req not in self.environment_status.get("dependencies", {}):
                    self.check_environment(verbose=False)

                if (
                    not self.environment_status["dependencies"]
                    .get(req, {})
                    .get("available", False)
                ):
                    missing.append(req)
            else:
                # Check capability
                if not self._check_capability(req):
                    missing.append(req)

        return len(missing) == 0, missing

    def _check_dependency(self, name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Check if a dependency is available and meets version requirements."""
        result = {
            "available": False,
            "version": None,
            "meets_requirements": False,
            "error": None,
        }

        try:
            # Handle special cases
            if name == "sklearn":
                module = importlib.import_module("sklearn")
            else:
                module = importlib.import_module(name)

            result["available"] = True

            # Get version
            if hasattr(module, "__version__"):
                result["version"] = module.__version__

                # Check version requirements
                if "min_version" in config:
                    result["meets_requirements"] = self._version_check(
                        result["version"], config["min_version"]
                    )
                else:
                    result["meets_requirements"] = True
            else:
                result["meets_requirements"] = True  # Assume OK if no version info

        except ImportError as e:
            result["error"] = str(e)
        except Exception as e:
            result["error"] = f"Unexpected error: {str(e)}"

        return result

    def _version_check(self, current: str, minimum: str) -> bool:
        """Check if current version meets minimum requirement."""
        try:
            from packaging import version

            return version.parse(current) >= version.parse(minimum)
        except ImportError:
            # Fallback version comparison
            current_parts = [int(x) for x in current.split(".")]
            minimum_parts = [int(x) for x in minimum.split(".")]

            # Pad shorter version with zeros
            max_len = max(len(current_parts), len(minimum_parts))
            current_parts.extend([0] * (max_len - len(current_parts)))
            minimum_parts.extend([0] * (max_len - len(minimum_parts)))

            return current_parts >= minimum_parts

    def _get_tutorial_dependencies(self, tutorial_type: str) -> List[str]:
        """Get dependencies required for specific tutorial type."""
        base_deps = ["numpy", "pandas", "matplotlib", "rdkit", "sklearn"]

        if tutorial_type == "basic":
            return base_deps
        elif tutorial_type == "advanced":
            return base_deps + ["deepchem", "torch"]
        elif tutorial_type == "quantum":
            return base_deps + ["qiskit", "psi4"]
        elif tutorial_type == "ml":
            return base_deps + ["torch", "tensorflow", "deepchem"]
        elif tutorial_type == "md":
            return base_deps + ["openmm", "mdtraj"]
        else:
            return base_deps

    def _install_dependencies(self, dependencies: List[str]) -> bool:
        """Attempt to install missing dependencies."""
        logger.info(f"Attempting to install: {dependencies}")

        success_count = 0
        for dep in dependencies:
            try:
                if dep == "rdkit":
                    subprocess.check_call(
                        [sys.executable, "-m", "pip", "install", "rdkit-pypi"]
                    )
                elif dep == "sklearn":
                    subprocess.check_call(
                        [sys.executable, "-m", "pip", "install", "scikit-learn"]
                    )
                else:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", dep])

                success_count += 1
                logger.info(f"Successfully installed {dep}")

            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to install {dep}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error installing {dep}: {e}")

        return success_count == len(dependencies)

    def _create_fallback(self, fallback_type: str, original_dep: str) -> Any:
        """Create fallback implementation for missing dependency."""
        if fallback_type == "rdkit_fallback":
            return self._create_rdkit_fallback()
        elif fallback_type == "deepchem_fallback":
            return self._create_deepchem_fallback()
        elif fallback_type == "torch_fallback":
            return self._create_torch_fallback()
        elif fallback_type == "quantum_fallback":
            return self._create_quantum_fallback()
        elif fallback_type == "visualization_fallback":
            return self._create_visualization_fallback()
        else:
            return None

    def _create_rdkit_fallback(self):
        """Create fallback for RDKit functionality."""

        class RDKitFallback:
            def __init__(self):
                self.available = False

            def MolFromSmiles(self, smiles):
                warnings.warn(
                    "RDKit not available. Install with: pip install rdkit-pypi"
                )
                return None

            def MolToSmiles(self, mol):
                warnings.warn(
                    "RDKit not available. Install with: pip install rdkit-pypi"
                )
                return "unavailable"

        return RDKitFallback()

    def _create_deepchem_fallback(self):
        """Create fallback for DeepChem functionality."""

        def fallback_function(*args, **kwargs):
            warnings.warn("DeepChem not available. Install with: pip install deepchem")
            return None

        return fallback_function

    def _create_torch_fallback(self):
        """Create fallback for PyTorch functionality."""

        def fallback_function(*args, **kwargs):
            warnings.warn("PyTorch not available. Install with: pip install torch")
            return None

        return fallback_function

    def _create_quantum_fallback(self):
        """Create fallback for quantum computing libraries."""

        def fallback_function(*args, **kwargs):
            warnings.warn("Quantum libraries not available. Install Qiskit or Psi4")
            return None

        return fallback_function

    def _create_visualization_fallback(self):
        """Create fallback for advanced visualization."""

        def fallback_function(*args, **kwargs):
            warnings.warn("Advanced visualization not available. Install py3Dmol")
            return None

        return fallback_function

    def _check_capability(self, capability: str) -> bool:
        """Check if a specific capability is available."""
        capabilities = {
            "gpu": self._check_gpu_availability(),
            "jupyter": self._check_jupyter_availability(),
            "visualization": self._check_visualization_capability(),
            "quantum": self._check_quantum_capability(),
            "molecular_dynamics": self._check_md_capability(),
        }

        return capabilities.get(capability, False)

    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available for computation."""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            try:
                import tensorflow as tf

                return len(tf.config.list_physical_devices("GPU")) > 0
            except ImportError:
                return False

    def _check_jupyter_availability(self) -> bool:
        """Check if running in Jupyter environment."""
        try:
            from IPython import get_ipython

            return get_ipython() is not None
        except ImportError:
            return False

    def _check_visualization_capability(self) -> bool:
        """Check advanced visualization capabilities."""
        try:
            import py3Dmol

            return True
        except ImportError:
            return False

    def _check_quantum_capability(self) -> bool:
        """Check quantum computing capabilities."""
        try:
            import qiskit

            return True
        except ImportError:
            try:
                import psi4

                return True
            except ImportError:
                return False

    def _check_md_capability(self) -> bool:
        """Check molecular dynamics capabilities."""
        try:
            import openmm

            return True
        except ImportError:
            try:
                import mdtraj

                return True
            except ImportError:
                return False

    def _suggest_installation(self, missing_deps: List[str]):
        """Suggest installation commands for missing dependencies."""
        print("\n🔧 Missing Dependencies Installation Guide:")
        print("=" * 50)

        for dep in missing_deps:
            if dep == "rdkit":
                print(f"📦 {dep}: pip install rdkit-pypi")
            elif dep == "sklearn":
                print(f"📦 {dep}: pip install scikit-learn")
            elif dep == "deepchem":
                print(f"📦 {dep}: pip install deepchem")
            elif dep == "qiskit":
                print(f"📦 {dep}: pip install qiskit")
            elif dep == "psi4":
                print(f"📦 {dep}: conda install -c conda-forge psi4 (recommended)")
            else:
                print(f"📦 {dep}: pip install {dep}")

        print("\n💡 Install all at once:")
        install_cmd = "pip install " + " ".join(
            (
                "rdkit-pypi"
                if dep == "rdkit"
                else "scikit-learn" if dep == "sklearn" else dep
            )
            for dep in missing_deps
            if dep != "psi4"
        )
        print(f"   {install_cmd}")

        if "psi4" in missing_deps:
            print("   conda install -c conda-forge psi4")

    def _generate_recommendations(self, status: Dict[str, Any]):
        """Generate setup recommendations based on environment status."""
        if status["overall_status"] == "insufficient":
            status["recommendations"].append(
                "Install required dependencies before proceeding with tutorials"
            )

        if len(status["missing_optional"]) > 5:
            status["recommendations"].append(
                "Consider using conda environment for easier dependency management"
            )

        if (
            "torch" not in status["dependencies"]
            or not status["dependencies"]["torch"]["available"]
        ):
            status["recommendations"].append(
                "Install PyTorch for machine learning tutorials"
            )

        if (
            "deepchem" not in status["dependencies"]
            or not status["dependencies"]["deepchem"]["available"]
        ):
            status["recommendations"].append(
                "Install DeepChem for advanced chemical informatics"
            )

    def _print_status_report(self, status: Dict[str, Any]):
        """Print formatted environment status report."""
        print("\n🔍 ChemML Tutorial Environment Check")
        print("=" * 50)

        # Overall status
        status_emoji = {
            "excellent": "🟢",
            "good": "🟡",
            "adequate": "🟠",
            "insufficient": "🔴",
            "unknown": "⚪",
        }

        print(
            f"\n📊 Overall Status: {status_emoji[status['overall_status']]} {status['overall_status'].upper()}"
        )

        # Python version
        print(f"🐍 Python: {status['python_version'].split()[0]}")

        # Dependencies
        print("\n📦 Dependencies:")
        for dep, dep_status in status["dependencies"].items():
            if dep_status["available"]:
                if dep_status["meets_requirements"]:
                    emoji = "✅"
                else:
                    emoji = "⚠️"
                version_info = (
                    f" (v{dep_status['version']})" if dep_status["version"] else ""
                )
                print(f"   {emoji} {dep}{version_info}")
            else:
                print(f"   ❌ {dep} - {dep_status['error'][:50]}...")

        # Recommendations
        if status["recommendations"]:
            print("\n💡 Recommendations:")
            for rec in status["recommendations"]:
                print(f"   • {rec}")


def check_dependencies(
    dependencies: List[str] = None, verbose: bool = True
) -> Dict[str, bool]:
    """
    Check if specific dependencies are available.

    Args:
        dependencies (List[str], optional): List of dependencies to check
        verbose (bool): Whether to print status information

    Returns:
        Dict[str, bool]: Dependency availability status
    """
    manager = EnvironmentManager()

    if dependencies is None:
        dependencies = list(manager.core_dependencies.keys())

    status = manager.check_environment(verbose=False)

    result = {}
    for dep in dependencies:
        result[dep] = status["dependencies"].get(dep, {}).get("available", False)

    if verbose:
        print("\n📋 Dependency Check Results:")
        for dep, available in result.items():
            emoji = "✅" if available else "❌"
            print(f"   {emoji} {dep}")

    return result


def setup_fallbacks(tutorial_type: str = "basic") -> Dict[str, Any]:
    """
    Setup fallback mechanisms for missing dependencies.

    Args:
        tutorial_type (str): Type of tutorial requiring fallbacks

    Returns:
        Dict[str, Any]: Configured fallback implementations
    """
    manager = EnvironmentManager(tutorial_type)
    return manager.setup_fallbacks()


def validate_environment(requirements: List[str]) -> bool:
    """
    Validate that the environment meets specific requirements.

    Args:
        requirements (List[str]): List of required capabilities or dependencies

    Returns:
        bool: True if all requirements are met
    """
    manager = EnvironmentManager()
    all_met, missing = manager.validate_tutorial_requirements(requirements)

    if not all_met:
        print(f"⚠️  Missing requirements: {missing}")
        print("Run environment setup to resolve these issues.")

    return all_met
