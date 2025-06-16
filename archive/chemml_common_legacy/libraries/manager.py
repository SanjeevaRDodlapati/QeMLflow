"""
Unified Library Management for ChemML Scripts
============================================

Centralized library management that eliminates code duplication and provides
consistent handling of optional dependencies across all ChemML scripts.
"""

import importlib
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class LibraryInfo:
    """Information about a library dependency."""

    name: str
    import_name: str
    install_name: str = ""
    required: bool = False
    fallback_available: bool = False
    description: str = ""

    def __post_init__(self):
        """Set defaults after initialization."""
        if not self.install_name:
            self.install_name = self.name


class LibraryManager:
    """
    Unified library management for ChemML scripts.

    Provides consistent handling of library imports, installations,
    and fallback mechanisms across all scripts.
    """

    # Standard library definitions
    LIBRARY_REGISTRY = {
        "rdkit": LibraryInfo(
            name="rdkit",
            import_name="rdkit.Chem",
            install_name="rdkit",
            required=False,
            fallback_available=True,
            description="Chemical informatics toolkit",
        ),
        "torch": LibraryInfo(
            name="torch",
            import_name="torch",
            install_name="torch",
            required=False,
            fallback_available=True,
            description="PyTorch deep learning framework",
        ),
        "torch_geometric": LibraryInfo(
            name="torch_geometric",
            import_name="torch_geometric",
            install_name="torch-geometric",
            required=False,
            fallback_available=True,
            description="Graph neural networks for PyTorch",
        ),
        "sklearn": LibraryInfo(
            name="sklearn",
            import_name="sklearn",
            install_name="scikit-learn",
            required=False,
            fallback_available=True,
            description="Machine learning library",
        ),
        "deepchem": LibraryInfo(
            name="deepchem",
            import_name="deepchem",
            install_name="deepchem",
            required=False,
            fallback_available=True,
            description="Deep learning for drug discovery",
        ),
        "qiskit": LibraryInfo(
            name="qiskit",
            import_name="qiskit",
            install_name="qiskit",
            required=False,
            fallback_available=True,
            description="Quantum computing framework",
        ),
        "pyscf": LibraryInfo(
            name="pyscf",
            import_name="pyscf",
            install_name="pyscf",
            required=False,
            fallback_available=True,
            description="Python-based simulations of chemistry framework",
        ),
        "openfermion": LibraryInfo(
            name="openfermion",
            import_name="openfermion",
            install_name="openfermion",
            required=False,
            fallback_available=True,
            description="Electronic structure package for quantum computers",
        ),
        "mdanalysis": LibraryInfo(
            name="mdanalysis",
            import_name="MDAnalysis",
            install_name="MDAnalysis",
            required=False,
            fallback_available=True,
            description="Molecular dynamics analysis",
        ),
        "openbabel": LibraryInfo(
            name="openbabel",
            import_name="openbabel",
            install_name="openbabel",
            required=False,
            fallback_available=True,
            description="Chemical toolbox",
        ),
    }

    def __init__(self, auto_install: bool = True, force_fallbacks: bool = False):
        """
        Initialize library manager.

        Args:
            auto_install: Whether to attempt automatic installation
            force_fallbacks: Force use of fallbacks even if libraries are available
        """
        self.auto_install = auto_install
        self.force_fallbacks = force_fallbacks
        self.logger = logging.getLogger(__name__)

        # Track library status
        self.available_libraries: Dict[str, Any] = {}
        self.failed_libraries: Set[str] = set()
        self.fallback_libraries: Set[str] = set()

        # Initialize status tracking
        self._library_status: Dict[str, bool] = {}

    def import_library(self, library_name: str) -> tuple[bool, Optional[Any]]:
        """
        Import a library with fallback handling.

        Args:
            library_name: Name of library to import

        Returns:
            Tuple of (success, module_or_none)
        """
        if library_name not in self.LIBRARY_REGISTRY:
            self.logger.warning("Unknown library: %s", library_name)
            return False, None

        lib_info = self.LIBRARY_REGISTRY[library_name]

        # Check if already loaded
        if library_name in self.available_libraries:
            return True, self.available_libraries[library_name]

        # Check if previously failed
        if library_name in self.failed_libraries:
            return False, None

        # Force fallback if requested
        if self.force_fallbacks and lib_info.fallback_available:
            self.fallback_libraries.add(library_name)
            return False, None

        try:
            # Attempt import
            module = importlib.import_module(lib_info.import_name)
            self.available_libraries[library_name] = module
            self._library_status[library_name] = True
            self.logger.info("Successfully imported %s", library_name)
            return True, module

        except ImportError as e:
            self.logger.warning("Failed to import %s: %s", library_name, str(e))
            self.failed_libraries.add(library_name)
            self._library_status[library_name] = False

            # Attempt installation if enabled
            if self.auto_install:
                success = self._attempt_install(lib_info)
                if success:
                    return self.import_library(library_name)  # Retry after install

            return False, None

    def _attempt_install(self, lib_info: LibraryInfo) -> bool:
        """
        Attempt to install a library.

        Args:
            lib_info: Library information

        Returns:
            True if installation succeeded
        """
        try:
            self.logger.info("Attempting to install %s...", lib_info.name)

            # Try pip install
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", lib_info.install_name],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            if result.returncode == 0:
                self.logger.info("Successfully installed %s", lib_info.name)
                return True
            else:
                self.logger.error(
                    "Failed to install %s: %s", lib_info.name, result.stderr
                )
                return False

        except subprocess.TimeoutExpired:
            self.logger.error("Installation timeout for %s", lib_info.name)
            return False
        except Exception as e:
            self.logger.error("Installation error for %s: %s", lib_info.name, str(e))
            return False

    def is_available(self, library_name: str) -> bool:
        """Check if a library is available."""
        return library_name in self.available_libraries

    def get_module(self, library_name: str) -> Optional[Any]:
        """Get an imported module."""
        return self.available_libraries.get(library_name)

    def get_status_summary(self) -> Dict[str, Any]:
        """Get summary of library status."""
        total_libraries = len(self.LIBRARY_REGISTRY)
        available_count = len(self.available_libraries)
        failed_count = len(self.failed_libraries)
        fallback_count = len(self.fallback_libraries)

        return {
            "total_libraries": total_libraries,
            "available": available_count,
            "failed": failed_count,
            "using_fallbacks": fallback_count,
            "success_rate": (
                available_count / total_libraries if total_libraries > 0 else 0
            ),
            "library_status": self._library_status.copy(),
        }

    def print_status_report(self) -> None:
        """Print a formatted status report."""
        summary = self.get_status_summary()

        print("\n📚 Library Status Report")
        print("=" * 50)
        print(f"Total Libraries: {summary['total_libraries']}")
        print(f"Available: {summary['available']}")
        print(f"Failed: {summary['failed']}")
        print(f"Using Fallbacks: {summary['using_fallbacks']}")
        print(f"Success Rate: {summary['success_rate']:.1%}")

        print("\n📋 Detailed Status:")
        for lib_name, lib_info in self.LIBRARY_REGISTRY.items():
            if lib_name in self.available_libraries:
                status = "✅ Available"
            elif lib_name in self.fallback_libraries:
                status = "🔄 Using Fallback"
            elif lib_name in self.failed_libraries:
                status = "❌ Failed"
            else:
                status = "⏳ Not Checked"

            print(f"  {status} {lib_name:15} - {lib_info.description}")

        print("=" * 50)

    def import_multiple(
        self, library_names: List[str]
    ) -> Dict[str, tuple[bool, Optional[Any]]]:
        """
        Import multiple libraries at once.

        Args:
            library_names: List of library names to import

        Returns:
            Dictionary mapping library names to (success, module) tuples
        """
        results = {}
        for lib_name in library_names:
            results[lib_name] = self.import_library(lib_name)
        return results

    def require_libraries(
        self, library_names: List[str], allow_fallbacks: bool = True
    ) -> bool:
        """
        Ensure required libraries are available.

        Args:
            library_names: List of required library names
            allow_fallbacks: Whether fallbacks are acceptable

        Returns:
            True if all required libraries are available
        """
        all_available = True

        for lib_name in library_names:
            success, _ = self.import_library(lib_name)

            if not success:
                if allow_fallbacks and lib_name in self.fallback_libraries:
                    self.logger.info(
                        "Using fallback for required library: %s", lib_name
                    )
                else:
                    self.logger.error("Required library not available: %s", lib_name)
                    all_available = False

        return all_available
