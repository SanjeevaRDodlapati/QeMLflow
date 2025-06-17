QeMLflow System Diagnostics
========================

Consolidated diagnostic utilities for QeMLflow system health checks.
Replaces the various scattered diagnostic scripts with a unified tool.
"""

import importlib
import os
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

class QeMLflowDiagnostics:
    """Comprehensive diagnostic tool for QeMLflow system health."""

    def __init__(self):
        self.results = {}
        self.warnings = []
        self.errors = []

    def run_full_diagnostics(self) -> Dict[str, Any]:
        """Run all diagnostic checks."""
        print("🔬 QeMLflow System Diagnostics")
        print("=" * 50)

        # Core system checks
        self.check_python_environment()
        self.check_core_dependencies()
        self.check_optional_dependencies()
        self.check_qemlflow_installation()
        self.check_data_directories()
        self.check_configuration()

        # Specialized checks
        self.check_quantum_setup()
        self.check_gpu_availability()
        self.check_experiment_tracking()

        # Generate summary
        return self.generate_summary()

    def check_python_environment(self):
        """Check Python environment."""
        print("\n🐍 Python Environment")
        print("-" * 30)

        python_version = sys.version
        python_executable = sys.executable

        self.results["python"] = {
            "version": python_version,
            "executable": python_executable,
            "version_info": sys.version_info,
        }

        print(
            f"Python Version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        )
        print(f"Executable: {python_executable}")

        # Check if we're in a virtual environment
        in_venv = hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
        self.results["python"]["in_virtual_env"] = in_venv
        print(f"Virtual Environment: {'Yes' if in_venv else 'No'}")

    def check_core_dependencies(self):
        """Check core dependencies."""
        print("\n📦 Core Dependencies")
        print("-" * 30)

        core_deps = [
            "numpy",
            "pandas",
            "scikit-learn",
            "matplotlib",
            "rdkit",
            "deepchem",
            "tensorflow",
        ]

        dependency_status = {}
        for dep in core_deps:
            try:
                module = importlib.import_module(dep.replace("-", "_"))
                version = getattr(module, "__version__", "unknown")
                dependency_status[dep] = {"installed": True, "version": version}
                print(f"✅ {dep}: {version}")
            except ImportError:
                dependency_status[dep] = {"installed": False, "version": None}
                print(f"❌ {dep}: Not installed")
                self.warnings.append(f"Core dependency {dep} not available")

        self.results["core_dependencies"] = dependency_status

    def check_optional_dependencies(self):
        """Check optional dependencies."""
        print("\n🔧 Optional Dependencies")
        print("-" * 30)

        optional_deps = [
            "qiskit",
            "psi4",
            "wandb",
            "plotly",
            "ipywidgets",
            "jupyter",
            "notebook",
            "jupyterlab",
        ]

        optional_status = {}
        for dep in optional_deps:
            try:
                module = importlib.import_module(dep)
                version = getattr(module, "__version__", "unknown")
                optional_status[dep] = {"installed": True, "version": version}
                print(f"✅ {dep}: {version}")
            except ImportError:
                optional_status[dep] = {"installed": False, "version": None}
                print(f"⚠️  {dep}: Not installed (optional)")

        self.results["optional_dependencies"] = optional_status

    def check_qemlflow_installation(self):
        """Check QeMLflow installation."""
        print("\n🧪 QeMLflow Installation")
        print("-" * 30)

        try:
            import qemlflow

            version = qemlflow.__version__
            install_path = qemlflow.__file__

            self.results["qemlflow"] = {
                "installed": True,
                "version": version,
                "path": install_path,
            }

            print(f"✅ QeMLflow Version: {version}")
            print(f"Installation Path: {Path(install_path).parent}")

            # Check submodules
            submodules = ["preprocessing", "models", "visualization", "integrations"]
            submodule_status = {}

            for submodule in submodules:
                try:
                    importlib.import_module(f"qemlflow.{submodule}")
                    submodule_status[submodule] = True
                    print(f"✅ qemlflow.{submodule}")
                except ImportError:
                    submodule_status[submodule] = False
                    print(f"❌ qemlflow.{submodule}")
                    self.warnings.append(f"QeMLflow submodule {submodule} not available")

            self.results["qemlflow"]["submodules"] = submodule_status

        except ImportError:
            self.results["qemlflow"] = {"installed": False}
            print("❌ QeMLflow not installed")
            self.errors.append("QeMLflow not installed")

    def check_data_directories(self):
        """Check data directories and permissions."""
        print("\n📁 Data Directories")
        print("-" * 30)

        directories = ["data", "cache", "logs", "notebooks", "examples"]
        directory_status = {}

        for directory in directories:
            dir_path = Path(directory)
            exists = dir_path.exists()
            writable = (
                dir_path.is_dir() and os.access(dir_path, os.W_OK) if exists else False
            )

            directory_status[directory] = {
                "exists": exists,
                "writable": writable,
                "path": str(dir_path.absolute()),
            }

            if exists:
                print(f"✅ {directory}: {dir_path.absolute()}")
                if not writable:
                    print(f"⚠️  {directory}: Not writable")
                    self.warnings.append(f"Directory {directory} not writable")
            else:
                print(f"❌ {directory}: Not found")

        self.results["directories"] = directory_status

    def check_configuration(self):
        """Check configuration system."""
        print("\n⚙️  Configuration")
        print("-" * 30)

        try:
            from qemlflow.config import get_config

            config = get_config()

            self.results["configuration"] = {
                "working": True,
                "environment": config.environment,
                "debug_mode": config.debug_mode,
            }

            print("✅ Configuration loaded")
            print(f"Environment: {config.environment}")
            print(f"Debug Mode: {config.debug_mode}")

        except Exception as e:
            self.results["configuration"] = {"working": False, "error": str(e)}
            print(f"❌ Configuration error: {e}")
            self.errors.append(f"Configuration system error: {e}")

    def check_quantum_setup(self):
        """Check quantum computing setup."""
        print("\n⚛️  Quantum Computing")
        print("-" * 30)

        quantum_status = {}

        # Check Qiskit
        try:
            import qiskit
            from qiskit import BasicAer

            # Try to get a backend
            backend = BasicAer.get_backend("qasm_simulator")
            quantum_status["qiskit"] = {
                "installed": True,
                "version": qiskit.__version__,
                "backend_available": True,
            }
            print(f"✅ Qiskit: {qiskit.__version__}")
            print(f"✅ Qiskit Backend: {backend.name()}")

        except ImportError:
            quantum_status["qiskit"] = {"installed": False}
            print("❌ Qiskit: Not installed")
        except Exception as e:
            quantum_status["qiskit"] = {
                "installed": True,
                "backend_available": False,
                "error": str(e),
            }
            print(f"⚠️  Qiskit: Installed but backend issues - {e}")

        # Check Psi4
        try:
            import psi4

            quantum_status["psi4"] = {"installed": True, "version": psi4.__version__}
            print(f"✅ Psi4: {psi4.__version__}")
        except ImportError:
            quantum_status["psi4"] = {"installed": False}
            print("❌ Psi4: Not installed")

        self.results["quantum"] = quantum_status

    def check_gpu_availability(self):
        """Check GPU availability."""
        print("\n🖥️  GPU Support")
        print("-" * 30)

        gpu_status = {}

        # Check TensorFlow GPU
        try:
            import tensorflow as tf

            gpu_available = len(tf.config.list_physical_devices("GPU")) > 0
            gpu_status["tensorflow"] = {
                "installed": True,
                "gpu_available": gpu_available,
                "devices": len(tf.config.list_physical_devices("GPU")),
            }

            if gpu_available:
                print(
                    f"✅ TensorFlow GPU: {len(tf.config.list_physical_devices('GPU'))} device(s)"
                )
            else:
                print("⚠️  TensorFlow: No GPU devices found")

        except ImportError:
            gpu_status["tensorflow"] = {"installed": False}
            print("❌ TensorFlow: Not installed")

        # Check PyTorch GPU
        try:
            import torch

            gpu_available = torch.cuda.is_available()
            gpu_status["pytorch"] = {
                "installed": True,
                "gpu_available": gpu_available,
                "devices": torch.cuda.device_count() if gpu_available else 0,
            }

            if gpu_available:
                print(f"✅ PyTorch GPU: {torch.cuda.device_count()} device(s)")
            else:
                print("⚠️  PyTorch: No GPU devices found")

        except ImportError:
            gpu_status["pytorch"] = {"installed": False}
            print("❌ PyTorch: Not installed")

        self.results["gpu"] = gpu_status

    def check_experiment_tracking(self):
        """Check experiment tracking setup."""
        print("\n📊 Experiment Tracking")
        print("-" * 30)

        tracking_status = {}

        # Check Weights & Biases
        try:
            import wandb

            # Check if logged in
            try:
                api = wandb.Api()
                user = api.viewer
                tracking_status["wandb"] = {
                    "installed": True,
                    "logged_in": True,
                    "user": user.username if hasattr(user, "username") else "unknown",
                }
                print(
                    f"✅ Weights & Biases: Logged in as {user.username if hasattr(user, 'username') else 'user'}"
                )
            except Exception:
                tracking_status["wandb"] = {"installed": True, "logged_in": False}
                print("⚠️  Weights & Biases: Not logged in")

        except ImportError:
            tracking_status["wandb"] = {"installed": False}
            print("❌ Weights & Biases: Not installed")

        self.results["experiment_tracking"] = tracking_status

    def generate_summary(self) -> Dict[str, Any]:
        """Generate diagnostic summary."""
        print("\n" + "=" * 50)
        print("📋 Diagnostic Summary")
        print("=" * 50)

        summary = {
            "timestamp": str(os.path.getmtime(__file__)),
            "total_warnings": len(self.warnings),
            "total_errors": len(self.errors),
            "warnings": self.warnings,
            "errors": self.errors,
            "results": self.results,
        }

        # Overall health score
        total_checks = 0
        passed_checks = 0

        for category, data in self.results.items():
            if isinstance(data, dict) and "installed" in data:
                total_checks += 1
                if data["installed"]:
                    passed_checks += 1

        health_score = (passed_checks / total_checks * 100) if total_checks > 0 else 0
        summary["health_score"] = health_score

        print(f"Overall Health Score: {health_score:.1f}%")
        print(f"Warnings: {len(self.warnings)}")
        print(f"Errors: {len(self.errors)}")

        if self.warnings:
            print("\n⚠️  Warnings:")
            for warning in self.warnings:
                print(f"  • {warning}")

        if self.errors:
            print("\n❌ Errors:")
            for error in self.errors:
                print(f"  • {error}")

        if health_score >= 80:
            print("\n✅ System is healthy and ready for QeMLflow!")
        elif health_score >= 60:
            print("\n⚠️  System has some issues but should work for basic usage.")
        else:
            print(
                "\n❌ System has significant issues. Please address errors before using QeMLflow."
            )

        return summary

def quick_check():
    """Quick health check."""
    print("🔬 QeMLflow Quick Health Check")
    print("=" * 40)

    checks = [
        ("Python", lambda: importlib.import_module("sys")),
        ("NumPy", lambda: importlib.import_module("numpy")),
        ("QeMLflow", lambda: importlib.import_module("qemlflow")),
        ("Config", lambda: importlib.import_module("qemlflow.config")),
    ]

    for name, check_func in checks:
        try:
            check_func()
            print(f"✅ {name}")
        except Exception:
            print(f"❌ {name}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="QeMLflow System Diagnostics")
    parser.add_argument("--quick", action="store_true", help="Run quick check only")
    parser.add_argument("--save", type=str, help="Save results to JSON file")

    args = parser.parse_args()

    if args.quick:
        quick_check()
    else:
        diagnostics = QeMLflowDiagnostics()
        results = diagnostics.run_full_diagnostics()

        if args.save:
            import json

            with open(args.save, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n💾 Results saved to {args.save}")
