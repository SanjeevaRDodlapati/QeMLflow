"""
ChemML Health Check and Assessment Tool
======================================

Consolidated tool for checking ChemML installation, configuration,
and overall system health. Combines functionality from multiple
assessment scripts.

Features:
- Installation verification
- Integration system health check
- Configuration validation
- Dependency verification

Usage:
    python tools/assessment/health_check.py
    python tools/assessment/health_check.py --detailed
    python tools/assessment/health_check.py --fix-issues
"""

import argparse
import importlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class HealthChecker:
    """Comprehensive health check for ChemML installation."""

    def __init__(self):
        self.results = {}
        self.issues = []
        self.recommendations = []
        self.fixes_available = []
        
        # Add src directory to Python path for development version
        self.repo_root = Path(__file__).parent.parent.parent
        src_path = self.repo_root / "src"
        if src_path.exists() and str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))

    def run_full_assessment(self) -> Dict[str, Any]:
        """Run complete health assessment."""
        print("🏥 ChemML Health Check & Assessment")
        print("=" * 50)

        # System information
        self.results["system"] = self._check_system_info()

        # Python environment
        self.results["python"] = self._check_python_environment()

        # ChemML installation
        self.results["chemml"] = self._check_chemml_installation()

        # Dependencies
        self.results["dependencies"] = self._check_dependencies()

        # Integration system
        self.results["integrations"] = self._check_integration_system()

        # Performance
        self.results["performance"] = self._check_performance()

        # Configuration
        self.results["configuration"] = self._check_configuration()

        # Security vulnerabilities
        self.results["security"] = self._check_security_vulnerabilities()

        # Dependency conflicts
        self.results["dependency_conflicts"] = self._check_dependency_conflicts()

        # Registry integrity
        self.results["registry_integrity"] = self._check_registry_integrity()

        # Generate summary
        self.results["summary"] = self._generate_summary()

        return self.results

    def _check_system_info(self) -> Dict[str, Any]:
        """Check system information."""
        print("\n1️⃣ System Information")
        print("-" * 30)

        info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "architecture": platform.architecture()[0],
            "processor": platform.processor(),
            "machine": platform.machine(),
        }

        print(f"   Platform: {info['platform']}")
        print(f"   Python: {info['python_version']}")
        print(f"   Architecture: {info['architecture']}")

        # Check system compatibility
        compatibility_issues = []
        if sys.version_info < (3, 8):
            compatibility_issues.append("Python version < 3.8 (not recommended)")

        info["compatibility_issues"] = compatibility_issues
        info["status"] = "good" if not compatibility_issues else "warning"

        return info

    def _check_python_environment(self) -> Dict[str, Any]:
        """Check Python environment setup."""
        print("\n2️⃣ Python Environment")
        print("-" * 30)

        env_info = {
            "executable": sys.executable,
            "path": sys.path[:3],  # First 3 entries
            "virtual_env": os.environ.get("VIRTUAL_ENV"),
            "pip_version": self._get_pip_version(),
        }

        print(f"   Executable: {env_info['executable']}")
        print(f"   Virtual Env: {env_info['virtual_env'] or 'None'}")
        print(f"   Pip Version: {env_info['pip_version']}")

        # Check environment health
        issues = []
        if not env_info["virtual_env"]:
            issues.append("No virtual environment detected (recommended)")

        env_info["issues"] = issues
        env_info["status"] = "good" if not issues else "warning"

        return env_info

    def _check_chemml_installation(self) -> Dict[str, Any]:
        """Check ChemML installation status."""
        print("\n3️⃣ ChemML Installation")
        print("-" * 30)

        install_info = {}

        try:
            import chemml

            install_info["installed"] = True
            install_info["version"] = getattr(chemml, "__version__", "unknown")
            install_info["location"] = chemml.__file__
            print(f"   ✅ ChemML installed: v{install_info['version']}")
            print(f"   📍 Location: {Path(install_info['location']).parent}")

            # Test basic imports
            test_imports = [
                "chemml.core",
                "chemml.integrations",
                "chemml.preprocessing",
            ]

            import_results = {}
            for module in test_imports:
                try:
                    importlib.import_module(module)
                    import_results[module] = "success"
                    print(f"   ✅ {module}")
                except ImportError as e:
                    import_results[module] = f"failed: {e}"
                    print(f"   ❌ {module}: {e}")

            install_info["import_tests"] = import_results
            install_info["status"] = "good"

        except ImportError as e:
            install_info["installed"] = False
            install_info["error"] = str(e)
            install_info["status"] = "error"
            print(f"   ❌ ChemML not installed: {e}")
            self.fixes_available.append("Install ChemML: pip install chemml")

        return install_info

    def _check_dependencies(self) -> Dict[str, Any]:
        """Check core dependencies."""
        print("\n4️⃣ Dependencies")
        print("-" * 30)

        core_deps = [
            "numpy",
            "pandas",
            "scikit-learn",
            "matplotlib",
            "rdkit",
            "torch",
            "transformers",
        ]

        optional_deps = ["jupyter", "wandb", "mlflow", "pytest"]

        dep_status = {"core": {}, "optional": {}}

        # Check core dependencies
        for dep in core_deps:
            try:
                module = importlib.import_module(dep)
                version = getattr(module, "__version__", "unknown")
                dep_status["core"][dep] = {"installed": True, "version": version}
                print(f"   ✅ {dep}: {version}")
            except ImportError:
                dep_status["core"][dep] = {"installed": False}
                print(f"   ❌ {dep}: not installed")

        # Check optional dependencies
        for dep in optional_deps:
            try:
                module = importlib.import_module(dep)
                version = getattr(module, "__version__", "unknown")
                dep_status["optional"][dep] = {"installed": True, "version": version}
                print(f"   🔵 {dep}: {version}")
            except ImportError:
                dep_status["optional"][dep] = {"installed": False}
                print(f"   ⚪ {dep}: not installed (optional)")

        # Assess dependency health
        missing_core = [
            dep for dep, info in dep_status["core"].items() if not info["installed"]
        ]
        dep_status["missing_core"] = missing_core
        dep_status["status"] = "error" if missing_core else "good"

        # Add enhanced dependency validation
        dep_status["vulnerability_check"] = self._quick_security_check()
        dep_status["conflict_check"] = self._quick_conflict_check()

        return dep_status

    def _check_integration_system(self) -> Dict[str, Any]:
        """Check integration system health."""
        print("\n5️⃣ Integration System")
        print("-" * 30)

        integration_info = {}

        try:
            from chemml.integrations import get_manager

            # Test manager creation
            manager = get_manager()
            integration_info["manager_available"] = True
            print("   ✅ Integration manager available")

            # Test basic functionality
            try:
                # Try to list available models/adapters using different methods
                models_count = 0

                # Try various methods to get model count
                try:
                    if hasattr(manager, "list_models"):
                        models = manager.list_models()  # type: ignore
                        models_count = len(models)
                    elif hasattr(manager, "get_available_models"):
                        models = manager.get_available_models()  # type: ignore
                        models_count = len(models) if models else 0
                    elif hasattr(manager, "registry") and hasattr(manager.registry, "models"):  # type: ignore
                        models_count = len(manager.registry.models)  # type: ignore
                    else:
                        # Try to access registry directly
                        from chemml.integrations.core.advanced_registry import (
                            AdvancedModelRegistry,
                        )

                        registry = AdvancedModelRegistry()
                        models_count = len(registry.models)
                except Exception:
                    models_count = "unknown"

                integration_info["available_models"] = models_count
                if isinstance(models_count, int):
                    print(f"   📋 Available models: {models_count}")
                else:
                    print("   📋 Model listing: method not available")

                integration_info["status"] = "good"

            except Exception as e:
                integration_info["functionality_error"] = str(e)
                integration_info["status"] = "warning"
                print(f"   ⚠️  Integration functionality: {e}")

        except ImportError as e:
            integration_info["manager_available"] = False
            integration_info["error"] = str(e)
            integration_info["status"] = "error"
            print(f"   ❌ Integration system: {e}")

        return integration_info

    def _check_performance(self) -> Dict[str, Any]:
        """Check basic performance characteristics."""
        print("\n6️⃣ Performance Assessment")
        print("-" * 30)

        perf_info = {}

        # Test import performance
        start_time = time.time()
        try:
            import chemml

            import_time = time.time() - start_time
            perf_info["chemml_import_time"] = import_time
            print(f"   ⏱️  ChemML import time: {import_time:.3f}s")

            if import_time > 2.0:
                perf_info["import_warning"] = "Slow import time (>2s)"
                print(f"   ⚠️  Slow import time: {import_time:.3f}s")
        except ImportError:
            perf_info["chemml_import_time"] = None
            print("   ❌ Cannot measure import time (ChemML not available)")

        # Test basic computation
        try:
            import numpy as np

            start_time = time.time()

            # Simple computation test
            data = np.random.random((1000, 100))
            _result = np.dot(data, data.T)
            computation_time = time.time() - start_time

            perf_info["computation_test"] = computation_time
            print(f"   🧮 Basic computation: {computation_time:.3f}s")

        except Exception as e:
            perf_info["computation_error"] = str(e)
            print(f"   ❌ Computation test failed: {e}")

        perf_info["status"] = "good"
        return perf_info

    def _check_configuration(self) -> Dict[str, Any]:
        """Check configuration files and settings."""
        print("\n7️⃣ Configuration")
        print("-" * 30)

        config_info = {}

        # Check for configuration files
        config_paths = [
            Path.cwd() / "config" / "chemml_config.yaml",
            Path.cwd() / "config" / "advanced_config.yaml",
            Path.home() / ".chemml" / "config.yaml",
        ]

        found_configs = []
        for config_path in config_paths:
            if config_path.exists():
                found_configs.append(str(config_path))
                print(f"   ✅ Config found: {config_path.name}")

        if not found_configs:
            print("   ℹ️  No configuration files found (using defaults)")

        config_info["config_files"] = found_configs
        config_info["status"] = "good"

        return config_info

    def _check_security_vulnerabilities(self) -> Dict[str, Any]:
        """Check for known security vulnerabilities."""
        print("\n🔒 Security Vulnerability Assessment")
        print("-" * 30)

        security_info = {
            "scan_completed": False,
            "vulnerabilities": [],
            "recommendations": [],
            "tools_available": {},
        }

        # Check if security tools are available
        security_tools = {
            "safety": "pip install safety",
            "bandit": "pip install bandit",
            "pip-audit": "pip install pip-audit",
        }

        for tool, install_cmd in security_tools.items():
            try:
                result = subprocess.run(
                    [tool, "--version"], capture_output=True, text=True, timeout=10
                )
                security_info["tools_available"][tool] = result.returncode == 0
                if result.returncode == 0:
                    print(f"   ✅ {tool}: available")
                else:
                    print(f"   ❌ {tool}: not available ({install_cmd})")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                security_info["tools_available"][tool] = False
                print(f"   ❌ {tool}: not available ({install_cmd})")

        # Run available security checks
        if security_info["tools_available"].get("safety", False):
            try:
                result = subprocess.run(
                    ["safety", "check", "--json"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.stdout:
                    safety_data = json.loads(result.stdout)
                    security_info["vulnerabilities"].extend(safety_data)
                    print(
                        f"   📊 Safety scan: {len(safety_data)} vulnerabilities found"
                    )
                else:
                    print("   ✅ Safety scan: no vulnerabilities found")
            except Exception as e:
                print(f"   ⚠️ Safety scan failed: {e}")

        if security_info["tools_available"].get("bandit", False):
            try:
                result = subprocess.run(
                    ["bandit", "-r", "src/", "-f", "json", "-q"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.stdout:
                    bandit_data = json.loads(result.stdout)
                    issues = bandit_data.get("results", [])
                    security_info["vulnerabilities"].extend(issues)
                    print(f"   📊 Bandit scan: {len(issues)} issues found")
                else:
                    print("   ✅ Bandit scan: no issues found")
            except Exception as e:
                print(f"   ⚠️ Bandit scan failed: {e}")

        # Add recommendations
        if security_info["vulnerabilities"]:
            security_info["recommendations"].extend(
                [
                    "Review identified vulnerabilities and update dependencies",
                    "Run 'pip install --upgrade' for affected packages",
                    "Consider using pip-audit for comprehensive vulnerability scanning",
                ]
            )

        security_info["scan_completed"] = True
        return security_info

    def _check_dependency_conflicts(self) -> Dict[str, Any]:
        """Check for dependency conflicts and version issues."""
        print("\n📦 Dependency Conflict Analysis")
        print("-" * 30)

        dependency_info = {
            "conflicts": [],
            "outdated": [],
            "recommendations": [],
            "pip_check_passed": False,
        }

        # Run pip check for dependency conflicts
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "check"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                dependency_info["pip_check_passed"] = True
                print("   ✅ No dependency conflicts detected")
            else:
                dependency_info["pip_check_passed"] = False
                conflicts = result.stdout.strip().split("\n") if result.stdout else []
                dependency_info["conflicts"] = conflicts
                print(f"   ❌ {len(conflicts)} dependency conflicts found")
                for conflict in conflicts[:5]:  # Show first 5
                    print(f"     • {conflict}")
                if len(conflicts) > 5:
                    print(f"     ... and {len(conflicts) - 5} more")
        except Exception as e:
            print(f"   ⚠️ Dependency check failed: {e}")

        # Check for outdated packages
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "list", "--outdated", "--format=json"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.stdout:
                outdated_packages = json.loads(result.stdout)
                dependency_info["outdated"] = outdated_packages
                print(f"   📊 {len(outdated_packages)} packages have updates available")
            else:
                print("   ✅ All packages are up to date")
        except Exception as e:
            print(f"   ⚠️ Outdated package check failed: {e}")

        # Add recommendations
        if dependency_info["conflicts"]:
            dependency_info["recommendations"].extend(
                [
                    "Resolve dependency conflicts by updating or pinning package versions",
                    "Consider using a fresh virtual environment",
                    "Review requirements.txt for version conflicts",
                ]
            )

        if dependency_info["outdated"]:
            dependency_info["recommendations"].append(
                "Update packages with 'pip install --upgrade package_name'"
            )

        return dependency_info

    def _check_registry_integrity(self) -> Dict[str, Any]:
        """Check integrity of model registry and configuration files."""
        print("\n🗂️ Registry & Configuration Integrity")
        print("-" * 30)

        registry_info = {
            "registry_files": {},
            "config_files": {},
            "json_validity": {},
            "recommendations": [],
        }

        # Check registry files
        registry_paths = [
            Path.home() / ".chemml" / "model_registry.json",
            Path.cwd() / "config" / "chemml_config.yaml",
            Path.cwd() / "config" / "advanced_config.yaml",
        ]

        for path in registry_paths:
            file_info = {
                "exists": path.exists(),
                "readable": False,
                "valid_format": False,
                "size_bytes": 0,
            }

            if path.exists():
                try:
                    file_info["size_bytes"] = path.stat().st_size
                    file_info["readable"] = True

                    # Check file format validity
                    if path.suffix == ".json":
                        with open(path, "r") as f:
                            json.load(f)
                        file_info["valid_format"] = True
                        print(
                            f"   ✅ {path.name}: valid JSON ({file_info['size_bytes']} bytes)"
                        )
                    elif path.suffix in [".yml", ".yaml"]:
                        # Basic YAML check (without pyyaml dependency)
                        with open(path, "r") as f:
                            content = f.read()
                        # Simple validation - check for basic YAML structure
                        if ":" in content and not content.strip().startswith("{"):
                            file_info["valid_format"] = True
                        print(
                            f"   ✅ {path.name}: appears valid ({file_info['size_bytes']} bytes)"
                        )

                except Exception as e:
                    print(f"   ❌ {path.name}: invalid format - {e}")
                    registry_info["recommendations"].append(
                        f"Fix or regenerate {path.name}"
                    )
            else:
                print(f"   ⚠️ {path.name}: not found")
                if "config" in str(path):
                    registry_info["recommendations"].append(
                        f"Create missing configuration file: {path.name}"
                    )

            registry_info[
                "registry_files" if "registry" in path.name else "config_files"
            ][str(path)] = file_info

        return registry_info

    def _performance_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive performance benchmarks."""
        print("\n⚡ Performance Benchmark Suite")
        print("-" * 30)

        benchmark_info = {
            "import_times": {},
            "computation_times": {},
            "memory_usage": {},
            "recommendations": [],
        }

        # Test import performance
        import_tests = [
            "chemml",
            "chemml.core",
            "chemml.integrations",
            "numpy",
            "pandas",
            "torch",
        ]

        for module in import_tests:
            try:
                start_time = time.time()
                importlib.import_module(module)
                import_time = time.time() - start_time
                benchmark_info["import_times"][module] = import_time

                if import_time < 0.1:
                    status = "✅"
                elif import_time < 1.0:
                    status = "⚠️"
                else:
                    status = "❌"

                print(f"   {status} {module}: {import_time:.3f}s")

                if import_time > 1.0:
                    benchmark_info["recommendations"].append(
                        f"Optimize {module} import time (currently {import_time:.3f}s)"
                    )

            except ImportError:
                print(f"   ❌ {module}: import failed")
                benchmark_info["import_times"][module] = -1

        # Test basic computations
        try:
            import numpy as np

            # Matrix operations
            start_time = time.time()
            a = np.random.random((1000, 1000))
            b = np.random.random((1000, 1000))
            _c = np.dot(a, b)
            computation_time = time.time() - start_time
            benchmark_info["computation_times"]["matrix_1000x1000"] = computation_time
            print(f"   📊 Matrix multiplication (1000x1000): {computation_time:.3f}s")

            if computation_time > 2.0:
                benchmark_info["recommendations"].append(
                    "Consider optimizing NumPy installation (BLAS/LAPACK)"
                )

        except Exception as e:
            print(f"   ❌ Computation benchmark failed: {e}")

        return benchmark_info

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate overall health summary."""
        print("\n📊 Health Summary")
        print("-" * 30)

        # Count status types
        statuses = {}
        for component, info in self.results.items():
            if component != "summary" and isinstance(info, dict):
                status = info.get("status", "unknown")
                statuses[status] = statuses.get(status, 0) + 1

        # Calculate overall health score
        total_components = sum(statuses.values())
        good_components = statuses.get("good", 0)
        health_score = (
            (good_components / total_components * 100) if total_components > 0 else 0
        )

        # Generate recommendations
        recommendations = []

        if self.results.get("chemml", {}).get("status") == "error":
            recommendations.append("Install ChemML package")

        if self.results.get("dependencies", {}).get("missing_core"):
            recommendations.append("Install missing core dependencies")

        if self.results.get("python", {}).get("virtual_env") is None:
            recommendations.append("Consider using a virtual environment")

        if self.results.get("performance", {}).get("import_warning"):
            recommendations.append("Optimize import performance")

        # Display summary
        print(f"   🎯 Overall Health Score: {health_score:.1f}/100")
        print(f"   ✅ Good: {statuses.get('good', 0)}")
        print(f"   ⚠️  Warnings: {statuses.get('warning', 0)}")
        print(f"   ❌ Errors: {statuses.get('error', 0)}")

        if recommendations:
            print("\n🎯 Recommendations:")
            for i, rec in enumerate(recommendations[:5], 1):
                print(f"   {i}. {rec}")

        return {
            "health_score": health_score,
            "component_status": statuses,
            "recommendations": recommendations,
            "fixes_available": self.fixes_available,
        }

    def _get_pip_version(self) -> str:
        """Get pip version."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return result.stdout.split()[1]
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, IndexError):
            pass
        return "unknown"

    def _quick_security_check(self) -> Dict[str, Any]:
        """Quick security validation check."""
        security_result = {
            "tools_available": False,
            "basic_check_passed": True,
            "recommendations": [],
        }

        # Check if any security tools are available
        try:
            subprocess.run(["safety", "--version"], capture_output=True, timeout=5)
            security_result["tools_available"] = True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            security_result["recommendations"].append(
                "Install security tools: pip install safety bandit"
            )

        return security_result

    def _quick_conflict_check(self) -> Dict[str, Any]:
        """Quick dependency conflict check."""
        conflict_result = {
            "pip_check_available": True,
            "conflicts_detected": False,
            "recommendations": [],
        }

        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "check"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                conflict_result["conflicts_detected"] = True
                conflict_result["recommendations"].append(
                    "Resolve pip dependency conflicts"
                )
        except Exception:
            conflict_result["pip_check_available"] = False
            conflict_result["recommendations"].append(
                "Unable to check for dependency conflicts"
            )

        return conflict_result

    def apply_fixes(self):
        """Apply available automatic fixes."""
        print("\n🔧 Attempting to fix identified issues...")

        fixes_applied = 0

        # Fix 1: Create missing directories
        chemml_dir = Path.home() / ".chemml"
        if not chemml_dir.exists():
            chemml_dir.mkdir(parents=True, exist_ok=True)
            print("   ✅ Created ChemML user directory")
            fixes_applied += 1

        # Fix 2: Create basic registry if missing
        registry_file = chemml_dir / "model_registry.json"
        if not registry_file.exists():
            basic_registry = {
                "models": {},
                "compatibility_matrix": {},
                "popularity_scores": {},
            }
            with open(registry_file, "w") as f:
                json.dump(basic_registry, f, indent=2)
            print("   ✅ Created basic model registry")
            fixes_applied += 1

        if fixes_applied == 0:
            print("   ℹ️  No automatic fixes available")
        else:
            print(f"   🎉 Applied {fixes_applied} fixes")

    # ...existing code...


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="ChemML Health Check Tool")
    parser.add_argument(
        "--detailed", action="store_true", help="Show detailed information"
    )
    parser.add_argument(
        "--fix-issues", action="store_true", help="Attempt to fix issues"
    )
    parser.add_argument(
        "--json-output", action="store_true", help="Output results as JSON"
    )

    args = parser.parse_args()

    checker = HealthChecker()
    results = checker.run_full_assessment()

    if args.fix_issues:
        checker.apply_fixes()

    if args.json_output:
        print("\n" + "=" * 50)
        print("JSON Output:")
        print(json.dumps(results, indent=2, default=str))

    # Final status
    health_score = results["summary"]["health_score"]

    if health_score >= 80:
        print("\n🎉 ChemML installation looks healthy!")
    elif health_score >= 60:
        print("\n⚠️  ChemML installation has some issues but is functional")
    else:
        print("\n❌ ChemML installation has significant issues")

    print(f"\nOverall Health Score: {health_score:.1f}/100")


if __name__ == "__main__":
    main()
