#!/usr/bin/env python3
"""
ChemML Performance Optimization Tool
Analyzes and optimizes import patterns, configuration loading, and other performance bottlenecks.
"""

import argparse
import importlib
import json
import os
import sys
import time
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional


def measure_time(func):
    """Decorator to measure function execution time."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        return result, execution_time

    return wrapper


class ImportProfiler:
    """Profiles import times for ChemML modules."""

    def __init__(self):
        self.import_times = {}
        self.failed_imports = []

    @measure_time
    def profile_import(self, module_name: str) -> bool:
        """Profile the import time of a module."""
        try:
            importlib.import_module(module_name)
            return True
        except ImportError as e:
            self.failed_imports.append((module_name, str(e)))
            return False

    def profile_chemml_imports(self) -> Dict[str, float]:
        """Profile all ChemML module imports."""
        chemml_modules = [
            "chemml",
            "chemml.core",
            "chemml.core.data",
            "chemml.core.models",
            "chemml.core.evaluation",
            "chemml.research",
            "chemml.research.drug_discovery",
            "chemml.integrations",
            "chemml.config",
            "chemml.utils",
            "chemml.notebooks",
        ]

        print("🔍 Profiling ChemML imports...")
        results = {}

        for module in chemml_modules:
            success, import_time = self.profile_import(module)
            if success:
                results[module] = import_time
                print(f"  ✅ {module}: {import_time:.3f}s")
            else:
                print(f"  ❌ {module}: Failed to import")

        return results

    def profile_external_dependencies(self) -> Dict[str, float]:
        """Profile external dependency import times."""
        external_deps = [
            "numpy",
            "pandas",
            "scikit-learn",
            "rdkit",
            "deepchem",
            "tensorflow",
            "torch",
            "matplotlib",
            "plotly",
            "wandb",
        ]

        print("\n🔍 Profiling external dependencies...")
        results = {}

        for dep in external_deps:
            try:
                success, import_time = self.profile_import(dep)
                if success:
                    results[dep] = import_time
                    status = "⚠️ SLOW" if import_time > 1.0 else "✅"
                    print(f"  {status} {dep}: {import_time:.3f}s")
            except Exception:
                print(f"  ❌ {dep}: Not available")

        return results


class ConfigPerformanceProfiler:
    """Profiles configuration loading performance."""

    @measure_time
    def profile_config_loading(self) -> Dict[str, Any]:
        """Profile configuration loading time."""
        try:
            from chemml.config import unified_config

            config = unified_config.ChemMLConfig()
            return config.to_dict()
        except Exception as e:
            return {"error": str(e)}

    @measure_time
    def profile_yaml_parsing(self, yaml_content: str) -> Dict[str, Any]:
        """Profile YAML parsing performance."""
        try:
            import yaml

            return yaml.safe_load(yaml_content)
        except Exception as e:
            return {"error": str(e)}

    def run_config_benchmarks(self) -> Dict[str, float]:
        """Run configuration loading benchmarks."""
        print("🔍 Profiling configuration performance...")
        results = {}

        # Test config loading
        config_data, config_time = self.profile_config_loading()
        results["config_loading"] = config_time
        print(f"  ✅ Config loading: {config_time:.3f}s")

        # Test YAML parsing with sample content
        sample_yaml = """
        features:
          molecular: true
          quantum: false
        models:
          default: "random_forest"
          available: ["linear", "random_forest", "neural_network"]
        performance:
          cache_size: 1000
          parallel_jobs: 4
        """

        yaml_data, yaml_time = self.profile_yaml_parsing(sample_yaml)
        results["yaml_parsing"] = yaml_time
        print(f"  ✅ YAML parsing: {yaml_time:.3f}s")

        return results


class MemoryProfiler:
    """Profiles memory usage of ChemML operations."""

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage information."""
        try:
            import psutil

            process = psutil.Process()
            memory_info = process.memory_info()

            return {
                "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size
                "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size
                "percent": process.memory_percent(),
            }
        except ImportError:
            return {"error": "psutil not available"}

    def profile_memory_usage(self) -> Dict[str, Any]:
        """Profile memory usage during various operations."""
        print("🔍 Profiling memory usage...")

        baseline = self.get_memory_usage()
        print(f"  📊 Baseline memory: {baseline.get('rss_mb', 0):.1f} MB")

        # Test import memory impact
        try:
            import numpy as np
            import pandas as pd

            after_basic_imports = self.get_memory_usage()

            # Create test data
            _test_data = pd.DataFrame(np.random.randn(1000, 10))
            after_data_creation = self.get_memory_usage()

            return {
                "baseline": baseline,
                "after_imports": after_basic_imports,
                "after_data": after_data_creation,
                "import_overhead_mb": after_basic_imports.get("rss_mb", 0)
                - baseline.get("rss_mb", 0),
                "data_overhead_mb": after_data_creation.get("rss_mb", 0)
                - after_basic_imports.get("rss_mb", 0),
            }

        except Exception as e:
            return {"error": str(e)}


def create_optimization_report() -> Dict[str, Any]:
    """Create a comprehensive performance optimization report."""
    print("🚀 ChemML Performance Analysis")
    print("=" * 50)

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python_version": sys.version,
        "results": {},
    }

    # Import profiling
    import_profiler = ImportProfiler()
    chemml_imports = import_profiler.profile_chemml_imports()
    external_imports = import_profiler.profile_external_dependencies()

    report["results"]["import_performance"] = {
        "chemml_modules": chemml_imports,
        "external_dependencies": external_imports,
        "slow_imports": {k: v for k, v in external_imports.items() if v > 1.0},
        "failed_imports": import_profiler.failed_imports,
    }

    # Config profiling
    config_profiler = ConfigPerformanceProfiler()
    config_results = config_profiler.run_config_benchmarks()
    report["results"]["config_performance"] = config_results

    # Memory profiling
    memory_profiler = MemoryProfiler()
    memory_results = memory_profiler.profile_memory_usage()
    report["results"]["memory_usage"] = memory_results

    return report


def generate_optimization_recommendations(report: Dict[str, Any]) -> List[str]:
    """Generate performance optimization recommendations."""
    recommendations = []
    results = report["results"]

    # Import optimizations
    import_perf = results.get("import_performance", {})
    slow_imports = import_perf.get("slow_imports", {})

    if slow_imports:
        recommendations.append("🔧 Implement lazy loading for slow imports:")
        for module, time_taken in slow_imports.items():
            recommendations.append(
                f"   • {module} ({time_taken:.2f}s) - consider lazy loading"
            )

    # Config optimizations
    config_perf = results.get("config_performance", {})
    config_time = config_perf.get("config_loading", 0)

    if config_time > 0.1:
        recommendations.append(
            f"⚡ Config loading is slow ({config_time:.3f}s) - implement caching"
        )

    # Memory optimizations
    memory_usage = results.get("memory_usage", {})
    import_overhead = memory_usage.get("import_overhead_mb", 0)

    if import_overhead > 100:
        recommendations.append(
            f"💾 High import memory overhead ({import_overhead:.1f} MB) - optimize imports"
        )

    # General recommendations
    recommendations.extend(
        [
            "📚 Consider implementing import caching for frequently used modules",
            "🔄 Add configuration caching to reduce file I/O",
            "⚡ Implement lazy evaluation for expensive computations",
            "🧹 Review and optimize hot code paths identified in profiling",
        ]
    )

    return recommendations


def print_optimization_report(report: Dict[str, Any]):
    """Print formatted optimization report."""
    print("\n📊 Performance Analysis Summary")
    print("-" * 30)

    results = report["results"]

    # Import performance summary
    import_perf = results.get("import_performance", {})
    chemml_total = sum(import_perf.get("chemml_modules", {}).values())
    external_total = sum(import_perf.get("external_dependencies", {}).values())

    print(f"⏱️  Total ChemML import time: {chemml_total:.3f}s")
    print(f"⏱️  Total external import time: {external_total:.3f}s")

    slow_count = len(import_perf.get("slow_imports", {}))
    if slow_count > 0:
        print(f"⚠️  Slow imports detected: {slow_count}")

    # Config performance
    config_perf = results.get("config_performance", {})
    config_time = config_perf.get("config_loading", 0)
    print(f"⚙️  Configuration loading: {config_time:.3f}s")

    # Memory usage
    memory_usage = results.get("memory_usage", {})
    if "baseline" in memory_usage:
        baseline_mb = memory_usage["baseline"].get("rss_mb", 0)
        print(f"💾 Baseline memory usage: {baseline_mb:.1f} MB")

    # Recommendations
    recommendations = generate_optimization_recommendations(report)
    print(f"\n💡 Optimization Recommendations ({len(recommendations)}):")
    for rec in recommendations[:10]:  # Show top 10
        print(f"  {rec}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="ChemML Performance Optimization")
    parser.add_argument(
        "--save-report", action="store_true", help="Save detailed report to JSON file"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run quick performance check only"
    )

    args = parser.parse_args()

    try:
        report = create_optimization_report()
        print_optimization_report(report)

        if args.save_report:
            with open("performance_optimization_report.json", "w") as f:
                json.dump(report, f, indent=2)
            print("\n💾 Detailed report saved to performance_optimization_report.json")

    except Exception as e:
        print(f"❌ Error during performance analysis: {e}")


if __name__ == "__main__":
    main()
