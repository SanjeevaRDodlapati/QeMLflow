"""
ChemML Registry and Model Discovery Demo
=======================================

This example demonstrates ChemML's advanced model registry and discovery
capabilities, including AI-powered model recommendations and management.

Features demonstrated:
- Model registry operations
- AI-powered model discovery
- Compatibility checking
- Performance tracking
- Model information retrieval

Prerequisites:
- ChemML with integrations installed
- Understanding of basic integration concepts

Expected runtime: 1-2 minutes
"""

import sys
from pathlib import Path

# Add ChemML to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent.parent.parent))


def main():
    """Run registry and discovery demonstration."""
    print("🔍 ChemML Registry & Model Discovery Demo")
    print("=" * 60)

    try:
        # Import ChemML components
        print("\n📋 Setting up ChemML integrations...")
        from chemml.integrations import get_manager
        from chemml.integrations.core import AdvancedRegistry

        # Get manager and registry
        manager = get_manager()
        registry = AdvancedRegistry()

        print("✅ Manager and registry initialized!")

        # Demo 1: Basic Registry Operations
        demo_basic_registry(registry)

        # Demo 2: Model Discovery
        demo_model_discovery(registry)

        # Demo 3: Compatibility Checking
        demo_compatibility_checking(registry)

        # Demo 4: Model Information
        demo_model_information(registry)

        # Demo 5: Performance Integration
        demo_performance_integration(manager, registry)

        print("\n🎉 Registry demo completed successfully!")
        print("\n📚 Related examples:")
        print("   - examples/integrations/framework/monitoring_demo.py")
        print("   - examples/integrations/framework/pipeline_demo.py")
        print("   - examples/utilities/performance_testing.py")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   This demo requires the full ChemML integrations package")
        print("   Make sure you have the latest version installed")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("   Some features may not be available in test mode")


def demo_basic_registry(registry):
    """Demonstrate basic registry operations."""
    print("\n" + "=" * 50)
    print("🗃️  DEMO 1: Basic Registry Operations")
    print("=" * 50)

    try:
        # Register a model
        print("\n📝 Registering models...")
        registry.register_model(
            "test_model",
            {
                "category": "molecular",
                "task": "property_prediction",
                "description": "Test molecular property predictor",
            },
        )
        print("   ✅ Test model registered")

        # List registered models
        print("\n📋 Listing registered models...")
        models = registry.list_models()
        print(f"   Found {len(models)} registered models:")
        for model in models[:5]:  # Show first 5
            print(f"      - {model}")

        # Check if model exists
        print("\n🔍 Checking model availability...")
        if registry.has_model("test_model"):
            print("   ✅ test_model is available")
        else:
            print("   ℹ️  test_model not found")

    except Exception as e:
        print(f"   ⚠️  Registry operations: {e}")


def demo_model_discovery(registry):
    """Demonstrate AI-powered model discovery."""
    print("\n" + "=" * 50)
    print("🤖 DEMO 2: AI-Powered Model Discovery")
    print("=" * 50)

    try:
        # Task-based discovery
        print("\n🎯 Discovering models by task...")
        task_models = registry.discover_models_by_task("structure_prediction")
        print(f"   Found {len(task_models)} structure prediction models:")
        for model in task_models:
            print(f"      - {model}")

        # Category-based discovery
        print("\n📂 Discovering models by category...")
        molecular_models = registry.discover_models_by_category("molecular")
        print(f"   Found {len(molecular_models)} molecular modeling tools:")
        for model in molecular_models:
            print(f"      - {model}")

        # Search functionality
        print("\n🔎 Searching models...")
        search_results = registry.search_models("protein")
        print(f"   Found {len(search_results)} protein-related models:")
        for model in search_results:
            print(f"      - {model}")

    except Exception as e:
        print(f"   ⚠️  Model discovery: {e}")


def demo_compatibility_checking(registry):
    """Demonstrate model compatibility checking."""
    print("\n" + "=" * 50)
    print("🔗 DEMO 3: Model Compatibility Checking")
    print("=" * 50)

    try:
        # Check compatibility between models
        print("\n🔍 Checking model compatibility...")

        # Example compatibility checks
        compatibility_tests = [
            ("boltz", "alphafold"),
            ("deepchem", "rdkit"),
            ("pytorch", "sklearn"),
        ]

        for model1, model2 in compatibility_tests:
            try:
                compatible = registry.check_compatibility(model1, model2)
                status = "✅ Compatible" if compatible else "❌ Incompatible"
                print(f"   {model1} + {model2}: {status}")
            except Exception:
                print(f"   {model1} + {model2}: ℹ️  Unable to check")

        # Pipeline compatibility
        print("\n🔄 Checking pipeline compatibility...")
        pipeline_models = ["preprocessing", "boltz", "analysis"]
        try:
            pipeline_valid = registry.validate_pipeline(pipeline_models)
            status = "✅ Valid" if pipeline_valid else "❌ Invalid"
            print(f"   Pipeline {' → '.join(pipeline_models)}: {status}")
        except Exception:
            print("   Pipeline validation: ℹ️  Feature not available")

    except Exception as e:
        print(f"   ⚠️  Compatibility checking: {e}")


def demo_model_information(registry):
    """Demonstrate detailed model information retrieval."""
    print("\n" + "=" * 50)
    print("📊 DEMO 4: Model Information Retrieval")
    print("=" * 50)

    try:
        # Get detailed model info
        print("\n📄 Getting detailed model information...")

        test_models = ["boltz", "test_model", "deepchem"]

        for model_name in test_models:
            try:
                info = registry.get_model_info(model_name)
                print(f"\n   📋 {model_name}:")
                if isinstance(info, dict):
                    for key, value in info.items():
                        if isinstance(value, list) and len(value) > 3:
                            print(f"      {key}: {len(value)} items")
                        else:
                            print(f"      {key}: {value}")
                else:
                    print(f"      Info: {info}")
            except Exception:
                print(f"   📋 {model_name}: ℹ️  Information not available")

        # Generate model report
        print("\n📊 Generating comprehensive model report...")
        try:
            report = registry.generate_model_report("boltz")
            print("   ✅ Model report generated")
            print(f"   📄 Report preview: {str(report)[:100]}...")
        except Exception:
            print("   📊 Model report: ℹ️  Feature not available")

    except Exception as e:
        print(f"   ⚠️  Model information: {e}")


def demo_performance_integration(manager, registry):
    """Demonstrate performance monitoring integration."""
    print("\n" + "=" * 50)
    print("📈 DEMO 5: Performance Integration")
    print("=" * 50)

    try:
        # Performance monitoring setup
        print("\n📊 Setting up performance monitoring...")
        from chemml.integrations.core import PerformanceMonitor

        monitor = PerformanceMonitor()
        print("   ✅ Performance monitor initialized")

        # Simulate model operation with monitoring
        print("\n🔄 Simulating monitored model operation...")
        with monitor.track_performance():
            # Simulate some work
            import time

            time.sleep(0.1)  # Small delay to simulate computation

            # Registry operations
            models = registry.list_models()

        # Get performance stats
        stats = monitor.get_stats()
        print(f"   ⏱️  Operation time: {stats.get('total_time', 'N/A'):.3f}s")
        print(f"   💾 Memory usage: {stats.get('peak_memory_mb', 'N/A')} MB")

        # Performance benchmarking
        print("\n🏁 Performance benchmarking...")
        try:
            benchmark_results = monitor.benchmark_operation(
                lambda: registry.list_models(), iterations=5
            )
            print(f"   📊 Average time: {benchmark_results.get('avg_time', 'N/A'):.3f}s")
            print(f"   📊 Min time: {benchmark_results.get('min_time', 'N/A'):.3f}s")
            print(f"   📊 Max time: {benchmark_results.get('max_time', 'N/A'):.3f}s")
        except Exception:
            print("   📊 Benchmarking: ℹ️  Feature not available")

    except ImportError:
        print("   ⚠️  Performance monitoring not available")
    except Exception as e:
        print(f"   ⚠️  Performance integration: {e}")


if __name__ == "__main__":
    main()
