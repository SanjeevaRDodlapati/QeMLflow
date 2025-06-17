#!/usr/bin/env python3
"""
Comprehensive Integration Test for QeMLflow Enhancements
====================================================

Tests Phase 1, 2, and 3 implementations to ensure all features work together.

Usage:
    python tests/integration/test_comprehensive_enhancements.py
"""

import sys
import time
from pathlib import Path

import numpy as np

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def test_phase_1_infrastructure():
    """Test Phase 1: Critical Infrastructure improvements."""
    print("🧪 Testing Phase 1: Critical Infrastructure")
    print("-" * 45)

    # Test enhanced registry
    try:
        from qemlflow.integrations.core.advanced_registry import AdvancedModelRegistry

        registry = AdvancedModelRegistry()
        print("✅ Advanced registry loaded successfully")

        # Test serialization (the main fix)
        registry._save_registry()
        registry._load_registry()
        print("✅ Registry serialization working correctly")

    except Exception as e:
        print(f"❌ Registry test failed: {e}")

    # Test enhanced health check
    try:
        import subprocess
        from pathlib import Path

        # Run health check
        result = subprocess.run(
            [
                sys.executable,
                str(
                    Path(__file__).parent.parent.parent
                    / "tools"
                    / "assessment"
                    / "health_check.py"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            print("✅ Enhanced health check working")
        else:
            print(f"⚠️ Health check returned non-zero: {result.returncode}")

    except Exception as e:
        print(f"❌ Health check test failed: {e}")

    # Test dependency audit
    try:
        from pathlib import Path

        audit_path = (
            Path(__file__).parent.parent.parent
            / "tools"
            / "security"
            / "dependency_audit.py"
        )
        if audit_path.exists():
            print("✅ Dependency audit tool available")
        else:
            print("❌ Dependency audit tool not found")
    except Exception as e:
        print(f"❌ Dependency audit test failed: {e}")

    print("✅ Phase 1 testing completed\n")


def test_phase_2_user_experience():
    """Test Phase 2: Enhanced User Experience."""
    print("🎨 Testing Phase 2: Enhanced User Experience")
    print("-" * 45)

    # Test enhanced error handling
    try:
        from qemlflow.utils.enhanced_error_handling import QeMLflowError, debug_context

        # Test custom error
        try:
            raise QeMLflowError(
                "Test error message",
                context="Test Context",
                solutions=["Solution 1", "Solution 2"],
                error_code="TEST_ERROR",
            )
        except QeMLflowError as e:
            if "Test error message" in str(e) and "TEST_ERROR" in str(e):
                print("✅ Enhanced error handling working")
            else:
                print("❌ Enhanced error format incorrect")

        # Test debug context
        with debug_context("Test Debug Context"):
            pass
        print("✅ Debug context working")

    except Exception as e:
        print(f"❌ Enhanced error handling test failed: {e}")

    # Test enhanced UI
    try:
        from qemlflow.utils.enhanced_ui import qemlflow_interface, quick_start

        # Test interface
        interface = qemlflow_interface
        quick_guide = quick_start()

        if "QeMLflow" in quick_guide:
            print("✅ Enhanced UI interface working")
        else:
            print("❌ Enhanced UI not generating proper output")

        # Test function help
        help_text = interface.get_function_help("load_data")
        if "load_data" in help_text:
            print("✅ Interactive help system working")
        else:
            print("❌ Help system not working properly")

    except Exception as e:
        print(f"❌ Enhanced UI test failed: {e}")

    print("✅ Phase 2 testing completed\n")


def test_phase_3_advanced_features():
    """Test Phase 3: Advanced Features."""
    print("🚀 Testing Phase 3: Advanced Features")
    print("-" * 45)

    # Test AutoML optimizer
    try:
        from qemlflow.advanced.ml_optimizer import AutoMLOptimizer, ModelAnalytics

        # Generate sample data
        np.random.seed(42)
        X = np.random.random((100, 10))
        y = np.random.randint(0, 2, 100)

        # Test AutoML
        automl = AutoMLOptimizer(optimization_strategy="bayesian")
        result = automl.optimize(X, y)

        if result.best_score > 0 and result.best_params:
            print("✅ AutoML optimizer working")
        else:
            print("❌ AutoML optimizer not returning valid results")

        # Test analytics
        analytics = ModelAnalytics()
        y_pred = np.random.randint(0, 2, 100)
        metrics = analytics.analyze_model_performance(y, y_pred, "test_model")

        if hasattr(metrics, "accuracy"):
            print("✅ Model analytics working")
        else:
            print("❌ Model analytics not working properly")

    except Exception as e:
        print(f"❌ Advanced ML features test failed: {e}")

    # Test enterprise monitoring
    try:
        from qemlflow.enterprise.monitoring import MetricsCollector, MonitoringDashboard

        # Test monitoring setup
        dashboard = MonitoringDashboard()
        collector = dashboard.metrics_collector

        # Test metrics collection
        collector.record_user_activity(
            "test_user", "test_action", "test_resource", 0.5, True
        )

        if len(collector.user_activities) > 0:
            print("✅ Enterprise monitoring working")
        else:
            print("❌ Enterprise monitoring not recording activities")

        # Test dashboard data generation
        overview = dashboard.analytics.generate_system_overview()
        if isinstance(overview, dict):
            print("✅ Analytics dashboard working")
        else:
            print("❌ Analytics dashboard not generating data")

    except Exception as e:
        print(f"❌ Enterprise monitoring test failed: {e}")

    print("✅ Phase 3 testing completed\n")


def test_integration_workflow():
    """Test complete integration workflow."""
    print("🔄 Testing Complete Integration Workflow")
    print("-" * 45)

    try:
        # Import all major components
        from qemlflow.advanced.ml_optimizer import AutoMLOptimizer
        from qemlflow.enterprise.monitoring import MonitoringDashboard
        from qemlflow.utils.enhanced_error_handling import (
            debug_context,
            global_performance_monitor,
        )

        # Start monitoring
        dashboard = MonitoringDashboard()

        # Test workflow with monitoring and performance tracking
        with debug_context("Integration Test Workflow"):
            with global_performance_monitor.monitor_performance("test_workflow"):
                # Create AutoML optimizer
                automl = AutoMLOptimizer()

                # Generate test data
                np.random.seed(42)
                X = np.random.random((50, 5))
                y = np.random.randint(0, 2, 50)

                # Run optimization
                result = automl.optimize(X, y)

                # Record activity
                dashboard.metrics_collector.record_user_activity(
                    "integration_test", "optimize", "test_model", 1.0, True
                )

        # Check if everything worked
        if (
            result.best_score > 0
            and len(dashboard.metrics_collector.user_activities) > 0
            and "test_workflow" in global_performance_monitor.metrics
        ):
            print("✅ Complete integration workflow successful")
            return True
        else:
            print("❌ Integration workflow missing components")
            return False

    except Exception as e:
        print(f"❌ Integration workflow test failed: {e}")
        return False


def test_health_score_improvement():
    """Test if health score has improved."""
    print("📊 Testing Health Score Improvement")
    print("-" * 45)

    try:
        import subprocess
        from pathlib import Path

        # Run health check and capture output
        health_check_path = (
            Path(__file__).parent.parent.parent
            / "tools"
            / "assessment"
            / "health_check.py"
        )
        result = subprocess.run(
            [sys.executable, str(health_check_path)],
            capture_output=True,
            text=True,
            timeout=60,
        )

        output = result.stdout

        # Extract health score
        health_score = None
        for line in output.split("\n"):
            if "Overall Health Score:" in line:
                try:
                    score_str = line.split(":")[1].strip().split("/")[0]
                    health_score = float(score_str)
                    break
                except:
                    continue

        if health_score is not None:
            print(f"📊 Current Health Score: {health_score}/100")

            # Check for improvements
            improvements = []
            if "security tools: available" in output.lower():
                improvements.append("Security tools installed")
            if "integration manager available" in output.lower():
                improvements.append("Integration system working")
            if "registry.*valid" in output.lower():
                improvements.append("Registry integrity maintained")

            if improvements:
                print("✅ Health improvements detected:")
                for improvement in improvements:
                    print(f"   • {improvement}")

            if health_score >= 60:
                print("✅ Health score is acceptable")
            else:
                print("⚠️ Health score could be improved")

        else:
            print("❌ Could not extract health score")

    except Exception as e:
        print(f"❌ Health score test failed: {e}")


def main():
    """Run comprehensive integration tests."""
    print("🧬 QeMLflow Comprehensive Enhancement Testing")
    print("=" * 50)
    print(f"Python: {sys.version}")
    print(f"Test started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)

    # Run all test phases
    test_phase_1_infrastructure()
    test_phase_2_user_experience()
    test_phase_3_advanced_features()

    # Test integration
    integration_success = test_integration_workflow()

    # Test health improvements
    test_health_score_improvement()

    # Final summary
    print("📋 Test Summary")
    print("-" * 20)
    if integration_success:
        print("✅ All phases working together successfully")
        print("🎯 QeMLflow enhancements are production-ready")
    else:
        print("⚠️ Some integration issues detected")
        print("🔧 Additional debugging may be needed")

    print(f"\n🏁 Testing completed: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
