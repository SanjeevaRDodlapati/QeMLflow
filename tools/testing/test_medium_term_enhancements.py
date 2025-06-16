#!/usr/bin/env python3
"""
Test script for ChemML medium-term enhancements
"""

import sys

import numpy as np
import pandas as pd

# Add current directory to path
sys.path.insert(0, "src")

def test_workflow_optimizer():
    """Test the workflow optimizer functionality."""
    print("🔧 Testing Workflow Optimizer...")

    try:
        from chemml.core.workflow_optimizer import WorkflowOptimizer, optimize_workflow

        # Create sample molecular data
        sample_smiles = ["CCO", "CCC", "CCCO", "CC(C)O"]
        sample_descriptors = np.random.random((100, 10))

        # Test workflow analysis
        result = optimize_workflow(
            molecular_data=sample_smiles,
            target_property="logP",
            preprocessing_steps=["validate_smiles", "generate_descriptors"],
        )

        print(
            f"   ✅ Data characteristics: {result['data_characteristics']['data_type']}"
        )
        print(
            f"   ✅ Recommended preprocessing: {len(result['recommended_preprocessing'])} steps"
        )
        print(
            f"   ✅ Optimization suggestions: {len(result['optimization_suggestions'])} suggestions"
        )

        # Test workflow optimizer directly
        optimizer = WorkflowOptimizer()
        analysis = optimizer.analyze_data_pipeline(
            sample_descriptors, target_property="solubility"
        )

        print(
            f"   ✅ Workflow analysis completed: {analysis['data_characteristics']['size']} samples"
        )

        return True

    except Exception as e:
        print(f"   ❌ Workflow optimizer test failed: {e}")
        return False

def test_advanced_ensembles():
    """Test the advanced ensemble methods."""
    print("🤖 Testing Advanced Ensemble Methods...")

    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.svm import SVR

        from chemml.core.ensemble_advanced import (
            AdaptiveEnsemble,
            MultiModalEnsemble,
            UncertaintyQuantifiedEnsemble,
            create_adaptive_ensemble,
        )

        # Create sample data
        X = np.random.random((100, 10))
        y = np.random.random(100)

        # Test Adaptive Ensemble
        base_models = [
            RandomForestRegressor(n_estimators=10, random_state=42),
            LinearRegression(),
            SVR(kernel="linear"),
        ]

        adaptive_ensemble = create_adaptive_ensemble(base_models)
        adaptive_ensemble.fit(X, y)
        predictions = adaptive_ensemble.predict(X[:10])

        print(f"   ✅ Adaptive ensemble predictions: {len(predictions)} samples")

        # Test predictions with uncertainty
        predictions_with_unc = adaptive_ensemble.predict(
            X[:10], return_uncertainty=True
        )
        if isinstance(predictions_with_unc, tuple):
            pred, unc = predictions_with_unc
            print(f"   ✅ Uncertainty estimates: {len(unc)} uncertainty values")

        # Test Multi-Modal Ensemble
        modality_models = {
            "descriptors": RandomForestRegressor(n_estimators=10, random_state=42),
            "fingerprints": LinearRegression(),
        }

        multimodal_ensemble = MultiModalEnsemble(modality_models)
        modality_data = {
            "descriptors": X,
            "fingerprints": X[:, :5],  # Different feature space
        }

        multimodal_ensemble.fit(modality_data, y)
        multimodal_pred = multimodal_ensemble.predict(modality_data)

        print(f"   ✅ Multi-modal ensemble predictions: {len(multimodal_pred)} samples")

        # Test Uncertainty Quantified Ensemble
        unc_ensemble = UncertaintyQuantifiedEnsemble(
            base_models[:2], bootstrap_samples=10  # Use fewer models for faster testing
        )

        unc_ensemble.fit(X, y)
        pred_with_full_unc = unc_ensemble.predict(X[:10], return_uncertainties=True)

        if isinstance(pred_with_full_unc, tuple):
            pred, uncertainties = pred_with_full_unc
            print(
                f"   ✅ Full uncertainty analysis: {len(uncertainties)} uncertainty types"
            )

        return True

    except Exception as e:
        print(f"   ❌ Advanced ensemble test failed: {e}")
        return False

def test_integration():
    """Test integration with existing ChemML features."""
    print("🔗 Testing Integration with Existing Features...")

    try:
        import chemml

        # Test that new features are accessible from main module
        from chemml import (
            AdaptiveEnsemble,
            ModelRecommendationEngine,
            PerformanceDashboard,
            WorkflowOptimizer,
        )

        print("   ✅ All enhanced features imported successfully")

        # Test combined workflow
        sample_data = np.random.random((50, 8))

        # Model recommendation
        recommender = ModelRecommendationEngine()
        recommendation = recommender.recommend_best_model(
            sample_data, "toxicity", task_type="classification"
        )

        print(f"   ✅ Model recommendation: {recommendation['recommended_model']}")

        # Workflow optimization
        optimizer = WorkflowOptimizer()
        workflow_analysis = optimizer.analyze_data_pipeline(sample_data)

        print(
            f"   ✅ Workflow analysis: {workflow_analysis['data_characteristics']['size']} samples"
        )

        # Performance monitoring (test single report generation)
        dashboard = PerformanceDashboard()
        report = dashboard.generate_real_time_report()

        print("   ✅ Performance monitoring tested (single report)")
        print(f"   ✅ Report contains {len(report)} metrics sections")

        return True

    except Exception as e:
        print(f"   ❌ Integration test failed: {e}")
        return False

def main():
    """Run all tests for medium-term enhancements."""
    print("🧪 Testing ChemML Medium-Term Enhancements")
    print("=" * 50)

    tests = [test_workflow_optimizer, test_advanced_ensembles, test_integration]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"   ❌ Test failed with exception: {e}")
            results.append(False)
        print()

    # Summary
    passed = sum(results)
    total = len(results)

    print("=" * 50)
    print(f"🏁 Test Summary: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All medium-term enhancements are working perfectly!")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
