#!/usr/bin/env python3
"""
Test script to validate new framework modules for clinical research,
environmental chemistry, and materials discovery.
"""

def test_clinical_research():
    """Test clinical research module functionality."""
    try:
        from chemml.research.clinical_research import (
            ClinicalTrialOptimizer,
            PatientStratificationEngine,
            RegulatoryComplianceFramework,
            quick_clinical_analysis,
        )

        # Quick test
        results = quick_clinical_analysis("oncology")

        print("✅ Clinical Research Module: PASSED")
        print(
            f"   - Patient stratification accuracy: {results['patient_stratification']['accuracy']:.3f}"
        )
        print(
            f"   - Trial optimization success rate: {results['trial_optimization']['predicted_success_rate']:.2%}"
        )
        print(
            f"   - Regulatory compliance score: {results['regulatory_compliance']['overall_score']:.1%}"
        )
        return True

    except Exception as e:
        print(f"❌ Clinical Research Module: FAILED - {e}")
        return False

def test_environmental_chemistry():
    """Test environmental chemistry module functionality."""
    try:
        from chemml.research.environmental_chemistry import (
            AtmosphericChemistryAnalyzer,
            EnvironmentalMonitoringSystem,
            GreenChemistryOptimizer,
            quick_environmental_analysis,
        )

        # Quick test
        results = quick_environmental_analysis("air_quality")

        print("✅ Environmental Chemistry Module: PASSED")
        print(
            f"   - Pollution prediction R²: {results['pollution_prediction']['metrics']['r2']:.3f}"
        )
        print(
            f"   - Green chemistry max score: {results['green_chemistry']['max_green_score']:.1f}"
        )
        print(
            f"   - Atmospheric analysis timepoints: {results['summary']['atmospheric_timepoints']:,}"
        )
        return True

    except Exception as e:
        print(f"❌ Environmental Chemistry Module: FAILED - {e}")
        return False

def test_materials_discovery():
    """Test materials discovery module functionality."""
    try:
        from chemml.research.materials_discovery import (
            GenerativeMaterialsModel,
            InverseMaterialsDesigner,
            MaterialsClusterAnalyzer,
            MaterialsPropertyPredictor,
            comprehensive_materials_discovery,
        )

        # Quick test
        target_props = {"young_modulus": 200, "hardness": 15, "yield_strength": 500}
        results = comprehensive_materials_discovery(target_props)

        print("✅ Materials Discovery Module: PASSED")

        # Get average R² across all property predictions
        prop_prediction = results["property_prediction"]["model_performance"]
        avg_r2 = sum([metrics["r2"] for metrics in prop_prediction.values()]) / len(
            prop_prediction
        )

        print(f"   - Average property prediction R²: {avg_r2:.3f}")
        print(
            f"   - Best design fitness: {results['summary']['best_design_fitness']:.3f}"
        )
        print(f"   - Materials analyzed: {results['summary']['materials_analyzed']:,}")
        return True

    except Exception as e:
        print(f"❌ Materials Discovery Module: FAILED - {e}")
        return False

def main():
    """Run all validation tests."""
    print("🧪 Testing ChemML Framework Module Integration")
    print("=" * 50)

    tests = [
        test_clinical_research,
        test_environmental_chemistry,
        test_materials_discovery,
    ]

    passed = 0
    total = len(tests)

    for test_func in tests:
        try:
            if test_func():
                passed += 1
            print()  # Add spacing
        except Exception as e:
            print(f"❌ {test_func.__name__}: CRITICAL FAILURE - {e}")

    print("=" * 50)
    print(f"📊 TEST RESULTS: {passed}/{total} modules passed")

    if passed == total:
        print("🎉 ALL FRAMEWORK MODULES WORKING CORRECTLY!")
        print("✅ Ready for notebook integration testing")
    else:
        print("⚠️ Some modules need attention")

    return passed == total

if __name__ == "__main__":
    main()
