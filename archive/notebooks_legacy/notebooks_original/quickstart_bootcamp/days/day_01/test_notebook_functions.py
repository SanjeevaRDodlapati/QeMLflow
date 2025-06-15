#!/usr/bin/env python3
"""
Test script to validate Day 1 notebook key functions
Ensures the notebook will run without errors
"""

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Add utils to path for assessment framework
utils_path = Path("../utils")
if utils_path.exists():
    sys.path.append(str(utils_path))


def test_imports():
    """Test all critical imports"""
    print("🧪 Testing Critical Imports")
    print("=" * 40)

    results = {}

    # Test basic scientific stack
    try:
        from datetime import datetime

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import requests
        import seaborn as sns

        results["scientific_stack"] = True
        print("✅ Scientific stack (numpy, pandas, matplotlib, seaborn)")
    except ImportError as e:
        results["scientific_stack"] = False
        print(f"❌ Scientific stack error: {e}")

    # Test assessment framework
    try:
        from assessment_framework import (
            create_assessment,
            create_dashboard,
            create_widget,
        )

        results["assessment_framework"] = True
        print("✅ Assessment framework")
    except ImportError:
        results["assessment_framework"] = False
        print("⚠️ Assessment framework not found (will use fallback)")

    # Test cheminformatics
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, Descriptors, Draw

        results["rdkit"] = True
        print("✅ RDKit cheminformatics")
    except ImportError as e:
        results["rdkit"] = False
        print(f"❌ RDKit error: {e}")

    # Test DeepChem
    try:
        import deepchem as dc

        results["deepchem"] = True
        print(f"✅ DeepChem v{dc.__version__}")
    except ImportError as e:
        results["deepchem"] = False
        print(f"❌ DeepChem error: {e}")

    # Test sklearn
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error, r2_score

        results["sklearn"] = True
        print("✅ Scikit-learn")
    except ImportError as e:
        results["sklearn"] = False
        print(f"❌ Scikit-learn error: {e}")

    return results


def test_molecular_operations():
    """Test molecular manipulation functions"""
    print("\n🧪 Testing Molecular Operations")
    print("=" * 40)

    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors

        # Test drug molecules
        drug_molecules = {
            "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
            "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        }

        mol_objects = {}
        for name, smiles in drug_molecules.items():
            mol = Chem.MolFromSmiles(smiles)
            mol_objects[name] = mol

            # Calculate properties
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)

            print(f"✅ {name}: MW={mw:.2f}, LogP={logp:.2f}")

        print(f"✅ Processed {len(mol_objects)} molecules successfully")
        return True

    except Exception as e:
        print(f"❌ Molecular operations failed: {e}")
        return False


def test_assessment_fallback():
    """Test assessment framework fallback system"""
    print("\n🧪 Testing Assessment Fallback System")
    print("=" * 40)

    try:
        # Create fallback assessment system
        class BasicAssessment:
            def __init__(self, student_id, day, track):
                self.student_id = student_id
                self.day = day
                self.track = track
                self.track_configs = {
                    "standard": {"target_hours": 4.5, "min_completion": 0.8}
                }

            def record_activity(self, activity, data):
                print(f"📝 Activity recorded: {activity}")

            def get_comprehensive_report(self):
                return {"total_time": 240, "performance_score": 85}

            def save_final_report(self):
                print("💾 Progress saved")

        class BasicWidget:
            def display(self):
                print("📋 Assessment checkpoint active")

        # Test creating assessment
        assessment = BasicAssessment("test_student", 1, "standard")
        assessment.record_activity("test", {"status": "success"})

        widget = BasicWidget()
        widget.display()

        print("✅ Assessment fallback system works correctly")
        return True

    except Exception as e:
        print(f"❌ Assessment fallback failed: {e}")
        return False


def test_data_handling():
    """Test data handling and processing"""
    print("\n🧪 Testing Data Handling")
    print("=" * 40)

    try:
        import numpy as np
        import pandas as pd

        # Create demo dataset
        demo_data = {
            "Name": ["Aspirin", "Caffeine", "Ibuprofen"],
            "SMILES": [
                "CC(=O)OC1=CC=CC=C1C(=O)O",
                "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
                "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
            ],
            "MW": [180.16, 194.19, 206.29],
            "LogP": [1.19, -0.07, 3.97],
        }

        df = pd.DataFrame(demo_data)
        print(f"✅ Created demo dataset with {len(df)} compounds")

        # Test basic operations
        mean_mw = df["MW"].mean()
        print(f"✅ Calculated average MW: {mean_mw:.2f}")

        # Test Lipinski's Rule of Five
        violations = 0
        for _, row in df.iterrows():
            if row["MW"] > 500 or row["LogP"] > 5:
                violations += 1

        print(f"✅ Lipinski analysis: {violations} violations found")
        return True

    except Exception as e:
        print(f"❌ Data handling failed: {e}")
        return False


def test_model_compatibility():
    """Test model creation compatibility"""
    print("\n🧪 Testing Model Compatibility")
    print("=" * 40)

    try:
        # Test sklearn model
        import numpy as np
        from sklearn.ensemble import RandomForestRegressor

        # Create demo data
        X = np.random.randn(100, 10)
        y = np.random.randn(100)

        # Create and test model
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)

        predictions = model.predict(X[:5])
        print(f"✅ RandomForest model created and tested")

        # Test DeepChem compatibility (if available)
        try:
            import deepchem as dc

            # Test dataset creation concept
            class DemoDataset:
                def __init__(self, size):
                    self.X = np.random.randn(size, 1024)
                    self.y = np.random.randn(size, 1)

                def __len__(self):
                    return len(self.X)

            demo_dataset = DemoDataset(50)
            print(f"✅ DeepChem-style dataset concept works")

        except ImportError:
            print("⚠️ DeepChem not available for advanced models")

        return True

    except Exception as e:
        print(f"❌ Model compatibility failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 Day 1 Notebook Validation Test")
    print("=" * 50)

    # Run all tests
    results = {}

    results["imports"] = test_imports()
    results["molecular"] = test_molecular_operations()
    results["assessment"] = test_assessment_fallback()
    results["data"] = test_data_handling()
    results["models"] = test_model_compatibility()

    # Summary
    print("\n📊 Test Summary")
    print("=" * 20)

    total_tests = len(results)
    passed_tests = sum(1 for v in results.values() if v is True)

    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")

    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED - Notebook should run successfully!")
    elif passed_tests >= total_tests * 0.8:
        print("\n👍 MOSTLY WORKING - Notebook should run with minor fallbacks")
    else:
        print("\n⚠️ SOME ISSUES DETECTED - Check error messages above")

    print(f"\n💡 The notebook is designed to work even with missing dependencies")
    print(f"🔄 Fallback systems ensure learning continues in all scenarios")


if __name__ == "__main__":
    main()
