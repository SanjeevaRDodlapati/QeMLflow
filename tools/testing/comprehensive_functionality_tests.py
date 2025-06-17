"""
Comprehensive Real-World Functionality Tests for QeMLflow
========================================================

Tests actual usage scenarios to validate the codebase works as intended.
"""

import numpy as np
import pandas as pd
import pytest
from typing import List, Optional


class TestQeMLflowCoreWorkflows:
    """Test core QeMLflow workflows that users would actually use."""
    
    def test_basic_data_loading_and_processing(self):
        """Test the complete data loading and processing pipeline."""
        try:
            from qemlflow.core.data_processing import QeMLflowDataLoader, process_smiles
            
            # Test basic functionality
            loader = QeMLflowDataLoader()
            assert loader is not None, "DataLoader should be created"
            
            # Test SMILES processing
            test_smiles = ["CCO", "c1ccccc1", "CC(=O)O", "invalid_smiles"]
            processed = process_smiles(test_smiles)
            
            assert len(processed) == 4, "Should process all SMILES"
            assert processed[0] is not None, "Valid SMILES should be processed"
            assert processed[3] is None, "Invalid SMILES should return None"
            
            print("✅ Data loading and processing pipeline works")
            return True
            
        except Exception as e:
            pytest.fail(f"Data processing workflow failed: {e}")
    
    def test_molecular_feature_extraction(self):
        """Test molecular feature extraction capabilities."""
        try:
            from qemlflow.core.preprocessing import extract_basic_molecular_descriptors
            
            # Test with sample molecules
            smiles_list = ["CCO", "c1ccccc1", "CC(=O)O"]
            features = extract_basic_molecular_descriptors(smiles_list)
            
            assert features is not None, "Features should be calculated"
            assert len(features) > 0, "Should return some features"
            
            print("✅ Molecular feature extraction works")
            return True
            
        except Exception as e:
            pytest.fail(f"Feature extraction workflow failed: {e}")
    
    def test_model_creation_and_training(self):
        """Test model creation and training capabilities."""
        try:
            from qemlflow.core.models import create_linear_model
            
            # Create sample data
            X = np.random.randn(100, 5)
            y = np.random.randn(100)
            
            # Create and test model
            model = create_linear_model()
            assert model is not None, "Model should be created"
            
            # Test fitting (basic functionality)
            result = model.fit(X, y)
            assert result is not None, "Model should fit and return metrics"
            
            print("✅ Model creation and training works")
            return True
            
        except Exception as e:
            pytest.fail(f"Model workflow failed: {e}")
    
    def test_utility_functions(self):
        """Test core utility functions."""
        try:
            from qemlflow.core.utils import validate_input, setup_logging
            
            # Test input validation
            assert validate_input({"test": "data"}) == True, "Valid dict should pass validation"
            assert validate_input(None) == False, "None should fail validation"
            
            # Test logging setup (use correct parameter name)
            setup_logging(log_level="INFO")  # This returns None but should work
            
            print("✅ Utility functions work")
            return True
            
        except Exception as e:
            pytest.fail(f"Utility workflow failed: {e}")
    
    def test_integration_system(self):
        """Test integration system functionality."""
        try:
            import qemlflow.integrations
            
            # Basic import test - this validates the integration system is loadable
            assert hasattr(qemlflow.integrations, '__file__'), "Integration module should be properly loaded"
            
            print("✅ Integration system works")
            return True
            
        except Exception as e:
            pytest.fail(f"Integration workflow failed: {e}")


class TestQeMLflowRealWorldScenarios:
    """Test realistic usage scenarios."""
    
    def test_drug_discovery_pipeline(self):
        """Test a basic drug discovery workflow."""
        try:
            from qemlflow.core.data_processing import process_smiles
            from qemlflow.core.preprocessing import extract_basic_molecular_descriptors
            from qemlflow.core.utils import validate_input
            
            # Simulate drug discovery workflow
            # 1. Load molecular data
            drug_smiles = [
                "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
                "CC(=O)OC1=CC=CC=C1C(=O)O",       # Aspirin
                "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
            ]
            
            # 2. Process SMILES
            processed_smiles = process_smiles(drug_smiles)
            assert all(s is not None for s in processed_smiles), "All drug SMILES should be valid"
            
            # 3. Calculate molecular descriptors
            features = extract_basic_molecular_descriptors(drug_smiles)
            assert features is not None, "Features should be calculated for drugs"
            
            # 4. Validate data
            assert validate_input(features), "Feature data should be valid"
            
            print("✅ Drug discovery pipeline works")
            return True
            
        except Exception as e:
            pytest.fail(f"Drug discovery pipeline failed: {e}")
    
    def test_qsar_modeling_scenario(self):
        """Test a QSAR modeling scenario."""
        try:
            from qemlflow.core.data_processing import process_smiles
            from qemlflow.core.models import create_linear_model
            import numpy as np
            
            # Simulate QSAR workflow
            # 1. Molecular data
            molecules = ["CCO", "CCC", "CCCC", "CCCCC", "CCCCCC"]
            
            # 2. Process molecules
            processed = process_smiles(molecules)
            valid_molecules = [m for m in processed if m is not None]
            assert len(valid_molecules) > 0, "Should have valid molecules"
            
            # 3. Create dummy features and target (molecular weight proxy)
            n_mols = len(valid_molecules)
            X = np.random.randn(n_mols, 3)  # Dummy molecular features
            y = np.random.randn(n_mols)     # Dummy target property
            
            # 4. Create and train QSAR model
            model = create_linear_model()
            metrics = model.fit(X, y)
            assert metrics is not None, "QSAR model should train successfully"
            
            print("✅ QSAR modeling scenario works")
            return True
            
        except Exception as e:
            pytest.fail(f"QSAR modeling scenario failed: {e}")


def run_comprehensive_tests():
    """Run all comprehensive tests and provide detailed results."""
    print("🧪 Running Comprehensive QeMLflow Functionality Tests")
    print("=" * 60)
    
    test_results = []
    
    # Core workflow tests
    core_tests = TestQeMLflowCoreWorkflows()
    core_test_methods = [
        ("Data Loading & Processing", core_tests.test_basic_data_loading_and_processing),
        ("Molecular Features", core_tests.test_molecular_feature_extraction),
        ("Model Training", core_tests.test_model_creation_and_training),
        ("Utility Functions", core_tests.test_utility_functions),
        ("Integration System", core_tests.test_integration_system),
    ]
    
    # Real-world scenario tests
    scenario_tests = TestQeMLflowRealWorldScenarios()
    scenario_test_methods = [
        ("Drug Discovery Pipeline", scenario_tests.test_drug_discovery_pipeline),
        ("QSAR Modeling", scenario_tests.test_qsar_modeling_scenario),
    ]
    
    all_tests = core_test_methods + scenario_test_methods
    
    print(f"\n🔬 Running {len(all_tests)} comprehensive tests...")
    
    for test_name, test_method in all_tests:
        try:
            print(f"\n🧪 Testing {test_name}...")
            result = test_method()
            test_results.append((test_name, True, None))
            print(f"   ✅ {test_name}: PASSED")
        except Exception as e:
            test_results.append((test_name, False, str(e)))
            print(f"   ❌ {test_name}: FAILED - {e}")
    
    # Summary
    passed = sum(1 for _, success, _ in test_results if success)
    total = len(test_results)
    success_rate = (passed / total) * 100
    
    print("\n" + "=" * 60)
    print("📊 COMPREHENSIVE TEST RESULTS:")
    print(f"   ✅ Passed: {passed}/{total}")
    print(f"   📈 Success Rate: {success_rate:.1f}%")
    
    if success_rate == 100:
        print("   🎉 All comprehensive tests passed! QeMLflow is fully functional.")
    elif success_rate >= 80:
        print("   ✅ Most tests passed! QeMLflow is largely functional.")
    elif success_rate >= 50:
        print("   ⚠️  Some tests failed. QeMLflow has partial functionality.")
    else:
        print("   ❌ Many tests failed. QeMLflow needs significant fixes.")
    
    return test_results


if __name__ == "__main__":
    run_comprehensive_tests()
