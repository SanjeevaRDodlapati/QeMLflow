"""
Performance and Stress Testing for QeMLflow
==========================================

Tests performance with larger datasets and stress scenarios.
"""

import time
import numpy as np
import pandas as pd
from typing import List


def test_large_dataset_processing():
    """Test QeMLflow with larger datasets."""
    print("🧪 Testing Large Dataset Processing...")
    
    try:
        from qemlflow.core.data_processing import process_smiles
        
        # Generate a larger set of SMILES for testing
        test_smiles = [
            "CCO", "CCC", "CCCC", "CCCCC", "CCCCCC",
            "c1ccccc1", "c1ccc(C)cc1", "c1ccc(CC)cc1",
            "CC(=O)O", "CC(=O)OC", "CC(=O)CC",
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        ] * 50  # 600 molecules
        
        start_time = time.time()
        processed = process_smiles(test_smiles)
        end_time = time.time()
        
        processing_time = end_time - start_time
        molecules_per_second = len(test_smiles) / processing_time
        
        print(f"   ✅ Processed {len(test_smiles)} molecules in {processing_time:.2f}s")
        print(f"   📊 Rate: {molecules_per_second:.1f} molecules/second")
        
        # Validate results
        valid_count = sum(1 for mol in processed if mol is not None)
        success_rate = (valid_count / len(test_smiles)) * 100
        print(f"   ✅ Success Rate: {success_rate:.1f}%")
        
        return processing_time < 10.0  # Should process in under 10 seconds
        
    except Exception as e:
        print(f"   ❌ Large dataset test failed: {e}")
        return False


def test_feature_extraction_performance():
    """Test molecular feature extraction performance."""
    print("\n🧪 Testing Feature Extraction Performance...")
    
    try:
        from qemlflow.core.preprocessing import extract_basic_molecular_descriptors
        
        # Test with moderately sized dataset
        smiles_list = [
            "CCO", "CCC", "CCCC", "c1ccccc1", "CC(=O)O",
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "c1ccc(C)cc1",
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        ] * 10  # 80 molecules
        
        start_time = time.time()
        features = extract_basic_molecular_descriptors(smiles_list)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        print(f"   ✅ Extracted features for {len(smiles_list)} molecules in {processing_time:.2f}s")
        
        if features is not None:
            print(f"   📊 Feature matrix shape: {np.array(features).shape}")
            return True
        else:
            print("   ⚠️  No features returned")
            return False
            
    except Exception as e:
        print(f"   ❌ Feature extraction test failed: {e}")
        return False


def test_model_training_performance():
    """Test model training with various dataset sizes."""
    print("\n🧪 Testing Model Training Performance...")
    
    try:
        from qemlflow.core.models import create_linear_model, create_rf_model
        
        results = []
        
        # Test with different dataset sizes
        for n_samples in [100, 500, 1000]:
            print(f"   📊 Testing with {n_samples} samples...")
            
            # Generate synthetic data
            X = np.random.randn(n_samples, 10)
            y = np.random.randn(n_samples)
            
            # Test linear model
            start_time = time.time()
            linear_model = create_linear_model()
            linear_metrics = linear_model.fit(X, y)
            linear_time = time.time() - start_time
            
            # Test random forest
            start_time = time.time()
            rf_model = create_rf_model()
            rf_metrics = rf_model.fit(X, y)
            rf_time = time.time() - start_time
            
            print(f"      • Linear Model: {linear_time:.3f}s")
            print(f"      • Random Forest: {rf_time:.3f}s")
            
            results.append({
                'n_samples': n_samples,
                'linear_time': linear_time,
                'rf_time': rf_time
            })
        
        print("   ✅ Model training performance test completed")
        return True
        
    except Exception as e:
        print(f"   ❌ Model training test failed: {e}")
        return False


def test_memory_usage():
    """Test memory usage with various operations."""
    print("\n🧪 Testing Memory Usage...")
    
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"   📊 Initial memory usage: {initial_memory:.1f} MB")
        
        # Test memory usage during large operations
        from qemlflow.core.data_processing import process_smiles
        from qemlflow.core.preprocessing import extract_basic_molecular_descriptors
        
        # Large SMILES processing
        large_smiles = ["CCO", "c1ccccc1", "CC(=O)O"] * 200
        processed = process_smiles(large_smiles)
        
        mid_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"   📊 Memory after SMILES processing: {mid_memory:.1f} MB")
        
        # Feature extraction
        features = extract_basic_molecular_descriptors(large_smiles[:50])
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"   📊 Memory after feature extraction: {final_memory:.1f} MB")
        
        memory_increase = final_memory - initial_memory
        print(f"   📊 Total memory increase: {memory_increase:.1f} MB")
        
        # Memory should not increase dramatically (< 500MB for these tests)
        return memory_increase < 500
        
    except Exception as e:
        print(f"   ❌ Memory usage test failed: {e}")
        return False


def test_error_handling_robustness():
    """Test robustness with invalid inputs and edge cases."""
    print("\n🧪 Testing Error Handling Robustness...")
    
    test_cases = [
        ("Empty SMILES list", []),
        ("Invalid SMILES", ["invalid", "also_invalid", "still_invalid"]),
        ("Mixed valid/invalid", ["CCO", "invalid", "c1ccccc1", "bad_smiles"]),
        ("Very long SMILES", ["C" * 1000]),
        ("None input", None),
        ("Non-string input", [123, 456, 789]),
    ]
    
    passed_tests = 0
    
    for test_name, test_input in test_cases:
        try:
            from qemlflow.core.data_processing import process_smiles
            
            print(f"   🧪 Testing: {test_name}")
            
            if test_input is None:
                # This should raise an exception or handle gracefully
                try:
                    result = process_smiles(test_input)
                    print(f"      ⚠️  Unexpectedly handled None input: {result}")
                except (TypeError, AttributeError):
                    print(f"      ✅ Properly rejected None input")
                    passed_tests += 1
            else:
                result = process_smiles(test_input)
                print(f"      ✅ Handled gracefully: {len(result) if result else 0} results")
                passed_tests += 1
                
        except Exception as e:
            print(f"      ❌ Failed with error: {e}")
    
    success_rate = (passed_tests / len(test_cases)) * 100
    print(f"   📊 Error handling success rate: {success_rate:.1f}%")
    
    return success_rate >= 80


def run_performance_tests():
    """Run all performance and stress tests."""
    print("🚀 QeMLflow Performance and Stress Testing")
    print("=" * 50)
    
    tests = [
        ("Large Dataset Processing", test_large_dataset_processing),
        ("Feature Extraction Performance", test_feature_extraction_performance),
        ("Model Training Performance", test_model_training_performance),
        ("Memory Usage", test_memory_usage),
        ("Error Handling Robustness", test_error_handling_robustness),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"   ❌ {test_name}: Critical failure - {e}")
            results.append((test_name, False))
    
    # Summary
    passed = sum(1 for _, success in results if success)
    total = len(results)
    success_rate = (passed / total) * 100
    
    print("\n" + "=" * 50)
    print("📊 PERFORMANCE TEST RESULTS:")
    print(f"   ✅ Passed: {passed}/{total}")
    print(f"   📈 Success Rate: {success_rate:.1f}%")
    
    if success_rate == 100:
        print("   🎉 All performance tests passed! QeMLflow is production-ready.")
    elif success_rate >= 80:
        print("   ✅ Most tests passed! QeMLflow performs well under stress.")
    else:
        print("   ⚠️  Some performance issues detected.")
    
    return results


if __name__ == "__main__":
    run_performance_tests()
