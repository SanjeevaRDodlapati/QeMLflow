"""
Comprehensive test suite for experiment tracking functionality.

This module tests all aspects of the experiment tracking system including:
- Experiment creation and management
- Parameter tracking and validation
- Data versioning and integrity
- Result validation and comparison
- Integration with environment determinism
- Performance benchmarks
"""

import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
import yaml

try:
    from qemlflow.reproducibility.experiment_tracking import (
        ExperimentParameter,
        ExperimentRecord,
        ExperimentTracker,
        DataVersion
    )
    from qemlflow.reproducibility.result_validation import ResultValidator
    from qemlflow.reproducibility.environment import get_environment_manager
except ImportError:
    # For development/testing when package is not installed
    import sys
    from pathlib import Path
    src_path = Path(__file__).parent.parent.parent / "src"
    sys.path.insert(0, str(src_path))
    
    from qemlflow.reproducibility.experiment_tracking import (
        ExperimentParameter,
        ExperimentRecord,
        ExperimentTracker,
        DataVersion
    )
    from qemlflow.reproducibility.result_validation import ResultValidator
    from qemlflow.reproducibility.environment import get_environment_manager


class TestExperimentCreation:
    """Test experiment creation and basic functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.tracker = ExperimentTracker(base_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_experiment_creation(self):
        """Test basic experiment creation."""
        exp = self.tracker.create_experiment(
            name="test_experiment",
            description="Test experiment for validation"
        )
        
        assert exp is not None
        assert exp.name == "test_experiment"
        assert exp.description == "Test experiment for validation"
        assert exp.id is not None
        assert exp.start_time is not None
        assert exp.status == "running"
    
    def test_experiment_with_tags(self):
        """Test experiment creation with tags."""
        tags = ["unit_test", "validation", "automated"]
        exp = self.tracker.create_experiment(
            name="tagged_experiment",
            description="Experiment with tags",
            tags=tags
        )
        
        assert exp.tags == tags
        assert "unit_test" in exp.tags
    
    def test_experiment_with_metadata(self):
        """Test experiment creation with custom metadata."""
        metadata = {
            "project": "QeMLflow",
            "researcher": "Test User",
            "hypothesis": "Testing experiment tracking"
        }
        
        exp = self.tracker.create_experiment(
            name="metadata_experiment",
            description="Experiment with metadata",
            metadata=metadata
        )
        
        assert exp.metadata["project"] == "QeMLflow"
        assert exp.metadata["researcher"] == "Test User"
    
    def test_experiment_finish(self):
        """Test experiment finishing."""
        exp = self.tracker.create_experiment(
            name="finish_test",
            description="Test experiment finishing"
        )
        
        # Add some data
        exp.log_parameter("test_param", 42)
        exp.log_metric("test_metric", 0.95)
        
        # Finish experiment
        exp.finish()
        
        assert exp.status == "completed"
        assert exp.end_time is not None
        assert exp.duration > 0
    
    def test_experiment_persistence(self):
        """Test experiment data persistence."""
        exp = self.tracker.create_experiment(
            name="persistence_test",
            description="Test experiment persistence"
        )
        
        exp.log_parameter("persist_param", "test_value")
        exp.log_metric("persist_metric", 0.85)
        exp.finish()
        
        # Load experiment from disk
        loaded_exp = self.tracker.load_experiment(exp.id)
        
        assert loaded_exp.id == exp.id
        assert loaded_exp.name == exp.name
        assert loaded_exp.get_parameter("persist_param") == "test_value"
        assert loaded_exp.get_metric("persist_metric") == 0.85


class TestParameterTracking:
    """Test parameter tracking functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.tracker = ExperimentTracker(base_dir=self.temp_dir)
        self.exp = self.tracker.create_experiment(
            name="param_test",
            description="Parameter tracking test"
        )
    
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_parameter_logging(self):
        """Test basic parameter logging."""
        self.exp.log_parameter("learning_rate", 0.01)
        self.exp.log_parameter("batch_size", 32)
        self.exp.log_parameter("model_type", "transformer")
        
        assert self.exp.get_parameter("learning_rate") == 0.01
        assert self.exp.get_parameter("batch_size") == 32
        assert self.exp.get_parameter("model_type") == "transformer"
    
    def test_parameter_types(self):
        """Test different parameter types."""
        self.exp.log_parameter("int_param", 42)
        self.exp.log_parameter("float_param", 3.14)
        self.exp.log_parameter("str_param", "test")
        self.exp.log_parameter("bool_param", True)
        self.exp.log_parameter("list_param", [1, 2, 3])
        self.exp.log_parameter("dict_param", {"key": "value"})
        
        assert isinstance(self.exp.get_parameter("int_param"), int)
        assert isinstance(self.exp.get_parameter("float_param"), float)
        assert isinstance(self.exp.get_parameter("str_param"), str)
        assert isinstance(self.exp.get_parameter("bool_param"), bool)
        assert isinstance(self.exp.get_parameter("list_param"), list)
        assert isinstance(self.exp.get_parameter("dict_param"), dict)
    
    def test_parameter_validation(self):
        """Test parameter validation."""
        # Test with validation enabled
        param = ExperimentParameter(
            name="validated_param",
            value=0.5,
            param_type="float",
            constraints={"min": 0.0, "max": 1.0}
        )
        
        self.exp.log_parameter_object(param)
        
        # Test invalid parameter
        with pytest.raises(ValueError):
            invalid_param = ExperimentParameter(
                name="invalid_param",
                value=1.5,
                param_type="float",
                constraints={"min": 0.0, "max": 1.0}
            )
            self.exp.log_parameter_object(invalid_param)
    
    def test_parameter_versioning(self):
        """Test parameter versioning."""
        # Log initial parameter
        self.exp.log_parameter("versioned_param", 1.0, version="v1")
        
        # Update parameter
        self.exp.log_parameter("versioned_param", 2.0, version="v2")
        
        # Check versions
        assert self.exp.get_parameter("versioned_param", version="v1") == 1.0
        assert self.exp.get_parameter("versioned_param", version="v2") == 2.0
        assert self.exp.get_parameter("versioned_param") == 2.0  # Latest version
    
    def test_sensitive_parameters(self):
        """Test sensitive parameter handling."""
        # These should be excluded or hashed
        self.exp.log_parameter("password", "secret123")
        self.exp.log_parameter("api_key", "abc123xyz")
        self.exp.log_parameter("email", "user@example.com")
        
        # Check that sensitive parameters are handled properly
        stored_params = self.exp.get_all_parameters()
        
        # Password should be excluded
        assert "password" not in stored_params
        
        # API key should be excluded
        assert "api_key" not in stored_params
        
        # Email should be hashed
        if "email" in stored_params:
            assert stored_params["email"] != "user@example.com"


class TestDataVersioning:
    """Test data versioning functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.tracker = ExperimentTracker(base_dir=self.temp_dir)
        self.exp = self.tracker.create_experiment(
            name="data_test",
            description="Data versioning test"
        )
    
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_data_logging(self):
        """Test basic data logging."""
        data = np.random.rand(100, 10)
        self.exp.log_data("input_data", data)
        
        # Retrieve data
        retrieved_data = self.exp.get_data("input_data")
        np.testing.assert_array_equal(data, retrieved_data)
    
    def test_data_fingerprinting(self):
        """Test data fingerprinting."""
        data1 = np.random.rand(100, 10)
        data2 = np.random.rand(100, 10)
        
        self.exp.log_data("data1", data1)
        self.exp.log_data("data2", data2)
        
        # Get fingerprints
        fp1 = self.exp.get_data_fingerprint("data1")
        fp2 = self.exp.get_data_fingerprint("data2")
        
        assert fp1 != fp2
        assert len(fp1) == 64  # SHA256 hash length
    
    def test_data_versioning(self):
        """Test data versioning."""
        # Log initial data
        data_v1 = np.random.rand(50, 5)
        self.exp.log_data("versioned_data", data_v1, version="v1")
        
        # Log updated data
        data_v2 = np.random.rand(60, 5)
        self.exp.log_data("versioned_data", data_v2, version="v2")
        
        # Check versions
        retrieved_v1 = self.exp.get_data("versioned_data", version="v1")
        retrieved_v2 = self.exp.get_data("versioned_data", version="v2")
        
        np.testing.assert_array_equal(data_v1, retrieved_v1)
        np.testing.assert_array_equal(data_v2, retrieved_v2)
    
    def test_data_lineage(self):
        """Test data lineage tracking."""
        # Create data with lineage
        input_data = np.random.rand(100, 10)
        self.exp.log_data("input", input_data)
        
        # Transform data
        processed_data = input_data * 2
        self.exp.log_data(
            "processed", 
            processed_data,
            lineage=["input"]
        )
        
        # Further transform
        final_data = processed_data + 1
        self.exp.log_data(
            "final",
            final_data,
            lineage=["processed"]
        )
        
        # Check lineage
        lineage = self.exp.get_data_lineage("final")
        assert "processed" in lineage
        assert "input" in lineage  # Should trace back to original
    
    def test_large_data_handling(self):
        """Test handling of large data arrays."""
        # Create large data (>10MB)
        large_data = np.random.rand(1000, 1000)
        
        start_time = time.time()
        self.exp.log_data("large_data", large_data)
        log_time = time.time() - start_time
        
        start_time = time.time()
        retrieved_data = self.exp.get_data("large_data")
        retrieve_time = time.time() - start_time
        
        # Verify data integrity
        np.testing.assert_array_equal(large_data, retrieved_data)
        
        # Performance should be reasonable
        assert log_time < 10.0  # Should complete within 10 seconds
        assert retrieve_time < 5.0  # Should retrieve within 5 seconds


class TestResultValidation:
    """Test result validation and comparison."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.tracker = ExperimentTracker(base_dir=self.temp_dir)
        self.validator = ResultValidator()
        self.exp = self.tracker.create_experiment(
            name="result_test",
            description="Result validation test"
        )
    
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_metric_logging(self):
        """Test metric logging."""
        self.exp.log_metric("accuracy", 0.95)
        self.exp.log_metric("loss", 0.05)
        self.exp.log_metric("precision", 0.92)
        
        assert self.exp.get_metric("accuracy") == 0.95
        assert self.exp.get_metric("loss") == 0.05
        assert self.exp.get_metric("precision") == 0.92
    
    def test_result_validation(self):
        """Test result validation against baseline."""
        # Log baseline results
        baseline = {
            "accuracy": 0.90,
            "precision": 0.88,
            "recall": 0.92
        }
        
        for metric, value in baseline.items():
            self.exp.log_metric(f"baseline_{metric}", value)
        
        # Log current results
        current = {
            "accuracy": 0.91,
            "precision": 0.89,
            "recall": 0.91
        }
        
        for metric, value in current.items():
            self.exp.log_metric(f"current_{metric}", value)
        
        # Validate results
        validation_result = self.validator.validate_results(
            current_results=current,
            baseline_results=baseline,
            tolerance=0.05
        )
        
        assert validation_result["valid"]
        assert validation_result["accuracy"]["within_tolerance"]
    
    def test_statistical_validation(self):
        """Test statistical validation of results."""
        # Generate repeated measurements
        measurements1 = [0.90, 0.91, 0.89, 0.92, 0.90]
        measurements2 = [0.85, 0.86, 0.84, 0.87, 0.85]
        
        # Test statistical significance
        significance_test = self.validator.test_statistical_significance(
            measurements1, measurements2
        )
        
        assert "p_value" in significance_test
        assert "significant" in significance_test
        assert "test_statistic" in significance_test
    
    def test_result_comparison(self):
        """Test result comparison between experiments."""
        # Create second experiment
        exp2 = self.tracker.create_experiment(
            name="comparison_test",
            description="Second experiment for comparison"
        )
        
        # Log different results
        self.exp.log_metric("accuracy", 0.90)
        self.exp.log_metric("f1_score", 0.88)
        
        exp2.log_metric("accuracy", 0.92)
        exp2.log_metric("f1_score", 0.90)
        
        # Compare experiments
        comparison = self.tracker.compare_experiments([self.exp.id, exp2.id])
        
        assert "metrics" in comparison
        assert "accuracy" in comparison["metrics"]
        assert len(comparison["experiments"]) == 2
    
    def test_performance_regression_detection(self):
        """Test performance regression detection."""
        # Simulate historical performance
        historical_accuracy = [0.89, 0.90, 0.91, 0.90, 0.92]
        
        for i, acc in enumerate(historical_accuracy):
            self.exp.log_metric(f"historical_accuracy_{i}", acc)
        
        # Current performance
        current_accuracy = 0.85  # Regression
        self.exp.log_metric("current_accuracy", current_accuracy)
        
        # Detect regression
        regression_test = self.validator.detect_regression(
            historical_values=historical_accuracy,
            current_value=current_accuracy,
            threshold=0.05
        )
        
        assert regression_test["regression_detected"]
        assert regression_test["performance_drop"] > 0.05


class TestExperimentComparison:
    """Test experiment comparison functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.tracker = ExperimentTracker(base_dir=self.temp_dir)
        self.experiments = []
        
        # Create multiple experiments
        for i in range(3):
            exp = self.tracker.create_experiment(
                name=f"comparison_exp_{i}",
                description=f"Experiment {i} for comparison testing"
            )
            
            # Log parameters
            exp.log_parameter("learning_rate", 0.01 * (i + 1))
            exp.log_parameter("batch_size", 32 * (i + 1))
            exp.log_parameter("model_type", f"model_{i}")
            
            # Log metrics
            exp.log_metric("accuracy", 0.8 + i * 0.05)
            exp.log_metric("loss", 0.5 - i * 0.1)
            exp.log_metric("f1_score", 0.75 + i * 0.03)
            
            exp.finish()
            self.experiments.append(exp)
    
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_basic_comparison(self):
        """Test basic experiment comparison."""
        exp_ids = [exp.id for exp in self.experiments]
        comparison = self.tracker.compare_experiments(exp_ids)
        
        assert "experiments" in comparison
        assert "parameters" in comparison
        assert "metrics" in comparison
        assert len(comparison["experiments"]) == 3
    
    def test_parameter_comparison(self):
        """Test parameter comparison across experiments."""
        exp_ids = [exp.id for exp in self.experiments]
        comparison = self.tracker.compare_experiments(exp_ids)
        
        params = comparison["parameters"]
        
        # Check learning rate comparison
        assert "learning_rate" in params
        lr_values = params["learning_rate"]["values"]
        assert len(lr_values) == 3
        assert lr_values[0] == 0.01
        assert lr_values[1] == 0.02
        assert lr_values[2] == 0.03
    
    def test_metric_comparison(self):
        """Test metric comparison across experiments."""
        exp_ids = [exp.id for exp in self.experiments]
        comparison = self.tracker.compare_experiments(exp_ids)
        
        metrics = comparison["metrics"]
        
        # Check accuracy comparison
        assert "accuracy" in metrics
        acc_values = metrics["accuracy"]["values"]
        assert len(acc_values) == 3
        assert acc_values[0] == 0.8
        assert acc_values[1] == 0.85
        assert acc_values[2] == 0.9
    
    def test_best_experiment_selection(self):
        """Test automatic best experiment selection."""
        exp_ids = [exp.id for exp in self.experiments]
        
        # Select best by accuracy
        best_exp = self.tracker.get_best_experiment(
            experiment_ids=exp_ids,
            metric="accuracy",
            objective="maximize"
        )
        
        assert best_exp.id == self.experiments[2].id  # Highest accuracy
        
        # Select best by loss
        best_exp_loss = self.tracker.get_best_experiment(
            experiment_ids=exp_ids,
            metric="loss",
            objective="minimize"
        )
        
        assert best_exp_loss.id == self.experiments[2].id  # Lowest loss
    
    def test_pareto_frontier_analysis(self):
        """Test Pareto frontier analysis for multi-objective optimization."""
        exp_ids = [exp.id for exp in self.experiments]
        
        pareto_frontier = self.tracker.get_pareto_frontier(
            experiment_ids=exp_ids,
            objectives=["accuracy", "f1_score"],
            directions=["maximize", "maximize"]
        )
        
        assert len(pareto_frontier) >= 1
        assert all(exp_id in exp_ids for exp_id in pareto_frontier)


class TestIntegrationWithEnvironmentDeterminism:
    """Test integration with environment determinism."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.tracker = ExperimentTracker(base_dir=self.temp_dir)
        self.env_manager = get_environment_manager()
    
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_environment_capture_integration(self):
        """Test integration with environment capture."""
        exp = self.tracker.create_experiment(
            name="env_integration_test",
            description="Test environment capture integration"
        )
        
        # Capture environment
        env_fingerprint = self.env_manager.capture_environment()
        exp.add_environment_info(env_fingerprint.to_dict())
        
        # Verify environment info is stored
        env_info = exp.get_environment_info()
        assert env_info is not None
        assert "python_version" in env_info
        assert "packages" in env_info
        assert "system_info" in env_info
    
    def test_reproducibility_validation(self):
        """Test reproducibility validation."""
        exp = self.tracker.create_experiment(
            name="reproducibility_test",
            description="Test reproducibility validation"
        )
        
        # Capture environment and parameters
        env_fingerprint = self.env_manager.capture_environment()
        exp.add_environment_info(env_fingerprint.to_dict())
        
        exp.log_parameter("seed", 42)
        exp.log_parameter("algorithm", "test_algo")
        
        # Simulate results
        results = np.random.seed(42)
        test_results = np.random.rand(100)
        exp.log_data("results", test_results)
        
        exp.finish()
        
        # Verify reproducibility info
        repro_info = exp.get_reproducibility_info()
        assert repro_info["environment_captured"]
        assert repro_info["parameters_logged"]
        assert repro_info["results_stored"]


class TestPerformanceBenchmarks:
    """Performance benchmarks for experiment tracking."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.tracker = ExperimentTracker(base_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @pytest.mark.benchmark
    def test_experiment_creation_performance(self, benchmark):
        """Benchmark experiment creation performance."""
        def create_experiment():
            return self.tracker.create_experiment(
                name="perf_test",
                description="Performance test experiment"
            )
        
        result = benchmark(create_experiment)
        assert result is not None
    
    @pytest.mark.benchmark
    def test_parameter_logging_performance(self, benchmark):
        """Benchmark parameter logging performance."""
        exp = self.tracker.create_experiment(
            name="param_perf_test",
            description="Parameter performance test"
        )
        
        def log_parameters():
            for i in range(100):
                exp.log_parameter(f"param_{i}", i * 0.01)
        
        benchmark(log_parameters)
    
    @pytest.mark.benchmark
    def test_data_logging_performance(self, benchmark):
        """Benchmark data logging performance."""
        exp = self.tracker.create_experiment(
            name="data_perf_test",
            description="Data performance test"
        )
        
        data = np.random.rand(1000, 100)
        
        def log_data():
            exp.log_data("performance_data", data)
        
        benchmark(log_data)
    
    @pytest.mark.benchmark
    def test_experiment_comparison_performance(self, benchmark):
        """Benchmark experiment comparison performance."""
        # Create multiple experiments
        experiments = []
        for i in range(10):
            exp = self.tracker.create_experiment(
                name=f"perf_exp_{i}",
                description=f"Performance test experiment {i}"
            )
            
            # Add data
            for j in range(50):
                exp.log_parameter(f"param_{j}", j * 0.01)
                exp.log_metric(f"metric_{j}", j * 0.1)
            
            exp.finish()
            experiments.append(exp)
        
        exp_ids = [exp.id for exp in experiments]
        
        def compare_experiments():
            return self.tracker.compare_experiments(exp_ids)
        
        result = benchmark(compare_experiments)
        assert result is not None
        assert len(result["experiments"]) == 10
    
    def test_performance_requirements(self):
        """Test that performance meets minimum requirements."""
        # Test experiment creation time
        start_time = time.time()
        exp = self.tracker.create_experiment(
            name="requirement_test",
            description="Performance requirement test"
        )
        creation_time = time.time() - start_time
        
        assert creation_time < 1.0  # Should create experiment in < 1 second
        
        # Test parameter logging time
        start_time = time.time()
        for i in range(1000):
            exp.log_parameter(f"param_{i}", i)
        param_time = time.time() - start_time
        
        assert param_time < 5.0  # Should log 1000 parameters in < 5 seconds
        
        # Test data logging time
        data = np.random.rand(10000, 10)
        start_time = time.time()
        exp.log_data("large_data", data)
        data_time = time.time() - start_time
        
        assert data_time < 10.0  # Should log large data in < 10 seconds


class TestConfigurationValidation:
    """Test configuration file validation."""
    
    def test_config_file_exists(self):
        """Test that configuration file exists and is valid."""
        config_path = Path("config/experiment_tracking.yml")
        assert config_path.exists(), "Configuration file not found"
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # Test required sections
        required_sections = [
            "storage", "metadata", "parameters", 
            "data_versioning", "results", "reproducibility"
        ]
        
        for section in required_sections:
            assert section in config, f"Missing configuration section: {section}"
    
    def test_config_values_valid(self):
        """Test that configuration values are valid."""
        config_path = Path("config/experiment_tracking.yml")
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # Test storage configuration
        storage = config["storage"]
        assert storage["backend"] in ["local", "s3", "gcs", "azure"]
        assert isinstance(storage["compression"]["enabled"], bool)
        assert storage["retention"]["default_days"] > 0
        
        # Test reproducibility configuration
        repro = config["reproducibility"]
        assert isinstance(repro["seed_management"]["enabled"], bool)
        assert isinstance(repro["environment_capture"]["enabled"], bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
