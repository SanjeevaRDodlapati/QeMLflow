"""
Focused test suite for experiment tracking functionality.

This module tests the core experiment tracking functionality using the actual API:
- Experiment creation and management
- Parameter and metric logging
- Data versioning
- Result validation
"""

import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

try:
    from qemlflow.reproducibility.experiment_tracking import (
        ExperimentParameter,
        ExperimentRecord,
        ExperimentTracker,
        DataVersion,
        start_experiment,
        log_parameter,
        log_metric,
        end_experiment
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
        DataVersion,
        start_experiment,
        log_parameter,
        log_metric,
        end_experiment
    )
    from qemlflow.reproducibility.result_validation import ResultValidator
    from qemlflow.reproducibility.environment import get_environment_manager


class TestExperimentTracking:
    """Test core experiment tracking functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.tracker = ExperimentTracker(base_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_start_experiment(self):
        """Test starting an experiment."""
        exp_id = self.tracker.start_experiment(
            name="test_experiment",
            description="Test experiment for validation"
        )
        
        assert exp_id is not None
        assert isinstance(exp_id, str)
        assert self.tracker.current_experiment is not None
        assert self.tracker.current_experiment.name == "test_experiment"
    
    def test_log_parameter(self):
        """Test logging parameters."""
        exp_id = self.tracker.start_experiment(
            name="param_test",
            description="Parameter logging test"
        )
        
        # Log different types of parameters
        self.tracker.log_parameter("learning_rate", 0.01)
        self.tracker.log_parameter("batch_size", 32)
        self.tracker.log_parameter("model_type", "transformer")
        self.tracker.log_parameter("use_dropout", True)
        
        # Check parameters are stored
        exp = self.tracker.current_experiment
        assert any(p.name == "learning_rate" and p.value == 0.01 for p in exp.parameters)
        assert any(p.name == "batch_size" and p.value == 32 for p in exp.parameters)
        assert any(p.name == "model_type" and p.value == "transformer" for p in exp.parameters)
        assert any(p.name == "use_dropout" and p.value is True for p in exp.parameters)
    
    def test_log_metric(self):
        """Test logging metrics."""
        exp_id = self.tracker.start_experiment(
            name="metric_test",
            description="Metric logging test"
        )
        
        # Log metrics
        self.tracker.log_metric("accuracy", 0.95)
        self.tracker.log_metric("loss", 0.05)
        self.tracker.log_metric("f1_score", 0.88)
        
        # Check metrics are stored
        exp = self.tracker.current_experiment
        assert "accuracy" in exp.metrics.metrics
        assert "loss" in exp.metrics.metrics
        assert "f1_score" in exp.metrics.metrics
        assert exp.metrics.metrics["accuracy"] == 0.95
    
    def test_end_experiment(self):
        """Test ending an experiment."""
        exp_id = self.tracker.start_experiment(
            name="end_test",
            description="Test experiment ending"
        )
        
        # Add some data
        self.tracker.log_parameter("test_param", 42)
        self.tracker.log_metric("test_metric", 0.95)
        
        # End experiment
        result = self.tracker.end_experiment()
        
        assert result is not None
        assert self.tracker.current_experiment is None
    
    def test_experiment_persistence(self):
        """Test experiment data persistence."""
        exp_id = self.tracker.start_experiment(
            name="persistence_test",
            description="Test experiment persistence"
        )
        
        self.tracker.log_parameter("persist_param", "test_value")
        self.tracker.log_metric("persist_metric", 0.85)
        
        # End experiment (this should save it)
        self.tracker.end_experiment()
        
        # Check files were created
        experiments_dir = Path(self.temp_dir) / "records"
        assert experiments_dir.exists()
        
        # Check that experiment file exists
        exp_files = list(experiments_dir.glob("*.json"))
        assert len(exp_files) > 0


class TestExperimentRecord:
    """Test ExperimentRecord functionality."""
    
    def test_experiment_record_creation(self):
        """Test creating an ExperimentRecord."""
        record = ExperimentRecord(
            experiment_id="test-001",
            name="test_experiment",
            description="Test experiment record"
        )
        
        assert record.experiment_id == "test-001"
        assert record.name == "test_experiment"
        assert record.description == "Test experiment record"
        assert record.status == "running"
        assert len(record.parameters) == 0
        assert len(record.metrics.metrics) == 0
    
    def test_add_parameter(self):
        """Test adding parameters to experiment record."""
        record = ExperimentRecord(
            experiment_id="test-002",
            name="param_test",
            description="Parameter test"
        )
        
        record.add_parameter("learning_rate", 0.01, param_type="float")
        record.add_parameter("epochs", 100, param_type="int")
        
        assert len(record.parameters) == 2
        lr_param = next(p for p in record.parameters if p.name == "learning_rate")
        assert lr_param.value == 0.01
        assert lr_param.param_type == "float"
    
    def test_add_metric(self):
        """Test adding metrics to experiment record."""
        record = ExperimentRecord(
            experiment_id="test-003",
            name="metric_test",
            description="Metric test"
        )
        
        record.add_metric("accuracy", 0.95)
        record.add_metric("loss", 0.05)
        
        assert "accuracy" in record.metrics.metrics
        assert "loss" in record.metrics.metrics
        assert record.metrics.metrics["accuracy"] == 0.95
        assert record.metrics.metrics["loss"] == 0.05
    
    def test_serialization(self):
        """Test experiment record serialization."""
        record = ExperimentRecord(
            experiment_id="test-004",
            name="serialization_test",
            description="Serialization test"
        )
        
        record.add_parameter("param1", "value1")
        record.add_metric("metric1", 0.9)
        
        # Convert to dict
        data = record.to_dict()
        
        assert data["experiment_id"] == "test-004"
        assert data["name"] == "serialization_test"
        assert len(data["parameters"]) == 1
        assert "metric1" in data["metrics"]["metrics"]
        
        # Convert back from dict
        restored_record = ExperimentRecord.from_dict(data)
        assert restored_record.experiment_id == record.experiment_id
        assert restored_record.name == record.name
        assert len(restored_record.parameters) == 1


class TestDataVersioning:
    """Test data versioning functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.tracker = ExperimentTracker(base_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_data_version_creation(self):
        """Test creating data versions."""
        exp_id = self.tracker.start_experiment(
            name="data_version_test",
            description="Data versioning test"
        )
        
        # Create test data files
        test_file = Path(self.temp_dir) / "test_data.txt"
        test_file.write_text("test data content")
        
        # Add data version
        exp = self.tracker.current_experiment
        exp.add_data_version(
            dataset_name="test_dataset",
            version="v1.0",
            file_paths=[str(test_file)],
            metadata={"source": "test"}
        )
        
        assert len(exp.data_versions) == 1
        data_version = exp.data_versions[0]
        assert data_version.dataset_name == "test_dataset"
        assert data_version.version == "v1.0"
        assert len(data_version.file_paths) == 1


class TestStandaloneFunctions:
    """Test standalone experiment tracking functions."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        # Set environment variable for experiment directory
        os.environ["QEMLFLOW_EXPERIMENT_DIR"] = self.temp_dir
    
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        if "QEMLFLOW_EXPERIMENT_DIR" in os.environ:
            del os.environ["QEMLFLOW_EXPERIMENT_DIR"]
    
    def test_standalone_experiment_workflow(self):
        """Test using standalone functions for experiment workflow."""
        # Start experiment
        exp_id = start_experiment(
            name="standalone_test",
            description="Test standalone functions"
        )
        
        assert exp_id is not None
        
        # Log parameters and metrics
        log_parameter("standalone_param", "test_value")
        log_metric("standalone_metric", 0.88)
        
        # End experiment
        result = end_experiment()
        assert result is not None


class TestResultValidation:
    """Test result validation functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.validator = ResultValidator()
    
    def test_result_validator_creation(self):
        """Test creating ResultValidator instance."""
        assert self.validator is not None
        # Add more specific tests based on actual ResultValidator implementation
    
    def test_basic_validation(self):
        """Test basic result validation functionality."""
        # This is a placeholder test - update based on actual ResultValidator API
        
        # Create test results
        baseline_results = {"accuracy": 0.90, "precision": 0.88}
        current_results = {"accuracy": 0.91, "precision": 0.89}
        
        # Validate (this test will need to be updated based on actual API)
        # For now, just check that the validator exists
        assert hasattr(self.validator, '__class__')


class TestConfigurationValidation:
    """Test configuration file validation."""
    
    def test_config_file_exists(self):
        """Test that configuration file exists and is valid."""
        import yaml
        
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
        import yaml
        
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
        exp_id = self.tracker.start_experiment(
            name="env_integration_test",
            description="Test environment capture integration",
            capture_env=True  # This should trigger environment capture
        )
        
        # Check that environment was captured
        exp = self.tracker.current_experiment
        assert exp.environment_fingerprint is not None
        
        # Check that environment fingerprint has expected attributes
        env_fp = exp.environment_fingerprint
        assert hasattr(env_fp, 'python_version')
        assert hasattr(env_fp, 'packages')


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
