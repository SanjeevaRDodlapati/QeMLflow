"""
Tests for Validation Framework Module

This module provides comprehensive tests for the validation framework including
cross-validation, benchmark testing, validation reporting, and continuous
validation functionality.
"""

import json
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Import the modules to test
from qemlflow.reproducibility.validation_framework import (
    ValidationResult,
    BenchmarkResult,
    ValidationReport,
    CrossValidator,
    BenchmarkTester,
    StatisticalValidator,
    ValidationFramework,
    ContinuousValidator,
    validate_model,
    run_benchmark_test,
    generate_validation_report,
    get_validation_framework
)

class TestValidationResult:
    """Test ValidationResult class."""

    def test_validation_result_creation(self):
        """Test ValidationResult creation."""
        result = ValidationResult(
            validation_type="cross_validation",
            method="kfold",
            n_folds=5
        )

        assert result.validation_type == "cross_validation"
        assert result.method == "kfold"
        assert result.n_folds == 5
        assert result.status == "pending"
        assert len(result.validation_id) > 0
        assert len(result.timestamp) > 0

    def test_calculate_statistics(self):
        """Test statistics calculation."""
        result = ValidationResult()
        result.scores = [0.8, 0.85, 0.9, 0.75, 0.88]
        result.calculate_statistics()

        assert abs(result.mean_score - 0.836) < 0.01
        assert result.std_score > 0
        assert result.min_score == 0.75
        assert result.max_score == 0.9

    def test_add_detailed_metric(self):
        """Test adding detailed metrics."""
        result = ValidationResult()
        result.add_detailed_metric("precision", 0.85)
        result.add_detailed_metric("recall", 0.80)

        assert result.detailed_metrics["precision"] == 0.85
        assert result.detailed_metrics["recall"] == 0.80

    def test_add_statistical_test(self):
        """Test adding statistical test results."""
        result = ValidationResult()
        test_result = {"statistic": 0.95, "p_value": 0.001}
        result.add_statistical_test("shapiro_wilk", test_result)

        assert result.statistical_tests["shapiro_wilk"] == test_result

    def test_to_dict_conversion(self):
        """Test dictionary conversion."""
        result = ValidationResult(
            validation_type="cross_validation",
            method="kfold"
        )
        result.scores = [0.8, 0.9]
        result.calculate_statistics()

        data = result.to_dict()

        assert data["validation_type"] == "cross_validation"
        assert data["method"] == "kfold"
        assert data["scores"] == [0.8, 0.9]
        assert "mean_score" in data

    def test_from_dict_creation(self):
        """Test creation from dictionary."""
        data = {
            "validation_id": "test-id",
            "validation_type": "cross_validation",
            "method": "kfold",
            "scores": [0.8, 0.9],
            "mean_score": 0.85
        }

        result = ValidationResult.from_dict(data)

        assert result.validation_id == "test-id"
        assert result.validation_type == "cross_validation"
        assert result.scores == [0.8, 0.9]
        assert result.mean_score == 0.85

class TestBenchmarkResult:
    """Test BenchmarkResult class."""

    def test_benchmark_result_creation(self):
        """Test BenchmarkResult creation."""
        result = BenchmarkResult(
            benchmark_name="test_benchmark",
            dataset_name="test_dataset",
            model_name="test_model"
        )

        assert result.benchmark_name == "test_benchmark"
        assert result.dataset_name == "test_dataset"
        assert result.model_name == "test_model"
        assert len(result.benchmark_id) > 0

    def test_add_baseline_comparison(self):
        """Test adding baseline comparisons."""
        result = BenchmarkResult()
        metrics = {"accuracy": 0.85, "precision": 0.80}
        result.add_baseline_comparison("baseline_model", metrics)

        assert result.baseline_comparisons["baseline_model"] == metrics

    def test_to_dict_conversion(self):
        """Test dictionary conversion."""
        result = BenchmarkResult(
            benchmark_name="test",
            accuracy=0.85,
            precision=0.80
        )

        data = result.to_dict()

        assert data["benchmark_name"] == "test"
        assert data["accuracy"] == 0.85
        assert data["precision"] == 0.80

class TestValidationReport:
    """Test ValidationReport class."""

    def test_validation_report_creation(self):
        """Test ValidationReport creation."""
        report = ValidationReport(
            title="Test Report",
            description="Test validation report"
        )

        assert report.title == "Test Report"
        assert report.description == "Test validation report"
        assert len(report.report_id) > 0
        assert report.report_type == "validation_report"

    def test_add_validation_result(self):
        """Test adding validation results."""
        report = ValidationReport()
        result = ValidationResult(validation_type="cross_validation")

        report.add_validation_result(result)

        assert len(report.validation_results) == 1
        assert report.validation_results[0] == result

    def test_add_benchmark_result(self):
        """Test adding benchmark results."""
        report = ValidationReport()
        result = BenchmarkResult(benchmark_name="test")

        report.add_benchmark_result(result)

        assert len(report.benchmark_results) == 1
        assert report.benchmark_results[0] == result

    def test_calculate_summary(self):
        """Test summary calculation."""
        report = ValidationReport()

        # Add validation results
        result1 = ValidationResult(mean_score=0.8)
        result2 = ValidationResult(mean_score=0.9)
        report.add_validation_result(result1)
        report.add_validation_result(result2)

        # Add benchmark results
        bench1 = BenchmarkResult(accuracy=0.85)
        bench2 = BenchmarkResult(accuracy=0.75)
        report.add_benchmark_result(bench1)
        report.add_benchmark_result(bench2)

        report.calculate_summary()

        assert report.summary["total_validations"] == 2
        assert report.summary["total_benchmarks"] == 2
        assert abs(report.summary["mean_validation_score"] - 0.85) < 0.01
        assert abs(report.summary["mean_benchmark_accuracy"] - 0.8) < 0.01

    def test_generate_recommendations(self):
        """Test recommendation generation."""
        report = ValidationReport()

        # Add results with high variance
        result = ValidationResult(std_score=0.15)
        report.add_validation_result(result)

        # Add low-performance benchmark
        bench = BenchmarkResult(accuracy=0.6)
        report.add_benchmark_result(bench)

        report.generate_recommendations()

        assert len(report.recommendations) > 0
        assert any("variance" in rec.lower() for rec in report.recommendations)
        assert any("performance" in rec.lower() for rec in report.recommendations)

class TestCrossValidator:
    """Test CrossValidator class."""

    def setup_method(self):
        """Setup test data."""
        self.X, self.y = make_classification(
            n_samples=100, n_features=10, n_classes=2, random_state=42
        )
        self.model = LogisticRegression(random_state=42)
        self.validator = CrossValidator()

    @patch('qemlflow.reproducibility.validation_framework.log_audit_event')
    def test_kfold_validation(self, mock_log):
        """Test k-fold cross-validation."""
        result = self.validator.validate(
            self.X, self.y, self.model,
            cv_method="kfold",
            n_folds=3,
            scoring="accuracy"
        )

        assert result.validation_type == "cross_validation"
        assert result.method == "kfold"
        assert result.n_folds == 3
        assert result.status == "completed"
        assert len(result.scores) == 3
        assert result.mean_score > 0
        assert result.std_score >= 0

        # Check audit logging
        assert mock_log.call_count >= 2  # start and completion

    @patch('qemlflow.reproducibility.validation_framework.log_audit_event')
    def test_stratified_validation(self, mock_log):
        """Test stratified cross-validation."""
        result = self.validator.validate(
            self.X, self.y, self.model,
            cv_method="stratified",
            n_folds=3
        )

        assert result.method == "stratified"
        assert result.stratify is True
        assert result.status == "completed"
        assert len(result.scores) == 3

    @patch('qemlflow.reproducibility.validation_framework.log_audit_event')
    def test_timeseries_validation(self, mock_log):
        """Test time series cross-validation."""
        result = self.validator.validate(
            self.X, self.y, self.model,
            cv_method="timeseries",
            n_folds=3
        )

        assert result.method == "timeseries"
        assert result.status == "completed"
        assert len(result.scores) == 3

    @patch('qemlflow.reproducibility.validation_framework.log_audit_event')
    def test_invalid_cv_method(self, mock_log):
        """Test invalid cross-validation method."""
        result = self.validator.validate(
            self.X, self.y, self.model,
            cv_method="invalid_method"
        )

        assert result.status == "failed"
        assert "Unknown CV method" in result.error_message

    @patch('qemlflow.reproducibility.validation_framework.log_audit_event')
    def test_detailed_metrics(self, mock_log):
        """Test detailed metrics calculation."""
        result = self.validator.validate(
            self.X, self.y, self.model,
            cv_method="kfold",
            n_folds=3
        )

        assert result.status == "completed"
        assert "accuracy_mean" in result.detailed_metrics
        assert "precision_macro_mean" in result.detailed_metrics
        assert "recall_macro_mean" in result.detailed_metrics
        assert "f1_macro_mean" in result.detailed_metrics

class TestBenchmarkTester:
    """Test BenchmarkTester class."""

    def setup_method(self):
        """Setup test data."""
        self.X, self.y = make_classification(
            n_samples=100, n_features=10, n_classes=2, random_state=42
        )
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42
        )
        self.model = LogisticRegression(random_state=42)
        self.tester = BenchmarkTester()

    @patch('qemlflow.reproducibility.validation_framework.log_audit_event')
    def test_classification_benchmark(self, mock_log):
        """Test classification benchmark."""
        result = self.tester.validate(
            self.X_train, self.X_test, self.y_train, self.y_test, self.model,
            dataset_name="test_dataset",
            model_name="logistic_regression"
        )

        assert result.dataset_name == "test_dataset"
        assert result.model_name == "logistic_regression"
        assert result.accuracy > 0
        assert result.precision >= 0
        assert result.recall >= 0
        assert result.f1_score >= 0
        assert result.training_time > 0
        assert result.inference_time > 0

    def test_regression_benchmark(self):
        """Test regression benchmark."""
        # Create regression data
        X_reg, y_reg = make_regression(n_samples=100, n_features=10, random_state=42, noise=0.1)
        X_train, X_test, y_train, y_test = train_test_split(
            X_reg, y_reg, test_size=0.3, random_state=42
        )

        model = LinearRegression()

        result = self.tester.validate(
            X_train, X_test, y_train, y_test, model,
            dataset_name="test_regression",
            model_name="linear_regression"
        )

        assert result.mse >= 0
        assert result.mae >= 0
        assert result.r2 <= 1.0  # R² can be negative for very bad models

    def test_baseline_comparison(self):
        """Test baseline model comparison."""
        # Add baseline model
        baseline_model = LogisticRegression(random_state=24)
        baseline_model.fit(self.X_train, self.y_train)
        baseline_metrics = {"accuracy": 0.8, "precision": 0.75}

        self.tester.add_baseline("baseline_lr", baseline_model, baseline_metrics)

        result = self.tester.validate(
            self.X_train, self.X_test, self.y_train, self.y_test, self.model
        )

        assert "baseline_lr" in result.baseline_comparisons
        assert "accuracy" in result.baseline_comparisons["baseline_lr"]

class TestStatisticalValidator:
    """Test StatisticalValidator class."""

    def setup_method(self):
        """Setup test data."""
        self.validator = StatisticalValidator()
        self.results = [0.8, 0.85, 0.9, 0.75, 0.88, 0.82, 0.87]

    def test_basic_statistical_validation(self):
        """Test basic statistical validation."""
        result = self.validator.validate(self.results)

        assert result.validation_type == "statistical"
        assert result.method == "statistical_tests"
        assert result.status == "completed"
        assert result.scores == self.results
        assert result.mean_score > 0
        assert result.std_score >= 0

    def test_normality_test(self):
        """Test normality test."""
        result = self.validator.validate(self.results)

        assert "shapiro_wilk" in result.statistical_tests
        shapiro_result = result.statistical_tests["shapiro_wilk"]
        assert "statistic" in shapiro_result
        assert "p_value" in shapiro_result
        assert "is_normal" in shapiro_result

    def test_comparison_with_reference(self):
        """Test comparison with reference results."""
        reference_results = [0.7, 0.75, 0.8, 0.65, 0.78]

        result = self.validator.validate(
            self.results,
            reference_results=reference_results
        )

        assert "mean_difference" in result.statistical_tests
        comparison = result.statistical_tests["mean_difference"]
        assert "mean_difference" in comparison
        assert "reference_mean" in comparison
        assert "current_mean" in comparison

class TestValidationFramework:
    """Test ValidationFramework class."""

    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.framework = ValidationFramework(
            results_dir=self.temp_dir,
            config={"cross_validation": {"default_folds": 3}}
        )

        # Create test data
        self.X, self.y = make_classification(
            n_samples=100, n_features=10, n_classes=2, random_state=42
        )
        self.model = LogisticRegression(random_state=42)

    def teardown_method(self):
        """Cleanup test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('qemlflow.reproducibility.validation_framework.log_audit_event')
    def test_run_cross_validation(self, mock_log):
        """Test running cross-validation."""
        result = self.framework.run_cross_validation(
            self.X, self.y, self.model,
            cv_method="kfold",
            n_folds=3
        )

        assert isinstance(result, ValidationResult)
        assert result.status == "completed"
        assert len(self.framework.validation_results) == 1

        # Check if result was saved
        result_files = list(Path(self.temp_dir).glob("validation_*.json"))
        assert len(result_files) == 1

    @patch('qemlflow.reproducibility.validation_framework.log_audit_event')
    def test_run_benchmark(self, mock_log):
        """Test running benchmark."""
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42
        )

        result = self.framework.run_benchmark(
            X_train, X_test, y_train, y_test, self.model,
            dataset_name="test_dataset",
            model_name="test_model"
        )

        assert isinstance(result, BenchmarkResult)
        assert len(self.framework.benchmark_results) == 1

        # Check if result was saved
        benchmark_files = list(Path(self.temp_dir).glob("benchmark_*.json"))
        assert len(benchmark_files) == 1

    def test_run_statistical_validation(self):
        """Test running statistical validation."""
        results = [0.8, 0.85, 0.9, 0.75, 0.88]

        result = self.framework.run_statistical_validation(results)

        assert isinstance(result, ValidationResult)
        assert result.validation_type == "statistical"
        assert len(self.framework.validation_results) == 1

    @patch('qemlflow.reproducibility.validation_framework.log_audit_event')
    def test_comprehensive_validation(self, mock_log):
        """Test comprehensive validation suite."""
        report = self.framework.run_comprehensive_validation(
            self.X, self.y, self.model,
            dataset_name="test_dataset",
            model_name="test_model",
            cv_methods=["kfold", "stratified"]
        )

        assert isinstance(report, ValidationReport)
        assert len(report.validation_results) >= 1  # At least one CV method should work
        assert len(report.benchmark_results) >= 1
        assert report.title != ""
        assert report.description != ""

        # Check summary calculation
        assert "total_validations" in report.summary
        assert "total_benchmarks" in report.summary

        # Check if report was saved
        report_files = list(Path(self.temp_dir).glob("report_*.json"))
        assert len(report_files) == 1

    def test_get_validation_history(self):
        """Test getting validation history."""
        # Run some validations
        self.framework.run_cross_validation(self.X, self.y, self.model)
        self.framework.run_statistical_validation([0.8, 0.9])

        history = self.framework.get_validation_history()

        assert len(history) == 2
        assert all(isinstance(r, ValidationResult) for r in history)

    def test_get_benchmark_history(self):
        """Test getting benchmark history."""
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42
        )

        self.framework.run_benchmark(
            X_train, X_test, y_train, y_test, self.model
        )

        history = self.framework.get_benchmark_history()

        assert len(history) == 1
        assert isinstance(history[0], BenchmarkResult)

    def test_generate_summary_report(self):
        """Test generating summary report."""
        # Add some results
        self.framework.run_cross_validation(self.X, self.y, self.model)

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42
        )
        self.framework.run_benchmark(
            X_train, X_test, y_train, y_test, self.model
        )

        summary = self.framework.generate_summary_report()

        assert "total_validations" in summary
        assert "total_benchmarks" in summary
        assert "timestamp" in summary
        assert summary["total_validations"] >= 1
        assert summary["total_benchmarks"] >= 1

class TestContinuousValidator:
    """Test ContinuousValidator class."""

    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.framework = ValidationFramework(results_dir=self.temp_dir)
        self.continuous_validator = ContinuousValidator(
            self.framework,
            schedule_config={"interval": 1}
        )

    def teardown_method(self):
        """Cleanup test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_add_validation_task(self):
        """Test adding validation tasks."""
        task_config = {
            "type": "cross_validation",
            "name": "test_task",
            "dataset": "test_data"
        }

        self.continuous_validator.add_validation_task(task_config)

        assert len(self.continuous_validator.validation_queue) == 1
        assert self.continuous_validator.validation_queue[0] == task_config

    @patch('qemlflow.reproducibility.validation_framework.log_audit_event')
    def test_execute_validation_task(self, mock_log):
        """Test executing validation task."""
        task_config = {
            "type": "cross_validation",
            "name": "test_task"
        }

        self.continuous_validator._execute_validation_task(task_config)

        # Should log the task execution
        mock_log.assert_called_once()
        call_args = mock_log.call_args
        assert call_args[1]["action"] == "continuous_validation_task"

    def test_stop_continuous_validation(self):
        """Test stopping continuous validation."""
        assert not self.continuous_validator.is_running

        self.continuous_validator.stop()

        assert not self.continuous_validator.is_running

class TestStandaloneAPI:
    """Test standalone API functions."""

    def setup_method(self):
        """Setup test data."""
        self.X, self.y = make_classification(
            n_samples=100, n_features=10, n_classes=2, random_state=42
        )
        self.model = LogisticRegression(random_state=42)

    @patch('qemlflow.reproducibility.validation_framework.get_validation_framework')
    def test_validate_model_cross_validation(self, mock_get_framework):
        """Test validate_model function with cross-validation."""
        mock_framework = Mock()
        mock_result = ValidationResult(validation_type="cross_validation")
        mock_framework.run_cross_validation.return_value = mock_result
        mock_get_framework.return_value = mock_framework

        result = validate_model(
            self.X, self.y, self.model,
            validation_type="cross_validation"
        )

        assert result == mock_result
        mock_framework.run_cross_validation.assert_called_once()

    @patch('qemlflow.reproducibility.validation_framework.get_validation_framework')
    def test_validate_model_comprehensive(self, mock_get_framework):
        """Test validate_model function with comprehensive validation."""
        mock_framework = Mock()
        mock_report = ValidationReport(title="Test Report")
        mock_framework.run_comprehensive_validation.return_value = mock_report
        mock_get_framework.return_value = mock_framework

        result = validate_model(
            self.X, self.y, self.model,
            validation_type="comprehensive"
        )

        assert result == mock_report
        mock_framework.run_comprehensive_validation.assert_called_once()

    def test_validate_model_invalid_type(self):
        """Test validate_model with invalid validation type."""
        with pytest.raises(ValueError, match="Unknown validation type"):
            validate_model(
                self.X, self.y, self.model,
                validation_type="invalid_type"
            )

    @patch('qemlflow.reproducibility.validation_framework.get_validation_framework')
    def test_run_benchmark_test(self, mock_get_framework):
        """Test run_benchmark_test function."""
        mock_framework = Mock()
        mock_result = BenchmarkResult(benchmark_name="test")
        mock_framework.run_benchmark.return_value = mock_result
        mock_get_framework.return_value = mock_framework

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42
        )

        result = run_benchmark_test(
            X_train, X_test, y_train, y_test, self.model
        )

        assert result == mock_result
        mock_framework.run_benchmark.assert_called_once()

    @patch('qemlflow.reproducibility.validation_framework.get_validation_framework')
    def test_generate_validation_report(self, mock_get_framework):
        """Test generate_validation_report function."""
        mock_framework = Mock()
        mock_report = ValidationReport(title="Generated Report")
        mock_framework.run_comprehensive_validation.return_value = mock_report
        mock_get_framework.return_value = mock_framework

        result = generate_validation_report(self.X, self.y, self.model)

        assert result == mock_report
        mock_framework.run_comprehensive_validation.assert_called_once()

class TestGlobalFramework:
    """Test global framework management."""

    def test_get_validation_framework_singleton(self):
        """Test that get_validation_framework returns singleton."""
        framework1 = get_validation_framework()
        framework2 = get_validation_framework()

        assert framework1 is framework2

    def test_get_validation_framework_with_config(self):
        """Test getting framework with custom config."""
        config = {"test_key": "test_value"}

        # Clear global instance first
        import src.qemlflow.reproducibility.validation_framework as vf_module
        vf_module._global_validation_framework = None

        framework = get_validation_framework(config=config)

        assert framework.config == config

class TestIntegration:
    """Integration tests for validation framework."""

    def setup_method(self):
        """Setup integration test environment."""
        self.temp_dir = tempfile.mkdtemp()

        # Create more complex test data
        self.X_class, self.y_class = make_classification(
            n_samples=200, n_features=20, n_classes=3,
            n_informative=15, n_redundant=5, random_state=42
        )

        self.X_reg, self.y_reg = make_regression(
            n_samples=200, n_features=20,
            noise=0.1, random_state=42
        )

        # Multiple models to test
        self.classification_models = {
            "logistic_regression": LogisticRegression(random_state=42),
            "random_forest": RandomForestClassifier(n_estimators=10, random_state=42)
        }

        self.regression_models = {
            "linear_regression": LinearRegression()
        }

    def teardown_method(self):
        """Cleanup integration test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('qemlflow.reproducibility.validation_framework.log_audit_event')
    def test_end_to_end_classification_validation(self, mock_log):
        """Test end-to-end classification validation."""
        framework = ValidationFramework(results_dir=self.temp_dir)

        for model_name, model in self.classification_models.items():
            # Run comprehensive validation
            report = framework.run_comprehensive_validation(
                self.X_class, self.y_class, model,
                dataset_name="synthetic_classification",
                model_name=model_name,
                cv_methods=["kfold", "stratified"],
                test_size=0.2
            )

            # Validate report structure
            assert isinstance(report, ValidationReport)
            assert len(report.validation_results) >= 1
            assert len(report.benchmark_results) >= 1
            assert report.summary["total_validations"] >= 1
            assert report.summary["total_benchmarks"] >= 1

            # Check that results are reasonable
            for val_result in report.validation_results:
                if val_result.status == "completed":
                    assert val_result.mean_score > 0
                    assert val_result.std_score >= 0

            for bench_result in report.benchmark_results:
                assert bench_result.accuracy >= 0
                assert bench_result.training_time > 0
                assert bench_result.inference_time > 0

        # Check audit logging
        assert mock_log.call_count > 0

        # Verify file outputs
        result_files = list(Path(self.temp_dir).glob("*.json"))
        assert len(result_files) > 0

    def test_multiple_model_comparison(self):
        """Test comparing multiple models."""
        framework = ValidationFramework(results_dir=self.temp_dir)

        model_results = {}

        for model_name, model in self.classification_models.items():
            # Run cross-validation
            cv_result = framework.run_cross_validation(
                self.X_class, self.y_class, model,
                cv_method="kfold",
                n_folds=3,
                dataset_name="comparison_test",
                model_name=model_name
            )

            model_results[model_name] = cv_result.mean_score

        # Generate summary
        summary = framework.generate_summary_report()

        assert summary["total_validations"] == len(self.classification_models)
        assert "validation_summary" in summary

        # All models should have reasonable performance
        for model_name, score in model_results.items():
            assert score > 0, f"Model {model_name} has invalid score: {score}"

    def test_statistical_validation_pipeline(self):
        """Test statistical validation pipeline."""
        framework = ValidationFramework(results_dir=self.temp_dir)

        # Generate multiple validation runs
        scores_history = []

        for run in range(3):
            cv_result = framework.run_cross_validation(
                self.X_class, self.y_class,
                LogisticRegression(random_state=42 + run),
                cv_method="kfold",
                n_folds=5
            )
            scores_history.extend(cv_result.scores)

        # Run statistical validation on collected scores
        stat_result = framework.run_statistical_validation(
            scores_history,
            reference_results=[0.8, 0.85, 0.9, 0.75, 0.88]  # Reference baseline
        )

        assert stat_result.status == "completed"
        assert "shapiro_wilk" in stat_result.statistical_tests
        assert "mean_difference" in stat_result.statistical_tests

        # Check statistical test results
        shapiro_test = stat_result.statistical_tests["shapiro_wilk"]
        assert "statistic" in shapiro_test
        assert "p_value" in shapiro_test
        assert "is_normal" in shapiro_test

    def test_file_persistence_and_loading(self):
        """Test file persistence and loading."""
        framework = ValidationFramework(results_dir=self.temp_dir)

        # Run validation
        cv_result = framework.run_cross_validation(
            self.X_class, self.y_class,
            LogisticRegression(random_state=42),
            cv_method="kfold"
        )

        # Find and load the saved result
        result_files = list(Path(self.temp_dir).glob("validation_*.json"))
        assert len(result_files) == 1

        with open(result_files[0], 'r') as f:
            saved_data = json.load(f)

        # Verify data structure
        assert "validation_id" in saved_data
        assert "validation_type" in saved_data
        assert "scores" in saved_data
        assert "mean_score" in saved_data

        # Reconstruct ValidationResult
        loaded_result = ValidationResult.from_dict(saved_data)

        assert loaded_result.validation_id == cv_result.validation_id
        assert loaded_result.validation_type == cv_result.validation_type
        assert loaded_result.scores == cv_result.scores
        assert loaded_result.mean_score == cv_result.mean_score

if __name__ == "__main__":
    pytest.main([__file__])
