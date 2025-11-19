"""
Validation Framework Module

This module provides comprehensive validation infrastructure for scientific computing
including cross-validation, benchmark testing, validation reporting, and continuous
validation to ensure reproducible and reliable results.
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

import numpy as np
from sklearn.model_selection import (
    KFold, StratifiedKFold, TimeSeriesSplit, 
    cross_val_score, cross_validate
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)

from .audit_trail import log_audit_event


@dataclass
class ValidationResult:
    """Individual validation result with complete metrics."""
    
    validation_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    # Validation metadata
    validation_type: str = ""  # cross_validation, benchmark, statistical
    method: str = ""  # kfold, stratified, timeseries, etc.
    dataset_name: str = ""
    model_name: str = ""
    
    # Cross-validation parameters
    n_folds: int = 5
    random_state: Optional[int] = None
    stratify: bool = False
    
    # Results
    scores: List[float] = field(default_factory=list)
    mean_score: float = 0.0
    std_score: float = 0.0
    min_score: float = 0.0
    max_score: float = 0.0
    
    # Detailed metrics
    detailed_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Statistical tests
    statistical_tests: Dict[str, Any] = field(default_factory=dict)
    
    # Performance metrics
    execution_time: float = 0.0
    memory_usage: float = 0.0
    
    # Validation status
    status: str = "pending"  # pending, running, completed, failed
    error_message: str = ""
    
    # Compliance and reproducibility
    environment_hash: str = ""
    data_hash: str = ""
    model_hash: str = ""
    
    def calculate_statistics(self):
        """Calculate statistical measures from scores."""
        if self.scores:
            self.mean_score = float(np.mean(self.scores))
            self.std_score = float(np.std(self.scores))
            self.min_score = float(np.min(self.scores))
            self.max_score = float(np.max(self.scores))
    
    def add_detailed_metric(self, name: str, value: Any):
        """Add detailed metric."""
        self.detailed_metrics[name] = value
    
    def add_statistical_test(self, test_name: str, result: Dict[str, Any]):
        """Add statistical test result."""
        self.statistical_tests[test_name] = result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ValidationResult':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class BenchmarkResult:
    """Benchmark test result with performance comparison."""
    
    benchmark_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    # Benchmark metadata
    benchmark_name: str = ""
    dataset_name: str = ""
    model_name: str = ""
    
    # Performance metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    
    # Regression metrics (if applicable)
    mse: float = 0.0
    mae: float = 0.0
    r2: float = 0.0
    
    # Execution metrics
    training_time: float = 0.0
    inference_time: float = 0.0
    memory_usage: float = 0.0
    
    # Comparison with baselines
    baseline_comparisons: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Quality indicators
    convergence_achieved: bool = False
    stability_score: float = 0.0
    
    def add_baseline_comparison(self, baseline_name: str, metrics: Dict[str, float]):
        """Add baseline comparison."""
        self.baseline_comparisons[baseline_name] = metrics
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    
    report_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    # Report metadata
    report_type: str = "validation_report"
    title: str = ""
    description: str = ""
    
    # Validation results
    validation_results: List[ValidationResult] = field(default_factory=list)
    benchmark_results: List[BenchmarkResult] = field(default_factory=list)
    
    # Summary statistics
    summary: Dict[str, Any] = field(default_factory=dict)
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    
    # Quality assessment
    overall_quality_score: float = 0.0
    quality_indicators: Dict[str, Any] = field(default_factory=dict)
    
    def add_validation_result(self, result: ValidationResult):
        """Add validation result."""
        self.validation_results.append(result)
    
    def add_benchmark_result(self, result: BenchmarkResult):
        """Add benchmark result."""
        self.benchmark_results.append(result)
    
    def calculate_summary(self):
        """Calculate summary statistics."""
        if self.validation_results:
            scores = [r.mean_score for r in self.validation_results if r.mean_score > 0]
            if scores:
                self.summary['mean_validation_score'] = float(np.mean(scores))
                self.summary['std_validation_score'] = float(np.std(scores))
                self.summary['min_validation_score'] = float(np.min(scores))
                self.summary['max_validation_score'] = float(np.max(scores))
        
        if self.benchmark_results:
            accuracies = [r.accuracy for r in self.benchmark_results if r.accuracy > 0]
            if accuracies:
                self.summary['mean_benchmark_accuracy'] = float(np.mean(accuracies))
                self.summary['std_benchmark_accuracy'] = float(np.std(accuracies))
        
        self.summary['total_validations'] = len(self.validation_results)
        self.summary['total_benchmarks'] = len(self.benchmark_results)
    
    def generate_recommendations(self):
        """Generate recommendations based on results."""
        self.recommendations.clear()
        
        # Check validation consistency
        if self.validation_results:
            std_scores = [r.std_score for r in self.validation_results if r.std_score > 0]
            if std_scores and np.mean(std_scores) > 0.1:
                self.recommendations.append(
                    "High variance in validation scores detected. Consider increasing training data or regularization."
                )
        
        # Check benchmark performance
        if self.benchmark_results:
            low_performance = [r for r in self.benchmark_results if r.accuracy < 0.7]
            if low_performance:
                self.recommendations.append(
                    f"Low performance detected in {len(low_performance)} benchmarks. Consider model tuning."
                )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class BaseValidator(ABC):
    """Base class for all validators."""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
    @abstractmethod
    def validate(self, X, y, model, **kwargs) -> ValidationResult:
        """Perform validation."""
        pass


class CrossValidator(BaseValidator):
    """Cross-validation implementation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("cross_validator", config)
        
    def validate(self, X, y, model, 
                 cv_method: str = "kfold",
                 n_folds: int = 5,
                 scoring: str = "accuracy",
                 random_state: Optional[int] = None,
                 **kwargs) -> ValidationResult:
        """Perform cross-validation."""
        
        # Create validation result
        result = ValidationResult(
            validation_type="cross_validation",
            method=cv_method,
            n_folds=n_folds,
            random_state=random_state
        )
        
        # Log validation start
        log_audit_event(
            action="cross_validation_start",
            resource=f"validation_{result.validation_id}",
            resource_type="validation",
            metadata={
                "method": cv_method,
                "n_folds": n_folds,
                "scoring": scoring
            }
        )
        
        try:
            start_time = time.time()
            result.status = "running"
            
            # Choose cross-validation method
            if cv_method == "kfold":
                cv = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
            elif cv_method == "stratified":
                cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
                result.stratify = True
            elif cv_method == "timeseries":
                cv = TimeSeriesSplit(n_splits=n_folds)
            else:
                raise ValueError(f"Unknown CV method: {cv_method}")
            
            # Perform cross-validation
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
            result.scores = scores.tolist()
            result.calculate_statistics()
            
            # Detailed cross-validation with multiple metrics
            detailed_cv = cross_validate(
                model, X, y, cv=cv, 
                scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'],
                return_train_score=True
            )
            
            for metric, values in detailed_cv.items():
                if metric.startswith('test_'):
                    metric_name = metric.replace('test_', '')
                    result.add_detailed_metric(f"{metric_name}_mean", float(np.mean(values)))
                    result.add_detailed_metric(f"{metric_name}_std", float(np.std(values)))
            
            result.execution_time = time.time() - start_time
            result.status = "completed"
            
            # Log completion
            log_audit_event(
                action="cross_validation_completed",
                resource=f"validation_{result.validation_id}",
                resource_type="validation",
                metadata={
                    "mean_score": result.mean_score,
                    "std_score": result.std_score,
                    "execution_time": result.execution_time
                }
            )
            
        except Exception as e:
            result.status = "failed"
            result.error_message = str(e)
            self.logger.error(f"Cross-validation failed: {e}")
            
            # Log failure
            log_audit_event(
                action="cross_validation_failed",
                resource=f"validation_{result.validation_id}",
                resource_type="validation",
                metadata={"error": str(e)}
            )
        
        return result


class BenchmarkTester(BaseValidator):
    """Benchmark testing implementation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("benchmark_tester", config)
        self.baselines = {}
        
    def add_baseline(self, name: str, model, metrics: Dict[str, float]):
        """Add baseline model for comparison."""
        self.baselines[name] = {
            'model': model,
            'metrics': metrics
        }
    
    def validate(self, X_train, X_test, y_train, y_test, model, 
                 dataset_name: str = "",
                 model_name: str = "",
                 **kwargs) -> BenchmarkResult:
        """Perform benchmark testing."""
        
        result = BenchmarkResult(
            benchmark_name=f"benchmark_{dataset_name}_{model_name}",
            dataset_name=dataset_name,
            model_name=model_name
        )
        
        # Log benchmark start
        log_audit_event(
            action="benchmark_test_start",
            resource=f"benchmark_{result.benchmark_id}",
            resource_type="benchmark",
            metadata={
                "dataset": dataset_name,
                "model": model_name
            }
        )
        
        try:
            # Training
            start_time = time.time()
            model.fit(X_train, y_train)
            result.training_time = time.time() - start_time
            
            # Inference
            start_time = time.time()
            y_pred = model.predict(X_test)
            result.inference_time = time.time() - start_time
            
            # Calculate metrics
            if hasattr(model, 'predict_proba'):  # Classification
                result.accuracy = float(accuracy_score(y_test, y_pred))
                result.precision = float(precision_score(y_test, y_pred, average='macro', zero_division=0))
                result.recall = float(recall_score(y_test, y_pred, average='macro', zero_division=0))
                result.f1_score = float(f1_score(y_test, y_pred, average='macro', zero_division=0))
            else:  # Regression
                result.mse = float(mean_squared_error(y_test, y_pred))
                result.mae = float(mean_absolute_error(y_test, y_pred))
                result.r2 = float(r2_score(y_test, y_pred))
            
            # Compare with baselines
            for baseline_name, baseline_info in self.baselines.items():
                baseline_model = baseline_info['model']
                baseline_pred = baseline_model.predict(X_test)
                
                if hasattr(baseline_model, 'predict_proba'):
                    baseline_metrics = {
                        'accuracy': accuracy_score(y_test, baseline_pred),
                        'precision': precision_score(y_test, baseline_pred, average='macro', zero_division=0),
                        'recall': recall_score(y_test, baseline_pred, average='macro', zero_division=0),
                        'f1_score': f1_score(y_test, baseline_pred, average='macro', zero_division=0)
                    }
                else:
                    baseline_metrics = {
                        'mse': mean_squared_error(y_test, baseline_pred),
                        'mae': mean_absolute_error(y_test, baseline_pred),
                        'r2': r2_score(y_test, baseline_pred)
                    }
                
                result.add_baseline_comparison(baseline_name, baseline_metrics)
            
            # Log completion
            log_audit_event(
                action="benchmark_test_completed",
                resource=f"benchmark_{result.benchmark_id}",
                resource_type="benchmark",
                metadata={
                    "accuracy": result.accuracy,
                    "training_time": result.training_time,
                    "inference_time": result.inference_time
                }
            )
            
        except Exception as e:
            self.logger.error(f"Benchmark testing failed: {e}")
            
            # Log failure
            log_audit_event(
                action="benchmark_test_failed",
                resource=f"benchmark_{result.benchmark_id}",
                resource_type="benchmark",
                metadata={"error": str(e)}
            )
        
        return result


class StatisticalValidator(BaseValidator):
    """Statistical validation implementation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("statistical_validator", config)
        
    def validate(self, results: List[float], 
                 reference_results: Optional[List[float]] = None,
                 **kwargs) -> ValidationResult:
        """Perform statistical validation."""
        
        result = ValidationResult(
            validation_type="statistical",
            method="statistical_tests"
        )
        
        try:
            # Basic statistics
            result.scores = results
            result.calculate_statistics()
            
            # Normality test
            from scipy import stats
            stat, p_value = stats.shapiro(results)
            result.add_statistical_test("shapiro_wilk", {
                "statistic": float(stat),
                "p_value": float(p_value),
                "is_normal": p_value > 0.05
            })
            
            # If reference results provided, perform comparison tests
            if reference_results:
                try:
                    # Basic comparison
                    mean_diff = float(np.mean(results)) - float(np.mean(reference_results))
                    result.add_statistical_test("mean_difference", {
                        "mean_difference": mean_diff,
                        "reference_mean": float(np.mean(reference_results)),
                        "current_mean": float(np.mean(results))
                    })
                except Exception as e:
                    self.logger.warning(f"Statistical comparison failed: {e}")
            
            result.status = "completed"
            
        except Exception as e:
            result.status = "failed"
            result.error_message = str(e)
            self.logger.error(f"Statistical validation failed: {e}")
        
        return result


class ValidationFramework:
    """Main validation framework coordinating all validation activities."""
    
    def __init__(self, results_dir: str = "validation_results",
                 config: Optional[Dict[str, Any]] = None):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize validators
        self.cross_validator = CrossValidator(self.config.get('cross_validation', {}))
        self.benchmark_tester = BenchmarkTester(self.config.get('benchmark', {}))
        self.statistical_validator = StatisticalValidator(self.config.get('statistical', {}))
        
        # Results storage
        self.validation_results: List[ValidationResult] = []
        self.benchmark_results: List[BenchmarkResult] = []
        
        self.logger.info(f"Validation framework initialized: {self.results_dir}")
    
    def run_cross_validation(self, X, y, model, **kwargs) -> ValidationResult:
        """Run cross-validation."""
        result = self.cross_validator.validate(X, y, model, **kwargs)
        self.validation_results.append(result)
        self._save_result(result)
        return result
    
    def run_benchmark(self, X_train, X_test, y_train, y_test, model, **kwargs) -> BenchmarkResult:
        """Run benchmark test."""
        result = self.benchmark_tester.validate(
            X_train, X_test, y_train, y_test, model, **kwargs
        )
        self.benchmark_results.append(result)
        self._save_result(result)
        return result
    
    def run_statistical_validation(self, results: List[float], **kwargs) -> ValidationResult:
        """Run statistical validation."""
        result = self.statistical_validator.validate(results, **kwargs)
        self.validation_results.append(result)
        self._save_result(result)
        return result
    
    def run_comprehensive_validation(self, X, y, model,
                                     test_size: float = 0.2,
                                     cv_methods: Optional[List[str]] = None,
                                     dataset_name: str = "",
                                     model_name: str = "",
                                     **kwargs) -> ValidationReport:
        """Run comprehensive validation suite."""
        
        if cv_methods is None:
            cv_methods = ['kfold', 'stratified']
        
        # Log comprehensive validation start
        log_audit_event(
            action="comprehensive_validation_start",
            resource=f"validation_suite_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            resource_type="validation_suite",
            metadata={
                "dataset": dataset_name,
                "model": model_name,
                "cv_methods": cv_methods
            }
        )
        
        # Split data for benchmarking
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Create validation report
        report = ValidationReport(
            title=f"Comprehensive Validation Report - {model_name}",
            description=f"Complete validation suite for {model_name} on {dataset_name}"
        )
        
        # Run cross-validation with different methods
        for cv_method in cv_methods:
            try:
                cv_result = self.run_cross_validation(
                    X_train, y_train, model,
                    cv_method=cv_method,
                    dataset_name=dataset_name,
                    model_name=model_name,
                    **kwargs
                )
                report.add_validation_result(cv_result)
            except Exception as e:
                self.logger.warning(f"Cross-validation with {cv_method} failed: {e}")
        
        # Run benchmark test
        try:
            benchmark_result = self.run_benchmark(
                X_train, X_test, y_train, y_test, model,
                dataset_name=dataset_name,
                model_name=model_name,
                **kwargs
            )
            report.add_benchmark_result(benchmark_result)
        except Exception as e:
            self.logger.warning(f"Benchmark test failed: {e}")
        
        # Calculate summary and recommendations
        report.calculate_summary()
        report.generate_recommendations()
        
        # Save report
        self._save_report(report)
        
        # Log completion
        log_audit_event(
            action="comprehensive_validation_completed",
            resource=f"validation_suite_{report.report_id}",
            resource_type="validation_suite",
            metadata={
                "report_id": report.report_id,
                "total_validations": len(report.validation_results),
                "total_benchmarks": len(report.benchmark_results)
            }
        )
        
        return report
    
    def _save_result(self, result):
        """Save individual result to file."""
        try:
            if isinstance(result, ValidationResult):
                filename = f"validation_{result.validation_id}.json"
            elif isinstance(result, BenchmarkResult):
                filename = f"benchmark_{result.benchmark_id}.json"
            else:
                filename = f"result_{result.validation_id}.json"
            
            filepath = self.results_dir / filename
            with open(filepath, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save result: {e}")
    
    def _save_report(self, report: ValidationReport):
        """Save validation report to file."""
        try:
            filename = f"report_{report.report_id}.json"
            filepath = self.results_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(report.to_dict(), f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save report: {e}")
    
    def get_validation_history(self) -> List[ValidationResult]:
        """Get all validation results."""
        return self.validation_results
    
    def get_benchmark_history(self) -> List[BenchmarkResult]:
        """Get all benchmark results."""
        return self.benchmark_results
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate summary report of all validations."""
        summary = {
            "total_validations": len(self.validation_results),
            "total_benchmarks": len(self.benchmark_results),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        if self.validation_results:
            completed_validations = [r for r in self.validation_results if r.status == "completed"]
            if completed_validations:
                scores = [r.mean_score for r in completed_validations]
                summary["validation_summary"] = {
                    "mean_score": float(np.mean(scores)),
                    "std_score": float(np.std(scores)),
                    "min_score": float(np.min(scores)),
                    "max_score": float(np.max(scores)),
                    "success_rate": len(completed_validations) / len(self.validation_results)
                }
        
        if self.benchmark_results:
            accuracies = [r.accuracy for r in self.benchmark_results if r.accuracy > 0]
            if accuracies:
                summary["benchmark_summary"] = {
                    "mean_accuracy": float(np.mean(accuracies)),
                    "std_accuracy": float(np.std(accuracies)),
                    "min_accuracy": float(np.min(accuracies)),
                    "max_accuracy": float(np.max(accuracies))
                }
        
        return summary


# Continuous validation system
class ContinuousValidator:
    """Continuous validation system for automated quality assurance."""
    
    def __init__(self, validation_framework: ValidationFramework,
                 schedule_config: Optional[Dict[str, Any]] = None):
        self.framework = validation_framework
        self.schedule_config = schedule_config or {}
        self.logger = logging.getLogger(__name__)
        
        self.validation_queue = []
        self.is_running = False
        
    def add_validation_task(self, task_config: Dict[str, Any]):
        """Add validation task to queue."""
        self.validation_queue.append(task_config)
        
    def run_continuous_validation(self):
        """Run continuous validation process."""
        self.is_running = True
        
        while self.is_running and self.validation_queue:
            task = self.validation_queue.pop(0)
            
            try:
                # Execute validation task
                self._execute_validation_task(task)
                
            except Exception as e:
                self.logger.error(f"Continuous validation task failed: {e}")
            
            # Sleep between tasks
            time.sleep(self.schedule_config.get('interval', 60))
    
    def _execute_validation_task(self, task_config: Dict[str, Any]):
        """Execute individual validation task."""
        task_type = task_config.get('type', 'cross_validation')
        
        # Log task execution
        log_audit_event(
            action="continuous_validation_task",
            resource=f"task_{task_config.get('name', 'unknown')}",
            resource_type="validation_task",
            metadata=task_config
        )
        
        # Execute based on task type
        # Implementation would depend on specific task requirements
        self.logger.info(f"Executing continuous validation task: {task_type}")
    
    def stop(self):
        """Stop continuous validation."""
        self.is_running = False


# Global validation framework instance
_global_validation_framework = None

def get_validation_framework(results_dir: str = "validation_results",
                           config: Optional[Dict[str, Any]] = None) -> ValidationFramework:
    """Get global validation framework instance."""
    global _global_validation_framework
    
    if _global_validation_framework is None:
        _global_validation_framework = ValidationFramework(results_dir, config)
    
    return _global_validation_framework


# Standalone API functions
def validate_model(X, y, model, 
                   validation_type: str = "cross_validation",
                   **kwargs) -> Union[ValidationResult, BenchmarkResult, ValidationReport]:
    """Validate model using specified validation type."""
    framework = get_validation_framework()
    
    if validation_type == "cross_validation":
        return framework.run_cross_validation(X, y, model, **kwargs)
    elif validation_type == "comprehensive":
        return framework.run_comprehensive_validation(X, y, model, **kwargs)
    else:
        raise ValueError(f"Unknown validation type: {validation_type}")


def run_benchmark_test(X_train, X_test, y_train, y_test, model, **kwargs) -> BenchmarkResult:
    """Run benchmark test using global framework."""
    framework = get_validation_framework()
    return framework.run_benchmark(X_train, X_test, y_train, y_test, model, **kwargs)


def generate_validation_report(X, y, model, **kwargs) -> ValidationReport:
    """Generate comprehensive validation report."""
    framework = get_validation_framework()
    return framework.run_comprehensive_validation(X, y, model, **kwargs)
