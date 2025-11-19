"""
Result Validation Module

This module provides comprehensive result validation capabilities for reproducible
scientific computing including result comparison, statistical validation, and
reproducibility verification.
"""

import hashlib
import json
import logging
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from scipy import stats
import warnings

from .experiment_tracking import ExperimentRecord, ExperimentTracker


@dataclass
class ValidationMetric:
    """Individual validation metric with statistical analysis."""
    
    name: str
    expected_value: float
    actual_value: float
    tolerance: float = 1e-6
    relative_tolerance: float = 1e-3
    validation_method: str = "absolute"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate validation result after initialization."""
        self.is_valid = self._validate()
        self.difference = abs(self.actual_value - self.expected_value)
        self.relative_difference = (
            self.difference / abs(self.expected_value) 
            if self.expected_value != 0 else float('inf')
        )
    
    def _validate(self) -> bool:
        """Perform validation based on method."""
        if self.validation_method == "absolute":
            return abs(self.actual_value - self.expected_value) <= self.tolerance
        elif self.validation_method == "relative":
            if self.expected_value == 0:
                return abs(self.actual_value) <= self.tolerance
            return abs(self.actual_value - self.expected_value) / abs(self.expected_value) <= self.relative_tolerance
        elif self.validation_method == "statistical":
            # For statistical validation, we'd need multiple values
            return True  # Placeholder for now
        else:
            raise ValueError(f"Unknown validation method: {self.validation_method}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'expected_value': self.expected_value,
            'actual_value': self.actual_value,
            'tolerance': self.tolerance,
            'relative_tolerance': self.relative_tolerance,
            'validation_method': self.validation_method,
            'is_valid': self.is_valid,
            'difference': self.difference,
            'relative_difference': self.relative_difference,
            'metadata': self.metadata
        }


@dataclass
class StatisticalComparison:
    """Statistical comparison between two sets of results."""
    
    metric_name: str
    reference_values: List[float]
    test_values: List[float]
    statistical_tests: Dict[str, Any] = field(default_factory=dict)
    summary_stats: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Perform statistical analysis after initialization."""
        self._calculate_summary_stats()
        self._perform_statistical_tests()
    
    def _calculate_summary_stats(self):
        """Calculate summary statistics for both datasets."""
        self.summary_stats = {
            'reference': {
                'mean': np.mean(self.reference_values),
                'std': np.std(self.reference_values),
                'min': np.min(self.reference_values),
                'max': np.max(self.reference_values),
                'median': np.median(self.reference_values),
                'count': len(self.reference_values)
            },
            'test': {
                'mean': np.mean(self.test_values),
                'std': np.std(self.test_values),
                'min': np.min(self.test_values),
                'max': np.max(self.test_values),
                'median': np.median(self.test_values),
                'count': len(self.test_values)
            }
        }
    
    def _perform_statistical_tests(self):
        """Perform various statistical tests."""
        try:
            # T-test for means
            if len(self.reference_values) > 1 and len(self.test_values) > 1:
                t_stat, t_p_value = stats.ttest_ind(self.reference_values, self.test_values)
                self.statistical_tests['t_test'] = {
                    'statistic': float(t_stat),
                    'p_value': float(t_p_value),
                    'significant': t_p_value < 0.05
                }
            
            # Mann-Whitney U test (non-parametric)
            if len(self.reference_values) > 2 and len(self.test_values) > 2:
                u_stat, u_p_value = stats.mannwhitneyu(
                    self.reference_values, self.test_values, alternative='two-sided'
                )
                self.statistical_tests['mann_whitney'] = {
                    'statistic': float(u_stat),
                    'p_value': float(u_p_value),
                    'significant': u_p_value < 0.05
                }
            
            # Kolmogorov-Smirnov test for distribution comparison
            if len(self.reference_values) > 2 and len(self.test_values) > 2:
                ks_stat, ks_p_value = stats.ks_2samp(self.reference_values, self.test_values)
                self.statistical_tests['kolmogorov_smirnov'] = {
                    'statistic': float(ks_stat),
                    'p_value': float(ks_p_value),
                    'significant': ks_p_value < 0.05
                }
            
            # Effect size (Cohen's d)
            if len(self.reference_values) > 1 and len(self.test_values) > 1:
                pooled_std = np.sqrt(
                    ((len(self.reference_values) - 1) * np.var(self.reference_values, ddof=1) +
                     (len(self.test_values) - 1) * np.var(self.test_values, ddof=1)) /
                    (len(self.reference_values) + len(self.test_values) - 2)
                )
                if pooled_std > 0:
                    cohens_d = (np.mean(self.test_values) - np.mean(self.reference_values)) / pooled_std
                    self.statistical_tests['effect_size'] = {
                        'cohens_d': float(cohens_d),
                        'interpretation': self._interpret_cohens_d(cohens_d)
                    }
        
        except Exception as e:
            # Log warning but don't fail
            logging.getLogger(__name__).warning(f"Statistical test failed: {e}")
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'metric_name': self.metric_name,
            'reference_values': self.reference_values,
            'test_values': self.test_values,
            'statistical_tests': self.statistical_tests,
            'summary_stats': self.summary_stats
        }


@dataclass
class ValidationReport:
    """Comprehensive validation report for experiment results."""
    
    experiment_id: str
    reference_experiment_id: Optional[str]
    validation_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metric_validations: List[ValidationMetric] = field(default_factory=list)
    statistical_comparisons: List[StatisticalComparison] = field(default_factory=list)
    environment_validation: Dict[str, Any] = field(default_factory=dict)
    data_validation: Dict[str, Any] = field(default_factory=dict)
    overall_valid: bool = True
    summary: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate overall validation status."""
        self._calculate_summary()
    
    def _calculate_summary(self):
        """Calculate validation summary."""
        total_metrics = len(self.metric_validations)
        valid_metrics = sum(1 for mv in self.metric_validations if mv.is_valid)
        
        # Statistical significance summary
        significant_tests = 0
        total_statistical_tests = 0
        
        for sc in self.statistical_comparisons:
            for test_name, test_result in sc.statistical_tests.items():
                if 'significant' in test_result:
                    total_statistical_tests += 1
                    if test_result['significant']:
                        significant_tests += 1
        
        self.summary = {
            'total_metrics': total_metrics,
            'valid_metrics': valid_metrics,
            'metric_validity_rate': valid_metrics / total_metrics if total_metrics > 0 else 1.0,
            'total_statistical_tests': total_statistical_tests,
            'significant_statistical_differences': significant_tests,
            'environment_valid': self.environment_validation.get('valid', True),
            'data_valid': self.data_validation.get('valid', True)
        }
        
        # Overall validation
        self.overall_valid = (
            self.summary['metric_validity_rate'] >= 0.9 and  # 90% of metrics must be valid
            self.summary['environment_valid'] and
            self.summary['data_valid']
        )
    
    def add_metric_validation(self, validation: ValidationMetric):
        """Add metric validation result."""
        self.metric_validations.append(validation)
        self._calculate_summary()
    
    def add_statistical_comparison(self, comparison: StatisticalComparison):
        """Add statistical comparison result."""
        self.statistical_comparisons.append(comparison)
        self._calculate_summary()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'experiment_id': self.experiment_id,
            'reference_experiment_id': self.reference_experiment_id,
            'validation_timestamp': self.validation_timestamp,
            'metric_validations': [mv.to_dict() for mv in self.metric_validations],
            'statistical_comparisons': [sc.to_dict() for sc in self.statistical_comparisons],
            'environment_validation': self.environment_validation,
            'data_validation': self.data_validation,
            'overall_valid': self.overall_valid,
            'summary': self.summary
        }


class ResultValidator:
    """
    Comprehensive result validator for reproducible scientific computing.
    """
    
    def __init__(self, experiment_tracker: Optional[ExperimentTracker] = None):
        self.experiment_tracker = experiment_tracker or ExperimentTracker()
        self.logger = logging.getLogger(__name__)
    
    def validate_experiment_results(self, experiment_id: str, 
                                  reference_experiment_id: Optional[str] = None,
                                  metric_tolerances: Optional[Dict[str, Dict[str, float]]] = None) -> ValidationReport:
        """Validate experiment results against reference or expected values."""
        
        # Load experiment
        experiment = self.experiment_tracker.load_experiment(experiment_id)
        
        # Create validation report
        report = ValidationReport(
            experiment_id=experiment_id,
            reference_experiment_id=reference_experiment_id
        )
        
        if reference_experiment_id:
            # Validate against reference experiment
            reference_experiment = self.experiment_tracker.load_experiment(reference_experiment_id)
            self._validate_against_reference(experiment, reference_experiment, report, metric_tolerances)
        else:
            # Validate against expected values (if provided in tolerances)
            self._validate_against_expected(experiment, report, metric_tolerances)
        
        # Validate environment consistency
        if reference_experiment_id:
            reference_experiment = self.experiment_tracker.load_experiment(reference_experiment_id)
            report.environment_validation = self._validate_environment_consistency(
                experiment, reference_experiment
            )
        
        # Validate data consistency
        if reference_experiment_id:
            report.data_validation = self._validate_data_consistency(
                experiment, reference_experiment
            )
        
        self.logger.info(f"Validation completed for experiment {experiment_id}")
        return report
    
    def _validate_against_reference(self, experiment: ExperimentRecord, 
                                  reference: ExperimentRecord,
                                  report: ValidationReport,
                                  metric_tolerances: Optional[Dict[str, Dict[str, float]]]):
        """Validate experiment against reference experiment."""
        
        # Get all metric names from both experiments
        exp_metrics = experiment.metrics.metrics
        ref_metrics = reference.metrics.metrics
        
        all_metrics = set(exp_metrics.keys()) | set(ref_metrics.keys())
        
        for metric_name in all_metrics:
            if metric_name in exp_metrics and metric_name in ref_metrics:
                # Both experiments have this metric
                tolerances = metric_tolerances.get(metric_name, {}) if metric_tolerances else {}
                
                validation = ValidationMetric(
                    name=metric_name,
                    expected_value=ref_metrics[metric_name],
                    actual_value=exp_metrics[metric_name],
                    tolerance=tolerances.get('absolute', 1e-6),
                    relative_tolerance=tolerances.get('relative', 1e-3),
                    validation_method=tolerances.get('method', 'relative')
                )
                
                report.add_metric_validation(validation)
            else:
                # Metric missing in one experiment
                self.logger.warning(f"Metric {metric_name} missing in one of the experiments")
    
    def _validate_against_expected(self, experiment: ExperimentRecord,
                                 report: ValidationReport,
                                 metric_tolerances: Optional[Dict[str, Dict[str, float]]]):
        """Validate experiment against expected values."""
        
        if not metric_tolerances:
            self.logger.warning("No expected values provided for validation")
            return
        
        exp_metrics = experiment.metrics.metrics
        
        for metric_name, config in metric_tolerances.items():
            if 'expected' in config and metric_name in exp_metrics:
                validation = ValidationMetric(
                    name=metric_name,
                    expected_value=config['expected'],
                    actual_value=exp_metrics[metric_name],
                    tolerance=config.get('absolute', 1e-6),
                    relative_tolerance=config.get('relative', 1e-3),
                    validation_method=config.get('method', 'relative')
                )
                
                report.add_metric_validation(validation)
    
    def _validate_environment_consistency(self, experiment: ExperimentRecord,
                                        reference: ExperimentRecord) -> Dict[str, Any]:
        """Validate environment consistency between experiments."""
        
        validation = {
            'valid': True,
            'differences': []
        }
        
        exp_env = experiment.environment_fingerprint
        ref_env = reference.environment_fingerprint
        
        if not exp_env or not ref_env:
            validation['valid'] = False
            validation['differences'].append("Missing environment fingerprint")
            return validation
        
        # Check environment hash
        if exp_env.fingerprint_hash != ref_env.fingerprint_hash:
            validation['valid'] = False
            validation['differences'].append("Environment fingerprint mismatch")
        
        # Check Python version
        if exp_env.python_version != ref_env.python_version:
            validation['differences'].append(
                f"Python version: {exp_env.python_version} vs {ref_env.python_version}"
            )
        
        # Check platform
        exp_platform = exp_env.platform_info.get('system')
        ref_platform = ref_env.platform_info.get('system')
        if exp_platform != ref_platform:
            validation['differences'].append(
                f"Platform: {exp_platform} vs {ref_platform}"
            )
        
        return validation
    
    def _validate_data_consistency(self, experiment: ExperimentRecord,
                                 reference: ExperimentRecord) -> Dict[str, Any]:
        """Validate data consistency between experiments."""
        
        validation = {
            'valid': True,
            'differences': []
        }
        
        # Get dataset checksums
        exp_data = {dv.dataset_name: dv.checksum for dv in experiment.data_versions}
        ref_data = {dv.dataset_name: dv.checksum for dv in reference.data_versions}
        
        # Check for common datasets
        common_datasets = set(exp_data.keys()) & set(ref_data.keys())
        
        for dataset in common_datasets:
            if exp_data[dataset] != ref_data[dataset]:
                validation['valid'] = False
                validation['differences'].append(f"Data checksum mismatch for {dataset}")
        
        # Check for missing datasets
        missing_in_exp = set(ref_data.keys()) - set(exp_data.keys())
        missing_in_ref = set(exp_data.keys()) - set(ref_data.keys())
        
        if missing_in_exp:
            validation['differences'].append(f"Datasets missing in experiment: {missing_in_exp}")
        
        if missing_in_ref:
            validation['differences'].append(f"Datasets missing in reference: {missing_in_ref}")
        
        return validation
    
    def compare_multiple_experiments(self, experiment_ids: List[str],
                                   metrics_to_compare: Optional[List[str]] = None) -> Dict[str, StatisticalComparison]:
        """Compare metrics across multiple experiments statistically."""
        
        experiments = []
        for exp_id in experiment_ids:
            experiments.append(self.experiment_tracker.load_experiment(exp_id))
        
        if len(experiments) < 2:
            raise ValueError("Need at least 2 experiments for comparison")
        
        # Get all metrics if not specified
        if not metrics_to_compare:
            all_metrics = set()
            for exp in experiments:
                all_metrics.update(exp.metrics.metrics.keys())
            metrics_to_compare = list(all_metrics)
        
        comparisons = {}
        
        for metric_name in metrics_to_compare:
            # Collect values for this metric
            values = []
            for exp in experiments:
                if metric_name in exp.metrics.metrics:
                    values.append(exp.metrics.metrics[metric_name])
            
            if len(values) >= 2:
                # Split into reference (first) and test (rest)
                reference_values = [values[0]]
                test_values = values[1:]
                
                comparison = StatisticalComparison(
                    metric_name=metric_name,
                    reference_values=reference_values,
                    test_values=test_values
                )
                
                comparisons[metric_name] = comparison
        
        return comparisons
    
    def save_validation_report(self, report: ValidationReport, output_path: str):
        """Save validation report to file."""
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Validation report saved to {output_path}")
    
    def load_validation_report(self, report_path: str) -> ValidationReport:
        """Load validation report from file."""
        
        with open(report_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Reconstruct ValidationReport
        report = ValidationReport(
            experiment_id=data['experiment_id'],
            reference_experiment_id=data.get('reference_experiment_id'),
            validation_timestamp=data.get('validation_timestamp', datetime.now().isoformat())
        )
        
        # Add metric validations
        for mv_data in data.get('metric_validations', []):
            mv = ValidationMetric(
                name=mv_data['name'],
                expected_value=mv_data['expected_value'],
                actual_value=mv_data['actual_value'],
                tolerance=mv_data.get('tolerance', 1e-6),
                relative_tolerance=mv_data.get('relative_tolerance', 1e-3),
                validation_method=mv_data.get('validation_method', 'relative'),
                metadata=mv_data.get('metadata', {})
            )
            report.add_metric_validation(mv)
        
        # Add statistical comparisons
        for sc_data in data.get('statistical_comparisons', []):
            sc = StatisticalComparison(
                metric_name=sc_data['metric_name'],
                reference_values=sc_data['reference_values'],
                test_values=sc_data['test_values']
            )
            # Override computed values with saved ones
            sc.statistical_tests = sc_data.get('statistical_tests', {})
            sc.summary_stats = sc_data.get('summary_stats', {})
            report.add_statistical_comparison(sc)
        
        # Set other fields
        report.environment_validation = data.get('environment_validation', {})
        report.data_validation = data.get('data_validation', {})
        report.overall_valid = data.get('overall_valid', True)
        report.summary = data.get('summary', {})
        
        return report


# Global result validator instance
_result_validator: Optional[ResultValidator] = None


def get_result_validator() -> ResultValidator:
    """Get the global result validator instance."""
    global _result_validator
    if _result_validator is None:
        _result_validator = ResultValidator()
    return _result_validator


def validate_experiment_results(experiment_id: str, 
                              reference_experiment_id: Optional[str] = None,
                              metric_tolerances: Optional[Dict[str, Dict[str, float]]] = None) -> ValidationReport:
    """Validate experiment results."""
    return get_result_validator().validate_experiment_results(
        experiment_id, reference_experiment_id, metric_tolerances
    )


def compare_experiments_statistically(experiment_ids: List[str],
                                     metrics_to_compare: Optional[List[str]] = None) -> Dict[str, StatisticalComparison]:
    """Compare experiments statistically."""
    return get_result_validator().compare_multiple_experiments(experiment_ids, metrics_to_compare)
