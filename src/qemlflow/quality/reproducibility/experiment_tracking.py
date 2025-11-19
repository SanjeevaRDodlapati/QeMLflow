"""
Experiment Tracking Module

This module provides comprehensive experiment tracking capabilities for reproducible
scientific computing including experiment logging, data versioning, parameter tracking,
and result comparison.
"""

import hashlib
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import numpy as np

from .environment import EnvironmentFingerprint, capture_environment


@dataclass
class ExperimentParameter:
    """Individual experiment parameter with metadata."""
    
    name: str
    value: Any
    param_type: str
    description: str = ""
    constraints: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Set parameter type automatically if not provided."""
        if not self.param_type:
            self.param_type = type(self.value).__name__
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'value': self.value if self._is_serializable(self.value) else str(self.value),
            'param_type': self.param_type,
            'description': self.description,
            'constraints': self.constraints
        }
    
    @staticmethod
    def _is_serializable(value: Any) -> bool:
        """Check if value is JSON serializable."""
        try:
            json.dumps(value)
            return True
        except (TypeError, ValueError):
            return False


@dataclass
class ExperimentMetrics:
    """Experiment metrics and results."""
    
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    computed_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def add_metric(self, name: str, value: float, metadata: Optional[Dict[str, Any]] = None):
        """Add a metric with optional metadata."""
        self.metrics[name] = value
        if metadata:
            self.metadata[name] = metadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'metrics': self.metrics,
            'metadata': self.metadata,
            'computed_at': self.computed_at
        }


@dataclass
class DataVersion:
    """Data version tracking information."""
    
    dataset_name: str
    version: str
    checksum: str
    file_paths: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class ExperimentRecord:
    """Complete experiment record with all tracking information."""
    
    experiment_id: str
    name: str
    description: str
    parameters: List[ExperimentParameter] = field(default_factory=list)
    metrics: ExperimentMetrics = field(default_factory=ExperimentMetrics)
    data_versions: List[DataVersion] = field(default_factory=list)
    environment_fingerprint: Optional[EnvironmentFingerprint] = None
    status: str = "running"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    duration: float = 0.0
    tags: List[str] = field(default_factory=list)
    artifacts: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Generate experiment ID if not provided."""
        if not self.experiment_id:
            self.experiment_id = str(uuid4())
    
    def add_parameter(self, name: str, value: Any, param_type: str = "", 
                     description: str = "", constraints: Optional[Dict[str, Any]] = None):
        """Add experiment parameter."""
        param = ExperimentParameter(
            name=name,
            value=value,
            param_type=param_type or type(value).__name__,
            description=description,
            constraints=constraints or {}
        )
        self.parameters.append(param)
    
    def add_metric(self, name: str, value: float, metadata: Optional[Dict[str, Any]] = None):
        """Add experiment metric."""
        self.metrics.add_metric(name, value, metadata)
        self.updated_at = datetime.now().isoformat()
    
    def add_data_version(self, dataset_name: str, version: str, file_paths: List[str],
                        metadata: Optional[Dict[str, Any]] = None):
        """Add data version information."""
        checksum = self._calculate_data_checksum(file_paths)
        data_version = DataVersion(
            dataset_name=dataset_name,
            version=version,
            checksum=checksum,
            file_paths=file_paths,
            metadata=metadata or {}
        )
        self.data_versions.append(data_version)
    
    def add_artifact(self, name: str, file_path: str):
        """Add experiment artifact."""
        self.artifacts[name] = file_path
        self.updated_at = datetime.now().isoformat()
    
    def add_tag(self, tag: str):
        """Add experiment tag."""
        if tag not in self.tags:
            self.tags.append(tag)
    
    def set_status(self, status: str):
        """Set experiment status."""
        self.status = status
        self.updated_at = datetime.now().isoformat()
    
    def calculate_duration(self, start_time: float):
        """Calculate experiment duration."""
        self.duration = time.time() - start_time
    
    def _calculate_data_checksum(self, file_paths: List[str]) -> str:
        """Calculate checksum for data files."""
        hasher = hashlib.sha256()
        
        for file_path in sorted(file_paths):
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hasher.update(chunk)
            else:
                # Include filename in hash if file doesn't exist
                hasher.update(file_path.encode('utf-8'))
        
        return hasher.hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'experiment_id': self.experiment_id,
            'name': self.name,
            'description': self.description,
            'parameters': [param.to_dict() for param in self.parameters],
            'metrics': self.metrics.to_dict(),
            'data_versions': [dv.to_dict() for dv in self.data_versions],
            'environment_fingerprint': self.environment_fingerprint.to_dict() if self.environment_fingerprint else None,
            'status': self.status,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'duration': self.duration,
            'tags': self.tags,
            'artifacts': self.artifacts
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentRecord':
        """Create from dictionary."""
        # Convert parameters
        parameters = [
            ExperimentParameter(
                name=p['name'],
                value=p['value'],
                param_type=p['param_type'],
                description=p.get('description', ''),
                constraints=p.get('constraints', {})
            )
            for p in data.get('parameters', [])
        ]
        
        # Convert metrics
        metrics_data = data.get('metrics', {})
        metrics = ExperimentMetrics(
            metrics=metrics_data.get('metrics', {}),
            metadata=metrics_data.get('metadata', {}),
            computed_at=metrics_data.get('computed_at', datetime.now().isoformat())
        )
        
        # Convert data versions
        data_versions = [
            DataVersion(**dv_data) for dv_data in data.get('data_versions', [])
        ]
        
        # Convert environment fingerprint
        env_fp = None
        if data.get('environment_fingerprint'):
            from .environment import EnvironmentFingerprint
            env_fp = EnvironmentFingerprint.from_dict(data['environment_fingerprint'])
        
        return cls(
            experiment_id=data['experiment_id'],
            name=data['name'],
            description=data['description'],
            parameters=parameters,
            metrics=metrics,
            data_versions=data_versions,
            environment_fingerprint=env_fp,
            status=data.get('status', 'running'),
            created_at=data.get('created_at', datetime.now().isoformat()),
            updated_at=data.get('updated_at', datetime.now().isoformat()),
            duration=data.get('duration', 0.0),
            tags=data.get('tags', []),
            artifacts=data.get('artifacts', {})
        )


class ExperimentTracker:
    """
    Comprehensive experiment tracking system for reproducible scientific computing.
    """
    
    def __init__(self, base_dir: str = "experiments"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True, parents=True)
        
        # Create subdirectories
        self.experiments_dir = self.base_dir / "records"
        self.artifacts_dir = self.base_dir / "artifacts"
        self.data_dir = self.base_dir / "data"
        
        for dir_path in [self.experiments_dir, self.artifacts_dir, self.data_dir]:
            dir_path.mkdir(exist_ok=True, parents=True)
        
        self.logger = logging.getLogger(__name__)
        self.current_experiment: Optional[ExperimentRecord] = None
        self.start_time: Optional[float] = None
    
    def start_experiment(self, name: str, description: str = "", 
                        experiment_id: Optional[str] = None,
                        capture_env: bool = True) -> str:
        """Start a new experiment."""
        
        if self.current_experiment:
            self.logger.warning("Previous experiment still running. Ending it first.")
            self.end_experiment("interrupted")
        
        # Create new experiment record
        self.current_experiment = ExperimentRecord(
            experiment_id=experiment_id or str(uuid4()),
            name=name,
            description=description
        )
        
        # Capture environment if requested
        if capture_env:
            try:
                env_fp = capture_environment()
                self.current_experiment.environment_fingerprint = env_fp
                self.logger.info(f"Captured environment fingerprint: {env_fp.fingerprint_hash[:16]}...")
            except Exception as e:
                self.logger.warning(f"Failed to capture environment: {e}")
        
        self.start_time = time.time()
        
        self.logger.info(f"Started experiment: {name} ({self.current_experiment.experiment_id})")
        return self.current_experiment.experiment_id
    
    def log_parameter(self, name: str, value: Any, description: str = "",
                     constraints: Optional[Dict[str, Any]] = None):
        """Log experiment parameter."""
        if not self.current_experiment:
            raise RuntimeError("No active experiment. Call start_experiment() first.")
        
        self.current_experiment.add_parameter(name, value, description=description, constraints=constraints)
        self.logger.debug(f"Logged parameter: {name} = {value}")
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None,
                  metadata: Optional[Dict[str, Any]] = None):
        """Log experiment metric."""
        if not self.current_experiment:
            raise RuntimeError("No active experiment. Call start_experiment() first.")
        
        # Add step to metadata if provided
        if step is not None:
            metadata = metadata or {}
            metadata['step'] = step
        
        self.current_experiment.add_metric(name, value, metadata)
        self.logger.debug(f"Logged metric: {name} = {value}")
    
    def log_data_version(self, dataset_name: str, version: str, file_paths: List[str],
                        metadata: Optional[Dict[str, Any]] = None):
        """Log data version information."""
        if not self.current_experiment:
            raise RuntimeError("No active experiment. Call start_experiment() first.")
        
        self.current_experiment.add_data_version(dataset_name, version, file_paths, metadata)
        self.logger.info(f"Logged data version: {dataset_name} v{version}")
    
    def log_artifact(self, name: str, file_path: str, copy_to_artifacts: bool = True):
        """Log experiment artifact."""
        if not self.current_experiment:
            raise RuntimeError("No active experiment. Call start_experiment() first.")
        
        source_path = Path(file_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Artifact file not found: {file_path}")
        
        if copy_to_artifacts:
            # Copy artifact to experiment artifacts directory
            exp_artifacts_dir = self.artifacts_dir / self.current_experiment.experiment_id
            exp_artifacts_dir.mkdir(exist_ok=True)
            
            dest_path = exp_artifacts_dir / source_path.name
            shutil.copy2(source_path, dest_path)
            artifact_path = str(dest_path)
        else:
            artifact_path = str(source_path.absolute())
        
        self.current_experiment.add_artifact(name, artifact_path)
        self.logger.info(f"Logged artifact: {name} -> {artifact_path}")
    
    def add_tag(self, tag: str):
        """Add tag to current experiment."""
        if not self.current_experiment:
            raise RuntimeError("No active experiment. Call start_experiment() first.")
        
        self.current_experiment.add_tag(tag)
        self.logger.debug(f"Added tag: {tag}")
    
    def end_experiment(self, status: str = "completed") -> ExperimentRecord:
        """End current experiment and save record."""
        if not self.current_experiment:
            raise RuntimeError("No active experiment to end.")
        
        # Calculate duration
        if self.start_time:
            self.current_experiment.calculate_duration(self.start_time)
        
        # Set final status
        self.current_experiment.set_status(status)
        
        # Save experiment record
        self._save_experiment_record(self.current_experiment)
        
        experiment_id = self.current_experiment.experiment_id
        experiment_name = self.current_experiment.name
        
        # Clear current experiment
        current_exp = self.current_experiment
        self.current_experiment = None
        self.start_time = None
        
        self.logger.info(f"Ended experiment: {experiment_name} ({experiment_id}) - Status: {status}")
        return current_exp
    
    def _save_experiment_record(self, experiment: ExperimentRecord):
        """Save experiment record to disk."""
        record_file = self.experiments_dir / f"{experiment.experiment_id}.json"
        
        with open(record_file, 'w', encoding='utf-8') as f:
            json.dump(experiment.to_dict(), f, indent=2, ensure_ascii=False)
        
        self.logger.debug(f"Saved experiment record: {record_file}")
    
    def load_experiment(self, experiment_id: str) -> ExperimentRecord:
        """Load experiment record by ID."""
        record_file = self.experiments_dir / f"{experiment_id}.json"
        
        if not record_file.exists():
            raise FileNotFoundError(f"Experiment record not found: {experiment_id}")
        
        with open(record_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return ExperimentRecord.from_dict(data)
    
    def list_experiments(self, tags: Optional[List[str]] = None,
                        status: Optional[str] = None,
                        limit: Optional[int] = None) -> List[ExperimentRecord]:
        """List experiments with optional filtering."""
        experiments = []
        
        for record_file in sorted(self.experiments_dir.glob("*.json"), reverse=True):
            try:
                experiment = self.load_experiment(record_file.stem)
                
                # Apply filters
                if tags and not any(tag in experiment.tags for tag in tags):
                    continue
                
                if status and experiment.status != status:
                    continue
                
                experiments.append(experiment)
                
                # Apply limit
                if limit and len(experiments) >= limit:
                    break
                    
            except Exception as e:
                self.logger.warning(f"Failed to load experiment {record_file.stem}: {e}")
        
        return experiments
    
    def compare_experiments(self, experiment_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple experiments."""
        experiments = []
        
        for exp_id in experiment_ids:
            try:
                experiments.append(self.load_experiment(exp_id))
            except FileNotFoundError:
                self.logger.warning(f"Experiment not found: {exp_id}")
        
        if len(experiments) < 2:
            raise ValueError("Need at least 2 experiments to compare")
        
        comparison = {
            'experiment_ids': experiment_ids,
            'experiments': [exp.to_dict() for exp in experiments],
            'parameter_comparison': self._compare_parameters(experiments),
            'metric_comparison': self._compare_metrics(experiments),
            'environment_comparison': self._compare_environments(experiments),
            'data_comparison': self._compare_data_versions(experiments)
        }
        
        return comparison
    
    def _compare_parameters(self, experiments: List[ExperimentRecord]) -> Dict[str, Any]:
        """Compare parameters across experiments."""
        all_param_names = set()
        for exp in experiments:
            all_param_names.update(param.name for param in exp.parameters)
        
        comparison = {}
        for param_name in all_param_names:
            values = []
            for exp in experiments:
                param_value = None
                for param in exp.parameters:
                    if param.name == param_name:
                        param_value = param.value
                        break
                values.append(param_value)
            
            comparison[param_name] = {
                'values': values,
                'consistent': len(set(str(v) for v in values)) == 1
            }
        
        return comparison
    
    def _compare_metrics(self, experiments: List[ExperimentRecord]) -> Dict[str, Any]:
        """Compare metrics across experiments."""
        all_metric_names = set()
        for exp in experiments:
            all_metric_names.update(exp.metrics.metrics.keys())
        
        comparison = {}
        for metric_name in all_metric_names:
            values = []
            for exp in experiments:
                values.append(exp.metrics.metrics.get(metric_name))
            
            # Calculate statistics for numeric values
            numeric_values = [v for v in values if v is not None and isinstance(v, (int, float))]
            
            comparison[metric_name] = {
                'values': values,
                'statistics': {
                    'min': min(numeric_values) if numeric_values else None,
                    'max': max(numeric_values) if numeric_values else None,
                    'mean': np.mean(numeric_values) if numeric_values else None,
                    'std': np.std(numeric_values) if numeric_values else None
                } if numeric_values else None
            }
        
        return comparison
    
    def _compare_environments(self, experiments: List[ExperimentRecord]) -> Dict[str, Any]:
        """Compare environment fingerprints across experiments."""
        env_hashes = []
        python_versions = []
        platforms = []
        
        for exp in experiments:
            if exp.environment_fingerprint:
                env_hashes.append(exp.environment_fingerprint.fingerprint_hash)
                python_versions.append(exp.environment_fingerprint.python_version)
                platforms.append(exp.environment_fingerprint.platform_info.get('system'))
            else:
                env_hashes.append(None)
                python_versions.append(None)
                platforms.append(None)
        
        return {
            'environment_hashes': env_hashes,
            'python_versions': python_versions,
            'platforms': platforms,
            'environments_identical': len(set(h for h in env_hashes if h)) <= 1,
            'python_versions_consistent': len(set(v for v in python_versions if v)) <= 1,
            'platforms_consistent': len(set(p for p in platforms if p)) <= 1
        }
    
    def _compare_data_versions(self, experiments: List[ExperimentRecord]) -> Dict[str, Any]:
        """Compare data versions across experiments."""
        all_datasets = set()
        for exp in experiments:
            all_datasets.update(dv.dataset_name for dv in exp.data_versions)
        
        comparison = {}
        for dataset_name in all_datasets:
            versions = []
            checksums = []
            
            for exp in experiments:
                dataset_version = None
                dataset_checksum = None
                
                for dv in exp.data_versions:
                    if dv.dataset_name == dataset_name:
                        dataset_version = dv.version
                        dataset_checksum = dv.checksum
                        break
                
                versions.append(dataset_version)
                checksums.append(dataset_checksum)
            
            comparison[dataset_name] = {
                'versions': versions,
                'checksums': checksums,
                'data_consistent': len(set(c for c in checksums if c)) <= 1
            }
        
        return comparison
    
    def get_experiment_summary(self, experiment_id: str) -> Dict[str, Any]:
        """Get experiment summary."""
        experiment = self.load_experiment(experiment_id)
        
        return {
            'id': experiment.experiment_id,
            'name': experiment.name,
            'status': experiment.status,
            'duration': experiment.duration,
            'parameter_count': len(experiment.parameters),
            'metric_count': len(experiment.metrics.metrics),
            'data_version_count': len(experiment.data_versions),
            'artifact_count': len(experiment.artifacts),
            'tag_count': len(experiment.tags),
            'has_environment': experiment.environment_fingerprint is not None,
            'created_at': experiment.created_at,
            'updated_at': experiment.updated_at
        }


# Global experiment tracker instance
_experiment_tracker: Optional[ExperimentTracker] = None


def get_experiment_tracker() -> ExperimentTracker:
    """Get the global experiment tracker instance."""
    global _experiment_tracker
    if _experiment_tracker is None:
        _experiment_tracker = ExperimentTracker()
    return _experiment_tracker


def start_experiment(name: str, description: str = "", experiment_id: Optional[str] = None,
                    capture_env: bool = True) -> str:
    """Start a new experiment."""
    return get_experiment_tracker().start_experiment(name, description, experiment_id, capture_env)


def log_parameter(name: str, value: Any, description: str = "",
                 constraints: Optional[Dict[str, Any]] = None):
    """Log experiment parameter."""
    get_experiment_tracker().log_parameter(name, value, description, constraints)


def log_metric(name: str, value: float, step: Optional[int] = None,
              metadata: Optional[Dict[str, Any]] = None):
    """Log experiment metric."""
    get_experiment_tracker().log_metric(name, value, step, metadata)


def log_data_version(dataset_name: str, version: str, file_paths: List[str],
                    metadata: Optional[Dict[str, Any]] = None):
    """Log data version information."""
    get_experiment_tracker().log_data_version(dataset_name, version, file_paths, metadata)


def log_artifact(name: str, file_path: str, copy_to_artifacts: bool = True):
    """Log experiment artifact."""
    get_experiment_tracker().log_artifact(name, file_path, copy_to_artifacts)


def add_tag(tag: str):
    """Add tag to current experiment."""
    get_experiment_tracker().add_tag(tag)


def end_experiment(status: str = "completed") -> ExperimentRecord:
    """End current experiment."""
    return get_experiment_tracker().end_experiment(status)


def list_experiments(tags: Optional[List[str]] = None, status: Optional[str] = None,
                    limit: Optional[int] = None) -> List[ExperimentRecord]:
    """List experiments with optional filtering."""
    return get_experiment_tracker().list_experiments(tags, status, limit)


def compare_experiments(experiment_ids: List[str]) -> Dict[str, Any]:
    """Compare multiple experiments."""
    return get_experiment_tracker().compare_experiments(experiment_ids)
