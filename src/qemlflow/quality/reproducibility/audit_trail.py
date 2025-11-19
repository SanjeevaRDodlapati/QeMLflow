"""
Audit Trail System Module

This module provides comprehensive audit trail capabilities for scientific computing
including computational workflow tracking, data lineage, audit log analysis, and
compliance reporting for regulatory environments.
"""

import hashlib
import json
import logging
import os
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from contextlib import contextmanager
from functools import wraps
from uuid import uuid4

import psutil


@dataclass
class AuditEvent:
    """Individual audit event with complete tracking information."""
    
    event_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    event_type: str = ""
    user_id: str = ""
    session_id: str = ""
    process_id: int = field(default_factory=os.getpid)
    thread_id: int = field(default_factory=threading.get_ident)
    
    # Core event information
    action: str = ""
    resource: str = ""
    resource_type: str = ""
    resource_id: str = ""
    
    # Context information
    function_name: str = ""
    module_name: str = ""
    file_path: str = ""
    line_number: int = 0
    
    # Input/output tracking
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # System context
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    
    # Execution tracking
    execution_time: float = 0.0
    status: str = "success"  # success, failure, warning
    error_message: str = ""
    stack_trace: str = ""
    
    # Compliance and metadata
    compliance_tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    checksum: str = ""
    
    def __post_init__(self):
        """Calculate checksum after initialization."""
        self.checksum = self._calculate_checksum()
    
    def _calculate_checksum(self) -> str:
        """Calculate event integrity checksum."""
        # Create a deterministic representation
        data = {
            'timestamp': self.timestamp,
            'action': self.action,
            'resource': self.resource,
            'inputs': self._serialize_data(self.inputs),
            'outputs': self._serialize_data(self.outputs),
            'parameters': self._serialize_data(self.parameters)
        }
        
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    
    def _serialize_data(self, data: Any) -> str:
        """Safely serialize data for checksum calculation."""
        try:
            return json.dumps(data, sort_keys=True, default=str)
        except (TypeError, ValueError):
            return str(data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuditEvent':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class DataLineage:
    """Data lineage tracking for complete data provenance."""
    
    data_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    path: str = ""
    checksum: str = ""
    size: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    # Lineage relationships
    parent_data: List[str] = field(default_factory=list)  # Data IDs this depends on
    child_data: List[str] = field(default_factory=list)   # Data IDs that depend on this
    
    # Transformation information
    transformation: str = ""  # Function/process that created this data
    transformation_params: Dict[str, Any] = field(default_factory=dict)
    transformation_code: str = ""
    
    # Provenance metadata
    creator: str = ""
    purpose: str = ""
    quality_score: float = 1.0
    validation_status: str = "pending"
    
    # Compliance information
    sensitivity_level: str = "public"  # public, internal, confidential, restricted
    retention_policy: str = ""
    compliance_tags: List[str] = field(default_factory=list)
    
    def add_parent(self, parent_id: str):
        """Add parent data dependency."""
        if parent_id not in self.parent_data:
            self.parent_data.append(parent_id)
    
    def add_child(self, child_id: str):
        """Add child data dependency."""
        if child_id not in self.child_data:
            self.child_data.append(child_id)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class WorkflowStep:
    """Individual step in a computational workflow."""
    
    step_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    function_name: str = ""
    module_name: str = ""
    
    # Execution information
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: str = ""
    duration: float = 0.0
    status: str = "running"  # running, completed, failed, skipped
    
    # Dependencies
    depends_on: List[str] = field(default_factory=list)  # Step IDs
    dependencies_met: bool = False
    
    # Input/output tracking
    input_data: List[str] = field(default_factory=list)  # Data IDs
    output_data: List[str] = field(default_factory=list)  # Data IDs
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Resource usage
    cpu_time: float = 0.0
    memory_peak: float = 0.0
    disk_io: float = 0.0
    
    # Error handling
    error_message: str = ""
    retry_count: int = 0
    max_retries: int = 3
    
    def complete(self, status: str = "completed", error_message: str = ""):
        """Mark step as completed."""
        self.completed_at = datetime.now(timezone.utc).isoformat()
        self.status = status
        self.error_message = error_message
        
        # Calculate duration
        if self.started_at:
            start_time = datetime.fromisoformat(self.started_at.replace('Z', '+00:00'))
            end_time = datetime.fromisoformat(self.completed_at.replace('Z', '+00:00'))
            self.duration = (end_time - start_time).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class AuditTrailManager:
    """Comprehensive audit trail management system."""
    
    def __init__(self, audit_dir: str = "audit_logs", 
                 session_id: Optional[str] = None,
                 user_id: Optional[str] = None):
        self.audit_dir = Path(audit_dir)
        self.audit_dir.mkdir(exist_ok=True, parents=True)
        
        self.session_id = session_id or str(uuid4())
        self.user_id = user_id or os.getenv('USER', 'unknown')
        
        # Create subdirectories
        self.events_dir = self.audit_dir / "events"
        self.lineage_dir = self.audit_dir / "lineage"
        self.workflows_dir = self.audit_dir / "workflows"
        self.reports_dir = self.audit_dir / "reports"
        
        for dir_path in [self.events_dir, self.lineage_dir, self.workflows_dir, self.reports_dir]:
            dir_path.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # In-memory caches
        self.current_workflow: Optional[str] = None
        self.workflow_steps: Dict[str, WorkflowStep] = {}
        self.data_lineage: Dict[str, DataLineage] = {}
        
        # Performance tracking
        self.process = psutil.Process()
        
        # Thread safety
        self.lock = threading.Lock()
        
        self.logger.info(f"Audit trail manager initialized: session={self.session_id}")
    
    def log_event(self, action: str, resource: str = "", 
                  resource_type: str = "", **kwargs) -> str:
        """Log an audit event."""
        
        # Get system metrics
        cpu_percent = self.process.cpu_percent()
        memory_info = self.process.memory_info()
        memory_percent = memory_info.rss / (1024 * 1024)  # MB
        
        # Get call stack information
        frame = sys._getframe(1)
        function_name = frame.f_code.co_name
        module_name = frame.f_globals.get('__name__', 'unknown')
        file_path = frame.f_code.co_filename
        line_number = frame.f_lineno
        
        event = AuditEvent(
            event_type="audit_event",
            user_id=self.user_id,
            session_id=self.session_id,
            action=action,
            resource=resource,
            resource_type=resource_type,
            function_name=function_name,
            module_name=module_name,
            file_path=file_path,
            line_number=line_number,
            cpu_usage=cpu_percent,
            memory_usage=memory_percent,
            **kwargs
        )
        
        # Save event
        self._save_event(event)
        
        return event.event_id
    
    def _save_event(self, event: AuditEvent):
        """Save audit event to disk."""
        with self.lock:
            try:
                # Create daily log file
                date_str = datetime.now().strftime("%Y-%m-%d")
                log_file = self.events_dir / f"audit_{date_str}.jsonl"
                
                with open(log_file, 'a') as f:
                    f.write(json.dumps(event.to_dict()) + '\n')
                    
            except Exception as e:
                self.logger.error(f"Failed to save audit event: {e}")
    
    def track_data_lineage(self, data_id: str, name: str, path: str = "", 
                          parent_data: List[str] = None,
                          transformation: str = "",
                          **kwargs) -> DataLineage:
        """Track data lineage information."""
        
        # Calculate file checksum if path provided
        checksum = ""
        size = 0
        if path and os.path.exists(path):
            checksum = self._calculate_file_checksum(path)
            size = os.path.getsize(path)
        
        lineage = DataLineage(
            data_id=data_id,
            name=name,
            path=path,
            checksum=checksum,
            size=size,
            parent_data=parent_data or [],
            transformation=transformation,
            creator=self.user_id,
            **kwargs
        )
        
        # Update parent-child relationships
        for parent_id in lineage.parent_data:
            if parent_id in self.data_lineage:
                self.data_lineage[parent_id].add_child(data_id)
        
        self.data_lineage[data_id] = lineage
        
        # Save lineage
        self._save_lineage(lineage)
        
        # Log event
        self.log_event(
            action="data_created",
            resource=name,
            resource_type="data",
            resource_id=data_id,
            inputs={"parent_data": parent_data or []},
            metadata={"transformation": transformation, "checksum": checksum}
        )
        
        return lineage
    
    def _calculate_file_checksum(self, file_path: str) -> str:
        """Calculate SHA256 checksum of a file."""
        hasher = hashlib.sha256()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            self.logger.warning(f"Failed to calculate checksum for {file_path}: {e}")
            return ""
    
    def _save_lineage(self, lineage: DataLineage):
        """Save data lineage to disk."""
        try:
            lineage_file = self.lineage_dir / f"{lineage.data_id}.json"
            with open(lineage_file, 'w') as f:
                json.dump(lineage.to_dict(), f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save lineage: {e}")
    
    def start_workflow(self, workflow_name: str, workflow_id: Optional[str] = None) -> str:
        """Start tracking a computational workflow."""
        workflow_id = workflow_id or str(uuid4())
        self.current_workflow = workflow_id
        
        self.log_event(
            action="workflow_started",
            resource=workflow_name,
            resource_type="workflow",
            resource_id=workflow_id
        )
        
        return workflow_id
    
    def add_workflow_step(self, step_name: str, function_name: str = "",
                         depends_on: List[str] = None,
                         parameters: Dict[str, Any] = None) -> str:
        """Add a step to the current workflow."""
        
        frame = sys._getframe(1)
        module_name = frame.f_globals.get('__name__', 'unknown')
        function_name = function_name or frame.f_code.co_name
        
        step = WorkflowStep(
            name=step_name,
            function_name=function_name,
            module_name=module_name,
            depends_on=depends_on or [],
            parameters=parameters or {}
        )
        
        self.workflow_steps[step.step_id] = step
        
        self.log_event(
            action="workflow_step_started",
            resource=step_name,
            resource_type="workflow_step",
            resource_id=step.step_id,
            parameters=parameters or {}
        )
        
        return step.step_id
    
    def complete_workflow_step(self, step_id: str, status: str = "completed",
                              output_data: List[str] = None,
                              error_message: str = ""):
        """Complete a workflow step."""
        if step_id in self.workflow_steps:
            step = self.workflow_steps[step_id]
            step.complete(status, error_message)
            step.output_data = output_data or []
            
            # Save workflow step
            self._save_workflow_step(step)
            
            self.log_event(
                action="workflow_step_completed",
                resource=step.name,
                resource_type="workflow_step",
                resource_id=step_id,
                status=status,
                execution_time=step.duration,
                outputs={"output_data": output_data or []},
                error_message=error_message
            )
    
    def _save_workflow_step(self, step: WorkflowStep):
        """Save workflow step to disk."""
        try:
            if self.current_workflow:
                workflow_dir = self.workflows_dir / self.current_workflow
                workflow_dir.mkdir(exist_ok=True)
                
                step_file = workflow_dir / f"{step.step_id}.json"
                with open(step_file, 'w') as f:
                    json.dump(step.to_dict(), f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save workflow step: {e}")
    
    def get_data_lineage_tree(self, data_id: str, max_depth: int = 10) -> Dict[str, Any]:
        """Get complete lineage tree for a data item."""
        def build_tree(current_id: str, depth: int = 0) -> Dict[str, Any]:
            if depth > max_depth or current_id not in self.data_lineage:
                return {}
            
            lineage = self.data_lineage[current_id]
            tree = {
                'data_id': current_id,
                'name': lineage.name,
                'transformation': lineage.transformation,
                'created_at': lineage.created_at,
                'parents': [],
                'children': []
            }
            
            # Add parent trees
            for parent_id in lineage.parent_data:
                parent_tree = build_tree(parent_id, depth + 1)
                if parent_tree:
                    tree['parents'].append(parent_tree)
            
            # Add child trees (limited to avoid circular references)
            if depth < 3:  # Limit child depth
                for child_id in lineage.child_data:
                    child_tree = build_tree(child_id, depth + 1)
                    if child_tree:
                        tree['children'].append(child_tree)
            
            return tree
        
        return build_tree(data_id)
    
    def generate_compliance_report(self, start_date: Optional[str] = None,
                                  end_date: Optional[str] = None,
                                  compliance_tags: List[str] = None) -> Dict[str, Any]:
        """Generate compliance audit report."""
        
        # Load events within date range
        events = self._load_events_in_range(start_date, end_date)
        
        # Filter by compliance tags if specified
        if compliance_tags:
            events = [e for e in events if any(tag in e.get('compliance_tags', []) for tag in compliance_tags)]
        
        report = {
            'report_id': str(uuid4()),
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'period': {
                'start_date': start_date,
                'end_date': end_date
            },
            'summary': {
                'total_events': len(events),
                'unique_users': len(set(e.get('user_id', '') for e in events)),
                'unique_sessions': len(set(e.get('session_id', '') for e in events)),
                'event_types': {}
            },
            'compliance_metrics': {
                'data_access_events': 0,
                'data_modification_events': 0,
                'failed_operations': 0,
                'security_events': 0
            },
            'violations': [],
            'recommendations': []
        }
        
        # Analyze events
        for event in events:
            action = event.get('action', '')
            status = event.get('status', 'success')
            
            # Count event types
            report['summary']['event_types'][action] = report['summary']['event_types'].get(action, 0) + 1
            
            # Compliance metrics
            if 'data' in action:
                if 'read' in action or 'access' in action:
                    report['compliance_metrics']['data_access_events'] += 1
                elif 'write' in action or 'modify' in action or 'delete' in action:
                    report['compliance_metrics']['data_modification_events'] += 1
            
            if status == 'failure':
                report['compliance_metrics']['failed_operations'] += 1
            
            if 'security' in action or 'auth' in action:
                report['compliance_metrics']['security_events'] += 1
        
        # Save report
        report_file = self.reports_dir / f"compliance_report_{report['report_id']}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def _load_events_in_range(self, start_date: Optional[str] = None,
                             end_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load audit events within date range."""
        events = []
        
        try:
            # Get all event files
            for event_file in self.events_dir.glob("audit_*.jsonl"):
                with open(event_file, 'r') as f:
                    for line in f:
                        try:
                            event = json.loads(line.strip())
                            
                            # Filter by date range if specified
                            if start_date or end_date:
                                event_date = event.get('timestamp', '')
                                if start_date and event_date < start_date:
                                    continue
                                if end_date and event_date > end_date:
                                    continue
                            
                            events.append(event)
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            self.logger.error(f"Failed to load events: {e}")
        
        return events


# Decorator for automatic audit trail tracking
def audit_trail(action: str = "", resource_type: str = "function"):
    """Decorator to automatically track function calls in audit trail."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get audit manager from global context or create one
            audit_manager = getattr(wrapper, '_audit_manager', None)
            if not audit_manager:
                audit_manager = AuditTrailManager()
                wrapper._audit_manager = audit_manager
            
            # Start tracking
            start_time = time.time()
            action_name = action or f"{func.__module__}.{func.__name__}"
            
            event_id = audit_manager.log_event(
                action=f"{action_name}_started",
                resource=func.__name__,
                resource_type=resource_type,
                parameters={"args": str(args)[:1000], "kwargs": str(kwargs)[:1000]}
            )
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                
                # Log completion
                execution_time = time.time() - start_time
                audit_manager.log_event(
                    action=f"{action_name}_completed",
                    resource=func.__name__,
                    resource_type=resource_type,
                    execution_time=execution_time,
                    status="success",
                    outputs={"result_type": type(result).__name__}
                )
                
                return result
                
            except Exception as e:
                # Log failure
                execution_time = time.time() - start_time
                audit_manager.log_event(
                    action=f"{action_name}_failed",
                    resource=func.__name__,
                    resource_type=resource_type,
                    execution_time=execution_time,
                    status="failure",
                    error_message=str(e),
                    stack_trace=traceback.format_exc()
                )
                raise
        
        return wrapper
    return decorator


# Context manager for workflow tracking
@contextmanager
def audit_workflow(workflow_name: str, audit_manager: Optional[AuditTrailManager] = None):
    """Context manager for tracking computational workflows."""
    if not audit_manager:
        audit_manager = AuditTrailManager()
    
    workflow_id = audit_manager.start_workflow(workflow_name)
    
    try:
        yield audit_manager, workflow_id
        
        audit_manager.log_event(
            action="workflow_completed",
            resource=workflow_name,
            resource_type="workflow",
            resource_id=workflow_id,
            status="success"
        )
        
    except Exception as e:
        audit_manager.log_event(
            action="workflow_failed",
            resource=workflow_name,
            resource_type="workflow",
            resource_id=workflow_id,
            status="failure",
            error_message=str(e)
        )
        raise


# Global audit manager instance
_global_audit_manager = None

def get_audit_manager() -> AuditTrailManager:
    """Get global audit manager instance."""
    global _global_audit_manager
    if _global_audit_manager is None:
        audit_dir = os.getenv('QEMLFLOW_AUDIT_DIR', 'audit_logs')
        _global_audit_manager = AuditTrailManager(audit_dir=audit_dir)
    return _global_audit_manager


# Convenience functions
def log_audit_event(action: str, resource: str = "", **kwargs) -> str:
    """Log an audit event using global manager."""
    return get_audit_manager().log_event(action, resource, **kwargs)


def track_data_lineage(data_id: str, name: str, **kwargs) -> DataLineage:
    """Track data lineage using global manager."""
    return get_audit_manager().track_data_lineage(data_id, name, **kwargs)
