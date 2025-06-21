"""
QeMLflow High Availability Module

This module provides redundancy strategies, disaster recovery,
backup and restore capabilities, and failover mechanisms.
"""

import json
import logging
import shutil
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import psutil
import yaml

logger = logging.getLogger(__name__)

# Export public API
__all__ = [
    'HAStatus', 'BackupInfo', 'FailoverEvent', 'RedundancyManager',
    'DisasterRecoveryManager', 'BackupRestoreManager', 'FailoverManager',
    'HealthMonitor', 'HighAvailabilityManager', 'initialize_ha_system',
    'get_ha_manager', 'shutdown_ha_system'
]


@dataclass
class HAStatus:
    """High availability status."""
    timestamp: datetime
    overall_health: str  # healthy, degraded, critical
    services_status: Dict[str, str]
    redundancy_status: Dict[str, bool]
    backup_status: Dict[str, Any]
    failover_ready: bool
    last_backup: Optional[datetime] = None
    recovery_point: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'overall_health': self.overall_health,
            'services_status': self.services_status,
            'redundancy_status': self.redundancy_status,
            'backup_status': self.backup_status,
            'failover_ready': self.failover_ready,
            'last_backup': self.last_backup.isoformat() if self.last_backup else None,
            'recovery_point': self.recovery_point.isoformat() if self.recovery_point else None
        }


@dataclass
class BackupInfo:
    """Backup information."""
    backup_id: str
    timestamp: datetime
    type: str  # full, incremental
    size_bytes: int
    location: str
    checksum: str
    status: str  # success, failed, in_progress
    retention_until: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'backup_id': self.backup_id,
            'timestamp': self.timestamp.isoformat(),
            'type': self.type,
            'size_bytes': self.size_bytes,
            'location': self.location,
            'checksum': self.checksum,
            'status': self.status,
            'retention_until': self.retention_until.isoformat()
        }


@dataclass
class FailoverEvent:
    """Failover event information."""
    event_id: str
    timestamp: datetime
    trigger: str
    source_node: str
    target_node: str
    service: str
    status: str  # initiated, in_progress, completed, failed
    duration_seconds: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'trigger': self.trigger,
            'source_node': self.source_node,
            'target_node': self.target_node,
            'service': self.service,
            'status': self.status,
            'duration_seconds': self.duration_seconds
        }


class RedundancyManager:
    """Manages service redundancy and load balancing."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.services_status: Dict[str, Dict[str, Any]] = {}
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        
    def start_monitoring(self):
        """Start redundancy monitoring."""
        if self._monitoring:
            return
            
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        self.logger.info("Redundancy monitoring started")
        
    def stop_monitoring(self):
        """Stop redundancy monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        self.logger.info("Redundancy monitoring stopped")
        
    def _monitor_loop(self):
        """Main monitoring loop."""
        interval = self.config.get('redundancy', {}).get('services', {}).get('compute_nodes', {}).get('health_check_interval', 30)
        
        while self._monitoring:
            try:
                self._check_service_health()
                self._ensure_redundancy()
                time.sleep(interval)
            except Exception as e:
                self.logger.error(f"Error in redundancy monitoring: {e}")
                time.sleep(interval)
                
    def _check_service_health(self):
        """Check health of all services."""
        services_config = self.config.get('redundancy', {}).get('services', {})
        
        for service_name, service_config in services_config.items():
            try:
                health_status = self._check_single_service(service_name, service_config)
                self.services_status[service_name] = health_status
            except Exception as e:
                self.logger.error(f"Failed to check health for service {service_name}: {e}")
                self.services_status[service_name] = {'status': 'unhealthy', 'error': str(e)}
                
    def _check_single_service(self, service_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Check health of a single service."""
        # Simulate service health check
        # In a real implementation, this would check actual service endpoints
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.virtual_memory().percent
        
        # Determine health based on resource usage
        if cpu_usage > 90 or memory_usage > 95:
            status = 'critical'
        elif cpu_usage > 80 or memory_usage > 85:
            status = 'degraded'
        else:
            status = 'healthy'
            
        return {
            'status': status,
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage,
            'last_check': datetime.now().isoformat(),
            'instances': config.get('min_instances', 1)
        }
        
    def _ensure_redundancy(self):
        """Ensure minimum redundancy requirements are met."""
        services_config = self.config.get('redundancy', {}).get('services', {})
        
        for service_name, service_config in services_config.items():
            min_instances = service_config.get('min_instances', 1)
            current_status = self.services_status.get(service_name, {})
            current_instances = current_status.get('instances', 0)
            
            if current_instances < min_instances:
                self.logger.warning(f"Service {service_name} has {current_instances} instances, minimum required: {min_instances}")
                self._scale_service(service_name, min_instances)
                
    def _scale_service(self, service_name: str, target_instances: int):
        """Scale service to target number of instances."""
        self.logger.info(f"Scaling service {service_name} to {target_instances} instances")
        # In a real implementation, this would interact with orchestration systems
        # like Kubernetes, Docker Swarm, or cloud auto-scaling groups
        
    def get_redundancy_status(self) -> Dict[str, Any]:
        """Get current redundancy status."""
        return {
            'services': self.services_status,
            'monitoring_active': self._monitoring,
            'last_update': datetime.now().isoformat()
        }


class DisasterRecoveryManager:
    """Manages disaster recovery procedures."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.recovery_procedures: Dict[str, Any] = {}
        
    def create_recovery_plan(self) -> Dict[str, Any]:
        """Create a disaster recovery plan."""
        dr_config = self.config.get('disaster_recovery', {})
        
        plan = {
            'plan_id': f"dr_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'created': datetime.now().isoformat(),
            'rto': dr_config.get('rto', 300),
            'rpo': dr_config.get('rpo', 60),
            'procedures': self._generate_recovery_procedures(),
            'resources': self._identify_critical_resources(),
            'contacts': self._get_emergency_contacts()
        }
        
        self.recovery_procedures[plan['plan_id']] = plan
        return plan
        
    def _generate_recovery_procedures(self) -> List[Dict[str, Any]]:
        """Generate recovery procedures."""
        return [
            {
                'step': 1,
                'title': 'Assess Situation',
                'description': 'Determine scope and impact of the disaster',
                'estimated_time': 300,  # seconds
                'responsible': 'incident_commander'
            },
            {
                'step': 2,
                'title': 'Activate Backup Systems',
                'description': 'Switch to backup infrastructure',
                'estimated_time': 600,
                'responsible': 'system_admin'
            },
            {
                'step': 3,
                'title': 'Restore Data',
                'description': 'Restore data from most recent backup',
                'estimated_time': 1800,
                'responsible': 'data_admin'
            },
            {
                'step': 4,
                'title': 'Validate Recovery',
                'description': 'Test all systems and validate data integrity',
                'estimated_time': 900,
                'responsible': 'qa_team'
            },
            {
                'step': 5,
                'title': 'Resume Operations',
                'description': 'Return to normal operations',
                'estimated_time': 300,
                'responsible': 'incident_commander'
            }
        ]
        
    def _identify_critical_resources(self) -> List[Dict[str, Any]]:
        """Identify critical resources for recovery."""
        return [
            {'type': 'database', 'name': 'primary_db', 'priority': 1, 'backup_location': '/backup/db'},
            {'type': 'application', 'name': 'api_server', 'priority': 1, 'backup_location': '/backup/app'},
            {'type': 'storage', 'name': 'user_data', 'priority': 2, 'backup_location': '/backup/data'},
            {'type': 'configuration', 'name': 'system_config', 'priority': 2, 'backup_location': '/backup/config'}
        ]
        
    def _get_emergency_contacts(self) -> List[Dict[str, str]]:
        """Get emergency contact information."""
        return [
            {'role': 'incident_commander', 'name': 'John Doe', 'phone': '+1-555-0101', 'email': 'john@company.com'},
            {'role': 'system_admin', 'name': 'Jane Smith', 'phone': '+1-555-0102', 'email': 'jane@company.com'},
            {'role': 'data_admin', 'name': 'Bob Johnson', 'phone': '+1-555-0103', 'email': 'bob@company.com'}
        ]
        
    def execute_recovery(self, plan_id: str) -> Dict[str, Any]:
        """Execute a disaster recovery plan."""
        if plan_id not in self.recovery_procedures:
            raise ValueError(f"Recovery plan {plan_id} not found")
            
        plan = self.recovery_procedures[plan_id]
        execution_log = {
            'execution_id': f"exec_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'plan_id': plan_id,
            'started': datetime.now().isoformat(),
            'status': 'in_progress',
            'steps_completed': [],
            'current_step': 1
        }
        
        self.logger.info(f"Starting disaster recovery execution: {execution_log['execution_id']}")
        
        # In a real implementation, this would execute the actual recovery procedures
        # For now, we'll simulate the execution
        for procedure in plan['procedures']:
            step_result = self._execute_recovery_step(procedure)
            execution_log['steps_completed'].append(step_result)
            execution_log['current_step'] = procedure['step'] + 1
            
        execution_log['status'] = 'completed'
        execution_log['completed'] = datetime.now().isoformat()
        
        return execution_log
        
    def _execute_recovery_step(self, procedure: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single recovery step."""
        start_time = datetime.now()
        
        # Simulate step execution
        time.sleep(0.1)  # Simulate work
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        return {
            'step': procedure['step'],
            'title': procedure['title'],
            'status': 'completed',
            'started': start_time.isoformat(),
            'completed': end_time.isoformat(),
            'duration_seconds': duration
        }


class BackupRestoreManager:
    """Manages backup and restore operations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.backup_history: List[BackupInfo] = []
        self._backup_scheduled = False
        self._backup_thread: Optional[threading.Thread] = None
        
    def start_automated_backup(self):
        """Start automated backup process."""
        if self._backup_scheduled:
            return
            
        backup_config = self.config.get('disaster_recovery', {}).get('backup', {})
        if not backup_config.get('automated', False):
            return
            
        self._backup_scheduled = True
        self._backup_thread = threading.Thread(target=self._backup_loop, daemon=True)
        self._backup_thread.start()
        self.logger.info("Automated backup started")
        
    def stop_automated_backup(self):
        """Stop automated backup process."""
        self._backup_scheduled = False
        if self._backup_thread:
            self._backup_thread.join(timeout=5)
        self.logger.info("Automated backup stopped")
        
    def _backup_loop(self):
        """Main backup loop."""
        backup_config = self.config.get('disaster_recovery', {}).get('backup', {})
        frequency = backup_config.get('frequency', 'daily')
        
        # Convert frequency to seconds
        interval_map = {
            'hourly': 3600,
            'daily': 86400,
            'weekly': 604800
        }
        interval = interval_map.get(frequency, 86400)
        
        while self._backup_scheduled:
            try:
                self.create_backup('automated')
                time.sleep(interval)
            except Exception as e:
                self.logger.error(f"Error in automated backup: {e}")
                time.sleep(300)  # Wait 5 minutes before retry
                
    def create_backup(self, backup_type: str = 'manual') -> BackupInfo:
        """Create a backup."""
        backup_id = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        timestamp = datetime.now()
        
        self.logger.info(f"Creating backup: {backup_id}")
        
        try:
            # Simulate backup creation
            backup_data = self._perform_backup(backup_type)
            
            backup_info = BackupInfo(
                backup_id=backup_id,
                timestamp=timestamp,
                type=backup_type,
                size_bytes=backup_data['size'],
                location=backup_data['location'],
                checksum=backup_data['checksum'],
                status='success',
                retention_until=self._calculate_retention_date(backup_type)
            )
            
            self.backup_history.append(backup_info)
            self._cleanup_old_backups()
            
            self.logger.info(f"Backup created successfully: {backup_id}")
            return backup_info
            
        except Exception as e:
            self.logger.error(f"Backup creation failed: {e}")
            backup_info = BackupInfo(
                backup_id=backup_id,
                timestamp=timestamp,
                type=backup_type,
                size_bytes=0,
                location='',
                checksum='',
                status='failed',
                retention_until=timestamp
            )
            self.backup_history.append(backup_info)
            raise
            
    def _perform_backup(self, backup_type: str) -> Dict[str, Any]:
        """Perform the actual backup operation."""
        # Simulate backup process
        backup_size = 1024 * 1024 * 100  # 100MB simulated
        backup_location = f"/backup/{backup_type}/{datetime.now().strftime('%Y%m%d')}"
        checksum = "sha256:abcd1234efgh5678"  # Simulated checksum
        
        # In a real implementation, this would:
        # 1. Create backup directories
        # 2. Copy/compress data
        # 3. Calculate checksums
        # 4. Upload to remote locations
        # 5. Verify backup integrity
        
        return {
            'size': backup_size,
            'location': backup_location,
            'checksum': checksum
        }
        
    def _calculate_retention_date(self, backup_type: str) -> datetime:
        """Calculate backup retention date."""
        retention_map = {
            'automated': 7,    # 7 days
            'manual': 30,      # 30 days
            'full': 90,        # 90 days
            'archive': 365     # 365 days
        }
        
        retention_days = retention_map.get(backup_type, 30)
        return datetime.now() + timedelta(days=retention_days)
        
    def _cleanup_old_backups(self):
        """Clean up expired backups."""
        now = datetime.now()
        expired_backups = [b for b in self.backup_history if b.retention_until < now]
        
        for backup in expired_backups:
            self.logger.info(f"Cleaning up expired backup: {backup.backup_id}")
            self.backup_history.remove(backup)
            # In a real implementation, would also delete the actual backup files
            
    def restore_from_backup(self, backup_id: str, target_location: Optional[str] = None) -> Dict[str, Any]:
        """Restore from a specific backup."""
        backup = next((b for b in self.backup_history if b.backup_id == backup_id), None)
        if not backup:
            raise ValueError(f"Backup {backup_id} not found")
            
        if backup.status != 'success':
            raise ValueError(f"Backup {backup_id} is not in a valid state for restore")
            
        self.logger.info(f"Starting restore from backup: {backup_id}")
        
        restore_result = {
            'restore_id': f"restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'backup_id': backup_id,
            'started': datetime.now().isoformat(),
            'status': 'in_progress'
        }
        
        try:
            # Simulate restore process
            self._perform_restore(backup, target_location)
            
            restore_result['status'] = 'completed'
            restore_result['completed'] = datetime.now().isoformat()
            
            self.logger.info(f"Restore completed successfully: {restore_result['restore_id']}")
            return restore_result
            
        except Exception as e:
            restore_result['status'] = 'failed'
            restore_result['error'] = str(e)
            restore_result['failed'] = datetime.now().isoformat()
            
            self.logger.error(f"Restore failed: {e}")
            raise
            
    def _perform_restore(self, backup: BackupInfo, target_location: Optional[str]):
        """Perform the actual restore operation."""
        # In a real implementation, this would:
        # 1. Verify backup integrity
        # 2. Extract/decompress data
        # 3. Copy to target location
        # 4. Verify restore integrity
        # 5. Update permissions/ownership
        
        time.sleep(0.1)  # Simulate restore work
        
    def get_backup_status(self) -> Dict[str, Any]:
        """Get current backup status."""
        return {
            'automated_backup_active': self._backup_scheduled,
            'total_backups': len(self.backup_history),
            'recent_backups': [b.to_dict() for b in self.backup_history[-5:]],
            'last_backup': self.backup_history[-1].to_dict() if self.backup_history else None
        }


class FailoverManager:
    """Manages failover mechanisms."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.failover_history: List[FailoverEvent] = []
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        
    def start_monitoring(self):
        """Start failover monitoring."""
        if self._monitoring:
            return
            
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        self.logger.info("Failover monitoring started")
        
    def stop_monitoring(self):
        """Stop failover monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        self.logger.info("Failover monitoring stopped")
        
    def _monitor_loop(self):
        """Main failover monitoring loop."""
        while self._monitoring:
            try:
                self._check_failover_conditions()
                time.sleep(10)  # Check every 10 seconds
            except Exception as e:
                self.logger.error(f"Error in failover monitoring: {e}")
                time.sleep(10)
                
    def _check_failover_conditions(self):
        """Check conditions that might trigger failover."""
        failover_config = self.config.get('failover', {})
        if not failover_config.get('automatic', {}).get('health_checks', False):
            return
            
        # Check system health
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.virtual_memory().percent
        
        # Trigger failover if thresholds exceeded
        if cpu_usage > 95 or memory_usage > 98:
            self.logger.warning("Critical resource usage detected, considering failover")
            # In a real implementation, would implement more sophisticated logic
            
    def initiate_failover(self, service: str, reason: str, target_node: Optional[str] = None) -> FailoverEvent:
        """Initiate a failover for a service."""
        event_id = f"failover_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        timestamp = datetime.now()
        
        # Determine target node
        if not target_node:
            target_node = self._select_target_node(service)
            
        failover_event = FailoverEvent(
            event_id=event_id,
            timestamp=timestamp,
            trigger=reason,
            source_node='current_node',  # In real implementation, would be actual node
            target_node=target_node,
            service=service,
            status='initiated',
            duration_seconds=0.0
        )
        
        self.failover_history.append(failover_event)
        self.logger.info(f"Initiating failover: {event_id} for service {service}")
        
        try:
            start_time = time.time()
            self._execute_failover(failover_event)
            end_time = time.time()
            
            failover_event.duration_seconds = end_time - start_time
            failover_event.status = 'completed'
            
            self.logger.info(f"Failover completed successfully: {event_id}")
            return failover_event
            
        except Exception as e:
            failover_event.status = 'failed'
            self.logger.error(f"Failover failed: {e}")
            raise
            
    def _select_target_node(self, service: str) -> str:
        """Select the best target node for failover."""
        # In a real implementation, this would:
        # 1. Query available nodes
        # 2. Check their capacity and health
        # 3. Select the best candidate based on load, proximity, etc.
        
        return f"backup_node_for_{service}"
        
    def _execute_failover(self, event: FailoverEvent):
        """Execute the actual failover process."""
        event.status = 'in_progress'
        
        # Simulate failover steps
        steps = [
            'Validate target node',
            'Stop service on source node',
            'Transfer state to target node',
            'Start service on target node',
            'Update load balancer',
            'Verify service health'
        ]
        
        for step in steps:
            self.logger.info(f"Failover step: {step}")
            time.sleep(0.1)  # Simulate work
            
    def get_failover_status(self) -> Dict[str, Any]:
        """Get current failover status."""
        return {
            'monitoring_active': self._monitoring,
            'recent_failovers': [f.to_dict() for f in self.failover_history[-5:]],
            'total_failovers': len(self.failover_history),
            'automatic_failover_enabled': self.config.get('failover', {}).get('automatic', {}).get('health_checks', False)
        }


class HealthMonitor:
    """Monitors overall system health for HA purposes."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.health_history: List[Dict[str, Any]] = []
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        
    def start_monitoring(self):
        """Start health monitoring."""
        if self._monitoring:
            return
            
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        self.logger.info("Health monitoring started")
        
    def stop_monitoring(self):
        """Stop health monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        self.logger.info("Health monitoring stopped")
        
    def _monitor_loop(self):
        """Main health monitoring loop."""
        while self._monitoring:
            try:
                health_data = self._collect_health_metrics()
                self.health_history.append(health_data)
                
                # Trim history
                if len(self.health_history) > 100:
                    self.health_history.pop(0)
                    
                # Check alerts
                self._check_health_alerts(health_data)
                
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                self.logger.error(f"Error in health monitoring: {e}")
                time.sleep(30)
                
    def _collect_health_metrics(self) -> Dict[str, Any]:
        """Collect current health metrics."""
        return {
            'timestamp': datetime.now().isoformat(),
            'system': {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent,
                'load_average': psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0.0
            },
            'network': {
                'connections': len(psutil.net_connections()),
                'bytes_sent': psutil.net_io_counters().bytes_sent,
                'bytes_recv': psutil.net_io_counters().bytes_recv
            },
            'processes': {
                'total': len(psutil.pids()),
                'running': len([p for p in psutil.process_iter() if p.status() == psutil.STATUS_RUNNING])
            }
        }
        
    def _check_health_alerts(self, health_data: Dict[str, Any]):
        """Check for health-related alerts."""
        health_config = self.config.get('health_monitoring', {})
        system_thresholds = health_config.get('system', {})
        
        system_data = health_data.get('system', {})
        
        # Check CPU threshold
        cpu_threshold = system_thresholds.get('cpu_threshold', 80)
        if system_data.get('cpu_percent', 0) > cpu_threshold:
            self.logger.warning(f"High CPU usage: {system_data['cpu_percent']}%")
            
        # Check Memory threshold
        memory_threshold = system_thresholds.get('memory_threshold', 85)
        if system_data.get('memory_percent', 0) > memory_threshold:
            self.logger.warning(f"High memory usage: {system_data['memory_percent']}%")
            
        # Check Disk threshold
        disk_threshold = system_thresholds.get('disk_threshold', 90)
        if system_data.get('disk_percent', 0) > disk_threshold:
            self.logger.warning(f"High disk usage: {system_data['disk_percent']}%")
            
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        if not self.health_history:
            return {'status': 'no_data', 'monitoring_active': self._monitoring}
            
        latest_health = self.health_history[-1]
        system_data = latest_health.get('system', {})
        
        # Determine overall health
        cpu_percent = system_data.get('cpu_percent', 0)
        memory_percent = system_data.get('memory_percent', 0)
        disk_percent = system_data.get('disk_percent', 0)
        
        if cpu_percent > 90 or memory_percent > 95 or disk_percent > 95:
            overall_health = 'critical'
        elif cpu_percent > 80 or memory_percent > 85 or disk_percent > 90:
            overall_health = 'degraded'
        else:
            overall_health = 'healthy'
            
        return {
            'status': overall_health,
            'monitoring_active': self._monitoring,
            'latest_metrics': latest_health,
            'metrics_count': len(self.health_history)
        }


class HighAvailabilityManager:
    """Main high availability management system."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.redundancy_manager = RedundancyManager(self.config)
        self.disaster_recovery_manager = DisasterRecoveryManager(self.config)
        self.backup_restore_manager = BackupRestoreManager(self.config)
        self.failover_manager = FailoverManager(self.config)
        self.health_monitor = HealthMonitor(self.config)
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load high availability configuration."""
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    loaded_config = yaml.safe_load(f) or {}
                    # Handle nested high_availability config structure
                    if 'high_availability' in loaded_config:
                        return loaded_config['high_availability']
                    return loaded_config
            except Exception as e:
                self.logger.warning(f"Failed to load config from {config_path}: {e}")
        
        # Default configuration
        return {
            'redundancy': {'enabled': True},
            'disaster_recovery': {'enabled': True},
            'backup_restore': {'enabled': True},
            'failover': {'enabled': True},
            'health_monitoring': {'enabled': True}
        }
        
    def start(self):
        """Start high availability management."""
        self.logger.info("Starting high availability management")
        
        if self.config.get('redundancy', {}).get('enabled', False):
            self.redundancy_manager.start_monitoring()
            
        if self.config.get('backup_restore', {}).get('enabled', False):
            self.backup_restore_manager.start_automated_backup()
            
        if self.config.get('failover', {}).get('enabled', False):
            self.failover_manager.start_monitoring()
            
        if self.config.get('health_monitoring', {}).get('enabled', False):
            self.health_monitor.start_monitoring()
            
        self.logger.info("High availability management started")
        
    def stop(self):
        """Stop high availability management."""
        self.logger.info("Stopping high availability management")
        
        self.redundancy_manager.stop_monitoring()
        self.backup_restore_manager.stop_automated_backup()
        self.failover_manager.stop_monitoring()
        self.health_monitor.stop_monitoring()
        
        self.logger.info("High availability management stopped")
        
    def get_ha_status(self) -> HAStatus:
        """Get comprehensive HA status."""
        health_status = self.health_monitor.get_health_status()
        redundancy_status = self.redundancy_manager.get_redundancy_status()
        backup_status = self.backup_restore_manager.get_backup_status()
        failover_status = self.failover_manager.get_failover_status()
        
        # Determine overall health
        overall_health = health_status.get('status', 'unknown')
        
        # Get services status
        services_status = {}
        if 'services' in redundancy_status:
            for service, data in redundancy_status['services'].items():
                services_status[service] = data.get('status', 'unknown')
                
        # Get redundancy status
        redundancy_ready = {
            'monitoring': redundancy_status.get('monitoring_active', False),
            'backup': backup_status.get('automated_backup_active', False),
            'failover': failover_status.get('monitoring_active', False)
        }
        
        # Determine failover readiness
        failover_ready = all(redundancy_ready.values())
        
        # Get last backup time
        last_backup = None
        if backup_status.get('last_backup'):
            last_backup = datetime.fromisoformat(backup_status['last_backup']['timestamp'])
            
        return HAStatus(
            timestamp=datetime.now(),
            overall_health=overall_health,
            services_status=services_status,
            redundancy_status=redundancy_ready,
            backup_status=backup_status,
            failover_ready=failover_ready,
            last_backup=last_backup,
            recovery_point=last_backup  # Simplified - in reality would be more complex
        )
        
    def create_backup(self, backup_type: str = 'manual') -> BackupInfo:
        """Create a backup."""
        return self.backup_restore_manager.create_backup(backup_type)
        
    def restore_backup(self, backup_id: str) -> Dict[str, Any]:
        """Restore from backup."""
        return self.backup_restore_manager.restore_from_backup(backup_id)
        
    def initiate_failover(self, service: str, reason: str) -> FailoverEvent:
        """Initiate failover."""
        return self.failover_manager.initiate_failover(service, reason)
        
    def create_disaster_recovery_plan(self) -> Dict[str, Any]:
        """Create disaster recovery plan."""
        return self.disaster_recovery_manager.create_recovery_plan()
        
    def execute_disaster_recovery(self, plan_id: str) -> Dict[str, Any]:
        """Execute disaster recovery plan."""
        return self.disaster_recovery_manager.execute_recovery(plan_id)


# Global HA manager instance
_ha_manager: Optional[HighAvailabilityManager] = None


def initialize_ha_system(config_path: Optional[str] = None) -> HighAvailabilityManager:
    """Initialize the global high availability system."""
    global _ha_manager
    if _ha_manager is None:
        _ha_manager = HighAvailabilityManager(config_path)
        _ha_manager.start()
    return _ha_manager


def get_ha_manager() -> Optional[HighAvailabilityManager]:
    """Get the global HA manager."""
    return _ha_manager


def shutdown_ha_system():
    """Shutdown the global HA system."""
    global _ha_manager
    if _ha_manager:
        _ha_manager.stop()
        _ha_manager = None


if __name__ == "__main__":
    # Example usage
    manager = initialize_ha_system("config/high_availability.yml")
    
    try:
        # Run for a while
        time.sleep(60)  # 1 minute
        
        # Get status
        status = manager.get_ha_status()
        print(json.dumps(status.to_dict(), indent=2))
        
    finally:
        shutdown_ha_system()
