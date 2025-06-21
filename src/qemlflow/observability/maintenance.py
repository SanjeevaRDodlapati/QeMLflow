"""
Automated Maintenance Module

This module provides automated maintenance capabilities including dependency updates,
security patch automation, health-based scaling, and automated cleanup processes
for enterprise-grade system maintainability.
"""

import json
import logging
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from threading import Thread, Event, Lock
from typing import Dict, List, Optional, Any
try:
    import yaml
except ImportError:
    yaml = None  # type: ignore


@dataclass
class MaintenanceTask:
    """Represents a maintenance task to be executed."""
    task_id: str
    task_type: str  # 'dependency_update', 'security_patch', 'cleanup', 'scaling'
    priority: str = "medium"  # 'low', 'medium', 'high', 'critical'
    description: str = ""
    schedule: str = "manual"  # 'manual', 'daily', 'weekly', 'monthly', cron expression
    enabled: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    run_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    last_result: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary."""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
        return data


@dataclass
class MaintenanceResult:
    """Result of a maintenance task execution."""
    task_id: str
    status: str  # 'success', 'failure', 'skipped', 'partial'
    message: str
    duration_seconds: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    details: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        data = asdict(self)
        if isinstance(data['timestamp'], datetime):
            data['timestamp'] = data['timestamp'].isoformat()
        return data


@dataclass
class SystemHealth:
    """Current system health status."""
    cpu_usage_percent: float
    memory_usage_percent: float
    disk_usage_percent: float
    active_connections: int
    response_time_ms: float
    error_rate_percent: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    healthy: bool = True
    issues: List[str] = field(default_factory=list)


class MaintenanceExecutor(ABC):
    """Abstract base class for maintenance task executors."""
    
    @abstractmethod
    def execute(self, task: MaintenanceTask) -> MaintenanceResult:
        """Execute a maintenance task."""
        pass
    
    @abstractmethod
    def can_execute(self, task: MaintenanceTask) -> bool:
        """Check if this executor can handle the given task."""
        pass


class DependencyUpdateExecutor(MaintenanceExecutor):
    """Executes dependency update tasks."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.logger = logging.getLogger(__name__)
    
    def can_execute(self, task: MaintenanceTask) -> bool:
        """Check if this is a dependency update task."""
        return task.task_type == "dependency_update"
    
    def execute(self, task: MaintenanceTask) -> MaintenanceResult:
        """Execute dependency update task."""
        start_time = time.time()
        result = MaintenanceResult(
            task_id=task.task_id,
            status="failure",
            message="",
            duration_seconds=0
        )
        
        try:
            # Check for requirements files
            req_files = []
            for req_file in ["requirements.txt", "requirements-core.txt", "pyproject.toml"]:
                req_path = self.project_root / req_file
                if req_path.exists():
                    req_files.append(str(req_path))
            
            if not req_files:
                result.status = "skipped"
                result.message = "No dependency files found"
                return result
            
            updates = []
            
            # Update pip packages
            if any("requirements" in f for f in req_files):
                pip_result = self._update_pip_packages()
                updates.extend(pip_result)
            
            # Update pyproject.toml if present
            if any("pyproject.toml" in f for f in req_files):
                poetry_result = self._update_poetry_packages()
                updates.extend(poetry_result)
            
            result.status = "success" if updates else "skipped"
            result.message = f"Updated {len(updates)} packages" if updates else "No updates available"
            result.details = {"updated_packages": updates}
            
        except Exception as e:
            result.status = "failure"
            result.message = f"Dependency update failed: {str(e)}"
            result.errors.append(str(e))
            self.logger.error(f"Dependency update failed: {e}")
        
        finally:
            result.duration_seconds = time.time() - start_time
        
        return result
    
    def _update_pip_packages(self) -> List[Dict[str, str]]:
        """Update pip packages and return list of updates."""
        updates = []
        try:
            # Get outdated packages
            result = subprocess.run(
                ["pip", "list", "--outdated", "--format=json"],
                capture_output=True,
                text=True,
                check=True
            )
            
            outdated = json.loads(result.stdout)
            
            for package in outdated:
                package_name = package["name"]
                current_version = package["version"]
                latest_version = package["latest_version"]
                
                # Update package
                try:
                    subprocess.run(
                        ["pip", "install", "--upgrade", package_name],
                        capture_output=True,
                        check=True
                    )
                    updates.append({
                        "name": package_name,
                        "from_version": current_version,
                        "to_version": latest_version,
                        "method": "pip"
                    })
                except subprocess.CalledProcessError as e:
                    self.logger.warning(f"Failed to update {package_name}: {e}")
        
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            self.logger.error(f"Failed to check pip packages: {e}")
        
        return updates
    
    def _update_poetry_packages(self) -> List[Dict[str, str]]:
        """Update poetry packages and return list of updates."""
        updates = []
        try:
            # Check if poetry is available
            subprocess.run(["poetry", "--version"], capture_output=True, check=True)
            
            # Update packages
            result = subprocess.run(
                ["poetry", "update", "--dry-run"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            # Parse output for updates (simplified)
            if "Updating" in result.stdout:
                # Actually perform update
                subprocess.run(
                    ["poetry", "update"],
                    capture_output=True,
                    check=True,
                    cwd=self.project_root
                )
                updates.append({
                    "name": "poetry_packages",
                    "from_version": "various",
                    "to_version": "latest",
                    "method": "poetry"
                })
        
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.logger.debug("Poetry not available or failed")
        
        return updates


class SecurityPatchExecutor(MaintenanceExecutor):
    """Executes security patch tasks."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.logger = logging.getLogger(__name__)
    
    def can_execute(self, task: MaintenanceTask) -> bool:
        """Check if this is a security patch task."""
        return task.task_type == "security_patch"
    
    def execute(self, task: MaintenanceTask) -> MaintenanceResult:
        """Execute security patch task."""
        start_time = time.time()
        result = MaintenanceResult(
            task_id=task.task_id,
            status="failure",
            message="",
            duration_seconds=0
        )
        
        try:
            vulnerabilities = []
            
            # Check for security vulnerabilities using pip-audit
            pip_vulns = self._check_pip_vulnerabilities()
            vulnerabilities.extend(pip_vulns)
            
            # Check for security advisories
            advisories = self._check_security_advisories()
            vulnerabilities.extend(advisories)
            
            if not vulnerabilities:
                result.status = "success"
                result.message = "No security vulnerabilities found"
                result.details = {"vulnerabilities_checked": True}
            else:
                # Attempt to fix vulnerabilities
                fixed = self._fix_vulnerabilities(vulnerabilities)
                result.status = "partial" if fixed else "failure"
                result.message = f"Fixed {len(fixed)}/{len(vulnerabilities)} vulnerabilities"
                result.details = {
                    "total_vulnerabilities": len(vulnerabilities),
                    "fixed_vulnerabilities": len(fixed),
                    "remaining_vulnerabilities": len(vulnerabilities) - len(fixed),
                    "vulnerabilities": vulnerabilities,
                    "fixed": fixed
                }
        
        except Exception as e:
            result.status = "failure"
            result.message = f"Security patch failed: {str(e)}"
            result.errors.append(str(e))
            self.logger.error(f"Security patch failed: {e}")
        
        finally:
            result.duration_seconds = time.time() - start_time
        
        return result
    
    def _check_pip_vulnerabilities(self) -> List[Dict[str, Any]]:
        """Check for pip package vulnerabilities."""
        vulnerabilities = []
        try:
            # Use pip-audit if available
            result = subprocess.run(
                ["pip-audit", "--format=json"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                for vuln in data.get("vulnerabilities", []):
                    vulnerabilities.append({
                        "type": "pip_vulnerability",
                        "package": vuln.get("package"),
                        "version": vuln.get("version"),
                        "vulnerability_id": vuln.get("id"),
                        "description": vuln.get("description", ""),
                        "severity": vuln.get("severity", "unknown")
                    })
        
        except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
            self.logger.debug("pip-audit not available or failed")
        
        return vulnerabilities
    
    def _check_security_advisories(self) -> List[Dict[str, Any]]:
        """Check for security advisories."""
        advisories = []
        try:
            # Use safety if available
            result = subprocess.run(
                ["safety", "check", "--json"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                for advisory in data:
                    advisories.append({
                        "type": "security_advisory",
                        "package": advisory.get("package"),
                        "version": advisory.get("installed_version"),
                        "vulnerability_id": advisory.get("id"),
                        "description": advisory.get("advisory", ""),
                        "severity": "high"  # Default for safety advisories
                    })
        
        except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
            self.logger.debug("safety not available or failed")
        
        return advisories
    
    def _fix_vulnerabilities(self, vulnerabilities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Attempt to fix vulnerabilities by updating packages."""
        fixed = []
        
        for vuln in vulnerabilities:
            try:
                package = vuln.get("package")
                if package:
                    # Try to update the vulnerable package
                    subprocess.run(
                        ["pip", "install", "--upgrade", package],
                        capture_output=True,
                        check=True
                    )
                    fixed.append(vuln)
            
            except subprocess.CalledProcessError:
                self.logger.warning(f"Failed to fix vulnerability in {package}")
        
        return fixed


class CleanupExecutor(MaintenanceExecutor):
    """Executes cleanup tasks."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.logger = logging.getLogger(__name__)
    
    def can_execute(self, task: MaintenanceTask) -> bool:
        """Check if this is a cleanup task."""
        return task.task_type == "cleanup"
    
    def execute(self, task: MaintenanceTask) -> MaintenanceResult:
        """Execute cleanup task."""
        start_time = time.time()
        result = MaintenanceResult(
            task_id=task.task_id,
            status="success",
            message="",
            duration_seconds=0
        )
        
        try:
            cleanup_stats = {
                "files_removed": 0,
                "directories_removed": 0,
                "bytes_freed": 0,
                "actions": []
            }
            
            # Clean up common temporary/cache directories
            cleanup_paths = [
                "__pycache__",
                "*.pyc",
                ".pytest_cache",
                ".coverage",
                "htmlcov",
                ".mypy_cache",
                "*.egg-info",
                "build",
                "dist",
                ".tox"
            ]
            
            for pattern in cleanup_paths:
                removed = self._cleanup_pattern(pattern)
                cleanup_stats["files_removed"] += removed["files"]
                cleanup_stats["directories_removed"] += removed["directories"]
                cleanup_stats["bytes_freed"] += removed["bytes"]
                if removed["files"] > 0 or removed["directories"] > 0:
                    cleanup_stats["actions"].append(f"Cleaned {pattern}: {removed}")
            
            # Clean up old log files
            log_cleanup = self._cleanup_old_logs()
            cleanup_stats.update(log_cleanup)
            
            # Clean up temporary data files
            temp_cleanup = self._cleanup_temp_data()
            cleanup_stats.update(temp_cleanup)
            
            total_items = cleanup_stats["files_removed"] + cleanup_stats["directories_removed"]
            result.message = f"Cleaned up {total_items} items, freed {cleanup_stats['bytes_freed']} bytes"
            result.details = cleanup_stats
        
        except Exception as e:
            result.status = "failure"
            result.message = f"Cleanup failed: {str(e)}"
            result.errors.append(str(e))
            self.logger.error(f"Cleanup failed: {e}")
        
        finally:
            result.duration_seconds = time.time() - start_time
        
        return result
    
    def _cleanup_pattern(self, pattern: str) -> Dict[str, int]:
        """Clean up files/directories matching pattern."""
        stats = {"files": 0, "directories": 0, "bytes": 0}
        
        try:
            import glob
            import shutil
            
            matches = glob.glob(str(self.project_root / "**" / pattern), recursive=True)
            
            for match in matches:
                match_path = Path(match)
                if match_path.exists():
                    try:
                        if match_path.is_file():
                            stats["bytes"] += match_path.stat().st_size
                            match_path.unlink()
                            stats["files"] += 1
                        elif match_path.is_dir():
                            # Calculate directory size
                            for file_path in match_path.rglob("*"):
                                if file_path.is_file():
                                    stats["bytes"] += file_path.stat().st_size
                            shutil.rmtree(match_path)
                            stats["directories"] += 1
                    except (OSError, PermissionError) as e:
                        self.logger.warning(f"Failed to remove {match}: {e}")
        
        except Exception as e:
            self.logger.error(f"Failed to cleanup pattern {pattern}: {e}")
        
        return stats
    
    def _cleanup_old_logs(self, days_old: int = 30) -> Dict[str, Any]:
        """Clean up log files older than specified days."""
        stats = {"old_logs_removed": 0, "log_bytes_freed": 0}
        
        try:
            cutoff_time = datetime.now() - timedelta(days=days_old)
            
            # Look for log files
            log_patterns = ["*.log", "*.log.*", "logs/**/*.log"]
            
            for pattern in log_patterns:
                log_files = self.project_root.glob(pattern)
                for log_file in log_files:
                    if log_file.is_file():
                        file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                        if file_time < cutoff_time:
                            stats["log_bytes_freed"] += log_file.stat().st_size
                            log_file.unlink()
                            stats["old_logs_removed"] += 1
        
        except Exception as e:
            self.logger.error(f"Failed to cleanup old logs: {e}")
        
        return stats
    
    def _cleanup_temp_data(self) -> Dict[str, Any]:
        """Clean up temporary data files."""
        stats = {"temp_files_removed": 0, "temp_bytes_freed": 0}
        
        try:
            # Clean up common temp directories
            temp_dirs = [
                "data/processed/temp",
                "data/cache",
                "tmp",
                ".tmp"
            ]
            
            for temp_dir in temp_dirs:
                temp_path = self.project_root / temp_dir
                if temp_path.exists() and temp_path.is_dir():
                    for temp_file in temp_path.rglob("*"):
                        if temp_file.is_file():
                            stats["temp_bytes_freed"] += temp_file.stat().st_size
                            temp_file.unlink()
                            stats["temp_files_removed"] += 1
        
        except Exception as e:
            self.logger.error(f"Failed to cleanup temp data: {e}")
        
        return stats


class HealthBasedScalingExecutor(MaintenanceExecutor):
    """Executes health-based scaling tasks."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def can_execute(self, task: MaintenanceTask) -> bool:
        """Check if this is a scaling task."""
        return task.task_type == "scaling"
    
    def execute(self, task: MaintenanceTask) -> MaintenanceResult:
        """Execute health-based scaling task."""
        start_time = time.time()
        result = MaintenanceResult(
            task_id=task.task_id,
            status="success",
            message="",
            duration_seconds=0
        )
        
        try:
            # Get current system health
            health = self._get_system_health()
            
            # Determine scaling actions based on health
            scaling_actions = self._determine_scaling_actions(health, task.metadata)
            
            # Execute scaling actions
            executed_actions = []
            for action in scaling_actions:
                if self._execute_scaling_action(action):
                    executed_actions.append(action)
            
            result.message = f"Executed {len(executed_actions)} scaling actions"
            result.details = {
                "system_health": health.__dict__,
                "recommended_actions": scaling_actions,
                "executed_actions": executed_actions
            }
        
        except Exception as e:
            result.status = "failure"
            result.message = f"Scaling failed: {str(e)}"
            result.errors.append(str(e))
            self.logger.error(f"Health-based scaling failed: {e}")
        
        finally:
            result.duration_seconds = time.time() - start_time
        
        return result
    
    def _get_system_health(self) -> SystemHealth:
        """Get current system health metrics."""
        try:
            import psutil
            
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Get network connections count
            connections = len(psutil.net_connections())
            
            health = SystemHealth(
                cpu_usage_percent=cpu_percent,
                memory_usage_percent=memory.percent,
                disk_usage_percent=disk.percent,
                active_connections=connections,
                response_time_ms=0.0,  # Would be measured from health checks
                error_rate_percent=0.0  # Would be calculated from metrics
            )
            
            # Determine if system is healthy
            issues = []
            if cpu_percent > 80:
                issues.append("High CPU usage")
            if memory.percent > 85:
                issues.append("High memory usage")
            if disk.percent > 90:
                issues.append("High disk usage")
            
            health.healthy = len(issues) == 0
            health.issues = issues
            
            return health
        
        except ImportError:
            self.logger.warning("psutil not available, using mock health data")
            return SystemHealth(
                cpu_usage_percent=50.0,
                memory_usage_percent=60.0,
                disk_usage_percent=70.0,
                active_connections=100,
                response_time_ms=200.0,
                error_rate_percent=1.0
            )
    
    def _determine_scaling_actions(self, health: SystemHealth, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Determine what scaling actions should be taken."""
        actions = []
        
        # CPU-based scaling
        cpu_threshold = config.get("cpu_scale_threshold", 75.0)
        if health.cpu_usage_percent > cpu_threshold:
            actions.append({
                "type": "scale_up",
                "reason": f"CPU usage {health.cpu_usage_percent}% > {cpu_threshold}%",
                "resource": "cpu",
                "recommended_action": "Add more CPU cores or scale horizontally"
            })
        
        # Memory-based scaling
        memory_threshold = config.get("memory_scale_threshold", 80.0)
        if health.memory_usage_percent > memory_threshold:
            actions.append({
                "type": "scale_up",
                "reason": f"Memory usage {health.memory_usage_percent}% > {memory_threshold}%",
                "resource": "memory",
                "recommended_action": "Add more memory or optimize memory usage"
            })
        
        # Connection-based scaling
        connection_threshold = config.get("connection_scale_threshold", 1000)
        if health.active_connections > connection_threshold:
            actions.append({
                "type": "scale_out",
                "reason": f"Active connections {health.active_connections} > {connection_threshold}",
                "resource": "connections",
                "recommended_action": "Scale horizontally to handle more connections"
            })
        
        return actions
    
    def _execute_scaling_action(self, action: Dict[str, Any]) -> bool:
        """Execute a scaling action (placeholder for actual implementation)."""
        try:
            # In a real implementation, this would:
            # - Send scaling requests to orchestration system (Kubernetes, Docker Swarm, etc.)
            # - Adjust resource limits
            # - Trigger auto-scaling policies
            # - Notify monitoring systems
            
            self.logger.info(f"Scaling action executed: {action}")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to execute scaling action {action}: {e}")
            return False


class AutomatedMaintenanceManager:
    """Manages automated maintenance tasks and scheduling."""
    
    def __init__(self, config_path: Optional[str] = None, storage_dir: str = "maintenance"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.tasks: Dict[str, MaintenanceTask] = {}
        self.executors: List[MaintenanceExecutor] = []
        self.results: List[MaintenanceResult] = []
        
        # Threading for scheduled tasks
        self._scheduler_thread: Optional[Thread] = None
        self._stop_event = Event()
        self._task_lock = Lock()
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize default executors
        self._initialize_executors()
        
        # Load existing tasks
        self._load_tasks()
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load maintenance configuration."""
        default_config = {
            "enabled": True,
            "check_interval_seconds": 300,  # 5 minutes
            "max_concurrent_tasks": 3,
            "retention_days": 30,
            "default_executors": ["dependency", "security", "cleanup", "scaling"],
            "task_defaults": {
                "dependency_update": {
                    "schedule": "weekly",
                    "priority": "medium",
                    "enabled": True
                },
                "security_patch": {
                    "schedule": "daily",
                    "priority": "high",
                    "enabled": True
                },
                "cleanup": {
                    "schedule": "daily",
                    "priority": "low",
                    "enabled": True
                },
                "scaling": {
                    "schedule": "*/15 * * * *",  # Every 15 minutes
                    "priority": "medium",
                    "enabled": True,
                    "cpu_scale_threshold": 75.0,
                    "memory_scale_threshold": 80.0,
                    "connection_scale_threshold": 1000
                }
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    if yaml:
                        config = yaml.safe_load(f)
                    else:
                        config = json.load(f)  # Fallback to JSON if yaml not available
                # Merge with defaults
                default_config.update(config)
            except Exception as e:
                self.logger.error(f"Failed to load config from {config_path}: {e}")
        
        return default_config
    
    def _initialize_executors(self):
        """Initialize maintenance executors."""
        executor_map = {
            "dependency": DependencyUpdateExecutor,
            "security": SecurityPatchExecutor,
            "cleanup": CleanupExecutor,
            "scaling": HealthBasedScalingExecutor
        }
        
        enabled_executors = self.config.get("default_executors", [])
        
        for executor_name in enabled_executors:
            if executor_name in executor_map:
                try:
                    if executor_name in ["dependency", "security", "cleanup"]:
                        executor = executor_map[executor_name](".")
                    else:
                        executor = executor_map[executor_name]()
                    self.executors.append(executor)
                    self.logger.info(f"Initialized {executor_name} executor")
                except Exception as e:
                    self.logger.error(f"Failed to initialize {executor_name} executor: {e}")
    
    def _load_tasks(self):
        """Load existing maintenance tasks from storage."""
        tasks_file = self.storage_dir / "tasks.json"
        if tasks_file.exists():
            try:
                with open(tasks_file, 'r') as f:
                    tasks_data = json.load(f)
                
                for task_data in tasks_data:
                    # Convert datetime strings back to datetime objects
                    for key in ['created_at', 'last_run', 'next_run']:
                        if task_data.get(key):
                            task_data[key] = datetime.fromisoformat(task_data[key])
                    
                    task = MaintenanceTask(**task_data)
                    self.tasks[task.task_id] = task
                
                self.logger.info(f"Loaded {len(self.tasks)} maintenance tasks")
            
            except Exception as e:
                self.logger.error(f"Failed to load tasks: {e}")
    
    def _save_tasks(self):
        """Save maintenance tasks to storage."""
        tasks_file = self.storage_dir / "tasks.json"
        try:
            tasks_data = [task.to_dict() for task in self.tasks.values()]
            
            with open(tasks_file, 'w') as f:
                json.dump(tasks_data, f, indent=2)
        
        except Exception as e:
            self.logger.error(f"Failed to save tasks: {e}")
    
    def add_task(self, task: MaintenanceTask) -> bool:
        """Add a maintenance task."""
        try:
            with self._task_lock:
                # Set next run time based on schedule
                self._calculate_next_run(task)
                
                self.tasks[task.task_id] = task
                self._save_tasks()
                
                self.logger.info(f"Added maintenance task: {task.task_id}")
                return True
        
        except Exception as e:
            self.logger.error(f"Failed to add task {task.task_id}: {e}")
            return False
    
    def remove_task(self, task_id: str) -> bool:
        """Remove a maintenance task."""
        try:
            with self._task_lock:
                if task_id in self.tasks:
                    del self.tasks[task_id]
                    self._save_tasks()
                    self.logger.info(f"Removed maintenance task: {task_id}")
                    return True
                else:
                    self.logger.warning(f"Task not found: {task_id}")
                    return False
        
        except Exception as e:
            self.logger.error(f"Failed to remove task {task_id}: {e}")
            return False
    
    def execute_task(self, task_id: str) -> Optional[MaintenanceResult]:
        """Execute a specific maintenance task."""
        with self._task_lock:
            task = self.tasks.get(task_id)
            if not task:
                self.logger.error(f"Task not found: {task_id}")
                return None
            
            if not task.enabled:
                self.logger.info(f"Task disabled: {task_id}")
                return None
        
        # Find executor for task
        executor = None
        for exec_instance in self.executors:
            if exec_instance.can_execute(task):
                executor = exec_instance
                break
        
        if not executor:
            result = MaintenanceResult(
                task_id=task_id,
                status="failure",
                message="No executor found for task",
                duration_seconds=0
            )
        else:
            self.logger.info(f"Executing task: {task_id}")
            result = executor.execute(task)
        
        # Update task with result
        with self._task_lock:
            task.run_count += 1
            task.last_run = datetime.now(timezone.utc)
            task.last_result = result.to_dict()
            
            if result.status == "success":
                task.success_count += 1
            else:
                task.failure_count += 1
            
            # Calculate next run
            self._calculate_next_run(task)
            
            self._save_tasks()
        
        # Store result
        self.results.append(result)
        self._save_result(result)
        
        return result
    
    def _calculate_next_run(self, task: MaintenanceTask):
        """Calculate next run time for a task."""
        if task.schedule == "manual":
            task.next_run = None
            return
        
        now = datetime.now(timezone.utc)
        
        if task.schedule == "daily":
            task.next_run = now + timedelta(days=1)
        elif task.schedule == "weekly":
            task.next_run = now + timedelta(weeks=1)
        elif task.schedule == "monthly":
            task.next_run = now + timedelta(days=30)
        else:
            # For cron expressions or other schedules, set a default
            task.next_run = now + timedelta(hours=1)
    
    def _save_result(self, result: MaintenanceResult):
        """Save maintenance result to storage."""
        try:
            results_file = self.storage_dir / f"results_{datetime.now().strftime('%Y%m%d')}.json"
            
            # Load existing results for the day
            results_data = []
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results_data = json.load(f)
            
            # Add new result
            results_data.append(result.to_dict())
            
            # Save updated results
            with open(results_file, 'w') as f:
                json.dump(results_data, f, indent=2)
        
        except Exception as e:
            self.logger.error(f"Failed to save result: {e}")
    
    def start_scheduler(self):
        """Start the maintenance task scheduler."""
        if not self.config.get("enabled", True):
            self.logger.info("Automated maintenance is disabled")
            return
        
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            self.logger.warning("Scheduler already running")
            return
        
        self._stop_event.clear()
        self._scheduler_thread = Thread(target=self._scheduler_loop, daemon=True)
        self._scheduler_thread.start()
        
        self.logger.info("Started maintenance scheduler")
    
    def stop_scheduler(self):
        """Stop the maintenance task scheduler."""
        if self._scheduler_thread:
            self._stop_event.set()
            self._scheduler_thread.join(timeout=5)
            self.logger.info("Stopped maintenance scheduler")
    
    def _scheduler_loop(self):
        """Main scheduler loop."""
        check_interval = self.config.get("check_interval_seconds", 300)
        
        while not self._stop_event.is_set():
            try:
                self._check_scheduled_tasks()
            except Exception as e:
                self.logger.error(f"Error in scheduler loop: {e}")
            
            self._stop_event.wait(check_interval)
    
    def _check_scheduled_tasks(self):
        """Check for tasks that need to be executed."""
        now = datetime.now(timezone.utc)
        
        with self._task_lock:
            tasks_to_execute = []
            
            for task in self.tasks.values():
                if (task.enabled and 
                    task.next_run and 
                    task.next_run <= now):
                    tasks_to_execute.append(task.task_id)
        
        # Execute tasks (outside of lock to avoid blocking)
        for task_id in tasks_to_execute:
            try:
                self.execute_task(task_id)
            except Exception as e:
                self.logger.error(f"Failed to execute scheduled task {task_id}: {e}")
    
    def get_task_status(self, task_id: Optional[str] = None) -> Dict[str, Any]:
        """Get status of maintenance tasks."""
        with self._task_lock:
            if task_id:
                task = self.tasks.get(task_id)
                if task:
                    return {
                        "task": task.to_dict(),
                        "last_result": task.last_result
                    }
                else:
                    return {"error": "Task not found"}
            else:
                return {
                    "total_tasks": len(self.tasks),
                    "enabled_tasks": sum(1 for t in self.tasks.values() if t.enabled),
                    "tasks": [task.to_dict() for task in self.tasks.values()],
                    "scheduler_running": self._scheduler_thread and self._scheduler_thread.is_alive()
                }
    
    def get_maintenance_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get maintenance activity summary."""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        # Filter recent results
        recent_results = [
            r for r in self.results 
            if r.timestamp >= cutoff_date
        ]
        
        summary = {
            "period_days": days,
            "total_executions": len(recent_results),
            "successful_executions": sum(1 for r in recent_results if r.status == "success"),
            "failed_executions": sum(1 for r in recent_results if r.status == "failure"),
            "by_task_type": {},
            "average_duration": 0,
            "recent_results": [r.to_dict() for r in recent_results[-10:]]  # Last 10 results
        }
        
        # Group by task type
        task_types = {}
        total_duration = 0.0  # Use float type
        
        for result in recent_results:
            task = self.tasks.get(result.task_id)
            task_type = task.task_type if task else "unknown"
            
            if task_type not in task_types:
                task_types[task_type] = {"count": 0, "success": 0, "failure": 0}
            
            task_types[task_type]["count"] += 1
            if result.status == "success":
                task_types[task_type]["success"] += 1
            else:
                task_types[task_type]["failure"] += 1
            
            total_duration += result.duration_seconds
        
        summary["by_task_type"] = task_types
        if recent_results:
            summary["average_duration"] = total_duration / len(recent_results)
        
        return summary


# Global maintenance manager instance
_maintenance_manager: Optional[AutomatedMaintenanceManager] = None


def get_maintenance_manager() -> AutomatedMaintenanceManager:
    """Get global maintenance manager instance."""
    global _maintenance_manager
    if _maintenance_manager is None:
        _maintenance_manager = AutomatedMaintenanceManager()
    return _maintenance_manager


def initialize_maintenance(config_path: Optional[str] = None, storage_dir: str = "maintenance") -> AutomatedMaintenanceManager:
    """Initialize maintenance system."""
    global _maintenance_manager
    if _maintenance_manager is not None:
        _maintenance_manager.stop_scheduler()
    
    _maintenance_manager = AutomatedMaintenanceManager(config_path, storage_dir)
    return _maintenance_manager


def shutdown_maintenance():
    """Shutdown maintenance system."""
    global _maintenance_manager
    if _maintenance_manager is not None:
        _maintenance_manager.stop_scheduler()
        _maintenance_manager = None
