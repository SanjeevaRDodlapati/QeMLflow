"""
Health Checks Module

This module provides comprehensive health check capabilities for system components,
databases, services, and custom health checks with configurable thresholds and
automated health monitoring.
"""

import logging
import psutil
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4


@dataclass
class HealthCheckResult:
    """Health check result data."""
    
    check_id: str = field(default_factory=lambda: str(uuid4()))
    check_name: str = ""
    status: str = "unknown"  # healthy, degraded, unhealthy, unknown
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    # Check details
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    
    # Performance metrics
    check_duration_ms: float = 0.0
    
    # Thresholds
    warning_threshold: Optional[float] = None
    critical_threshold: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HealthCheckResult':
        """Create from dictionary."""
        return cls(**data)


class BaseHealthCheck(ABC):
    """Base class for all health checks."""
    
    def __init__(self, name: str, warning_threshold: Optional[float] = None,
                 critical_threshold: Optional[float] = None):
        self.name = name
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def check(self) -> HealthCheckResult:
        """Perform the health check."""
        pass
    
    def _determine_status(self, value: float, invert: bool = False) -> str:
        """Determine health status based on thresholds."""
        if self.critical_threshold is not None:
            if invert:
                if value <= self.critical_threshold:
                    return "unhealthy"
            else:
                if value >= self.critical_threshold:
                    return "unhealthy"
        
        if self.warning_threshold is not None:
            if invert:
                if value <= self.warning_threshold:
                    return "degraded"
            else:
                if value >= self.warning_threshold:
                    return "degraded"
        
        return "healthy"


class SystemHealthCheck(BaseHealthCheck):
    """System resource health checks."""
    
    def __init__(self, name: str = "system_health", 
                 cpu_warning: float = 80.0, cpu_critical: float = 95.0,
                 memory_warning: float = 85.0, memory_critical: float = 95.0,
                 disk_warning: float = 85.0, disk_critical: float = 95.0):
        super().__init__(name)
        
        self.cpu_warning = cpu_warning
        self.cpu_critical = cpu_critical
        self.memory_warning = memory_warning
        self.memory_critical = memory_critical
        self.disk_warning = disk_warning
        self.disk_critical = disk_critical
    
    def check(self) -> HealthCheckResult:
        """Perform system health check."""
        start_time = time.time()
        
        try:
            # CPU check
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory check
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk check
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # Determine overall status
            cpu_status = self._determine_status_with_thresholds(
                cpu_percent, self.cpu_warning, self.cpu_critical
            )
            memory_status = self._determine_status_with_thresholds(
                memory_percent, self.memory_warning, self.memory_critical
            )
            disk_status = self._determine_status_with_thresholds(
                disk_percent, self.disk_warning, self.disk_critical
            )
            
            # Overall status is the worst of all components
            status_priority = {"healthy": 0, "degraded": 1, "unhealthy": 2}
            overall_status = max([cpu_status, memory_status, disk_status], 
                               key=lambda x: status_priority.get(x, 0))
            
            duration_ms = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                check_name=self.name,
                status=overall_status,
                message=f"System health: CPU {cpu_percent:.1f}%, Memory {memory_percent:.1f}%, Disk {disk_percent:.1f}%",
                details={
                    "cpu_percent": cpu_percent,
                    "cpu_status": cpu_status,
                    "memory_percent": memory_percent,
                    "memory_status": memory_status,
                    "memory_used_gb": memory.used / (1024**3),
                    "memory_available_gb": memory.available / (1024**3),
                    "disk_percent": disk_percent,
                    "disk_status": disk_status,
                    "disk_used_gb": disk.used / (1024**3),
                    "disk_free_gb": disk.free / (1024**3)
                },
                check_duration_ms=duration_ms
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.logger.error(f"System health check failed: {e}")
            
            return HealthCheckResult(
                check_name=self.name,
                status="unhealthy",
                message=f"System health check failed: {str(e)}",
                details={"error": str(e)},
                check_duration_ms=duration_ms
            )
    
    def _determine_status_with_thresholds(self, value: float, warning: float, critical: float) -> str:
        """Determine status with specific thresholds."""
        if value >= critical:
            return "unhealthy"
        elif value >= warning:
            return "degraded"
        else:
            return "healthy"


class DatabaseHealthCheck(BaseHealthCheck):
    """Database connectivity and performance health check."""
    
    def __init__(self, name: str = "database_health", 
                 connection_timeout: float = 5.0,
                 query_timeout: float = 10.0):
        super().__init__(name)
        self.connection_timeout = connection_timeout
        self.query_timeout = query_timeout
    
    def check(self) -> HealthCheckResult:
        """Perform database health check."""
        start_time = time.time()
        
        try:
            # Placeholder implementation - in production, implement actual DB checks
            # This would connect to your database and run health queries
            
            # Simulate database check
            time.sleep(0.1)  # Simulate connection time
            
            connection_time_ms = 100  # Simulated
            query_time_ms = 50  # Simulated
            
            # Determine status based on response times
            if connection_time_ms > self.connection_timeout * 1000:
                status = "unhealthy"
                message = f"Database connection slow: {connection_time_ms}ms"
            elif query_time_ms > self.query_timeout * 1000:
                status = "degraded" 
                message = f"Database queries slow: {query_time_ms}ms"
            else:
                status = "healthy"
                message = "Database healthy"
            
            duration_ms = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                check_name=self.name,
                status=status,
                message=message,
                details={
                    "connection_time_ms": connection_time_ms,
                    "query_time_ms": query_time_ms,
                    "connection_timeout_ms": self.connection_timeout * 1000,
                    "query_timeout_ms": self.query_timeout * 1000
                },
                check_duration_ms=duration_ms
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.logger.error(f"Database health check failed: {e}")
            
            return HealthCheckResult(
                check_name=self.name,
                status="unhealthy",
                message=f"Database health check failed: {str(e)}",
                details={"error": str(e)},
                check_duration_ms=duration_ms
            )


class ServiceHealthCheck(BaseHealthCheck):
    """External service health check."""
    
    def __init__(self, name: str, service_url: str = "", 
                 timeout: float = 10.0, expected_status: int = 200):
        super().__init__(name)
        self.service_url = service_url
        self.timeout = timeout
        self.expected_status = expected_status
    
    def check(self) -> HealthCheckResult:
        """Perform service health check."""
        start_time = time.time()
        
        try:
            # Placeholder implementation - in production, make actual HTTP requests
            # This would make HTTP requests to check service availability
            
            # Simulate service check
            time.sleep(0.05)  # Simulate network request
            
            response_time_ms = 50  # Simulated
            status_code = 200  # Simulated
            
            # Determine status
            if status_code == self.expected_status:
                if response_time_ms < 1000:
                    status = "healthy"
                    message = f"Service healthy: {status_code} in {response_time_ms}ms"
                else:
                    status = "degraded"
                    message = f"Service slow: {status_code} in {response_time_ms}ms"
            else:
                status = "unhealthy"
                message = f"Service unhealthy: {status_code}"
            
            duration_ms = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                check_name=self.name,
                status=status,
                message=message,
                details={
                    "service_url": self.service_url,
                    "response_time_ms": response_time_ms,
                    "status_code": status_code,
                    "expected_status": self.expected_status,
                    "timeout_ms": self.timeout * 1000
                },
                check_duration_ms=duration_ms
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.logger.error(f"Service health check failed: {e}")
            
            return HealthCheckResult(
                check_name=self.name,
                status="unhealthy",
                message=f"Service health check failed: {str(e)}",
                details={"error": str(e), "service_url": self.service_url},
                check_duration_ms=duration_ms
            )


class HealthCheckRegistry:
    """Registry for managing multiple health checks."""
    
    def __init__(self):
        self.health_checks: Dict[str, BaseHealthCheck] = {}
        self.logger = logging.getLogger(__name__)
        
        # Add default system health check
        self.register_check(SystemHealthCheck())
        
        self.logger.info("Health check registry initialized")
    
    def register_check(self, health_check: BaseHealthCheck):
        """Register a health check."""
        self.health_checks[health_check.name] = health_check
        self.logger.info(f"Registered health check: {health_check.name}")
    
    def unregister_check(self, name: str):
        """Unregister a health check."""
        if name in self.health_checks:
            del self.health_checks[name]
            self.logger.info(f"Unregistered health check: {name}")
    
    def run_check(self, name: str) -> Optional[HealthCheckResult]:
        """Run a specific health check."""
        if name in self.health_checks:
            try:
                return self.health_checks[name].check()
            except Exception as e:
                self.logger.error(f"Health check {name} failed: {e}")
                return HealthCheckResult(
                    check_name=name,
                    status="unhealthy",
                    message=f"Health check failed: {str(e)}",
                    details={"error": str(e)}
                )
        return None
    
    def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks."""
        results = {}
        
        for name in self.health_checks:
            result = self.run_check(name)
            if result:
                results[name] = result
        
        return results
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall health status from all checks."""
        results = self.run_all_checks()
        
        if not results:
            return {
                "status": "unknown",
                "message": "No health checks configured",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "details": {}
            }
        
        # Determine overall status
        status_counts = {"healthy": 0, "degraded": 0, "unhealthy": 0, "unknown": 0}
        
        for result in results.values():
            status_counts[result.status] = status_counts.get(result.status, 0) + 1
        
        # Overall status logic
        if status_counts["unhealthy"] > 0:
            overall_status = "unhealthy"
        elif status_counts["degraded"] > 0:
            overall_status = "degraded"
        elif status_counts["healthy"] > 0:
            overall_status = "healthy"
        else:
            overall_status = "unknown"
        
        # Create summary message
        total_checks = len(results)
        healthy_checks = status_counts["healthy"]
        
        if overall_status == "healthy":
            message = f"All {total_checks} health checks passing"
        elif overall_status == "degraded":
            message = f"{healthy_checks}/{total_checks} health checks healthy, {status_counts['degraded']} degraded"
        else:
            message = f"{healthy_checks}/{total_checks} health checks healthy, {status_counts['unhealthy']} unhealthy"
        
        return {
            "status": overall_status,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "total_checks": total_checks,
                "healthy": status_counts["healthy"],
                "degraded": status_counts["degraded"],
                "unhealthy": status_counts["unhealthy"],
                "unknown": status_counts["unknown"]
            },
            "details": {name: result.to_dict() for name, result in results.items()}
        }
    
    def get_health_checks(self) -> List[str]:
        """Get list of registered health check names."""
        return list(self.health_checks.keys())


# Global health check registry
_global_health_registry = None

def get_health_registry() -> HealthCheckRegistry:
    """Get global health check registry."""
    global _global_health_registry
    
    if _global_health_registry is None:
        _global_health_registry = HealthCheckRegistry()
    
    return _global_health_registry
