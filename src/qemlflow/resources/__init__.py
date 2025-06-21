"""
QeMLflow Resource Management Package

This package provides comprehensive resource management capabilities including:
- Intelligent memory management
- Compute resource optimization
- Auto-scaling mechanisms
- Resource monitoring and alerting
"""

from .memory import MemoryManager, MemoryProfiler, MemoryOptimizer
from .compute import ComputeManager, CPUOptimizer, GPUManager
from .scaling import AutoScaler, ScalingPolicy, LoadBalancer
from .monitoring import ResourceMonitor, ResourceDashboard, AlertManager

__all__ = [
    'MemoryManager',
    'MemoryProfiler', 
    'MemoryOptimizer',
    'ComputeManager',
    'CPUOptimizer',
    'GPUManager',
    'AutoScaler',
    'ScalingPolicy',
    'LoadBalancer',
    'ResourceMonitor',
    'ResourceDashboard',
    'AlertManager'
]

__version__ = '1.0.0'
