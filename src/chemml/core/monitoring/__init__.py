"""
ChemML Monitoring Module
=======================

Performance monitoring, analytics, and optimization tools for ChemML.

Key Components:
- PerformanceDashboard: Real-time performance visualization
- System health monitoring
- Optimization suggestions
- Resource usage analytics
"""

from .dashboard import (
    PerformanceDashboard,
    create_performance_dashboard,
    show_performance_dashboard,
)

__all__ = [
    "PerformanceDashboard",
    "create_performance_dashboard",
    "show_performance_dashboard",
]
