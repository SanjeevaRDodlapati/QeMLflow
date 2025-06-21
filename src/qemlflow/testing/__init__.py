"""
Testing utilities for QeMLflow.

This package provides utilities for testing, including matplotlib configuration
and other testing helpers.
"""

from .matplotlib_config import (
    FigureManager,
    managed_figures,
    setup_testing_backend,
    create_test_figure,
    safe_show,
    is_testing,
    patch_matplotlib_show,
    MATPLOTLIB_AVAILABLE
)

__all__ = [
    'FigureManager',
    'managed_figures', 
    'setup_testing_backend',
    'create_test_figure',
    'safe_show',
    'is_testing',
    'patch_matplotlib_show',
    'MATPLOTLIB_AVAILABLE'
]
