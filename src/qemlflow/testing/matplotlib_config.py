"""
Matplotlib configuration for testing to prevent manual figure closing.

This module configures matplotlib to use non-interactive backends during testing
and provides utilities for automatic figure management.
"""

import os
import sys
from typing import Optional
from contextlib import contextmanager

# Set matplotlib backend to non-interactive before any imports
os.environ['MPLBACKEND'] = 'Agg'

try:
    import matplotlib
    matplotlib.use('Agg', force=True)  # Use non-interactive backend
    import matplotlib.pyplot as plt
    
    # Configure matplotlib for testing
    plt.ioff()  # Turn off interactive mode
    
    # Set default figure parameters for testing
    plt.rcParams.update({
        'figure.max_open_warning': 0,  # Disable warnings for too many figures
        'savefig.bbox': 'tight',
        'savefig.dpi': 100,
        'figure.figsize': (8, 6),
        'font.size': 10,
        'axes.grid': True,
    })
    
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None


class FigureManager:
    """Manages matplotlib figures during testing."""
    
    def __init__(self):
        self.created_figures = []
    
    def track_figure(self, fig):
        """Track a created figure for cleanup."""
        if MATPLOTLIB_AVAILABLE and fig is not None:
            self.created_figures.append(fig)
    
    def close_all_figures(self):
        """Close all tracked figures."""
        if MATPLOTLIB_AVAILABLE and plt is not None:
            # Close tracked figures
            for fig in self.created_figures:
                try:
                    plt.close(fig)
                except Exception:
                    pass
            
            # Close any remaining figures
            plt.close('all')
            
            # Clear the list
            self.created_figures.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_all_figures()


# Global figure manager instance
_figure_manager = FigureManager()


@contextmanager
def managed_figures():
    """Context manager for automatic figure cleanup."""
    with _figure_manager:
        yield _figure_manager


def setup_testing_backend():
    """Setup matplotlib for testing environment."""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    # Ensure we're using the non-interactive backend
    if plt.get_backend() != 'Agg':
        matplotlib.use('Agg', force=True)
    
    # Turn off interactive mode
    plt.ioff()
    
    # Close any existing figures
    plt.close('all')


def create_test_figure(*args, **kwargs):
    """Create a figure for testing that will be automatically managed."""
    if not MATPLOTLIB_AVAILABLE:
        return None
    
    fig = plt.figure(*args, **kwargs)
    _figure_manager.track_figure(fig)
    return fig


def safe_show(fig=None, save_path: Optional[str] = None):
    """
    Safe replacement for plt.show() that doesn't block during testing.
    
    Args:
        fig: Figure to show/save (optional, uses current figure if None)
        save_path: Path to save figure instead of showing (optional)
    """
    if not MATPLOTLIB_AVAILABLE:
        return
    
    try:
        if save_path:
            if fig:
                fig.savefig(save_path, bbox_inches='tight', dpi=100)
            else:
                plt.savefig(save_path, bbox_inches='tight', dpi=100)
        
        # In testing environment, don't actually show
        if is_testing():
            return
        
        # In interactive environment, show the figure
        if fig:
            # For specific figure
            fig.show()
        else:
            # For current figure
            plt.show()
            
    except Exception:
        # Silently handle any display errors
        pass
    finally:
        # Always close the figure after showing/saving
        if fig:
            plt.close(fig)
        else:
            plt.close()


def is_testing() -> bool:
    """Check if we're currently running in a testing environment."""
    return (
        'pytest' in sys.modules or
        'unittest' in sys.modules or
        os.environ.get('TESTING', '').lower() in ('1', 'true', 'yes') or
        os.environ.get('CI', '').lower() in ('1', 'true', 'yes') or
        'test' in sys.argv[0].lower()
    )


def patch_matplotlib_show():
    """Patch matplotlib.pyplot.show to use safe_show during testing."""
    if not MATPLOTLIB_AVAILABLE or not is_testing():
        return
    
    # Replace plt.show with our safe version
    original_show = plt.show
    
    def testing_show(*args, **kwargs):
        """Testing replacement for plt.show that auto-closes figures."""
        safe_show()
    
    plt.show = testing_show
    
    return original_show


# Automatically setup testing environment when imported
if is_testing():
    setup_testing_backend()
    patch_matplotlib_show()


# Export the main utilities
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
