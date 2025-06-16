"""
ChemML Notebook Integration
==========================

Tools and utilities for integrating ChemML with Jupyter notebooks.
"""

from .integration import (
    BasicChemMLTemplate,
    IntermediateChemMLTemplate,
    NotebookTemplate,
    create_notebook_templates,
    integrate_existing_notebooks,
)

__all__ = [
    "NotebookTemplate",
    "BasicChemMLTemplate",
    "IntermediateChemMLTemplate",
    "create_notebook_templates",
    "integrate_existing_notebooks",
]
