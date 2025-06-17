"""
ChemML Common - Shared Infrastructure for ChemML Bootcamp Scripts
================================================================

This package provides unified infrastructure components for all ChemML scripts,
eliminating code duplication and following Python best practices.

Author: ChemML Enhancement System
Version: 2.0.0
"""

__version__ = "2.0.0"
__author__ = "ChemML Enhancement System"

from .assessment.framework import AssessmentFramework
from .config.environment import ChemMLConfig, get_config, print_banner
from .core.base_runner import BaseRunner, SectionRunner
from .libraries.manager import LibraryManager

__all__ = [
    "ChemMLConfig",
    "get_config",
    "print_banner",
    "BaseRunner",
    "SectionRunner",
    "LibraryManager",
    "AssessmentFramework",
]
