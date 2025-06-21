"""
Deprecation Policy Framework

This module provides comprehensive deprecation management including:
- Deprecation warnings and notifications
- Deprecation timeline management
- Migration path documentation
- Automated deprecation tracking
"""

import functools
import logging
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar, cast

from .versioning import SemanticVersion, VersionManager

# Type variable for decorated functions
F = TypeVar('F', bound=Callable[..., Any])


class DeprecationLevel(Enum):
    """Deprecation severity levels."""
    NOTICE = "notice"          # Early warning, no timeline yet
    SCHEDULED = "scheduled"    # Deprecation scheduled for specific version
    PENDING = "pending"        # Removal imminent in next major version
    URGENT = "urgent"          # Removal in next minor version (breaking change)


@dataclass
class DeprecationInfo:
    """Information about a deprecated API element."""
    
    element_name: str
    element_type: str  # 'function', 'class', 'method', 'parameter', 'module'
    deprecated_since: str  # Version when deprecation started
    removal_version: Optional[str] = None  # Version when removal is planned
    reason: str = ""
    alternative: str = ""
    migration_guide: str = ""
    level: DeprecationLevel = DeprecationLevel.NOTICE
    
    # Tracking information
    first_warned: Optional[datetime] = None
    warning_count: int = 0
    last_warned: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'element_name': self.element_name,
            'element_type': self.element_type,
            'deprecated_since': self.deprecated_since,
            'removal_version': self.removal_version,
            'reason': self.reason,
            'alternative': self.alternative,
            'migration_guide': self.migration_guide,
            'level': self.level.value,
            'first_warned': self.first_warned.isoformat() if self.first_warned else None,
            'warning_count': self.warning_count,
            'last_warned': self.last_warned.isoformat() if self.last_warned else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DeprecationInfo':
        """Create from dictionary."""
        info = cls(
            element_name=data['element_name'],
            element_type=data['element_type'],
            deprecated_since=data['deprecated_since'],
            removal_version=data.get('removal_version'),
            reason=data.get('reason', ''),
            alternative=data.get('alternative', ''),
            migration_guide=data.get('migration_guide', ''),
            level=DeprecationLevel(data.get('level', 'notice')),
            warning_count=data.get('warning_count', 0)
        )
        
        if data.get('first_warned'):
            info.first_warned = datetime.fromisoformat(data['first_warned'])
        if data.get('last_warned'):
            info.last_warned = datetime.fromisoformat(data['last_warned'])
        
        return info


class DeprecationManager:
    """Manages deprecation policies and warnings across the codebase."""
    
    def __init__(self, deprecation_file: str = "deprecations.json"):
        self.deprecation_file = Path(deprecation_file)
        self.logger = logging.getLogger(__name__)
        self.version_manager = VersionManager()
        
        # Load existing deprecations
        self.deprecations: Dict[str, DeprecationInfo] = self._load_deprecations()
        
        # Warning settings
        self.warning_frequency = {
            DeprecationLevel.NOTICE: timedelta(days=7),     # Weekly
            DeprecationLevel.SCHEDULED: timedelta(days=1),  # Daily
            DeprecationLevel.PENDING: timedelta(hours=1),   # Hourly
            DeprecationLevel.URGENT: timedelta(minutes=1)   # Every call
        }
    
    def _load_deprecations(self) -> Dict[str, DeprecationInfo]:
        """Load deprecation information from file."""
        if not self.deprecation_file.exists():
            return {}
        
        try:
            import json
            with open(self.deprecation_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return {
                name: DeprecationInfo.from_dict(info_dict)
                for name, info_dict in data.items()
            }
        
        except Exception as e:
            self.logger.error(f"Failed to load deprecations from {self.deprecation_file}: {e}")
            return {}
    
    def _save_deprecations(self) -> None:
        """Save deprecation information to file."""
        try:
            import json
            data = {
                name: info.to_dict()
                for name, info in self.deprecations.items()
            }
            
            with open(self.deprecation_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            self.logger.debug(f"Saved {len(self.deprecations)} deprecations to {self.deprecation_file}")
        
        except Exception as e:
            self.logger.error(f"Failed to save deprecations to {self.deprecation_file}: {e}")
    
    def register_deprecation(self,
                           element_name: str,
                           element_type: str,
                           reason: str = "",
                           alternative: str = "",
                           removal_version: Optional[str] = None,
                           migration_guide: str = "") -> DeprecationInfo:
        """Register a new deprecation."""
        
        current_version = str(self.version_manager.current_version)
        
        # Determine deprecation level
        level = DeprecationLevel.NOTICE
        if removal_version:
            try:
                removal_ver = SemanticVersion.parse(removal_version)
                current_ver = self.version_manager.current_version
                
                if removal_ver.major == current_ver.major:
                    if removal_ver.minor == current_ver.minor + 1:
                        level = DeprecationLevel.URGENT
                    else:
                        level = DeprecationLevel.PENDING
                elif removal_ver.major == current_ver.major + 1:
                    level = DeprecationLevel.SCHEDULED
                    
            except Exception:
                pass
        
        info = DeprecationInfo(
            element_name=element_name,
            element_type=element_type,
            deprecated_since=current_version,
            removal_version=removal_version,
            reason=reason,
            alternative=alternative,
            migration_guide=migration_guide,
            level=level
        )
        
        self.deprecations[element_name] = info
        self._save_deprecations()
        
        self.logger.info(f"Registered deprecation: {element_name} (level: {level.value})")
        return info
    
    def should_warn(self, element_name: str) -> bool:
        """Check if a deprecation warning should be issued."""
        if element_name not in self.deprecations:
            return False
        
        info = self.deprecations[element_name]
        
        # Always warn on first use
        if info.first_warned is None:
            return True
        
        # Check frequency based on level
        frequency = self.warning_frequency.get(info.level, timedelta(days=1))
        
        if info.last_warned is None:
            return True
        
        return datetime.now() - info.last_warned >= frequency
    
    def warn_deprecated(self, element_name: str, 
                       additional_message: str = "",
                       stacklevel: int = 2) -> None:
        """Issue a deprecation warning for an element."""
        
        if element_name not in self.deprecations:
            return
        
        if not self.should_warn(element_name):
            return
        
        info = self.deprecations[element_name]
        
        # Update tracking information
        now = datetime.now()
        if info.first_warned is None:
            info.first_warned = now
        info.last_warned = now
        info.warning_count += 1
        
        # Build warning message
        message_parts = [f"{info.element_type.title()} '{element_name}' is deprecated"]
        
        if info.deprecated_since:
            message_parts.append(f"since version {info.deprecated_since}")
        
        if info.removal_version:
            message_parts.append(f"and will be removed in version {info.removal_version}")
        
        if info.reason:
            message_parts.append(f"Reason: {info.reason}")
        
        if info.alternative:
            message_parts.append(f"Use '{info.alternative}' instead")
        
        if additional_message:
            message_parts.append(additional_message)
        
        if info.migration_guide:
            message_parts.append(f"Migration guide: {info.migration_guide}")
        
        message = ". ".join(message_parts) + "."
        
        # Issue warning with appropriate category
        warning_category: type = DeprecationWarning
        if info.level == DeprecationLevel.URGENT:
            warning_category = FutureWarning
        
        warnings.warn(message, warning_category, stacklevel=stacklevel)
        
        # Save updated tracking info
        self._save_deprecations()
    
    def get_deprecation_status(self) -> Dict[str, Any]:
        """Get comprehensive deprecation status report."""
        
        level_counts = {}
        for level in DeprecationLevel:
            level_counts[level.value] = sum(
                1 for info in self.deprecations.values() 
                if info.level == level
            )
        
        # Find deprecations approaching removal
        approaching_removal = []
        current_version = self.version_manager.current_version
        
        for info in self.deprecations.values():
            if info.removal_version:
                try:
                    removal_ver = SemanticVersion.parse(info.removal_version)
                    if removal_ver <= current_version:
                        approaching_removal.append(info.element_name)
                except Exception:
                    pass
        
        return {
            "total_deprecations": len(self.deprecations),
            "by_level": level_counts,
            "approaching_removal": approaching_removal,
            "active_warnings": sum(info.warning_count for info in self.deprecations.values()),
            "deprecations": [info.to_dict() for info in self.deprecations.values()]
        }
    
    def cleanup_removed_deprecations(self, current_version: Optional[str] = None) -> int:
        """Remove deprecations for elements that should have been removed."""
        
        if current_version is None:
            current_version = str(self.version_manager.current_version)
        
        current_ver = SemanticVersion.parse(current_version)
        removed_count = 0
        
        to_remove = []
        for name, info in self.deprecations.items():
            if info.removal_version:
                try:
                    removal_ver = SemanticVersion.parse(info.removal_version)
                    if current_ver >= removal_ver:
                        to_remove.append(name)
                except Exception:
                    continue
        
        for name in to_remove:
            del self.deprecations[name]
            removed_count += 1
            self.logger.info(f"Removed deprecation record for {name} (should have been removed)")
        
        if removed_count > 0:
            self._save_deprecations()
        
        return removed_count


def deprecated(reason: str = "",
               alternative: str = "",
               removal_version: Optional[str] = None,
               migration_guide: str = "") -> Callable[[F], F]:
    """
    Decorator to mark functions, methods, or classes as deprecated.
    
    Args:
        reason: Reason for deprecation
        alternative: Suggested alternative to use
        removal_version: Version when the element will be removed
        migration_guide: URL or description of migration guide
    
    Returns:
        Decorated function with deprecation warning
    """
    
    def decorator(func: F) -> F:
        # Get deprecation manager
        manager = get_deprecation_manager()
        
        # Register the deprecation
        element_type = "class" if isinstance(func, type) else "function"
        full_name = f"{func.__module__}.{func.__qualname__}"
        
        manager.register_deprecation(
            element_name=full_name,
            element_type=element_type,
            reason=reason,
            alternative=alternative,
            removal_version=removal_version,
            migration_guide=migration_guide
        )
        
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Issue deprecation warning
            manager.warn_deprecated(full_name, stacklevel=2)
            
            # Call original function
            return func(*args, **kwargs)
        
        # For classes, we need a different approach
        if isinstance(func, type):
            # Handle class deprecation
            original_init = getattr(func, '__init__', None)
            if original_init:
                @functools.wraps(original_init)
                def new_init(self: Any, *args: Any, **kwargs: Any) -> None:
                    manager.warn_deprecated(full_name, stacklevel=2)
                    if original_init:
                        original_init(self, *args, **kwargs)
                
                setattr(func, '__init__', new_init)
            
            return func  # type: ignore[return-value]
        
        return wrapper  # type: ignore
    
    return decorator


def deprecate_parameter(parameter_name: str,
                       reason: str = "",
                       alternative: str = "",
                       removal_version: Optional[str] = None) -> Callable[[F], F]:
    """
    Decorator to mark specific parameters as deprecated.
    
    Args:
        parameter_name: Name of the deprecated parameter
        reason: Reason for deprecation
        alternative: Suggested alternative parameter
        removal_version: Version when the parameter will be removed
    
    Returns:
        Decorated function with parameter deprecation warning
    """
    
    def decorator(func: F) -> F:
        # Get deprecation manager
        manager = get_deprecation_manager()
        
        # Register the parameter deprecation
        full_name = f"{func.__module__}.{func.__qualname__}.{parameter_name}"
        
        manager.register_deprecation(
            element_name=full_name,
            element_type="parameter",
            reason=reason,
            alternative=alternative,
            removal_version=removal_version
        )
        
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Check if deprecated parameter is used
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            if parameter_name in bound_args.arguments:
                additional_msg = f"Parameter '{parameter_name}' was provided"
                manager.warn_deprecated(full_name, additional_msg, stacklevel=2)
            
            return func(*args, **kwargs)
        
        return wrapper  # type: ignore
    
    return decorator


# Global deprecation manager instance
_deprecation_manager: Optional[DeprecationManager] = None


def get_deprecation_manager() -> DeprecationManager:
    """Get the global deprecation manager instance."""
    global _deprecation_manager
    if _deprecation_manager is None:
        _deprecation_manager = DeprecationManager()
    return _deprecation_manager


def register_deprecation(element_name: str,
                        element_type: str,
                        reason: str = "",
                        alternative: str = "",
                        removal_version: Optional[str] = None,
                        migration_guide: str = "") -> DeprecationInfo:
    """Register a deprecation using the global manager."""
    return get_deprecation_manager().register_deprecation(
        element_name, element_type, reason, alternative, removal_version, migration_guide
    )


def warn_deprecated(element_name: str, additional_message: str = "") -> None:
    """Issue a deprecation warning using the global manager."""
    get_deprecation_manager().warn_deprecated(element_name, additional_message, stacklevel=2)


def get_deprecation_status() -> Dict[str, Any]:
    """Get deprecation status using the global manager."""
    return get_deprecation_manager().get_deprecation_status()
