"""
Semantic Versioning Implementation

This module provides comprehensive semantic versioning management including:
- Semantic version parsing and comparison
- Version compatibility checking
- Automated version bumping
- Version constraint validation
"""

import json
import logging
import re
import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Version pattern following semantic versioning 2.0.0
VERSION_PATTERN = re.compile(
    r'^(?P<major>0|[1-9]\d*)'
    r'\.(?P<minor>0|[1-9]\d*)'
    r'\.(?P<patch>0|[1-9]\d*)'
    r'(?:-(?P<prerelease>(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)'
    r'(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?'
    r'(?:\+(?P<buildmetadata>[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$'
)


class VersionType(Enum):
    """Version bump types."""
    MAJOR = "major"
    MINOR = "minor"  
    PATCH = "patch"
    PRERELEASE = "prerelease"


class CompatibilityLevel(Enum):
    """API compatibility levels."""
    BREAKING = "breaking"      # Major version change required
    COMPATIBLE = "compatible"  # Minor version change allowed
    PATCH = "patch"           # Patch version change
    IDENTICAL = "identical"   # No changes


@dataclass
class SemanticVersion:
    """Represents a semantic version with comparison and manipulation capabilities."""
    
    major: int
    minor: int
    patch: int
    prerelease: Optional[str] = None
    build_metadata: Optional[str] = None
    
    def __post_init__(self):
        """Validate version components."""
        if self.major < 0 or self.minor < 0 or self.patch < 0:
            raise ValueError("Version components must be non-negative integers")
    
    @classmethod
    def parse(cls, version_string: str) -> 'SemanticVersion':
        """Parse a version string into a SemanticVersion object."""
        match = VERSION_PATTERN.match(version_string.strip())
        if not match:
            raise ValueError(f"Invalid semantic version string: {version_string}")
        
        groups = match.groupdict()
        return cls(
            major=int(groups['major']),
            minor=int(groups['minor']),
            patch=int(groups['patch']),
            prerelease=groups.get('prerelease'),
            build_metadata=groups.get('buildmetadata')
        )
    
    def __str__(self) -> str:
        """String representation of the version."""
        version = f"{self.major}.{self.minor}.{self.patch}"
        
        if self.prerelease:
            version += f"-{self.prerelease}"
        
        if self.build_metadata:
            version += f"+{self.build_metadata}"
        
        return version
    
    def __eq__(self, other: object) -> bool:
        """Check version equality (excluding build metadata)."""
        if not isinstance(other, SemanticVersion):
            return NotImplemented
        
        return (
            self.major == other.major and
            self.minor == other.minor and
            self.patch == other.patch and
            self.prerelease == other.prerelease
        )
    
    def __lt__(self, other: 'SemanticVersion') -> bool:
        """Compare versions for ordering."""
        if not isinstance(other, SemanticVersion):
            return NotImplemented
        
        # Compare major.minor.patch
        if (self.major, self.minor, self.patch) != (other.major, other.minor, other.patch):
            return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)
        
        # Handle prerelease comparison
        if self.prerelease is None and other.prerelease is None:
            return False
        elif self.prerelease is None:
            return False  # 1.0.0 > 1.0.0-alpha
        elif other.prerelease is None:
            return True   # 1.0.0-alpha < 1.0.0
        else:
            # Compare prerelease versions
            return self._compare_prerelease(self.prerelease, other.prerelease)
    
    def __le__(self, other: 'SemanticVersion') -> bool:
        """Less than or equal comparison."""
        return self == other or self < other
    
    def __gt__(self, other: 'SemanticVersion') -> bool:
        """Greater than comparison."""
        return not self <= other
    
    def __ge__(self, other: 'SemanticVersion') -> bool:
        """Greater than or equal comparison."""
        return not self < other
    
    def _compare_prerelease(self, pre1: str, pre2: str) -> bool:
        """Compare prerelease versions according to semver rules."""
        parts1 = pre1.split('.')
        parts2 = pre2.split('.')
        
        for i in range(max(len(parts1), len(parts2))):
            p1 = parts1[i] if i < len(parts1) else None
            p2 = parts2[i] if i < len(parts2) else None
            
            if p1 is None:
                return True  # Shorter prerelease < longer
            elif p2 is None:
                return False
            
            # Try to compare as integers first
            try:
                n1, n2 = int(p1), int(p2)
                if n1 != n2:
                    return n1 < n2
            except ValueError:
                # Compare as strings if not integers
                if p1 != p2:
                    return p1 < p2
        
        return False
    
    def bump(self, version_type: VersionType, prerelease_label: Optional[str] = None) -> 'SemanticVersion':
        """Create a new version with the specified component bumped."""
        if version_type == VersionType.MAJOR:
            return SemanticVersion(self.major + 1, 0, 0)
        elif version_type == VersionType.MINOR:
            return SemanticVersion(self.major, self.minor + 1, 0)
        elif version_type == VersionType.PATCH:
            return SemanticVersion(self.major, self.minor, self.patch + 1)
        elif version_type == VersionType.PRERELEASE:
            if not prerelease_label:
                prerelease_label = "alpha"
            
            if self.prerelease:
                # Try to increment existing prerelease
                parts = self.prerelease.split('.')
                if parts[-1].isdigit():
                    parts[-1] = str(int(parts[-1]) + 1)
                    new_prerelease = '.'.join(parts)
                else:
                    new_prerelease = f"{self.prerelease}.1"
            else:
                new_prerelease = f"{prerelease_label}.1"
            
            return SemanticVersion(
                self.major, self.minor, self.patch,
                prerelease=new_prerelease
            )
        else:
            raise ValueError(f"Unknown version type: {version_type}")
    
    def is_compatible_with(self, other: 'SemanticVersion') -> bool:
        """Check if this version is backward compatible with another version."""
        # Same major version means compatible (assuming semver conventions)
        if self.major != other.major:
            return False
        
        # Higher minor/patch versions are compatible with lower ones
        return self >= other
    
    def compatibility_level(self, other: 'SemanticVersion') -> CompatibilityLevel:
        """Determine the compatibility level between two versions."""
        if self == other:
            return CompatibilityLevel.IDENTICAL
        
        if self.major != other.major:
            return CompatibilityLevel.BREAKING
        
        if self.minor != other.minor:
            return CompatibilityLevel.COMPATIBLE
        
        if self.patch != other.patch:
            return CompatibilityLevel.PATCH
        
        # Only prerelease/build metadata differs
        return CompatibilityLevel.PATCH


class VersionCompatibility:
    """Manages version compatibility rules and constraints."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.compatibility_matrix: Dict[str, Dict[str, bool]] = {}
    
    def add_compatibility_rule(self, from_version: str, to_version: str, compatible: bool) -> None:
        """Add a specific compatibility rule between versions."""
        if from_version not in self.compatibility_matrix:
            self.compatibility_matrix[from_version] = {}
        
        self.compatibility_matrix[from_version][to_version] = compatible
        self.logger.info(f"Added compatibility rule: {from_version} -> {to_version} = {compatible}")
    
    def is_compatible(self, from_version: Union[str, SemanticVersion], 
                     to_version: Union[str, SemanticVersion]) -> bool:
        """Check if upgrading from one version to another is compatible."""
        
        # Convert strings to SemanticVersion objects
        if isinstance(from_version, str):
            from_version = SemanticVersion.parse(from_version)
        if isinstance(to_version, str):
            to_version = SemanticVersion.parse(to_version)
        
        # Check explicit compatibility rules first
        from_str = str(from_version)
        to_str = str(to_version)
        
        if from_str in self.compatibility_matrix:
            if to_str in self.compatibility_matrix[from_str]:
                return self.compatibility_matrix[from_str][to_str]
        
        # Apply standard semver compatibility rules
        return to_version.is_compatible_with(from_version)
    
    def get_compatible_versions(self, base_version: Union[str, SemanticVersion],
                              available_versions: List[Union[str, SemanticVersion]]) -> List[SemanticVersion]:
        """Get all compatible versions from a list of available versions."""
        
        if isinstance(base_version, str):
            base_version = SemanticVersion.parse(base_version)
        
        compatible = []
        for version in available_versions:
            if isinstance(version, str):
                version = SemanticVersion.parse(version)
            
            if self.is_compatible(base_version, version):
                compatible.append(version)
        
        return sorted(compatible)


class VersionManager:
    """Manages version information and operations for QeMLflow."""
    
    def __init__(self, version_file: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.version_file = Path(version_file) if version_file else Path("VERSION")
        self.compatibility = VersionCompatibility()
        
        # Load current version
        self._current_version = self._load_version()
    
    def _load_version(self) -> SemanticVersion:
        """Load version from file or default."""
        if self.version_file.exists():
            try:
                version_text = self.version_file.read_text().strip()
                return SemanticVersion.parse(version_text)
            except Exception as e:
                self.logger.error(f"Failed to load version from {self.version_file}: {e}")
        
        # Default version if file doesn't exist or can't be parsed
        default_version = SemanticVersion(0, 1, 0)
        self._save_version(default_version)
        return default_version
    
    def _save_version(self, version: SemanticVersion) -> None:
        """Save version to file."""
        try:
            self.version_file.write_text(str(version))
            self.logger.info(f"Saved version {version} to {self.version_file}")
        except Exception as e:
            self.logger.error(f"Failed to save version to {self.version_file}: {e}")
    
    @property
    def current_version(self) -> SemanticVersion:
        """Get the current version."""
        return self._current_version
    
    def bump_version(self, version_type: VersionType, 
                    prerelease_label: Optional[str] = None,
                    save: bool = True) -> SemanticVersion:
        """Bump the version and optionally save it."""
        new_version = self._current_version.bump(version_type, prerelease_label)
        
        if save:
            self._save_version(new_version)
            self._current_version = new_version
        
        self.logger.info(f"Version bumped from {self._current_version} to {new_version}")
        return new_version
    
    def set_version(self, version: Union[str, SemanticVersion], save: bool = True) -> None:
        """Set a specific version."""
        if isinstance(version, str):
            version = SemanticVersion.parse(version)
        
        if save:
            self._save_version(version)
        
        self._current_version = version
        self.logger.info(f"Version set to {version}")
    
    def check_compatibility(self, target_version: Union[str, SemanticVersion]) -> CompatibilityLevel:
        """Check compatibility between current version and target version."""
        if isinstance(target_version, str):
            target_version = SemanticVersion.parse(target_version)
        
        return self._current_version.compatibility_level(target_version)
    
    def suggest_version_bump(self, changes: List[str]) -> VersionType:
        """Suggest appropriate version bump based on change descriptions."""
        # Simple heuristic - can be enhanced with ML or rule-based classification
        breaking_keywords = ['breaking', 'remove', 'delete', 'incompatible', 'major']
        feature_keywords = ['add', 'new', 'feature', 'enhance', 'minor']
        
        changes_text = ' '.join(changes).lower()
        
        for keyword in breaking_keywords:
            if keyword in changes_text:
                return VersionType.MAJOR
        
        for keyword in feature_keywords:
            if keyword in changes_text:
                return VersionType.MINOR
        
        # Default to patch for bug fixes and small changes
        return VersionType.PATCH
    
    def get_version_info(self) -> Dict[str, Any]:
        """Get comprehensive version information."""
        return {
            "version": str(self._current_version),
            "major": self._current_version.major,
            "minor": self._current_version.minor,
            "patch": self._current_version.patch,
            "prerelease": self._current_version.prerelease,
            "build_metadata": self._current_version.build_metadata,
            "version_file": str(self.version_file),
            "is_prerelease": self._current_version.prerelease is not None,
            "is_stable": self._current_version.prerelease is None and self._current_version.major > 0
        }


# Global version manager instance
_version_manager: Optional[VersionManager] = None


def get_version_manager() -> VersionManager:
    """Get the global version manager instance."""
    global _version_manager
    if _version_manager is None:
        _version_manager = VersionManager()
    return _version_manager


def get_version() -> str:
    """Get the current QeMLflow version string."""
    return str(get_version_manager().current_version)


def version_info() -> Dict[str, Any]:
    """Get comprehensive version information."""
    return get_version_manager().get_version_info()


def check_compatibility(from_version: Union[str, SemanticVersion],
                       to_version: Union[str, SemanticVersion]) -> bool:
    """Check if upgrading from one version to another is compatible."""
    return get_version_manager().compatibility.is_compatible(from_version, to_version)


def parse_version(version_string: str) -> SemanticVersion:
    """Parse a version string into a SemanticVersion object."""
    return SemanticVersion.parse(version_string)


def compare_versions(version1: Union[str, SemanticVersion],
                    version2: Union[str, SemanticVersion]) -> int:
    """Compare two versions. Returns -1, 0, or 1."""
    if isinstance(version1, str):
        version1 = SemanticVersion.parse(version1)
    if isinstance(version2, str):
        version2 = SemanticVersion.parse(version2)
    
    if version1 < version2:
        return -1
    elif version1 > version2:
        return 1
    else:
        return 0
