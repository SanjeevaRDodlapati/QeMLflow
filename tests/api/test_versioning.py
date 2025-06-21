"""
Test suite for API versioning functionality.
"""

import pytest
from unittest.mock import patch
from pathlib import Path

from src.qemlflow.api.versioning import (
    SemanticVersion, 
    VersionType,
    CompatibilityLevel,
    VersionManager,
    get_version_manager
)


class TestSemanticVersion:
    """Test semantic version parsing and comparison."""
    
    def test_version_parsing(self):
        """Test parsing of semantic version strings."""
        version = SemanticVersion.parse("1.2.3")
        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3
        assert version.prerelease is None
        assert version.build_metadata is None
    
    def test_version_parsing_with_prerelease(self):
        """Test parsing with prerelease information."""
        version = SemanticVersion.parse("1.2.3-alpha.1")
        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3
        assert version.prerelease == "alpha.1"
    
    def test_version_parsing_with_build_metadata(self):
        """Test parsing with build metadata."""
        version = SemanticVersion.parse("1.2.3+build.1")
        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3
        assert version.build_metadata == "build.1"
    
    def test_version_parsing_invalid(self):
        """Test parsing of invalid version strings."""
        with pytest.raises(ValueError):
            SemanticVersion.parse("invalid")
        
        with pytest.raises(ValueError):
            SemanticVersion.parse("1.2")
        
        with pytest.raises(ValueError):
            SemanticVersion.parse("1.2.3.4")
    
    def test_version_comparison(self):
        """Test version comparison operations."""
        v1 = SemanticVersion.parse("1.0.0")
        v2 = SemanticVersion.parse("1.1.0")
        v3 = SemanticVersion.parse("2.0.0")
        
        assert v1 < v2
        assert v2 < v3
        assert v1 < v3
        
        assert v2 > v1
        assert v3 > v2
        assert v3 > v1
        
        assert v1 == SemanticVersion.parse("1.0.0")
        assert v1 != v2
    
    def test_prerelease_comparison(self):
        """Test comparison with prerelease versions."""
        v1 = SemanticVersion.parse("1.0.0-alpha")
        v2 = SemanticVersion.parse("1.0.0-alpha.1")
        v3 = SemanticVersion.parse("1.0.0-alpha.2")
        v4 = SemanticVersion.parse("1.0.0")
        
        assert v1 < v2
        assert v2 < v3
        assert v3 < v4
    
    def test_version_bumping(self):
        """Test version bumping functionality."""
        base = SemanticVersion.parse("1.2.3")
        
        major = base.bump(VersionType.MAJOR)
        assert str(major) == "2.0.0"
        
        minor = base.bump(VersionType.MINOR)
        assert str(minor) == "1.3.0"
        
        patch = base.bump(VersionType.PATCH)
        assert str(patch) == "1.2.4"
        
        prerelease = base.bump(VersionType.PRERELEASE)
        assert str(prerelease) == "1.2.3-alpha.1"
    
    def test_compatibility_checking(self):
        """Test compatibility checking between versions."""
        v1 = SemanticVersion.parse("1.0.0")
        v2 = SemanticVersion.parse("1.1.0")
        v3 = SemanticVersion.parse("2.0.0")
        
        assert v2.is_compatible_with(v1)
        assert not v1.is_compatible_with(v2)
        assert not v3.is_compatible_with(v1)
        assert not v1.is_compatible_with(v3)
    
    def test_compatibility_level(self):
        """Test compatibility level determination."""
        base = SemanticVersion.parse("1.2.3")
        
        identical = SemanticVersion.parse("1.2.3")
        assert base.compatibility_level(identical) == CompatibilityLevel.IDENTICAL
        
        patch = SemanticVersion.parse("1.2.4")
        assert base.compatibility_level(patch) == CompatibilityLevel.PATCH
        
        minor = SemanticVersion.parse("1.3.0")
        assert base.compatibility_level(minor) == CompatibilityLevel.COMPATIBLE
        
        major = SemanticVersion.parse("2.0.0")
        assert base.compatibility_level(major) == CompatibilityLevel.BREAKING


class TestVersionManager:
    """Test version management functionality."""
    
    @pytest.fixture
    def temp_version_file(self, tmp_path):
        """Create a temporary version file."""
        version_file = tmp_path / "VERSION"
        return str(version_file)
    
    def test_version_manager_initialization(self, temp_version_file):
        """Test version manager initialization."""
        manager = VersionManager(temp_version_file)
        
        # Should create default version if file doesn't exist
        assert manager.current_version.major == 0
        assert manager.current_version.minor == 1
        assert manager.current_version.patch == 0
        
        # Version file should be created
        assert Path(temp_version_file).exists()
    
    def test_version_loading(self, temp_version_file):
        """Test loading version from file."""
        # Create version file with specific version
        Path(temp_version_file).write_text("2.1.0")
        
        manager = VersionManager(temp_version_file)
        assert str(manager.current_version) == "2.1.0"
    
    def test_version_bumping(self, temp_version_file):
        """Test version bumping through manager."""
        manager = VersionManager(temp_version_file)
        original_version = manager.current_version
        
        # Bump minor version
        new_version = manager.bump_version(VersionType.MINOR)
        assert new_version.minor == original_version.minor + 1
        assert manager.current_version == new_version
        
        # Version should be saved to file
        file_content = Path(temp_version_file).read_text().strip()
        assert file_content == str(new_version)
    
    def test_version_setting(self, temp_version_file):
        """Test setting specific version."""
        manager = VersionManager(temp_version_file)
        
        manager.set_version("3.5.7")
        assert str(manager.current_version) == "3.5.7"
        
        # Version should be saved to file
        file_content = Path(temp_version_file).read_text().strip()
        assert file_content == "3.5.7"
    
    def test_compatibility_checking(self, temp_version_file):
        """Test compatibility checking through manager."""
        manager = VersionManager(temp_version_file)
        manager.set_version("1.5.0")
        
        level = manager.check_compatibility("1.6.0")
        assert level == CompatibilityLevel.COMPATIBLE
        
        level = manager.check_compatibility("2.0.0")
        assert level == CompatibilityLevel.BREAKING
    
    def test_version_bump_suggestion(self, temp_version_file):
        """Test version bump suggestion based on changes."""
        manager = VersionManager(temp_version_file)
        
        # Breaking changes should suggest major bump
        suggestion = manager.suggest_version_bump(["breaking change", "remove old API"])
        assert suggestion == VersionType.MAJOR
        
        # New features should suggest minor bump
        suggestion = manager.suggest_version_bump(["add new feature", "enhance functionality"])
        assert suggestion == VersionType.MINOR
        
        # Bug fixes should suggest patch bump
        suggestion = manager.suggest_version_bump(["fix bug", "improve performance"])
        assert suggestion == VersionType.PATCH
    
    def test_version_info(self, temp_version_file):
        """Test version information retrieval."""
        manager = VersionManager(temp_version_file)
        manager.set_version("1.2.3-alpha.1")
        
        info = manager.get_version_info()
        
        assert info["version"] == "1.2.3-alpha.1"
        assert info["major"] == 1
        assert info["minor"] == 2
        assert info["patch"] == 3
        assert info["prerelease"] == "alpha.1"
        assert info["is_prerelease"] is True
        assert info["is_stable"] is False


class TestGlobalVersionManager:
    """Test global version manager functionality."""
    
    def test_global_manager_singleton(self):
        """Test that global manager is a singleton."""
        manager1 = get_version_manager()
        manager2 = get_version_manager()
        
        assert manager1 is manager2
    
    @patch('src.qemlflow.api.versioning._version_manager', None)
    def test_global_manager_initialization(self):
        """Test global manager initialization."""
        from src.qemlflow.api.versioning import get_version_manager
        
        manager = get_version_manager()
        assert manager is not None
        assert isinstance(manager, VersionManager)


if __name__ == "__main__":
    pytest.main([__file__])
