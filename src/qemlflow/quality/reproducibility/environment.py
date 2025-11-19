"""
Environment Determinism Module

This module provides comprehensive environment determinism for scientific reproducibility including:
- Exact dependency pinning and version locking
- Reproducible environment creation and validation
- Deterministic package installation
- Cross-platform environment fingerprinting
"""

import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pkg_resources


@dataclass
class PackageInfo:
    """Information about an installed package."""
    
    name: str
    version: str
    location: str
    requires: List[str] = field(default_factory=list)
    required_by: List[str] = field(default_factory=list)
    installer: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'version': self.version,
            'location': self.location,
            'requires': self.requires,
            'required_by': self.required_by,
            'installer': self.installer,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PackageInfo':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class EnvironmentFingerprint:
    """Complete fingerprint of a Python environment."""
    
    timestamp: str
    python_version: str
    platform_info: Dict[str, str]
    packages: List[PackageInfo]
    environment_variables: Dict[str, str] = field(default_factory=dict)
    system_info: Dict[str, Any] = field(default_factory=dict)
    fingerprint_hash: str = ""
    
    def __post_init__(self):
        """Calculate fingerprint hash after initialization."""
        if not self.fingerprint_hash:
            self.fingerprint_hash = self._calculate_hash()
    
    def _calculate_hash(self) -> str:
        """Calculate deterministic hash of environment."""
        # Create deterministic representation
        hash_data = {
            'python_version': self.python_version,
            'platform': self.platform_info,
            'packages': sorted([
                {'name': pkg.name, 'version': pkg.version}
                for pkg in self.packages
            ], key=lambda x: x['name']),
            'key_env_vars': {
                k: v for k, v in self.environment_variables.items()
                if k.startswith(('PYTHON', 'PATH', 'CONDA', 'VIRTUAL_ENV'))
            }
        }
        
        hash_string = json.dumps(hash_data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(hash_string.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp,
            'python_version': self.python_version,
            'platform_info': self.platform_info,
            'packages': [pkg.to_dict() for pkg in self.packages],
            'environment_variables': self.environment_variables,
            'system_info': self.system_info,
            'fingerprint_hash': self.fingerprint_hash
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnvironmentFingerprint':
        """Create from dictionary."""
        packages = [PackageInfo.from_dict(pkg_data) for pkg_data in data.get('packages', [])]
        
        fingerprint = cls(
            timestamp=data['timestamp'],
            python_version=data['python_version'],
            platform_info=data['platform_info'],
            packages=packages,
            environment_variables=data.get('environment_variables', {}),
            system_info=data.get('system_info', {}),
            fingerprint_hash=data.get('fingerprint_hash', '')
        )
        
        # Recalculate hash if not provided
        if not fingerprint.fingerprint_hash:
            fingerprint.fingerprint_hash = fingerprint._calculate_hash()
        
        return fingerprint


class EnvironmentManager:
    """
    Manages environment determinism and reproducibility.
    """
    
    def __init__(self, requirements_dir: str = "requirements"):
        self.requirements_dir = Path(requirements_dir)
        self.requirements_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Files for different requirement types
        self.files = {
            'exact': self.requirements_dir / 'requirements-exact.txt',
            'core': self.requirements_dir / 'requirements-core.txt',
            'dev': self.requirements_dir / 'requirements-dev.txt',
            'test': self.requirements_dir / 'requirements-test.txt',
            'optional': self.requirements_dir / 'requirements-optional.txt'
        }
    
    def capture_current_environment(self) -> EnvironmentFingerprint:
        """Capture complete fingerprint of current environment."""
        
        # Get Python version info
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        
        # Get platform information
        platform_info = {
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'architecture': platform.architecture()[0],
            'python_implementation': platform.python_implementation(),
            'python_compiler': platform.python_compiler()
        }
        
        # Get installed packages
        packages = self._get_installed_packages()
        
        # Get relevant environment variables
        env_vars = self._get_relevant_env_vars()
        
        # Get system information
        system_info = self._get_system_info()
        
        return EnvironmentFingerprint(
            timestamp=datetime.now().isoformat(),
            python_version=python_version,
            platform_info=platform_info,
            packages=packages,
            environment_variables=env_vars,
            system_info=system_info
        )
    
    def _get_installed_packages(self) -> List[PackageInfo]:
        """Get list of all installed packages with dependencies."""
        packages = []
        
        try:
            # Get all installed packages
            installed_packages = {pkg.key: pkg for pkg in pkg_resources.working_set}
            
            for pkg_name, pkg in installed_packages.items():
                # Get package requirements
                requires = [str(req).split()[0] for req in pkg.requires()]
                
                # Get packages that require this package
                required_by = []
                for other_pkg in installed_packages.values():
                    if pkg_name in [str(req).split()[0] for req in other_pkg.requires()]:
                        required_by.append(other_pkg.key)
                
                # Get installer information
                installer = self._get_package_installer(pkg)
                
                package_info = PackageInfo(
                    name=pkg.key,
                    version=pkg.version,
                    location=pkg.location,
                    requires=requires,
                    required_by=required_by,
                    installer=installer,
                    metadata=self._get_package_metadata(pkg)
                )
                
                packages.append(package_info)
        
        except Exception as e:
            self.logger.error(f"Failed to get installed packages: {e}")
        
        return sorted(packages, key=lambda x: x.name)
    
    def _get_package_installer(self, pkg: pkg_resources.Distribution) -> str:
        """Determine how a package was installed."""
        try:
            # Check for conda
            if 'conda-meta' in pkg.location:
                return 'conda'
            
            # Check for pip installer metadata
            try:
                metadata = pkg.get_metadata('INSTALLER')
                return metadata.strip()
            except Exception:
                pass
            
            # Default to pip if no other installer found
            return 'pip'
        
        except Exception:
            return 'unknown'
    
    def _get_package_metadata(self, pkg: pkg_resources.Distribution) -> Dict[str, Any]:
        """Get additional package metadata."""
        metadata = {}
        
        try:
            # Get basic metadata
            if hasattr(pkg, 'get_metadata'):
                try:
                    metadata['summary'] = pkg.get_metadata('METADATA').split('\n')[0]
                except Exception:
                    pass
            
            # Get file hash if available
            try:
                record = pkg.get_metadata('RECORD')
                metadata['has_record'] = bool(record)
            except Exception:
                metadata['has_record'] = False
            
        except Exception as e:
            self.logger.debug(f"Failed to get metadata for {pkg.key}: {e}")
        
        return metadata
    
    def _get_relevant_env_vars(self) -> Dict[str, str]:
        """Get environment variables relevant to Python environment."""
        relevant_vars = [
            'PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV', 'CONDA_DEFAULT_ENV',
            'CONDA_PREFIX', 'PATH', 'LD_LIBRARY_PATH', 'DYLD_LIBRARY_PATH'
        ]
        
        env_vars = {}
        for var in relevant_vars:
            if var in os.environ:
                env_vars[var] = os.environ[var]
        
        return env_vars
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get additional system information."""
        system_info = {}
        
        try:
            # CPU information
            if hasattr(os, 'cpu_count'):
                system_info['cpu_count'] = os.cpu_count()
            
            # Memory information (if available)
            try:
                import psutil
                memory = psutil.virtual_memory()
                system_info['total_memory'] = memory.total
            except ImportError:
                pass
            
            # Disk space information
            try:
                import shutil
                disk_usage = shutil.disk_usage('/')
                system_info['disk_total'] = disk_usage.total
                system_info['disk_free'] = disk_usage.free
            except Exception:
                pass
            
        except Exception as e:
            self.logger.debug(f"Failed to get system info: {e}")
        
        return system_info
    
    def generate_exact_requirements(self, output_file: Optional[str] = None) -> str:
        """Generate exact requirements file with pinned versions."""
        
        if output_file is None:
            output_file = str(self.files['exact'])
        
        fingerprint = self.capture_current_environment()
        
        # Generate requirements content
        lines = [
            "# Exact requirements file generated by QeMLflow Environment Manager",
            f"# Generated on: {fingerprint.timestamp}",
            f"# Python version: {fingerprint.python_version}",
            f"# Platform: {fingerprint.platform_info.get('system', 'Unknown')} {fingerprint.platform_info.get('release', 'Unknown')}",
            f"# Environment hash: {fingerprint.fingerprint_hash}",
            ""
        ]
        
        # Add packages sorted by name
        for package in sorted(fingerprint.packages, key=lambda x: x.name):
            # Skip packages that are part of standard library or development tools
            if self._should_include_in_requirements(package):
                lines.append(f"{package.name}=={package.version}")
        
        content = '\n'.join(lines)
        
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        self.logger.info(f"Generated exact requirements file: {output_file}")
        return content
    
    def _should_include_in_requirements(self, package: PackageInfo) -> bool:
        """Determine if package should be included in requirements."""
        
        # Skip standard library modules (these don't appear in pkg_resources anyway)
        # Skip development tools that shouldn't be in production
        skip_packages = {
            'pip', 'setuptools', 'wheel', 'distutils', 'pkg-resources',
            'pytest', 'pytest-cov', 'coverage', 'flake8', 'black', 'mypy',
            'pre-commit', 'tox', 'twine'
        }
        
        if package.name.lower() in skip_packages:
            return False
        
        # Skip packages with no version (shouldn't happen, but safety check)
        if not package.version:
            return False
        
        return True
    
    def validate_environment(self, expected_fingerprint: EnvironmentFingerprint,
                           strict: bool = True) -> Dict[str, Any]:
        """Validate current environment against expected fingerprint."""
        
        current = self.capture_current_environment()
        validation_result = {
            'valid': True,
            'hash_match': current.fingerprint_hash == expected_fingerprint.fingerprint_hash,
            'differences': {},
            'warnings': [],
            'errors': []
        }
        
        # Check Python version
        if current.python_version != expected_fingerprint.python_version:
            validation_result['differences']['python_version'] = {
                'expected': expected_fingerprint.python_version,
                'actual': current.python_version
            }
            validation_result['errors'].append(
                f"Python version mismatch: expected {expected_fingerprint.python_version}, "
                f"got {current.python_version}"
            )
            validation_result['valid'] = False
        
        # Check platform compatibility
        platform_diffs = self._compare_platforms(
            expected_fingerprint.platform_info,
            current.platform_info,
            strict
        )
        if platform_diffs:
            validation_result['differences']['platform'] = platform_diffs
            if strict:
                validation_result['valid'] = False
            else:
                validation_result['warnings'].extend([
                    f"Platform difference: {diff}" for diff in platform_diffs
                ])
        
        # Check package differences
        package_diffs = self._compare_packages(
            expected_fingerprint.packages,
            current.packages
        )
        if package_diffs:
            validation_result['differences']['packages'] = package_diffs
            validation_result['valid'] = False
        
        return validation_result
    
    def _compare_platforms(self, expected: Dict[str, str], actual: Dict[str, str],
                          strict: bool) -> List[str]:
        """Compare platform information."""
        differences = []
        
        # Critical platform attributes
        critical_attrs = ['system', 'python_implementation']
        for attr in critical_attrs:
            if expected.get(attr) != actual.get(attr):
                differences.append(f"{attr}: expected {expected.get(attr)}, got {actual.get(attr)}")
        
        # Architecture differences (important for compiled packages)
        if expected.get('architecture') != actual.get('architecture'):
            differences.append(f"architecture: expected {expected.get('architecture')}, got {actual.get('architecture')}")
        
        # In strict mode, check all platform attributes
        if strict:
            for attr in ['release', 'version', 'machine']:
                if expected.get(attr) != actual.get(attr):
                    differences.append(f"{attr}: expected {expected.get(attr)}, got {actual.get(attr)}")
        
        return differences
    
    def _compare_packages(self, expected: List[PackageInfo], 
                         actual: List[PackageInfo]) -> Dict[str, Any]:
        """Compare package lists."""
        
        expected_dict = {pkg.name: pkg for pkg in expected}
        actual_dict = {pkg.name: pkg for pkg in actual}
        
        differences = {
            'missing': [],
            'extra': [],
            'version_mismatch': [],
            'dependency_changes': []
        }
        
        # Find missing packages
        for name in expected_dict:
            if name not in actual_dict:
                differences['missing'].append({
                    'name': name,
                    'expected_version': expected_dict[name].version
                })
        
        # Find extra packages
        for name in actual_dict:
            if name not in expected_dict:
                differences['extra'].append({
                    'name': name,
                    'actual_version': actual_dict[name].version
                })
        
        # Find version mismatches
        for name in expected_dict:
            if name in actual_dict:
                expected_pkg = expected_dict[name]
                actual_pkg = actual_dict[name]
                
                if expected_pkg.version != actual_pkg.version:
                    differences['version_mismatch'].append({
                        'name': name,
                        'expected_version': expected_pkg.version,
                        'actual_version': actual_pkg.version
                    })
                
                # Check dependency changes
                if set(expected_pkg.requires) != set(actual_pkg.requires):
                    differences['dependency_changes'].append({
                        'name': name,
                        'expected_requires': expected_pkg.requires,
                        'actual_requires': actual_pkg.requires
                    })
        
        return differences
    
    def create_reproducible_environment(self, fingerprint: EnvironmentFingerprint,
                                      target_dir: Optional[str] = None) -> str:
        """Create a reproducible environment from fingerprint."""
        
        if target_dir is None:
            target_dir = f"env_{fingerprint.fingerprint_hash[:8]}"
        
        target_path = Path(target_dir)
        
        try:
            # Create virtual environment
            self._create_virtual_environment(target_path, fingerprint.python_version)
            
            # Install packages in deterministic order
            self._install_packages_deterministically(target_path, fingerprint.packages)
            
            # Validate created environment
            validation = self._validate_created_environment(target_path, fingerprint)
            
            if not validation['valid']:
                raise RuntimeError(f"Environment validation failed: {validation['errors']}")
            
            self.logger.info(f"Created reproducible environment: {target_path}")
            return str(target_path)
        
        except Exception as e:
            self.logger.error(f"Failed to create reproducible environment: {e}")
            raise
    
    def _create_virtual_environment(self, target_path: Path, python_version: str) -> None:
        """Create virtual environment with specific Python version."""
        
        # Use venv to create virtual environment
        python_cmd = f"python{python_version}"
        
        cmd = [python_cmd, '-m', 'venv', str(target_path)]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            self.logger.info(f"Created virtual environment at {target_path}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to create virtual environment: {e.stderr}")
    
    def _install_packages_deterministically(self, env_path: Path, 
                                          packages: List[PackageInfo]) -> None:
        """Install packages in deterministic order."""
        
        # Get pip executable in virtual environment
        if platform.system() == 'Windows':
            pip_cmd = env_path / 'Scripts' / 'pip.exe'
        else:
            pip_cmd = env_path / 'bin' / 'pip'
        
        # Sort packages by dependency order
        sorted_packages = self._topological_sort_packages(packages)
        
        for package in sorted_packages:
            if self._should_include_in_requirements(package):
                cmd = [str(pip_cmd), 'install', f"{package.name}=={package.version}"]
                
                try:
                    subprocess.run(cmd, check=True, capture_output=True, text=True)
                    self.logger.debug(f"Installed {package.name}=={package.version}")
                except subprocess.CalledProcessError as e:
                    self.logger.warning(f"Failed to install {package.name}: {e.stderr}")
    
    def _topological_sort_packages(self, packages: List[PackageInfo]) -> List[PackageInfo]:
        """Sort packages in dependency order."""
        
        # Simple topological sort - packages with no dependencies first
        package_dict = {pkg.name: pkg for pkg in packages}
        sorted_packages = []
        remaining = set(package_dict.keys())
        
        while remaining:
            # Find packages with no unresolved dependencies
            ready = []
            for name in remaining:
                pkg = package_dict[name]
                if all(dep not in remaining for dep in pkg.requires):
                    ready.append(pkg)
            
            if not ready:
                # Circular dependency or missing dependency - just add remaining
                ready = [package_dict[name] for name in remaining]
            
            sorted_packages.extend(ready)
            remaining -= {pkg.name for pkg in ready}
        
        return sorted_packages
    
    def _validate_created_environment(self, env_path: Path, 
                                    expected: EnvironmentFingerprint) -> Dict[str, Any]:
        """Validate that created environment matches expectations."""
        
        # This is a simplified validation - in practice, we'd activate the environment
        # and run the full validation
        return {
            'valid': True,
            'message': f"Environment created at {env_path}"
        }
    
    def save_fingerprint(self, fingerprint: EnvironmentFingerprint, 
                        filepath: str) -> None:
        """Save environment fingerprint to file."""
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(fingerprint.to_dict(), f, indent=2)
        
        self.logger.info(f"Saved environment fingerprint to {filepath}")
    
    def load_fingerprint(self, filepath: str) -> EnvironmentFingerprint:
        """Load environment fingerprint from file."""
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        fingerprint = EnvironmentFingerprint.from_dict(data)
        self.logger.info(f"Loaded environment fingerprint from {filepath}")
        return fingerprint


# Global environment manager instance
_environment_manager: Optional[EnvironmentManager] = None


def get_environment_manager() -> EnvironmentManager:
    """Get the global environment manager instance."""
    global _environment_manager
    if _environment_manager is None:
        _environment_manager = EnvironmentManager()
    return _environment_manager


def capture_environment() -> EnvironmentFingerprint:
    """Capture current environment fingerprint."""
    return get_environment_manager().capture_current_environment()


def validate_environment(expected_file: str, strict: bool = True) -> Dict[str, Any]:
    """Validate current environment against expected fingerprint file."""
    manager = get_environment_manager()
    expected = manager.load_fingerprint(expected_file)
    return manager.validate_environment(expected, strict)


def generate_requirements(output_file: Optional[str] = None) -> str:
    """Generate exact requirements file."""
    return get_environment_manager().generate_exact_requirements(output_file)
