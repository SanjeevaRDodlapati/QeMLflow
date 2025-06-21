"""
Deterministic Package Installation Module

This module provides deterministic package installation capabilities to ensure
reproducible environments with exact dependency resolution and installation order.
"""

import hashlib
import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .environment import PackageInfo, EnvironmentFingerprint


@dataclass
class InstallationPlan:
    """Plan for deterministic package installation."""
    
    packages: List[PackageInfo]
    installation_order: List[str]
    constraints: Dict[str, str] = field(default_factory=dict)
    pip_options: List[str] = field(default_factory=list)
    environment_variables: Dict[str, str] = field(default_factory=dict)
    platform_requirements: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'packages': [pkg.to_dict() for pkg in self.packages],
            'installation_order': self.installation_order,
            'constraints': self.constraints,
            'pip_options': self.pip_options,
            'environment_variables': self.environment_variables,
            'platform_requirements': self.platform_requirements
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InstallationPlan':
        """Create from dictionary."""
        packages = [PackageInfo.from_dict(pkg_data) for pkg_data in data.get('packages', [])]
        
        return cls(
            packages=packages,
            installation_order=data.get('installation_order', []),
            constraints=data.get('constraints', {}),
            pip_options=data.get('pip_options', []),
            environment_variables=data.get('environment_variables', {}),
            platform_requirements=data.get('platform_requirements', {})
        )


class DeterministicInstaller:
    """
    Deterministic package installer for reproducible environments.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / '.qemlflow' / 'pip_cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Default pip options for deterministic installation
        self.default_pip_options = [
            '--no-deps',  # Install only the package, not dependencies
            '--force-reinstall',  # Always reinstall to ensure clean state
            '--no-binary=:all:',  # Prefer source distributions for reproducibility
            '--compile',  # Compile .py files to .pyc
            '--cache-dir', str(self.cache_dir)
        ]
    
    def create_installation_plan(self, fingerprint: EnvironmentFingerprint,
                               include_dev: bool = False) -> InstallationPlan:
        """Create deterministic installation plan from environment fingerprint."""
        
        # Filter packages based on inclusion criteria
        packages_to_install = [
            pkg for pkg in fingerprint.packages
            if self._should_include_package(pkg, include_dev)
        ]
        
        # Calculate installation order using topological sort
        installation_order = self._calculate_installation_order(packages_to_install)
        
        # Generate constraints for exact versions
        constraints = {pkg.name: f"=={pkg.version}" for pkg in packages_to_install}
        
        # Set up environment variables for reproducible builds
        env_vars = self._get_build_environment_variables(fingerprint)
        
        # Platform-specific requirements
        platform_reqs = {
            'python_version': fingerprint.python_version,
            'platform': fingerprint.platform_info.get('system', 'Unknown'),
            'architecture': fingerprint.platform_info.get('architecture', 'Unknown')
        }
        
        return InstallationPlan(
            packages=packages_to_install,
            installation_order=installation_order,
            constraints=constraints,
            pip_options=self.default_pip_options.copy(),
            environment_variables=env_vars,
            platform_requirements=platform_reqs
        )
    
    def _should_include_package(self, package: PackageInfo, include_dev: bool) -> bool:
        """Determine if package should be included in installation."""
        
        # Always exclude these packages
        exclude_always = {
            'pip', 'setuptools', 'wheel', 'pkg-resources', 'distribute'
        }
        
        # Development packages to exclude unless specifically requested
        dev_packages = {
            'pytest', 'pytest-cov', 'coverage', 'flake8', 'black', 'mypy',
            'pre-commit', 'tox', 'twine', 'sphinx', 'pylint', 'bandit'
        }
        
        if package.name.lower() in exclude_always:
            return False
        
        if not include_dev and package.name.lower() in dev_packages:
            return False
        
        return True
    
    def _calculate_installation_order(self, packages: List[PackageInfo]) -> List[str]:
        """Calculate deterministic installation order using topological sort."""
        
        # Create dependency graph
        package_dict = {pkg.name: pkg for pkg in packages}
        dependencies = {}
        
        for pkg in packages:
            dependencies[pkg.name] = [
                dep for dep in pkg.requires 
                if dep in package_dict
            ]
        
        # Topological sort with deterministic ordering for stability
        ordered = []
        remaining = set(package_dict.keys())
        
        while remaining:
            # Find packages with no unresolved dependencies
            ready = []
            for name in sorted(remaining):  # Sort for deterministic order
                if all(dep not in remaining for dep in dependencies.get(name, [])):
                    ready.append(name)
            
            if not ready:
                # Handle circular dependencies by selecting alphabetically first
                ready = [min(remaining)]
                self.logger.warning(f"Circular dependency detected, forcing installation of {ready[0]}")
            
            # Add ready packages in sorted order for determinism
            for name in sorted(ready):
                ordered.append(name)
                remaining.remove(name)
        
        return ordered
    
    def _get_build_environment_variables(self, fingerprint: EnvironmentFingerprint) -> Dict[str, str]:
        """Get environment variables for reproducible builds."""
        
        build_env = {
            # Ensure reproducible builds
            'PYTHONHASHSEED': '0',
            'PYTHONDONTWRITEBYTECODE': '1',
            'PIP_NO_CACHE_DIR': 'false',  # We want to use our cache
            'PIP_CACHE_DIR': str(self.cache_dir),
            
            # Compiler flags for reproducible compilation
            'CFLAGS': '-O2 -g -pipe',
            'CPPFLAGS': '-O2 -g -pipe',
            
            # Ensure consistent locale
            'LC_ALL': 'C.UTF-8',
            'LANG': 'C.UTF-8'
        }
        
        # Platform-specific settings
        if fingerprint.platform_info['system'] == 'Darwin':  # macOS
            build_env['MACOSX_DEPLOYMENT_TARGET'] = '10.9'
        
        return build_env
    
    def install_from_plan(self, plan: InstallationPlan, 
                         target_env: Optional[str] = None) -> Dict[str, Any]:
        """Install packages according to deterministic plan."""
        
        installation_log = {
            'started_at': None,
            'completed_at': None,
            'success': False,
            'installed_packages': [],
            'failed_packages': [],
            'errors': []
        }
        
        try:
            from datetime import datetime
            installation_log['started_at'] = datetime.now().isoformat()
            
            # Set up environment
            install_env = os.environ.copy()
            install_env.update(plan.environment_variables)
            
            # Determine pip command
            pip_cmd = self._get_pip_command(target_env)
            
            # Create constraints file
            constraints_file = self._create_constraints_file(plan.constraints)
            
            try:
                # Install packages in order
                for package_name in plan.installation_order:
                    package_info = next(
                        pkg for pkg in plan.packages 
                        if pkg.name == package_name
                    )
                    
                    success = self._install_single_package(
                        package_info, pip_cmd, constraints_file, 
                        plan.pip_options, install_env
                    )
                    
                    if success:
                        installation_log['installed_packages'].append({
                            'name': package_info.name,
                            'version': package_info.version
                        })
                    else:
                        installation_log['failed_packages'].append({
                            'name': package_info.name,
                            'version': package_info.version
                        })
                        self.logger.error(f"Failed to install {package_info.name}")
                
                installation_log['success'] = len(installation_log['failed_packages']) == 0
                
            finally:
                # Clean up constraints file
                if constraints_file.exists():
                    constraints_file.unlink()
            
            installation_log['completed_at'] = datetime.now().isoformat()
            
        except Exception as e:
            installation_log['errors'].append(str(e))
            self.logger.error(f"Installation failed: {e}")
        
        return installation_log
    
    def _get_pip_command(self, target_env: Optional[str]) -> List[str]:
        """Get pip command for target environment."""
        
        if target_env:
            # Use pip from specific environment
            env_path = Path(target_env)
            if os.name == 'nt':  # Windows
                pip_cmd = [str(env_path / 'Scripts' / 'pip.exe')]
            else:
                pip_cmd = [str(env_path / 'bin' / 'pip')]
        else:
            # Use current environment pip
            pip_cmd = ['pip']
        
        return pip_cmd
    
    def _create_constraints_file(self, constraints: Dict[str, str]) -> Path:
        """Create temporary constraints file."""
        
        constraints_content = '\n'.join(
            f"{name}{version}" for name, version in sorted(constraints.items())
        )
        
        temp_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.txt', delete=False, encoding='utf-8'
        )
        temp_file.write(constraints_content)
        temp_file.close()
        
        return Path(temp_file.name)
    
    def _install_single_package(self, package: PackageInfo, pip_cmd: List[str],
                               constraints_file: Path, pip_options: List[str],
                               env: Dict[str, str]) -> bool:
        """Install a single package with constraints."""
        
        cmd = pip_cmd + ['install'] + pip_options + [
            '--constraint', str(constraints_file),
            f"{package.name}=={package.version}"
        ]
        
        try:
            result = subprocess.run(
                cmd, 
                env=env,
                capture_output=True, 
                text=True, 
                timeout=300  # 5 minute timeout per package
            )
            
            if result.returncode == 0:
                self.logger.info(f"Successfully installed {package.name}=={package.version}")
                return True
            else:
                self.logger.error(f"Failed to install {package.name}: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"Timeout installing {package.name}")
            return False
        except Exception as e:
            self.logger.error(f"Error installing {package.name}: {e}")
            return False
    
    def create_lockfile(self, plan: InstallationPlan, output_file: str) -> None:
        """Create a lockfile from installation plan."""
        
        lockfile_data = {
            'version': '1.0.0',
            'generated_at': None,
            'platform_requirements': plan.platform_requirements,
            'packages': {},
            'installation_order': plan.installation_order,
            'checksum': None
        }
        
        from datetime import datetime
        lockfile_data['generated_at'] = datetime.now().isoformat()
        
        # Add package information
        for pkg in plan.packages:
            lockfile_data['packages'][pkg.name] = {
                'version': pkg.version,
                'requires': pkg.requires,
                'installer': pkg.installer,
                'metadata': pkg.metadata
            }
        
        # Calculate checksum
        checksum_data = json.dumps(
            {k: v for k, v in lockfile_data.items() if k != 'checksum'},
            sort_keys=True, separators=(',', ':')
        )
        lockfile_data['checksum'] = hashlib.sha256(checksum_data.encode()).hexdigest()
        
        # Write lockfile
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(lockfile_data, f, indent=2, sort_keys=True)
        
        self.logger.info(f"Created lockfile: {output_file}")
    
    def install_from_lockfile(self, lockfile_path: str, 
                             target_env: Optional[str] = None) -> Dict[str, Any]:
        """Install packages from lockfile."""
        
        with open(lockfile_path, 'r', encoding='utf-8') as f:
            lockfile_data = json.load(f)
        
        # Verify lockfile integrity
        if not self._verify_lockfile_integrity(lockfile_data):
            raise ValueError("Lockfile integrity check failed")
        
        # Convert lockfile to installation plan
        packages = []
        for name, info in lockfile_data['packages'].items():
            package = PackageInfo(
                name=name,
                version=info['version'],
                location='',  # Will be determined during installation
                requires=info.get('requires', []),
                installer=info.get('installer', 'pip'),
                metadata=info.get('metadata', {})
            )
            packages.append(package)
        
        plan = InstallationPlan(
            packages=packages,
            installation_order=lockfile_data['installation_order'],
            pip_options=self.default_pip_options.copy()
        )
        
        return self.install_from_plan(plan, target_env)
    
    def _verify_lockfile_integrity(self, lockfile_data: Dict[str, Any]) -> bool:
        """Verify lockfile hasn't been tampered with."""
        
        expected_checksum = lockfile_data.get('checksum')
        if not expected_checksum:
            return False
        
        # Recalculate checksum
        checksum_data = json.dumps(
            {k: v for k, v in lockfile_data.items() if k != 'checksum'},
            sort_keys=True, separators=(',', ':')
        )
        actual_checksum = hashlib.sha256(checksum_data.encode()).hexdigest()
        
        return expected_checksum == actual_checksum


# Global deterministic installer instance
_deterministic_installer: Optional[DeterministicInstaller] = None


def get_deterministic_installer() -> DeterministicInstaller:
    """Get the global deterministic installer instance."""
    global _deterministic_installer
    if _deterministic_installer is None:
        _deterministic_installer = DeterministicInstaller()
    return _deterministic_installer


def create_installation_plan(fingerprint: EnvironmentFingerprint,
                           include_dev: bool = False) -> InstallationPlan:
    """Create installation plan from environment fingerprint."""
    return get_deterministic_installer().create_installation_plan(fingerprint, include_dev)


def install_deterministically(plan: InstallationPlan, 
                             target_env: Optional[str] = None) -> Dict[str, Any]:
    """Install packages deterministically according to plan."""
    return get_deterministic_installer().install_from_plan(plan, target_env)


def create_lockfile(fingerprint: EnvironmentFingerprint, output_file: str,
                   include_dev: bool = False) -> None:
    """Create lockfile from environment fingerprint."""
    installer = get_deterministic_installer()
    plan = installer.create_installation_plan(fingerprint, include_dev)
    installer.create_lockfile(plan, output_file)


def install_from_lockfile(lockfile_path: str, 
                         target_env: Optional[str] = None) -> Dict[str, Any]:
    """Install from lockfile."""
    return get_deterministic_installer().install_from_lockfile(lockfile_path, target_env)
