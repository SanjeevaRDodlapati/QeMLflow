"""
API Compatibility Testing Framework

This module provides comprehensive API compatibility testing including:
- API signature analysis and comparison
- Breaking change detection
- Compatibility matrix generation
- Regression testing for API changes
"""

import ast
import inspect
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, cast

from .versioning import SemanticVersion


def _ast_unparse(node: ast.AST) -> str:
    """Compatibility wrapper for ast.unparse (Python 3.9+) or fallback."""
    if hasattr(ast, 'unparse'):
        return cast(str, ast.unparse(node))
    else:
        # Simple fallback - just use the class name and basic info
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            return repr(node.value)
        elif isinstance(node, ast.Attribute):
            return f"{_ast_unparse(node.value)}.{node.attr}"
        else:
            return node.__class__.__name__


@dataclass
class APISignature:
    """Represents the signature of an API element (function, class, method)."""
    
    name: str
    type: str  # 'function', 'class', 'method', 'property'
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    return_type: Optional[str] = None
    docstring: Optional[str] = None
    decorators: List[str] = field(default_factory=list)
    is_public: bool = True
    module_path: str = ""
    
    def __post_init__(self):
        """Validate and normalize signature data."""
        self.is_public = not self.name.startswith('_') or self.name.startswith('__') and self.name.endswith('__')
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert signature to dictionary for serialization."""
        return {
            'name': self.name,
            'type': self.type,
            'parameters': self.parameters,
            'return_type': self.return_type,
            'docstring': self.docstring,
            'decorators': self.decorators,
            'is_public': self.is_public,
            'module_path': self.module_path
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'APISignature':
        """Create signature from dictionary."""
        return cls(**data)


@dataclass
class APIChange:
    """Represents a change in the API between versions."""
    
    change_type: str  # 'added', 'removed', 'modified', 'deprecated'
    element_type: str  # 'function', 'class', 'method', 'parameter'
    element_name: str
    old_signature: Optional[APISignature] = None
    new_signature: Optional[APISignature] = None
    breaking: bool = False
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert change to dictionary for serialization."""
        return {
            'change_type': self.change_type,
            'element_type': self.element_type,
            'element_name': self.element_name,
            'old_signature': self.old_signature.to_dict() if self.old_signature else None,
            'new_signature': self.new_signature.to_dict() if self.new_signature else None,
            'breaking': self.breaking,
            'description': self.description
        }


class APIAnalyzer:
    """Analyzes Python modules to extract API signatures."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_module(self, module_path: str, module_name: str = "") -> Dict[str, APISignature]:
        """Analyze a Python module and extract API signatures."""
        signatures = {}
        
        try:
            # Read and parse the module
            with open(module_path, 'r', encoding='utf-8') as f:
                source = f.read()
            
            tree = ast.parse(source)
            
            # Extract signatures from AST
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    sig = self._extract_function_signature(node, module_name)
                    signatures[sig.name] = sig
                elif isinstance(node, ast.ClassDef):
                    sig = self._extract_class_signature(node, module_name)
                    signatures[sig.name] = sig
                    
                    # Extract methods from class
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            method_sig = self._extract_function_signature(
                                item, module_name, class_name=node.name
                            )
                            signatures[f"{node.name}.{item.name}"] = method_sig
        
        except Exception as e:
            self.logger.error(f"Failed to analyze module {module_path}: {e}")
        
        return signatures
    
    def _extract_function_signature(self, node: ast.FunctionDef, 
                                   module_name: str, class_name: str = "") -> APISignature:
        """Extract function signature from AST node."""
        parameters = []
        
        # Extract parameters
        for arg in node.args.args:
            param_info = {
                'name': arg.arg,
                'annotation': _ast_unparse(arg.annotation) if arg.annotation else None,
                'default': None
            }
            parameters.append(param_info)
        
        # Handle defaults
        if node.args.defaults:
            defaults_start = len(parameters) - len(node.args.defaults)
            for i, default in enumerate(node.args.defaults):
                parameters[defaults_start + i]['default'] = _ast_unparse(default)
        
        # Extract return type annotation
        return_type = _ast_unparse(node.returns) if node.returns else None
        
        # Extract docstring
        docstring = ast.get_docstring(node)
        
        # Extract decorators
        decorators = [_ast_unparse(dec) for dec in node.decorator_list]
        
        element_type = "method" if class_name else "function"
        full_name = f"{class_name}.{node.name}" if class_name else node.name
        
        return APISignature(
            name=full_name,
            type=element_type,
            parameters=parameters,
            return_type=return_type,
            docstring=docstring,
            decorators=decorators,
            module_path=module_name
        )
    
    def _extract_class_signature(self, node: ast.ClassDef, module_name: str) -> APISignature:
        """Extract class signature from AST node."""
        # Extract base classes
        bases = [_ast_unparse(base) for base in node.bases]
        
        # Extract docstring
        docstring = ast.get_docstring(node)
        
        # Extract decorators
        decorators = [_ast_unparse(dec) for dec in node.decorator_list]
        
        return APISignature(
            name=node.name,
            type="class",
            parameters=[{'name': 'bases', 'value': bases}],
            docstring=docstring,
            decorators=decorators,
            module_path=module_name
        )
    
    def analyze_live_module(self, module: Any) -> Dict[str, APISignature]:
        """Analyze a live Python module using introspection."""
        signatures = {}
        
        try:
            for name in dir(module):
                if name.startswith('_') and not (name.startswith('__') and name.endswith('__')):
                    continue  # Skip private members
                
                obj = getattr(module, name)
                
                if inspect.isfunction(obj):
                    sig = self._extract_live_function_signature(obj, name, module.__name__)
                    signatures[name] = sig
                elif inspect.isclass(obj):
                    sig = self._extract_live_class_signature(obj, name, module.__name__)
                    signatures[name] = sig
                    
                    # Extract public methods
                    for method_name in dir(obj):
                        if not method_name.startswith('_') or (
                            method_name.startswith('__') and method_name.endswith('__')
                        ):
                            try:
                                method_obj = getattr(obj, method_name)
                                if inspect.ismethod(method_obj) or inspect.isfunction(method_obj):
                                    method_sig = self._extract_live_function_signature(
                                        method_obj, f"{name}.{method_name}", module.__name__
                                    )
                                    signatures[f"{name}.{method_name}"] = method_sig
                            except Exception:
                                continue  # Skip problematic methods
        
        except Exception as e:
            self.logger.error(f"Failed to analyze live module {module}: {e}")
        
        return signatures
    
    def _extract_live_function_signature(self, func: Callable, name: str, 
                                       module_name: str) -> APISignature:
        """Extract function signature from live function object."""
        try:
            sig = inspect.signature(func)
            parameters = []
            
            for param_name, param in sig.parameters.items():
                param_info = {
                    'name': param_name,
                    'annotation': str(param.annotation) if param.annotation != inspect.Parameter.empty else None,
                    'default': str(param.default) if param.default != inspect.Parameter.empty else None,
                    'kind': param.kind.name
                }
                parameters.append(param_info)
            
            return_type = str(sig.return_annotation) if sig.return_annotation != inspect.Signature.empty else None
            
            return APISignature(
                name=name,
                type="function",
                parameters=parameters,
                return_type=return_type,
                docstring=inspect.getdoc(func),
                module_path=module_name
            )
        
        except Exception as e:
            self.logger.warning(f"Failed to extract signature for {name}: {e}")
            return APISignature(name=name, type="function", module_path=module_name)
    
    def _extract_live_class_signature(self, cls: type, name: str, 
                                    module_name: str) -> APISignature:
        """Extract class signature from live class object."""
        try:
            bases = [base.__name__ for base in cls.__bases__ if base is not object]
            
            return APISignature(
                name=name,
                type="class",
                parameters=[{'name': 'bases', 'value': bases}],
                docstring=inspect.getdoc(cls),
                module_path=module_name
            )
        
        except Exception as e:
            self.logger.warning(f"Failed to extract class signature for {name}: {e}")
            return APISignature(name=name, type="class", module_path=module_name)


class APICompatibilityChecker:
    """Checks compatibility between different API versions."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def compare_apis(self, old_api: Dict[str, APISignature], 
                    new_api: Dict[str, APISignature]) -> List[APIChange]:
        """Compare two API versions and identify changes."""
        changes = []
        
        # Find removed elements
        for name, old_sig in old_api.items():
            if name not in new_api:
                changes.append(APIChange(
                    change_type="removed",
                    element_type=old_sig.type,
                    element_name=name,
                    old_signature=old_sig,
                    breaking=old_sig.is_public,
                    description=f"Removed {old_sig.type} '{name}'"
                ))
        
        # Find added elements
        for name, new_sig in new_api.items():
            if name not in old_api:
                changes.append(APIChange(
                    change_type="added",
                    element_type=new_sig.type,
                    element_name=name,
                    new_signature=new_sig,
                    breaking=False,
                    description=f"Added {new_sig.type} '{name}'"
                ))
        
        # Find modified elements
        for name in old_api:
            if name in new_api:
                old_sig = old_api[name]
                new_sig = new_api[name]
                
                if self._signatures_differ(old_sig, new_sig):
                    is_breaking = self._is_breaking_change(old_sig, new_sig)
                    changes.append(APIChange(
                        change_type="modified",
                        element_type=old_sig.type,
                        element_name=name,
                        old_signature=old_sig,
                        new_signature=new_sig,
                        breaking=is_breaking,
                        description=f"Modified {old_sig.type} '{name}'"
                    ))
        
        return changes
    
    def _signatures_differ(self, old_sig: APISignature, new_sig: APISignature) -> bool:
        """Check if two signatures are different."""
        # Compare key signature elements
        return (
            old_sig.parameters != new_sig.parameters or
            old_sig.return_type != new_sig.return_type or
            old_sig.decorators != new_sig.decorators
        )
    
    def _is_breaking_change(self, old_sig: APISignature, new_sig: APISignature) -> bool:
        """Determine if a signature change is breaking."""
        if not old_sig.is_public:
            return False  # Private API changes are not breaking
        
        # Check parameter changes
        old_params = {p['name']: p for p in old_sig.parameters}
        new_params = {p['name']: p for p in new_sig.parameters}
        
        # Removed parameters are breaking
        for param_name in old_params:
            if param_name not in new_params:
                return True
        
        # Changed parameter types are breaking
        for param_name in old_params:
            if param_name in new_params:
                old_param = old_params[param_name]
                new_param = new_params[param_name]
                
                if old_param.get('annotation') != new_param.get('annotation'):
                    return True
                
                # Removed default values are breaking
                if old_param.get('default') and not new_param.get('default'):
                    return True
        
        # Return type changes are potentially breaking
        if old_sig.return_type != new_sig.return_type:
            return True
        
        return False
    
    def generate_compatibility_report(self, changes: List[APIChange]) -> Dict[str, Any]:
        """Generate a comprehensive compatibility report."""
        breaking_changes = [c for c in changes if c.breaking]
        non_breaking_changes = [c for c in changes if not c.breaking]
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_changes": len(changes),
            "breaking_changes": len(breaking_changes),
            "non_breaking_changes": len(non_breaking_changes),
            "compatibility_level": self._determine_compatibility_level(changes),
            "changes": [change.to_dict() for change in changes],
            "summary": {
                "added": len([c for c in changes if c.change_type == "added"]),
                "removed": len([c for c in changes if c.change_type == "removed"]),
                "modified": len([c for c in changes if c.change_type == "modified"]),
                "deprecated": len([c for c in changes if c.change_type == "deprecated"])
            }
        }
        
        return report
    
    def _determine_compatibility_level(self, changes: List[APIChange]) -> str:
        """Determine overall compatibility level from changes."""
        breaking_changes = [c for c in changes if c.breaking]
        
        if breaking_changes:
            return "MAJOR"
        elif any(c.change_type == "added" for c in changes):
            return "MINOR"
        elif changes:
            return "PATCH"
        else:
            return "IDENTICAL"


class APISnapshot:
    """Manages API snapshots for compatibility testing."""
    
    def __init__(self, snapshot_dir: str = "api_snapshots"):
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.analyzer = APIAnalyzer()
    
    def create_snapshot(self, version: Union[str, SemanticVersion], 
                       modules: List[str], description: str = "") -> str:
        """Create an API snapshot for the given version."""
        if isinstance(version, SemanticVersion):
            version = str(version)
        
        snapshot_data = {
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "description": description,
            "modules": {},
            "metadata": {
                "total_signatures": 0,
                "public_signatures": 0
            }
        }  # type: Dict[str, Any]
        
        total_signatures = 0
        public_signatures = 0
        
        for module_path in modules:
            module_name = Path(module_path).stem
            
            try:
                signatures = self.analyzer.analyze_module(module_path, module_name)
                snapshot_data["modules"][module_name] = {
                    name: sig.to_dict() for name, sig in signatures.items()
                }
                
                total_signatures += len(signatures)
                public_signatures += sum(1 for sig in signatures.values() if sig.is_public)
                
            except Exception as e:
                self.logger.error(f"Failed to analyze module {module_path}: {e}")
        
        snapshot_data["metadata"]["total_signatures"] = total_signatures
        snapshot_data["metadata"]["public_signatures"] = public_signatures
        
        # Save snapshot
        snapshot_file = self.snapshot_dir / f"api_{version.replace('.', '_')}.json"
        with open(snapshot_file, 'w', encoding='utf-8') as f:
            json.dump(snapshot_data, f, indent=2)
        
        self.logger.info(f"Created API snapshot for version {version} with {total_signatures} signatures")
        return str(snapshot_file)
    
    def load_snapshot(self, version: Union[str, SemanticVersion]) -> Optional[Dict[str, Any]]:
        """Load an API snapshot for the given version."""
        if isinstance(version, SemanticVersion):
            version = str(version)
        
        snapshot_file = self.snapshot_dir / f"api_{version.replace('.', '_')}.json"
        
        if not snapshot_file.exists():
            self.logger.warning(f"Snapshot not found for version {version}")
            return None
        
        try:
            with open(snapshot_file, 'r', encoding='utf-8') as f:
                return cast(Dict[str, Any], json.load(f))
        except Exception as e:
            self.logger.error(f"Failed to load snapshot for version {version}: {e}")
            return None
    
    def compare_snapshots(self, old_version: Union[str, SemanticVersion],
                         new_version: Union[str, SemanticVersion]) -> Optional[Dict[str, Any]]:
        """Compare two API snapshots and generate compatibility report."""
        old_snapshot = self.load_snapshot(old_version)
        new_snapshot = self.load_snapshot(new_version)
        
        if not old_snapshot or not new_snapshot:
            return None
        
        checker = APICompatibilityChecker()
        all_changes = []
        
        # Compare each module
        for module_name in set(old_snapshot["modules"].keys()) | set(new_snapshot["modules"].keys()):
            old_module = old_snapshot["modules"].get(module_name, {})
            new_module = new_snapshot["modules"].get(module_name, {})
            
            # Convert dict data back to APISignature objects
            old_signatures = {
                name: APISignature.from_dict(data) 
                for name, data in old_module.items()
            }
            new_signatures = {
                name: APISignature.from_dict(data) 
                for name, data in new_module.items()
            }
            
            module_changes = checker.compare_apis(old_signatures, new_signatures)
            all_changes.extend(module_changes)
        
        return checker.generate_compatibility_report(all_changes)
    
    def list_snapshots(self) -> List[Dict[str, Any]]:
        """List all available API snapshots."""
        snapshots = []
        
        for snapshot_file in self.snapshot_dir.glob("api_*.json"):
            try:
                with open(snapshot_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    snapshots.append({
                        "version": data["version"],
                        "timestamp": data["timestamp"],
                        "description": data.get("description", ""),
                        "file": str(snapshot_file),
                        "metadata": data.get("metadata", {})
                    })
            except Exception as e:
                self.logger.warning(f"Failed to read snapshot {snapshot_file}: {e}")
        
        return sorted(snapshots, key=lambda x: x["version"])


# Global API snapshot manager
_api_snapshot: Optional[APISnapshot] = None


def get_api_snapshot() -> APISnapshot:
    """Get the global API snapshot manager."""
    global _api_snapshot
    if _api_snapshot is None:
        _api_snapshot = APISnapshot()
    return _api_snapshot
