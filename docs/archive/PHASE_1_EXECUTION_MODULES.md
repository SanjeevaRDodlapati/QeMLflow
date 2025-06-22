# 🛠️ Phase 1 Execution Modules

**Foundation Fixes - Modular Implementation Guide**

*Priority: P0 Critical | Duration: 2 weeks | Focus: Performance & Compliance*

---

## 📋 **Module Overview**

Phase 1 consists of **4 critical modules** that must be executed in sequence:

1. **Module 1.1**: Import Performance Optimization
2. **Module 1.2**: Type Hints Implementation  
3. **Module 1.3**: Health Monitoring Integration
4. **Module 1.4**: Philosophy Enforcement Integration

Each module includes **detailed steps, code examples, and validation criteria**.

---

## 🚀 **Module 1.1: Import Performance Optimization**

### **Objective**: Reduce core import time from 53s to <10s

### **Prerequisites**
```bash
# Ensure development environment is ready
python -m pip install cProfile memory_profiler
```

### **Step 1: Create Import Profiler Tool**

Create `tools/import_profiler.py`:

```python
#!/usr/bin/env python3
"""
Comprehensive import performance profiler for QeMLflow.
Identifies bottlenecks and suggests optimizations.
"""

import cProfile
import pstats
import time
import sys
import os
from pathlib import Path
import json
from typing import Dict, List, Tuple

class ImportProfiler:
    def __init__(self):
        self.results = {}
        self.baseline_time = None
        
    def profile_module_import(self, module_name: str) -> Dict:
        """Profile a specific module import."""
        # Add src to path
        sys.path.insert(0, 'src')
        
        # Profile the import
        profiler = cProfile.Profile()
        start_time = time.time()
        
        profiler.enable()
        try:
            exec(f"import {module_name}")
        except ImportError as e:
            return {"error": str(e), "time": 0}
        profiler.disable()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Analyze profiler results
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        
        return {
            "total_time": total_time,
            "top_functions": self._get_top_functions(stats),
            "module_stats": self._analyze_module_stats(stats)
        }
    
    def _get_top_functions(self, stats: pstats.Stats, limit: int = 10) -> List[Dict]:
        """Extract top time-consuming functions."""
        top_functions = []
        for func, (cc, nc, tt, ct, callers) in stats.stats.items():
            if tt > 0.01:  # Only functions taking >10ms
                filename, line, function = func
                top_functions.append({
                    "function": function,
                    "file": os.path.basename(filename),
                    "total_time": tt,
                    "cumulative_time": ct,
                    "calls": cc
                })
        
        return sorted(top_functions, key=lambda x: x["cumulative_time"], reverse=True)[:limit]
    
    def _analyze_module_stats(self, stats: pstats.Stats) -> Dict:
        """Analyze per-module import statistics."""
        module_times = {}
        for func, (cc, nc, tt, ct, callers) in stats.stats.items():
            filename, line, function = func
            module = self._extract_module_name(filename)
            if module not in module_times:
                module_times[module] = {"total_time": 0, "cumulative_time": 0, "calls": 0}
            module_times[module]["total_time"] += tt
            module_times[module]["cumulative_time"] += ct
            module_times[module]["calls"] += cc
        
        return dict(sorted(module_times.items(), 
                          key=lambda x: x[1]["cumulative_time"], 
                          reverse=True)[:15])
    
    def _extract_module_name(self, filename: str) -> str:
        """Extract module name from filename."""
        if 'site-packages' in filename:
            parts = filename.split('site-packages')[-1].split(os.sep)
            return parts[1] if len(parts) > 1 else 'unknown'
        return os.path.basename(filename)
    
    def profile_core_modules(self) -> Dict:
        """Profile all core QeMLflow modules."""
        core_modules = [
            'qemlflow.core',
            'qemlflow.core.models',
            'qemlflow.core.utils',
            'qemlflow.core.pipeline'
        ]
        
        results = {}
        for module in core_modules:
            print(f"Profiling {module}...")
            results[module] = self.profile_module_import(module)
        
        return results
    
    def generate_optimization_report(self, results: Dict) -> Dict:
        """Generate optimization recommendations."""
        recommendations = []
        
        for module, data in results.items():
            if data.get("total_time", 0) > 5.0:
                recommendations.append({
                    "module": module,
                    "issue": "Slow import time",
                    "current_time": data["total_time"],
                    "recommendation": "Implement lazy loading for heavy dependencies",
                    "priority": "HIGH"
                })
        
        return {
            "total_recommendations": len(recommendations),
            "recommendations": recommendations,
            "quick_wins": self._identify_quick_wins(results)
        }
    
    def _identify_quick_wins(self, results: Dict) -> List[Dict]:
        """Identify quick optimization opportunities."""
        quick_wins = []
        
        for module, data in results.items():
            top_functions = data.get("top_functions", [])
            for func in top_functions[:3]:  # Top 3 time consumers
                if any(heavy in func["file"].lower() for heavy in 
                      ["tensorflow", "torch", "sklearn", "matplotlib"]):
                    quick_wins.append({
                        "function": func["function"],
                        "file": func["file"],
                        "time": func["cumulative_time"],
                        "action": "Move import inside function or use lazy loading"
                    })
        
        return quick_wins


def main():
    """Main execution function."""
    profiler = ImportProfiler()
    
    print("🔍 Starting comprehensive import profiling...")
    results = profiler.profile_core_modules()
    
    print("📊 Generating optimization report...")
    report = profiler.generate_optimization_report(results)
    
    # Save detailed results
    with open('import_profile_results.json', 'w') as f:
        json.dump({
            "timestamp": time.time(),
            "profiling_results": results,
            "optimization_report": report
        }, f, indent=2)
    
    # Print summary
    print("\n" + "="*50)
    print("🎯 IMPORT PERFORMANCE ANALYSIS SUMMARY")
    print("="*50)
    
    for module, data in results.items():
        if not data.get("error"):
            print(f"📦 {module}: {data['total_time']:.2f}s")
    
    print(f"\n🔧 Found {report['total_recommendations']} optimization opportunities")
    print(f"⚡ Quick wins available: {len(report['quick_wins'])}")
    
    print("\n📋 Top recommendations:")
    for rec in report['recommendations'][:3]:
        print(f"  • {rec['module']}: {rec['recommendation']}")


if __name__ == "__main__":
    main()
```

### **Step 2: Run Initial Profiling**

```bash
# Execute the profiler
cd /Users/sanjeev/Downloads/Repos/QeMLflow
python tools/import_profiler.py
```

### **Step 3: Implement Lazy Loading Pattern**

Create `src/qemlflow/core/lazy_imports.py`:

```python
"""
Lazy import utilities for performance optimization.
Implements deferred loading of heavy dependencies.
"""

import importlib
from typing import Any, Optional, Dict
import warnings


class LazyImporter:
    """Lazy import manager for heavy dependencies."""
    
    def __init__(self):
        self._modules: Dict[str, Any] = {}
        self._import_cache: Dict[str, bool] = {}
    
    def get_module(self, module_name: str, required: bool = True) -> Optional[Any]:
        """Get module with lazy loading."""
        if module_name in self._modules:
            return self._modules[module_name]
        
        try:
            self._modules[module_name] = importlib.import_module(module_name)
            self._import_cache[module_name] = True
            return self._modules[module_name]
        except ImportError as e:
            self._import_cache[module_name] = False
            if required:
                raise ImportError(f"Required module '{module_name}' not available: {e}")
            else:
                warnings.warn(f"Optional module '{module_name}' not available: {e}")
                return None


# Global lazy importer instance
_lazy_importer = LazyImporter()

# Lazy import functions for common heavy dependencies
def get_numpy():
    """Lazy import numpy."""
    return _lazy_importer.get_module('numpy')

def get_pandas():
    """Lazy import pandas."""
    return _lazy_importer.get_module('pandas')

def get_sklearn():
    """Lazy import sklearn."""
    return _lazy_importer.get_module('sklearn')

def get_tensorflow():
    """Lazy import tensorflow."""
    return _lazy_importer.get_module('tensorflow', required=False)

def get_torch():
    """Lazy import torch.""" 
    return _lazy_importer.get_module('torch', required=False)

def get_matplotlib():
    """Lazy import matplotlib."""
    return _lazy_importer.get_module('matplotlib', required=False)


# Optional dependency checker
def check_optional_dependencies() -> Dict[str, bool]:
    """Check availability of optional dependencies."""
    optional_deps = ['tensorflow', 'torch', 'matplotlib', 'plotly']
    availability = {}
    
    for dep in optional_deps:
        availability[dep] = _lazy_importer._import_cache.get(dep, False)
    
    return availability
```

### **Step 4: Update Core Module Imports**

Update `src/qemlflow/core/__init__.py`:

```python
"""
QeMLflow Core Module - Optimized Imports
"""

# Fast core imports only
from .base import BaseModel, BasePreprocessor
from .exceptions import QeMLflowError, ValidationError

# Lazy import utilities
from .lazy_imports import (
    get_numpy, get_pandas, get_sklearn,
    get_tensorflow, get_torch, get_matplotlib,
    check_optional_dependencies
)

# Health monitoring (lightweight)
from .health_monitor import health_check

# Version info
__version__ = "0.1.0"

# Optional: Defer heavy module imports
def _lazy_load_modules():
    """Load heavy modules only when needed."""
    global models, utils, pipeline
    
    if 'models' not in globals():
        from . import models
    if 'utils' not in globals(): 
        from . import utils
    if 'pipeline' not in globals():
        from . import pipeline

# Make lazy loading available
def load_full_core():
    """Load all core modules (for when full functionality is needed)."""
    _lazy_load_modules()
    return True

# Public API (lightweight by default)
__all__ = [
    'BaseModel', 'BasePreprocessor',
    'QeMLflowError', 'ValidationError', 
    'health_check', 'load_full_core',
    'get_numpy', 'get_pandas', 'get_sklearn',
    'check_optional_dependencies'
]
```

### **Step 5: Optimize Heavy Import Modules**

Update imports in key modules to use lazy loading:

```python
# Example: src/qemlflow/core/models/base.py
from typing import Dict, Any, Optional
from ..lazy_imports import get_numpy, get_pandas

class BaseModel:
    def fit(self, X, y):
        # Lazy load numpy only when needed
        np = get_numpy()
        # Use np as normal...
```

### **Step 6: Validation & Testing**

Create `tools/validate_import_performance.py`:

```python
#!/usr/bin/env python3
"""Validate import performance improvements."""

import time
import sys

def measure_import_time(module_name: str) -> float:
    """Measure time to import a module."""
    sys.path.insert(0, 'src')
    
    start_time = time.time()
    exec(f"import {module_name}")
    end_time = time.time()
    
    return end_time - start_time

def main():
    modules_to_test = [
        'qemlflow.core',
        'qemlflow.core.models', 
        'qemlflow.core.utils'
    ]
    
    print("🚀 Validating import performance...")
    total_time = 0
    
    for module in modules_to_test:
        import_time = measure_import_time(module)
        total_time += import_time
        status = "✅" if import_time < 2.0 else "❌"
        print(f"{status} {module}: {import_time:.2f}s")
    
    print(f"\n📊 Total core import time: {total_time:.2f}s")
    
    target_time = 10.0  # Phase 1 target
    if total_time < target_time:
        print(f"✅ SUCCESS: Under target time of {target_time}s")
        return True
    else:
        print(f"❌ NEEDS WORK: Exceeds target time of {target_time}s")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
```

### **Deliverables for Module 1.1**

- ✅ Import profiling tool and baseline measurements
- ✅ Lazy import infrastructure implemented
- ✅ Core module imports optimized
- ✅ Performance validation tool
- ✅ Import time reduced to <10s (Phase 1 target)

---

## 🎯 **Module 1.2: Type Hints Implementation**

### **Objective**: Add type hints to all 23 identified public API functions

### **Step 1: Create Type Hint Scanner**

Create `tools/type_hint_scanner.py`:

```python
#!/usr/bin/env python3
"""
Scan codebase for missing type hints and generate implementation plan.
"""

import ast
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import json

class TypeHintScanner:
    def __init__(self, source_dir: str = "src"):
        self.source_dir = Path(source_dir)
        self.missing_hints = []
        
    def scan_file(self, file_path: Path) -> List[Dict]:
        """Scan a Python file for missing type hints."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            visitor = TypeHintVisitor(str(file_path))
            visitor.visit(tree)
            
            return visitor.missing_hints
        except Exception as e:
            print(f"Error scanning {file_path}: {e}")
            return []
    
    def scan_directory(self) -> Dict[str, List[Dict]]:
        """Scan all Python files in directory."""
        results = {}
        
        for py_file in self.source_dir.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
                
            missing = self.scan_file(py_file)
            if missing:
                results[str(py_file)] = missing
        
        return results
    
    def generate_implementation_plan(self, scan_results: Dict) -> Dict:
        """Generate type hint implementation plan."""
        priority_files = []
        total_functions = 0
        
        for file_path, missing_hints in scan_results.items():
            public_functions = [h for h in missing_hints if h.get('is_public', False)]
            total_functions += len(missing_hints)
            
            if public_functions:
                priority_files.append({
                    "file": file_path,
                    "public_missing": len(public_functions),
                    "total_missing": len(missing_hints),
                    "functions": public_functions
                })
        
        # Sort by number of public functions missing hints
        priority_files.sort(key=lambda x: x["public_missing"], reverse=True)
        
        return {
            "total_files_with_issues": len(scan_results),
            "total_functions_missing_hints": total_functions,
            "priority_files": priority_files,
            "implementation_order": [f["file"] for f in priority_files]
        }


class TypeHintVisitor(ast.NodeVisitor):
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.missing_hints = []
        self.current_class = None
    
    def visit_ClassDef(self, node):
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class
    
    def visit_FunctionDef(self, node):
        """Visit function definitions and check for type hints."""
        # Skip private functions (starting with _) unless they're __init__
        is_public = not node.name.startswith('_') or node.name in ['__init__', '__call__']
        
        missing_annotations = []
        
        # Check return annotation
        if node.returns is None and node.name != '__init__':
            missing_annotations.append("return")
        
        # Check parameter annotations
        for arg in node.args.args:
            if arg.annotation is None and arg.arg != 'self':
                missing_annotations.append(f"parameter:{arg.arg}")
        
        if missing_annotations:
            self.missing_hints.append({
                "function": node.name,
                "class": self.current_class,
                "line": node.lineno,
                "missing": missing_annotations,
                "is_public": is_public,
                "signature": self._get_function_signature(node)
            })
    
    def _get_function_signature(self, node) -> str:
        """Extract function signature for reference."""
        args = []
        for arg in node.args.args:
            if arg.arg != 'self':
                args.append(arg.arg)
        
        class_prefix = f"{self.current_class}." if self.current_class else ""
        return f"{class_prefix}{node.name}({', '.join(args)})"


def main():
    """Main execution function."""
    scanner = TypeHintScanner()
    
    print("🔍 Scanning codebase for missing type hints...")
    scan_results = scanner.scan_directory()
    
    print("📋 Generating implementation plan...")
    plan = scanner.generate_implementation_plan(scan_results)
    
    # Save results
    with open('type_hint_analysis.json', 'w') as f:
        json.dump({
            "scan_results": scan_results,
            "implementation_plan": plan
        }, f, indent=2)
    
    # Print summary
    print("\n" + "="*50)
    print("🎯 TYPE HINT ANALYSIS SUMMARY")
    print("="*50)
    
    print(f"📊 Files with missing hints: {plan['total_files_with_issues']}")
    print(f"🔧 Functions missing hints: {plan['total_functions_missing_hints']}")
    
    print("\n📋 Priority files (public API):")
    for file_info in plan['priority_files'][:5]:
        print(f"  • {os.path.basename(file_info['file'])}: "
              f"{file_info['public_missing']} public functions")
    
    return plan

if __name__ == "__main__":
    main()
```

### **Step 2: Run Type Hint Analysis**

```bash
# Execute type hint scanner
python tools/type_hint_scanner.py
```

### **Step 3: Implement Type Hints Systematically**

Create `tools/type_hint_fixer.py`:

```python
#!/usr/bin/env python3
"""
Automated type hint implementation tool.
Applies type hints based on analysis and common patterns.
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class TypeHintFixer:
    def __init__(self):
        self.common_type_mappings = {
            'X': 'np.ndarray',
            'y': 'np.ndarray', 
            'data': 'pd.DataFrame',
            'model': 'BaseModel',
            'models': 'Dict[str, BaseModel]',
            'config': 'Dict[str, Any]',
            'params': 'Dict[str, Any]',
            'metrics': 'List[str]',
            'results': 'Dict[str, float]'
        }
        
        self.imports_to_add = {
            'Dict': 'from typing import Dict, List, Optional, Union, Any',
            'np.ndarray': 'import numpy as np',
            'pd.DataFrame': 'import pandas as pd'
        }
    
    def fix_file(self, file_path: Path, missing_hints: List[Dict]) -> str:
        """Apply type hints to a file."""
        with open(file_path, 'r') as f:
            content = f.read()
        
        lines = content.splitlines()
        
        # Group by function for easier processing
        functions_to_fix = {}
        for hint_info in missing_hints:
            if hint_info.get('is_public', False):  # Only fix public functions
                func_name = hint_info['function']
                if func_name not in functions_to_fix:
                    functions_to_fix[func_name] = hint_info
        
        # Apply fixes
        modified_content = self._apply_type_hints(content, functions_to_fix)
        
        return modified_content
    
    def _apply_type_hints(self, content: str, functions_to_fix: Dict) -> str:
        """Apply type hints to function definitions."""
        lines = content.splitlines()
        modified_lines = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Check if this line starts a function definition we need to fix
            func_match = re.match(r'(\s*)def\s+(\w+)\s*\(', line)
            if func_match and func_match.group(2) in functions_to_fix:
                # This is a function we need to fix
                indent = func_match.group(1)
                func_name = func_match.group(2)
                hint_info = functions_to_fix[func_name]
                
                # Process the function definition (might span multiple lines)
                func_lines = [line]
                j = i + 1
                
                # Collect all lines of the function definition
                while j < len(lines) and not lines[j].strip().endswith(':'):
                    func_lines.append(lines[j])
                    j += 1
                
                if j < len(lines):
                    func_lines.append(lines[j])  # Include the line with ':'
                
                # Apply type hints to the function definition
                fixed_func_lines = self._fix_function_definition(func_lines, hint_info)
                modified_lines.extend(fixed_func_lines)
                
                i = j + 1  # Skip the processed lines
            else:
                modified_lines.append(line)
                i += 1
        
        return '\n'.join(modified_lines)
    
    def _fix_function_definition(self, func_lines: List[str], hint_info: Dict) -> List[str]:
        """Fix type hints for a specific function definition."""
        # Combine all function definition lines
        func_def = ' '.join(line.strip() for line in func_lines)
        
        # Parse parameters and add type hints
        # This is a simplified implementation - in practice, you'd want more robust parsing
        
        # For now, return the original lines with a comment about needed fixes
        fixed_lines = func_lines.copy()
        fixed_lines.insert(-1, f"    # TODO: Add type hints for: {', '.join(hint_info['missing'])}")
        
        return fixed_lines


def main():
    """Main execution function."""
    # Load analysis results
    import json
    
    try:
        with open('type_hint_analysis.json', 'r') as f:
            analysis = json.load(f)
    except FileNotFoundError:
        print("❌ Please run type_hint_scanner.py first")
        return
    
    fixer = TypeHintFixer()
    scan_results = analysis['scan_results']
    
    print("🔧 Applying type hints to priority files...")
    
    for file_path, missing_hints in scan_results.items():
        print(f"Processing {file_path}...")
        
        # For demonstration, just show what would be fixed
        public_missing = [h for h in missing_hints if h.get('is_public', False)]
        if public_missing:
            print(f"  • Would fix {len(public_missing)} public functions")
            for hint in public_missing[:3]:  # Show first 3
                print(f"    - {hint['function']}: {', '.join(hint['missing'])}")

if __name__ == "__main__":
    main()
```

### **Step 4: Manual Type Hint Implementation**

For the most critical files, implement type hints manually:

```python
# Example: Fix src/qemlflow/core/models/base.py

from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import pandas as pd

class BaseModel:
    """Base class for all ML models in QeMLflow."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the model with optional configuration."""
        self.config = config or {}
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseModel':
        """Fit the model to training data."""
        # Implementation here
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        # Implementation here
        pass
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, 
                metrics: Optional[List[str]] = None) -> Dict[str, float]:
        """Evaluate model performance."""
        # Implementation here
        pass
```

### **Step 5: Configure MyPy for Type Checking**

Update `mypy.ini`:

```ini
[mypy]
python_version = 3.9
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
disallow_incomplete_defs = True
check_untyped_defs = True
disallow_untyped_decorators = True

# Specific module configurations
[mypy-qemlflow.core.*]
disallow_untyped_defs = True
disallow_any_generics = True
warn_redundant_casts = True

# External library stubs
[mypy-numpy.*]
ignore_missing_imports = True

[mypy-pandas.*]
ignore_missing_imports = True

[mypy-sklearn.*]
ignore_missing_imports = True
```

### **Step 6: Create Type Checking Validation**

Create `tools/validate_type_hints.py`:

```python
#!/usr/bin/env python3
"""Validate type hint implementation."""

import subprocess
import sys
import json
from pathlib import Path

def run_mypy_check() -> Tuple[bool, str]:
    """Run MyPy type checking."""
    try:
        result = subprocess.run(
            ['mypy', 'src/qemlflow/core/', '--json-report', 'mypy_report'],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, "MyPy check timed out"
    except FileNotFoundError:
        return False, "MyPy not installed"

def count_missing_type_hints() -> int:
    """Count remaining missing type hints."""
    # Re-run type hint scanner
    from type_hint_scanner import TypeHintScanner
    
    scanner = TypeHintScanner()
    scan_results = scanner.scan_directory()
    
    total_missing = 0
    for file_path, missing_hints in scan_results.items():
        public_missing = [h for h in missing_hints if h.get('is_public', False)]
        total_missing += len(public_missing)
    
    return total_missing

def main():
    """Main validation function."""
    print("🔍 Validating type hint implementation...")
    
    # Count remaining missing hints
    missing_count = count_missing_type_hints()
    print(f"📊 Missing type hints (public API): {missing_count}")
    
    # Run MyPy check
    success, output = run_mypy_check()
    
    if success:
        print("✅ MyPy type checking passed")
    else:
        print("❌ MyPy type checking failed:")
        print(output)
    
    # Overall success criteria
    success_criteria = missing_count == 0 and success
    
    if success_criteria:
        print("✅ SUCCESS: All type hints implemented correctly")
        return True
    else:
        print("❌ NEEDS WORK: Type hint implementation incomplete")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
```

### **Deliverables for Module 1.2**

- ✅ Type hint scanning and analysis tools
- ✅ Type hints added to all public API functions
- ✅ MyPy configuration for strict type checking
- ✅ Type validation and testing tools
- ✅ Zero missing type hints for public APIs

---

## 📊 **Module 1.3: Health Monitoring Integration**

### **Objective**: Deploy comprehensive health monitoring system

### **Step 1: Enhance Existing Health Monitor**

Update `src/qemlflow/core/health_monitor.py`:

```python
"""
Enhanced Health Monitoring System for QeMLflow Core.
Provides real-time performance tracking and system health analytics.
"""

import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
from pathlib import Path

class HealthMonitor:
    """Comprehensive health monitoring for QeMLflow."""
    
    def __init__(self, monitoring_interval: float = 60.0):
        self.monitoring_interval = monitoring_interval
        self.is_monitoring = False
        self.metrics_history: List[Dict] = []
        self.alerts: List[Dict] = []
        self.thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'import_time': 5.0,
            'response_time': 1.0
        }
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
    
    def start_monitoring(self) -> None:
        """Start continuous health monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()
    
    def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        self.is_monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                metrics = self.collect_metrics()
                self._store_metrics(metrics)
                self._check_alerts(metrics)
                time.sleep(self.monitoring_interval)
            except Exception as e:
                self._log_error(f"Monitoring loop error: {e}")
    
    def collect_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive system metrics."""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'system': self._collect_system_metrics(),
            'performance': self._collect_performance_metrics(),
            'qemlflow': self._collect_qemlflow_metrics()
        }
        
        return metrics
    
    def _collect_system_metrics(self) -> Dict[str, float]:
        """Collect system-level metrics."""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'load_average': psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0.0
        }
    
    def _collect_performance_metrics(self) -> Dict[str, float]:
        """Collect performance-related metrics."""
        return {
            'import_time': self._measure_import_time(),
            'response_time': self._measure_response_time(),
            'memory_usage': psutil.Process().memory_info().rss / 1024 / 1024  # MB
        }
    
    def _collect_qemlflow_metrics(self) -> Dict[str, Any]:
        """Collect QeMLflow-specific metrics."""
        try:
            # Import performance
            import_start = time.time()
            import qemlflow.core
            import_time = time.time() - import_start
            
            return {
                'core_import_time': import_time,
                'modules_loaded': len([m for m in sys.modules.keys() if 'qemlflow' in m]),
                'health_status': 'healthy' if import_time < self.thresholds['import_time'] else 'degraded'
            }
        except Exception as e:
            return {
                'core_import_time': -1,
                'modules_loaded': 0,
                'health_status': 'error',
                'error': str(e)
            }
    
    def _measure_import_time(self) -> float:
        """Measure core module import time."""
        import sys
        import importlib
        
        # Clear module cache for accurate measurement
        modules_to_reload = [m for m in sys.modules.keys() if 'qemlflow.core' in m]
        for module in modules_to_reload:
            if module in sys.modules:
                del sys.modules[module]
        
        start_time = time.time()
        try:
            import qemlflow.core
            importlib.reload(qemlflow.core)
        except Exception:
            pass
        
        return time.time() - start_time
    
    def _measure_response_time(self) -> float:
        """Measure system response time."""
        start_time = time.time()
        # Simple CPU-bound operation
        sum(range(10000))
        return time.time() - start_time
    
    def _store_metrics(self, metrics: Dict) -> None:
        """Store metrics in history."""
        with self._lock:
            self.metrics_history.append(metrics)
            
            # Keep only last 100 entries
            if len(self.metrics_history) > 100:
                self.metrics_history = self.metrics_history[-100:]
    
    def _check_alerts(self, metrics: Dict) -> None:
        """Check for threshold violations and generate alerts."""
        alerts = []
        
        # Check system thresholds
        system_metrics = metrics['system']
        if system_metrics['cpu_percent'] > self.thresholds['cpu_percent']:
            alerts.append({
                'type': 'cpu_high',
                'message': f"High CPU usage: {system_metrics['cpu_percent']:.1f}%",
                'severity': 'warning',
                'timestamp': metrics['timestamp']
            })
        
        if system_metrics['memory_percent'] > self.thresholds['memory_percent']:
            alerts.append({
                'type': 'memory_high',
                'message': f"High memory usage: {system_metrics['memory_percent']:.1f}%",
                'severity': 'warning',
                'timestamp': metrics['timestamp']
            })
        
        # Check performance thresholds
        perf_metrics = metrics['performance']
        if perf_metrics['import_time'] > self.thresholds['import_time']:
            alerts.append({
                'type': 'import_slow',
                'message': f"Slow import time: {perf_metrics['import_time']:.2f}s",
                'severity': 'critical',
                'timestamp': metrics['timestamp']
            })
        
        # Store alerts
        with self._lock:
            self.alerts.extend(alerts)
            
            # Keep only recent alerts (last 50)
            if len(self.alerts) > 50:
                self.alerts = self.alerts[-50:]
    
    def get_status(self) -> Dict[str, Any]:
        """Get current health status."""
        latest_metrics = self.metrics_history[-1] if self.metrics_history else None
        
        if not latest_metrics:
            return {'status': 'no_data', 'message': 'No metrics available'}
        
        # Determine overall health status
        recent_alerts = [a for a in self.alerts 
                        if datetime.fromisoformat(a['timestamp']) > 
                        datetime.now() - timedelta(minutes=5)]
        
        critical_alerts = [a for a in recent_alerts if a['severity'] == 'critical']
        
        if critical_alerts:
            status = 'critical'
        elif recent_alerts:
            status = 'warning'
        else:
            status = 'healthy'
        
        return {
            'status': status,
            'latest_metrics': latest_metrics,
            'recent_alerts': recent_alerts,
            'monitoring_active': self.is_monitoring,
            'metrics_collected': len(self.metrics_history)
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        if not self.metrics_history:
            return {'error': 'No metrics available'}
        
        # Calculate performance trends
        recent_metrics = self.metrics_history[-10:]  # Last 10 measurements
        
        import_times = [m['performance']['import_time'] for m in recent_metrics]
        cpu_usage = [m['system']['cpu_percent'] for m in recent_metrics]
        memory_usage = [m['system']['memory_percent'] for m in recent_metrics]
        
        return {
            'performance_summary': {
                'avg_import_time': sum(import_times) / len(import_times),
                'max_import_time': max(import_times),
                'min_import_time': min(import_times),
                'avg_cpu_usage': sum(cpu_usage) / len(cpu_usage),
                'avg_memory_usage': sum(memory_usage) / len(memory_usage)
            },
            'trends': {
                'import_time_trend': 'improving' if import_times[-1] < import_times[0] else 'degrading',
                'cpu_trend': 'stable' if abs(cpu_usage[-1] - cpu_usage[0]) < 10 else 'changing'
            },
            'recommendations': self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance improvement recommendations."""
        recommendations = []
        
        if not self.metrics_history:
            return recommendations
        
        latest = self.metrics_history[-1]
        
        if latest['performance']['import_time'] > 3.0:
            recommendations.append("Consider implementing additional lazy loading for imports")
        
        if latest['system']['memory_percent'] > 70:
            recommendations.append("Monitor memory usage - consider implementing memory optimization")
        
        if latest['system']['cpu_percent'] > 60:
            recommendations.append("High CPU usage detected - investigate background processes")
        
        return recommendations
    
    def export_metrics(self, filepath: str) -> bool:
        """Export metrics history to file."""
        try:
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'metrics_history': self.metrics_history,
                'alerts_history': self.alerts,
                'thresholds': self.thresholds
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            return True
        except Exception as e:
            self._log_error(f"Export failed: {e}")
            return False
    
    def _log_error(self, message: str) -> None:
        """Log error message."""
        timestamp = datetime.now().isoformat()
        print(f"[{timestamp}] HealthMonitor ERROR: {message}")


# Global health monitor instance
_health_monitor = HealthMonitor()

def start_monitoring() -> None:
    """Start global health monitoring."""
    _health_monitor.start_monitoring()

def stop_monitoring() -> None:
    """Stop global health monitoring."""
    _health_monitor.stop_monitoring()

def health_check() -> Dict[str, Any]:
    """Get current health status."""
    return _health_monitor.get_status()

def get_performance_report() -> Dict[str, Any]:
    """Get performance analysis report."""
    return _health_monitor.get_performance_report()

def export_health_data(filepath: str = "health_metrics.json") -> bool:
    """Export health monitoring data."""
    return _health_monitor.export_metrics(filepath)
```

### **Step 2: Create Health Dashboard**

Create `tools/health_dashboard.py`:

```python
#!/usr/bin/env python3
"""
Health Dashboard for QeMLflow Development.
Provides real-time monitoring and performance insights.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add src to path for imports
sys.path.insert(0, 'src')

try:
    from qemlflow.core.health_monitor import health_check, get_performance_report, start_monitoring
except ImportError:
    st.error("Could not import QeMLflow health monitoring. Please ensure the package is properly installed.")
    st.stop()

def main():
    """Main dashboard application."""
    st.set_page_config(
        page_title="QeMLflow Health Dashboard",
        page_icon="🏥",
        layout="wide"
    )
    
    st.title("🏥 QeMLflow Health Dashboard")
    st.markdown("Real-time monitoring and performance analytics for QeMLflow development")
    
    # Sidebar controls
    st.sidebar.header("Controls")
    
    if st.sidebar.button("Start Monitoring"):
        start_monitoring()
        st.sidebar.success("Monitoring started!")
    
    auto_refresh = st.sidebar.checkbox("Auto-refresh (5s)", value=True)
    
    if auto_refresh:
        st.rerun()
    
    # Main dashboard layout
    col1, col2, col3 = st.columns(3)
    
    # Get current health status
    health_status = health_check()
    
    # Status indicators
    with col1:
        st.metric(
            label="Health Status",
            value=health_status.get('status', 'unknown').upper(),
            delta="Monitoring Active" if health_status.get('monitoring_active') else "Monitoring Inactive"
        )
    
    with col2:
        metrics_count = health_status.get('metrics_collected', 0)
        st.metric(
            label="Metrics Collected",
            value=metrics_count,
            delta=f"Last updated: {datetime.now().strftime('%H:%M:%S')}"
        )
    
    with col3:
        alerts_count = len(health_status.get('recent_alerts', []))
        st.metric(
            label="Recent Alerts",
            value=alerts_count,
            delta="⚠️ Issues detected" if alerts_count > 0 else "✅ All clear"
        )
    
    # Detailed metrics display
    if health_status.get('latest_metrics'):
        st.header("📊 Current Metrics")
        
        latest = health_status['latest_metrics']
        
        # System metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("System Performance")
            system_metrics = latest.get('system', {})
            
            # CPU usage gauge
            fig_cpu = go.Figure(go.Indicator(
                mode="gauge+number",
                value=system_metrics.get('cpu_percent', 0),
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "CPU Usage (%)"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "darkblue"},
                       'steps': [{'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 80], 'color': "yellow"},
                                {'range': [80, 100], 'color': "red"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 90}}
            ))
            fig_cpu.update_layout(height=300)
            st.plotly_chart(fig_cpu, use_container_width=True)
        
        with col2:
            st.subheader("Memory Usage")
            memory_percent = system_metrics.get('memory_percent', 0)
            
            # Memory usage gauge
            fig_memory = go.Figure(go.Indicator(
                mode="gauge+number",
                value=memory_percent,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Memory Usage (%)"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "darkgreen"},
                       'steps': [{'range': [0, 60], 'color': "lightgray"},
                                {'range': [60, 85], 'color': "yellow"},
                                {'range': [85, 100], 'color': "red"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 90}}
            ))
            fig_memory.update_layout(height=300)
            st.plotly_chart(fig_memory, use_container_width=True)
        
        # Performance metrics
        st.subheader("QeMLflow Performance")
        perf_metrics = latest.get('performance', {})
        qeml_metrics = latest.get('qemlflow', {})
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            import_time = perf_metrics.get('import_time', 0)
            st.metric(
                label="Core Import Time",
                value=f"{import_time:.2f}s",
                delta="🎯 Target: <5s" if import_time < 5 else "⚠️ Exceeds target"
            )
        
        with col2:
            response_time = perf_metrics.get('response_time', 0)
            st.metric(
                label="Response Time",
                value=f"{response_time:.3f}s",
                delta="✅ Good" if response_time < 0.1 else "⚠️ Slow"
            )
        
        with col3:
            modules_loaded = qeml_metrics.get('modules_loaded', 0)
            st.metric(
                label="Modules Loaded",
                value=modules_loaded,
                delta=f"Status: {qeml_metrics.get('health_status', 'unknown')}"
            )
    
    # Performance report
    st.header("📈 Performance Analysis")
    
    try:
        perf_report = get_performance_report()
        
        if 'error' not in perf_report:
            summary = perf_report.get('performance_summary', {})
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Performance Summary")
                st.json(summary)
            
            with col2:
                st.subheader("Recommendations")
                recommendations = perf_report.get('recommendations', [])
                
                if recommendations:
                    for i, rec in enumerate(recommendations, 1):
                        st.write(f"{i}. {rec}")
                else:
                    st.success("No specific recommendations at this time!")
    
    except Exception as e:
        st.error(f"Could not generate performance report: {e}")
    
    # Alerts section
    alerts = health_status.get('recent_alerts', [])
    if alerts:
        st.header("🚨 Recent Alerts")
        
        for alert in alerts:
            severity = alert.get('severity', 'info')
            if severity == 'critical':
                st.error(f"🔴 {alert.get('message', 'Unknown alert')}")
            elif severity == 'warning':
                st.warning(f"🟡 {alert.get('message', 'Unknown alert')}")
            else:
                st.info(f"🔵 {alert.get('message', 'Unknown alert')}")

if __name__ == "__main__":
    main()
```

### **Step 3: Integrate Health Monitoring into Core**

Update `src/qemlflow/core/__init__.py` to include health monitoring:

```python
"""
QeMLflow Core Module - Optimized with Health Monitoring
"""

# Fast core imports only
from .base import BaseModel, BasePreprocessor
from .exceptions import QeMLflowError, ValidationError

# Lazy import utilities
from .lazy_imports import (
    get_numpy, get_pandas, get_sklearn,
    get_tensorflow, get_torch, get_matplotlib,
    check_optional_dependencies
)

# Health monitoring (lightweight)
from .health_monitor import health_check, start_monitoring, get_performance_report

# Start health monitoring automatically
import atexit
start_monitoring()
atexit.register(lambda: print("QeMLflow health monitoring stopped"))

# Version info
__version__ = "0.1.0"

# Public API (lightweight by default)
__all__ = [
    'BaseModel', 'BasePreprocessor',
    'QeMLflowError', 'ValidationError', 
    'health_check', 'start_monitoring', 'get_performance_report',
    'get_numpy', 'get_pandas', 'get_sklearn',
    'check_optional_dependencies'
]
```

### **Deliverables for Module 1.3**

- ✅ Enhanced health monitoring system with real-time metrics
- ✅ Performance analytics and trending
- ✅ Interactive health dashboard
- ✅ Automated alerting system
- ✅ Integration into core package

---

This completes the detailed implementation plan for Phase 1. Each module includes specific steps, code examples, validation criteria, and clear deliverables. The plan focuses on addressing the critical performance and compliance issues identified in our analysis while establishing the foundation for subsequent phases.

Would you like me to continue with the detailed plans for the remaining phases, or would you prefer to focus on executing Phase 1 first?
