"""
ChemML Lazy Loading Module
Implements smart lazy loading for heavy dependencies to improve import performance.
"""

import importlib
import importlib.util
import threading
import warnings
from functools import wraps
from typing import Any, Callable, Dict, Optional


class LazyImporter:
    """Smart lazy importer that loads modules only when needed."""

    def __init__(self, module_name: str, package: Optional[str] = None) -> None:
        self.module_name = module_name
        self.package = package
        self._module = None
        self._lock = threading.Lock()
        self._import_attempted = False
        self._import_error = None

    def __getattr__(self, name: str) -> Any:
        """Lazy load the module and return the requested attribute."""
        module = self._get_module()
        if module is None:
            raise self._import_error or ImportError(
                f"Failed to import {self.module_name}"
            )
        return getattr(module, name)

    def __call__(self, *args, **kwargs) -> Any:
        """Allow the lazy importer to be called if the module is callable."""
        module = self._get_module()
        if module is None:
            raise self._import_error or ImportError(
                f"Failed to import {self.module_name}"
            )
        return module(*args, **kwargs)

    def _get_module(self) -> Optional[Any]:
        """Get the module, importing it if necessary."""
        if self._module is not None:
            return self._module

        if self._import_attempted and self._import_error:
            return None

        with self._lock:
            if self._module is not None:
                return self._module

            if self._import_attempted and self._import_error:
                return None

            try:
                self._module = importlib.import_module(self.module_name, self.package)
                return self._module
            except ImportError as e:
                self._import_error = e
                warnings.warn(
                    f"Optional dependency {self.module_name} not available: {e}",
                    ImportWarning,
                    stacklevel=3,
                )
                return None
            finally:
                self._import_attempted = True

    def is_available(self) -> bool:
        """Check if the module is available without forcing import."""
        if self._module is not None:
            return True
        if self._import_attempted and self._import_error:
            return False

        # Quick availability check without full import
        try:
            importlib.util.find_spec(self.module_name)
            return True
        except (ImportError, ValueError, ModuleNotFoundError):
            return False


class ConditionalImporter:
    """Imports modules only when specific conditions are met."""

    def __init__(self, module_name: str, condition_func: Callable[[], bool]):
        self.module_name = module_name
        self.condition_func = condition_func
        self.lazy_importer = LazyImporter(module_name)

    def __getattr__(self, name: str) -> Any:
        """Get attribute only if condition is met."""
        if not self.condition_func():
            raise ImportError(f"{self.module_name} not available - condition not met")
        return getattr(self.lazy_importer, name)


# Define lazy imports for heavy dependencies
tensorflow = LazyImporter("tensorflow")
torch = LazyImporter("torch")
deepchem = LazyImporter("deepchem")
rdkit = LazyImporter("rdkit")
qiskit = ConditionalImporter("qiskit", lambda: False)  # Disabled by default

# ML libraries
sklearn = LazyImporter("sklearn")
xgboost = LazyImporter("xgboost")
lightgbm = LazyImporter("lightgbm")

# Visualization libraries
matplotlib = LazyImporter("matplotlib")
plotly = LazyImporter("plotly")
seaborn = LazyImporter("seaborn")

# Experiment tracking
wandb = LazyImporter("wandb")
mlflow = LazyImporter("mlflow")

# Molecular analysis
openmm = LazyImporter("openmm")
mdanalysis = LazyImporter("MDAnalysis")


def lazy_import(module_name: str, package: Optional[str] = None) -> LazyImporter:
    """Create a lazy importer for a module."""
    return LazyImporter(module_name, package)


def conditional_import(
    module_name: str, condition: Callable[[], bool]
) -> ConditionalImporter:
    """Create a conditional importer for a module."""
    return ConditionalImporter(module_name, condition)


def require_dependency(dependency_name: str) -> Any:
    """Decorator that ensures a dependency is available before running a function."""

    def decorator(func) -> Any:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Get the lazy importer from globals
            lazy_importer = globals().get(dependency_name.lower())
            if lazy_importer and not lazy_importer.is_available():
                raise ImportError(
                    f"Function {func.__name__} requires {dependency_name} "
                    f"but it's not available"
                )
            return func(*args, **kwargs)

        return wrapper

    return decorator


def check_dependencies() -> Dict[str, bool]:
    """Check availability of all lazy dependencies without importing them."""
    dependencies = {
        "tensorflow": tensorflow.is_available(),
        "torch": torch.is_available(),
        "deepchem": deepchem.is_available(),
        "rdkit": rdkit.is_available(),
        "sklearn": sklearn.is_available(),
        "matplotlib": matplotlib.is_available(),
        "wandb": wandb.is_available(),
    }
    return dependencies


# Performance monitoring
import time


def monitor_import_time(func) -> Any:
    """Decorator to monitor import times."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        import_time = end_time - start_time
        if import_time > 0.1:  # Only log slow imports
            print(f"⏱️  Import {func.__name__}: {import_time:.3f}s")
        return result

    return wrapper
