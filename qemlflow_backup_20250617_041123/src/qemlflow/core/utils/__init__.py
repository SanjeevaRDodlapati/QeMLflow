"""
ChemML Core Utils package providing utility functions for chemistry applications.
"""

import importlib.util
import os

# Import from the original utils.py file

# Get the path to the utils.py file
utils_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "utils.py")
spec = importlib.util.spec_from_file_location("utils_module", utils_path)
utils_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils_module)

# Import the functions we need from the original utils.py
setup_logging = getattr(utils_module, "setup_logging", None)
check_environment = getattr(utils_module, "check_environment", None)
get_sample_data = getattr(utils_module, "get_sample_data", None)
validate_input = getattr(utils_module, "validate_input", None)
ensure_reproducibility = getattr(utils_module, "ensure_reproducibility", None)
configure_warnings = getattr(utils_module, "configure_warnings", None)

# Import from individual utility modules
try:
    from .io_utils import *
except ImportError:
    pass

try:
    from .molecular_utils import *
except ImportError:
    pass

try:
    from .ml_utils import *
except ImportError:
    pass

try:
    from .metrics import *
except ImportError:
    pass

try:
    from .quantum_utils import *
except ImportError:
    pass

try:
    from .visualization import *
except ImportError:
    pass

__all__ = [
    "setup_logging",
    "check_environment",
    "get_sample_data",
    "ensure_reproducibility",
    "configure_warnings",
]
