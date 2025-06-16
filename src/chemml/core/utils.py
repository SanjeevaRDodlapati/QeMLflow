"""
ChemML Core Utilities
====================

Common utilities and helper functions for ChemML.
Provides logging, environment setup, and convenience functions.

Key Features:
- Logging configuration
- Environment checking and setup
- Sample data generation
- Common helper functions
"""
import importlib.util
import logging
import os
import sys
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def setup_logging(
    level: str = "INFO",
    filepath: Optional[str] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Setup logging configuration for ChemML.

    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: Path to log file (optional)
        format_string: Custom format string (optional)

    Returns:
        Configured logger instance
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format=format_string,
        handlers=[
            logging.StreamHandler(sys.stdout),
            *([] if filepath is None else [logging.FileHandler(filepath)]),
        ],
    )
    logger = logging.getLogger("chemml")
    logger.info(f"ChemML logging initialized at {level} level")
    return logger


def check_environment() -> Dict[str, Any]:
    """
    Check the current environment and available dependencies.

    Returns:
        Dictionary with environment information
    """
    env_info = {
        "python_version": sys.version,
        "platform": sys.platform,
        "available_packages": {},
        "missing_packages": [],
        "warnings": [],
    }
    required_packages = ["numpy", "pandas", "sklearn", "matplotlib", "seaborn"]
    optional_packages = ["rdkit", "deepchem", "torch", "tensorflow", "wandb", "jupyter"]
    all_packages = required_packages + optional_packages
    for package in all_packages:
        try:
            spec = importlib.util.find_spec(package)
            if spec is not None:
                try:
                    module = importlib.import_module(package)
                    version = getattr(module, "__version__", "unknown")
                    env_info["available_packages"][package] = {
                        "version": version,
                        "required": package in required_packages,
                    }
                except Exception as e:
                    env_info["available_packages"][package] = {
                        "version": "import_error",
                        "error": str(e),
                        "required": package in required_packages,
                    }
            else:
                env_info["missing_packages"].append(package)
                if package in required_packages:
                    env_info["warnings"].append(
                        f"Required package '{package}' not found"
                    )
        except Exception as e:
            env_info["missing_packages"].append(package)
            env_info["warnings"].append(f"Error checking package '{package}': {e}")
    return env_info


def print_environment_report(env_info: Optional[Dict[str, Any]] = None) -> Any:
    """
    Print a formatted environment report.

    Args:
        env_info: Environment info from check_environment() (optional)
    """
    if env_info is None:
        env_info = check_environment()
    print("ChemML Environment Report")
    print("=" * 50)
    print(f"Python Version: {env_info['python_version']}")
    print(f"Platform: {env_info['platform']}")
    print()
    print("Available Packages:")
    print("-" * 30)
    for package, info in env_info["available_packages"].items():
        status = "✅" if info.get("version") != "import_error" else "❌"
        required = " (required)" if info.get("required") else " (optional)"
        print(f"{status} {package}: {info.get('version')}{required}")
    if env_info["missing_packages"]:
        print("\nMissing Packages:")
        print("-" * 30)
        for package in env_info["missing_packages"]:
            print(f"❌ {package}")
    if env_info["warnings"]:
        print("\nWarnings:")
        print("-" * 30)
        for warning in env_info["warnings"]:
            print(f"⚠️  {warning}")
    print()


def get_sample_data(dataset: str = "molecules", size: int = 100) -> pd.DataFrame:
    """
    Generate sample data for testing and tutorials.

    Args:
        dataset: Type of dataset ('molecules', 'properties', 'toxicity')
        size: Number of samples to generate

    Returns:
        Sample DataFrame
    """
    np.random.seed(42)
    if dataset == "molecules":
        sample_smiles = [
            "CCO",
            "CC(=O)OC1=CC=CC=C1C(=O)O",
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
            "C1=CC=CC=C1",
            "CCN(CC)CCCC(=O)O",
            "CC(C)(C)NCC(C1=CC(=C(C=C1)O)CO)O",
            "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",
        ]
        data = []
        for i in range(size):
            base_smiles = sample_smiles[i % len(sample_smiles)]
            data.append(
                {
                    "smiles": base_smiles,
                    "molecular_weight": np.random.uniform(100, 500),
                    "logp": np.random.uniform(-3, 5),
                    "num_heavy_atoms": np.random.randint(5, 30),
                    "is_drug_like": np.random.choice([True, False], p=[0.7, 0.3]),
                }
            )
        return pd.DataFrame(data)
    elif dataset == "properties":
        data = []
        for i in range(size):
            data.append(
                {
                    "compound_id": f"COMP_{i:04d}",
                    "solubility": np.random.normal(-3, 2),
                    "permeability": np.random.lognormal(0, 1),
                    "clearance": np.random.exponential(20),
                    "bioavailability": np.random.beta(2, 2),
                    "toxicity_score": np.random.uniform(0, 1),
                }
            )
        return pd.DataFrame(data)
    elif dataset == "toxicity":
        data = []
        endpoints = [
            "hepatotoxicity",
            "cardiotoxicity",
            "nephrotoxicity",
            "neurotoxicity",
        ]
        for i in range(size):
            row = {"compound_id": f"TOX_{i:04d}"}
            for endpoint in endpoints:
                row[endpoint] = np.random.choice([0, 1], p=[0.8, 0.2])
            row["severity"] = np.random.choice(
                ["low", "medium", "high"], p=[0.5, 0.3, 0.2]
            )
            data.append(row)
        return pd.DataFrame(data)
    else:
        raise ValueError(f"Unknown dataset type: {dataset}")


def validate_smiles(smiles_list: List[str]) -> Tuple[List[bool], List[str]]:
    """
    Validate SMILES strings using RDKit (if available).

    Args:
        smiles_list: List of SMILES strings to validate

    Returns:
        Tuple of (validity_flags, error_messages)
    """
    try:
        from rdkit import Chem

        validity_flags = []
        error_messages = []
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(str(smiles))
                if mol is not None:
                    validity_flags.append(True)
                    error_messages.append("")
                else:
                    validity_flags.append(False)
                    error_messages.append("Invalid SMILES structure")
            except Exception as e:
                validity_flags.append(False)
                error_messages.append(str(e))
        return validity_flags, error_messages
    except ImportError:
        warnings.warn("RDKit not available. Cannot validate SMILES.")
        return [True] * len(smiles_list), [""] * len(smiles_list)


def download_sample_datasets(data_dir: str = "./data") -> Dict[str, str]:
    """
    Download common chemistry datasets for tutorials and testing.

    Args:
        data_dir: Directory to save datasets

    Returns:
        Dictionary mapping dataset names to file paths
    """
    os.makedirs(data_dir, exist_ok=True)
    datasets = {}
    solubility_data = get_sample_data("molecules", 200)
    solubility_data["logS"] = np.random.normal(-3, 2, len(solubility_data))
    solubility_path = os.path.join(data_dir, "solubility_sample.csv")
    solubility_data.to_csv(solubility_path, index=False)
    datasets["solubility"] = solubility_path
    toxicity_data = get_sample_data("toxicity", 150)
    toxicity_path = os.path.join(data_dir, "toxicity_sample.csv")
    toxicity_data.to_csv(toxicity_path, index=False)
    datasets["toxicity"] = toxicity_path
    admet_data = get_sample_data("properties", 300)
    admet_path = os.path.join(data_dir, "admet_sample.csv")
    admet_data.to_csv(admet_path, index=False)
    datasets["admet"] = admet_path
    print(f"Sample datasets created in {data_dir}:")
    for name, path in datasets.items():
        print(f"  {name}: {path}")
    return datasets


def memory_usage_mb() -> float:
    """
    Get current memory usage in MB.

    Returns:
        Memory usage in megabytes
    """
    try:
        import psutil

        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        warnings.warn("psutil not available. Cannot get memory usage.")
        return 0.0


def configure_warnings(
    action: str = "ignore", categories: Optional[List[str]] = None
) -> Any:
    """
    Configure warning filters for cleaner output.

    Args:
        action: Warning action ('ignore', 'default', 'error', 'once')
        categories: List of warning categories to filter
    """
    if categories is None:
        categories = ["DeprecationWarning", "FutureWarning", "UserWarning"]
    for category in categories:
        try:
            category_class = getattr(warnings, category, None)
            if category_class:
                warnings.filterwarnings(action, category=category_class)
        except Exception:
            pass


def ensure_reproducibility(seed: int = 42) -> Any:
    """
    Set random seeds for reproducible results.

    Args:
        seed: Random seed to use
    """
    np.random.seed(seed)
    try:
        import random

        random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass


def create_directory_structure(base_filepath: str) -> Any:
    """
    Create standard ChemML project directory structure.

    Args:
        base_path: Base directory path
    """
    directories = [
        "data/raw",
        "data/processed",
        "data/external",
        "notebooks",
        "scripts",
        "models",
        "reports/figures",
        "reports/tables",
    ]
    for dir_path in directories:
        full_path = os.path.join(base_filepath, dir_path)
        os.makedirs(full_path, exist_ok=True)
    print(f"Created ChemML project structure in {base_filepath}")


def format_large_number(number: float) -> str:
    """
    Format large numbers with appropriate units.

    Args:
        number: Number to format

    Returns:
        Formatted string
    """
    if number >= 1000000000.0:
        return f"{number / 1000000000.0:.1f}B"
    elif number >= 1000000.0:
        return f"{number / 1000000.0:.1f}M"
    elif number >= 1000.0:
        return f"{number / 1000.0:.1f}K"
    else:
        return f"{number:.1f}"


def time_function(func: Any) -> Any:
    """
    Decorator to time function execution.

    Args:
        func: Function to time

    Returns:
        Wrapped function
    """
    import functools
    import time

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{func.__name__} executed in {execution_time:.2f} seconds")
        return result

    return wrapper


__all__ = [
    "setup_logging",
    "check_environment",
    "print_environment_report",
    "get_sample_data",
    "validate_smiles",
    "download_sample_datasets",
    "memory_usage_mb",
    "configure_warnings",
    "ensure_reproducibility",
    "create_directory_structure",
    "format_large_number",
    "time_function",
]
