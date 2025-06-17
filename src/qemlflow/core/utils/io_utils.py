from typing import Dict\nfrom typing import List\nfrom typing import Optional\nfrom typing import Union\n"""
Input/Output utilities for QeMLflow

This module provides utilities for data loading, saving results,
and configuration management.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import yaml


class DataLoader:
    """Load data from various formats for QeMLflow workflows"""

    def __init__(self, data_dir: Optional[str] = None) -> None:
        self.data_dir = Path(data_dir) if data_dir else Path("data")

    def load_csv(self, filename: str, **kwargs) -> pd.DataFrame:
        """Load CSV file"""
        filepath = self.data_dir / filename
        return pd.read_csv(filepath, **kwargs)

    def load_sdf(self, filename: str) -> List[Dict]:
        """Load SDF (Structure Data File) for molecular data"""
        try:
            from rdkit import Chem
        except ImportError:
            raise ImportError("RDKit required for SDF loading")

        filepath = self.data_dir / filename
        supplier = Chem.SDMolSupplier(str(filepath))

        molecules = []
        for mol in supplier:
            if mol is not None:
                mol_dict = {
                    "mol": mol,
                    "smiles": Chem.MolToSmiles(mol),
                    "properties": mol.GetPropsAsDict(),
                }
                molecules.append(mol_dict)

        return molecules

    def load_smiles_file(
        self, filename: str, smiles_column: str = "SMILES", delimiter: str = "\t"
    ) -> pd.DataFrame:
        """Load file containing SMILES strings"""
        filepath = self.data_dir / filename

        if filename.endswith(".csv"):
            return pd.read_csv(filepath)
        else:
            return pd.read_csv(filepath, delimiter=delimiter)

    def load_protein_fasta(self, filename: str) -> Dict[str, str]:
        """Load protein sequences from FASTA file"""
        try:
            from Bio import SeqIO
        except ImportError:
            raise ImportError("Biopython required for FASTA loading")

        filepath = self.data_dir / filename
        sequences = {}

        for record in SeqIO.parse(filepath, "fasta"):
            sequences[record.id] = str(record.seq)

        return sequences

    def load_chembl_data(
        self, target_id: str, activity_type: str = "IC50"
    ) -> pd.DataFrame:
        """Load data from ChEMBL database"""
        try:
            from chembl_webresource_client.new_client import new_client
        except ImportError:
            raise ImportError("ChEMBL web client required for ChEMBL data loading")

        # Get target information
        target = new_client.target
        _ = target.get(target_id)  # Validate target exists

        # Get activities
        activity = new_client.activity
        activities = activity.filter(
            target_chembl_id=target_id, standard_type=activity_type, assay_type="B"
        )  # Binding assays

        # Convert to DataFrame
        df = pd.DataFrame(activities)

        logging.info(f"Loaded {len(df)} activities for target {target_id}")
        return df

    def load_json(self, filename: str) -> Dict:
        """Load JSON file"""
        filepath = self.data_dir / filename
        with open(filepath, "r") as f:
            return json.load(f)

    def load_pickle(self, filename: str) -> Any:
        """Load pickled object"""
        filepath = self.data_dir / filename
        with open(filepath, "rb") as f:
            return pickle.load(f)


def load_molecular_data(
    filepath: Union[str, Path],
    smiles_column: str = "SMILES",
    target_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load molecular data from various file formats.

    Args:
        filepath: Path to the molecular data file
        smiles_column: Name of the SMILES column (default: "SMILES")
        target_column: Name of the target column (optional)

    Returns:
        DataFrame with molecular data

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If required columns are missing
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Determine file format from extension
    extension = filepath.suffix.lower()

    try:
        if extension == ".csv":
            df = pd.read_csv(filepath)
        elif extension in [".xlsx", ".xls"]:
            df = pd.read_excel(filepath)
        elif extension == ".json":
            df = pd.read_json(filepath)
        elif extension == ".parquet":
            df = pd.read_parquet(filepath)
        elif extension == ".pkl" or extension == ".pickle":
            with open(filepath, "rb") as f:
                df = pickle.load(f)
        else:
            # Try CSV as default
            df = pd.read_csv(filepath)

        # Try to find SMILES column with different case variations
        smiles_col_found = None
        for col in df.columns:
            if col.lower() == smiles_column.lower():
                smiles_col_found = col
                break

        if smiles_col_found is None:
            # Try common variations
            for variant in ["SMILES", "smiles", "Smiles", "SMILE", "smile"]:
                if variant in df.columns:
                    smiles_col_found = variant
                    break

        if smiles_col_found is None:
            raise ValueError(f"SMILES column '{smiles_column}' not found in data")

        # Rename to standard name if needed (always use lowercase 'smiles')
        if smiles_col_found != "smiles":
            df = df.rename(columns={smiles_col_found: "smiles"})

        if target_column and target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")

        # Basic data validation
        if df.empty:
            raise ValueError("Loaded data is empty")

        logging.info(
            f"Successfully loaded molecular data: {len(df)} molecules from {filepath}"
        )
        return df

    except Exception as e:
        logging.error(f"Error loading molecular data from {filepath}: {e}")
        raise


def save_molecular_data(
    data: pd.DataFrame, filepath: Union[str, Path], format: Optional[str] = None
) -> None:
    """
    Save molecular data to various file formats.

    Args:
        data: DataFrame with molecular data
        filepath: Path to save the data
        format: File format (auto-detected from extension if None)
    """
    filepath = Path(filepath)

    # Create parent directories if they don't exist
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Determine format from extension if not specified
    if format is None:
        format = filepath.suffix.lower().lstrip(".")

    try:
        if format == "csv":
            data.to_csv(filepath, index=False)
        elif format in ["xlsx", "xls"]:
            data.to_excel(filepath, index=False)
        elif format == "json":
            data.to_json(filepath, orient="records", indent=2)
        elif format == "parquet":
            data.to_parquet(filepath, index=False)
        elif format in ["pkl", "pickle"]:
            with open(filepath, "wb") as f:
                pickle.dump(data, f)
        else:
            # Default to CSV
            data.to_csv(filepath, index=False)

        logging.info(
            f"Successfully saved molecular data: {len(data)} molecules to {filepath}"
        )

    except Exception as e:
        logging.error(f"Error saving molecular data to {filepath}: {e}")
        raise


class ResultsExporter:
    """Export results and visualizations"""

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("results")
        self.output_dir.mkdir(exist_ok=True)

    def save_dataframe(
        self, df: pd.DataFrame, filename: str, format: str = "csv"
    ) -> None:
        """Save DataFrame in various formats"""
        filepath = self.output_dir / filename

        if format == "csv":
            df.to_csv(filepath, index=False)
        elif format == "excel":
            df.to_excel(filepath, index=False)
        elif format == "parquet":
            df.to_parquet(filepath, index=False)
        elif format == "json":
            df.to_json(filepath, orient="records", indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logging.info(f"Saved DataFrame to {filepath}")

    def save_model_results(self, results: Dict, filename: str) -> None:
        """Save model training/evaluation results"""
        filepath = self.output_dir / filename

        # Convert numpy arrays to lists for JSON serialization
        serializable_results = self._make_json_serializable(results)

        with open(filepath, "w") as f:
            json.dump(serializable_results, f, indent=2)

        logging.info(f"Saved model results to {filepath}")

    def save_plot(
        self, fig, filename: str, dpi: int = 300, bbox_inches: str = "tight"
    ) -> None:
        """Save matplotlib figure"""
        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)
        logging.info(f"Saved plot to {filepath}")

    def export_smiles_list(
        self, smiles_list: List[str], filename: str, include_index: bool = True
    ):
        """Export list of SMILES to file"""
        filepath = self.output_dir / filename

        if include_index:
            df = pd.DataFrame({"ID": range(len(smiles_list)), "SMILES": smiles_list})
            df.to_csv(filepath, index=False)
        else:
            with open(filepath, "w") as f:
                for smiles in smiles_list:
                    f.write(f"{smiles}\n")

        logging.info(f"Exported {len(smiles_list)} SMILES to {filepath}")

    def save_molecular_descriptors(
        self,
        descriptors: pd.DataFrame,
        smiles_column: str = "SMILES",
        filename: str = "molecular_descriptors.csv",
    ):
        """Save molecular descriptors with SMILES"""
        filepath = self.output_dir / filename
        descriptors.to_csv(filepath, index=False)
        logging.info(f"Saved molecular descriptors to {filepath}")

    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj


class ConfigManager:
    """Manage configuration files for experiments"""

    def __init__(self, config_dir: Optional[str] = None):
        self.config_dir = Path(config_dir) if config_dir else Path("configs")
        self.config_dir.mkdir(exist_ok=True)

    def load_config(self, filename: str) -> Dict:
        """Load configuration from YAML or JSON file"""
        filepath = self.config_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")

        if filename.endswith(".yaml") or filename.endswith(".yml"):
            with open(filepath, "r") as f:
                return yaml.safe_load(f)
        elif filename.endswith(".json"):
            with open(filepath, "r") as f:
                return json.load(f)
        else:
            raise ValueError("Config file must be YAML or JSON format")

    def save_config(self, config: Dict, filename: str) -> None:
        """Save configuration to file"""
        filepath = self.config_dir / filename

        if filename.endswith(".yaml") or filename.endswith(".yml"):
            with open(filepath, "w") as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
        elif filename.endswith(".json"):
            with open(filepath, "w") as f:
                json.dump(config, f, indent=2)
        else:
            raise ValueError("Config file must be YAML or JSON format")

        logging.info(f"Saved config to {filepath}")

    def create_experiment_config(
        self,
        experiment_name: str,
        model_params: Dict,
        data_params: Dict,
        training_params: Dict,
    ) -> str:
        """Create comprehensive experiment configuration"""
        config = {
            "experiment": {
                "name": experiment_name,
                "description": f"QeMLflow experiment: {experiment_name}",
                "created_at": pd.Timestamp.now().isoformat(),
            },
            "data": data_params,
            "model": model_params,
            "training": training_params,
            "paths": {
                "data_dir": "data/",
                "results_dir": "results/",
                "models_dir": "models/",
            },
        }

        filename = f"{experiment_name}_config.yaml"
        self.save_config(config, filename)
        return filename


class FileManager:
    """Manage file operations and directory structure"""

    @staticmethod
    def create_directory_structure(base_dir: str, subdirs: List[str]) -> None:
        """Create directory structure for project"""
        base_path = Path(base_dir)
        base_path.mkdir(exist_ok=True)

        for subdir in subdirs:
            (base_path / subdir).mkdir(exist_ok=True)

        logging.info(f"Created directory structure in {base_dir}")

    @staticmethod
    def list_files(directory: str, pattern: str = "*") -> List[str]:
        """List files matching pattern in directory"""
        dir_path = Path(directory)
        return [str(f) for f in dir_path.glob(pattern)]

    @staticmethod
    def get_file_info(filepath: str) -> Dict:
        """Get file information"""
        path = Path(filepath)
        if not path.exists():
            return {"exists": False}

        stat = path.stat()
        return {
            "exists": True,
            "size_bytes": stat.st_size,
            "size_mb": stat.st_size / (1024 * 1024),
            "modified": pd.Timestamp.fromtimestamp(stat.st_mtime),
            "is_file": path.is_file(),
            "is_dir": path.is_dir(),
            "suffix": path.suffix,
        }

    @staticmethod
    def backup_file(filepath: str, backup_dir: Optional[str] = None) -> str:
        """Create backup of file"""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        if backup_dir:
            backup_path = Path(backup_dir) / f"{path.stem}_backup{path.suffix}"
        else:
            backup_path = path.parent / f"{path.stem}_backup{path.suffix}"

        import shutil

        shutil.copy2(path, backup_path)

        logging.info(f"Created backup: {backup_path}")
        return str(backup_path)


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Setup logging configuration"""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    handlers = [logging.StreamHandler()]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, log_level.upper()), format=log_format, handlers=handlers
    )

    logging.info("Logging configured")


def validate_data_integrity(df: pd.DataFrame, required_columns: List[str]) -> Dict:
    """Validate data integrity for ML workflows"""
    issues = {
        "missing_columns": [],
        "missing_values": {},
        "duplicate_rows": 0,
        "data_types": {},
        "summary": {},
    }

    # Check required columns
    for col in required_columns:
        if col not in df.columns:
            issues["missing_columns"].append(col)

    # Check missing values
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            issues["missing_values"][col] = missing_count

    # Check duplicates
    issues["duplicate_rows"] = df.duplicated().sum()

    # Data types
    issues["data_types"] = df.dtypes.to_dict()

    # Summary
    issues["summary"] = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
    }

    return issues


def export_results(
    results: Dict[str, Any], filepath: str, format: str = "json", **kwargs
) -> None:
    """
    Export results to various formats.

    Args:
        results: Dictionary of results to export
        filepath: Output file path
        format: Export format ('json', 'yaml', 'csv', 'pickle')
        **kwargs: Additional arguments for specific formats
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if format.lower() == "json":
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2, default=_json_serializer)

    elif format.lower() == "yaml":
        with open(filepath, "w") as f:
            yaml.dump(results, f, default_flow_style=False)

    elif format.lower() == "csv":
        # Convert dict to DataFrame if possible
        if isinstance(results, dict):
            df = pd.DataFrame([results])
        else:
            df = pd.DataFrame(results)
        df.to_csv(filepath, index=False, **kwargs)

    elif format.lower() == "pickle":
        with open(filepath, "wb") as f:
            pickle.dump(results, f)

    else:
        raise ValueError(f"Unsupported format: {format}")


def save_model_results(
    model_results: Dict[str, Any], experiment_name: str, output_dir: str = "results"
) -> str:
    """
    Save model training/evaluation results.

    Args:
        model_results: Dictionary containing model results
        experiment_name: Name of the experiment
        output_dir: Output directory

    Returns:
        Path to saved results file
    """
    output_path = Path(output_dir) / f"{experiment_name}_results.json"
    export_results(model_results, str(output_path), format="json")
    return str(output_path)


def load_experiment_results(
    experiment_name: str, results_dir: str = "results"
) -> Dict[str, Any]:
    """
    Load previously saved experiment results.

    Args:
        experiment_name: Name of the experiment
        results_dir: Directory containing results

    Returns:
        Dictionary of experiment results
    """
    results_path = Path(results_dir) / f"{experiment_name}_results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    with open(results_path, "r") as f:
        return json.load(f)


def _json_serializer(obj) -> Any:
    """Custom JSON serializer for numpy arrays and other objects."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict("records")
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif hasattr(obj, "__dict__"):
        return obj.__dict__
    else:
        return str(obj)
