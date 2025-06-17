"""
Boltz Model Integration for QeMLflow
==================================

Specialized adapter for integrating Boltz biomolecular interaction models
into the QeMLflow framework. Boltz is a state-of-the-art model for protein
structure prediction and binding affinity prediction.

Repository: https://github.com/jwohlwend/boltz
Papers:
- Boltz-1: Democratizing Biomolecular Interaction Modeling (2024)
- Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction (2025)
"""

import json
import os
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml

try:
    from ...core.external_models import ExternalModelWrapper
except ImportError:
    from qemlflow.integrations.core.external_models import ExternalModelWrapper


class BoltzAdapter(ExternalModelWrapper):
    """
    Specialized adapter for Boltz biomolecular interaction models.

    Provides integration for:
    - Protein structure prediction
    - Protein-ligand complex prediction
    - Binding affinity prediction
    - Multi-chain complex modeling
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        use_msa_server: bool = True,
        model_version: str = "latest",
        device: str = "auto",
    ):
        """
        Initialize Boltz adapter.

        Args:
            cache_dir: Directory to cache Boltz models and data
            use_msa_server: Whether to use MSA server for automatic MSA generation
            model_version: Boltz model version to use
            device: Computing device ('gpu', 'cpu', 'auto')
        """
        super().__init__(
            repo_url="https://github.com/jwohlwend/boltz",
            model_class_name="Boltz",
            model_name="Boltz",
        )

        self.cache_dir = (
            Path(cache_dir) if cache_dir else Path.home() / ".qemlflow" / "boltz"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.use_msa_server = use_msa_server
        self.model_version = model_version
        self.device = device

        # Boltz configuration
        self.boltz_installed = False
        self.boltz_cache_dir = self.cache_dir / "boltz_cache"

        # Task tracking
        self.supported_tasks = [
            "structure_prediction",
            "complex_prediction",
            "affinity_prediction",
            "protein_folding",
        ]

        self._install_boltz()

    def _install_boltz(self):
        """Install Boltz if not already available."""
        try:
            import boltz

            self.boltz_installed = True
            print("✓ Boltz already installed")
        except ImportError:
            print("Installing Boltz...")
            try:
                subprocess.run(
                    ["pip", "install", "boltz", "-U"], check=True, capture_output=True
                )

                # Verify installation
                import boltz

                self.boltz_installed = True
                print("✓ Boltz installed successfully")

            except subprocess.CalledProcessError as e:
                warnings.warn(f"Failed to install Boltz: {e}")
                self.boltz_installed = False
            except ImportError:
                warnings.warn("Boltz installation succeeded but import failed")
                self.boltz_installed = False

    def prepare_input(self, data: Dict[str, Any]) -> str:
        """
        Prepare input data for Boltz prediction.

        Args:
            data: Input data containing sequences, structures, etc.

        Returns:
            Path to prepared input file (YAML or FASTA)
        """
        if not self.boltz_installed:
            raise RuntimeError("Boltz is not installed. Please install it first.")

        # Determine input format based on complexity
        needs_yaml = any(
            [
                data.get("constraints"),
                data.get("templates"),
                data.get("properties"),
                data.get("modifications"),
                len(data.get("sequences", [])) > 1
                and any(
                    "ligand" in str(seq).lower() for seq in data.get("sequences", [])
                ),
            ]
        )

        if needs_yaml:
            return self._prepare_yaml_input(data)
        else:
            return self._prepare_fasta_input(data)

    def _prepare_yaml_input(self, data: Dict[str, Any]) -> str:
        """Prepare YAML input file for complex predictions."""
        yaml_data = {"version": 1, "sequences": []}

        # Process sequences
        for seq_data in data.get("sequences", []):
            if seq_data.get("type") == "protein":
                seq_entry = {
                    "protein": {
                        "id": seq_data.get("id", "A"),
                        "sequence": seq_data["sequence"],
                    }
                }

                # Add MSA if provided
                if seq_data.get("msa_path"):
                    seq_entry["protein"]["msa"] = seq_data["msa_path"]
                elif not self.use_msa_server:
                    seq_entry["protein"]["msa"] = "empty"  # Single sequence mode

            elif seq_data.get("type") == "ligand":
                seq_entry = {
                    "ligand": {
                        "id": seq_data.get("id", "L"),
                    }
                }

                if seq_data.get("smiles"):
                    seq_entry["ligand"]["smiles"] = seq_data["smiles"]
                elif seq_data.get("ccd"):
                    seq_entry["ligand"]["ccd"] = seq_data["ccd"]
                else:
                    raise ValueError("Ligand must have either 'smiles' or 'ccd'")

            elif seq_data.get("type") in ["dna", "rna"]:
                seq_entry = {
                    seq_data["type"]: {
                        "id": seq_data.get("id", "N"),
                        "sequence": seq_data["sequence"],
                    }
                }

            yaml_data["sequences"].append(seq_entry)

        # Add constraints if provided
        if data.get("constraints"):
            yaml_data["constraints"] = data["constraints"]

        # Add templates if provided
        if data.get("templates"):
            yaml_data["templates"] = data["templates"]

        # Add properties (e.g., affinity prediction)
        if data.get("properties"):
            yaml_data["properties"] = data["properties"]

        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        yaml.dump(yaml_data, temp_file, default_flow_style=False)
        temp_file.close()

        return temp_file.name

    def _prepare_fasta_input(self, data: Dict[str, Any]) -> str:
        """Prepare FASTA input file for simple predictions."""
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False)

        for seq_data in data.get("sequences", []):
            chain_id = seq_data.get("id", "A")
            entity_type = seq_data.get("type", "protein")
            sequence = seq_data.get("sequence", "")

            if entity_type == "ligand":
                if seq_data.get("smiles"):
                    entity_type = "smiles"
                    sequence = seq_data["smiles"]
                elif seq_data.get("ccd"):
                    entity_type = "ccd"
                    sequence = seq_data["ccd"]

            # MSA path for proteins
            msa_path = seq_data.get(
                "msa_path", "empty" if not self.use_msa_server else ""
            )

            if entity_type == "protein" and msa_path:
                header = f">{chain_id}|{entity_type}|{msa_path}"
            else:
                header = f">{chain_id}|{entity_type}"

            temp_file.write(f"{header}\n{sequence}\n")

        temp_file.close()
        return temp_file.name

    def predict(
        self,
        input_data: Union[str, Dict[str, Any]],
        task: str = "structure_prediction",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run Boltz prediction.

        Args:
            input_data: Input data (file path or structured data)
            task: Prediction task type
            **kwargs: Additional Boltz parameters

        Returns:
            Dictionary containing prediction results
        """
        if not self.boltz_installed:
            raise RuntimeError("Boltz is not installed")

        if task not in self.supported_tasks:
            raise ValueError(
                f"Task '{task}' not supported. Available: {self.supported_tasks}"
            )

        # Prepare input file
        if isinstance(input_data, str):
            input_file = input_data
        else:
            input_file = self.prepare_input(input_data)

        # Prepare output directory
        output_dir = self.cache_dir / f"predictions_{os.path.basename(input_file)}"
        output_dir.mkdir(exist_ok=True)

        # Build Boltz command
        cmd = [
            "boltz",
            "predict",
            input_file,
            "--out_dir",
            str(output_dir),
            "--cache",
            str(self.boltz_cache_dir),
        ]

        # Add MSA server if enabled
        if self.use_msa_server:
            cmd.append("--use_msa_server")

        # Add device configuration
        if self.device != "auto":
            cmd.extend(["--accelerator", self.device])

        # Add task-specific parameters
        if task == "affinity_prediction":
            cmd.extend(
                [
                    "--diffusion_samples_affinity",
                    str(kwargs.get("affinity_samples", 5)),
                    "--sampling_steps_affinity",
                    str(kwargs.get("affinity_steps", 200)),
                ]
            )

        # Add general parameters
        cmd.extend(
            [
                "--recycling_steps",
                str(kwargs.get("recycling_steps", 3)),
                "--diffusion_samples",
                str(kwargs.get("diffusion_samples", 1)),
                "--sampling_steps",
                str(kwargs.get("sampling_steps", 200)),
            ]
        )

        # Add additional options
        if kwargs.get("use_potentials", False):
            cmd.append("--use_potentials")

        if kwargs.get("override", True):
            cmd.append("--override")

        try:
            # Run Boltz prediction
            print(f"Running Boltz prediction: {' '.join(cmd)}")
            _result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            # Parse results
            results = self._parse_results(output_dir, task)

            # Clean up temporary input file if we created it
            if isinstance(input_data, dict) and os.path.exists(input_file):
                os.unlink(input_file)

            return results

        except subprocess.CalledProcessError as e:
            error_msg = f"Boltz prediction failed: {e.stderr}"
            print(f"Error: {error_msg}")
            raise RuntimeError(error_msg)

    def _parse_results(self, output_dir: Path, task: str) -> Dict[str, Any]:
        """Parse Boltz prediction results."""
        results = {
            "task": task,
            "output_dir": str(output_dir),
            "structures": [],
            "confidence": {},
            "affinity": {},
            "status": "completed",
        }

        # Find prediction directories
        predictions_dir = output_dir / "predictions"
        if not predictions_dir.exists():
            warnings.warn(f"No predictions directory found in {output_dir}")
            results["status"] = "failed"
            return results

        # Process each prediction subdirectory
        for pred_subdir in predictions_dir.iterdir():
            if pred_subdir.is_dir():
                pred_results = self._parse_single_prediction(pred_subdir)
                results["structures"].extend(pred_results.get("structures", []))

                if pred_results.get("confidence"):
                    results["confidence"][pred_subdir.name] = pred_results["confidence"]

                if pred_results.get("affinity"):
                    results["affinity"][pred_subdir.name] = pred_results["affinity"]

        return results

    def _parse_single_prediction(self, pred_dir: Path) -> Dict[str, Any]:
        """Parse results from a single prediction directory."""
        results = {"structures": [], "confidence": {}, "affinity": {}}

        # Find structure files
        for cif_file in pred_dir.glob("*.cif"):
            results["structures"].append(
                {"path": str(cif_file), "format": "cif", "model_name": cif_file.stem}
            )

        # Parse confidence scores
        for conf_file in pred_dir.glob("confidence_*.json"):
            try:
                with open(conf_file, "r") as f:
                    conf_data = json.load(f)
                results["confidence"] = conf_data
                break
            except Exception as e:
                warnings.warn(f"Failed to parse confidence file {conf_file}: {e}")

        # Parse affinity scores
        for aff_file in pred_dir.glob("affinity_*.json"):
            try:
                with open(aff_file, "r") as f:
                    aff_data = json.load(f)
                results["affinity"] = aff_data
                break
            except Exception as e:
                warnings.warn(f"Failed to parse affinity file {aff_file}: {e}")

        return results

    def predict_structure(
        self, sequence: str, chain_id: str = "A", **kwargs
    ) -> Dict[str, Any]:
        """
        Predict protein structure from sequence.

        Args:
            sequence: Protein amino acid sequence
            chain_id: Chain identifier
            **kwargs: Additional parameters

        Returns:
            Structure prediction results
        """
        input_data = {
            "sequences": [{"type": "protein", "id": chain_id, "sequence": sequence}]
        }

        return self.predict(input_data, task="structure_prediction", **kwargs)

    def predict_complex(
        self,
        protein_sequence: str,
        ligand_smiles: str,
        protein_id: str = "A",
        ligand_id: str = "L",
        predict_affinity: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Predict protein-ligand complex structure and optionally binding affinity.

        Args:
            protein_sequence: Protein amino acid sequence
            ligand_smiles: Ligand SMILES string
            protein_id: Protein chain identifier
            ligand_id: Ligand chain identifier
            predict_affinity: Whether to predict binding affinity
            **kwargs: Additional parameters

        Returns:
            Complex prediction results
        """
        input_data = {
            "sequences": [
                {"type": "protein", "id": protein_id, "sequence": protein_sequence},
                {"type": "ligand", "id": ligand_id, "smiles": ligand_smiles},
            ]
        }

        if predict_affinity:
            input_data["properties"] = [{"affinity": {"binder": ligand_id}}]

        task = "affinity_prediction" if predict_affinity else "complex_prediction"
        return self.predict(input_data, task=task, **kwargs)

    def predict_affinity_only(
        self, protein_sequence: str, ligand_smiles: str, **kwargs
    ) -> Dict[str, Any]:
        """
        Predict binding affinity without full structure prediction.

        Args:
            protein_sequence: Protein amino acid sequence
            ligand_smiles: Ligand SMILES string
            **kwargs: Additional parameters

        Returns:
            Affinity prediction results
        """
        result = self.predict_complex(
            protein_sequence, ligand_smiles, predict_affinity=True, **kwargs
        )

        # Extract just affinity information
        affinity_results = {}
        for key, value in result.get("affinity", {}).items():
            affinity_results[key] = value

        return {
            "affinity_predictions": affinity_results,
            "task": "affinity_prediction",
            "status": result.get("status", "completed"),
        }

    def batch_predict(
        self,
        input_list: List[Dict[str, Any]],
        task: str = "structure_prediction",
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Run batch predictions on multiple inputs.

        Args:
            input_list: List of input data dictionaries
            task: Prediction task type
            **kwargs: Additional parameters

        Returns:
            List of prediction results
        """
        results = []

        for i, input_data in enumerate(input_list):
            try:
                print(f"Processing batch item {i+1}/{len(input_list)}")
                result = self.predict(input_data, task=task, **kwargs)
                results.append(result)
            except Exception as e:
                print(f"Failed to process batch item {i+1}: {e}")
                results.append({"status": "failed", "error": str(e), "input_index": i})

        return results

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the Boltz model."""
        return {
            "model_name": "Boltz",
            "version": self.model_version,
            "repository": "https://github.com/jwohlwend/boltz",
            "papers": [
                "Boltz-1: Democratizing Biomolecular Interaction Modeling (2024)",
                "Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction (2025)",
            ],
            "capabilities": [
                "Protein structure prediction",
                "Protein-ligand complex prediction",
                "Binding affinity prediction",
                "Multi-chain complex modeling",
                "MSA-based folding",
                "Template-based modeling",
            ],
            "supported_inputs": [
                "Protein sequences",
                "DNA/RNA sequences",
                "Ligand SMILES",
                "CCD codes",
                "Multiple sequence alignments",
            ],
            "output_formats": ["CIF", "PDB"],
            "installed": self.boltz_installed,
            "cache_dir": str(self.cache_dir),
        }


# Convenience wrapper for easy integration
class BoltzModel(ExternalModelWrapper):
    """
    High-level wrapper for Boltz model integration into QeMLflow workflows.
    """

    def __init__(self, **kwargs):
        """Initialize Boltz model wrapper."""
        super().__init__(
            repo_url="https://github.com/jwohlwend/boltz", model_name="Boltz", **kwargs
        )

        self.adapter = BoltzAdapter(**kwargs)
        self.model_info = self.adapter.get_model_info()

    def predict(self, X, task="structure_prediction", **kwargs):
        """
        QeMLflow-compatible predict method.

        Args:
            X: Input data (sequences, SMILES, etc.)
            task: Prediction task
            **kwargs: Additional parameters

        Returns:
            Prediction results
        """
        # Convert QeMLflow inputs to Boltz format
        if isinstance(X, pd.DataFrame):
            input_data = self._dataframe_to_boltz_input(X)
        elif isinstance(X, str):
            # Single sequence input
            input_data = {"sequences": [{"type": "protein", "id": "A", "sequence": X}]}
        elif isinstance(X, dict):
            input_data = X
        else:
            raise ValueError(f"Unsupported input type: {type(X)}")

        return self.adapter.predict(input_data, task=task, **kwargs)

    def _dataframe_to_boltz_input(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Convert pandas DataFrame to Boltz input format."""
        sequences = []

        for idx, row in df.iterrows():
            if "sequence" in row and pd.notna(row["sequence"]):
                seq_data = {
                    "type": "protein",
                    "id": row.get("chain_id", f"chain_{idx}"),
                    "sequence": row["sequence"],
                }
                sequences.append(seq_data)

            elif "smiles" in row and pd.notna(row["smiles"]):
                seq_data = {
                    "type": "ligand",
                    "id": row.get("chain_id", f"ligand_{idx}"),
                    "smiles": row["smiles"],
                }
                sequences.append(seq_data)

        return {"sequences": sequences}

    def fit(self, X, y=None):
        """Boltz is a pre-trained model, no fitting required."""
        print("Boltz is a pre-trained model. No training required.")
        return self

    def score(self, X, y=None):
        """Return model confidence scores."""
        results = self.predict(X)

        # Extract confidence scores
        confidences = []
        for pred_key, conf_data in results.get("confidence", {}).items():
            confidences.append(conf_data.get("confidence_score", 0.0))

        return np.mean(confidences) if confidences else 0.0
