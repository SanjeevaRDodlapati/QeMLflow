# 📖 ChemML API Reference

**Complete API documentation for ChemML Framework and Core Components**

---

## 📚 Table of Contents

- [🧩 Core Framework API](#-core-framework-api)
- [🔧 Configuration Management](#-configuration-management)
- [📦 Library Management](#-library-management)
- [⚙️ Base Runner Classes](#️-base-runner-classes)
- [📊 Assessment Framework](#-assessment-framework)
- [🛠️ Data Processing](#️-data-processing)
- [🤖 Model Classes](#-model-classes)
- [⚛️ Quantum Computing](#️-quantum-computing)
- [💊 Drug Design](#-drug-design)
- [📊 Utilities](#-utilities)

---

## 🧩 Core Framework API

### `chemml_common` Package

The main framework package providing unified infrastructure for all ChemML components.

```python
from chemml_common import (
    ChemMLConfig,
    get_config,
    print_banner,
    BaseRunner,
    SectionRunner,
    LibraryManager,
    AssessmentFramework
)
```

**Package Information:**
- **Version**: 2.0.0
- **Author**: ChemML Enhancement System
- **License**: MIT

---

## 🔧 Configuration Management

### `ChemMLConfig`

Central configuration management for ChemML framework.

```python
from chemml_common.config.environment import ChemMLConfig

# Initialize configuration
config = ChemMLConfig()

# Access configuration properties
student_id = config.student_id
track = config.track
output_dir = config.output_dir
```

**Properties:**
- `student_id: str` - Unique student identifier
- `track: str` - Learning track (quantum_ml, classical_ml, drug_design)
- `output_dir: Path` - Directory for outputs and results
- `data_dir: Path` - Directory for data files
- `cache_dir: Path` - Directory for cached computations
- `log_level: str` - Logging level (DEBUG, INFO, WARNING, ERROR)

**Methods:**
- `get_section_dir(section_name: str) -> Path` - Get section-specific directory
- `get_output_path(filename: str) -> Path` - Get output file path
- `validate_config() -> bool` - Validate configuration settings

### `get_config()`

Convenience function to get current configuration instance.

```python
from chemml_common import get_config

config = get_config()
```

**Returns:** `ChemMLConfig` - Current configuration instance

### `print_banner()`

Display ChemML framework banner with version information.

```python
from chemml_common import print_banner

print_banner()
```

---

## 📦 Library Management

### `LibraryManager`

Manages library dependencies with intelligent fallbacks and optional installations.

```python
from chemml_common.libraries.manager import LibraryManager

# Initialize library manager
lib_manager = LibraryManager()

# Check library availability
available = lib_manager.is_available('rdkit')
missing = lib_manager.get_missing_libraries()

# Get library or fallback
rdkit = lib_manager.get_library('rdkit')
```

**Methods:**

#### `is_available(library_name: str) -> bool`
Check if a library is available for import.

**Parameters:**
- `library_name: str` - Name of library to check

**Returns:** `bool` - True if library is available

#### `get_library(library_name: str) -> Any`
Import and return library, or provide fallback/mock.

**Parameters:**
- `library_name: str` - Name of library to import

**Returns:** `Any` - Imported library or fallback

#### `get_missing_libraries() -> List[str]`
Get list of libraries that are not available.

**Returns:** `List[str]` - Names of missing libraries

#### `install_missing() -> Dict[str, bool]`
Attempt to install missing libraries via pip.

**Returns:** `Dict[str, bool]` - Installation results

#### `get_installation_status() -> Dict[str, Any]`
Get detailed status of all libraries.

**Returns:** `Dict[str, Any]` - Library status information

---

## ⚙️ Base Runner Classes

### `BaseRunner`

Abstract base class for creating modular ChemML scripts.

```python
from chemml_common.core.base_runner import BaseRunner

class MyChemMLScript(BaseRunner):
    def setup(self):
        # Initialize your script
        pass

    def execute(self):
        # Main execution logic
        pass

    def cleanup(self):
        # Cleanup resources
        pass

# Run script
script = MyChemMLScript()
result = script.run()
```

**Abstract Methods:**
- `setup()` - Initialize script resources
- `execute()` - Main execution logic
- `cleanup()` - Clean up resources

**Methods:**
- `run() -> SectionResult` - Execute complete script lifecycle
- `get_config() -> ChemMLConfig` - Get current configuration
- `get_logger() -> logging.Logger` - Get script logger

### `SectionRunner`

Base class for individual script sections with dependency management.

```python
from chemml_common.core.base_runner import SectionRunner

class DataProcessingSection(SectionRunner):
    def check_dependencies(self) -> bool:
        return self.lib_manager.is_available('pandas')

    def execute_section(self) -> Dict[str, Any]:
        # Section logic here
        return {"processed_data": data}

# Execute section
section = DataProcessingSection()
result = section.run()
```

**Abstract Methods:**
- `check_dependencies() -> bool` - Check if section can run
- `execute_section() -> Dict[str, Any]` - Execute section logic

**Properties:**
- `lib_manager: LibraryManager` - Access to library manager
- `config: ChemMLConfig` - Access to configuration
- `logger: logging.Logger` - Section logger

### `SectionResult`

Data class representing the result of section execution.

```python
@dataclass
class SectionResult:
    section_name: str
    success: bool = True
    execution_time: float = 0.0
    outputs: Dict[str, Any] = None
    errors: List[str] = None
    warnings: List[str] = None
    metadata: Dict[str, Any] = None
```

**Attributes:**
- `section_name: str` - Name of executed section
- `success: bool` - Whether execution was successful
- `execution_time: float` - Execution time in seconds
- `outputs: Dict[str, Any]` - Outputs produced by section
- `errors: List[str]` - List of errors encountered
- `warnings: List[str]` - List of warnings generated
- `metadata: Dict[str, Any]` - Additional execution metadata

---

## 📊 Assessment Framework

### `AssessmentFramework`

Comprehensive assessment and progress tracking system.

```python
from chemml_common.assessment.framework import AssessmentFramework

# Initialize assessment
assessment = AssessmentFramework()

# Track section execution
assessment.start_section("data_processing")
# ... execute section ...
assessment.complete_section("data_processing", outputs)

# Generate reports
summary = assessment.generate_summary()
report = assessment.generate_detailed_report()
```

**Methods:**

#### `start_section(section_name: str)`
Begin tracking a new section.

**Parameters:**
- `section_name: str` - Name of section to track

#### `complete_section(section_name: str, outputs: Dict[str, Any])`
Complete section tracking with outputs.

**Parameters:**
- `section_name: str` - Name of completed section
- `outputs: Dict[str, Any]` - Section outputs

#### `add_warning(section_name: str, message: str)`
Add warning to section.

**Parameters:**
- `section_name: str` - Section name
- `message: str` - Warning message

#### `add_error(section_name: str, error: str)`
Add error to section.

**Parameters:**
- `section_name: str` - Section name
- `error: str` - Error message

#### `generate_summary() -> Dict[str, Any]`
Generate execution summary.

**Returns:** `Dict[str, Any]` - Summary information

#### `generate_detailed_report() -> str`
Generate detailed execution report.

**Returns:** `str` - Formatted report

---

## 🛠️ Data Processing

### Molecular Processing

```python
from chemml.data_processing import MolecularProcessor

# Initialize processor
processor = MolecularProcessor(
    standardize=True,
    remove_salts=True,
    max_atoms=1000
)

# Process molecules
molecules = processor.process_smiles(smiles_list)
features = processor.extract_features(molecules, feature_type="morgan")
```

**Methods:**
- `process_smiles(smiles_list: List[str]) -> List[Mol]` - Process SMILES strings
- `process_sdf(file_path: str) -> List[Mol]` - Process SDF files
- `extract_features(molecules: List[Mol], feature_type: str) -> np.ndarray` - Extract molecular features
- `validate_molecules(molecules: List[Mol]) -> Tuple[List[Mol], List[int]]` - Validate molecular structures

### Feature Extraction

```python
from chemml.data_processing import (
    calculate_descriptors,
    generate_fingerprints,
    molecular_descriptors
)

# Calculate molecular descriptors
descriptors = calculate_descriptors(
    molecules,
    descriptor_types=["lipinski", "crippen", "tpsa"]
)

# Generate molecular fingerprints
fingerprints = generate_fingerprints(
    molecules,
    fingerprint_type="morgan",
    radius=2,
    n_bits=2048
)
```

---

## 🤖 Model Classes

### Classical Machine Learning

```python
from chemml.models.classical import (
    RegressionModel,
    ClassificationModel,
    EnsembleModel
)

# Regression for continuous properties
regressor = RegressionModel(algorithm="random_forest")
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)

# Classification for categorical properties
classifier = ClassificationModel(algorithm="svm")
classifier.fit(X_train, y_train)
classes = classifier.predict(X_test)
```

### Deep Learning Models

```python
from chemml.models.deep_learning import (
    MolecularNeuralNetwork,
    GraphNeuralNetwork,
    TransformerModel
)

# Standard neural network
nn_model = MolecularNeuralNetwork(
    hidden_layers=[512, 256, 128],
    dropout_rate=0.2,
    activation="relu"
)

# Graph neural network for molecular graphs
gnn_model = GraphNeuralNetwork(
    node_features=32,
    edge_features=16,
    hidden_dim=128,
    n_layers=3
)
```

### QSAR Models

```python
from chemml.models import QSARModel

# Initialize QSAR model
model = QSARModel(
    algorithm="random_forest",
    cv_folds=5,
    random_state=42
)

# Train and evaluate
model.fit(X_train, y_train)
predictions = model.predict(X_test)
metrics = model.evaluate(X_test, y_test)
```

**Methods:**
- `fit(X, y)` - Train the model
- `predict(X)` - Make predictions
- `predict_proba(X)` - Prediction probabilities
- `evaluate(X, y)` - Model evaluation metrics
- `get_feature_importance()` - Feature importance scores

---

## ⚛️ Quantum Computing

### Quantum Models

```python
from chemml.models.quantum import QuantumMolecularModel

# Initialize quantum model
qml_model = QuantumMolecularModel(
    n_qubits=4,
    circuit_depth=3,
    backend="qiskit_aer"
)

# Build and train quantum circuit
circuit = qml_model.build_circuit(molecular_features)
qml_model.fit(X_train, y_train, epochs=100)
quantum_predictions = qml_model.predict(X_test)
```

**Methods:**
- `build_circuit(features)` - Construct quantum circuit
- `fit(X, y, epochs)` - Train quantum model
- `predict(X)` - Quantum predictions
- `optimize_circuit()` - Circuit optimization
- `get_quantum_metrics()` - Quantum-specific metrics

### Quantum Algorithms

```python
from chemml.quantum.algorithms import VQE, QAOA

# Variational Quantum Eigensolver
vqe = VQE(molecule=h2_molecule, backend="qiskit_aer")
ground_state_energy = vqe.compute_minimum_eigenvalue()

# Quantum Approximate Optimization Algorithm
qaoa = QAOA(
    problem="molecular_design",
    layers=3,
    mixer="X_mixer"
)
optimized_structure = qaoa.optimize(initial_molecule)
```

---

## 💊 Drug Design

### Molecular Generation

```python
from chemml.drug_design.generation import (
    VAE,
    GAN,
    generate_molecules
)

# Variational Autoencoder for molecules
vae = VAE(latent_dim=512, max_length=120)
vae.fit(smiles_data)
new_molecules = vae.generate(n_samples=1000)

# Generate molecules with properties
molecules = generate_molecules(
    target_properties={"logP": (2, 4), "MW": (200, 500)},
    n_molecules=100
)
```

### Property Prediction

```python
from chemml.drug_design.properties import (
    PropertyPredictor,
    predict_admet_properties
)

# Predict ADMET properties
admet_props = predict_admet_properties(molecules)

# Custom property predictor
predictor = PropertyPredictor(property_type="solubility")
solubility = predictor.predict(molecules)
```

---

## 📊 Utilities

### Visualization

```python
from chemml.utils.visualization import (
    plot_molecule,
    plot_results,
    plot_property_distribution
)

# Visualize molecular structure
plot_molecule(molecule, save_path="molecule.png")

# Plot model results
plot_results(y_true, y_pred, metrics=metrics)

# Plot property distributions
plot_property_distribution(properties, property_name="logP")
```

### Metrics

```python
from chemml.utils.metrics import (
    calculate_accuracy,
    calculate_r2_score,
    calculate_molecular_diversity
)

# Calculate prediction accuracy
accuracy = calculate_accuracy(y_true, y_pred)

# Calculate R² score for regression
r2 = calculate_r2_score(y_true, y_pred)

# Calculate molecular diversity
diversity = calculate_molecular_diversity(molecules)
```

### File I/O

```python
from chemml.utils.io import (
    load_sdf,
    save_molecules,
    load_dataset,
    save_results
)

# Load molecular data
molecules = load_sdf("compounds.sdf")

# Save molecules to file
save_molecules(molecules, "output.sdf", format="sdf")

# Load/save datasets
dataset = load_dataset("training_data.csv")
save_results(results, "predictions.csv")
```

---

## 🔧 Configuration Options

### Environment Variables

ChemML supports configuration via environment variables:

```bash
export CHEMML_STUDENT_ID="your_student_id"
export CHEMML_TRACK="quantum_ml"
export CHEMML_OUTPUT_DIR="./outputs"
export CHEMML_LOG_LEVEL="INFO"
export CHEMML_DATA_DIR="./data"
export CHEMML_CACHE_DIR="./cache"
```

### Configuration File

Create `config.yaml` for advanced configuration:

```yaml
chemml:
  data_dir: "./data"
  cache_dir: "./data/cache"
  results_dir: "./data/results"

  molecular:
    max_atoms: 1000
    sanitize_molecules: true
    remove_hydrogens: false

  ml:
    random_seed: 42
    cv_folds: 5
    test_size: 0.2

  quantum:
    backend: "qiskit_aer"
    shots: 1024
    optimization_level: 3
```

---

## 🚀 Usage Examples

### Basic Framework Usage

```python
from chemml_common import (
    ChemMLConfig,
    LibraryManager,
    BaseRunner
)

class MyMolecularMLScript(BaseRunner):
    def setup(self):
        self.lib_manager = LibraryManager()
        self.rdkit = self.lib_manager.get_library('rdkit')

    def execute(self):
        # Your molecular ML logic
        molecules = self.process_molecules()
        model = self.train_model(molecules)
        return {"model": model, "accuracy": 0.95}

    def cleanup(self):
        # Clean up resources
        pass

# Run script
script = MyMolecularMLScript()
result = script.run()
print(f"Success: {result.success}")
print(f"Outputs: {result.outputs}")
```

### Section-Based Development

```python
from chemml_common.core.base_runner import SectionRunner

class DataPreprocessingSection(SectionRunner):
    def check_dependencies(self):
        return all([
            self.lib_manager.is_available('pandas'),
            self.lib_manager.is_available('rdkit')
        ])

    def execute_section(self):
        pd = self.lib_manager.get_library('pandas')
        rdkit = self.lib_manager.get_library('rdkit')

        # Process data
        data = pd.read_csv("molecules.csv")
        processed = self.clean_molecules(data)

        return {"processed_data": processed}

# Execute section
section = DataPreprocessingSection()
result = section.run()
```

---

## 🛠️ Error Handling

### Common Exceptions

```python
from chemml.exceptions import (
    ChemMLError,
    LibraryNotFoundError,
    MolecularProcessingError,
    ModelTrainingError
)

try:
    molecules = processor.process_smiles(smiles_list)
except MolecularProcessingError as e:
    logger.error(f"Failed to process molecules: {e}")

try:
    rdkit = lib_manager.get_library('rdkit')
except LibraryNotFoundError as e:
    logger.warning(f"RDKit not available: {e}")
    # Use fallback or skip section
```

### Error Recovery

```python
from chemml_common.core.base_runner import SectionRunner

class RobustSection(SectionRunner):
    def execute_section(self):
        try:
            return self.main_logic()
        except Exception as e:
            self.logger.error(f"Main logic failed: {e}")
            return self.fallback_logic()

    def fallback_logic(self):
        # Simplified or alternative implementation
        return {"status": "fallback_used"}
```

---

## 📚 Type Hints

ChemML uses comprehensive type hints for better IDE support:

```python
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import numpy as np
from rdkit import Chem

def process_molecules(
    smiles_list: List[str],
    standardize: bool = True,
    max_atoms: Optional[int] = None
) -> Tuple[List[Chem.Mol], List[str]]:
    """
    Process SMILES strings into RDKit molecule objects.

    Args:
        smiles_list: List of SMILES strings
        standardize: Whether to standardize molecules
        max_atoms: Maximum number of atoms allowed

    Returns:
        Tuple of (processed molecules, error messages)
    """
    pass
```

---

## 🔍 Advanced Features

### Custom Backends

```python
from chemml.backends import register_backend

@register_backend("custom_quantum")
class CustomQuantumBackend:
    def execute_circuit(self, circuit):
        # Custom quantum execution
        pass

# Use custom backend
model = QuantumMolecularModel(backend="custom_quantum")
```

### Plugin System

```python
from chemml.plugins import register_plugin

@register_plugin("molecular_transformer")
class MolecularTransformerPlugin:
    def transform(self, molecules):
        # Custom molecular transformation
        pass

# Use plugin
transformer = get_plugin("molecular_transformer")
transformed = transformer.transform(molecules)
```

---

## 📄 Version Information

- **API Version**: 2.0.0
- **Framework Version**: 2.0.0
- **Last Updated**: June 14, 2025
- **Compatibility**: Python 3.8+

---

## 🔗 Related Documentation

- **[User Guide](USER_GUIDE.md)** - Usage instructions and examples
- **[Complete Reference](REFERENCE.md)** - Comprehensive technical documentation
- **[Quick Start](GET_STARTED.md)** - Getting started guide
- **[Learning Paths](LEARNING_PATHS.md)** - Structured learning programs

---

*This API reference is automatically updated with each release. For the latest version, check the [GitHub repository](https://github.com/yourusername/ChemML).*
