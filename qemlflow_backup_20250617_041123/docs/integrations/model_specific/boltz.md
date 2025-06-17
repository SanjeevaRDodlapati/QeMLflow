# Boltz Integration Guide

**Complete guide for integrating the Boltz biomolecular interaction model into ChemML**

---

## 🔬 Overview

Boltz is a state-of-the-art deep learning model for predicting biomolecular structures and interactions. This guide covers its integration into ChemML for protein structure prediction, protein-ligand docking, and binding affinity prediction.

## 🚀 Quick Start

### Basic Usage

```python
from chemml.integrations import get_manager

# Get integration manager
manager = get_manager()

# Create Boltz adapter
boltz = manager.get_adapter("boltz")

# Predict protein structure
structure = boltz.predict_structure(protein_sequence)

# Protein-ligand docking
docking_result = boltz.dock_ligand(protein_pdb, ligand_smiles)
```

### Installation Requirements

```bash
# Install Boltz dependencies
pip install torch torchvision torchaudio
pip install biopython rdkit

# Install Boltz (if available)
pip install boltz-pytorch  # or clone from repository
```

## 📋 Supported Tasks

### 1. Protein Structure Prediction

```python
# Single protein prediction
protein_seq = "MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGE..."
structure = boltz.predict_structure(protein_seq)

# Save structure
structure.save_pdb("predicted_structure.pdb")
```

### 2. Protein-Ligand Complex Prediction

```python
# Complex prediction
complex_result = boltz.predict_complex(
    protein_sequence=protein_seq,
    ligand_smiles="CCO",  # ethanol
    binding_site_info=site_coords
)

# Get binding affinity
affinity = complex_result.binding_affinity
print(f"Predicted Kd: {affinity:.2f} μM")
```

### 3. Batch Processing

```python
# Batch protein predictions
sequences = ["SEQUENCE1", "SEQUENCE2", "SEQUENCE3"]
results = boltz.batch_predict_structures(sequences, batch_size=4)

# Batch docking
ligands = ["CCO", "CC(C)O", "CCC"]
docking_results = boltz.batch_dock_ligands(protein_pdb, ligands)
```

## ⚙️ Configuration Options

### Model Parameters

```python
boltz_config = {
    "model_type": "boltz-large",  # or "boltz-base"
    "confidence_threshold": 0.8,
    "max_sequence_length": 1000,
    "use_gpu": True,
    "batch_size": 2
}

boltz = manager.get_adapter("boltz", config=boltz_config)
```

### Advanced Settings

```python
# Custom model path
boltz = manager.get_adapter("boltz", model_path="/path/to/custom/model")

# Memory optimization
boltz.set_memory_efficient(True)

# Custom preprocessing
boltz.set_preprocessing_options(
    sequence_padding=True,
    structure_relaxation=True
)
```

## 📊 Performance Monitoring

### Tracking Predictions

```python
from chemml.integrations.core import PerformanceMonitor

monitor = PerformanceMonitor()

with monitor.track_performance():
    structure = boltz.predict_structure(sequence)

stats = monitor.get_stats()
print(f"Prediction time: {stats['prediction_time']:.2f}s")
print(f"Memory usage: {stats['peak_memory_mb']:.1f} MB")
```

### Benchmarking

```python
# Run benchmark suite
benchmark_results = boltz.run_benchmark(
    test_sequences=["SEQ1", "SEQ2"],
    reference_structures=["ref1.pdb", "ref2.pdb"]
)

print(f"Average RMSD: {benchmark_results['avg_rmsd']:.2f} Å")
```

## 🔧 Troubleshooting

### Common Issues

**GPU Memory Errors**
```python
# Reduce batch size
boltz.config.batch_size = 1

# Enable memory efficient mode
boltz.set_memory_efficient(True)
```

**Long Sequences**
```python
# Split long sequences
max_length = boltz.get_max_sequence_length()
if len(sequence) > max_length:
    # Implement sequence splitting logic
    fragments = split_sequence(sequence, max_length)
    results = [boltz.predict_structure(frag) for frag in fragments]
```

**Model Loading Issues**
```python
# Verify model availability
if not boltz.is_model_available():
    print("Boltz model not found. Please install or check path.")

# Use fallback model
boltz = manager.get_adapter("boltz", model_type="boltz-base")
```

## 📈 Advanced Features

### Custom Loss Functions

```python
# Define custom loss for fine-tuning
def custom_structure_loss(predicted, target):
    # Your custom loss implementation
    pass

boltz.set_loss_function(custom_structure_loss)
```

### Integration with Other Tools

```python
# Combine with DeepChem
from chemml.integrations.adapters.molecular import DeepChemIntegration

deepchem = manager.get_adapter("deepchem")

# Predict properties of Boltz-generated structures
structure = boltz.predict_structure(sequence)
properties = deepchem.predict_properties(structure)
```

### Experiment Tracking

```python
from chemml.integrations.utils import ExperimentTracker

tracker = ExperimentTracker(backend="wandb")

# Track Boltz experiments
tracker.start_experiment("boltz_structure_prediction")
tracker.log_parameters(boltz_config)

structure = boltz.predict_structure(sequence)

tracker.log_results({
    "sequence_length": len(sequence),
    "prediction_confidence": structure.confidence,
    "processing_time": structure.metadata["time"]
})
```

## 📚 Examples

### Complete Workflow Example

```python
"""
Complete protein analysis workflow using Boltz
"""
from chemml.integrations import get_manager
from chemml.integrations.utils import ExperimentTracker

def protein_analysis_workflow(protein_sequence, ligand_smiles):
    """Complete protein-ligand analysis using Boltz."""

    # Initialize
    manager = get_manager()
    boltz = manager.get_adapter("boltz")
    tracker = ExperimentTracker()

    # Track experiment
    tracker.start_experiment("protein_ligand_analysis")

    # Step 1: Predict protein structure
    print("Predicting protein structure...")
    structure = boltz.predict_structure(protein_sequence)

    # Step 2: Dock ligand
    print("Docking ligand...")
    docking_result = boltz.dock_ligand(structure, ligand_smiles)

    # Step 3: Analyze results
    results = {
        "structure_confidence": structure.confidence,
        "binding_affinity": docking_result.binding_affinity,
        "binding_site": docking_result.binding_site,
        "interaction_score": docking_result.interaction_score
    }

    # Log results
    tracker.log_results(results)

    return results

# Run workflow
results = protein_analysis_workflow(
    protein_sequence="MKWVTFISLLLLFSSAYSRGVFRRD...",
    ligand_smiles="CCO"
)
```

## 🔗 See Also

- **[Integration Overview](../README.md)** - General integration guide
- **[Examples](../../../examples/integrations/boltz/)** - Working examples
- **[API Reference](../../REFERENCE.md)** - Complete API documentation
- **[Performance Guide](../performance_optimization.md)** - Optimization tips
