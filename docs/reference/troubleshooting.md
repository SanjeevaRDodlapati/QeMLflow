# Troubleshooting Guide

## Overview

This guide addresses common technical issues you may encounter while following the computational drug discovery roadmap. Issues are organized by category for quick reference.

---

## Environment Setup Issues

### Python Installation and Virtual Environments

**Problem**: "Python command not found" or version conflicts
**Solutions**:
```bash
# Check Python installation
python --version
python3 --version

# Create virtual environment (recommended)
python -m venv qemlflow_env
source qemlflow_env/bin/activate  # On macOS/Linux
qemlflow_env\Scripts\activate     # On Windows

# Install required packages
pip install -r requirements.txt
```

**Problem**: Package installation failures
**Solutions**:
1. Update pip: `pip install --upgrade pip`
2. Use conda instead: `conda install package_name`
3. Install from source: `pip install git+https://github.com/repo/package.git`
4. Check compatibility: Ensure Python version matches package requirements

**Problem**: Jupyter notebook not working with virtual environment
**Solution**:
```bash
# Install ipykernel in your virtual environment
pip install ipykernel
python -m ipykernel install --user --name=qemlflow_env
# Select the kernel in Jupyter notebook
```

### RDKit Installation Issues

**Problem**: RDKit installation fails with pip
**Solutions**:
1. Use conda (recommended): `conda install -c conda-forge rdkit`
2. Try pip with specific version: `pip install rdkit-pypi`
3. Use Docker: Pull pre-configured RDKit container
4. Build from source (advanced users)

**Problem**: Import errors with RDKit
**Check**:
```python
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    print("RDKit imported successfully")
except ImportError as e:
    print(f"RDKit import failed: {e}")
```

### GPU/CUDA Issues

**Problem**: PyTorch not using GPU
**Diagnostics**:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
```

**Solutions**:
1. Install GPU-enabled PyTorch: Visit [pytorch.org](https://pytorch.org/) for installation commands
2. Check CUDA drivers: `nvidia-smi` command
3. Verify CUDA compatibility between PyTorch and system CUDA

---

## Data Processing Issues

### File Format Problems

**Problem**: Cannot read molecular files (SDF, MOL, SMILES)
**Solutions**:
```python
# For RDKit
from rdkit import Chem

# Reading SDF files
supplier = Chem.SDMolSupplier('molecules.sdf')
mols = [mol for mol in supplier if mol is not None]

# Reading SMILES
with open('smiles.txt', 'r') as f:
    smiles_list = [line.strip() for line in f]
mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]

# Handle invalid molecules
valid_mols = [mol for mol in mols if mol is not None]
```

**Problem**: Large dataset memory issues
**Solutions**:
1. Process in chunks: Use pandas `chunksize` parameter
2. Use generators instead of loading all data
3. Implement lazy loading with `dask` or similar
4. Use memory mapping for large files

### Database Connection Issues

**Problem**: Cannot connect to ChEMBL or other databases
**Solutions**:
```python
# For ChEMBL web resource client
from chembl_webresource_client.new_client import new_client

# Test connection
try:
    molecule = new_client.molecule
    test_mol = molecule.get('CHEMBL25')
    print("ChEMBL connection successful")
except Exception as e:
    print(f"Connection failed: {e}")
    # Try with different settings or offline data
```

---

## Machine Learning Issues

### Model Training Problems

**Problem**: Model not converging or poor performance
**Diagnostics**:
1. Check data distribution and outliers
2. Verify train/validation/test splits
3. Monitor training curves
4. Check for data leakage

**Solutions**:
```python
# Basic model diagnostics
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

# Plot learning curves
train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10)
)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
plt.plot(train_sizes, np.mean(val_scores, axis=1), label='Validation score')
plt.legend()
plt.show()
```

**Problem**: Overfitting
**Solutions**:
1. Increase training data
2. Add regularization (L1/L2)
3. Use cross-validation
4. Implement early stopping
5. Try ensemble methods

**Problem**: Memory errors during training
**Solutions**:
1. Reduce batch size
2. Use gradient accumulation
3. Mixed precision training
4. Implement data streaming

### Feature Engineering Issues

**Problem**: Molecular descriptors calculation fails
**Solutions**:
```python
from rdkit.Chem import Descriptors
import pandas as pd

def safe_descriptor_calculation(mol, descriptor_func):
    """Safely calculate molecular descriptors"""
    try:
        return descriptor_func(mol)
    except:
        return None

# Calculate descriptors safely
mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
descriptors = []
for mol in mols:
    if mol is not None:
        desc_dict = {}
        for name, func in Descriptors.descList:
            desc_dict[name] = safe_descriptor_calculation(mol, func)
        descriptors.append(desc_dict)

df_descriptors = pd.DataFrame(descriptors)
```

---

## Quantum Computing Issues

### Qiskit Problems

**Problem**: Qiskit installation or import issues
**Solutions**:
```bash
# Install latest Qiskit
pip install qiskit[visualization]

# For IBM Quantum access
pip install qiskit-ibm-provider
```

**Problem**: Quantum circuit execution errors
**Diagnostics**:
```python
from qiskit import QuantumCircuit, execute, Aer

# Test basic circuit
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

# Execute on simulator
backend = Aer.get_backend('qasm_simulator')
job = execute(qc, backend, shots=1024)
result = job.result()
counts = result.get_counts()
print(counts)
```

**Problem**: VQE convergence issues
**Solutions**:
1. Try different optimizers (COBYLA, SPSA, SLSQP)
2. Adjust initial parameters
3. Increase maximum iterations
4. Use noise mitigation techniques

---

## Performance Issues

### Slow Computation

**Problem**: Code running too slowly
**Optimization strategies**:
1. **Profiling**: Use `cProfile` to identify bottlenecks
```python
import cProfile
cProfile.run('your_function()')
```

2. **Vectorization**: Use NumPy operations instead of loops
```python
# Instead of
result = []
for x in data:
    result.append(x ** 2)

# Use
result = np.array(data) ** 2
```

3. **Parallel processing**: Use `multiprocessing` or `joblib`
```python
from joblib import Parallel, delayed

results = Parallel(n_jobs=-1)(
    delayed(your_function)(item) for item in data_list
)
```

### Memory Issues

**Problem**: Out of memory errors
**Solutions**:
1. **Batch processing**: Process data in smaller chunks
2. **Memory monitoring**: Use `memory_profiler`
```python
from memory_profiler import profile

@profile
def your_function():
    # Your code here
    pass
```

3. **Garbage collection**: Explicitly manage memory
```python
import gc
gc.collect()  # Force garbage collection
```

---

## Specific Tool Issues

### Molecular Dynamics (OpenMM)

**Problem**: OpenMM installation or simulation setup
**Solutions**:
```python
# Test OpenMM installation
import simtk.openmm as mm
import simtk.openmm.app as app

# Basic system setup
pdb = app.PDBFile('protein.pdb')
forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
system = forcefield.createSystem(pdb.topology)

# Check for common issues
print(f"Number of particles: {system.getNumParticles()}")
print(f"Number of forces: {system.getNumForces()}")
```

### Docking Software

**Problem**: AutoDock Vina installation or execution
**Solutions**:
1. Download from [official site](http://vina.scripps.edu/)
2. Check PATH variable includes Vina executable
3. Verify input file formats (PDBQT)
4. Use Python wrapper: `pip install vina`

---

## Getting Additional Help

### When to Seek Help
- After trying multiple solutions from this guide
- When encountering unusual error messages
- For domain-specific questions beyond technical issues

### Where to Get Help
1. **Stack Overflow**: Programming and technical issues
2. **GitHub Issues**: Tool-specific problems
3. **Community Forums**: RDKit, Qiskit, PyTorch communities
4. **Academic Support**: Advisors, colleagues, lab mates
5. **Professional Networks**: LinkedIn, ResearchGate connections

### How to Ask for Help Effectively
1. **Provide context**: What you're trying to accomplish
2. **Include error messages**: Copy the full error traceback
3. **Share minimal code**: Reproduce the issue with simple example
4. **Describe attempts**: What you've already tried
5. **Specify environment**: Python version, OS, package versions

### Creating Minimal Reproducible Examples
```python
# Example template for bug reports
import package_name

# Minimal data that reproduces issue
test_data = [...]

# Minimal code that fails
try:
    result = package_name.function(test_data)
except Exception as e:
    print(f"Error: {e}")
    print(f"Package version: {package_name.__version__}")
    print(f"Python version: {sys.version}")
```

---

*This troubleshooting guide is continuously updated. If you encounter issues not covered here, please consider contributing solutions to help the community.*
