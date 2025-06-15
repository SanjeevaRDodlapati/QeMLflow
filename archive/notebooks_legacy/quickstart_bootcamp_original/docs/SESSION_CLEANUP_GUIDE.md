# ChemML Bootcamp Session Cleanup Guide
## Fresh Session Management & Error Prevention

### 🎯 **Purpose**
This guide helps maintain a clean, error-free environment when starting fresh ChemML Bootcamp sessions, especially after long-running sessions that may have accumulated state, memory issues, or conflicting imports.

---

## 🧹 **Pre-Session Cleanup Checklist**

### **1. Jupyter Kernel & Memory Cleanup**
```bash
# Restart Jupyter kernel completely
# In Jupyter: Kernel → Restart & Clear Output

# Or restart from terminal:
jupyter notebook stop
jupyter lab stop
# Wait 5 seconds, then restart
jupyter lab
```

### **2. Python Environment Reset**
```bash
# Deactivate and reactivate conda/virtual environment
conda deactivate
conda activate chemml_env

# Or for venv:
deactivate
source chemml_env/bin/activate
```

### **3. Clear Python Cache & Bytecode**
```bash
# Navigate to ChemML directory
cd /Users/sanjeevadodlapati/Downloads/Repos/ChemML

# Remove Python cache files
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -name "*.pyc" -delete
find . -name "*.pyo" -delete

# Clear specific bootcamp cache
rm -rf notebooks/quickstart_bootcamp/__pycache__/
rm -rf notebooks/quickstart_bootcamp/assessment/__pycache__/
```

### **4. Clear Temporary Files**
```bash
# Remove temporary assessment data
rm -f notebooks/quickstart_bootcamp/assessment/*.tmp
rm -f notebooks/quickstart_bootcamp/assessment/temp_*
rm -f notebooks/quickstart_bootcamp/*.tmp

# Remove any lock files
find . -name "*.lock" -delete
find . -name ".ipynb_checkpoints" -exec rm -rf {} +
```

---

## 🚨 **Common Long-Session Errors & Solutions**

### **Error 1: Import Conflicts**
**Symptoms:**
- `ModuleNotFoundError` for previously working imports
- `ImportError: cannot import name 'X' from 'Y'`
- Conflicting library versions

**Solution:**
```python
# Add to first cell of any notebook session:
import sys
import importlib

# Clear specific modules from cache
modules_to_reload = [
    'assessment_framework',
    'deepchem',
    'rdkit',
    'simple_progress_tracker',
    'completion_badges'
]

for module in modules_to_reload:
    if module in sys.modules:
        importlib.reload(sys.modules[module])

# Force garbage collection
import gc
gc.collect()
```

### **Error 2: Memory Issues**
**Symptoms:**
- Kernel dying unexpectedly
- Slow notebook performance
- "Memory Error" messages

**Solution:**
```python
# Memory cleanup function (add to notebooks)
def cleanup_memory():
    import gc
    import psutil
    import os

    # Clear large variables
    for var in list(globals().keys()):
        if var.startswith('large_') or var.endswith('_dataset'):
            if var in globals():
                del globals()[var]

    # Force garbage collection
    gc.collect()

    # Print memory status
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

# Call at start of each section
cleanup_memory()
```

### **Error 3: Assessment Framework State Issues**
**Symptoms:**
- Assessment widgets not displaying
- Progress tracking errors
- Duplicate assessment instances

**Solution:**
```python
# Reset assessment framework state
def reset_assessment_state():
    # Clear assessment globals
    assessment_vars = [var for var in globals().keys() if 'assessment' in var.lower()]
    for var in assessment_vars:
        if var in globals():
            del globals()[var]

    # Reimport assessment framework
    try:
        from assessment_framework import create_assessment, create_widget, create_dashboard
        print("✅ Assessment framework reset successfully")
    except ImportError:
        print("⚠️ Assessment framework not found - continuing without assessments")

    return True

# Call at start of notebooks
reset_assessment_state()
```

### **Error 4: DeepChem/RDKit State Issues**
**Symptoms:**
- RDKit molecules not displaying
- DeepChem model training failures
- Inconsistent molecular parsing

**Solution:**
```python
# Reset cheminformatics libraries
def reset_cheminformatics():
    import warnings
    warnings.filterwarnings('ignore')

    # Clear RDKit state
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        # Reset RDKit display options
        from rdkit.Chem.Draw import IPythonConsole
        IPythonConsole.ipython_useSVG = True
        print("✅ RDKit reset")
    except ImportError:
        print("⚠️ RDKit not available")

    # Reset DeepChem
    try:
        import deepchem as dc
        # Clear any cached models
        if hasattr(dc, 'clear_cache'):
            dc.clear_cache()
        print("✅ DeepChem reset")
    except ImportError:
        print("⚠️ DeepChem not available")

# Call at start of chemistry notebooks
reset_cheminformatics()
```

---

## 📋 **Fresh Session Startup Sequence**

### **Step 1: Environment Preparation**
```bash
# 1. Terminal cleanup
cd /Users/sanjeevadodlapati/Downloads/Repos/ChemML
find . -name "__pycache__" -exec rm -rf {} +
find . -name "*.pyc" -delete

# 2. Activate environment
conda activate chemml_env  # or source chemml_env/bin/activate

# 3. Verify installation
python -c "import rdkit; import deepchem; print('Libraries OK')"

# 4. Start Jupyter
jupyter lab --no-browser --port=8888
```

### **Step 2: Notebook Initialization Template**
```python
# Add this as first cell in every notebook session:

# =============================================================================
# ChemML Bootcamp - Fresh Session Initialization
# =============================================================================

import sys
import os
import gc
import warnings
warnings.filterwarnings('ignore')

# Memory and cache cleanup
gc.collect()

# Standard imports with error handling
try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    print("✅ Core libraries imported")
except ImportError as e:
    print(f"❌ Core library import error: {e}")

# Cheminformatics imports with error handling
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Draw, AllChem
    from rdkit.Chem.Draw import IPythonConsole
    IPythonConsole.ipython_useSVG = True
    print("✅ RDKit imported and configured")
except ImportError as e:
    print(f"❌ RDKit import error: {e}")

try:
    import deepchem as dc
    print(f"✅ DeepChem v{dc.__version__} imported")
except ImportError as e:
    print(f"❌ DeepChem import error: {e}")

# Assessment framework (optional)
try:
    from assessment_framework import create_assessment, create_widget, create_dashboard
    print("✅ Assessment framework imported")
except ImportError:
    print("⚠️ Assessment framework not available (optional)")

# Set plotting defaults
plt.style.use('default')
sns.set_palette("husl")
if 'get_ipython' in globals():
    get_ipython().run_line_magic('matplotlib', 'inline')

print("\n🚀 Fresh session initialized successfully!")
print(f"Working directory: {os.getcwd()}")
print(f"Python path: {sys.executable}")
print("=" * 50)
```

---

## 🔧 **Specific Cleanup by Notebook Type**

### **Day 1: ML & Cheminformatics**
```python
# Specific cleanup for Day 1 notebooks
def day1_cleanup():
    # Clear molecular data
    mol_vars = ['mol_objects', 'drug_molecules', 'df_properties', 'fingerprint_data']
    for var in mol_vars:
        if var in globals():
            del globals()[var]

    # Clear datasets
    dataset_vars = ['train_dataset', 'valid_dataset', 'test_dataset', 'datasets_dict']
    for var in dataset_vars:
        if var in globals():
            del globals()[var]

    # Clear models
    if 'model' in globals():
        del globals()['model']
    if 'rf_model' in globals():
        del globals()['rf_model']

    gc.collect()
    print("✅ Day 1 variables cleaned")

# Call before starting Day 1 exercises
day1_cleanup()
```

### **Days 5-7: Modular Notebooks**
```python
# Cleanup for modular notebooks
def modular_notebook_cleanup():
    # Clear any cross-module variables
    module_vars = [var for var in globals().keys() if 'module_' in var]
    for var in module_vars:
        if var in globals():
            del globals()[var]

    # Reset progress tracking
    if 'progress_tracker' in globals():
        del globals()['progress_tracker']

    # Clear large arrays/datasets
    large_vars = [var for var in globals().keys()
                  if isinstance(globals().get(var), (np.ndarray, pd.DataFrame))
                  and sys.getsizeof(globals().get(var)) > 1024*1024]  # >1MB

    for var in large_vars:
        del globals()[var]

    gc.collect()
    print("✅ Modular notebook variables cleaned")
```

---

## ⚠️ **Signs You Need a Fresh Session**

### **Performance Indicators:**
- Notebook cells taking >30 seconds to execute simple operations
- Memory usage >2GB for basic operations
- Frequent "Kernel appears to have died" messages
- Import statements failing randomly

### **Assessment Framework Issues:**
- Widgets not displaying properly
- Progress tracking showing incorrect data
- Duplicate assessment instances
- Database lock errors

### **Chemistry Library Issues:**
- Molecular structures not rendering
- RDKit/DeepChem throwing unexpected errors
- Model training failing with cryptic errors
- Inconsistent SMILES parsing

---

## 🎯 **Quick Fresh Session Commands**

### **Complete Reset (Nuclear Option)**
```bash
# Stop all Jupyter processes
pkill -f jupyter
pkill -f python

# Wait 10 seconds
sleep 10

# Clear everything
cd /Users/sanjeevadodlapati/Downloads/Repos/ChemML
find . -name "__pycache__" -exec rm -rf {} +
find . -name "*.pyc" -delete
find . -name ".ipynb_checkpoints" -exec rm -rf {} +

# Restart environment
conda deactivate
conda activate chemml_env

# Launch fresh Jupyter
jupyter lab --no-browser --port=8888
```

### **Quick Memory Reset**
```python
# Add to any notebook cell for quick reset
import gc
import sys

# Clear large variables
for var in list(globals().keys()):
    obj = globals()[var]
    if sys.getsizeof(obj) > 1024*1024:  # >1MB
        del globals()[var]

gc.collect()
print("✅ Memory cleaned")
```

---

## 📝 **Session Management Best Practices**

### **1. Modular Development**
- Start each notebook section with cleanup
- Avoid storing large datasets in global variables
- Use functions to encapsulate operations
- Clear variables after use

### **2. Regular Checkpoints**
- Save progress every 30 minutes
- Export important results to files
- Use `gc.collect()` between major sections
- Monitor memory usage regularly

### **3. Error Prevention**
- Import all libraries at the start
- Use try/except blocks for imports
- Check library versions before proceeding
- Validate data shapes and types

### **4. Documentation**
- Keep notes of any custom modifications
- Document successful workarounds
- Track performance issues
- Maintain environment specifications

---

## 🚀 **Emergency Recovery Commands**

If everything fails and you need to start completely fresh:

```bash
# Complete environment reset
conda remove --name chemml_env --all
conda create -n chemml_env python=3.11
conda activate chemml_env

# Reinstall core packages
pip install jupyter jupyterlab
pip install rdkit-pypi deepchem
pip install numpy pandas matplotlib seaborn scikit-learn

# Restart from clean slate
cd /Users/sanjeevadodlapati/Downloads/Repos/ChemML
jupyter lab
```

---

## 📞 **Quick Reference**

| Issue | Quick Fix |
|-------|-----------|
| Import errors | Restart kernel + clear cache |
| Memory issues | `gc.collect()` + clear large variables |
| Widget problems | Reset assessment framework |
| RDKit not displaying | Reset cheminformatics libraries |
| Slow performance | Fresh session restart |
| Assessment duplicates | Clear assessment globals |

---

**💡 Pro Tip:** Bookmark this guide and run the "Fresh Session Startup Sequence" at the beginning of each major bootcamp session to avoid 90% of common errors!

---

*Last updated: Session Management Guide v1.0*
