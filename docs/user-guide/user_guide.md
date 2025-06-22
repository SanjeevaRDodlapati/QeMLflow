# 📚 QeMLflow User Guide

## 🚀 Quick Start (15 minutes to success!)

### **1. Setup (2 minutes)**
```bash
# Clone and setup
git clone <repository>
cd QeMLflow
pip install -r requirements.txt
```

### **2. Launch Interactive Demo (5 minutes)**
```bash
# Main entry point - interactive launcher
python quick_access_demo.py
```

**What you'll see:**
- Browse Day 1-7 bootcamp scripts
- Run any script with one click
- Framework demonstrations
- Code analysis tools

### **3. Run Your First Script (8 minutes)**
```bash
# Option 1: Through interactive demo
python quick_access_demo.py
# Then select: 1 → 1 → y

# Option 2: Direct execution
cd notebooks/quickstart_bootcamp/days/day_01
python day_01_ml_cheminformatics_final.py
```

---

## 🧩 QeMLflow Framework Guide

### **Framework Overview**
QeMLflow provides a unified framework for molecular machine learning with quantum computing integration.

**Core Components:**
- **`qemlflow_common/`** - Main framework package
- **Configuration Management** - Environment-based setup
- **Library Management** - Automatic fallbacks for missing dependencies
- **Assessment Framework** - Progress tracking and reporting

### **Framework Usage**

#### **Basic Usage**
```python
from qemlflow_common.config import QeMLflowConfig
from qemlflow_common.library_manager import LibraryManager
from qemlflow_common.utils import setup_logging

# Initialize framework
config = QeMLflowConfig()
lib_manager = LibraryManager()
logger = setup_logging()

# Check available libraries
missing_libs = lib_manager.get_missing_libraries()
logger.info(f"Missing libraries: {missing_libs}")
```

#### **Advanced Configuration**
```python
# Environment variables for configuration
import os
os.environ['QEMLFLOW_STUDENT_ID'] = 'your_student_id'
os.environ['QEMLFLOW_TRACK'] = 'quantum_ml'
os.environ['QEMLFLOW_OUTPUT_DIR'] = './outputs'

# Use configuration
from qemlflow_common.config import QeMLflowConfig
config = QeMLflowConfig()
```

### **Production Scripts**
All Day 1-7 scripts are production-ready:
- **Non-interactive execution** - No prompts, runs automatically
- **Environment configuration** - Configurable via environment variables
- **Robust error handling** - Graceful degradation when libraries missing
- **Progress tracking** - Detailed logging and progress reports

---

## 📁 File Locations & Navigation

### **Main Entry Points**
```
QeMLflow/
├── README.md                    # 📖 Project overview
├── quick_access_demo.py         # 🚀 MAIN LAUNCHER
├── requirements.txt             # 📦 Dependencies
└── notebooks/quickstart_bootcamp/  # 📚 Learning materials
```

### **Learning Materials**
```
notebooks/quickstart_bootcamp/
├── days/                        # 📅 Day-by-day materials
│   ├── day_01/                 # ML & Cheminformatics
│   ├── day_02/                 # Deep Learning for Molecules
│   ├── day_03/                 # Molecular Docking
│   ├── day_04/                 # Quantum Chemistry
│   ├── day_05/                 # Quantum ML Integration
│   ├── day_06/                 # Quantum Computing
│   └── day_07/                 # End-to-End Integration
└── qemlflow_common/              # 🧩 Framework (local copy)
```

### **Core Framework**
```
qemlflow_common/
├── config.py                   # Configuration management
├── library_manager.py          # Library detection & fallbacks
├── utils.py                    # Common utilities
└── assessment.py               # Progress tracking
```

### **Development & Support**
```
tools/                          # 🔧 Development tools
├── analysis/                   # Code analysis tools
├── diagnostics/                # Diagnostic scripts
├── development/                # Development utilities
└── legacy_fixes/               # Legacy fix scripts

tests/                          # 🧪 Test suite
├── integration/                # Integration tests
├── unit/                       # Unit tests
└── legacy/                     # Legacy test files

archive/                        # 📂 Development history
└── development/                # Development documentation

logs/                           # 📋 Execution logs & outputs
├── outputs/                    # Script output files
├── cache/                      # Cache directories
└── development_artifacts/      # Coverage & build files
```

---

## 🎯 Usage Examples

### **For Learning (Systematic Approach)**
```bash
# Start with Day 1
cd notebooks/quickstart_bootcamp/days/day_01
python day_01_ml_cheminformatics_final.py

# Progress through days
cd ../day_02
python day_02_deep_learning_molecules_final.py

# Continue through Day 7...
```

### **For Quick Exploration**
```bash
# Use interactive launcher
python quick_access_demo.py

# Browse and run any day script
# Framework handles all dependencies automatically
```

### **For Development**
```python
# Import and use framework components
from qemlflow_common import QeMLflowConfig, LibraryManager

# Create custom applications
config = QeMLflowConfig()
lib_manager = LibraryManager()

# Your custom molecular ML code here...
```

### **For Production Deployment**
```bash
# Environment-based configuration
export QEMLFLOW_STUDENT_ID="production_user"
export QEMLFLOW_TRACK="quantum_ml"
export QEMLFLOW_OUTPUT_DIR="/data/outputs"

# Run scripts in production
python notebooks/quickstart_bootcamp/days/day_05/day_05_quantum_ml_final.py
```

---

## 🔧 Configuration Options

### **Environment Variables**
```bash
# Student identification
export QEMLFLOW_STUDENT_ID="your_id"              # Default: auto-generated

# Learning track selection
export QEMLFLOW_TRACK="quantum_ml"                 # Options: standard, quantum_ml, etc.

# Output configuration
export QEMLFLOW_OUTPUT_DIR="./custom_outputs"     # Default: ./day_XX_outputs

# Library preferences
export QEMLFLOW_PREFER_CPU="true"                 # Force CPU execution
export QEMLFLOW_DISABLE_GPU="true"                # Disable GPU libraries
```

### **Framework Configuration**
```python
# In your code
from qemlflow_common.config import QeMLflowConfig

config = QeMLflowConfig(
    student_id="custom_id",
    track="quantum_ml",
    output_dir="./results",
    enable_gpu=False
)
```

---

## ❓ Troubleshooting

### **Common Issues**

#### **Missing Libraries**
```bash
# Check what's missing
python -c "from qemlflow_common import LibraryManager; print(LibraryManager().get_missing_libraries())"

# Install missing dependencies
pip install rdkit-pypi torch deepchem
```

#### **Import Errors**
```python
# If importing from wrong location, use:
import sys
sys.path.append('notebooks/quickstart_bootcamp')
from qemlflow_common import LibraryManager
```

#### **Script Won't Run**
```bash
# Check from correct directory
cd notebooks/quickstart_bootcamp
python days/day_01/day_01_ml_cheminformatics_final.py

# Or use the launcher
cd ../../  # Back to main directory
python quick_access_demo.py
```

### **Getting Help**
1. **Check logs** - All scripts generate detailed logs
2. **Use interactive demo** - `python quick_access_demo.py`
3. **Check framework status** - LibraryManager shows what's available
4. **Review examples** - All scripts include comprehensive examples

---

## 🎯 Next Steps

### **After Quick Start**
1. **Complete all 7 days** systematically
2. **Explore framework components** for custom development
3. **Try quantum computing examples** in Days 4-6
4. **Build production pipelines** using Day 7 integration

### **For Advanced Users**
1. **Customize framework** for specific needs
2. **Add new assessment methods** to the framework
3. **Integrate with production systems** using environment configuration
4. **Contribute improvements** to the codebase

---

**🎉 You're ready to start your QeMLflow journey!**

Run `python quick_access_demo.py` to begin exploring.
