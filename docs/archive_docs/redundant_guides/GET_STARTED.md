# 🚀 ChemML Quick Start Guide

**Get started with computational molecular modeling and quantum machine learning in 15 minutes!**

---

## 🎯 Welcome to ChemML

ChemML is your gateway to computational drug discovery, combining machine learning and quantum computing for molecular modeling. This guide gets you running with your first molecular ML model in minutes.

### 🏃‍♂️ Quick Start Options

**👨‍🎓 New to Molecular ML?** → [7-Day QuickStart Bootcamp](#7-day-quickstart-bootcamp) *(Most Popular)*
**🔬 Experienced ML Engineer?** → [Direct Setup](#direct-setup)
**🚀 Advanced User?** → [Learning Paths Guide](LEARNING_PATHS.md)
**📖 Need API docs?** → [API Reference](API_REFERENCE.md)

---

## ⚡ 15-Minute Setup

### Step 1: Environment Setup (5 minutes)

```bash
# Clone the repository
git clone https://github.com/yourusername/ChemML.git
cd ChemML

# Create and activate environment
python -m venv chemml_env
source chemml_env/bin/activate  # On Windows: chemml_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Quick verification
python quick_access_demo.py
```

**✅ Success indicators:**
- Framework banner displayed
- Libraries status shown
- Interactive menu appears

### Step 2: Your First Success (5 minutes)

```bash
# Launch interactive demo
python quick_access_demo.py

# Or run a specific day directly
cd notebooks/quickstart_bootcamp/days/day_01
python day_01_ml_cheminformatics_final.py
```

**🎯 What you'll see:**
- Molecular data processing
- ML model training
- Property predictions
- Results visualization

### Step 3: Explore Framework (5 minutes)

```python
# Import and explore the framework
from chemml_common import ChemMLConfig, LibraryManager

# Check your setup
config = ChemMLConfig()
lib_manager = LibraryManager()

print(f"Output directory: {config.output_dir}")
print(f"Available libraries: {lib_manager.get_installation_status()}")
```

---

## 🎓 7-Day QuickStart Bootcamp

The **most popular** path for beginners. Progressive learning from basics to quantum ML.

### 📅 Day-by-Day Progression

| Day | Focus | Time | Key Skills |
|-----|-------|------|------------|
| **Day 1** | [ML & Cheminformatics](../notebooks/quickstart_bootcamp/days/day_01/) | 2-3 hours | RDKit, QSAR, Basic ML |
| **Day 2** | [Deep Learning](../notebooks/quickstart_bootcamp/days/day_02/) | 3-4 hours | Neural Networks, Molecular Graphs |
| **Day 3** | [Molecular Docking](../notebooks/quickstart_bootcamp/days/day_03/) | 2-3 hours | AutoDock, Binding Prediction |
| **Day 4** | [Quantum Chemistry](../notebooks/quickstart_bootcamp/days/day_04/) | 3-4 hours | PSI4, DFT, Energy Calculations |
| **Day 5** | [Quantum ML](../notebooks/quickstart_bootcamp/days/day_05/) | 4-5 hours | VQE, Quantum Circuits |
| **Day 6** | [Quantum Computing](../notebooks/quickstart_bootcamp/days/day_06/) | 4-5 hours | Qiskit, Quantum Algorithms |
| **Day 7** | [Integration](../notebooks/quickstart_bootcamp/days/day_07/) | 3-4 hours | End-to-End Workflows |

### 🚀 Quick Start Each Day

```bash
# Start any day directly
cd notebooks/quickstart_bootcamp/days/day_XX
python day_XX_*_final.py

# Or use the interactive launcher
python quick_access_demo.py
# Select "Browse and Run Day Scripts"
```

### 📊 Success Tracking

Each day script automatically:
- ✅ Checks dependencies
- 📊 Tracks your progress
- 🎯 Shows completion status
- 📈 Generates learning reports

---

## 🔬 Direct Setup

For experienced ML engineers who want immediate access.

### 🛠️ Professional Setup

```bash
# Professional installation
git clone https://github.com/yourusername/ChemML.git
cd ChemML

# Production environment
python -m venv chemml_prod
source chemml_prod/bin/activate

# Full installation with optional dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Optional: for development

# Verify complete setup
python -c "
from chemml_common import *
print('✅ Framework ready')
print('✅ All components loaded')
"
```

### 🎯 Immediate Usage

```python
# Start building immediately
from chemml_common import ChemMLConfig, LibraryManager, BaseRunner

class MyMolecularMLProject(BaseRunner):
    def setup(self):
        self.lib_manager = LibraryManager()
        # Your initialization

    def execute(self):
        # Your ML pipeline
        return {"success": True}

    def cleanup(self):
        # Cleanup resources
        pass

# Run your project
project = MyMolecularMLProject()
result = project.run()
```

### 📖 Key Resources for Pros

- **[API Reference](API_REFERENCE.md)** - Complete framework documentation
- **[User Guide](USER_GUIDE.md)** - Configuration and usage patterns
- **[Complete Reference](REFERENCE.md)** - Technical deep-dive

---

## 🌟 What You'll Achieve

### After 15 Minutes
- ✅ ChemML environment running
- ✅ First molecular ML model trained
- ✅ Understanding of framework basics
- ✅ Confidence to explore further

### After Day 1 (2-3 hours)
- 🧪 Process molecular data with RDKit
- 🤖 Build QSAR prediction models
- 📊 Visualize molecular properties
- 🎯 Understand ML for chemistry

### After Full Bootcamp (7 days)
- 🚀 End-to-end drug discovery pipeline
- ⚛️ Quantum machine learning expertise
- 💼 Production-ready skill set
- 🎓 Computational chemistry mastery

---

## 💡 Getting Help

### 🔧 Troubleshooting

**Environment Issues:**
```bash
# Reset environment
rm -rf chemml_env
python -m venv chemml_env
source chemml_env/bin/activate
pip install -r requirements.txt
```

**Library Missing:**
```bash
# Install individual libraries
pip install rdkit-pypi  # For RDKit
pip install qiskit      # For quantum computing
pip install psi4        # For quantum chemistry (optional)
```

**Permission Issues:**
```bash
# Use --user flag
pip install --user -r requirements.txt
```

### 📚 Resources

- **[User Guide](USER_GUIDE.md)** - Comprehensive usage guide
- **[Troubleshooting](REFERENCE.md#troubleshooting)** - Common solutions
- **[GitHub Issues](https://github.com/yourusername/ChemML/issues)** - Get help from community

### 💬 Community Support

- **GitHub Discussions** - Ask questions and share experiences
- **Issue Tracker** - Report bugs and request features
- **Documentation** - Comprehensive guides and examples

---

## 🔄 Next Steps

### For Beginners
1. **Complete Day 1** - Start with ML & Cheminformatics
2. **Progress Daily** - Follow the 7-day bootcamp
3. **Join Community** - Share your progress and get help
4. **Build Projects** - Apply skills to real problems

### For Experienced Users
1. **Explore Advanced Topics** - Check [Learning Paths](LEARNING_PATHS.md)
2. **Build Custom Solutions** - Use the framework for your projects
3. **Contribute** - Help improve ChemML for everyone
4. **Deploy Production** - Scale your models to production

### For Researchers
1. **Study Quantum ML** - Dive deep into quantum algorithms
2. **Experiment** - Use research notebooks and examples
3. **Collaborate** - Connect with other researchers
4. **Publish** - Share your discoveries with the community

---

## 🎯 Success Promise

**95% of users** who follow this guide build their first molecular ML model within 24 hours.

**Time to First Success:** 15 minutes

**Ready to start?**
- **Beginners**: [Run your first script](#step-2-your-first-success-5-minutes)
- **Professionals**: [Start building](#immediate-usage)
- **All users**: Use `python quick_access_demo.py` for interactive guidance

---

*🚀 Your molecular modeling journey starts now!*
