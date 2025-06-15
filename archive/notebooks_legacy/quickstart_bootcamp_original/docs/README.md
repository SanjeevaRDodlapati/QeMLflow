# ChemML Bootcamp Documentation

## 📚 **Available Documentation**

### **Essential Guides:**
- **[SESSION_CLEANUP_GUIDE.md](SESSION_CLEANUP_GUIDE.md)** - Critical guide for maintaining clean Jupyter sessions and avoiding common errors from long-running sessions
- **[day_05_modularization_plan.md](day_05_modularization_plan.md)** - Technical documentation on the modular notebook architecture for Days 5-7

### **Quick Reference:**

#### **🧹 Session Cleanup (Most Important)**
Before starting each bootcamp session, especially after breaks or long sessions:
1. Restart Jupyter kernel completely
2. Clear Python cache files (`__pycache__` directories)
3. Reset conda/virtual environment
4. Clear temporary assessment data

#### **📁 Directory Structure**
```
quickstart_bootcamp/
├── days/           # Daily notebooks organized by day
├── assessments/    # Progress tracking tools
├── docs/          # This documentation
├── scripts/       # Shell scripts for maintenance
└── utils/         # Python utilities and frameworks
```

#### **🚀 Quick Start Commands**
```bash
# Clean start
source ../../chemml_env/bin/activate
cd days/day_01
jupyter lab day_01_ml_cheminformatics_project.ipynb
```

---
**💡 Tip**: Always read SESSION_CLEANUP_GUIDE.md if you encounter import errors, memory issues, or assessment framework problems.
