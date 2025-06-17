# ⚡ QeMLflow Quick Start Guide

## **🎯 Get Started in 30 Seconds**

### **1. Lightning-Fast Import**
```python
import qemlflow  # Takes < 0.1 seconds!
print(f"QeMLflow {qemlflow.__version__} ready!")
```

### **2. Verify Performance**
```python
import time
start = time.time()
import qemlflow
end = time.time()
print(f"Import time: {end-start:.3f}s")  # Should be < 0.1s
```

### **3. Basic Usage**
```python
# Core functionality available immediately
try:
    # Your chemistry/ML workflow here
    print("QeMLflow is ready for your chemistry workflows!")
except qemlflow.QeMLflowError as e:
    print(f"QeMLflow handled error gracefully: {e}")
```

---

## **🏃‍♂️ Common Workflows**

### **Data Processing Pipeline**
```python
import qemlflow

# Load and validate data
try:
    data = qemlflow.load_data('molecules.csv')
    print("Data loaded successfully!")
except qemlflow.DataError as e:
    print(f"Data issue handled: {e}")
```

### **Feature Engineering**
```python
# Feature calculation (lazy-loaded when needed)
features = qemlflow.calculate_features(molecules)
print(f"Calculated {len(features)} features")
```

### **Model Integration**
```python
# ML model integration with sklearn
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
# QeMLflow features work seamlessly with sklearn
model.fit(features, target_values)
```

---

## **🔧 Troubleshooting**

### **Import Issues**
If imports are slow:
```python
# Check if you have conflicting installations
import sys
print(sys.path)

# Verify QeMLflow installation
import qemlflow
print(qemlflow.__file__)
```

### **Memory Issues**
```python
# QeMLflow handles memory automatically
from qemlflow.utils.edge_case_handler import edge_case_handler

# Automatic memory management for large datasets
memory_config = edge_case_handler.handle_memory_constraints(
    data_size=1000000,
    available_memory=8000000
)
print(memory_config)
```

### **Dependency Issues**
```python
# Check for missing dependencies
from qemlflow.utils.edge_case_handler import edge_case_handler

available, msg = edge_case_handler.handle_missing_dependencies('rdkit')
print(f"RDKit status: {msg}")
```

---

## **🏆 Production Features**

### **Built-in Validation**
```python
from qemlflow.utils.workflow_validator import workflow_validator

# Validate your complete workflow
results = workflow_validator.run_comprehensive_workflow_test()
if results['overall_score'] > 85:
    print("✅ Workflow is production-ready!")
else:
    print("⚠️ Workflow needs optimization")
```

### **Performance Monitoring**
```python
# Built-in performance tracking
import time
start = time.time()

# Your QeMLflow operations
result = qemlflow.some_heavy_operation()

duration = time.time() - start
print(f"Operation completed in {duration:.3f}s")
```

---

## **📈 Next Steps**

1. **Explore Examples**: Check `/examples/` directory
2. **Read Full Documentation**: See `/docs/API_COMPLETE.md`
3. **Run Validation**: Use built-in workflow validators
4. **Monitor Performance**: Leverage built-in profiling

---

## **🆘 Need Help?**

- **API Reference**: `/docs/API_COMPLETE.md`
- **Error Handling**: Built-in exception hierarchy
- **Validation Tools**: Workflow and edge case validators
- **Performance Tips**: Use lazy loading and built-in optimizations

**QeMLflow**: Production-ready chemistry + machine learning
**Performance**: Sub-100ms imports, enterprise-grade reliability
