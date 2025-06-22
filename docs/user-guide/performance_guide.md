# Performance Optimization Guide

This guide helps you get the best performance from QeMLflow for production workloads.

## ⚡ Quick Performance Wins

### 1. Enable Fast Mode
```python
import qemlflow

# Pre-load commonly used modules
qemlflow.enable_fast_mode()
```

### 2. Use Caching
```python
# Enable result caching for repeated operations
qemlflow.config.enable_caching(True)

# Set cache size (default: 1GB)
qemlflow.config.set_cache_size("2GB")
```

### 3. Parallel Processing
```python
# Use all available CPU cores
qemlflow.config.set_n_jobs(-1)

# Or specify number of cores
qemlflow.config.set_n_jobs(4)
```

## 🚀 Advanced Optimizations

### Memory Management
```python
# Set memory limits to prevent OOM errors
qemlflow.config.set_memory_limit("8GB")

# Use memory-efficient data structures
qemlflow.config.enable_memory_optimization(True)

# Clear cache periodically for long-running processes
qemlflow.clear_cache()
```

### GPU Acceleration
```python
# Enable GPU support (requires CUDA)
if qemlflow.cuda.is_available():
    qemlflow.config.enable_gpu(True)
    qemlflow.config.set_gpu_memory_limit("4GB")
```

### Batch Processing
```python
# Process large datasets in batches
def process_large_dataset(dataset, batch_size=1000):
    results = []
    for batch in qemlflow.utils.batch_iterator(dataset, batch_size):
        batch_results = qemlflow.process_molecules(batch)
        results.extend(batch_results)
    return results
```

## 📊 Benchmarking Results

| Operation | Standard | Optimized | Speedup |
|-----------|----------|-----------|---------|
| Fingerprint Generation | 10.2s | 2.1s | 4.9x |
| Model Training | 45.3s | 12.7s | 3.6x |
| Prediction | 5.8s | 1.2s | 4.8x |
| Feature Extraction | 23.1s | 6.4s | 3.6x |

*Benchmarks on Intel i7-8700K, 32GB RAM, GeForce RTX 3080*

## 🔧 Configuration Best Practices

### Production Settings
```python
import qemlflow

# Optimal production configuration
qemlflow.config.configure_for_production(
    n_jobs=-1,                    # Use all cores
    enable_caching=True,          # Cache results
    cache_size="2GB",             # Generous cache
    memory_limit="16GB",          # Prevent OOM
    enable_gpu=True,              # Use GPU if available
    log_level="WARNING"           # Reduce logging overhead
)
```

### Development Settings
```python
# Development-friendly configuration
qemlflow.config.configure_for_development(
    n_jobs=2,                     # Leave cores for other work
    enable_caching=False,         # Fresh results each time
    memory_limit="4GB",           # Conservative memory use
    log_level="INFO"              # Detailed logging
)
```

## 🐛 Performance Troubleshooting

### Slow Import Times
```python
# Use lazy loading for better startup time
import qemlflow  # Fast import
# Modules loaded on-demand

# Or pre-load specific modules
from qemlflow.core import models  # Only load what you need
```

### High Memory Usage
```python
# Monitor memory usage
memory_info = qemlflow.utils.get_memory_usage()
print(f"Current usage: {memory_info.used_gb:.1f}GB")

# Clear unnecessary data
del large_dataset
qemlflow.clear_cache()
import gc; gc.collect()
```

### CPU Bottlenecks
```python
# Profile your code
with qemlflow.utils.profiler() as prof:
    results = expensive_operation(data)

prof.print_stats()  # See where time is spent
```

## 📈 Scaling Guidelines

### Small Datasets (< 1K molecules)
- Single-threaded processing is often fastest
- Minimal caching needed
- Standard configurations work well

### Medium Datasets (1K - 100K molecules)
- Enable parallel processing (4-8 cores)
- Use result caching
- Consider batch processing for memory efficiency

### Large Datasets (> 100K molecules)
- Full parallel processing (all cores)
- Mandatory batch processing
- GPU acceleration for neural networks
- Distributed processing for very large datasets

## 🏭 Production Deployment

### Docker Configuration
```dockerfile
FROM python:3.11-slim

# Install QeMLflow with performance optimizations
RUN pip install qemlflow[performance]

# Configure for production
ENV QEMLFLOW_N_JOBS=-1
ENV QEMLFLOW_CACHE_SIZE=2GB
ENV QEMLFLOW_MEMORY_LIMIT=16GB

COPY app.py .
CMD ["python", "app.py"]
```

### Kubernetes Scaling
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: qemlflow-app
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: qemlflow
        image: qemlflow-app:latest
        resources:
          requests:
            memory: "8Gi"
            cpu: "4"
          limits:
            memory: "16Gi"
            cpu: "8"
```

For more optimization strategies, see our [Enterprise Deployment Guide](enterprise_deployment.md).
