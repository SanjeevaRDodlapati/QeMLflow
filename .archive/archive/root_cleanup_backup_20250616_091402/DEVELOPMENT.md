# QeMLflow Development Guide

## Quick Start

The development environment has been automatically configured with enhanced features:

### 🚀 Quick Commands

```bash
# Show performance dashboard
python quick_dev.py dashboard

# Demo model recommendation
python quick_dev.py recommend

# Generate API documentation
python quick_dev.py docs

# Run tests
python quick_dev.py tests

# Launch Jupyter Lab
python quick_dev.py notebook
```

### 📊 Performance Monitoring

```python
from qemlflow.core.monitoring import show_performance_dashboard
show_performance_dashboard()
```

### 🤖 Model Recommendations

```python
from qemlflow.core.recommendations import recommend_model

# Get model recommendation
recommendation = recommend_model(
    molecular_data=["CCO", "CCC", "c1ccccc1"],
    target_property="logP",
    computational_budget="medium"
)

print(recommendation['recommended_model'])
```

### 📚 API Documentation

Auto-generated API documentation is available at: `docs/api_auto/index.html`

### 🧪 Testing

```bash
pytest tests/ -v
```

### 🔧 Development Features

- **Performance Monitoring**: Real-time performance tracking and optimization suggestions
- **Model Recommendations**: AI-powered model selection based on data characteristics
- **Auto-Documentation**: Automatically generated API documentation with examples
- **Quick Setup**: One-command environment setup

## Enhanced Workflow

1. **Develop**: Write your code with automatic performance monitoring
2. **Test**: Use `python quick_dev.py tests` to run comprehensive tests
3. **Document**: API docs are auto-generated with examples
4. **Optimize**: Use performance dashboard to identify bottlenecks
5. **Deploy**: Enhanced model recommendation helps users choose optimal approaches

Happy coding! 🧬✨
