# 🔴 CRITICAL FILES REGISTRY

## Core Framework Files (Require 2+ Reviewer Approval)

### Framework Entry Points

### Framework Entry Points
- `src/qemlflow/core/__init__.py` - Core module API exports
- `src/qemlflow/core/common/__init__.py` - Core framework component
- `src/qemlflow/core/monitoring/__init__.py` - Core framework component
- `src/qemlflow/core/preprocessing/__init__.py` - Core framework component
- `src/qemlflow/core/utils/__init__.py` - Core framework component

### Core Modules
- `.gitignore` - Version control exclusion rules
- `src/qemlflow/__init__.py` - Framework entry point and API exports
- `src/qemlflow/core/common/config.py` - Core framework component
- `src/qemlflow/core/common/errors.py` - Core framework component
- `src/qemlflow/core/common/performance.py` - Core framework component
- `src/qemlflow/core/data.py` - Data handling and I/O operations
- `src/qemlflow/core/data_processing.py` - Core framework component
- `src/qemlflow/core/enhanced_models.py` - Core framework component
- `src/qemlflow/core/ensemble_advanced.py` - Core framework component
- `src/qemlflow/core/evaluation.py` - Model evaluation and metrics
- `src/qemlflow/core/exceptions.py` - Custom exception classes
- `src/qemlflow/core/featurizers.py` - Molecular featurization framework
- `src/qemlflow/core/models.py` - Base model classes and interfaces
- `src/qemlflow/core/monitoring/dashboard.py` - Core framework component
- `src/qemlflow/core/pipeline.py` - Core framework component
- `src/qemlflow/core/preprocessing/feature_extraction.py` - Core framework component
- `src/qemlflow/core/preprocessing/molecular_preprocessing.py` - Core framework component
- `src/qemlflow/core/preprocessing/protein_preparation.py` - Core framework component
- `src/qemlflow/core/recommendations.py` - Core framework component
- `src/qemlflow/core/utils.py` - Core utility functions
- `src/qemlflow/core/utils/io_utils.py` - Core framework component
- `src/qemlflow/core/utils/metrics.py` - Core framework component
- `src/qemlflow/core/utils/ml_utils.py` - Core framework component
- `src/qemlflow/core/utils/molecular_utils.py` - Core framework component
- `src/qemlflow/core/utils/quantum_utils.py` - Core framework component
- `src/qemlflow/core/utils/visualization.py` - Core framework component
- `src/qemlflow/core/workflow_optimizer.py` - Core framework component
- `src/qemlflow/utils/__init__.py` - Core framework component

### Configuration Files
- `src/qemlflow/config/__init__.py` - Configuration management
- `src/qemlflow/config/unified_config.py` - Core framework component

### Build and Setup
- `pyproject.toml` - Project configuration and metadata
- `requirements-core.txt` - Essential dependencies
- `setup.py` - Package installation configuration


## Review Requirements for Core Files
- **2+ reviewer approval** required for all core file changes
- **Comprehensive testing** must pass before merge
- **Rollback plan** must be documented
- **Impact assessment** must be completed
- **Breaking change analysis** required

## Protection Status
- Total Core Files: 38
- Total Middle Layer Files: 70770
- Total Outer Layer Files: 1263
- Last Updated: 2025-06-17 15:14:54

## Emergency Contact
- Core Maintainer: QeMLflow Development Team
- Emergency Contact: qemlflow-emergency@example.com

## Quick Commands
```bash
# Check file classification
python tools/maintenance/file_classifier.py --classify <file>

# Apply protection
python tools/maintenance/file_classifier.py --protect

# Audit permissions
python tools/maintenance/file_classifier.py --audit
```
