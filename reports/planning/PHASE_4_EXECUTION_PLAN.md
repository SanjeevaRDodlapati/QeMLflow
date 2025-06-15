# Phase 4: Legacy Architecture Consolidation - Execution Plan

## 🎯 Current Status
- ✅ Phase 3: Monster File Split Complete (all tests passing: 25/25)
- ✅ New modular drug_discovery structure functional
- ✅ ADMET return type tests fixed
- ✅ TOML syntax error resolved

## 🔧 Phase 4 Implementation Tasks

### Task 1: Import Pattern Migration (CRITICAL)
**Priority**: HIGH - Multiple files have old import patterns

#### Files requiring import updates:

**Core Module Updates:**
- `src/chemml/core/data.py:547` - Update property prediction import

**Test Files (Legacy Import Patterns):**
- `tests/integration/test_pipelines.py` - Lines 22, 140
- `tests/legacy/test_integration_quick.py` - Lines 11, 34, 63
- `tests/unit/test_admet_prediction.py` - Lines 14, 169
- `tests/unit/test_property_prediction_comprehensive.py` - Line 1003
- `tests/unit/test_virtual_screening_comprehensive.py` - Line 17
- `tests/unit/test_qsar_modeling_comprehensive.py` - Multiple lines (46, 54, 62, 72, 94, 121, 140, 156, 168, 195)

### Task 2: Validation & Testing
- Update and run all affected tests
- Verify import compatibility
- Performance validation

### Task 3: Documentation Updates
- Update import examples in documentation
- Create migration guide for users
- Update API reference

## 🚀 Implementation Strategy

### Step 1: Core Module Import Fix
Update `src/chemml/core/data.py` to use new modular imports

### Step 2: Test Suite Migration
Systematically update all test files with legacy imports

### Step 3: Validation
Run comprehensive test suite to ensure all changes work

### Step 4: Documentation
Update documentation with new import patterns

## 📋 Success Criteria
- [ ] All imports use new modular structure
- [ ] All tests pass with new imports
- [ ] No performance degradation
- [ ] Documentation updated
- [ ] Migration guide created

## 🔄 Next Actions
1. Start with core module import fix
2. Update test files systematically
3. Run validation suite
4. Update documentation

---
**Estimated Duration**: 1-2 hours
**Risk Level**: Low (isolated changes, backward compatibility maintained)
