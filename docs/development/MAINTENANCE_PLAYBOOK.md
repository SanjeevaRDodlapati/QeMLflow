# ChemML Maintenance Playbook

**Version**: 1.0  
**Last Updated**: June 16, 2025  
**Purpose**: Step-by-step procedures for maintaining ChemML codebase health

---

## 🚨 Emergency Response Procedures

### Critical Health Score (<50)
**Immediate Actions (within 2 hours):**

1. **Assess Scope**:
   ```bash
   python tools/monitoring/health_monitor.py
   python tools/linting/comprehensive_linter.py --quiet
   ```

2. **Auto-fix Critical Issues**:
   ```bash
   python tools/linting/comprehensive_linter.py --auto-fix
   python tools/linting/critical_fixes.py --severity=error
   ```

3. **Validate Fix**:
   ```bash
   scripts/quick_validate.sh
   python tools/linting/health_tracker.py --report
   ```

4. **If Health Score Still <50**:
   - Alert team lead immediately
   - Consider rolling back recent changes
   - Schedule emergency maintenance session

### Test Collection Failures (>10% failure rate)
**Immediate Actions:**

1. **Identify Failed Tests**:
   ```bash
   pytest --collect-only -v > test_collection.log 2>&1
   grep -E "ERROR|FAILED" test_collection.log
   ```

2. **Fix Import Issues**:
   ```bash
   python tools/linting/syntax_fixer.py --target=tests/
   python tools/linting/incremental_refactor.py --scope=imports
   ```

3. **Validate Collections**:
   ```bash
   pytest --collect-only --tb=no
   ```

### Build/CI Failures
**Immediate Actions:**

1. **Check Build Logs**:
   - Review GitHub Actions workflow logs
   - Identify specific failure point

2. **Local Reproduction**:
   ```bash
   scripts/full_validate.sh
   ```

3. **Common Fixes**:
   - Dependency conflicts: `pip install -r requirements.txt --force-reinstall`
   - Import errors: `python tools/linting/syntax_fixer.py`
   - Test failures: `pytest tests/failing_module/ -v`

---

## 📅 Routine Maintenance Procedures

### Daily (Automated via CI)
- Health score monitoring
- Security dependency scans
- Quick validation tests

### Weekly (Manual)
**Every Monday Morning (30 minutes):**

1. **Health Assessment**:
   ```bash
   python tools/monitoring/health_monitor.py
   ```

2. **Address Auto-fixable Issues**:
   ```bash
   python tools/linting/comprehensive_linter.py --auto-fix
   git add -A && git commit -m "🔧 Weekly auto-fix maintenance"
   ```

3. **Run Medium Validation**:
   ```bash
   scripts/medium_validate.sh
   ```

4. **Review and Triage Alerts**:
   - Check monitoring alerts
   - Create GitHub issues for non-trivial problems
   - Schedule fixes based on priority

### Monthly (Manual)
**First Friday of Each Month (2 hours):**

1. **Comprehensive Health Review**:
   ```bash
   scripts/full_validate.sh
   python tools/monitoring/health_monitor.py
   ```

2. **Technical Debt Analysis**:
   ```bash
   python tools/linting/comprehensive_linter.py --detailed > monthly_debt_analysis.txt
   ```

3. **Dependency Updates**:
   ```bash
   pip list --outdated
   # Update non-breaking dependencies
   pip install --upgrade $(pip list --outdated --format=freeze | grep -v '^\-e' | cut -d = -f 1)
   scripts/medium_validate.sh  # Validate after updates
   ```

4. **Performance Benchmarking**:
   ```bash
   python tools/analysis/performance_profiler.py --baseline
   ```

5. **Test Coverage Analysis**:
   ```bash
   pytest tests/ --cov=src/chemml --cov-report=html
   # Review htmlcov/index.html for gaps
   ```

### Quarterly (Manual)
**Strategic Review Session (4 hours):**

1. **Architecture Review**:
   - Review module dependencies
   - Identify refactoring opportunities
   - Plan major improvements

2. **Documentation Updates**:
   - Update API documentation
   - Refresh examples and tutorials
   - Update maintenance procedures

3. **Major Dependency Upgrades**:
   - Plan major framework updates (PyTorch, TensorFlow, etc.)
   - Test compatibility in isolated environment
   - Create migration plans

---

## 🔧 Troubleshooting Guide

### Common Issues and Solutions

#### 1. High Technical Debt (>300 hours)
**Symptoms**: Slow development, frequent bugs, difficult maintenance

**Solution Strategy**:
```bash
# Phase 1: Auto-fix (2-4 hours)
python tools/linting/comprehensive_linter.py --auto-fix
python tools/linting/final_polish.py --apply-all

# Phase 2: Targeted fixing (8-16 hours over 2 weeks)
python tools/linting/targeted_fixer.py --focus=complexity
python tools/linting/targeted_fixer.py --focus=imports
python tools/linting/targeted_fixer.py --focus=unused

# Phase 3: Manual review (schedule with team)
# - Complex function refactoring
# - Architecture improvements
# - Code review sessions
```

#### 2. Low Test Coverage (<60%)
**Symptoms**: Frequent regressions, low confidence in changes

**Solution Strategy**:
```bash
# Identify coverage gaps
pytest tests/ --cov=src/chemml --cov-report=html
open htmlcov/index.html

# Add tests for critical modules first
# Priority order: core > integrations > research > advanced

# Template for new tests:
cp tests/unit/test_template.py tests/unit/test_new_module.py
```

#### 3. Slow Import Times (>5 seconds)
**Symptoms**: Slow development cycle, poor user experience

**Investigation**:
```bash
python -c "
import time
start = time.time()
import chemml
print(f'Import time: {time.time() - start:.2f}s')
"

# Profile imports
python -m cProfile -s cumtime -c "import chemml" | head -20
```

**Solutions**:
- Implement lazy loading for heavy modules
- Move expensive imports inside functions
- Use optional dependencies pattern

#### 4. Memory Usage Issues
**Symptoms**: High memory consumption, OOM errors in CI

**Investigation**:
```bash
python -c "
import psutil
import chemml
process = psutil.Process()
print(f'Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB')
"
```

**Solutions**:
- Profile memory usage with memory_profiler
- Implement data streaming for large datasets
- Add memory cleanup in long-running processes

#### 5. CI/CD Pipeline Failures
**Common Causes and Solutions**:

- **Timeout Issues**: Increase timeout limits or optimize slow tests
- **Dependency Conflicts**: Pin problematic dependencies
- **Platform-specific Issues**: Add platform-specific conditions
- **Resource Limits**: Optimize memory/CPU usage in tests

---

## 📊 Health Score Interpretation

### Score Ranges and Actions

#### 90-100: Excellent
- **Status**: Production ready
- **Action**: Maintain current practices
- **Frequency**: Monthly light maintenance

#### 80-89: Good  
- **Status**: Stable with minor issues
- **Action**: Address issues during regular cycles
- **Frequency**: Weekly maintenance

#### 70-79: Fair
- **Status**: Acceptable but needs attention
- **Action**: Schedule focused improvement sprint
- **Frequency**: Bi-weekly maintenance

#### 60-69: Poor
- **Status**: Quality concerns
- **Action**: Immediate attention required
- **Frequency**: Daily monitoring until improved

#### 50-59: Critical
- **Status**: Significant problems
- **Action**: Emergency maintenance mode
- **Frequency**: Continuous attention until >70

#### <50: Crisis
- **Status**: System stability at risk
- **Action**: All hands on deck
- **Frequency**: Immediate intervention required

---

## 🎯 Performance Targets

### Development Metrics
- **Import Time**: <2 seconds for core modules
- **Test Suite**: <15 minutes for full validation
- **CI/CD Pipeline**: <20 minutes end-to-end
- **Memory Usage**: <500MB for typical workflows

### Quality Metrics
- **Health Score**: >80 (target), >70 (minimum)
- **Test Coverage**: >80% (target), >65% (minimum)
- **Technical Debt**: <200 hours (target), <300 hours (acceptable)
- **Critical Issues**: 0 (always)

### Reliability Metrics
- **Test Success Rate**: >98%
- **CI/CD Success Rate**: >95%
- **Documentation Coverage**: >90%
- **Example Success Rate**: 100%

---

## 🚀 Improvement Workflows

### Weekly Improvement Cycle
1. **Monday**: Health assessment and planning
2. **Tuesday-Thursday**: Implementation of fixes
3. **Friday**: Validation and documentation
4. **Weekend**: Automated monitoring and alerts

### Monthly Improvement Sprint
1. **Week 1**: Assessment and planning
2. **Week 2**: Major fixes and improvements
3. **Week 3**: Testing and validation
4. **Week 4**: Documentation and process improvement

### Quarterly Strategic Planning
1. **Month 1**: Current state analysis
2. **Month 2**: Strategy development and planning
3. **Month 3**: Implementation and validation

---

## 📞 Escalation Procedures

### Level 1: Developer Self-Service
- Health score 70-100
- Use maintenance scripts and auto-fix tools
- Follow weekly maintenance procedures

### Level 2: Team Lead Involvement
- Health score 50-69
- Multiple CI failures
- Performance degradation
- **Action**: Alert team lead, schedule team session

### Level 3: Emergency Response
- Health score <50
- Critical functionality broken
- Security vulnerabilities
- **Action**: Immediate team mobilization, consider rollback

### Level 4: Management Escalation
- Extended outages (>4 hours)
- Major architecture problems
- Resource constraints
- **Action**: Management involvement, external support

---

## 📝 Change Management

### Before Making Changes
1. Run `scripts/quick_validate.sh`
2. Check current health score
3. Create feature branch
4. Document expected impact

### During Development
1. Run `scripts/quick_validate.sh` frequently
2. Monitor health score changes
3. Address issues immediately
4. Update tests and documentation

### Before Merging
1. Run `scripts/full_validate.sh`
2. Ensure health score hasn't decreased
3. Review test coverage impact
4. Get code review approval

### After Merging
1. Monitor CI/CD pipeline
2. Check health score within 24 hours
3. Address any regressions immediately
4. Update documentation if needed

---

This playbook should be reviewed and updated quarterly to reflect changing needs and lessons learned.
