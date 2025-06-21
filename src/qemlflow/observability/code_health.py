"""
Code Health Metrics Module

This module provides comprehensive code health tracking including technical debt
analysis, code quality metrics, complexity monitoring, and maintenance scheduling
for enterprise-grade code maintainability.
"""

import ast
import json
import logging
import math
import subprocess
import time
from collections import defaultdict, Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

try:
    import radon.complexity
    import radon.metrics
    from radon.raw import analyze
    from radon.visitors import ComplexityVisitor
except ImportError:
    radon = None

try:
    import pylint.lint
    from pylint.reporters.text import TextReporter
except ImportError:
    pylint = None


@dataclass
class TechnicalDebt:
    """Technical debt item tracking."""
    
    debt_id: str = field(default_factory=lambda: str(uuid4()))
    file_path: str = ""
    line_number: int = 0
    debt_type: str = ""  # code_smell, bug_risk, security_issue, performance, maintainability
    severity: str = "medium"  # low, medium, high, critical
    description: str = ""
    estimated_fix_time_hours: float = 0.0
    
    # Metadata
    detected_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    resolved_at: Optional[str] = None
    status: str = "open"  # open, in_progress, resolved, ignored
    
    # Context
    code_snippet: str = ""
    suggested_fix: str = ""
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def resolve(self):
        """Mark debt as resolved."""
        self.status = "resolved"
        self.resolved_at = datetime.now(timezone.utc).isoformat()


@dataclass
class CodeQualityMetrics:
    """Code quality metrics collection."""
    
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    # File-level metrics
    file_path: str = ""
    lines_of_code: int = 0
    logical_lines: int = 0
    source_lines: int = 0
    comment_lines: int = 0
    blank_lines: int = 0
    
    # Complexity metrics
    cyclomatic_complexity: float = 0.0
    halstead_difficulty: float = 0.0
    halstead_volume: float = 0.0
    maintainability_index: float = 0.0
    
    # Quality scores
    pylint_score: float = 0.0
    code_coverage: float = 0.0
    test_coverage: float = 0.0
    
    # Code smells
    code_smells: List[str] = field(default_factory=list)
    duplicated_lines: int = 0
    
    # Dependencies
    import_count: int = 0
    external_dependencies: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ComplexityMetrics:
    """Code complexity metrics."""
    
    file_path: str = ""
    function_name: str = ""
    class_name: str = ""
    
    # Complexity measures
    cyclomatic_complexity: int = 0
    cognitive_complexity: int = 0
    nesting_depth: int = 0
    
    # Size metrics
    lines_of_code: int = 0
    parameter_count: int = 0
    local_variables: int = 0
    
    # Quality indicators
    complexity_rank: str = "A"  # A, B, C, D, E, F
    maintainability_rating: str = "good"  # excellent, good, fair, poor
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class MaintenanceTask:
    """Scheduled maintenance task."""
    
    task_id: str = field(default_factory=lambda: str(uuid4()))
    task_type: str = ""  # dependency_update, security_patch, refactoring, cleanup
    priority: str = "medium"  # low, medium, high, critical
    
    title: str = ""
    description: str = ""
    estimated_duration: str = ""  # "2h", "1d", "1w"
    
    # Scheduling
    scheduled_date: str = ""
    due_date: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = None
    
    # Status
    status: str = "scheduled"  # scheduled, in_progress, completed, cancelled
    assigned_to: str = ""
    
    # Context
    affected_files: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def complete(self):
        """Mark task as completed."""
        self.status = "completed"
        self.completed_at = datetime.now(timezone.utc).isoformat()


class TechnicalDebtAnalyzer:
    """Analyzes and tracks technical debt."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.logger = logging.getLogger(__name__)
        
        # Debt patterns
        self.debt_patterns = {
            "TODO": {"type": "maintainability", "severity": "low"},
            "FIXME": {"type": "bug_risk", "severity": "medium"},
            "HACK": {"type": "code_smell", "severity": "medium"},
            "XXX": {"type": "code_smell", "severity": "high"},
            "TEMP": {"type": "maintainability", "severity": "medium"},
            "NOTE": {"type": "maintainability", "severity": "low"}
        }
        
        # Code smell patterns
        self.smell_patterns = [
            ("Long Method", lambda metrics: metrics.lines_of_code > 50),
            ("Large Class", lambda metrics: metrics.lines_of_code > 500),
            ("Long Parameter List", lambda metrics: getattr(metrics, 'parameter_count', 0) > 5),
            ("Complex Method", lambda metrics: metrics.cyclomatic_complexity > 10),
            ("High Coupling", lambda metrics: metrics.import_count > 20),
            ("Low Cohesion", lambda metrics: metrics.maintainability_index < 20)
        ]
    
    def analyze_file(self, file_path: Path) -> List[TechnicalDebt]:
        """Analyze a single file for technical debt."""
        debt_items = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Analyze comment-based debt
            for line_num, line in enumerate(lines, 1):
                for pattern, config in self.debt_patterns.items():
                    if pattern in line.upper():
                        debt = TechnicalDebt(
                            file_path=str(file_path.relative_to(self.project_root)),
                            line_number=line_num,
                            debt_type=config["type"],
                            severity=config["severity"],
                            description=f"{pattern} comment found",
                            code_snippet=line.strip(),
                            estimated_fix_time_hours=self._estimate_fix_time(config["severity"])
                        )
                        debt_items.append(debt)
            
            # Analyze structural debt
            if file_path.suffix == '.py':
                debt_items.extend(self._analyze_python_structure(file_path, lines))
        
        except Exception as e:
            self.logger.error(f"Failed to analyze {file_path}: {e}")
        
        return debt_items
    
    def _analyze_python_structure(self, file_path: Path, lines: List[str]) -> List[TechnicalDebt]:
        """Analyze Python file structure for debt."""
        debt_items = []
        
        try:
            source = ''.join(lines)
            tree = ast.parse(source)
            
            for node in ast.walk(tree):
                # Long functions
                if isinstance(node, ast.FunctionDef):
                    func_lines = (node.end_lineno - node.lineno) if (hasattr(node, 'end_lineno') and node.end_lineno is not None) else 0
                    if func_lines > 50:
                        debt = TechnicalDebt(
                            file_path=str(file_path.relative_to(self.project_root)),
                            line_number=node.lineno,
                            debt_type="code_smell",
                            severity="medium",
                            description=f"Long function '{node.name}' ({func_lines} lines)",
                            estimated_fix_time_hours=2.0,
                            tags=["long_method"]
                        )
                        debt_items.append(debt)
                
                # Many parameters
                if isinstance(node, ast.FunctionDef) and len(node.args.args) > 5:
                    debt = TechnicalDebt(
                        file_path=str(file_path.relative_to(self.project_root)),
                        line_number=node.lineno,
                        debt_type="code_smell",
                        severity="low",
                        description=f"Function '{node.name}' has {len(node.args.args)} parameters",
                        estimated_fix_time_hours=1.0,
                        tags=["parameter_list"]
                    )
                    debt_items.append(debt)
                
                # Large classes
                if isinstance(node, ast.ClassDef):
                    class_lines = (node.end_lineno - node.lineno) if (hasattr(node, 'end_lineno') and node.end_lineno is not None) else 0
                    if class_lines > 200:
                        debt = TechnicalDebt(
                            file_path=str(file_path.relative_to(self.project_root)),
                            line_number=node.lineno,
                            debt_type="code_smell",
                            severity="high",
                            description=f"Large class '{node.name}' ({class_lines} lines)",
                            estimated_fix_time_hours=4.0,
                            tags=["large_class"]
                        )
                        debt_items.append(debt)
        
        except Exception as e:
            self.logger.error(f"Failed to analyze Python structure in {file_path}: {e}")
        
        return debt_items
    
    def _estimate_fix_time(self, severity: str) -> float:
        """Estimate fix time based on severity."""
        estimates = {
            "low": 0.5,
            "medium": 2.0,
            "high": 4.0,
            "critical": 8.0
        }
        return estimates.get(severity, 2.0)
    
    def analyze_project(self, include_patterns: Optional[List[str]] = None) -> List[TechnicalDebt]:
        """Analyze entire project for technical debt."""
        if include_patterns is None:
            include_patterns = ["**/*.py", "**/*.js", "**/*.ts", "**/*.java", "**/*.cpp"]
        
        debt_items = []
        
        for pattern in include_patterns:
            for file_path in self.project_root.glob(pattern):
                if file_path.is_file() and not self._should_skip_file(file_path):
                    debt_items.extend(self.analyze_file(file_path))
        
        return debt_items
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped."""
        skip_patterns = [
            ".git", "__pycache__", ".pytest_cache", "node_modules",
            ".venv", "venv", "build", "dist", ".tox"
        ]
        
        return any(pattern in str(file_path) for pattern in skip_patterns)


class CodeQualityAnalyzer:
    """Analyzes code quality metrics."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.logger = logging.getLogger(__name__)
    
    def analyze_file(self, file_path: Path) -> CodeQualityMetrics:
        """Analyze code quality for a single file."""
        metrics = CodeQualityMetrics(file_path=str(file_path.relative_to(self.project_root)))
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.splitlines()
            
            # Basic line metrics
            metrics.lines_of_code = len(lines)
            metrics.blank_lines = sum(1 for line in lines if not line.strip())
            metrics.comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
            metrics.source_lines = metrics.lines_of_code - metrics.blank_lines - metrics.comment_lines
            
            # Python-specific analysis
            if file_path.suffix == '.py':
                self._analyze_python_quality(file_path, content, metrics)
        
        except Exception as e:
            self.logger.error(f"Failed to analyze {file_path}: {e}")
        
        return metrics
    
    def _analyze_python_quality(self, file_path: Path, content: str, metrics: CodeQualityMetrics):
        """Analyze Python-specific quality metrics."""
        try:
            # Radon analysis
            if radon:
                # Raw metrics
                raw_metrics = analyze(content)
                metrics.logical_lines = raw_metrics.lloc
                
                # Complexity analysis
                complexity_visitor = ComplexityVisitor.from_code(content)
                if complexity_visitor.functions:
                    avg_complexity = sum(f.complexity for f in complexity_visitor.functions) / len(complexity_visitor.functions)
                    metrics.cyclomatic_complexity = avg_complexity
                
                # Halstead metrics
                try:
                    halstead = radon.metrics.h_visit(content)
                    if halstead:
                        metrics.halstead_difficulty = halstead.difficulty
                        metrics.halstead_volume = halstead.volume
                        
                        # Calculate maintainability index
                        if metrics.cyclomatic_complexity > 0:
                            metrics.maintainability_index = max(0, (171 - 5.2 * 
                                math.log(metrics.halstead_volume) - 
                                0.23 * metrics.cyclomatic_complexity - 
                                16.2 * math.log(metrics.source_lines)) * 100 / 171)
                except Exception:
                    pass
            
            # Import analysis
            try:
                tree = ast.parse(content)
                imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
                metrics.import_count = len(imports)
                
                # Count external dependencies
                stdlib_modules = {'os', 'sys', 'json', 'time', 'datetime', 'pathlib', 'typing', 're', 'math', 'collections'}
                external_imports = set()
                for node in imports:
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            module = alias.name.split('.')[0]
                            if module not in stdlib_modules:
                                external_imports.add(module)
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        module = node.module.split('.')[0]
                        if module not in stdlib_modules:
                            external_imports.add(module)
                
                metrics.external_dependencies = len(external_imports)
            except Exception:
                pass
            
            # Pylint analysis
            if pylint:
                try:
                    metrics.pylint_score = self._run_pylint(file_path)
                except Exception:
                    pass
        
        except Exception as e:
            self.logger.error(f"Failed Python analysis for {file_path}: {e}")
    
    def _run_pylint(self, file_path: Path) -> float:
        """Run pylint on file and extract score."""
        try:
            # Create a minimal pylint runner
            from io import StringIO
            
            output = StringIO()
            reporter = TextReporter(output)
            
            # Run pylint with minimal configuration
            from pylint.lint import Run
            
            # Suppress pylint output by redirecting stderr
            import sys
            old_stderr = sys.stderr
            sys.stderr = StringIO()
            
            try:
                Run([str(file_path), '--score=y', '--reports=n', '--msg-template='], reporter=reporter, exit=False)
            finally:
                sys.stderr = old_stderr
            
            # Extract score from output
            output_text = output.getvalue()
            for line in output_text.split('\n'):
                if 'Your code has been rated at' in line:
                    score_part = line.split('rated at')[1].split('/')[0].strip()
                    return float(score_part)
            
            return 5.0  # Default neutral score
        except Exception:
            return 5.0


class ComplexityAnalyzer:
    """Analyzes code complexity metrics."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.logger = logging.getLogger(__name__)
    
    def analyze_file(self, file_path: Path) -> List[ComplexityMetrics]:
        """Analyze complexity metrics for a file."""
        complexity_metrics: List[ComplexityMetrics] = []
        
        if file_path.suffix != '.py':
            return complexity_metrics
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Analyze functions and methods
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    metrics = self._analyze_function_complexity(file_path, node, content)
                    complexity_metrics.append(metrics)
        
        except Exception as e:
            self.logger.error(f"Failed to analyze complexity for {file_path}: {e}")
        
        return complexity_metrics
    
    def _analyze_function_complexity(self, file_path: Path, node: ast.FunctionDef, content: str) -> ComplexityMetrics:
        """Analyze complexity of a single function."""
        metrics = ComplexityMetrics(
            file_path=str(file_path.relative_to(self.project_root)),
            function_name=node.name
        )
        
        # Find enclosing class if any
        # This is a simplified approach - in practice you'd need proper scope tracking
        lines = content.splitlines()
        for i in range(node.lineno - 1, -1, -1):
            line = lines[i].strip()
            if line.startswith('class ') and ':' in line:
                class_name = line.split('class ')[1].split('(')[0].split(':')[0].strip()
                metrics.class_name = class_name
                break
        
        # Calculate basic metrics
        if hasattr(node, 'end_lineno'):
            metrics.lines_of_code = (node.end_lineno - node.lineno) if (hasattr(node, 'end_lineno') and node.end_lineno is not None) else 0
        
        metrics.parameter_count = len(node.args.args)
        
        # Calculate cyclomatic complexity
        metrics.cyclomatic_complexity = self._calculate_cyclomatic_complexity(node)
        
        # Calculate cognitive complexity (simplified)
        metrics.cognitive_complexity = self._calculate_cognitive_complexity(node)
        
        # Calculate nesting depth
        metrics.nesting_depth = self._calculate_nesting_depth(node)
        
        # Assign complexity rank
        metrics.complexity_rank = self._assign_complexity_rank(metrics.cyclomatic_complexity)
        
        # Assign maintainability rating
        metrics.maintainability_rating = self._assign_maintainability_rating(metrics)
        
        return metrics
    
    def _calculate_cyclomatic_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity of a function."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.With, ast.AsyncWith)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    def _calculate_cognitive_complexity(self, node: ast.AST) -> int:
        """Calculate cognitive complexity (simplified version)."""
        complexity = 0
        
        def visit_node(n, level):
            nonlocal complexity
            
            if isinstance(n, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1 + level
            elif isinstance(n, ast.ExceptHandler):
                complexity += 1 + level
            elif isinstance(n, (ast.With, ast.AsyncWith)):
                complexity += 1 + level
            elif isinstance(n, ast.BoolOp):
                complexity += len(n.values) - 1
            
            # Increase nesting for certain constructs
            if isinstance(n, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.With, ast.AsyncWith, ast.ExceptHandler)):
                for child in ast.iter_child_nodes(n):
                    visit_node(child, level + 1)
            else:
                for child in ast.iter_child_nodes(n):
                    visit_node(child, level)
        
        for child in ast.iter_child_nodes(node):
            visit_node(child, 0)
        
        return complexity
    
    def _calculate_nesting_depth(self, node: ast.AST) -> int:
        """Calculate maximum nesting depth."""
        max_depth = 0
        
        def visit_node(n, depth):
            nonlocal max_depth
            max_depth = max(max_depth, depth)
            
            if isinstance(n, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.With, ast.AsyncWith, ast.Try)):
                for child in ast.iter_child_nodes(n):
                    visit_node(child, depth + 1)
            else:
                for child in ast.iter_child_nodes(n):
                    visit_node(child, depth)
        
        for child in ast.iter_child_nodes(node):
            visit_node(child, 1)
        
        return max_depth
    
    def _assign_complexity_rank(self, complexity: int) -> str:
        """Assign complexity rank based on cyclomatic complexity."""
        if complexity <= 5:
            return "A"
        elif complexity <= 10:
            return "B"
        elif complexity <= 20:
            return "C"
        elif complexity <= 30:
            return "D"
        elif complexity <= 40:
            return "E"
        else:
            return "F"
    
    def _assign_maintainability_rating(self, metrics: ComplexityMetrics) -> str:
        """Assign maintainability rating based on multiple factors."""
        score = 0
        
        # Complexity factors
        if metrics.cyclomatic_complexity <= 5:
            score += 3
        elif metrics.cyclomatic_complexity <= 10:
            score += 2
        elif metrics.cyclomatic_complexity <= 20:
            score += 1
        
        # Size factors
        if metrics.lines_of_code <= 20:
            score += 2
        elif metrics.lines_of_code <= 50:
            score += 1
        
        # Parameter count
        if metrics.parameter_count <= 3:
            score += 1
        
        # Nesting depth
        if metrics.nesting_depth <= 3:
            score += 1
        
        if score >= 6:
            return "excellent"
        elif score >= 4:
            return "good"
        elif score >= 2:
            return "fair"
        else:
            return "poor"


class MaintenanceScheduler:
    """Schedules and manages maintenance tasks."""
    
    def __init__(self, storage_dir: str = "maintenance"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True, parents=True)
        self.logger = logging.getLogger(__name__)
        
        self.scheduled_tasks: Dict[str, MaintenanceTask] = {}
        self.completed_tasks: List[MaintenanceTask] = []
    
    def schedule_task(self, task: MaintenanceTask) -> str:
        """Schedule a maintenance task."""
        self.scheduled_tasks[task.task_id] = task
        self.logger.info(f"Scheduled maintenance task: {task.title}")
        return task.task_id
    
    def create_automated_tasks(self, debt_items: List[TechnicalDebt]) -> List[str]:
        """Create automated maintenance tasks based on technical debt."""
        task_ids = []
        
        # Group debt by type
        debt_by_type = defaultdict(list)
        for debt in debt_items:
            debt_by_type[debt.debt_type].append(debt)
        
        # Create tasks for high-priority debt
        for debt_type, items in debt_by_type.items():
            high_priority_items = [item for item in items if item.severity in ["high", "critical"]]
            
            if high_priority_items:
                task = MaintenanceTask(
                    task_type="refactoring",
                    priority="high" if any(item.severity == "critical" for item in high_priority_items) else "medium",
                    title=f"Fix {debt_type} issues",
                    description=f"Address {len(high_priority_items)} {debt_type} issues",
                    estimated_duration=f"{sum(item.estimated_fix_time_hours for item in high_priority_items):.1f}h",
                    scheduled_date=(datetime.now() + timedelta(days=7)).isoformat(),
                    due_date=(datetime.now() + timedelta(days=14)).isoformat(),
                    affected_files=list(set(item.file_path for item in high_priority_items)),
                    tags=[debt_type]
                )
                task_ids.append(self.schedule_task(task))
        
        return task_ids
    
    def schedule_dependency_updates(self) -> List[str]:
        """Schedule dependency update tasks."""
        task_ids = []
        
        # Check for outdated dependencies
        try:
            result = subprocess.run(
                ["pip", "list", "--outdated", "--format=json"],
                capture_output=True, text=True, timeout=30, check=False
            )
            
            if result.returncode == 0:
                outdated = json.loads(result.stdout)
                
                if outdated:
                    task = MaintenanceTask(
                        task_type="dependency_update",
                        priority="medium",
                        title="Update outdated dependencies",
                        description=f"Update {len(outdated)} outdated Python packages",
                        estimated_duration="2h",
                        scheduled_date=(datetime.now() + timedelta(days=3)).isoformat(),
                        due_date=(datetime.now() + timedelta(days=10)).isoformat(),
                        dependencies=[pkg["name"] for pkg in outdated],
                        tags=["dependencies", "security"]
                    )
                    task_ids.append(self.schedule_task(task))
        
        except Exception as e:
            self.logger.error(f"Failed to check for outdated dependencies: {e}")
        
        return task_ids
    
    def get_due_tasks(self, days_ahead: int = 7) -> List[MaintenanceTask]:
        """Get tasks due within specified days."""
        cutoff_date = datetime.now() + timedelta(days=days_ahead)
        cutoff_str = cutoff_date.isoformat()
        
        due_tasks = []
        for task in self.scheduled_tasks.values():
            if task.status == "scheduled" and task.due_date <= cutoff_str:
                due_tasks.append(task)
        
        return sorted(due_tasks, key=lambda t: t.due_date)
    
    def complete_task(self, task_id: str) -> bool:
        """Mark a task as completed."""
        if task_id in self.scheduled_tasks:
            task = self.scheduled_tasks.pop(task_id)
            task.complete()
            self.completed_tasks.append(task)
            self.logger.info(f"Completed maintenance task: {task.title}")
            return True
        return False
    
    def get_maintenance_summary(self) -> Dict[str, Any]:
        """Get maintenance summary statistics."""
        return {
            "scheduled_tasks": len(self.scheduled_tasks),
            "completed_tasks": len(self.completed_tasks),
            "due_soon": len(self.get_due_tasks(7)),
            "overdue": len(self.get_due_tasks(-1)),
            "total_estimated_hours": sum(
                float(task.estimated_duration.rstrip('h')) 
                for task in self.scheduled_tasks.values() 
                if task.estimated_duration.endswith('h')
            )
        }


class CodeHealthDashboard:
    """Main dashboard for code health metrics."""
    
    def __init__(self, project_root: str = ".", storage_dir: str = "code_health"):
        self.project_root = Path(project_root)
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True, parents=True)
        
        self.debt_analyzer = TechnicalDebtAnalyzer(project_root)
        self.quality_analyzer = CodeQualityAnalyzer(project_root)
        self.complexity_analyzer = ComplexityAnalyzer(project_root)
        self.maintenance_scheduler = MaintenanceScheduler(str(self.storage_dir / "maintenance"))
        
        self.logger = logging.getLogger(__name__)
    
    def analyze_project_health(self) -> Dict[str, Any]:
        """Perform comprehensive project health analysis."""
        self.logger.info("Starting comprehensive code health analysis...")
        
        start_time = time.time()
        
        # Analyze technical debt
        debt_items = self.debt_analyzer.analyze_project()
        
        # Analyze code quality
        quality_metrics = []
        complexity_metrics = []
        
        python_files = list(self.project_root.glob("**/*.py"))
        for file_path in python_files:
            if not self.debt_analyzer._should_skip_file(file_path):
                quality_metrics.append(self.quality_analyzer.analyze_file(file_path))
                complexity_metrics.extend(self.complexity_analyzer.analyze_file(file_path))
        
        # Schedule maintenance tasks
        scheduled_tasks = self.maintenance_scheduler.create_automated_tasks(debt_items)
        dependency_tasks = self.maintenance_scheduler.schedule_dependency_updates()
        
        analysis_time = time.time() - start_time
        
        # Generate comprehensive report
        report = {
            "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
            "analysis_duration_seconds": analysis_time,
            "project_overview": self._generate_project_overview(quality_metrics),
            "technical_debt": self._generate_debt_summary(debt_items),
            "code_quality": self._generate_quality_summary(quality_metrics),
            "complexity_analysis": self._generate_complexity_summary(complexity_metrics),
            "maintenance_plan": self._generate_maintenance_summary(scheduled_tasks + dependency_tasks),
            "recommendations": self._generate_recommendations(debt_items, quality_metrics, complexity_metrics)
        }
        
        # Save report
        self._save_report(report)
        
        self.logger.info(f"Code health analysis completed in {analysis_time:.2f} seconds")
        
        return report
    
    def _generate_project_overview(self, quality_metrics: List[CodeQualityMetrics]) -> Dict[str, Any]:
        """Generate project overview metrics."""
        if not quality_metrics:
            return {}
        
        total_loc = sum(m.lines_of_code for m in quality_metrics)
        total_source_lines = sum(m.source_lines for m in quality_metrics)
        total_files = len(quality_metrics)
        
        avg_complexity = sum(m.cyclomatic_complexity for m in quality_metrics) / total_files
        avg_maintainability = sum(m.maintainability_index for m in quality_metrics if m.maintainability_index > 0)
        if avg_maintainability > 0:
            avg_maintainability /= sum(1 for m in quality_metrics if m.maintainability_index > 0)
        
        return {
            "total_files": total_files,
            "total_lines_of_code": total_loc,
            "total_source_lines": total_source_lines,
            "average_file_size": total_loc / total_files if total_files > 0 else 0,
            "average_complexity": avg_complexity,
            "average_maintainability_index": avg_maintainability,
            "code_to_comment_ratio": total_source_lines / max(1, total_loc - total_source_lines)
        }
    
    def _generate_debt_summary(self, debt_items: List[TechnicalDebt]) -> Dict[str, Any]:
        """Generate technical debt summary."""
        if not debt_items:
            return {"total_items": 0, "total_estimated_hours": 0}
        
        by_severity = Counter(item.severity for item in debt_items)
        by_type = Counter(item.debt_type for item in debt_items)
        
        total_hours = sum(item.estimated_fix_time_hours for item in debt_items)
        
        return {
            "total_items": len(debt_items),
            "total_estimated_hours": total_hours,
            "by_severity": dict(by_severity),
            "by_type": dict(by_type),
            "critical_items": [item.to_dict() for item in debt_items if item.severity == "critical"],
            "debt_density": len(debt_items) / max(1, len(set(item.file_path for item in debt_items)))
        }
    
    def _generate_quality_summary(self, quality_metrics: List[CodeQualityMetrics]) -> Dict[str, Any]:
        """Generate code quality summary."""
        if not quality_metrics:
            return {}
        
        total_files = len(quality_metrics)
        
        return {
            "total_files_analyzed": total_files,
            "average_pylint_score": sum(m.pylint_score for m in quality_metrics) / total_files,
            "average_maintainability_index": sum(m.maintainability_index for m in quality_metrics if m.maintainability_index > 0) / max(1, sum(1 for m in quality_metrics if m.maintainability_index > 0)),
            "files_with_code_smells": sum(1 for m in quality_metrics if m.code_smells),
            "high_complexity_files": sum(1 for m in quality_metrics if m.cyclomatic_complexity > 10),
            "large_files": sum(1 for m in quality_metrics if m.lines_of_code > 500)
        }
    
    def _generate_complexity_summary(self, complexity_metrics: List[ComplexityMetrics]) -> Dict[str, Any]:
        """Generate complexity analysis summary."""
        if not complexity_metrics:
            return {}
        
        complexity_distribution = Counter(m.complexity_rank for m in complexity_metrics)
        maintainability_distribution = Counter(m.maintainability_rating for m in complexity_metrics)
        
        return {
            "total_functions_analyzed": len(complexity_metrics),
            "complexity_distribution": dict(complexity_distribution),
            "maintainability_distribution": dict(maintainability_distribution),
            "high_complexity_functions": [m.to_dict() for m in complexity_metrics if m.complexity_rank in ["E", "F"]],
            "average_cyclomatic_complexity": sum(m.cyclomatic_complexity for m in complexity_metrics) / len(complexity_metrics),
            "average_cognitive_complexity": sum(m.cognitive_complexity for m in complexity_metrics) / len(complexity_metrics)
        }
    
    def _generate_maintenance_summary(self, task_ids: List[str]) -> Dict[str, Any]:
        """Generate maintenance plan summary."""
        summary = self.maintenance_scheduler.get_maintenance_summary()
        summary["newly_scheduled"] = len(task_ids)
        summary["due_tasks"] = [task.to_dict() for task in self.maintenance_scheduler.get_due_tasks(7)]
        return summary
    
    def _generate_recommendations(self, debt_items: List[TechnicalDebt], 
                                quality_metrics: List[CodeQualityMetrics],
                                complexity_metrics: List[ComplexityMetrics]) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Debt-based recommendations
        critical_debt = [item for item in debt_items if item.severity == "critical"]
        if critical_debt:
            recommendations.append(f"Address {len(critical_debt)} critical technical debt items immediately")
        
        high_debt = [item for item in debt_items if item.severity == "high"]
        if len(high_debt) > 10:
            recommendations.append(f"Plan refactoring sprints to address {len(high_debt)} high-priority debt items")
        
        # Quality-based recommendations
        if quality_metrics:
            low_quality_files = [m for m in quality_metrics if m.pylint_score < 7.0]
            if len(low_quality_files) > len(quality_metrics) * 0.2:
                recommendations.append("Improve code quality - over 20% of files have low quality scores")
            
            large_files = [m for m in quality_metrics if m.lines_of_code > 500]
            if large_files:
                recommendations.append(f"Consider breaking down {len(large_files)} large files")
        
        # Complexity-based recommendations
        if complexity_metrics:
            complex_functions = [m for m in complexity_metrics if m.complexity_rank in ["E", "F"]]
            if complex_functions:
                recommendations.append(f"Refactor {len(complex_functions)} highly complex functions")
            
            poor_maintainability = [m for m in complexity_metrics if m.maintainability_rating == "poor"]
            if len(poor_maintainability) > len(complexity_metrics) * 0.15:
                recommendations.append("Focus on improving maintainability - many functions are hard to maintain")
        
        if not recommendations:
            recommendations.append("Code health looks good! Continue with regular maintenance.")
        
        return recommendations
    
    def _save_report(self, report: Dict[str, Any]):
        """Save the health report to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.storage_dir / f"code_health_report_{timestamp}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        # Also save as latest
        latest_file = self.storage_dir / "latest_report.json"
        with open(latest_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Code health report saved to {report_file}")


# Global dashboard instance
_code_health_dashboard = None

def get_code_health_dashboard(project_root: str = ".", storage_dir: str = "code_health") -> CodeHealthDashboard:
    """Get global code health dashboard instance."""
    global _code_health_dashboard
    if _code_health_dashboard is None:
        _code_health_dashboard = CodeHealthDashboard(project_root, storage_dir)
    return _code_health_dashboard
