"""
🔍 Comprehensive Codebase Reality Check
Analyze if we've truly built a complete core framework or just optimized infrastructure.
"""

import ast
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List


class CodebaseRealityCheck:
    """Comprehensive analysis of ChemML's actual core framework completion."""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.analysis = {
            "core_modules": {},
            "research_modules": {},
            "actual_functionality": {},
            "notebooks_vs_framework": {},
            "framework_gaps": [],
            "reality_assessment": {},
        }

    def analyze_comprehensive_framework_status(self):
        """Comprehensive analysis of actual framework completion vs optimization."""
        print("🔍 COMPREHENSIVE CODEBASE REALITY CHECK")
        print("=" * 70)

        # Analyze core framework
        self.analyze_core_framework()

        # Analyze research modules
        self.analyze_research_modules()

        # Analyze notebook dependencies
        self.analyze_notebook_framework_usage()

        # Check actual functionality depth
        self.analyze_functionality_depth()

        # Generate reality assessment
        self.generate_reality_assessment()

        return self.analysis

    def analyze_core_framework(self):
        """Analyze core framework modules and their actual capabilities."""
        core_path = self.project_root / "src" / "chemml" / "core"

        print("🔧 ANALYZING CORE FRAMEWORK...")

        core_modules = {}
        if core_path.exists():
            for py_file in core_path.rglob("*.py"):
                if py_file.name != "__init__.py":
                    module_name = py_file.stem
                    core_modules[module_name] = self.analyze_python_file(py_file)

        self.analysis["core_modules"] = core_modules

        # Print summary
        print(f"   Core modules found: {len(core_modules)}")
        for module, info in core_modules.items():
            print(
                f"   📄 {module}: {info['classes']} classes, {info['functions']} functions, {info['lines']} lines"
            )

    def analyze_research_modules(self):
        """Analyze research modules and their capabilities."""
        research_path = self.project_root / "src" / "chemml" / "research"

        print("\n🔬 ANALYZING RESEARCH MODULES...")

        research_modules = {}
        if research_path.exists():
            for py_file in research_path.rglob("*.py"):
                if py_file.name != "__init__.py":
                    module_name = py_file.stem
                    research_modules[module_name] = self.analyze_python_file(py_file)

        self.analysis["research_modules"] = research_modules

        # Print summary
        print(f"   Research modules found: {len(research_modules)}")
        for module, info in research_modules.items():
            print(
                f"   📄 {module}: {info['classes']} classes, {info['functions']} functions, {info['lines']} lines"
            )

    def analyze_python_file(self, file_path):
        """Analyze a Python file for classes, functions, and complexity."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Parse AST
            try:
                tree = ast.parse(content)
            except SyntaxError:
                return {
                    "classes": 0,
                    "functions": 0,
                    "lines": 0,
                    "error": "syntax_error",
                }

            classes = [
                node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
            ]
            functions = [
                node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
            ]
            lines = len(content.splitlines())

            return {
                "classes": len(classes),
                "functions": len(functions),
                "lines": lines,
                "class_names": [cls.name for cls in classes],
                "function_names": [
                    func.name for func in functions if not func.name.startswith("_")
                ],
            }
        except Exception as e:
            return {"classes": 0, "functions": 0, "lines": 0, "error": str(e)}

    def analyze_notebook_framework_usage(self):
        """Analyze how much notebooks actually use framework vs custom code."""
        notebooks_path = self.project_root / "notebooks"

        print("\n📚 ANALYZING NOTEBOOK FRAMEWORK USAGE...")

        notebook_analysis = {}
        if notebooks_path.exists():
            for notebook in notebooks_path.rglob("*.ipynb"):
                if "INTEGRATED" in notebook.name:
                    # Analyze integrated notebooks
                    usage = self.analyze_notebook_imports(notebook)
                    notebook_analysis[notebook.name] = usage

        self.analysis["notebooks_vs_framework"] = notebook_analysis

        # Print summary
        integrated_count = len(notebook_analysis)
        print(f"   Integrated notebooks found: {integrated_count}")
        for notebook, info in notebook_analysis.items():
            framework_imports = info.get("framework_imports", 0)
            custom_code_lines = info.get("custom_code_estimate", 0)
            print(
                f"   📓 {notebook}: {framework_imports} framework imports, ~{custom_code_lines} custom lines"
            )

    def analyze_notebook_imports(self, notebook_path):
        """Analyze a notebook's imports and framework usage."""
        try:
            import json

            with open(notebook_path, "r", encoding="utf-8") as f:
                notebook = json.load(f)

            framework_imports = 0
            custom_code_lines = 0

            for cell in notebook.get("cells", []):
                if cell.get("cell_type") == "code":
                    source = "".join(cell.get("source", []))

                    # Count framework imports
                    if "from chemml" in source or "import chemml" in source:
                        framework_imports += source.count("from chemml") + source.count(
                            "import chemml"
                        )

                    # Estimate custom code (rough heuristic)
                    lines = source.split("\n")
                    for line in lines:
                        if (
                            line.strip()
                            and not line.strip().startswith("#")
                            and not line.strip().startswith("from chemml")
                            and not line.strip().startswith("import chemml")
                            and "chemml" not in line
                        ):
                            custom_code_lines += 1

            return {
                "framework_imports": framework_imports,
                "custom_code_estimate": custom_code_lines,
            }
        except Exception as e:
            return {"error": str(e)}

    def analyze_functionality_depth(self):
        """Analyze the actual depth and completeness of framework functionality."""
        print("\n🎯 ANALYZING FUNCTIONALITY DEPTH...")

        # Key areas that should be in a complete chemistry ML framework
        expected_capabilities = {
            "molecular_featurization": [
                "morgan_fingerprints",
                "descriptors",
                "graph_features",
            ],
            "data_processing": ["data_loaders", "preprocessing", "splitting"],
            "models": ["ml_models", "deep_learning", "ensemble_methods"],
            "evaluation": ["metrics", "cross_validation", "visualization"],
            "chemistry_specific": ["qsar", "admet", "docking", "quantum"],
            "production_ready": ["apis", "deployment", "monitoring", "pipelines"],
        }

        functionality_coverage = {}

        # Check core modules coverage
        core_modules = self.analysis.get("core_modules", {})
        research_modules = self.analysis.get("research_modules", {})
        all_modules = {**core_modules, **research_modules}

        for capability, requirements in expected_capabilities.items():
            coverage = self.check_capability_coverage(all_modules, requirements)
            functionality_coverage[capability] = coverage

            coverage_percent = (coverage["found"] / max(len(requirements), 1)) * 100
            status = (
                "✅"
                if coverage_percent >= 70
                else "⚠️"
                if coverage_percent >= 40
                else "❌"
            )
            print(
                f"   {status} {capability}: {coverage_percent:.0f}% coverage ({coverage['found']}/{len(requirements)})"
            )

        self.analysis["actual_functionality"] = functionality_coverage

    def check_capability_coverage(self, modules, requirements):
        """Check if requirements are covered by existing modules."""
        found = 0
        details = {}

        for req in requirements:
            found_in = []
            for module_name, module_info in modules.items():
                # Check function names
                if any(
                    req.lower() in func.lower()
                    for func in module_info.get("function_names", [])
                ):
                    found_in.append(module_name)
                # Check class names
                if any(
                    req.lower() in cls.lower()
                    for cls in module_info.get("class_names", [])
                ):
                    found_in.append(module_name)

            if found_in:
                found += 1
                details[req] = found_in
            else:
                details[req] = []

        return {"found": found, "total": len(requirements), "details": details}

    def generate_reality_assessment(self):
        """Generate overall assessment of framework vs optimization status."""
        print("\n🎯 REALITY ASSESSMENT...")

        # Calculate metrics
        core_modules_count = len(self.analysis.get("core_modules", {}))
        research_modules_count = len(self.analysis.get("research_modules", {}))
        total_modules = core_modules_count + research_modules_count

        # Calculate functionality coverage
        functionality = self.analysis.get("actual_functionality", {})
        if functionality:
            coverage_scores = []
            for cap, info in functionality.items():
                coverage_scores.append((info["found"] / max(info["total"], 1)) * 100)
            avg_functionality_coverage = sum(coverage_scores) / len(coverage_scores)
        else:
            avg_functionality_coverage = 0

        # Calculate total code volume
        total_lines = 0
        all_modules = {
            **self.analysis.get("core_modules", {}),
            **self.analysis.get("research_modules", {}),
        }
        for module_info in all_modules.values():
            total_lines += module_info.get("lines", 0)

        # Assessment logic
        framework_maturity = self.assess_framework_maturity(
            total_modules, avg_functionality_coverage, total_lines
        )

        assessment = {
            "total_framework_modules": total_modules,
            "avg_functionality_coverage": avg_functionality_coverage,
            "total_framework_lines": total_lines,
            "framework_maturity": framework_maturity,
            "primary_focus": self.determine_primary_focus(),
            "next_priority": self.determine_next_priority(
                framework_maturity, avg_functionality_coverage
            ),
        }

        self.analysis["reality_assessment"] = assessment

        # Print assessment
        print(f"   📊 Framework modules: {total_modules}")
        print(f"   📈 Functionality coverage: {avg_functionality_coverage:.1f}%")
        print(f"   📝 Total framework code: {total_lines:,} lines")
        print(f"   🎯 Framework maturity: {framework_maturity}")
        print(f"   🔍 Primary focus so far: {assessment['primary_focus']}")
        print(f"   ⭐ Next priority: {assessment['next_priority']}")

    def assess_framework_maturity(self, modules, coverage, lines):
        """Assess the maturity level of the framework."""
        if modules >= 20 and coverage >= 80 and lines >= 10000:
            return "Production Ready"
        elif modules >= 15 and coverage >= 60 and lines >= 5000:
            return "Well Developed"
        elif modules >= 10 and coverage >= 40 and lines >= 2000:
            return "Moderately Developed"
        elif modules >= 5 and coverage >= 20 and lines >= 1000:
            return "Early Stage"
        else:
            return "Infrastructure Only"

    def determine_primary_focus(self):
        """Determine what has been the primary focus so far."""
        # Analyze what we've spent most effort on
        core_focus_areas = []

        # Check if we have substantial infrastructure
        if any(
            "optimization" in name.lower()
            for name in self.analysis.get("core_modules", {}).keys()
        ):
            core_focus_areas.append("Performance Optimization")

        if any(
            "lazy" in name.lower() or "import" in name.lower()
            for name in self.analysis.get("core_modules", {}).keys()
        ):
            core_focus_areas.append("Import Optimization")

        # Check research modules
        research_modules = self.analysis.get("research_modules", {})
        if len(research_modules) > 5:
            core_focus_areas.append("Research Module Development")

        if not core_focus_areas:
            return "Basic Infrastructure"
        else:
            return " + ".join(core_focus_areas)

    def determine_next_priority(self, maturity, coverage):
        """Determine what should be the next priority."""
        if maturity == "Infrastructure Only":
            return "🚨 CORE FRAMEWORK DEVELOPMENT (high priority)"
        elif coverage < 50:
            return "🔧 FUNCTIONALITY COMPLETION (medium priority)"
        elif maturity in ["Early Stage", "Moderately Developed"]:
            return "📈 FRAMEWORK EXPANSION (continue development)"
        else:
            return "✨ VALIDATION & POLISH (low priority - framework is solid)"


def main():
    """Run comprehensive codebase reality check."""
    checker = CodebaseRealityCheck()
    analysis = checker.analyze_comprehensive_framework_status()

    print("\n" + "=" * 70)
    print("🏁 FINAL VERDICT")
    print("=" * 70)

    assessment = analysis["reality_assessment"]
    maturity = assessment["framework_maturity"]
    coverage = assessment["avg_functionality_coverage"]
    next_priority = assessment["next_priority"]

    if "Infrastructure Only" in maturity or coverage < 30:
        print("🚨 CONCLUSION: Framework development should be THE PRIORITY")
        print("   Current focus on validation is PREMATURE")
        print("   We have optimization infrastructure but limited core functionality")
        print("   Recommendation: Focus on core framework development first")
    elif coverage < 60:
        print("⚠️  CONCLUSION: Framework development should continue")
        print("   Current validation efforts are somewhat premature")
        print("   We have good foundation but need more core functionality")
        print("   Recommendation: Balance framework development with light validation")
    else:
        print("✅ CONCLUSION: Framework is well-developed")
        print("   Validation and polish are appropriate now")
        print("   We have substantial core functionality")
        print("   Recommendation: Continue with validation and polish")

    print(f"\n🎯 Next Priority: {next_priority}")

    return analysis


if __name__ == "__main__":
    main()
