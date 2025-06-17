#!/usr/bin/env python3
"""
Quick Development Commands for QeMLflow
====================================

Easy access to enhanced development features.
"""

import subprocess
import sys


def show_dashboard():
    """Show performance dashboard."""
    from qemlflow.core.monitoring import show_performance_dashboard

    show_performance_dashboard()


def recommend_model_demo():
    """Demo model recommendation system."""
    from qemlflow.core.recommendations import recommend_model

    # Demo data
    demo_smiles = ["CCO", "CCC", "c1ccccc1", "CCN", "CCCN"]

    print("🤖 Model Recommendation Demo")
    print("=" * 40)

    # Get recommendation
    recommendation = recommend_model(
        molecular_data=demo_smiles,
        target_property="logP",
        computational_budget="medium",
    )

    print(f"📊 Recommended Model: {recommendation['recommended_model']}")
    print(f"🎯 Confidence: {recommendation['confidence']:.2f}")
    print(f"💡 Rationale: {recommendation['rationale']}")
    print(
        f"⚡ Expected Accuracy: {recommendation['expected_performance']['estimated_accuracy']}"
    )


def generate_docs():
    """Generate API documentation."""
    subprocess.run([sys.executable, "tools/development/auto_docs.py"])


def run_tests():
    """Run test suite."""
    subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"])


def setup_notebook():
    """Setup Jupyter notebook with QeMLflow."""
    subprocess.run([sys.executable, "-m", "jupyter", "lab", "--ip=0.0.0.0"])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("🧬 QeMLflow Quick Development Commands")
        print("=" * 40)
        print("python quick_dev.py dashboard     - Show performance dashboard")
        print("python quick_dev.py recommend     - Demo model recommendation")
        print("python quick_dev.py docs          - Generate API documentation")
        print("python quick_dev.py tests         - Run test suite")
        print("python quick_dev.py notebook      - Launch Jupyter Lab")
        sys.exit(0)

    command = sys.argv[1]

    if command == "dashboard":
        show_dashboard()
    elif command == "recommend":
        recommend_model_demo()
    elif command == "docs":
        generate_docs()
    elif command == "tests":
        run_tests()
    elif command == "notebook":
        setup_notebook()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
