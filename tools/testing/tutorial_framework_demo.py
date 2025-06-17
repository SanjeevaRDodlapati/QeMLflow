"""
QeMLflow Tutorial Framework Demonstration
======================================

This script demonstrates the new tutorial framework capabilities including:
- Educational datasets
- Environment management
- Interactive components
- Progress tracking
- Quantum computing integration

Run this script to see the tutorial framework in action.
"""

import warnings

import numpy as np

from src.qemlflow.tutorials import (
    EducationalDatasets,
    EnvironmentManager,
    LearningAssessment,
    ProgressTracker,
    check_quantum_requirements,
    create_progress_dashboard,
    get_quantum_tutorial_overview,
    setup_learning_environment,
    visualize_molecules,
)


def main():
    """Demonstrate the QeMLflow tutorial framework."""
    print("🧪 QeMLflow Tutorial Framework Demonstration")
    print("=" * 60)

    # 1. Environment Setup and Checking
    print("\n🔧 Environment Management")
    print("-" * 30)

    env_manager = EnvironmentManager("demo")
    env_status = env_manager.check_environment(verbose=True)

    # 2. Educational Datasets
    print("\n📚 Educational Datasets")
    print("-" * 30)

    datasets = EducationalDatasets()

    # Load drug molecules
    drugs = datasets.load_drug_molecules()
    print(f"✅ Loaded {len(drugs)} drug molecules:")
    for name, smiles in list(drugs.items())[:3]:
        print(f"   • {name}: {smiles}")

    # Get molecular dataset with properties
    drug_df = datasets.get_molecule_dataset("drugs")
    print(
        f"✅ Created dataset with {len(drug_df)} molecules and {len(drug_df.columns)-2} properties"
    )

    # Show property ranges
    prop_ranges = datasets.get_property_ranges("drugs")
    print("📊 Property ranges:")
    for prop, (min_val, max_val) in list(prop_ranges.items())[:3]:
        print(f"   • {prop}: {min_val:.2f} - {max_val:.2f}")

    # 3. Learning Assessment
    print("\n🎯 Learning Assessment Framework")
    print("-" * 30)

    # Create sample assessment
    assessment = LearningAssessment(
        student_id="demo_student", section="molecular_properties"
    )

    # Add some sample activities
    assessment.add_concept_checkpoint(
        "molecular_weight", 0.85, 0.80, "Good understanding of MW calculations"
    )
    assessment.add_concept_checkpoint(
        "logp", 0.72, 0.65, "Need more practice with LogP prediction"
    )
    assessment.record_activity("property_calculation", 120)  # 2 minutes

    # Get assessment summary
    summary = assessment.get_progress_summary()
    print(f"✅ Assessment created for section: {summary['section']}")
    print(f"   • Total activities: {summary['total_activities']}")
    print(f"   • Completed activities: {summary['activities_completed']}")
    print(f"   • Total time: {summary['total_time_hours']:.2f} hours")

    # 4. Progress Tracking
    print("\n📈 Progress Tracking")
    print("-" * 30)

    tracker = ProgressTracker(assessment)

    # Simulate some learning activities
    tracker.log_progress("molecular_descriptors", "completed", {"time_spent": 45})
    tracker.log_progress("hydrogen_bonding", "completed", {"understanding": 0.78})
    tracker.log_progress("lipophilicity", "completed", {"understanding": 0.82})

    session_data = tracker.get_session_summary()
    print("✅ Session completed:")
    print(f"   • Activities logged: {session_data['total_steps']}")
    print(f"   • Session duration: {session_data['elapsed_time_minutes']:.1f} minutes")

    # 5. Molecular Visualization
    print("\n🧬 Molecular Visualization")
    print("-" * 30)

    # Select a few molecules for visualization
    sample_molecules = {
        "aspirin": drugs["aspirin"],
        "caffeine": drugs["caffeine"],
        "ibuprofen": drugs["ibuprofen"],
    }

    print("📊 Visualizing molecules (check output above)...")
    _mol_vis = visualize_molecules(sample_molecules, grid_size=(1, 3))

    # 6. Quantum Computing Integration
    print("\n🌌 Quantum Computing Integration")
    print("-" * 30)

    quantum_status = check_quantum_requirements()
    print("📋 Quantum dependencies:")
    for package, available in quantum_status.items():
        emoji = "✅" if available else "❌"
        print(f"   {emoji} {package}")

    # Show quantum tutorial overview
    if any(quantum_status.values()):
        print("\n🎓 Available quantum tutorials:")
        overview = get_quantum_tutorial_overview()
        # Print first few lines of overview
        for line in overview.split("\n")[:10]:
            if line.strip():
                print(f"   {line}")

    # 7. Synthetic Data Generation
    print("\n🔬 Synthetic Data Generation")
    print("-" * 30)

    # Create synthetic molecular examples
    synthetic_data = datasets.create_synthetic_examples(10, complexity="simple")
    print(f"✅ Generated {len(synthetic_data)} synthetic molecules")
    print(
        f"   • Average molecular weight: {synthetic_data['molecular_weight'].mean():.2f}"
    )
    print(
        f"   • Property columns: {list(synthetic_data.columns)[3:6]}"
    )  # Skip name, smiles, mol

    # 8. Tutorial Environment Setup
    print("\n⚙️ Tutorial Environment Setup")
    print("-" * 30)

    # Setup environment for basic tutorials
    env_ready = env_manager.setup_tutorial_environment("basic", auto_install=False)
    print(f"✅ Basic tutorial environment ready: {env_ready}")

    # Setup fallbacks for missing dependencies
    fallbacks = env_manager.setup_fallbacks()
    if fallbacks:
        print(f"🔧 Fallbacks configured for {len(fallbacks)} missing dependencies")
    else:
        print("🎉 No fallbacks needed - all dependencies available!")

    # 9. Summary Statistics
    print("\n📊 Demo Summary")
    print("-" * 30)

    print(f"🧪 Educational datasets: {len(datasets.molecules)} categories")
    print("🎯 Assessment framework: ✅ Operational")
    print("📈 Progress tracking: ✅ Operational")
    print(f"🔧 Environment management: ✅ {env_status['overall_status'].title()}")
    print(
        f"🌌 Quantum integration: {'✅ Available' if any(quantum_status.values()) else '⚠️ Limited'}"
    )
    print("🧬 Visualization: ✅ Operational")

    print("\n🎉 QeMLflow Tutorial Framework is ready for educational use!")
    print("\n💡 Next steps:")
    print("   • Explore notebooks/learning/fundamentals/ for tutorial examples")
    print("   • Use the tutorial framework in your own educational content")
    print("   • Check UPDATED_NOTEBOOKS_INTEGRATION_PLAN.md for the full roadmap")


if __name__ == "__main__":
    # Suppress some warnings for cleaner demo output
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    main()
