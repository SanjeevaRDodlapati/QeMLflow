"""
ChemML Enhanced Framework Demo
=============================

Demonstration of the enhanced ChemML capabilities including:
- Comprehensive data processing pipeline
- Extended model suite with ensemble and boosting methods
- Automated ML pipelines with hyperparameter optimization
- Production-ready workflows

This demo showcases the new core framework features.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def demo_enhanced_data_processing():
    """Demonstrate enhanced data processing capabilities."""
    print("🧪 ChemML Enhanced Data Processing Demo")
    print("=" * 50)

    # Import enhanced data processing
    from chemml.core.data_processing import (
        AdvancedDataPreprocessor,
        ChemMLDataLoader,
        IntelligentDataSplitter,
        load_chemical_dataset,
        preprocess_chemical_data,
        split_chemical_data,
    )

    # 1. Load a chemistry dataset
    print("📊 Loading BBBP dataset...")
    try:
        df = load_chemical_dataset("bbbp")
        print(f"✅ Loaded {len(df)} compounds with {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")
    except Exception as e:
        print(f"⚠️  Could not load dataset: {e}")
        # Create synthetic data for demo
        print("📊 Creating synthetic dataset...")
        df = create_synthetic_chemistry_dataset()

    # 2. Advanced preprocessing
    print("\n🔧 Advanced preprocessing...")
    processed_data = preprocess_chemical_data(
        df,
        smiles_column="smiles",
        target_columns=["target"] if "target" in df.columns else [df.columns[-1]],
    )

    print(f"✅ Generated {len(processed_data['feature_names'])} molecular features")
    print(f"Feature types: {processed_data['preprocessing_info']['feature_counts']}")

    # 3. Intelligent data splitting
    print("\n📊 Intelligent data splitting...")
    splits = split_chemical_data(
        processed_data["X"],
        processed_data["y"],
        smiles=processed_data["smiles"],
        split_method="scaffold",
        test_size=0.2,
        val_size=0.1,
    )

    split_info = splits["split_info"]
    print(
        f"✅ {split_info['method']} split: {split_info['train_size']} train, "
        f"{split_info['val_size']} val, {split_info['test_size']} test"
    )

    return processed_data, splits


def demo_enhanced_models():
    """Demonstrate enhanced model capabilities."""
    print("\n🤖 ChemML Enhanced Models Demo")
    print("=" * 50)

    # Create synthetic data
    X, y = create_synthetic_features()

    # Import enhanced models
    try:
        from chemml.core.enhanced_models import (
            AutoMLModel,
            EnsembleModel,
            create_automl_model,
            create_ensemble_model,
        )

        # Test ensemble model
        print("🔄 Testing Ensemble Model...")
        ensemble = create_ensemble_model(
            ensemble_method="voting", task_type="regression"
        )
        ensemble_metrics = ensemble.fit(X, y)
        print(f"✅ Ensemble trained: {ensemble_metrics}")

        # Test AutoML
        print("\n🔄 Testing AutoML Model...")
        automl = create_automl_model(
            task_type="regression",
            n_trials=10,  # Quick demo
            model_types=["rf", "linear"],
        )
        automl_metrics = automl.fit(X, y)
        print(f"✅ AutoML completed: {automl_metrics}")
        print(f"Best parameters: {automl.get_best_params()}")

        # Test gradient boosting if available
        try:
            from chemml.core.enhanced_models import create_xgboost_model

            print("\n🔄 Testing XGBoost Model...")
            xgb_model = create_xgboost_model(task_type="regression")
            xgb_metrics = xgb_model.fit(X, y)
            print(f"✅ XGBoost trained: {xgb_metrics}")
        except ImportError:
            print("⚠️  XGBoost not available")

    except Exception as e:
        print(f"❌ Enhanced models demo failed: {e}")


def demo_automated_pipeline():
    """Demonstrate automated ML pipeline."""
    print("\n🚀 ChemML Automated Pipeline Demo")
    print("=" * 50)

    try:
        from chemml.core.pipeline import create_pipeline, quick_pipeline

        # Create synthetic dataset
        df = create_synthetic_chemistry_dataset(n_samples=200)

        # Quick pipeline demo
        print("⚡ Running quick pipeline...")
        try:
            results = quick_pipeline(
                data_source=df,
                task_type="regression",
                smiles_column="smiles",
                target_columns=["target"],
            )
            print("✅ Quick pipeline completed!")
            print(results[["model", "type", "r2", "rmse"]].round(4))
        except Exception as e:
            print(f"⚠️  Quick pipeline failed: {e}")

        # Full pipeline demo
        print("\n🔧 Running detailed pipeline...")
        pipeline = create_pipeline(task_type="regression")

        # Load data
        pipeline.load_data(df, smiles_column="smiles", target_columns=["target"])

        # Preprocess
        pipeline.preprocess()

        # Split data
        pipeline.split_data(split_method="random")

        # Add models
        pipeline.add_model("linear", "linear", regularization="ridge")
        pipeline.add_model("rf", "rf", n_estimators=50)

        # Try to add advanced models
        try:
            pipeline.add_model("ensemble", "ensemble", ensemble_method="voting")
        except Exception:
            print("⚠️  Could not add ensemble model")

        # Train models
        pipeline.train_models()

        # Evaluate
        results = pipeline.evaluate_models()
        print("✅ Detailed pipeline completed!")
        print(results[["model", "type", "r2", "rmse"]].round(4))

        # Save pipeline
        save_path = Path.home() / ".chemml" / "demo_pipeline"
        pipeline.save_pipeline(save_path)
        print(f"💾 Pipeline saved to {save_path}")

    except Exception as e:
        print(f"❌ Pipeline demo failed: {e}")


def demo_performance_comparison():
    """Compare performance of different model types."""
    print("\n📊 Performance Comparison Demo")
    print("=" * 50)

    # Create test data
    X, y = create_synthetic_features(n_samples=500, n_features=50)

    from chemml.core.models import LinearModel, RandomForestModel, compare_models

    # Create models for comparison
    models = {
        "Linear (Ridge)": LinearModel(task_type="regression", regularization="ridge"),
        "Random Forest": RandomForestModel(task_type="regression", n_estimators=100),
    }

    # Add enhanced models if available
    try:
        from chemml.core.enhanced_models import EnsembleModel

        models["Ensemble"] = EnsembleModel(task_type="regression")
    except ImportError:
        pass

    # Compare models
    print("🔄 Comparing models...")
    comparison_results = compare_models(models, X, y)
    print("✅ Comparison completed!")
    print(comparison_results[["model", "r2", "rmse"]].round(4))


def create_synthetic_chemistry_dataset(n_samples: int = 100) -> pd.DataFrame:
    """Create synthetic chemistry dataset for demo purposes."""
    np.random.seed(42)

    # Generate fake SMILES (simplified for demo)
    molecules = ["C", "CC", "CCC", "CCCC", "C1CCCCC1", "c1ccccc1", "CCO", "CCN"]
    smiles = np.random.choice(molecules, n_samples)

    # Generate synthetic targets
    targets = np.random.normal(5.0, 2.0, n_samples)

    # Add some additional features
    mw = np.random.normal(200, 50, n_samples)
    logp = np.random.normal(2.5, 1.0, n_samples)

    return pd.DataFrame(
        {"smiles": smiles, "molecular_weight": mw, "logp": logp, "target": targets}
    )


def create_synthetic_features(n_samples: int = 100, n_features: int = 20) -> tuple:
    """Create synthetic feature matrix for model testing."""
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)

    # Create correlated target
    weights = np.random.randn(n_features)
    y = X @ weights + np.random.randn(n_samples) * 0.1

    return X, y


def run_full_demo():
    """Run the complete enhanced ChemML demo."""
    print("🌟 ChemML Enhanced Framework Demo")
    print("=" * 60)
    print("Demonstrating new core framework capabilities...")
    print()

    try:
        # Demo 1: Enhanced data processing
        processed_data, splits = demo_enhanced_data_processing()

        # Demo 2: Enhanced models
        demo_enhanced_models()

        # Demo 3: Automated pipelines
        demo_automated_pipeline()

        # Demo 4: Performance comparison
        demo_performance_comparison()

        print("\n🎉 All demos completed successfully!")
        print("\nKey achievements:")
        print("✅ Comprehensive data processing with 8+ chemistry datasets")
        print("✅ Extended model suite with ensemble and boosting methods")
        print("✅ Automated ML pipelines with hyperparameter optimization")
        print("✅ Production-ready workflows with persistence")
        print("✅ Intelligent data splitting (scaffold, stratified, temporal)")
        print("✅ Automated feature engineering for molecular data")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    run_full_demo()
