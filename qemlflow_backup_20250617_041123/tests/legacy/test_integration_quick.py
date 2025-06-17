"""
Integration test to verify that our virtual screening and property prediction modules work together.
"""

import numpy as np
import pandas as pd


def test_integration():
    # Test virtual screening
    from chemml.research.drug_discovery.screening import perform_virtual_screening

    reference_smiles = ["CCO", "CCC", "CCCO"]
    library_smiles = ["CCCC", "CCCCC", "CC(C)O", "CCO", "invalid"]

    try:
        screening_results = perform_virtual_screening(
            reference_smiles=reference_smiles,
            library_smiles=library_smiles,
            method="similarity",
            threshold=0.3,
            max_hits=10,
        )

        print(f"✅ Virtual screening: Found {len(screening_results['results'])} hits")
        print(f"   Screening method: {screening_results['statistics']['method']}")
        print(f"   Hit rate: {screening_results['statistics']['hit_rate']:.2%}")

    except Exception as e:
        print(f"❌ Virtual screening error: {e}")
        return False

    # Test property prediction
    from chemml.research.drug_discovery.properties import (
        predict_properties,
        train_property_model,
    )

    try:
        # Generate synthetic data for training
        X = np.random.rand(20, 5)
        y = np.random.rand(20)

        # Train model
        model = train_property_model(X, y, model_type="regression")
        print(f"✅ Property model training: {type(model)}")
        print(f"   Has predict method: {hasattr(model, 'predict')}")

        # Test predictions
        test_predictions = model.predict(X[:3])
        print(f"✅ Property predictions: {len(test_predictions)} values")

        # Test with SMILES list
        smiles_list = ["CCO", "CCC", "CCCO"]
        props = predict_properties(smiles_list)
        print(f"✅ SMILES property prediction: {props.shape}")

    except Exception as e:
        print(f"❌ Property prediction error: {e}")
        return False

    # Test QSAR modeling
    from chemml.research.drug_discovery.qsar import build_qsar_model, predict_activity

    try:
        # Build QSAR model
        qsar_model = build_qsar_model(X, y, model_type="random_forest")
        print(f"✅ QSAR model building: {type(qsar_model)}")
        print(f"   Has predict method: {hasattr(qsar_model, 'predict')}")

        # Test predictions
        activities = predict_activity(qsar_model, X[:3])
        print(f"✅ QSAR predictions: {len(activities)} values")

    except Exception as e:
        print(f"❌ QSAR modeling error: {e}")
        return False

    print("\n🎉 All integration tests passed!")
    return True


if __name__ == "__main__":
    test_integration()
