Additional integration tests to improve coverage for ChemML modules.
These tests exercise more code paths to increase coverage percentage.
"""
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

def test_comprehensive_molecular_processing():
    """Test comprehensive molecular processing pipeline."""
    from src.data_processing.feature_extraction import (
        calculate_properties,
        extract_descriptors,
        extract_fingerprints,
        extract_structural_features,
    )
    from src.data_processing.molecular_preprocessing import (
        clean_molecular_data,
        filter_molecules_by_properties,
        standardize_molecules,
    )
    from src.utils.molecular_utils import (
        calculate_similarity,
        mol_to_smiles,
        smiles_to_mol,
    )

    # Test data
    smiles_list = [
        "CCO",  # ethanol
        "CCC",  # propane
        "CCCO",  # propanol
        "CC(C)O",  # isopropanol
        "INVALID_SMILES",  # invalid for error testing
        None,  # None for error testing
    ]

    # Test molecular preprocessing
    cleaned_data = clean_molecular_data(pd.DataFrame({"SMILES": smiles_list}))
    assert not cleaned_data.empty

    # Test molecule conversion
    valid_smiles = [s for s in smiles_list if s and s != "INVALID_SMILES"]
    molecules = [smiles_to_mol(smiles) for smiles in valid_smiles]
    back_to_smiles = [mol_to_smiles(mol) for mol in molecules if mol is not None]
    assert len(back_to_smiles) > 0

    # Test similarity calculation
    if len(valid_smiles) >= 2:
        similarity = calculate_similarity(valid_smiles[0], valid_smiles[1])
        assert 0 <= similarity <= 1

    # Test property calculation
    properties = calculate_properties(valid_smiles)
    assert not properties.empty
    assert len(properties) == len(valid_smiles)

    # Test descriptor extraction with different sets
    for desc_set in ["rdkit", "basic"]:
        descriptors = extract_descriptors(valid_smiles, descriptor_set=desc_set)
        assert not descriptors.empty

    # Test fingerprint extraction
    for fp_type in ["morgan", "maccs"]:
        fingerprints = extract_fingerprints(valid_smiles, fp_type=fp_type)
        assert not fingerprints.empty

    # Test structural features
    struct_features = extract_structural_features(valid_smiles)
    assert not struct_features.empty

    print("✅ Comprehensive molecular processing tests passed")

def test_comprehensive_ml_pipeline():
    """Test comprehensive ML pipeline with different configurations."""
    from src.models.classical_ml.regression_models import RegressionModel
    from src.utils.metrics import (
        calculate_accuracy,
        calculate_f1_score,
        calculate_mse,
        calculate_precision,
        calculate_r2,
        calculate_recall,
    )
    from src.utils.ml_utils import evaluate_model, normalize_features, split_data

    # Generate synthetic data
    np.random.seed(42)
    X = np.random.rand(50, 8)
    y_regression = X.sum(axis=1) + np.random.normal(0, 0.1, 50)
    y_classification = (y_regression > y_regression.mean()).astype(int)

    # Test data splitting
    X_train, X_test, y_train, y_test = split_data(X, y_regression, test_size=0.3)
    assert X_train.shape[0] + X_test.shape[0] == X.shape[0]

    # Test feature normalization
    X_train_norm, scaler = normalize_features(X_train)
    X_test_norm = scaler.transform(X_test)
    assert X_train_norm.shape == X_train.shape

    # Test regression models
    for model_type in ["linear", "ridge", "lasso"]:
        model = RegressionModel(model_type=model_type)
        model.fit(X_train_norm, y_train)
        predictions = model.predict(X_test_norm)

        # Test evaluation
        mse = calculate_mse(y_test, predictions)
        r2 = calculate_r2(y_test, predictions)

        assert mse >= 0
        assert -1 <= r2 <= 1

    # Test classification metrics
    y_pred_class = (np.random.rand(len(y_classification)) > 0.5).astype(int)

    accuracy = calculate_accuracy(y_classification, y_pred_class)
    precision = calculate_precision(y_classification, y_pred_class)
    recall = calculate_recall(y_classification, y_pred_class)
    f1 = calculate_f1_score(y_classification, y_pred_class)

    assert 0 <= accuracy <= 1
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= f1 <= 1

    print("✅ Comprehensive ML pipeline tests passed")

def test_quantum_computing_integration():
    """Test quantum computing integration."""
    from src.models.quantum_ml.quantum_circuits import QuantumCircuit
    from src.utils.quantum_utils import (
        apply_quantum_gate,
        create_quantum_circuit,
        measure_quantum_state,
    )

    # Test quantum circuit creation and operations
    qc = QuantumCircuit(n_qubits=4)
    assert qc.n_qubits == 4

    # Test different quantum operations
    angles = [0.1, 0.2, 0.3, 0.4]
    qc.add_rotation_layer(angles)
    qc.add_entangling_layer()

    # Test parameterized circuit
    param_circuit = qc.create_parameterized_circuit(n_layers=2)
    assert param_circuit is not None

    # Test parameter binding
    params = np.random.rand(qc.num_parameters)
    bound_circuit = qc.bind_parameters(params)
    assert bound_circuit is not None

    # Test simulation
    result = qc.simulate()
    assert isinstance(result, dict)
    assert "counts" in result

    # Test VQE
    vqe_result = qc.run_vqe("mock_hamiltonian")
    assert isinstance(vqe_result, dict)
    assert "energy" in vqe_result

    # Test feature mapping
    classical_data = np.random.rand(4)
    feature_map = qc.create_feature_map(classical_data)
    assert feature_map is not None

    # Test utility functions
    util_qc = create_quantum_circuit(3)
    apply_quantum_gate(util_qc, "H", 0)
    apply_quantum_gate(util_qc, "CNOT", [0, 1])

    measurement = measure_quantum_state(util_qc)
    assert measurement is not None

    print("✅ Quantum computing integration tests passed")

def test_visualization_functions():
    """Test visualization functions."""
    from src.utils.visualization import (
        ModelVisualizer,
        MolecularVisualizer,
        plot_feature_importance,
        plot_model_performance,
        plot_molecular_structure,
    )

    # Test molecular structure plotting
    with tempfile.TemporaryDirectory() as tmpdir:
        plot_file = os.path.join(tmpdir, "molecule.png")
        _result = plot_molecular_structure("CCO", filename=plot_file)
        # Should not raise error

    # Test feature importance plotting
    feature_names = ["feature1", "feature2", "feature3", "feature4"]
    importances = np.array([0.4, 0.3, 0.2, 0.1])

    with tempfile.TemporaryDirectory() as tmpdir:
        plot_file = os.path.join(tmpdir, "importance.png")
        plot_feature_importance(importances, feature_names, filename=plot_file)

    # Test model performance plotting
    history = {
        "loss": [1.0, 0.8, 0.6, 0.4, 0.2],
        "accuracy": [0.6, 0.7, 0.8, 0.9, 0.95],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        plot_file = os.path.join(tmpdir, "performance.png")
        plot_model_performance(history, filename=plot_file)

    # Test visualizer classes
    _mol_viz = MolecularVisualizer()
    _model_viz = ModelVisualizer()

    # Test with sample data
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])

    with tempfile.TemporaryDirectory() as tmpdir:
        plot_file = os.path.join(tmpdir, "predictions.png")
        ModelVisualizer.plot_predictions_vs_actual(y_true, y_pred, filename=plot_file)

    print("✅ Visualization function tests passed")

def test_io_and_utility_functions():
    """Test I/O and utility functions."""
    from src.utils.io_utils import (
        export_results,
        load_molecular_data,
        load_smiles_from_file,
        save_molecular_data,
        save_smiles_to_file,
    )

    # Create test data
    test_data = pd.DataFrame(
        {
            "SMILES": ["CCO", "CCC", "CCCO"],
            "Name": ["ethanol", "propane", "propanol"],
            "Activity": [0.8, 0.3, 0.6],
        }
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test saving and loading CSV
        csv_file = os.path.join(tmpdir, "molecules.csv")
        save_molecular_data(test_data, csv_file)
        loaded_data = load_molecular_data(csv_file)
        assert not loaded_data.empty
        assert len(loaded_data) == len(test_data)

        # Test SMILES file operations
        smiles_file = os.path.join(tmpdir, "smiles.txt")
        save_smiles_to_file(test_data["SMILES"].tolist(), smiles_file)
        loaded_smiles = load_smiles_from_file(smiles_file)
        assert len(loaded_smiles) == len(test_data)

        # Test results export
        results = {
            "predictions": [0.1, 0.2, 0.3],
            "metrics": {"accuracy": 0.85, "precision": 0.82},
        }
        results_file = os.path.join(tmpdir, "results.json")
        export_results(results, results_file)
        assert os.path.exists(results_file)

    print("✅ I/O and utility function tests passed")

if __name__ == "__main__":
    test_comprehensive_molecular_processing()
    test_comprehensive_ml_pipeline()
    test_quantum_computing_integration()
    test_visualization_functions()
    test_io_and_utility_functions()
    print("\n🎉 All comprehensive integration tests passed!")
    print("These tests exercise many more code paths to improve coverage.")
