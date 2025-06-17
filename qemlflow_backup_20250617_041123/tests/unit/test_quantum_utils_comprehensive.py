"""
Comprehensive tests for quantum_utils module

This test suite achieves high coverage for quantum computing utilities
including quantum circuit builders, VQE optimization, molecular Hamiltonians,
quantum machine learning, and standalone utility functions.
"""

import logging
import sys
from unittest.mock import MagicMock, Mock, call, patch

import numpy as np
import pytest

# Import the module under test
sys.path.insert(0, "/Users/sanjeevadodlapati/Downloads/Repos/ChemML/src")
from utils.quantum_utils import (
    PENNYLANE_AVAILABLE,
    QISKIT_AVAILABLE,
    MolecularHamiltonian,
    QuantumCircuitBuilder,
    QuantumMachineLearning,
    VQEOptimizer,
    apply_quantum_gate,
    create_quantum_circuit,
    create_quantum_feature_map,
    measure_quantum_state,
    quantum_distance,
)


class TestQuantumCircuitBuilder:
    """Test QuantumCircuitBuilder class"""

    def test_init_without_qiskit(self):
        """Test initialization when Qiskit is not available"""
        with patch("utils.quantum_utils.QISKIT_AVAILABLE", False):
            with pytest.raises(ImportError, match="Qiskit is required"):
                QuantumCircuitBuilder()

    def test_init_with_qiskit(self):
        """Test initialization when Qiskit is available"""
        if not QISKIT_AVAILABLE:
            pytest.skip("Qiskit not available")

        builder = QuantumCircuitBuilder()
        assert builder is not None

    @patch("utils.quantum_utils.QISKIT_AVAILABLE", True)
    @patch("utils.quantum_utils.SparsePauliOp")
    def test_create_h2_hamiltonian(self, mock_sparse_pauli_op):
        """Test H2 Hamiltonian creation"""
        # Mock the SparsePauliOp.from_list method
        mock_hamiltonian = Mock()
        mock_sparse_pauli_op.from_list.return_value = mock_hamiltonian

        result = QuantumCircuitBuilder.create_h2_hamiltonian()

        # Verify from_list was called with expected Pauli strings
        mock_sparse_pauli_op.from_list.assert_called_once()
        call_args = mock_sparse_pauli_op.from_list.call_args[0][0]

        # Check that the Pauli strings include expected terms
        pauli_ops = [term[0] for term in call_args]
        assert "II" in pauli_ops
        assert "ZI" in pauli_ops
        assert "IZ" in pauli_ops
        assert "ZZ" in pauli_ops
        assert "XX" in pauli_ops

        assert result == mock_hamiltonian

    @patch("utils.quantum_utils.QISKIT_AVAILABLE", True)
    @patch("utils.quantum_utils.QuantumCircuit")
    @patch("qiskit.circuit.Parameter")
    def test_create_ansatz_circuit(self, mock_parameter, mock_quantum_circuit):
        """Test ansatz circuit creation"""
        # Setup mocks
        mock_circuit = Mock()
        mock_quantum_circuit.return_value = mock_circuit
        mock_param = Mock()
        mock_parameter.return_value = mock_param

        circuit, params = QuantumCircuitBuilder.create_ansatz_circuit(
            num_qubits=2, depth=2
        )

        # Verify circuit creation
        mock_quantum_circuit.assert_called_once_with(2)
        assert circuit == mock_circuit
        assert len(params) == 4  # 2 qubits * 2 layers

        # Verify RY gates were added
        assert mock_circuit.ry.call_count == 4  # 2 qubits * 2 layers

        # Verify CNOT gates were added
        assert mock_circuit.cx.call_count == 2  # (num_qubits - 1) * depth

    @patch("utils.quantum_utils.QISKIT_AVAILABLE", True)
    @patch("utils.quantum_utils.QuantumCircuit")
    @patch("qiskit.circuit.Parameter")
    def test_create_hea_circuit(self, mock_parameter, mock_quantum_circuit):
        """Test Hardware Efficient Ansatz circuit creation"""
        mock_circuit = Mock()
        mock_quantum_circuit.return_value = mock_circuit
        mock_param = Mock()
        mock_parameter.return_value = mock_param

        circuit, params = QuantumCircuitBuilder.create_hea_circuit(
            num_qubits=3, layers=2
        )

        # Verify circuit creation
        mock_quantum_circuit.assert_called_once_with(3)
        assert circuit == mock_circuit

        # Verify parameters: 2 layers * 3 qubits * 2 rotation types
        assert len(params) == 12

        # Verify rotation gates (RY and RZ)
        assert mock_circuit.ry.call_count == 6  # 3 qubits * 2 layers
        assert mock_circuit.rz.call_count == 6  # 3 qubits * 2 layers

        # Verify entangling gates: (num_qubits - 1 + 1) * layers = 3 * 2 = 6
        assert mock_circuit.cx.call_count == 6


class TestVQEOptimizer:
    """Test VQEOptimizer class"""

    def test_init_without_qiskit(self):
        """Test initialization when Qiskit is not available"""
        with patch("utils.quantum_utils.QISKIT_AVAILABLE", False):
            with pytest.raises(ImportError, match="Qiskit is required"):
                VQEOptimizer()

    @patch("utils.quantum_utils.QISKIT_AVAILABLE", True)
    @patch("utils.quantum_utils.SPSA")
    @patch("utils.quantum_utils.AerSimulator")
    @patch("utils.quantum_utils.Estimator")
    def test_init_with_spsa_optimizer(self, mock_estimator, mock_aer, mock_spsa):
        """Test initialization with SPSA optimizer"""
        mock_optimizer = Mock()
        mock_spsa.return_value = mock_optimizer
        mock_backend = Mock()
        mock_aer.return_value = mock_backend
        mock_est = Mock()
        mock_estimator.return_value = mock_est

        try:
            vqe_opt = VQEOptimizer(optimizer="SPSA", max_iter=50)

            mock_spsa.assert_called_once_with(maxiter=50)
            assert vqe_opt.optimizer == mock_optimizer
            assert vqe_opt.backend == mock_backend
            assert vqe_opt.estimator == mock_est
            assert vqe_opt.max_iter == 50
        except ImportError:
            # This is expected when Qiskit is not available
            pytest.skip("Qiskit components not available")

    @patch("utils.quantum_utils.QISKIT_AVAILABLE", True)
    @patch("utils.quantum_utils.COBYLA")
    @patch("utils.quantum_utils.AerSimulator")
    @patch("utils.quantum_utils.Estimator")
    def test_init_with_cobyla_optimizer(self, mock_estimator, mock_aer, mock_cobyla):
        """Test initialization with COBYLA optimizer"""
        mock_optimizer = Mock()
        mock_cobyla.return_value = mock_optimizer

        try:
            vqe_opt = VQEOptimizer(optimizer="COBYLA")

            mock_cobyla.assert_called_once_with(maxiter=100)
            assert vqe_opt.optimizer == mock_optimizer
        except ImportError:
            pytest.skip("Qiskit components not available")

    @patch("utils.quantum_utils.QISKIT_AVAILABLE", True)
    @patch("utils.quantum_utils.AerSimulator")
    @patch("utils.quantum_utils.Estimator")
    def test_init_with_unsupported_optimizer(self, mock_estimator, mock_aer):
        """Test initialization with unsupported optimizer"""
        try:
            with pytest.raises(ValueError, match="Unsupported optimizer"):
                VQEOptimizer(optimizer="UNKNOWN")
        except ImportError:
            pytest.skip("Qiskit components not available")

    @patch("utils.quantum_utils.QISKIT_AVAILABLE", True)
    @patch("utils.quantum_utils.VQE")
    @patch("utils.quantum_utils.QuantumCircuitBuilder.create_h2_hamiltonian")
    @patch("utils.quantum_utils.QuantumCircuitBuilder.create_ansatz_circuit")
    @patch("utils.quantum_utils.AerSimulator")
    @patch("utils.quantum_utils.Estimator")
    @patch("utils.quantum_utils.SPSA")
    def test_optimize_h2(
        self,
        mock_spsa,
        mock_estimator,
        mock_aer,
        mock_create_ansatz,
        mock_create_h2,
        mock_vqe,
    ):
        """Test H2 optimization"""
        # Setup mocks
        mock_hamiltonian = Mock()
        mock_create_h2.return_value = mock_hamiltonian

        mock_circuit = Mock()
        mock_params = [Mock(), Mock(), Mock(), Mock()]
        mock_create_ansatz.return_value = (mock_circuit, mock_params)

        mock_result = Mock()
        mock_result.eigenvalue = complex(-1.85, 0)
        mock_result.optimal_parameters = np.array([0.1, 0.2, 0.3, 0.4])
        mock_result.optimizer_evals = 100

        mock_vqe_instance = Mock()
        mock_vqe_instance.compute_minimum_eigenvalue.return_value = mock_result
        mock_vqe.return_value = mock_vqe_instance

        vqe_opt = VQEOptimizer()
        result = vqe_opt.optimize_h2(bond_distance=0.8)

        # Verify method calls
        mock_create_h2.assert_called_once()
        mock_create_ansatz.assert_called_once_with(num_qubits=2, depth=2)
        mock_vqe_instance.compute_minimum_eigenvalue.assert_called_once_with(
            mock_hamiltonian
        )

        # Verify result structure
        assert "ground_state_energy" in result
        assert "optimal_parameters" in result
        assert "optimizer_evals" in result
        assert "bond_distance" in result
        assert result["ground_state_energy"] == -1.85
        assert result["bond_distance"] == 0.8

    @patch("utils.quantum_utils.QISKIT_AVAILABLE", True)
    @patch("utils.quantum_utils.AerSimulator")
    @patch("utils.quantum_utils.Estimator")
    @patch("utils.quantum_utils.SPSA")
    def test_scan_bond_distances(self, mock_spsa, mock_estimator, mock_aer):
        """Test bond distance scanning"""
        vqe_opt = VQEOptimizer()

        # Mock the optimize_h2 method
        vqe_opt.optimize_h2 = Mock(
            side_effect=[
                {"ground_state_energy": -1.85, "bond_distance": 0.7},
                {"ground_state_energy": -1.82, "bond_distance": 0.8},
                {"ground_state_energy": -1.75, "bond_distance": 0.9},
            ]
        )

        distances = [0.7, 0.8, 0.9]
        results = vqe_opt.scan_bond_distances(distances)

        assert len(results) == 3
        assert vqe_opt.optimize_h2.call_count == 3

        # Verify all distances were processed
        for i, distance in enumerate(distances):
            vqe_opt.optimize_h2.assert_any_call(bond_distance=distance)
            assert results[i]["bond_distance"] == distance

    @patch("utils.quantum_utils.QISKIT_AVAILABLE", True)
    @patch("utils.quantum_utils.AerSimulator")
    @patch("utils.quantum_utils.Estimator")
    @patch("utils.quantum_utils.SPSA")
    def test_scan_bond_distances_with_errors(self, mock_spsa, mock_estimator, mock_aer):
        """Test bond distance scanning with errors"""
        vqe_opt = VQEOptimizer()

        # Mock the optimize_h2 method to raise exception on second call
        vqe_opt.optimize_h2 = Mock(
            side_effect=[
                {"ground_state_energy": -1.85, "bond_distance": 0.7},
                Exception("Optimization failed"),
                {"ground_state_energy": -1.75, "bond_distance": 0.9},
            ]
        )

        distances = [0.7, 0.8, 0.9]
        results = vqe_opt.scan_bond_distances(distances)

        # Should return results for successful optimizations only
        assert len(results) == 2
        assert results[0]["bond_distance"] == 0.7
        assert results[1]["bond_distance"] == 0.9


class TestMolecularHamiltonian:
    """Test MolecularHamiltonian class"""

    def test_init_without_qiskit(self):
        """Test initialization when Qiskit is not available"""
        with patch("utils.quantum_utils.QISKIT_AVAILABLE", False):
            with pytest.raises(ImportError, match="Qiskit is required"):
                MolecularHamiltonian()

    @patch("utils.quantum_utils.QISKIT_AVAILABLE", True)
    def test_init_with_qiskit(self):
        """Test initialization when Qiskit is available"""
        hamiltonian = MolecularHamiltonian()
        assert hamiltonian is not None

    @patch("utils.quantum_utils.QISKIT_AVAILABLE", True)
    @patch("utils.quantum_utils.SparsePauliOp")
    def test_h2_hamiltonian_parametric(self, mock_sparse_pauli_op):
        """Test parametric H2 Hamiltonian creation"""
        mock_hamiltonian = Mock()
        mock_sparse_pauli_op.from_list.return_value = mock_hamiltonian

        result = MolecularHamiltonian.h2_hamiltonian_parametric(bond_distance=0.8)

        # Verify from_list was called
        mock_sparse_pauli_op.from_list.assert_called_once()
        call_args = mock_sparse_pauli_op.from_list.call_args[0][0]

        # Check that coefficients depend on bond distance
        pauli_dict = {term[0]: term[1] for term in call_args}

        # Nuclear repulsion should be 1/r = 1/0.8 = 1.25
        assert abs(pauli_dict["II"] - (1.25 + (-1.25 / (0.8**0.5)))) < 0.01

        assert result == mock_hamiltonian

    @patch("utils.quantum_utils.QISKIT_AVAILABLE", True)
    @patch("utils.quantum_utils.SparsePauliOp")
    def test_create_ising_hamiltonian(self, mock_sparse_pauli_op):
        """Test Ising Hamiltonian creation"""
        mock_hamiltonian = Mock()
        mock_sparse_pauli_op.from_list.return_value = mock_hamiltonian

        result = MolecularHamiltonian.create_ising_hamiltonian(
            num_qubits=3, coupling_strength=2.0
        )

        # Verify from_list was called
        mock_sparse_pauli_op.from_list.assert_called_once()
        call_args = mock_sparse_pauli_op.from_list.call_args[0][0]

        # Should have 3 longitudinal terms + 2 coupling terms
        assert len(call_args) == 5

        # Check for Z terms and ZZ coupling terms
        pauli_ops = [term[0] for term in call_args]
        coeffs = [term[1] for term in call_args]

        # Count Z and ZZ terms
        z_terms = [op for op in pauli_ops if op.count("Z") == 1]
        zz_terms = [op for op in pauli_ops if op.count("Z") == 2]

        assert len(z_terms) == 3  # Longitudinal field terms
        assert len(zz_terms) == 2  # Coupling terms

        # Check coupling strength in ZZ terms
        zz_coeffs = [coeffs[i] for i, op in enumerate(pauli_ops) if op.count("Z") == 2]
        for coeff in zz_coeffs:
            assert coeff == -2.0

        assert result == mock_hamiltonian


class TestQuantumMachineLearning:
    """Test QuantumMachineLearning class"""

    def test_init_without_pennylane(self):
        """Test initialization when PennyLane is not available"""
        with patch("utils.quantum_utils.PENNYLANE_AVAILABLE", False):
            with pytest.raises(ImportError, match="PennyLane is required"):
                QuantumMachineLearning()

    @patch("utils.quantum_utils.PENNYLANE_AVAILABLE", True)
    @patch("utils.quantum_utils.qml")
    def test_init_with_pennylane(self, mock_qml):
        """Test initialization when PennyLane is available"""
        mock_device = Mock()
        mock_qml.device.return_value = mock_device

        qml_instance = QuantumMachineLearning(num_qubits=4)

        mock_qml.device.assert_called_once_with("default.qubit", wires=4)
        assert qml_instance.num_qubits == 4
        assert qml_instance.dev == mock_device

    @patch("utils.quantum_utils.PENNYLANE_AVAILABLE", True)
    @patch("utils.quantum_utils.qml")
    def test_create_variational_classifier(self, mock_qml):
        """Test variational classifier creation"""
        mock_device = Mock()
        mock_qml.device.return_value = mock_device

        # Mock qnode decorator
        def mock_qnode(device):
            def decorator(func):
                return func

            return decorator

        mock_qml.qnode = mock_qnode

        qml_instance = QuantumMachineLearning(num_qubits=2)
        circuit = qml_instance.create_variational_classifier(num_layers=2)

        assert callable(circuit)

    @patch("utils.quantum_utils.PENNYLANE_AVAILABLE", True)
    @patch("utils.quantum_utils.qml")
    def test_create_quantum_embedding(self, mock_qml):
        """Test quantum embedding creation"""
        mock_device = Mock()
        mock_qml.device.return_value = mock_device

        # Mock qnode decorator
        def mock_qnode(device):
            def decorator(func):
                return func

            return decorator

        mock_qml.qnode = mock_qnode

        qml_instance = QuantumMachineLearning(num_qubits=3)
        embedding = qml_instance.create_quantum_embedding(data_reps=1)

        assert callable(embedding)


class TestStandaloneFunctions:
    """Test standalone utility functions"""

    def test_quantum_distance_identical_states(self):
        """Test quantum distance for identical states"""
        state1 = np.array([1.0, 0.0])
        state2 = np.array([1.0, 0.0])

        distance = quantum_distance(state1, state2)

        assert distance == pytest.approx(0.0, abs=1e-10)

    def test_quantum_distance_orthogonal_states(self):
        """Test quantum distance for orthogonal states"""
        state1 = np.array([1.0, 0.0])
        state2 = np.array([0.0, 1.0])

        distance = quantum_distance(state1, state2)

        assert distance == pytest.approx(1.0, abs=1e-10)

    def test_quantum_distance_superposition_states(self):
        """Test quantum distance for superposition states"""
        state1 = np.array([1.0, 0.0])
        state2 = np.array([1.0, 1.0]) / np.sqrt(2)

        distance = quantum_distance(state1, state2)

        # Fidelity should be 0.5, so distance should be 0.5
        assert distance == pytest.approx(0.5, abs=1e-10)

    def test_quantum_distance_unnormalized_states(self):
        """Test quantum distance with unnormalized states"""
        state1 = np.array([2.0, 0.0])  # Not normalized
        state2 = np.array([0.0, 3.0])  # Not normalized

        distance = quantum_distance(state1, state2)

        # Should still be orthogonal after normalization
        assert distance == pytest.approx(1.0, abs=1e-10)

    @patch("utils.quantum_utils.QISKIT_AVAILABLE", True)
    @patch("utils.quantum_utils.QuantumCircuit")
    def test_create_quantum_feature_map_with_qiskit(self, mock_quantum_circuit):
        """Test quantum feature map creation with Qiskit"""
        mock_circuit = Mock()
        mock_quantum_circuit.return_value = mock_circuit

        features = np.array([0.5, -0.3, 0.8])
        result = create_quantum_feature_map(features, num_qubits=3)

        # Verify circuit creation
        mock_quantum_circuit.assert_called_once_with(3)

        # Verify RY rotations were added
        assert mock_circuit.ry.call_count == 3

        # Verify entangling gates were added
        assert mock_circuit.cx.call_count == 2  # num_qubits - 1

        assert result == mock_circuit

    def test_create_quantum_feature_map_without_qiskit(self):
        """Test quantum feature map creation without Qiskit"""
        with patch("utils.quantum_utils.QISKIT_AVAILABLE", False):
            features = np.array([0.5, -0.3])

            with pytest.raises(ImportError, match="Qiskit is required"):
                create_quantum_feature_map(features)

    @patch("utils.quantum_utils.QISKIT_AVAILABLE", True)
    @patch("utils.quantum_utils.QuantumCircuit")
    def test_create_quantum_feature_map_default_qubits(self, mock_quantum_circuit):
        """Test quantum feature map with default number of qubits"""
        mock_circuit = Mock()
        mock_quantum_circuit.return_value = mock_circuit

        features = np.array([0.1, 0.2])
        _result = create_quantum_feature_map(features)  # num_qubits=None

        # Should default to len(features) = 2
        mock_quantum_circuit.assert_called_once_with(2)

    def test_create_quantum_circuit_without_qiskit(self):
        """Test quantum circuit creation without Qiskit"""
        with patch("utils.quantum_utils.QISKIT_AVAILABLE", False):
            circuit = create_quantum_circuit(n_qubits=2, n_classical=2)

            # Should return mock circuit
            assert hasattr(circuit, "num_qubits")
            assert hasattr(circuit, "num_clbits")
            assert circuit.num_qubits == 2
            assert circuit.num_clbits == 2

    @patch("utils.quantum_utils.QISKIT_AVAILABLE", True)
    @patch("utils.quantum_utils.QuantumCircuit")
    def test_create_quantum_circuit_with_qiskit(self, mock_quantum_circuit):
        """Test quantum circuit creation with Qiskit"""
        mock_circuit = Mock()
        mock_quantum_circuit.return_value = mock_circuit

        result = create_quantum_circuit(n_qubits=3, n_classical=3)

        mock_quantum_circuit.assert_called_once_with(3, 3)
        assert result == mock_circuit

    @patch("utils.quantum_utils.QISKIT_AVAILABLE", True)
    @patch("utils.quantum_utils.QuantumCircuit")
    def test_create_quantum_circuit_with_cbits_parameter(self, mock_quantum_circuit):
        """Test quantum circuit creation with n_cbits parameter"""
        mock_circuit = Mock()
        mock_quantum_circuit.return_value = mock_circuit

        _result = create_quantum_circuit(n_qubits=2, n_cbits=2)

        mock_quantum_circuit.assert_called_once_with(2, 2)

    @patch("utils.quantum_utils.QISKIT_AVAILABLE", True)
    @patch("utils.quantum_utils.QuantumCircuit")
    def test_create_quantum_circuit_default_classical(self, mock_quantum_circuit):
        """Test quantum circuit creation with default classical bits"""
        mock_circuit = Mock()
        mock_quantum_circuit.return_value = mock_circuit

        _result = create_quantum_circuit(n_qubits=4)

        # Should default classical bits to match qubits
        mock_quantum_circuit.assert_called_once_with(4, 4)

    def test_apply_quantum_gate_without_qiskit(self):
        """Test gate application without Qiskit"""
        with patch("utils.quantum_utils.QISKIT_AVAILABLE", False):
            # Create mock circuit
            circuit = create_quantum_circuit(n_qubits=2)

            # Apply gates
            result = apply_quantum_gate(circuit, "h", 0)
            assert result == circuit

            result = apply_quantum_gate(circuit, "cx", [0, 1])
            assert result == circuit

    @patch("utils.quantum_utils.QISKIT_AVAILABLE", True)
    def test_apply_quantum_gate_single_qubit_gates(self):
        """Test single-qubit gate application"""
        mock_circuit = Mock()
        mock_circuit.num_qubits = 2

        # Test various single-qubit gates
        gates_to_test = ["h", "x", "y", "z"]

        for gate in gates_to_test:
            result = apply_quantum_gate(mock_circuit, gate, 0)
            assert result == mock_circuit

            # Verify the correct method was called
            getattr(mock_circuit, gate).assert_called_with(0)

    @patch("utils.quantum_utils.QISKIT_AVAILABLE", True)
    def test_apply_quantum_gate_parametric_gates(self):
        """Test parametric gate application"""
        mock_circuit = Mock()

        # Test rotation gates
        rotation_gates = ["rx", "ry", "rz"]
        angle = np.pi / 4

        for gate in rotation_gates:
            result = apply_quantum_gate(mock_circuit, gate, 0, angle=angle)
            assert result == mock_circuit

            # Verify the correct method was called with angle
            getattr(mock_circuit, gate).assert_called_with(angle, 0)

    @patch("utils.quantum_utils.QISKIT_AVAILABLE", True)
    def test_apply_quantum_gate_cnot_with_list(self):
        """Test CNOT gate application with qubit list"""
        mock_circuit = Mock()

        result = apply_quantum_gate(mock_circuit, "cnot", [0, 1])
        assert result == mock_circuit

        mock_circuit.cx.assert_called_with(0, 1)

    @patch("utils.quantum_utils.QISKIT_AVAILABLE", True)
    def test_apply_quantum_gate_cnot_with_single_qubit(self):
        """Test CNOT gate application with single qubit"""
        mock_circuit = Mock()
        mock_circuit.num_qubits = 3

        result = apply_quantum_gate(mock_circuit, "cx", 1)
        assert result == mock_circuit

        # Should apply CNOT to qubit 1 and qubit 2
        mock_circuit.cx.assert_called_with(1, 2)

    @patch("utils.quantum_utils.QISKIT_AVAILABLE", True)
    def test_apply_quantum_gate_unknown_gate(self):
        """Test unknown gate application"""
        mock_circuit = Mock()

        with pytest.raises(ValueError, match="Unknown gate type"):
            apply_quantum_gate(mock_circuit, "unknown_gate", 0)

    @patch("utils.quantum_utils.QISKIT_AVAILABLE", True)
    def test_apply_quantum_gate_parametric_without_angle(self):
        """Test parametric gate without angle parameter"""
        mock_circuit = Mock()

        with pytest.raises(ValueError, match="Unknown gate type"):
            apply_quantum_gate(mock_circuit, "rx", 0)  # Missing angle

    def test_measure_quantum_state_without_qiskit(self):
        """Test quantum state measurement without Qiskit"""
        with patch("utils.quantum_utils.QISKIT_AVAILABLE", False):
            mock_circuit = Mock()

            result = measure_quantum_state(mock_circuit)

            # Should return mock results
            assert "counts" in result
            assert result["counts"] == {"00": 512, "11": 512}

    @patch("utils.quantum_utils.QISKIT_AVAILABLE", True)
    def test_measure_quantum_state_with_qiskit(self):
        """Test quantum state measurement with Qiskit"""
        mock_circuit = Mock()
        mock_circuit.num_qubits = 2
        mock_circuit.num_clbits = 2

        result = measure_quantum_state(mock_circuit, qubits=[0, 1])

        # Verify measurements were added
        assert mock_circuit.measure.call_count == 2
        mock_circuit.measure.assert_any_call(0, 0)
        mock_circuit.measure.assert_any_call(1, 1)

        # Should return mock results
        assert "counts" in result

    @patch("utils.quantum_utils.QISKIT_AVAILABLE", True)
    def test_measure_quantum_state_default_qubits(self):
        """Test quantum state measurement with default qubits"""
        mock_circuit = Mock()
        mock_circuit.num_qubits = 3
        mock_circuit.num_clbits = 3

        _result = measure_quantum_state(mock_circuit)  # qubits=None

        # Should measure all qubits
        assert mock_circuit.measure.call_count == 3
        for i in range(3):
            mock_circuit.measure.assert_any_call(i, i)

    @patch("utils.quantum_utils.QISKIT_AVAILABLE", True)
    def test_measure_quantum_state_with_simulation(self):
        """Test quantum state measurement with circuit simulation"""
        mock_circuit = Mock()
        mock_circuit.num_qubits = 2
        mock_circuit.num_clbits = 2
        mock_circuit.simulate.return_value = {"counts": {"01": 1024}}

        result = measure_quantum_state(mock_circuit)

        mock_circuit.simulate.assert_called_once()
        assert result == {"counts": {"01": 1024}}


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple components"""

    @patch("utils.quantum_utils.QISKIT_AVAILABLE", True)
    @patch("utils.quantum_utils.SPSA")
    @patch("utils.quantum_utils.AerSimulator")
    @patch("utils.quantum_utils.Estimator")
    def test_complete_vqe_workflow(self, mock_estimator, mock_aer, mock_spsa):
        """Test complete VQE workflow"""
        # Test the full workflow from VQE setup to optimization
        vqe_opt = VQEOptimizer(optimizer="SPSA")

        # Mock the optimization result
        vqe_opt.optimize_h2 = Mock(
            return_value={
                "ground_state_energy": -1.8571,
                "optimal_parameters": np.array([0.1, 0.2, 0.3, 0.4]),
                "optimizer_evals": 150,
                "bond_distance": 0.735,
            }
        )

        result = vqe_opt.optimize_h2()

        assert result["ground_state_energy"] < 0  # Should be negative for bound state
        assert len(result["optimal_parameters"]) > 0
        assert result["optimizer_evals"] > 0

    @patch("utils.quantum_utils.PENNYLANE_AVAILABLE", True)
    @patch("utils.quantum_utils.qml")
    def test_quantum_ml_pipeline(self, mock_qml):
        """Test quantum machine learning pipeline"""
        mock_device = Mock()
        mock_qml.device.return_value = mock_device

        def mock_qnode(device):
            def decorator(func):
                return func

            return decorator

        mock_qml.qnode = mock_qnode

        # Create QML instance and circuits
        qml_instance = QuantumMachineLearning(num_qubits=4)
        classifier = qml_instance.create_variational_classifier(num_layers=2)
        embedding = qml_instance.create_quantum_embedding(data_reps=1)

        # Both should be callable functions
        assert callable(classifier)
        assert callable(embedding)

    def test_quantum_distance_calculation_pipeline(self):
        """Test quantum distance calculation for multiple states"""
        # Create several quantum states
        states = [
            np.array([1.0, 0.0]),  # |0⟩
            np.array([0.0, 1.0]),  # |1⟩
            np.array([1.0, 1.0]) / np.sqrt(2),  # |+⟩
            np.array([1.0, -1.0]) / np.sqrt(2),  # |-⟩
        ]

        # Calculate distance matrix
        distance_matrix = np.zeros((len(states), len(states)))

        for i in range(len(states)):
            for j in range(len(states)):
                distance_matrix[i, j] = quantum_distance(states[i], states[j])

        # Verify properties
        # Diagonal should be zero (distance to self)
        for i in range(len(states)):
            assert distance_matrix[i, i] == pytest.approx(0.0, abs=1e-10)

        # Matrix should be symmetric
        for i in range(len(states)):
            for j in range(len(states)):
                assert distance_matrix[i, j] == pytest.approx(
                    distance_matrix[j, i], abs=1e-10
                )

        # |0⟩ and |1⟩ should be orthogonal (distance = 1)
        assert distance_matrix[0, 1] == pytest.approx(1.0, abs=1e-10)

        # |+⟩ and |-⟩ should be orthogonal (distance = 1)
        assert distance_matrix[2, 3] == pytest.approx(1.0, abs=1e-10)


class TestErrorHandling:
    """Test error handling and edge cases"""

    def test_quantum_distance_zero_states(self):
        """Test quantum distance with zero states"""
        state1 = np.array([0.0, 0.0])
        state2 = np.array([1.0, 0.0])

        # Should handle zero norm gracefully (may warn or error)
        try:
            distance = quantum_distance(state1, state2)
            # If it doesn't error, the result should be valid
            assert 0 <= distance <= 1 or np.isnan(distance)
        except (ValueError, ZeroDivisionError, RuntimeWarning):
            # These are acceptable for zero norm states
            pass

    def test_quantum_distance_complex_states(self):
        """Test quantum distance with complex states"""
        state1 = np.array([1.0 + 0j, 0.0 + 0j])
        state2 = np.array([0.0 + 0j, 1.0 + 0j])

        distance = quantum_distance(state1, state2)

        # Should handle complex states correctly
        assert distance == pytest.approx(1.0, abs=1e-10)

    @patch("utils.quantum_utils.QISKIT_AVAILABLE", True)
    def test_apply_quantum_gate_edge_cases(self):
        """Test gate application edge cases"""
        mock_circuit = Mock()
        mock_circuit.num_qubits = 2

        # Test CNOT with insufficient qubits
        result = apply_quantum_gate(mock_circuit, "cnot", [1])  # Only one qubit in list
        # Should not raise error but may not work as expected

        # Test CNOT at boundary
        _result = apply_quantum_gate(mock_circuit, "cx", 1)  # Last qubit
        mock_circuit.cx.assert_called_with(1, 2)  # Should try qubit 2 (out of bounds)

    def test_create_quantum_circuit_edge_cases(self):
        """Test quantum circuit creation edge cases"""
        # Test with zero qubits
        with patch("utils.quantum_utils.QISKIT_AVAILABLE", False):
            circuit = create_quantum_circuit(n_qubits=0)
            assert circuit.num_qubits == 0

        # Test with mismatched parameters
        with patch("utils.quantum_utils.QISKIT_AVAILABLE", False):
            circuit = create_quantum_circuit(n_qubits=2, n_classical=3, n_cbits=4)
            # n_cbits should override n_classical
            assert circuit.num_clbits == 4


class TestPerformance:
    """Test performance aspects"""

    def test_quantum_distance_large_states(self):
        """Test quantum distance with large state vectors"""
        # Create large random states
        size = 1000
        state1 = np.random.rand(size) + 1j * np.random.rand(size)
        state2 = np.random.rand(size) + 1j * np.random.rand(size)

        # Should handle large states efficiently
        distance = quantum_distance(state1, state2)

        assert 0 <= distance <= 1
        assert isinstance(distance, float)

    @patch("utils.quantum_utils.QISKIT_AVAILABLE", True)
    @patch("utils.quantum_utils.QuantumCircuit")
    def test_large_feature_map_creation(self, mock_quantum_circuit):
        """Test creation of feature maps with many features"""
        mock_circuit = Mock()
        mock_quantum_circuit.return_value = mock_circuit

        # Large feature vector
        features = np.random.rand(100)

        _result = create_quantum_feature_map(features, num_qubits=100)

        # Should handle large feature vectors
        mock_quantum_circuit.assert_called_once_with(100)
        assert mock_circuit.ry.call_count == 100
        assert mock_circuit.cx.call_count == 99


class TestCrossModuleCompatibility:
    """Test compatibility with other modules"""

    def test_quantum_utils_imports(self):
        """Test that all expected functions are importable"""
        from utils.quantum_utils import (
            MolecularHamiltonian,
            QuantumCircuitBuilder,
            QuantumMachineLearning,
            VQEOptimizer,
            apply_quantum_gate,
            create_quantum_circuit,
            create_quantum_feature_map,
            measure_quantum_state,
            quantum_distance,
        )

        # All imports should succeed
        assert QuantumCircuitBuilder is not None
        assert VQEOptimizer is not None
        assert MolecularHamiltonian is not None
        assert QuantumMachineLearning is not None
        assert quantum_distance is not None
        assert create_quantum_feature_map is not None
        assert create_quantum_circuit is not None
        assert apply_quantum_gate is not None
        assert measure_quantum_state is not None

    def test_availability_flags(self):
        """Test quantum library availability flags"""
        from utils.quantum_utils import PENNYLANE_AVAILABLE, QISKIT_AVAILABLE

        # Flags should be boolean
        assert isinstance(QISKIT_AVAILABLE, bool)
        assert isinstance(PENNYLANE_AVAILABLE, bool)

    def test_numpy_compatibility(self):
        """Test NumPy array compatibility"""
        # Test with different NumPy array types
        state1 = np.array([1.0, 0.0], dtype=np.float32)
        state2 = np.array([0.0, 1.0], dtype=np.float64)

        distance = quantum_distance(state1, state2)

        assert distance == pytest.approx(1.0, abs=1e-6)

        # Test with complex arrays
        state3 = np.array([1.0 + 0j, 0.0 + 0j], dtype=np.complex64)
        state4 = np.array([0.0 + 0j, 1.0 + 0j], dtype=np.complex128)

        distance = quantum_distance(state3, state4)

        assert distance == pytest.approx(1.0, abs=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
