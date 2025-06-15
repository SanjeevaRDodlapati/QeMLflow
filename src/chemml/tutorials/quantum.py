"""
Quantum Computing Integration for ChemML Tutorials
=================================================

This module provides quantum computing educational components and integration
with quantum chemistry libraries for ChemML tutorials.

Key Features:
- Quantum circuit visualization for chemistry
- VQE (Variational Quantum Eigensolver) tutorials
- Quantum molecular simulation examples
- Quantum machine learning demonstrations
- Integration with Qiskit and quantum chemistry packages
"""

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Core dependencies
try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Quantum computing dependencies
try:
    import qiskit
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    from qiskit.circuit import Parameter
    from qiskit.primitives import Estimator
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.visualization import circuit_drawer, plot_histogram

    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    warnings.warn("Qiskit not available. Quantum features will be limited.")

# Quantum chemistry dependencies
try:
    import psi4

    PSI4_AVAILABLE = True
except ImportError:
    PSI4_AVAILABLE = False

# Chemistry dependencies
try:
    from rdkit import Chem

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

# Optional visualization
try:
    import ipywidgets as widgets
    from IPython.display import HTML, display

    WIDGETS_AVAILABLE = True
except ImportError:
    WIDGETS_AVAILABLE = False

# Set up logging
logger = logging.getLogger(__name__)


class QuantumChemistryTutorial:
    """
    Educational quantum chemistry tutorial framework.

    This class provides interactive quantum chemistry tutorials including
    VQE demonstrations, quantum molecular simulation, and quantum machine learning.
    """

    def __init__(self, molecule_name: str = "H2"):
        """
        Initialize quantum chemistry tutorial.

        Args:
            molecule_name (str): Name of the molecule for tutorial
        """
        self.molecule_name = molecule_name
        self.quantum_circuit = None
        self.hamiltonian = None
        self.parameters = {}
        self.results = {}

    def create_h2_tutorial(self, bond_distance: float = 0.74) -> Dict[str, Any]:
        """
        Create H2 molecule tutorial with VQE demonstration.

        Args:
            bond_distance (float): H-H bond distance in Angstroms

        Returns:
            Dict[str, Any]: Tutorial components and results
        """
        tutorial_data = {
            "molecule": "H2",
            "bond_distance": bond_distance,
            "description": "Hydrogen molecule VQE tutorial",
            "components": {},
        }

        if not QISKIT_AVAILABLE:
            tutorial_data["components"][
                "error"
            ] = "Qiskit not available for quantum simulation"
            return tutorial_data

        # Create simple H2 Hamiltonian (educational approximation)
        h2_hamiltonian = self._create_h2_hamiltonian(bond_distance)
        tutorial_data["components"]["hamiltonian"] = h2_hamiltonian

        # Create VQE ansatz circuit
        vqe_circuit = self._create_vqe_ansatz(num_qubits=2)
        tutorial_data["components"]["circuit"] = vqe_circuit

        # Create visualization
        if MATPLOTLIB_AVAILABLE:
            circuit_visualization = self._visualize_circuit(vqe_circuit)
            tutorial_data["components"]["circuit_plot"] = circuit_visualization

        # Educational explanation
        tutorial_data["components"]["explanation"] = self._create_h2_explanation(
            bond_distance
        )

        return tutorial_data

    def create_quantum_feature_mapping(
        self, classical_data: np.ndarray
    ) -> Dict[str, Any]:
        """
        Demonstrate quantum feature mapping for machine learning.

        Args:
            classical_data (np.ndarray): Classical data to map to quantum features

        Returns:
            Dict[str, Any]: Quantum feature mapping demonstration
        """
        mapping_data = {
            "input_data": classical_data,
            "description": "Quantum feature mapping for ML",
            "components": {},
        }

        if not QISKIT_AVAILABLE:
            mapping_data["components"]["error"] = "Qiskit not available"
            return mapping_data

        # Create feature mapping circuit
        num_features = min(
            classical_data.shape[-1]
            if classical_data.ndim > 1
            else len(classical_data),
            4,
        )
        feature_map = self._create_feature_map(num_features)
        mapping_data["components"]["feature_map"] = feature_map

        # Demonstrate encoding
        encoded_circuit = self._encode_classical_data(
            feature_map, classical_data[:num_features]
        )
        mapping_data["components"]["encoded_circuit"] = encoded_circuit

        # Create visualization
        if MATPLOTLIB_AVAILABLE:
            mapping_visualization = self._visualize_feature_mapping(
                feature_map, classical_data[:num_features]
            )
            mapping_data["components"]["visualization"] = mapping_visualization

        return mapping_data

    def create_quantum_molecular_simulation(
        self, molecule_smiles: str
    ) -> Dict[str, Any]:
        """
        Create quantum molecular simulation tutorial.

        Args:
            molecule_smiles (str): SMILES string of the molecule

        Returns:
            Dict[str, Any]: Molecular simulation components
        """
        simulation_data = {
            "molecule_smiles": molecule_smiles,
            "description": "Quantum molecular simulation tutorial",
            "components": {},
        }

        if not RDKIT_AVAILABLE:
            simulation_data["components"][
                "warning"
            ] = "RDKit not available for molecular processing"

        # Process molecule
        if RDKIT_AVAILABLE:
            mol = Chem.MolFromSmiles(molecule_smiles)
            if mol is not None:
                simulation_data["components"][
                    "molecule_info"
                ] = self._extract_molecule_info(mol)
            else:
                simulation_data["components"][
                    "error"
                ] = f"Invalid SMILES: {molecule_smiles}"
                return simulation_data

        # Create simplified quantum simulation
        if QISKIT_AVAILABLE:
            # Estimate number of qubits needed (simplified)
            num_atoms = mol.GetNumAtoms() if RDKIT_AVAILABLE and mol else 2
            num_qubits = min(num_atoms * 2, 8)  # Limit for educational purposes

            simulation_circuit = self._create_molecular_simulation_circuit(num_qubits)
            simulation_data["components"]["simulation_circuit"] = simulation_circuit

            # Create educational Hamiltonian
            mock_hamiltonian = self._create_educational_hamiltonian(num_qubits)
            simulation_data["components"]["hamiltonian"] = mock_hamiltonian

        return simulation_data

    def create_interactive_vqe_demo(self) -> Any:
        """
        Create interactive VQE demonstration widget.

        Returns:
            Interactive VQE demo widget or fallback interface
        """
        if not WIDGETS_AVAILABLE:
            return self._create_fallback_vqe_demo()

        if not QISKIT_AVAILABLE:
            return widgets.HTML("<p>Qiskit not available for VQE demonstration</p>")

        # Create parameter sliders
        theta_slider = widgets.FloatSlider(
            value=0.0,
            min=0.0,
            max=2 * np.pi,
            step=0.1,
            description="θ (theta):",
            readout_format=".2f",
        )

        phi_slider = widgets.FloatSlider(
            value=0.0,
            min=0.0,
            max=2 * np.pi,
            step=0.1,
            description="φ (phi):",
            readout_format=".2f",
        )

        # Output area
        output = widgets.Output()

        def update_vqe_demo(change):
            """Update VQE demonstration when parameters change."""
            with output:
                output.clear_output(wait=True)

                theta = theta_slider.value
                phi = phi_slider.value

                # Create parametric circuit
                circuit = self._create_parametric_vqe_circuit(theta, phi)

                # Calculate expected energy (mock calculation)
                energy = self._calculate_mock_energy(theta, phi)

                print(f"🔬 VQE Parameters:")
                print(f"θ = {theta:.2f}, φ = {phi:.2f}")
                print(f"📊 Expected Energy: {energy:.4f}")

                # Display circuit
                if MATPLOTLIB_AVAILABLE:
                    try:
                        fig = circuit_drawer(circuit, output="mpl", style="clifford")
                        plt.show()
                    except Exception:
                        print("Circuit visualization not available")

        # Attach observers
        theta_slider.observe(update_vqe_demo, names="value")
        phi_slider.observe(update_vqe_demo, names="value")

        # Initial display
        update_vqe_demo(None)

        # Create interface
        controls = widgets.VBox([theta_slider, phi_slider])
        interface = widgets.HBox([controls, output])

        return interface

    def _create_h2_hamiltonian(self, bond_distance: float) -> Dict[str, Any]:
        """Create educational H2 Hamiltonian."""
        if not QISKIT_AVAILABLE:
            return {"error": "Qiskit not available"}

        # Simplified H2 Hamiltonian for educational purposes
        # This is a mock implementation for demonstration

        # Pauli operators for H2 (simplified)
        pauli_strings = ["II", "ZZ", "XX", "YY"]
        coefficients = [-1.0523732, 0.39793742, -0.39793742, -0.01128010]

        # Adjust coefficients based on bond distance (mock relationship)
        distance_factor = np.exp(-abs(bond_distance - 0.74))
        adjusted_coeffs = [coeff * distance_factor for coeff in coefficients]

        try:
            hamiltonian = SparsePauliOp(pauli_strings, adjusted_coeffs)
            return {
                "hamiltonian": hamiltonian,
                "pauli_strings": pauli_strings,
                "coefficients": adjusted_coeffs,
                "bond_distance": bond_distance,
            }
        except Exception as e:
            return {"error": f"Error creating Hamiltonian: {e}"}

    def _create_vqe_ansatz(self, num_qubits: int) -> QuantumCircuit:
        """Create VQE ansatz circuit."""
        if not QISKIT_AVAILABLE:
            return None

        # Create parameterized circuit
        circuit = QuantumCircuit(num_qubits)

        # Add parameterized gates
        theta = Parameter("θ")
        phi = Parameter("φ")

        # Simple ansatz: RY rotations + entangling gates
        for i in range(num_qubits):
            circuit.ry(theta, i)

        # Entangling gates
        for i in range(num_qubits - 1):
            circuit.cx(i, i + 1)

        # Additional parameterized layer
        for i in range(num_qubits):
            circuit.rz(phi, i)

        return circuit

    def _create_feature_map(self, num_features: int) -> QuantumCircuit:
        """Create quantum feature mapping circuit."""
        if not QISKIT_AVAILABLE:
            return None

        circuit = QuantumCircuit(num_features)

        # Create parameters for each feature
        parameters = [Parameter(f"x_{i}") for i in range(num_features)]

        # Encode features using RY rotations
        for i, param in enumerate(parameters):
            circuit.ry(param, i)

        # Add entangling gates for feature interaction
        for i in range(num_features - 1):
            circuit.cx(i, i + 1)

        # Second layer with RZ rotations
        for i, param in enumerate(parameters):
            circuit.rz(param, i)

        return circuit

    def _encode_classical_data(
        self, feature_map: QuantumCircuit, data: np.ndarray
    ) -> QuantumCircuit:
        """Encode classical data into quantum circuit."""
        if not QISKIT_AVAILABLE or feature_map is None:
            return None

        # Bind parameters with data
        parameter_dict = {}
        for i, param in enumerate(feature_map.parameters):
            if i < len(data):
                parameter_dict[param] = float(data[i])

        bound_circuit = feature_map.bind_parameters(parameter_dict)
        return bound_circuit

    def _create_molecular_simulation_circuit(self, num_qubits: int) -> QuantumCircuit:
        """Create molecular simulation circuit."""
        if not QISKIT_AVAILABLE:
            return None

        circuit = QuantumCircuit(num_qubits)

        # Initialize molecular ground state (simplified)
        for i in range(0, num_qubits, 2):
            circuit.x(i)  # Fill some orbitals

        # Add molecular interactions (simplified)
        for i in range(num_qubits - 1):
            circuit.cx(i, i + 1)
            circuit.rz(0.1, i + 1)  # Mock interaction

        return circuit

    def _create_educational_hamiltonian(self, num_qubits: int) -> Dict[str, Any]:
        """Create educational Hamiltonian for molecular simulation."""
        if not QISKIT_AVAILABLE:
            return {"error": "Qiskit not available"}

        # Create simple Hamiltonian with random coefficients for education
        pauli_strings = []
        coefficients = []

        # Add single-qubit terms
        for i in range(num_qubits):
            pauli_strings.append("I" * i + "Z" + "I" * (num_qubits - i - 1))
            coefficients.append(np.random.uniform(-0.5, 0.5))

        # Add two-qubit terms
        for i in range(num_qubits - 1):
            pauli_string = "I" * i + "ZZ" + "I" * (num_qubits - i - 2)
            pauli_strings.append(pauli_string)
            coefficients.append(np.random.uniform(-0.2, 0.2))

        try:
            hamiltonian = SparsePauliOp(pauli_strings, coefficients)
            return {
                "hamiltonian": hamiltonian,
                "pauli_strings": pauli_strings,
                "coefficients": coefficients,
                "num_qubits": num_qubits,
            }
        except Exception as e:
            return {"error": f"Error creating Hamiltonian: {e}"}

    def _visualize_circuit(self, circuit: QuantumCircuit) -> Any:
        """Visualize quantum circuit."""
        if not MATPLOTLIB_AVAILABLE or circuit is None:
            return None

        try:
            fig = circuit_drawer(circuit, output="mpl", style="clifford")
            return fig
        except Exception as e:
            logger.warning(f"Circuit visualization failed: {e}")
            return None

    def _visualize_feature_mapping(
        self, feature_map: QuantumCircuit, data: np.ndarray
    ) -> Any:
        """Visualize quantum feature mapping."""
        if not MATPLOTLIB_AVAILABLE:
            return None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Plot classical data
        ax1.bar(range(len(data)), data, alpha=0.7)
        ax1.set_title("Classical Data")
        ax1.set_xlabel("Feature Index")
        ax1.set_ylabel("Value")

        # Plot quantum encoding representation
        angles = data * np.pi  # Convert to rotation angles
        ax2.bar(range(len(angles)), angles, alpha=0.7, color="red")
        ax2.set_title("Quantum Encoding (Rotation Angles)")
        ax2.set_xlabel("Qubit Index")
        ax2.set_ylabel("Rotation Angle (radians)")

        plt.tight_layout()
        return fig

    def _extract_molecule_info(self, mol) -> Dict[str, Any]:
        """Extract molecular information for tutorial."""
        if not RDKIT_AVAILABLE:
            return {}

        return {
            "num_atoms": mol.GetNumAtoms(),
            "num_bonds": mol.GetNumBonds(),
            "molecular_formula": Chem.rdMolDescriptors.CalcMolFormula(mol),
            "molecular_weight": Chem.rdMolDescriptors.CalcExactMolWt(mol),
        }

    def _create_parametric_vqe_circuit(
        self, theta: float, phi: float
    ) -> QuantumCircuit:
        """Create VQE circuit with specific parameter values."""
        if not QISKIT_AVAILABLE:
            return None

        circuit = QuantumCircuit(2)

        # Apply rotation gates with given parameters
        circuit.ry(theta, 0)
        circuit.ry(theta, 1)
        circuit.cx(0, 1)
        circuit.rz(phi, 0)
        circuit.rz(phi, 1)

        return circuit

    def _calculate_mock_energy(self, theta: float, phi: float) -> float:
        """Calculate mock energy for VQE demonstration."""
        # Simple mock energy function for educational purposes
        return -1.0 + 0.5 * np.cos(theta) + 0.3 * np.sin(phi)

    def _create_h2_explanation(self, bond_distance: float) -> str:
        """Create educational explanation for H2 tutorial."""
        return f"""
        🎓 H2 Molecule VQE Tutorial

        In this tutorial, we explore the hydrogen molecule (H2) using the
        Variational Quantum Eigensolver (VQE) algorithm.

        📏 Bond Distance: {bond_distance:.2f} Å

        🔬 Key Concepts:
        • VQE finds the ground state energy of molecules
        • Quantum circuits represent molecular wavefunctions
        • Parametric gates allow optimization of the wavefunction
        • The Hamiltonian encodes molecular interactions

        🎯 Learning Objectives:
        1. Understand quantum representation of molecules
        2. Learn how VQE optimizes quantum circuits
        3. Explore the relationship between bond distance and energy
        4. Connect quantum computing to chemistry problems
        """

    def _create_fallback_vqe_demo(self) -> str:
        """Create fallback VQE demo when widgets are not available."""
        return """
        🔬 VQE Interactive Demo (Fallback Mode)

        This would be an interactive VQE demonstration where you could:
        • Adjust circuit parameters (θ, φ)
        • See real-time energy calculations
        • Visualize quantum circuits
        • Explore parameter optimization

        💡 Install ipywidgets for interactive features:
           pip install ipywidgets

        🔧 Install Qiskit for quantum simulation:
           pip install qiskit
        """


class QuantumMachineLearning:
    """
    Quantum machine learning tutorial components.

    This class provides educational examples of quantum machine learning
    algorithms and their applications to chemistry problems.
    """

    def __init__(self):
        """Initialize quantum ML tutorial."""
        self.classifiers = {}
        self.datasets = {}

    def create_qsvm_tutorial(
        self, classical_data: np.ndarray, labels: np.ndarray
    ) -> Dict[str, Any]:
        """
        Create Quantum Support Vector Machine tutorial.

        Args:
            classical_data (np.ndarray): Training data
            labels (np.ndarray): Training labels

        Returns:
            Dict[str, Any]: QSVM tutorial components
        """
        tutorial_data = {
            "algorithm": "Quantum SVM",
            "data_shape": classical_data.shape,
            "num_classes": len(np.unique(labels)),
            "components": {},
        }

        if not QISKIT_AVAILABLE:
            tutorial_data["components"]["error"] = "Qiskit not available for QSVM"
            return tutorial_data

        # Create quantum feature map
        num_features = min(classical_data.shape[1], 4)  # Limit for education
        feature_map = self._create_qml_feature_map(num_features)
        tutorial_data["components"]["feature_map"] = feature_map

        # Mock QSVM training (educational)
        qsvm_results = self._mock_qsvm_training(classical_data[:10], labels[:10])
        tutorial_data["components"]["training_results"] = qsvm_results

        return tutorial_data

    def create_vqc_tutorial(self, num_qubits: int = 3) -> Dict[str, Any]:
        """
        Create Variational Quantum Classifier tutorial.

        Args:
            num_qubits (int): Number of qubits for the classifier

        Returns:
            Dict[str, Any]: VQC tutorial components
        """
        tutorial_data = {
            "algorithm": "Variational Quantum Classifier",
            "num_qubits": num_qubits,
            "components": {},
        }

        if not QISKIT_AVAILABLE:
            tutorial_data["components"]["error"] = "Qiskit not available for VQC"
            return tutorial_data

        # Create VQC ansatz
        vqc_circuit = self._create_vqc_ansatz(num_qubits)
        tutorial_data["components"]["ansatz"] = vqc_circuit

        # Create measurement circuit
        measurement_circuit = self._create_measurement_circuit(num_qubits)
        tutorial_data["components"]["measurement"] = measurement_circuit

        return tutorial_data

    def _create_qml_feature_map(self, num_features: int) -> QuantumCircuit:
        """Create quantum ML feature map."""
        if not QISKIT_AVAILABLE:
            return None

        circuit = QuantumCircuit(num_features)

        # Create parameters
        parameters = [Parameter(f"x_{i}") for i in range(num_features)]

        # Feature encoding
        for i, param in enumerate(parameters):
            circuit.h(i)
            circuit.rz(param, i)

        # Feature interaction
        for i in range(num_features - 1):
            circuit.cx(i, i + 1)
            circuit.rz(parameters[i] * parameters[i + 1], i + 1)

        return circuit

    def _mock_qsvm_training(
        self, data: np.ndarray, labels: np.ndarray
    ) -> Dict[str, Any]:
        """Mock QSVM training for educational purposes."""
        return {
            "training_samples": len(data),
            "accuracy": 0.85 + np.random.uniform(-0.1, 0.1),  # Mock accuracy
            "support_vectors": np.random.randint(2, len(data)),
            "training_time": np.random.uniform(1.0, 5.0),  # Mock time
            "convergence": True,
        }

    def _create_vqc_ansatz(self, num_qubits: int) -> QuantumCircuit:
        """Create VQC ansatz circuit."""
        if not QISKIT_AVAILABLE:
            return None

        circuit = QuantumCircuit(num_qubits)

        # Create parameters
        params = [Parameter(f"θ_{i}") for i in range(num_qubits * 2)]

        # First layer: RY rotations
        for i in range(num_qubits):
            circuit.ry(params[i], i)

        # Entangling layer
        for i in range(num_qubits - 1):
            circuit.cx(i, i + 1)

        # Second layer: RZ rotations
        for i in range(num_qubits):
            circuit.rz(params[i + num_qubits], i)

        return circuit

    def _create_measurement_circuit(self, num_qubits: int) -> QuantumCircuit:
        """Create measurement circuit for classification."""
        if not QISKIT_AVAILABLE:
            return None

        circuit = QuantumCircuit(num_qubits, 1)  # 1 classical bit for classification

        # Measure first qubit for binary classification
        circuit.measure(0, 0)

        return circuit


# Convenience functions for easy tutorial creation


def create_h2_vqe_tutorial(bond_distance: float = 0.74) -> Dict[str, Any]:
    """
    Create H2 VQE tutorial.

    Args:
        bond_distance (float): H-H bond distance in Angstroms

    Returns:
        Dict[str, Any]: Complete H2 VQE tutorial
    """
    tutorial = QuantumChemistryTutorial("H2")
    return tutorial.create_h2_tutorial(bond_distance)


def create_quantum_ml_demo(data: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    """
    Create quantum machine learning demonstration.

    Args:
        data (np.ndarray): Training data
        labels (np.ndarray): Training labels

    Returns:
        Dict[str, Any]: Quantum ML demo components
    """
    qml = QuantumMachineLearning()
    return qml.create_qsvm_tutorial(data, labels)


def check_quantum_requirements() -> Dict[str, bool]:
    """
    Check quantum computing tutorial requirements.

    Returns:
        Dict[str, bool]: Status of quantum dependencies
    """
    return {
        "qiskit": QISKIT_AVAILABLE,
        "psi4": PSI4_AVAILABLE,
        "rdkit": RDKIT_AVAILABLE,
        "matplotlib": MATPLOTLIB_AVAILABLE,
        "widgets": WIDGETS_AVAILABLE,
    }


def get_quantum_tutorial_overview() -> str:
    """
    Get overview of available quantum tutorials.

    Returns:
        str: Overview of quantum tutorial capabilities
    """
    status = check_quantum_requirements()

    overview = """
    🌌 ChemML Quantum Computing Tutorials
    =====================================

    Available Tutorials:

    🔬 Quantum Chemistry:
    • H2 molecule VQE demonstration
    • Molecular Hamiltonian construction
    • Quantum molecular simulation
    • Interactive parameter optimization

    🤖 Quantum Machine Learning:
    • Quantum Support Vector Machines (QSVM)
    • Variational Quantum Classifiers (VQC)
    • Quantum feature mapping
    • Hybrid classical-quantum algorithms

    📊 Interactive Components:
    • Real-time circuit visualization
    • Parameter tuning interfaces
    • Energy landscape exploration
    • Quantum state visualization

    Requirements Status:
    """

    for package, available in status.items():
        emoji = "✅" if available else "❌"
        overview += f"    {emoji} {package}\n"

    if not status["qiskit"]:
        overview += "\n💡 Install Qiskit for quantum features: pip install qiskit"

    return overview
