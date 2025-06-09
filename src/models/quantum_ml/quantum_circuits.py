from qiskit import QuantumCircuit, transpile, assemble, Aer, execute
import numpy as np

class QuantumMLCircuit:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.circuit = QuantumCircuit(num_qubits)

    def add_hadamard(self, qubit):
        self.circuit.h(qubit)

    def add_cnot(self, control_qubit, target_qubit):
        self.circuit.cx(control_qubit, target_qubit)

    def add_rotation(self, qubit, angle):
        self.circuit.ry(angle, qubit)

    def measure(self):
        self.circuit.measure_all()

    def simulate(self):
        backend = Aer.get_backend('qasm_simulator')
        transpiled_circuit = transpile(self.circuit, backend)
        qobj = assemble(transpiled_circuit)
        result = execute(qobj, backend).result()
        counts = result.get_counts(self.circuit)
        return counts

    def get_circuit(self):
        return self.circuit

    def reset_circuit(self):
        self.circuit = QuantumCircuit(self.num_qubits)