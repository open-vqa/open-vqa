from enum import Enum

class QuantumCircuitWrapper(Enum):
    QISKIT = 'qiskit'
    CIRQ = 'cirq'
    QLM = 'qlm'
