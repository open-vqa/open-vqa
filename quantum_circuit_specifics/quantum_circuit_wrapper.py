from enum import Enum

# TODO mhh: remove circuit_type.ciruit and adapt this one for circuit type
class QuantumCircuitWrapper(Enum):
    QISKIT = 'qiskit'
    CIRQ = 'cirq'
    QLM = 'qlm'
