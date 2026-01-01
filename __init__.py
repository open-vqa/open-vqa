"""
OpenVQA: A high-level quantum variational algorithm framework
=========================================================

This package provides an easy-to-use interface for running Variational Quantum
Algorithms (VQA) including VQE and QAOA on multiple quantum backends.

Features:
    - Unified interface for VQA algorithms (VQE, QAOA)
    - Support for multiple backends (QLM, Qiskit, etc.)
    - Pre-built ansatze and Hamiltonians
    - Automatic optimization and result tracking
    - Scientific output formatting

Example:
    >>> from openvqa import VQA_myQLM
    >>> from openvqa.algorithms import VQE
    >>> 
    >>> engine = VQA_myQLM(n_qubits=4, shots=1000)
    >>> algorithm = VQE(hamiltonian="H2", ansatz="uccsd", optimizer="cobyla")
    >>> results = engine.run(algorithm)
    >>> print(f"Ground state energy: {results.energy:.10f}")
"""

from openvqa.engine import VQA_myQLM
from openvqa.algorithms import VQE, QAOA
from openvqa.options import AVAILABLE_OPTIONS

__version__ = "0.1.0"
__author__ = "OpenVQA Contributors"

__all__ = [
    "VQA_myQLM",
    "VQE",
    "QAOA",
    "AVAILABLE_OPTIONS",
]
