"""
VQA Algorithms: VQE and QAOA implementations
=============================================

This module provides high-level implementations of Variational Quantum Algorithms
including Variational Quantum Eigensolver (VQE) and Quantum Approximate
Optimization Algorithm (QAOA).
"""

from openvqa.algorithms.vqe import VQE
from openvqa.algorithms.qaoa import QAOA

__all__ = ["VQE", "QAOA"]
