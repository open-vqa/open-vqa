"""
VQA Engine: Core execution engine for variational quantum algorithms
====================================================================

This module provides the main VQA_myQLM class that orchestrates quantum circuit
execution, optimization, and result collection across different backends.
"""

import time
import numpy as np
from typing import Dict, Any, Optional, Tuple
from qat.qpus import get_default_qpu
from qat.plugins import ScipyMinimizePlugin, ObservableSplitter
from qiskit_algorithms.optimizers import SLSQP, COBYLA, NELDER_MEAD


class VQAResults:
    """Container for VQA execution results with scientific output formatting."""
    
    def __init__(self):
        self.energy: float = None
        self.optimal_parameters: np.ndarray = None
        self.n_qubits: int = None
        self.circuit_depth: int = None
        self.n_evaluations: int = None
        self.convergence: bool = False
        self.barren_plateau_detected: bool = False
        self.wall_time: float = 0.0
        self.backend: str = "myQLM (LinAlg)"
        self.optimizer_name: str = None
        self.intermediate_energies: list = []
        self.eigenvalues: np.ndarray = None
        self.success: bool = False
    
    def __repr__(self) -> str:
        """Print results in a formatted table."""
        lines = [
            "\n" + "=" * 70,
            "VQA OPTIMIZATION RESULTS",
            "=" * 70,
            f"Status                       : {'✓ Success' if self.success else '✗ Failed'}",
            f"Ground-State Energy (Ha)     : {self.energy:.10f}" if self.energy else "",
            f"Optimal Parameters           : {self.optimal_parameters}",
            f"Number of Qubits             : {self.n_qubits}",
            f"Circuit Depth               : {self.circuit_depth}",
            f"Optimizer                    : {self.optimizer_name}",
            f"Function Evaluations        : {self.n_evaluations}",
            f"Converged                    : {self.convergence}",
            f"Barren Plateau Detected     : {self.barren_plateau_detected}",
            f"Execution Time (s)          : {self.wall_time:.2f}",
            f"Backend Used                : {self.backend}",
            "=" * 70,
        ]
        return "\n".join([line for line in lines if line])


class VQA_myQLM:
    """
    Main VQA Engine using myQLM (QLM) backend.
    
    This class manages quantum circuit execution, optimization, and result tracking
    for variational quantum algorithms.
    
    Parameters:
        n_qubits (int): Number of qubits to allocate
        backend (str): Backend simulator type (default: "qat.simulators.qubit")
        shots (int): Number of measurement shots (default: 1000)
        seed (int): Random seed for reproducibility (default: None)
        verbose (bool): Enable verbose output (default: False)
    
    Example:
        >>> engine = VQA_myQLM(n_qubits=4, shots=1000, seed=42)
        >>> from openvqa.algorithms import VQE
        >>> algorithm = VQE(hamiltonian="H2", ansatz="uccsd")
        >>> results = engine.run(algorithm)
    """
    
    def __init__(
        self,
        n_qubits: int = 4,
        backend: str = "qat.simulators.qubit",
        shots: int = 1000,
        seed: Optional[int] = None,
        verbose: bool = False,
        use_pennylane_via_qlm: bool = False,
    ):
        self.n_qubits = n_qubits
        self.backend_name = backend
        self.is_pennylane = isinstance(backend, str) and "pennylane" in backend.lower()
        self.shots = shots
        self.seed = seed
        self.verbose = verbose
        self.use_pennylane_via_qlm = use_pennylane_via_qlm
        
        # Initialize QPU (used for non-PennyLane backends)
        self.qpu = None if self.is_pennylane else get_default_qpu()
        
        if seed is not None:
            np.random.seed(seed)
        
        if verbose:
            print(f"✓ VQA Engine initialized")
            print(f"  Qubits: {n_qubits}")
            print(f"  Backend: {backend}")
            print(f"  Shots: {shots}")
            if seed:
                print(f"  Seed: {seed}")
    
    def run(self, algorithm) -> VQAResults:
        """
        Execute a VQA algorithm.
        
        Parameters:
            algorithm: A VQA algorithm instance (VQE or QAOA)
        
        Returns:
            VQAResults: Results object containing energy, parameters, metrics
        """
        if self.verbose:
            print(f"\n{'=' * 70}")
            print(f"Executing {algorithm.__class__.__name__} algorithm")
            print(f"{'=' * 70}")
        
        start_time = time.time()
        
        try:
            # Run the algorithm. If configured to convert PennyLane templates to
            # QLM jobs (`use_pennylane_via_qlm`), prefer that execution path when
            # the algorithm provides it. Otherwise, if the engine is configured
            # with a PennyLane backend and the algorithm supports direct
            # PennyLane execution, use that. Fallback to the QLM execution path.
            if self.use_pennylane_via_qlm and hasattr(algorithm, 'execute_pennylane_via_qlm'):
                results = algorithm.execute_pennylane_via_qlm(self)
            elif self.is_pennylane and hasattr(algorithm, 'execute_pennylane'):
                results = algorithm.execute_pennylane(self)
            else:
                results = algorithm.execute(self, self.qpu)
            results.backend = self.backend_name
            results.wall_time = time.time() - start_time
            results.success = True
            
            if self.verbose:
                print(results)
            
            return results
        
        except Exception as e:
            print(f"✗ Error during execution: {e}")
            results = VQAResults()
            results.success = False
            results.wall_time = time.time() - start_time
            return results
    
    def create_optimizer(self, optimizer_name: str = "cobyla", max_iterations: int = 100):
        """
        Create an optimizer plugin for QLM.
        
        Parameters:
            optimizer_name (str): Name of optimizer (cobyla, nelder-mead, bfgs, slsqp)
            max_iterations (int): Maximum number of iterations
        
        Returns:
            Optimizer plugin for QLM
        """
        optimizer_name_lower = optimizer_name.lower()
        
        if optimizer_name_lower in ["cobyla", "nelder-mead", "bfgs"]:
            return ScipyMinimizePlugin(
                method=optimizer_name_lower,
                tol=1e-6,
                options={"maxiter": max_iterations}
            )
        elif optimizer_name_lower == "slsqp":
            return ScipyMinimizePlugin(
                method="SLSQP",
                tol=1e-6,
                options={"maxiter": max_iterations}
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
