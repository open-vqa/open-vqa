"""
VQA Engine: Core execution engine for variational quantum algorithms
====================================================================

This module provides the main VQA_myQLM class that orchestrates quantum circuit
execution, optimization, and result collection across different backends.
"""

import ast
import re
import time
import numpy as np
from typing import Dict, Any, Optional, Tuple
from qat.qpus import get_default_qpu
from qat.plugins import ScipyMinimizePlugin


class VQAResults:
    """Container for VQA execution results with scientific output formatting.
    
    Diagnostics that were not measured stay ``None`` and are displayed as
    "Not reported" / "Not assessed", never as a successful check.
    """
    
    def __init__(self):
        self.energy: float = None
        self.optimal_parameters: np.ndarray = None
        self.n_qubits: int = None
        self.circuit_depth: int = None
        self.n_evaluations: int = None
        self.convergence: Optional[bool] = None
        self.barren_plateau_detected: Optional[bool] = None
        self.wall_time: float = 0.0
        self.backend: str = None
        self.shots: int = None
        self.optimizer_name: str = None
        self.intermediate_energies: list = []
        self.eigenvalues: np.ndarray = None
        self.success: bool = False
        self.error: str = None
    
    def __repr__(self) -> str:
        """Print results in a formatted table."""
        def shown(value, missing="Not reported"):
            return missing if value is None else value
        
        if self.shots is None:
            shots = "Not reported"
        else:
            shots = "0 (exact expectation)" if self.shots == 0 else f"{self.shots} (sampled)"
        
        lines = [
            "\n" + "=" * 70,
            "VQA OPTIMIZATION RESULTS",
            "=" * 70,
            f"Status                       : {'✓ Completed' if self.success else '✗ Failed'}",
            f"Error                        : {self.error}" if self.error else "",
            f"Objective Value (energy)     : {self.energy:.10f}" if self.energy is not None else "",
            f"Optimal Parameters           : {self.optimal_parameters}",
            f"Number of Qubits             : {self.n_qubits}",
            f"Circuit Depth                : {shown(self.circuit_depth, 'Not measured')}",
            f"Optimizer                    : {self.optimizer_name}",
            f"Function Evaluations         : {shown(self.n_evaluations)}",
            f"Optimizer Converged          : {shown(self.convergence)}",
            f"Barren Plateau Check         : {shown(self.barren_plateau_detected, 'Not assessed')}",
            f"Shots                        : {shots}",
            f"Execution Time (s)           : {self.wall_time:.2f}",
            f"Backend Used                 : {shown(self.backend)}",
            "=" * 70,
        ]
        return "\n".join([line for line in lines if line])


def record_optimizer_metadata(results: VQAResults, qlm_result) -> None:
    """Fill evaluation count, energy history and convergence from a myQLM optimizer result.
    
    myQLM's ScipyMinimizePlugin stores the energy of every evaluation in
    ``meta_data["optimization_trace"]`` and scipy's report (including
    ``success``) in ``meta_data["optimizer_data"]``. Anything missing stays None.
    """
    meta = getattr(qlm_result, "meta_data", None) or {}
    trace = meta.get("optimization_trace")
    if trace:
        try:
            results.intermediate_energies = [float(v) for v in ast.literal_eval(trace)]
            results.n_evaluations = len(results.intermediate_energies)
        except (ValueError, SyntaxError, TypeError):
            pass
    match = re.search(r"success:\s*(True|False)", meta.get("optimizer_data") or "")
    if match:
        results.convergence = match.group(1) == "True"


def _executor_name(qpu) -> str:
    """Name the simulator at the bottom of a myQLM plugin stack."""
    for _ in range(10):
        inner = getattr(qpu, "qpu", None)
        if inner is None:
            break
        qpu = inner
    return f"myQLM {type(qpu).__module__}.{type(qpu).__name__}"


class VQA_myQLM:
    """
    Main VQA Engine using myQLM (QLM) backend.
    
    This class manages quantum circuit execution, optimization, and result tracking
    for variational quantum algorithms.
    
    Parameters:
        n_qubits (int): Number of qubits to allocate
        backend (str): "qat.simulators.qubit" (default myQLM simulator) or
            "pennylane" (PennyLane default.qubit); see SUPPORTED_BACKENDS
        shots (int): Number of measurement shots, 0 for exact expectation values
            (default: 0; applied on the myQLM path). Use e.g. 1000 to add the
            sampling noise of real hardware.
        seed (int): Random seed for reproducibility (default: None)
        verbose (bool): Enable verbose output (default: False)
        raise_on_error (bool): Re-raise errors from run() (default: True). If
            False, run() returns a failed VQAResults with the message in ``error``.
    
    Example:
        >>> engine = VQA_myQLM(n_qubits=2, seed=42)
        >>> from openvqa.algorithms import VQE
        >>> algorithm = VQE(hamiltonian="H2", ansatz="ryrz")
        >>> results = engine.run(algorithm)
    """
    
    SUPPORTED_BACKENDS = ("qat.simulators.qubit", "pennylane")
    
    def __init__(
        self,
        n_qubits: int = 4,
        backend: str = "qat.simulators.qubit",
        shots: int = 0,
        seed: Optional[int] = None,
        verbose: bool = False,
        use_pennylane_via_qlm: bool = False,
        raise_on_error: bool = True,
    ):
        # Validate inputs before anything is built or submitted
        if not isinstance(backend, str) or backend.lower() not in self.SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported backend {backend!r}. Choose from: {list(self.SUPPORTED_BACKENDS)}. "
                "No job was submitted."
            )
        if isinstance(n_qubits, bool) or not isinstance(n_qubits, int) or n_qubits < 1:
            raise ValueError(f"n_qubits must be a positive integer, got {n_qubits!r}.")
        if isinstance(shots, bool) or not isinstance(shots, int) or shots < 0:
            raise ValueError(f"shots must be a non-negative integer (0 = exact expectation), got {shots!r}.")
        
        self.n_qubits = n_qubits
        self.backend_name = backend
        self.is_pennylane = backend.lower() == "pennylane"
        self.shots = shots
        self.seed = seed
        self.verbose = verbose
        self.use_pennylane_via_qlm = use_pennylane_via_qlm
        self.raise_on_error = raise_on_error
        
        # Initialize QPU (used for non-PennyLane backends)
        self.qpu = None if self.is_pennylane else get_default_qpu()
        # Simulator that actually runs the jobs, reported in the results
        self.executor_name = "PennyLane default.qubit" if self.is_pennylane else _executor_name(self.qpu)
        
        if seed is not None:
            np.random.seed(seed)
        
        if verbose:
            print(f"✓ VQA Engine initialized")
            print(f"  Qubits: {n_qubits}")
            print(f"  Backend: {backend} ({self.executor_name})")
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
            # results.backend and results.shots are set by the execution path
            # that actually ran (e.g. optimizer="adam" runs on PennyLane)
            results.wall_time = time.time() - start_time
            results.success = True
            
            if self.verbose:
                print(results)
            
            return results
        
        except Exception as e:
            if self.raise_on_error:
                raise
            print(f"✗ Error during execution: {e}")
            results = VQAResults()
            results.success = False
            results.error = f"{type(e).__name__}: {e}"
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
