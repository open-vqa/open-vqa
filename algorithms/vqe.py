"""
Variational Quantum Eigensolver (VQE)
====================================

Finds the ground state energy of a Hamiltonian using a parameterized ansatz circuit
and classical optimization.
"""

import numpy as np
from typing import Optional, List, Dict, Any
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
import pennylane as qml
from pennylane import templates
from scipy.optimize import minimize
from qat.core import Observable, Term
from qat.interop.qiskit import qiskit_to_qlm
from openvqa.engine import VQAResults
from openvqa.interop.pennylane_to_qlm import template_to_qiskit, build_vqe_circuit_qiskit
from openvqa.mathematical_tools import ScipyMinimizePluginMethod
from openvqa.mathematical_tools.adam_optimizer import AdamOptimizer


class VQE:
    """
    Variational Quantum Eigensolver (VQE).
    
    Computes the ground state energy of a Hamiltonian using a variational ansatz.
    
    Parameters:
        hamiltonian (str): Predefined Hamiltonian (e.g., "H2", "LiH") or custom matrix
        ansatz (str): Ansatz circuit type (e.g., "uccsd", "ryrz", "linear", "RealAmplitudes")
        embedding (str): Embedding type for encoding data (e.g., "angle", "amplitude") - for PennyLane backends
        feature_map (str): Feature map for encoding data (e.g., "PauliFeatureMap", "ZZFeatureMap") - for Qiskit backends
        optimizer (str): Classical optimizer - "cobyla", "nelder-mead", "bfgs", or "adam" (default: "cobyla")
        max_iterations (int): Maximum optimization iterations (default: 100)
        n_layers (int): Number of ansatz layers (default: 2)
        optimizer_options (dict): Additional optimizer-specific options (e.g., learning_rate for adam)
    
    Example:
        >>> algorithm = VQE(
        ...     hamiltonian="H2",
        ...     ansatz="RealAmplitudes",
        ...     feature_map="PauliFeatureMap",
        ...     optimizer="cobyla",
        ...     max_iterations=50
        ... )
        >>> results = engine.run(algorithm)
        >>> print(f"Energy: {results.energy:.10f}")
        
    Supported Optimizers:
        - "cobyla": Constrained Optimization BY Linear Approximation (default)
        - "nelder-mead": Nelder-Mead simplex algorithm
        - "bfgs": Broyden–Fletcher–Goldfarb–Shanno algorithm
        - "adam": Adaptive Moment Estimation (gradient-based)
    """

    # Public, discoverable option lists (intended for notebooks / quick inspection)
    SUPPORTED_OPTIMIZERS = ("cobyla", "nelder-mead", "bfgs", "adam")
    SUPPORTED_FEATURE_MAPS = ("PauliFeatureMap", "ZZFeatureMap", "ZFeatureMap")
    SUPPORTED_ANSATZ_INTERNAL = ("ryrz", "linear", "uccsd")
    SUPPORTED_ANSATZ_QISKIT = ("RealAmplitudes", "EfficientSU2", "TwoLocal")
    
    def __init__(
        self,
        hamiltonian: str = "H2",
        ansatz: str = "ryrz",
        embedding: str = None,
        feature_map: str = None,
        optimizer: str = "cobyla",
        max_iterations: int = 100,
        n_layers: int = 2,
        optimizer_options: dict = None,
    ):
        self.hamiltonian_name = hamiltonian
        self.ansatz_name = ansatz
        self.embedding_name = embedding
        self.feature_map_name = feature_map
        self.optimizer_name = optimizer.lower()
        self.max_iterations = max_iterations
        self.n_layers = n_layers
        self.optimizer_options = optimizer_options or {}
        
        # Validate optimizer choice
        if self.optimizer_name not in self.SUPPORTED_OPTIMIZERS:
            raise ValueError(
                f"Optimizer '{optimizer}' not supported. Choose from: {list(self.SUPPORTED_OPTIMIZERS)}"
            )
        
        self.hamiltonian = None
        self.ansatz_circuit = None
        self.n_params = None
    
    def _build_h2_hamiltonian(self) -> Observable:
        """Build H2 Hamiltonian (2 qubits)."""
        terms = [
            Term(0.5, [], []),
            Term(0.5, ["Z"], [0]),
            Term(0.5, ["Z"], [1]),
            Term(0.25, ["X", "X"], [0, 1]),
            Term(0.25, ["Y", "Y"], [0, 1]),
        ]
        return Observable(2, pauli_terms=terms, constant_coeff=0.0)
    
    def _build_lih_hamiltonian(self) -> Observable:
        """Build LiH Hamiltonian (4 qubits, simplified)."""
        terms = [
            Term(-0.198, [], []),
            Term(0.396, ["Z"], [0]),
            Term(-0.396, ["Z"], [1]),
            Term(-0.396, ["Z"], [2]),
            Term(0.396, ["Z"], [3]),
            Term(0.099, ["Z", "Z"], [0, 1]),
            Term(0.099, ["Z", "Z"], [2, 3]),
            Term(0.025, ["X", "X"], [0, 1]),
            Term(0.025, ["Y", "Y"], [0, 1]),
            Term(0.025, ["X", "X"], [2, 3]),
            Term(0.025, ["Y", "Y"], [2, 3]),
        ]
        return Observable(4, pauli_terms=terms, constant_coeff=0.0)
    
    def _build_ansatz_ryrz(self, n_qubits: int, n_layers: int) -> QuantumCircuit:
        """Build RY-RZ parametric ansatz."""
        params = []
        for layer in range(n_layers):
            for qubit in range(n_qubits):
                params.append(Parameter(f"θ_RY_{layer}_{qubit}"))
                params.append(Parameter(f"θ_RZ_{layer}_{qubit}"))
        
        self.n_params = len(params)
        
        qc = QuantumCircuit(n_qubits)
        
        # Initial Hadamard
        for q in range(n_qubits):
            qc.h(q)
        
        # Parametric layers
        param_idx = 0
        for layer in range(n_layers):
            # RY rotations
            for q in range(n_qubits):
                qc.ry(params[param_idx], q)
                param_idx += 1
            
            # RZ rotations
            for q in range(n_qubits):
                qc.rz(params[param_idx], q)
                param_idx += 1
            
            # Entangling layer (CNOT chain)
            for q in range(n_qubits - 1):
                qc.cx(q, q + 1)
            qc.cx(n_qubits - 1, 0)  # Wrap around
        
        return qc, params
    
    def _build_ansatz_linear(self, n_qubits: int, n_layers: int) -> tuple:
        """Build linear parametric ansatz (RX-RZ)."""
        params = []
        for layer in range(n_layers):
            for qubit in range(n_qubits):
                params.append(Parameter(f"θ_{layer}_{qubit}"))
        
        self.n_params = len(params)
        
        qc = QuantumCircuit(n_qubits)
        
        # Initial superposition
        for q in range(n_qubits):
            qc.h(q)
        
        # Parametric layers
        param_idx = 0
        for layer in range(n_layers):
            for q in range(n_qubits):
                qc.rx(params[param_idx], q)
                param_idx += 1
            
            # Entangling CNOTs
            for q in range(n_qubits - 1):
                qc.cx(q, q + 1)
        
        return qc, params
    
    def _build_ansatz_uccsd(self, n_qubits: int) -> tuple:
        """Build simplified UCC-SD ansatz."""
        # Simplified UCCSD for small systems
        params = [Parameter(f"t_{i}") for i in range(n_qubits)]
        self.n_params = len(params)
        
        qc = QuantumCircuit(n_qubits)
        
        # Initialize in reference state (HF)
        for q in range(n_qubits // 2):
            qc.x(q)
        
        # Add parameterized gates
        for i, param in enumerate(params):
            qc.ry(param, i % n_qubits)
            qc.cx(i % n_qubits, (i + 1) % n_qubits)
        
        return qc, params
    
    def build_feature_map(self, name: str, n_qubits: int):
        """Build a Qiskit feature map."""
        if name.lower() == "paulifeaturemap":
            from qiskit.circuit.library import PauliFeatureMap
            return PauliFeatureMap(n_qubits, reps=1)
        elif name.lower() == "zzfeaturemap":
            from qiskit.circuit.library import ZZFeatureMap
            return ZZFeatureMap(n_qubits, reps=1)
        elif name.lower() == "zfeaturemap":
            from qiskit.circuit.library import ZFeatureMap
            return ZFeatureMap(n_qubits, reps=1)
        else:
            raise ValueError(f"Unknown feature map: {name}")
    
    def build_ansatz(self, n_qubits: int) -> tuple:
        """Build the ansatz circuit."""
        if self.ansatz_name.lower() == "ryrz":
            qc, params = self._build_ansatz_ryrz(n_qubits, self.n_layers)
        elif self.ansatz_name.lower() == "linear":
            qc, params = self._build_ansatz_linear(n_qubits, self.n_layers)
        elif self.ansatz_name.lower() == "uccsd":
            qc, params = self._build_ansatz_uccsd(n_qubits)
        elif self.ansatz_name.lower() == "realamplitudes":
            from qiskit.circuit.library import RealAmplitudes
            qc = RealAmplitudes(n_qubits, reps=self.n_layers)
            params = list(qc.parameters)
            self.n_params = len(params)
        elif self.ansatz_name.lower() == "efficientsu2":
            from qiskit.circuit.library import EfficientSU2
            qc = EfficientSU2(n_qubits, reps=self.n_layers)
            params = list(qc.parameters)
            self.n_params = len(params)
        elif self.ansatz_name.lower() == "twolocal":
            from qiskit.circuit.library import TwoLocal
            qc = TwoLocal(n_qubits, reps=self.n_layers)
            params = list(qc.parameters)
            self.n_params = len(params)
        else:
            # Try treating it as a PennyLane template name (for Pennylane->QLM path)
            # Will be handled by template_to_qiskit in execute_pennylane_via_qlm
            from openvqa.interop.pennylane_to_qlm import build_vqe_circuit_qiskit
            try:
                if self.embedding_name:
                    qc, params = build_vqe_circuit_qiskit(self.embedding_name, self.ansatz_name, n_qubits, self.n_layers)
                else:
                    qc, params = template_to_qiskit(self.ansatz_name, n_qubits, self.n_layers)
                # Convert back to Parameter objects for consistency
                from qiskit.circuit import Parameter
                param_list = [Parameter(f"p_{i}") for i in range(params)]
                self.n_params = params
                params = param_list
            except Exception:
                raise ValueError(f"Unknown ansatz: {self.ansatz_name}")
        
        # Compose with feature map if specified
        if self.feature_map_name:
            fm = self.build_feature_map(self.feature_map_name, n_qubits)
            qc = fm.compose(qc)
            params = list(qc.parameters)
            self.n_params = len(params)
        
        return qc, params
    
    def build_hamiltonian(self, n_qubits: int) -> Observable:
        """Build the target Hamiltonian."""
        if self.hamiltonian_name.lower() == "h2":
            if n_qubits < 2:
                raise ValueError("H2 Hamiltonian requires at least 2 qubits")
            return self._build_h2_hamiltonian()
        elif self.hamiltonian_name.lower() == "lih":
            if n_qubits < 4:
                raise ValueError("LiH Hamiltonian requires at least 4 qubits")
            return self._build_lih_hamiltonian()
        else:
            raise ValueError(f"Unknown Hamiltonian: {self.hamiltonian_name}")
    
    def execute(self, engine, qpu) -> VQAResults:
        """Execute VQE algorithm."""
        # The QLM optimizer pipeline uses ScipyMinimizePlugin methods.
        # For Adam, use a direct PennyLane evaluation loop with the local AdamOptimizer.
        if self.optimizer_name == "adam":
            return self.execute_pennylane(engine)

        results = VQAResults()
        results.optimizer_name = self.optimizer_name
        results.n_qubits = engine.n_qubits
        
        # Build components
        if engine.verbose:
            print(f"Building {self.hamiltonian_name} Hamiltonian...")
        self.hamiltonian = self.build_hamiltonian(engine.n_qubits)
        
        if engine.verbose:
            print(f"Building {self.ansatz_name} ansatz ({self.n_layers} layers)...")
        ansatz_circ, ansatz_params = self.build_ansatz(engine.n_qubits)
        
        results.n_qubits = engine.n_qubits
        results.circuit_depth = len(ansatz_circ)
        results.n_evaluations = 0
        
        # Convert to QLM
        ansatz_circ = ansatz_circ.decompose()
        qlm_circuit = qiskit_to_qlm(ansatz_circ)
        
        # Create optimization job
        # nbshots=0 => exact expectation (deterministic simulator)
        # nbshots>0 => sampling (shot noise)
        nbshots = int(getattr(engine, "shots", 0) or 0)
        job = qlm_circuit.to_job(job_type="OBS", observable=self.hamiltonian, nbshots=nbshots)
        
        # Create optimizer
        optimizer = engine.create_optimizer(self.optimizer_name, self.max_iterations)
        
        # Run optimization
        if engine.verbose:
            print(f"Running optimization with {self.optimizer_name}...")
        
        qpu_with_optimizer = optimizer | qpu
        result = qpu_with_optimizer.submit(job)
        
        results.energy = result.value
        results.optimal_parameters = result.parameter_map
        results.convergence = True
        results.barren_plateau_detected = False
        
        return results

    def execute_pennylane(self, engine) -> VQAResults:
        """Execute VQE using PennyLane (fallback path when backend is PennyLane)."""
        results = VQAResults()
        results.optimizer_name = self.optimizer_name
        n_qubits = engine.n_qubits

        # Build Hamiltonian as pennylane observable
        if engine.verbose:
            print(f"Building {self.hamiltonian_name} Hamiltonian (PennyLane)...")
        self.hamiltonian = self.build_hamiltonian(n_qubits)

        # Convert QAT Observable to PennyLane Hamiltonian safely
        try:
            if hasattr(self.hamiltonian, 'pauli_terms'):
                # QAT Observable with pauli_terms attribute
                coeffs = []
                ops = []
                for term in self.hamiltonian.pauli_terms:
                    coeffs.append(term.coeff)
                    if len(term.pauli) == 0:
                        ops.append(qml.Identity(wires=0))
                    else:
                        subops = []
                        for p, w in zip(term.pauli, term.wires):
                            if p == 'X':
                                subops.append(qml.PauliX(w))
                            elif p == 'Y':
                                subops.append(qml.PauliY(w))
                            elif p == 'Z':
                                subops.append(qml.PauliZ(w))
                        if len(subops) == 1:
                            ops.append(subops[0])
                        else:
                            prod = subops[0]
                            for so in subops[1:]:
                                prod = prod @ so
                            ops.append(prod)
                hamiltonian_pl = qml.Hamiltonian(coeffs, ops)
            else:
                # Fallback: build a simple H2 Hamiltonian in PennyLane directly
                if engine.verbose:
                    print("Warning: Could not extract pauli_terms, using default H2 Hamiltonian")
                hamiltonian_pl = qml.Hamiltonian(
                    [0.5, 0.5, 0.5, 0.25, 0.25],
                    [qml.Identity(wires=0), qml.PauliZ(wires=0), qml.PauliZ(wires=1),
                     qml.PauliX(wires=0) @ qml.PauliX(wires=1),
                     qml.PauliY(wires=0) @ qml.PauliY(wires=1)]
                )
        except Exception as e:
            if engine.verbose:
                print(f"Warning: Error converting to PennyLane Hamiltonian: {e}")
            # Use default H2 Hamiltonian
            hamiltonian_pl = qml.Hamiltonian(
                [0.5, 0.5, 0.5, 0.25, 0.25],
                [qml.Identity(wires=0), qml.PauliZ(wires=0), qml.PauliZ(wires=1),
                 qml.PauliX(wires=0) @ qml.PauliX(wires=1),
                 qml.PauliY(wires=0) @ qml.PauliY(wires=1)]
            )

        # Build ansatz function: accept either a pennylane template (callable) or fallback to internal builder
        def ansatz_fn(params):
            if callable(self.ansatz_name):
                # If ansatz_name is a template function/class
                templates.StronglyEntanglingLayers(params, wires=list(range(n_qubits)))
            elif isinstance(self.ansatz_name, str) and self.ansatz_name.lower() in ['stronglyentanglinglayers', 'basicentanglerlayers', 'randomlayers']:
                # Use PennyLane templates
                if self.ansatz_name.lower() == 'stronglyentanglinglayers':
                    weights = np.array(params).reshape(self.n_layers, n_qubits, 3)
                    templates.StronglyEntanglingLayers(weights, wires=list(range(n_qubits)))
                elif self.ansatz_name.lower() == 'basicentanglerlayers':
                    weights = np.array(params).reshape(self.n_layers, n_qubits)
                    templates.BasicEntanglerLayers(weights, wires=list(range(n_qubits)))
                elif self.ansatz_name.lower() == 'randomlayers':
                    weights = np.array(params).reshape(self.n_layers, n_qubits, 3)
                    templates.RandomLayers(weights, wires=list(range(n_qubits)))
            else:
                # Fallback to RY-RZ built circuit translated to pennylane
                n_layers = self.n_layers
                idx = 0
                for layer in range(n_layers):
                    for q in range(n_qubits):
                        qml.RY(params[idx], wires=q)
                        idx += 1
                    for q in range(n_qubits):
                        qml.RZ(params[idx], wires=q)
                        idx += 1

        # Build embedding function
        def embedding_fn():
            if self.embedding_name:
                if self.embedding_name.lower() == 'angleembedding':
                    # For H2, use some fixed angles (could be molecular parameters)
                    data = [0.5, 0.5]  # example data
                    qml.AngleEmbedding(data, wires=list(range(min(len(data), n_qubits))))
                elif self.embedding_name.lower() == 'amplitudeembedding':
                    # Use fixed state vector
                    data = [1.0, 0.0]  # |0> state
                    qml.AmplitudeEmbedding(data, wires=list(range(min(len(data), n_qubits))))
                elif self.embedding_name.lower() == 'basisembedding':
                    # Encode binary string, e.g., |00>
                    data = [0, 0]
                    qml.BasisEmbedding(data, wires=list(range(min(len(data), n_qubits))))
                elif self.embedding_name.lower() == 'displacementembedding':
                    # For qubits, approximate with rotations
                    data = [0.5, 0.5]
                    qml.DisplacementEmbedding(data, wires=list(range(min(len(data), n_qubits))))
                # Add more embeddings as needed
            else:
                # Default: Hadamard layer
                for w in range(n_qubits):
                    qml.Hadamard(wires=w)

        # Estimate number of parameters
        if callable(self.ansatz_name):
            n_params = n_qubits * max(1, self.n_layers) * 3
        elif isinstance(self.ansatz_name, str) and self.ansatz_name.lower() in ['stronglyentanglinglayers', 'basicentanglerlayers', 'randomlayers']:
            # PennyLane templates parameter counts
            if self.ansatz_name.lower() == 'stronglyentanglinglayers':
                n_params = 3 * n_qubits * self.n_layers
            elif self.ansatz_name.lower() == 'basicentanglerlayers':
                n_params = n_qubits * self.n_layers
            elif self.ansatz_name.lower() == 'randomlayers':
                n_params = 3 * n_qubits * self.n_layers
        else:
            n_params = n_qubits * 2 * self.n_layers

        dev = qml.device('default.qubit', wires=n_qubits)

        @qml.qnode(dev)
        def circuit(params):
            embedding_fn()
            ansatz_fn(params)
            return qml.expval(hamiltonian_pl)

        # If configured to run Pennylane templates via QLM, convert to Qiskit and
        # submit through the QLM optimizer job pipeline. Otherwise, run the
        # direct PennyLane QNode minimization above.
        if engine.use_pennylane_via_qlm and isinstance(self.ansatz_name, str):
            # Convert selected template to a Qiskit circuit
            if self.embedding_name:
                qc, param_count = build_vqe_circuit_qiskit(self.embedding_name, self.ansatz_name, n_qubits, self.n_layers)
            else:
                qc, param_count = template_to_qiskit(self.ansatz_name, n_qubits, self.n_layers)

            # Convert to QLM and run using existing job/optimizer flow
            qlm_circuit = qiskit_to_qlm(qc)
            nbshots = int(getattr(engine, "shots", 0) or 0)
            job = qlm_circuit.to_job(job_type="OBS", observable=self.hamiltonian, nbshots=nbshots)
            optimizer = engine.create_optimizer(self.optimizer_name, self.max_iterations)
            qpu = engine.qpu
            if qpu is None:
                raise RuntimeError("QLM QPU not available in engine for execute_pennylane_via_qlm")
            qpu_with_optimizer = optimizer | qpu
            result = qpu_with_optimizer.submit(job)

            results.energy = result.value
            results.optimal_parameters = result.parameter_map
            results.convergence = True
            results.circuit_depth = len(qc)
            results.n_qubits = n_qubits
            return results

        # Otherwise fallback to direct PennyLane SciPy minimize path
        def cost(p):
            return float(circuit(p))

        method = self.optimizer_name.lower()

        if method == "adam":
            learning_rate = float(self.optimizer_options.get("learning_rate", 0.01))
            beta1 = float(self.optimizer_options.get("beta1", 0.9))
            beta2 = float(self.optimizer_options.get("beta2", 0.999))
            epsilon = float(self.optimizer_options.get("epsilon", 1e-8))
            finite_diff_eps = float(self.optimizer_options.get("finite_diff_eps", 1e-6))

            optimizer = AdamOptimizer(
                learning_rate=learning_rate,
                beta1=beta1,
                beta2=beta2,
                epsilon=epsilon,
            )

            def finite_difference_grad(x: np.ndarray) -> np.ndarray:
                grad = np.zeros_like(x, dtype=float)
                for i in range(x.size):
                    x_fwd = np.array(x, dtype=float, copy=True)
                    x_bwd = np.array(x, dtype=float, copy=True)
                    x_fwd[i] += finite_diff_eps
                    x_bwd[i] -= finite_diff_eps
                    grad[i] = (cost(x_fwd) - cost(x_bwd)) / (2.0 * finite_diff_eps)
                return grad

            x = np.zeros(n_params, dtype=float)
            results.intermediate_energies = []
            results.n_evaluations = 0

            for _ in range(int(self.max_iterations)):
                energy = cost(x)
                results.intermediate_energies.append(float(energy))
                results.n_evaluations += 1

                grad = finite_difference_grad(x)
                # Count evaluations done inside finite_difference_grad
                results.n_evaluations += 2 * x.size
                x = optimizer.step(x, grad)

            final_energy = cost(x)
            results.n_evaluations += 1
            results.energy = float(final_energy)
            results.optimal_parameters = np.array(x, dtype=float)
            results.convergence = True
            results.circuit_depth = 0
            results.n_qubits = n_qubits
            return results

        x0 = np.zeros(n_params)
        res = minimize(
            cost,
            x0,
            method=method.upper() if method in ["cobyla", "nelder-mead"] else None,
            options={"maxiter": self.max_iterations},
        )

        results.energy = float(res.fun)
        results.optimal_parameters = np.array(res.x, dtype=float)
        results.convergence = bool(res.success)
        results.circuit_depth = 0
        results.n_qubits = n_qubits
        return results
