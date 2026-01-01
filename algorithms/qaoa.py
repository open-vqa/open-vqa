"""
Quantum Approximate Optimization Algorithm (QAOA)
================================================

Solves combinatorial optimization problems using parameterized quantum circuits.
"""

import numpy as np
from typing import Optional, Callable, List
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
import networkx as nx
from qat.core import Observable, Term
from qat.interop.qiskit import qiskit_to_qlm
from openvqa.engine import VQAResults
from openvqa.mathematical_tools import ScipyMinimizePluginMethod
from openvqa.mathematical_tools.adam_optimizer import AdamOptimizer
import pennylane as qml
import numpy as np
from scipy.optimize import minimize


class QAOA:
    """Quantum Approximate Optimization Algorithm (QAOA).
    
    Approximately solves MaxCut and similar combinatorial optimization problems.
    
    Parameters:
        problem (str): Problem type ("maxcut", "maxsat", "traveling_salesman")
        graph_size (int): Size of problem (nodes for MaxCut, etc.)
        optimizer (str): Classical optimizer - "cobyla", "nelder-mead", "bfgs", or "adam" (default: "cobyla")
        max_iterations (int): Maximum optimization iterations (default: 50)
        p (int): QAOA depth (number of layers) (default: 1)
        ansatz (str): Ansatz circuit type (e.g., "StronglyEntanglingLayers", "RealAmplitudes")
        embedding (str): Embedding type for encoding data (e.g., "AngleEmbedding") - for PennyLane backends
        feature_map (str): Feature map for encoding data (e.g., "PauliFeatureMap") - for Qiskit backends
        optimizer_options (dict): Additional optimizer-specific options (e.g., learning_rate for adam)
    
    Example:
        >>> algorithm = QAOA(
        ...     problem="maxcut",
        ...     graph_size=4,
        ...     optimizer="cobyla",
        ...     p=2,
        ...     ansatz="RealAmplitudes",
        ...     feature_map="PauliFeatureMap"
        ... )
        >>> results = engine.run(algorithm)
        >>> print(f"Max Cut Value: {results.energy:.4f}")
        
    Supported Optimizers:
        - "cobyla": Constrained Optimization BY Linear Approximation (default)
        - "nelder-mead": Nelder-Mead simplex algorithm
        - "bfgs": Broyden–Fletcher–Goldfarb–Shanno algorithm
        - "adam": Adaptive Moment Estimation (gradient-based)
    """

    # Public, discoverable option lists (intended for notebooks / quick inspection)
    SUPPORTED_OPTIMIZERS = ("cobyla", "nelder-mead", "bfgs", "adam")
    SUPPORTED_FEATURE_MAPS = ("PauliFeatureMap", "ZZFeatureMap", "ZFeatureMap")
    
    def __init__(
        self,
        problem: str = "maxcut",
        graph_size: int = 4,
        optimizer: str = "cobyla",
        max_iterations: int = 50,
        p: int = 1,
        ansatz: str = None,
        embedding: str = None,
        feature_map: str = None,
        optimizer_options: dict = None,
    ):
        self.problem = problem
        self.graph_size = graph_size
        self.optimizer_name = optimizer.lower()
        self.max_iterations = max_iterations
        self.p = p  # QAOA depth
        self.ansatz = ansatz
        self.embedding = embedding
        self.feature_map = feature_map
        self.optimizer_options = optimizer_options or {}
        
        # Validate optimizer choice
        if self.optimizer_name not in self.SUPPORTED_OPTIMIZERS:
            raise ValueError(
                f"Optimizer '{optimizer}' not supported. Choose from: {list(self.SUPPORTED_OPTIMIZERS)}"
            )
        
        self.graph = None
        self.n_params = None  # Will be set when building circuit
    
    def _create_maxcut_graph(self, n_nodes: int) -> nx.Graph:
        """Create a random MaxCut problem graph."""
        G = nx.complete_graph(n_nodes)
        # Add random weights to edges
        for u, v in G.edges():
            G[u][v]['weight'] = np.random.uniform(0.5, 2.0)
        return G
    
    def _create_regular_graph(self, n_nodes: int, k: int = 3) -> nx.Graph:
        """Create a k-regular graph."""
        if n_nodes * k % 2 != 0:
            k = k - 1
        return nx.random_regular_graph(k, n_nodes)
    
    def _build_maxcut_circuit(self, graph: nx.Graph, p: int) -> tuple:
        """Build QAOA circuit for MaxCut problem."""
        n_qubits = len(graph.nodes())
        
        # If custom ansatz is specified, use it instead of standard QAOA
        if self.ansatz:
            qc, params = self.build_ansatz(n_qubits)
            # Add feature map if specified (for initialization)
            if self.feature_map:
                fm = self.build_feature_map(self.feature_map, n_qubits)
                qc = fm.compose(qc)
                params = list(qc.parameters)
            return qc, params
        
        # Standard QAOA circuit
        params = []
        
        # Create parameters: gamma (problem) and beta (mixer) for each layer
        for layer in range(p):
            params.append(Parameter(f"γ_{layer}"))  # Problem Hamiltonian
            params.append(Parameter(f"β_{layer}"))  # Mixer Hamiltonian
        
        qc = QuantumCircuit(n_qubits)
        
        # Initialize in superposition (or use feature map)
        if self.feature_map:
            fm = self.build_feature_map(self.feature_map, n_qubits).decompose(reps=10)
            # Inline FM instructions explicitly to avoid composite instructions
            # (e.g., 'ZZFeatureMap') that some converters don't handle.
            for inst in fm.data:
                qargs = [qc.qubits[fm.find_bit(q).index] for q in inst.qubits]
                qc.append(inst.operation, qargs)
        else:
            for q in range(n_qubits):
                qc.h(q)
        
        # QAOA layers
        param_idx = 0
        for layer in range(p):
            gamma = params[param_idx]
            beta = params[param_idx + 1]
            param_idx += 2
            
            # Problem Hamiltonian: ZZ interactions on edges
            for u, v in graph.edges():
                angle = 2 * gamma * graph[u][v].get('weight', 1.0)
                qc.cx(u, v)
                qc.rz(angle, v)
                qc.cx(u, v)
            
            # Single qubit Z
            for q in range(n_qubits):
                qc.rz(gamma, q)
            
            # Mixer Hamiltonian: X rotations (RX)
            for q in range(n_qubits):
                qc.rx(2 * beta, q)
        
        return qc, params
    
    def _build_maxcut_observable(self, graph: nx.Graph) -> Observable:
        """Build MaxCut objective as observable (negative for minimization)."""
        terms = []
        
        for u, v in graph.edges():
            weight = graph[u][v].get('weight', 1.0)
            # MaxCut objective: -0.5 * sum_{edges} (1 - Z_u*Z_v)
            # Which equals: 0.5 * sum (Z_u*Z_v) - constant
            term = Term(-weight / 2, ["Z", "Z"], [u, v])
            terms.append(term)
        
        return Observable(len(graph.nodes()), pauli_terms=terms, constant_coeff=0.0)
    
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
        """Build the ansatz circuit for QAOA."""
        if self.ansatz.lower() == "realamplitudes":
            from qiskit.circuit.library import RealAmplitudes
            qc = RealAmplitudes(n_qubits, reps=self.p)
            params = list(qc.parameters)
            self.n_params = len(params)
        elif self.ansatz.lower() == "efficientsu2":
            from qiskit.circuit.library import EfficientSU2
            qc = EfficientSU2(n_qubits, reps=self.p)
            params = list(qc.parameters)
            self.n_params = len(params)
        elif self.ansatz.lower() == "twolocal":
            from qiskit.circuit.library import TwoLocal
            qc = TwoLocal(n_qubits, reps=self.p)
            params = list(qc.parameters)
            self.n_params = len(params)
        else:
            # Try treating it as a PennyLane template name
            from openvqa.interop.pennylane_to_qlm import build_vqe_circuit_qiskit
            try:
                if self.embedding:
                    qc, params = build_vqe_circuit_qiskit(self.embedding, self.ansatz, n_qubits, self.p)
                else:
                    from openvqa.interop.pennylane_to_qlm import template_to_qiskit
                    qc, params = template_to_qiskit(self.ansatz, n_qubits, self.p)
                # Convert back to Parameter objects for consistency
                from qiskit.circuit import Parameter
                param_list = [Parameter(f"p_{i}") for i in range(params)]
                self.n_params = params
                params = param_list
            except Exception:
                raise ValueError(f"Unknown ansatz: {self.ansatz}")
        
        # Compose with feature map if specified
        if self.feature_map:
            fm = self.build_feature_map(self.feature_map, n_qubits).decompose(reps=10)
            combined = QuantumCircuit(n_qubits)
            for inst in fm.data:
                qargs = [combined.qubits[fm.find_bit(q).index] for q in inst.qubits]
                combined.append(inst.operation, qargs)
            for inst in qc.data:
                qargs = [combined.qubits[qc.find_bit(q).index] for q in inst.qubits]
                combined.append(inst.operation, qargs)
            qc = combined
            params = list(qc.parameters)
            self.n_params = len(params)
        
        return qc, params
    
    def build_circuit(self) -> tuple:
        """Build QAOA circuit for the specified problem."""
        if self.problem.lower() == "maxcut":
            self.graph = self._create_maxcut_graph(self.graph_size)
            qc, params = self._build_maxcut_circuit(self.graph, self.p)
            self.n_params = len(params)
            return qc, params
        else:
            raise ValueError(f"Unsupported problem: {self.problem}")
    
    def build_observable(self) -> Observable:
        """Build objective function as Observable."""
        if self.problem.lower() == "maxcut":
            if self.graph is None:
                self.graph = self._create_maxcut_graph(self.graph_size)
            return self._build_maxcut_observable(self.graph)
        else:
            raise ValueError(f"Unsupported problem: {self.problem}")
    
    def execute(self, engine, qpu) -> VQAResults:
        """Execute QAOA algorithm."""
        # The QLM optimizer pipeline uses ScipyMinimizePlugin methods.
        # For Adam, use a direct PennyLane evaluation loop with the local AdamOptimizer.
        if self.optimizer_name == "adam":
            return self.execute_pennylane(engine)

        results = VQAResults()
        results.optimizer_name = self.optimizer_name
        results.n_qubits = self.graph_size
        
        # Build circuit and observable
        if engine.verbose:
            print(f"Building QAOA circuit for {self.problem} (p={self.p})...")
        
        qaoa_circuit, qaoa_params = self.build_circuit()
        observable = self.build_observable()
        
        results.n_qubits = self.graph_size
        results.circuit_depth = len(qaoa_circuit)
        results.n_evaluations = 0
        
        # Convert to QLM
        # Some circuit-library constructs (e.g., feature maps) can appear as
        # composite instructions that the Qiskit->QLM converter does not map.
        # Transpile + decompose expands them into basic gates.
        try:
            from qiskit import transpile
            qaoa_circuit = transpile(qaoa_circuit, optimization_level=0)
        except Exception:
            pass

        qaoa_circuit = qaoa_circuit.decompose(reps=10)
        qlm_circuit = qiskit_to_qlm(qaoa_circuit)
        
        # Create job
        nbshots = int(getattr(engine, "shots", 0) or 0)
        job = qlm_circuit.to_job(job_type="OBS", observable=observable, nbshots=nbshots)
        
        # Create optimizer
        optimizer = engine.create_optimizer(self.optimizer_name, self.max_iterations)
        
        # Run optimization
        if engine.verbose:
            print(f"Running QAOA optimization with {self.optimizer_name}...")
        
        qpu_with_optimizer = optimizer | qpu
        result = qpu_with_optimizer.submit(job)
        
        results.energy = result.value
        results.optimal_parameters = result.parameter_map
        results.convergence = True
        results.barren_plateau_detected = False
        
        return results

    def execute_pennylane(self, engine) -> VQAResults:
        """Execute QAOA using PennyLane when engine backend is PennyLane."""
        results = VQAResults()
        results.optimizer_name = self.optimizer_name
        n_qubits = self.graph_size

        if engine.verbose:
            print(f"Building QAOA circuit for {self.problem} (PennyLane)...")

        # build graph and Hamiltonian (MaxCut)
        if self.graph is None:
            self.graph = self._create_maxcut_graph(self.graph_size)

        # If engine requests running PennyLane templates via QLM conversion,
        # and an ansatz template name was provided, convert it to Qiskit and
        # submit via the QLM job pipeline.
        if engine.use_pennylane_via_qlm and self.ansatz is not None:
            from openvqa.interop.pennylane_to_qlm import template_to_qiskit
            qc, param_count = template_to_qiskit(self.ansatz, n_qubits, self.p)
            qlm_circuit = qiskit_to_qlm(qc)
            observable = self.build_observable()
            nbshots = int(getattr(engine, "shots", 0) or 0)
            job = qlm_circuit.to_job(job_type="OBS", observable=observable, nbshots=nbshots)
            optimizer = engine.create_optimizer(self.optimizer_name, self.max_iterations)
            qpu = engine.qpu
            if qpu is None:
                raise RuntimeError("QLM QPU not available in engine for execute_pennylane_via_qlm")
            qpu_with_optimizer = optimizer | qpu
            result = qpu_with_optimizer.submit(job)

            results.energy = result.value
            results.optimal_parameters = result.parameter_map
            results.convergence = True
            results.n_qubits = n_qubits
            results.circuit_depth = len(qc)
            return results

        # Default: run local PennyLane QNode minimization as previous implementation
        # build pennylane Hamiltonian
        coeffs = []
        ops = []
        import pennylane as qml
        for u, v in self.graph.edges():
            weight = self.graph[u][v].get('weight', 1.0)
            coeffs.append(-weight / 2)
            ops.append(qml.PauliZ(wires=u) @ qml.PauliZ(wires=v))
        hamiltonian_pl = qml.Hamiltonian(coeffs, ops)

        # build qaoa ansatz function
        p = self.p

        def qaoa_ansatz(params):
            idx = 0
            for q in range(n_qubits):
                qml.Hadamard(wires=q)
            for layer in range(p):
                gamma = params[idx]; beta = params[idx+1]
                idx += 2
                for u, v in self.graph.edges():
                    qml.CNOT(wires=[u, v])
                    qml.RZ(2 * gamma * self.graph[u][v].get('weight',1.0), wires=v)
                    qml.CNOT(wires=[u, v])
                for q in range(n_qubits):
                    qml.RX(2 * beta, wires=q)

        n_params = 2 * p
        dev = qml.device('default.qubit', wires=n_qubits)

        @qml.qnode(dev)
        def circuit(params):
            qaoa_ansatz(params)
            return qml.expval(hamiltonian_pl)

        def cost(pvec):
            return float(circuit(pvec))

        x0 = np.zeros(n_params)
        method = self.optimizer_name.lower()

        if method == "adam":
            learning_rate = float(self.optimizer_options.get("learning_rate", 0.01))
            beta1 = float(self.optimizer_options.get("beta1", 0.9))
            beta2 = float(self.optimizer_options.get("beta2", 0.999))
            epsilon = float(self.optimizer_options.get("epsilon", 1e-8))
            finite_diff_eps = float(self.optimizer_options.get("finite_diff_eps", 1e-6))
            init_scale = float(self.optimizer_options.get("init_scale", 0.1))

            # Important: QAOA's cost can have a stationary point at all-zeros
            # (phases don't affect diagonal observables without a mixer effect).
            # If we start at exactly zero, finite-difference gradients can be ~0
            # and Adam may not move. Use a non-zero initialization unless the
            # caller provided an explicit initial point.
            initial_point = self.optimizer_options.get("initial_point", None)
            if initial_point is None:
                x0 = init_scale * np.random.normal(size=n_params)
            else:
                x0 = np.array(initial_point, dtype=float)
                if x0.shape != (n_params,):
                    raise ValueError(
                        f"initial_point must have shape ({n_params},), got {x0.shape}"
                    )

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

            x = np.array(x0, dtype=float, copy=True)
            results.intermediate_energies = []
            results.n_evaluations = 0

            for _ in range(int(self.max_iterations)):
                energy = cost(x)
                results.intermediate_energies.append(float(energy))
                results.n_evaluations += 1

                grad = finite_difference_grad(x)
                results.n_evaluations += 2 * x.size
                x = optimizer.step(x, grad)

            final_energy = cost(x)
            results.n_evaluations += 1
            results.energy = float(final_energy)
            results.optimal_parameters = np.array(x, dtype=float)
            results.convergence = True
            results.n_qubits = n_qubits
            results.circuit_depth = 0
            return results

        res = minimize(cost, x0, method=method, options={"maxiter": self.max_iterations})

        results.energy = res.fun
        results.optimal_parameters = res.x
        results.convergence = res.success
        results.n_qubits = n_qubits
        results.circuit_depth = 0
        return results
