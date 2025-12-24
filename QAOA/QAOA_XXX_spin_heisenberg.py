import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
import math

# =====================================================
# Qiskit Imports
# =====================================================
from qiskit import QuantumCircuit, Aer, execute, transpile
from qiskit.circuit import Parameter
from qiskit.circuit.library import RXXGate, RYYGate, RZZGate, TwoLocal
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.opflow import X, Y, Z, I
from qiskit_algorithms.optimizers import SLSQP  # or ADAM, etc.
from qiskit.primitives import Estimator
from qiskit_algorithms import VQE

# =====================================================
# QLM / QAT Imports
# =====================================================
from qat.core import Observable, Term
from qat.interop.qiskit import qiskit_to_qlm
from qat.qpus import get_default_qpu
from qat.plugins import ScipyMinimizePlugin

# =====================================================
# Cirq Imports
# =====================================================
import cirq
from qat.interop.cirq import qlm_to_cirq

# =====================================================
# Class: SpinHamiltonianBuilder
# =====================================================
class SpinHamiltonianBuilder:
    def __init__(self, graph, J1, J2):
        """
        Builds a spin Hamiltonian from a graph.
        
        Parameters:
            graph (networkx.Graph): A graph (for example, a 2d lattice).
            J1 (float): Coupling constant for pairs that are connected in the graph.
            J2 (float): Coupling constant for pairs that are not connected.
        """
        self.graph = nx.relabel.convert_node_labels_to_integers(graph)
        self.J1 = J1
        self.J2 = J2
        self.num_qubits = len(self.graph.nodes())
        self.edges = list(self.graph.edges())
        self.non_edges = list(nx.non_edges(self.graph))
        self.pauli_strings = self._build_pauli_strings()
    
    @staticmethod
    def replace_char(string, index, new_char):
        """Replace the character at position index in string with new_char."""
        return string[:index] + new_char + string[index+1:]
    
    def _build_pauli_strings(self):
        """
        Builds a list of Pauli–string representations.
        
        For every edge in the graph and every non–edge, for each operator
        in ['X', 'Y', 'Z'], we replace the corresponding two characters in
        an identity string.
        """
        pauli_list = []
        # For edges (connected pairs)
        for edge in self.edges:
            for p in ['X', 'Y', 'Z']:
                s = "I" * self.num_qubits
                s = self.replace_char(s, edge[0], p)
                s = self.replace_char(s, edge[1], p)
                pauli_list.append(s)
        # For non-edges (not connected)
        for edge in self.non_edges:
            for p in ['X', 'Y', 'Z']:
                s = "I" * self.num_qubits
                s = self.replace_char(s, edge[0], p)
                s = self.replace_char(s, edge[1], p)
                pauli_list.append(s)
        return pauli_list

    def get_qiskit_hamiltonian(self):
        """
        Returns the Hamiltonian as a Qiskit Operator (sum of Pauli operators).
        A constant offset is subtracted so that the identity term is removed.
        """
        H_total = self.J1 * Operator(Pauli("I" * self.num_qubits))
        for h in self.pauli_strings:
            # Identify the two indices where h is not identity.
            indices = [i for i, ch in enumerate(h) if ch != 'I']
            # Check if the pair (in either order) is in the graph’s edge list.
            if len(indices) == 2 and (tuple(indices) in self.edges or tuple(indices[::-1]) in self.edges):
                coeff = self.J1
            else:
                coeff = self.J2
            H_total += coeff * Operator(Pauli(h))
        # Remove the constant offset.
        H_total -= self.J1 * Operator(Pauli("I" * self.num_qubits))
        return H_total

    def get_qlm_observable(self) -> Observable:
        """
        Returns the Hamiltonian as a QLM Observable (using QAT’s Term objects).
        """
        pauli_terms = []
        for h in self.pauli_strings:
            indices = [i for i, ch in enumerate(h) if ch != 'I']
            if len(indices) == 2 and (tuple(indices) in self.edges or tuple(indices[::-1]) in self.edges):
                coeff = self.J1
            else:
                coeff = self.J2
            pauli_terms.append(Term(coeff, h, list(range(self.num_qubits))))
        constant_coeff = -self.J1
        return Observable(self.num_qubits, pauli_terms=pauli_terms, constant_coeff=constant_coeff)

    def get_cirq_hamiltonian(self):
        """
        Returns the Hamiltonian as a Cirq PauliSum.
        """
        qubits = [cirq.LineQubit(i) for i in range(self.num_qubits)]
        pauli_sum = cirq.PauliSum()
        for h in self.pauli_strings:
            indices = [i for i, ch in enumerate(h) if ch != 'I']
            if len(indices) == 2 and (tuple(indices) in self.edges or tuple(indices[::-1]) in self.edges):
                coeff = self.J1
            else:
                coeff = self.J2
            ps_dict = {}
            for i, ch in enumerate(h):
                if ch == 'I':
                    continue
                elif ch == 'X':
                    ps_dict[qubits[i]] = cirq.X
                elif ch == 'Y':
                    ps_dict[qubits[i]] = cirq.Y
                elif ch == 'Z':
                    ps_dict[qubits[i]] = cirq.Z
            term = cirq.PauliString(ps_dict, coefficient=coeff)
            pauli_sum += term
        identity = cirq.PauliString({q: cirq.I for q in qubits}, coefficient=-self.J1)
        pauli_sum += identity
        return pauli_sum

# =====================================================
# Class: SpinVQESolver
# =====================================================
class SpinVQESolver:
    def __init__(self, spin_ham_builder: SpinHamiltonianBuilder, n_layers: int = 2):
        """
        Initializes the VQE solver for spin models.
        
        Parameters:
            spin_ham_builder (SpinHamiltonianBuilder): An instance of the Hamiltonian builder.
            n_layers (int): Number of ansatz layers to compose.
        """
        self.ham_builder = spin_ham_builder
        self.num_qubits = spin_ham_builder.num_qubits
        self.n_layers = n_layers
        self.J1 = spin_ham_builder.J1
        self.J2 = spin_ham_builder.J2

    @staticmethod
    def build_spin_ansatz(hamiltonian_list, param, num_qubits, J1, J2):
        """
        Constructs an ansatz circuit that applies two-qubit rotations based on
        the provided list of Pauli strings.
        
        Parameters:
            hamiltonian_list (list): List of Pauli–strings.
            param (list): A list of two parameters. The first is used to set
                          the rotation angle in the two-qubit gate, and the second
                          is used for single-qubit RX rotations.
            num_qubits (int): Number of qubits.
            J1 (float): Coupling constant used for “edge” interactions.
            J2 (float): Coupling constant for non–edge interactions.
        
        Returns:
            QuantumCircuit: A circuit implementing the ansatz for one layer.
        """
        qc = QuantumCircuit(num_qubits, num_qubits)
        counter = 0
        for h in hamiltonian_list:
            counter += 1
            for p in ['X', 'Y', 'Z']:
                if p in h:
                    index1 = h.find(p)
                    index2 = h.rfind(p)
                    # Use one coupling if the gate’s index (counter) is less than 22, else use the other.
                    prmtr = J1 if counter < 22 else J2
                    angle = prmtr * param[0]
                    if p == 'X':
                        qc.append(RXXGate(angle), [index1, index2])
                    elif p == 'Y':
                        qc.append(RYYGate(angle), [index1, index2])
                    elif p == 'Z':
                        qc.append(RZZGate(angle), [index1, index2])
        qc = qc.decompose()  # decompose composite gates
        for i in range(num_qubits):
            qc.rx(param[1], i)
        return qc

    def build_variational_circuit(self, params_layers):
        """
        Composes the full variational circuit by adding an initial Hadamard
        layer and then composing n_layers of the spin ansatz.
        
        Parameters:
            params_layers (list): A list (of length n_layers) of parameter lists.
                (Each parameter list is assumed to have two elements.)
        
        Returns:
            QuantumCircuit: The full variational circuit.
        """
        qc = QuantumCircuit(self.num_qubits, self.num_qubits)
        # Initialize each qubit in superposition.
        for i in range(self.num_qubits):
            qc.h(i)
        # Compose each ansatz layer.
        for layer in range(self.n_layers):
            qc = qc.compose(self.build_spin_ansatz(self.ham_builder.pauli_strings,
                                                     params_layers[layer],
                                                     self.num_qubits,
                                                     self.J1,
                                                     self.J2))
        return qc

    def solve_with_qiskit(self, variational_circuit, optimizer=SLSQP(maxiter=50)):
        """
        Runs VQE using Qiskit’s VQE routine.
        
        Parameters:
            variational_circuit (QuantumCircuit): The ansatz circuit.
            optimizer: A Qiskit optimizer (default SLSQP with maxiter=50).
        
        Returns:
            result: The VQE result object.
        """
        # Callback to record convergence (optional)
        counts = []
        values = []
        def store_intermediate_result(eval_count, parameters, mean, std):
            counts.append(eval_count)
            values.append(mean)
        
        estimator = Estimator()
        vqe = VQE(estimator, variational_circuit, optimizer, callback=store_intermediate_result)
        result = vqe.compute_minimum_eigenvalue(self.ham_builder.get_qiskit_hamiltonian())
        print("Qiskit VQE result:")
        print(result)
        plt.figure(figsize=(12,4))
        plt.plot(counts, values, marker='o')
        plt.xlabel("Eval count")
        plt.ylabel("Energy")
        plt.title("Qiskit VQE Convergence")
        plt.grid(True)
        plt.show()
        return result

    def solve_with_qlm(self, variational_circuit, initial_params, optimization_methods=None):
        """
        Runs VQE using QLM.
        
        Parameters:
            variational_circuit (QuantumCircuit): The ansatz circuit.
            initial_params (np.ndarray): Initial parameter vector.
            optimization_methods (list): List of optimizer names (default: COBYLA, Nelder-Mead, BFGS).
        
        Returns:
            dict: A dictionary mapping optimizer names to result objects.
        """
        qlm_circ = qiskit_to_qlm(variational_circuit)
        observable = self.ham_builder.get_qlm_observable()
        job = qlm_circ.to_job(job_type="OBS", observable=observable, nbshots=0)
        linalg_qpu = get_default_qpu()
        if optimization_methods is None:
            optimization_methods = ["COBYLA", "Nelder-Mead", "BFGS"]
        results = {}
        for method in optimization_methods:
            optimizer_scipy = ScipyMinimizePlugin(method=method,
                                                  tol=1e-6,
                                                  options={"maxiter": 50},
                                                  x0=initial_params)
            qpu = optimizer_scipy | linalg_qpu
            res = qpu.submit(job)
            results[method] = res
            print(f"QLM VQE energy ({method}) = {res.value}")
        return results

    def solve_with_cirq(self, variational_circuit, num_samples=1000):
        """
        Simulates the VQE circuit using Cirq by sampling random parameter values.
        
        Parameters:
            variational_circuit (QuantumCircuit): The ansatz circuit.
            num_samples (int): Number of random samples.
        
        Returns:
            float: The minimum energy found.
        """
        qlm_circ = qiskit_to_qlm(variational_circuit)
        cirq_circ = qlm_to_cirq(qlm_circ)
        observable = self.ham_builder.get_cirq_hamiltonian()
        simulator = cirq.Simulator()
        energies = []
        random_params = []
        # Determine number of parameters (assumes the circuit has free parameters)
        num_params = len(variational_circuit.parameters)
        for _ in range(num_samples):
            params = [2*random.random()-1 for _ in range(num_params)]
            # Assume parameter names are like 'a0', 'a1', etc.
            resolver_dict = {f'a{i}': params[i] for i in range(num_params)}
            resolver = cirq.ParamResolver(resolver_dict)
            resolved_circ = cirq.resolve_parameters(cirq_circ, resolver)
            circuit_to_simulate = resolved_circ if isinstance(resolved_circ, cirq.Circuit) else cirq.Circuit(resolved_circ)
            ev_list = simulator.simulate_expectation_values(circuit_to_simulate,
                                                            observables=[observable],
                                                            permit_terminal_measurements=True)
            energies.append(ev_list[0].real)
            random_params.append(params)
        min_energy = min(energies)
        print("Minimum energy found via Cirq sampling:", min_energy)
        return min_energy

# =====================================================
# Example Usage
# =====================================================
if __name__ == '__main__':
    # --- Build the spin Hamiltonian from a 2D grid graph ---
    # Create a 3x2 grid (6 nodes) and relabel the nodes to 0,...,5.
    graph = nx.generators.lattice.grid_2d_graph(3, 2)
    graph = nx.relabel.convert_node_labels_to_integers(graph)
    nx.draw(graph, labels={i: i for i in range(len(graph.nodes()))})
    plt.show()
    
    # Coupling constants for connected (J1) and non-connected (J2) pairs:
    J1 = -1.5
    J2 = -0.257
    spin_ham_builder = SpinHamiltonianBuilder(graph, J1, J2)
    
    # --- Build a variational ansatz ---
    # Here we choose a layered ansatz with n_layers = 2.
    n_layers = 2
    solver = SpinVQESolver(spin_ham_builder, n_layers=n_layers)
    # For each layer we need a parameter set (here, two parameters per layer)
    params_layers = [[Parameter(f'a{2*layer}'), Parameter(f'a{2*layer+1}')] for layer in range(n_layers)]
    
    variational_circuit = solver.build_variational_circuit(params_layers)
    print("Variational circuit (decomposed):")
    print(variational_circuit.decompose().draw('mpl'))
    
    # --- Solve via Qiskit VQE ---
    print("\n--- Qiskit VQE ---")
    qiskit_result = solver.solve_with_qiskit(variational_circuit)
    
    # --- Solve via QLM ---
    print("\n--- QLM VQE ---")
    # For QLM we need an initial parameter vector. Here we assume 2*n_layers parameters.
    initial_params = np.random.random(size=n_layers*2)
    qlm_results = solver.solve_with_qlm(variational_circuit, initial_params)
    
    # --- Solve via Cirq Simulation ---
    print("\n--- Cirq Simulation ---")
    min_energy_cirq = solver.solve_with_cirq(variational_circuit, num_samples=1000)
