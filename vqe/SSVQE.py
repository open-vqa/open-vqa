import math
import random
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------
# Qiskit Imports
# ---------------------------------------------------
from qiskit import Aer, execute, QuantumCircuit, warnings
from qiskit.circuit import Parameter
from qiskit.quantum_info.operators import Operator, Pauli
# (Also importing opflow objects if needed)
from qiskit.opflow import I, X, Y, Z

# ---------------------------------------------------
# QLM / QAT Imports
# ---------------------------------------------------
from qat.core import Observable, Term
from qat.interop.qiskit import qiskit_to_qlm

# ---------------------------------------------------
# Cirq Imports
# ---------------------------------------------------
import cirq
from qat.interop.cirq import qlm_to_cirq

# ---------------------------------------------------
# Custom HamiltonianBuilder (as given)
# ---------------------------------------------------
class HamiltonianBuilder:
    def __init__(self, num_qubits, terms):
        """
        Parameters:
            num_qubits (int): Number of qubits (and length of each Pauli string).
            terms (list): List of terms (each either a tuple: (coefficient, pauli_string) or
                          a dict with keys 'coefficient' and 'pauli').
        """
        self.num_qubits = num_qubits
        self.terms = []
        for term in terms:
            if isinstance(term, dict):
                coeff = term.get('coefficient')
                pauli = term.get('pauli')
            elif isinstance(term, (tuple, list)):
                coeff, pauli = term
            else:
                raise ValueError("Each term must be a tuple or dict.")
            if len(pauli) != num_qubits:
                raise ValueError(f"Pauli string '{pauli}' length does not equal num_qubits={num_qubits}.")
            self.terms.append({'coefficient': coeff, 'pauli': pauli})

    def get_qiskit_hamiltonian(self):
        """
        Returns the Hamiltonian as a sum of Operator(Pauli) objects.
        """
        hamiltonian = None
        for term in self.terms:
            coeff = term['coefficient']
            pauli_str = term['pauli']
            term_op = coeff * Operator(Pauli(pauli_str))
            hamiltonian = term_op if hamiltonian is None else hamiltonian + term_op
        return hamiltonian

    def get_qlm_observable(self) -> Observable:
        """
        Returns the Hamiltonian as an Observable (using QAT's Term objects).
        """
        pauli_terms = []
        for term in self.terms:
            coeff = term['coefficient']
            pauli_str = term['pauli']
            qubit_indices = list(range(self.num_qubits))
            pauli_terms.append(Term(coeff, pauli_str, qubit_indices))
        observable = Observable(self.num_qubits, pauli_terms=pauli_terms, constant_coeff=0)
        return observable

    def get_cirq_observable(self):
        """
        Returns the Hamiltonian as a Cirq PauliSum.
        """
        qubits = [cirq.LineQubit(i) for i in range(self.num_qubits)]
        pauli_sum = cirq.PauliSum()
        for term in self.terms:
            coeff = term['coefficient']
            pauli_str = term['pauli']
            ps_dict = {}
            for i, letter in enumerate(pauli_str):
                if letter == 'I':
                    continue
                elif letter == 'X':
                    ps_dict[qubits[i]] = cirq.X
                elif letter == 'Y':
                    ps_dict[qubits[i]] = cirq.Y
                elif letter == 'Z':
                    ps_dict[qubits[i]] = cirq.Z
                else:
                    raise ValueError(f"Unsupported Pauli letter: '{letter}' in term '{pauli_str}'")
            term_pauli_string = cirq.PauliString(ps_dict, coefficient=coeff)
            pauli_sum += term_pauli_string
        return pauli_sum

# ---------------------------------------------------
# Generic Adam Optimizer (backend–independent)
# ---------------------------------------------------
class AdamOptimizer:
    def __init__(self, n_iter, alpha, beta1, beta2, eps=1e-8):
        """
        A simple Adam optimizer.

        Args:
            n_iter (int): Number of iterations.
            alpha (float): Step size.
            beta1 (float): Exponential decay factor for first moment.
            beta2 (float): Exponential decay factor for second moment.
            eps (float): A small number to avoid division by zero.
        """
        self.n_iter = n_iter
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.history = []  # Each entry: (iteration, parameter vector, cost)

    def optimize(self, initial_params, grad_func):
        """
        Optimize parameters using the supplied gradient function.

        Args:
            initial_params (np.ndarray): Initial parameter vector.
            grad_func (callable): Function f(x) returning (gradient, cost).

        Returns:
            (best_params, best_cost)
        """
        x = np.array(initial_params, dtype=np.float32)
        m = np.zeros_like(x)
        v = np.zeros_like(x)
        for t in range(self.n_iter):
            g, cost = grad_func(x)
            for i in range(len(x)):
                m[i] = self.beta1 * m[i] + (1.0 - self.beta1) * g[i]
                v[i] = self.beta2 * v[i] + (1.0 - self.beta2) * (g[i] ** 2)
                mhat = m[i] / (1.0 - self.beta1 ** (t + 1))
                vhat = v[i] / (1.0 - self.beta2 ** (t + 1))
                x[i] = x[i] - self.alpha * mhat / (math.sqrt(vhat) + self.eps)
            self.history.append((t, x.copy(), cost))
            print(f"Iteration {t}: cost = {cost:.5f}")
        return x, cost

# ---------------------------------------------------
# H2 Adam Solver: Encapsulates expectation, gradient, and Adam for three backends.
# ---------------------------------------------------
class H2AdamSolver:
    def __init__(self, qc_ground, qc_excited, ham_builder: HamiltonianBuilder, num_qubits=4):
        """
        Initializes the variational H₂ solver.

        Args:
            qc_ground (QuantumCircuit): Qiskit circuit for (approximate) ground state.
            qc_excited (QuantumCircuit): Qiskit circuit for (approximate) excited state.
            ham_builder (HamiltonianBuilder): Instance to build the Hamiltonian.
            num_qubits (int): Number of qubits.
        """
        self.num_qubits = num_qubits
        self.w = np.arange(num_qubits, num_qubits - 2, -1)  # e.g. for 4 qubits: [4, 3]

        # Qiskit circuits
        self.qcs_qiskit = [qc_ground, qc_excited]
        self.H_qiskit = ham_builder.get_qiskit_hamiltonian()

        # QLM circuits (converted from Qiskit)
        self.qcs_qlm = [qiskit_to_qlm(qc) for qc in self.qcs_qiskit]
        self.qlm_observable = ham_builder.get_qlm_observable()

        # Cirq circuits (converted from QLM)
        self.qcs_cirq = [qlm_to_cirq(q) for q in self.qcs_qlm]
        # For parameter resolution assume all circuits share the same free variables.
        self.all_variables = self.qcs_cirq[0].get_variables()
        self.cirq_observable = ham_builder.get_cirq_observable()

        # Simulators
        self.aer_simulator = Aer.get_backend('statevector_simulator')
        self.cirq_simulator = cirq.Simulator()

    # -------------------------
    # Qiskit expectation & gradient
    # -------------------------
    def expectation_qiskit(self, circuit: QuantumCircuit, params):
        circ = circuit.assign_parameters(params)
        result = execute(circ, self.aer_simulator).result().get_statevector()
        # Use Qiskit's built-in method (this syntax may vary with Qiskit versions)
        return result.expectation_value(self.H_qiskit).real

    def gradient_qiskit(self, params):
        gradients = np.zeros(len(params))
        cost = 0.0
        params = list(params)
        for i in range(self.num_qubits - 2):
            for j in range(len(params)):
                params1 = params[:]
                params1[j] += np.pi / 2
                params2 = params[:]
                params2[j] -= np.pi / 2
                term = self.w[i] * 0.5 * (
                    self.expectation_qiskit(self.qcs_qiskit[i], params1) -
                    self.expectation_qiskit(self.qcs_qiskit[i], params2)
                )
                gradients[j] += term
            cost += self.w[i] * self.expectation_qiskit(self.qcs_qiskit[i], params)
        return gradients, cost

    # -------------------------
    # QLM expectation & gradient
    # -------------------------
    def expectation_qlm(self, crq, params):
        vs = list(crq.get_variables())
        binding = {vs[i]: params[i] for i in range(len(vs))}
        crq_bound = crq.bind_variables(binding)
        statevector = crq_bound.eval().statevector
        result1 = np.matmul(statevector.conjugate().T, self.qlm_observable.to_matrix(sparse=False))
        return float(np.matmul(result1, statevector).real)

    def gradient_qlm(self, params):
        gradients = np.zeros(len(params))
        cost = 0.0
        params = list(params)
        for i in range(self.num_qubits - 2):
            for j in range(len(params)):
                params1 = params[:]
                params1[j] += np.pi / 2
                params2 = params[:]
                params2[j] -= np.pi / 2
                term = self.w[i] * 0.5 * (
                    self.expectation_qlm(self.qcs_qlm[i], params1) -
                    self.expectation_qlm(self.qcs_qlm[i], params2)
                )
                gradients[j] += term
            cost += self.w[i] * self.expectation_qlm(self.qcs_qlm[i], params)
        return gradients, cost

    # -------------------------
    # Cirq expectation & gradient
    # -------------------------
    def expectation_circ(self, circuit, params):
        resolver = cirq.ParamResolver({self.all_variables[i]: params[i] for i in range(len(params))})
        resolved_circuit = cirq.resolve_parameters(circuit, resolver)
        resolved_circuit = cirq.drop_terminal_measurements(resolved_circuit)
        result = self.cirq_simulator.simulate_expectation_values(
            cirq.Circuit(resolved_circuit),
            observables=[self.cirq_observable],
            permit_terminal_measurements=False
        )
        return result[0].real

    def gradient_circ(self, params):
        gradients = np.zeros(len(params))
        cost = 0.0
        params = list(params)
        for i in range(self.num_qubits - 2):
            for j in range(len(params)):
                params1 = params[:]
                params1[j] += np.pi / 2
                params2 = params[:]
                params2[j] -= np.pi / 2
                term = self.w[i] * 0.5 * (
                    self.expectation_circ(self.qcs_cirq[i], params1) -
                    self.expectation_circ(self.qcs_cirq[i], params2)
                )
                gradients[j] += term
            cost += self.w[i] * self.expectation_circ(self.qcs_cirq[i], params)
        return gradients, cost

    # -------------------------
    # Optimization routines (using Adam)
    # -------------------------
    def optimize_qiskit(self, n_iter=35, alpha=0.1, beta1=0.9, beta2=0.999, initial_params=None):
        if initial_params is None:
            # Assume the Qiskit circuit's parameters define the dimension.
            num_params = len(list(self.qcs_qiskit[0].parameters))
            initial_params = np.array([random.gauss(0, 2 * np.pi) for _ in range(num_params)], dtype=np.float32)
        optimizer = AdamOptimizer(n_iter, alpha, beta1, beta2)
        best_params, best_cost = optimizer.optimize(initial_params, self.gradient_qiskit)
        return best_params, best_cost, optimizer.history

    def optimize_qlm(self, n_iter=35, alpha=0.1, beta1=0.9, beta2=0.999, initial_params=None):
        if initial_params is None:
            num_params = len(self.qcs_qlm[0].get_variables())
            initial_params = np.array([random.gauss(0, 2 * np.pi) for _ in range(num_params)], dtype=np.float32)
        optimizer = AdamOptimizer(n_iter, alpha, beta1, beta2)
        best_params, best_cost = optimizer.optimize(initial_params, self.gradient_qlm)
        return best_params, best_cost, optimizer.history

    def optimize_cirq(self, n_iter=35, alpha=0.1, beta1=0.9, beta2=0.999, initial_params=None):
        if initial_params is None:
            num_params = len(self.all_variables)
            initial_params = np.array([random.gauss(0, 2 * np.pi) for _ in range(num_params)], dtype=np.float32)
        optimizer = AdamOptimizer(n_iter, alpha, beta1, beta2)
        best_params, best_cost = optimizer.optimize(initial_params, self.gradient_circ)
        return best_params, best_cost, optimizer.history

    # -------------------------
    # Helpers to obtain final energies
    # -------------------------
    def get_ground_and_excited_energy_qiskit(self, params):
        ground = self.expectation_qiskit(self.qcs_qiskit[0], params)
        excited = self.expectation_qiskit(self.qcs_qiskit[1], params)
        return ground, excited

    def get_ground_and_excited_energy_qlm(self, params):
        ground = self.expectation_qlm(self.qcs_qlm[0], params)
        excited = self.expectation_qlm(self.qcs_qlm[1], params)
        return ground, excited

    def get_ground_and_excited_energy_cirq(self, params):
        ground = self.expectation_circ(self.qcs_cirq[0], params)
        excited = self.expectation_circ(self.qcs_cirq[1], params)
        return ground, excited

# ---------------------------------------------------
# Example Usage
# ---------------------------------------------------
if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    # --- Build the H2 Hamiltonian ---
    # Here we assume that get_h2_hamiltonian_terms is defined in your hamiltonians.h_2 module.
    # For demonstration we mimic the terms used in your file.
    h2_terms = [
        (-0.24274280046588792, 'IIZI'),
        (-0.24274280046588792, 'IIIZ'),
        (-0.04207898539364302, 'IIII'),
        (0.17771287502681438, 'ZIII'),
        (0.1777128750268144,  'IZII'),
        (0.12293305045316086, 'ZIZI'),
        (0.12293305045316086, 'IZIZ'),
        (0.16768319431887935, 'ZIIZ'),
        (0.16768319431887935, 'IZZI'),
        (0.1705973836507714,  'ZZII'),
        (0.1762764072240811,  'IIZZ'),
        (-0.044750143865718496, 'YYXX'),
        (-0.044750143865718496, 'XXYY'),
        (0.044750143865718496, 'YXXY'),
        (0.044750143865718496, 'XYYX')
    ]
    num_qubits = 4
    ham_builder = HamiltonianBuilder(num_qubits, h2_terms)

    # --- Build the variational ansatz ---
    # Construct a Qiskit circuit for the ground state ansatz.
    qc = QuantumCircuit(num_qubits)
    # For demonstration, we define two layers (D1 and D2) as in your file.
    D1 = 2
    D2 = 8
    num_p = 4 * D1 + 8 * D2 + 8
    parameters = [Parameter(f'p{i}') for i in range(num_p)]
    k = 0
    for i in range(D1):
        qc.rx(parameters[k], 2); k += 1
        qc.rx(parameters[k], 3); k += 1
        qc.rz(parameters[k], 2); k += 1
        qc.rz(parameters[k], 3); k += 1
        qc.cz(2, 3)
    for i in range(D2):
        qc.rx(parameters[k], 0); k += 1
        qc.rx(parameters[k], 1); k += 1
        qc.rx(parameters[k], 2); k += 1
        qc.rx(parameters[k], 3); k += 1
        qc.rz(parameters[k], 0); k += 1
        qc.rz(parameters[k], 1); k += 1
        qc.rz(parameters[k], 2); k += 1
        qc.rz(parameters[k], 3); k += 1
        qc.cz(0, 1)
        qc.cz(1, 2)
        qc.cz(2, 3)
    qc.rx(parameters[k], 0); k += 1
    qc.rx(parameters[k], 1); k += 1
    qc.rx(parameters[k], 2); k += 1
    qc.rx(parameters[k], 3); k += 1
    qc.rz(parameters[k], 0); k += 1
    qc.rz(parameters[k], 1); k += 1
    qc.rz(parameters[k], 2); k += 1
    qc.rz(parameters[k], 3); k += 1

    # Define a second circuit (for an excited state guess) by applying an extra X gate.
    qc2 = QuantumCircuit(num_qubits)
    qc2.x(0)
    qc2 = qc2.compose(qc)

    # --- Instantiate the H2AdamSolver ---
    solver = H2AdamSolver(qc, qc2, ham_builder, num_qubits=num_qubits)

    # --- Optimize using Qiskit/Aer backend ---
    print("\nOptimizing using Qiskit/Aer:")
    best_params_qiskit, cost_qiskit, history_qiskit = solver.optimize_qiskit(n_iter=35, alpha=0.1)
    ground_qiskit, excited_qiskit = solver.get_ground_and_excited_energy_qiskit(best_params_qiskit)
    print("Qiskit ground state energy:", ground_qiskit)
    print("Qiskit excited state energy:", excited_qiskit)

    # --- Optimize using QLM backend ---
    print("\nOptimizing using QLM:")
    best_params_qlm, cost_qlm, history_qlm = solver.optimize_qlm(n_iter=35, alpha=0.1)
    ground_qlm, excited_qlm = solver.get_ground_and_excited_energy_qlm(best_params_qlm)
    print("QLM ground state energy:", ground_qlm)
    print("QLM excited state energy:", excited_qlm)

    # --- Optimize using Cirq backend ---
    print("\nOptimizing using Cirq:")
    best_params_cirq, cost_cirq, history_cirq = solver.optimize_cirq(n_iter=35, alpha=0.1)
    ground_cirq, excited_cirq = solver.get_ground_and_excited_energy_cirq(best_params_cirq)
    print("Cirq ground state energy:", ground_cirq)
    print("Cirq excited state energy:", excited_cirq)

    # --- Plot optimization history (using Qiskit as an example) ---
    energies = [entry[2] for entry in history_qiskit]
    plt.figure()
    plt.plot(range(len(energies)), energies, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title("Qiskit/Aer Optimization History (Adam)")
    plt.grid(True)
    plt.show()
