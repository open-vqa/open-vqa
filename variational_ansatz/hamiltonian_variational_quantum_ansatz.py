# 

import numpy as np
import random
import matplotlib.pyplot as plt

# ------------------------------
# QAT / QLM / Qiskit / Cirq Imports
# ------------------------------
import cirq
from qiskit.quantum_info.operators import Operator, Pauli
from qat.core import Observable, Term
from qat.interop.qiskit import qiskit_to_qlm
from qat.interop.cirq import qlm_to_cirq
from qat.qpus import get_default_qpu
from qat.plugins import ScipyMinimizePlugin

# ------------------------------
# The HamiltonianBuilder class
# ------------------------------

class HamiltonianBuilder:
    def __init__(self, num_qubits, terms):
        """
        Parameters:
            num_qubits (int): Number of qubits (and the length of each Pauli string).
            terms (list): A list of terms. Each term can be either:
                          - a tuple: (coefficient, pauli_string) or
                          - a dict with keys 'coefficient' and 'pauli'
                          For example: (-0.2427, 'IIZI')
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
            # Create an operator from the Pauli string.
            term_op = coeff * Operator(Pauli(pauli_str))
            if hamiltonian is None:
                hamiltonian = term_op
            else:
                hamiltonian = hamiltonian + term_op
        return hamiltonian

    def get_qlm_observable(self) -> Observable:
        """
        Returns the Hamiltonian as an Observable object made of Term entries.
        """
        pauli_terms = []
        for term in self.terms:
            coeff = term['coefficient']
            pauli_str = term['pauli']
            # Here the qubit indices are assumed to be 0, 1, ..., num_qubits-1.
            qubit_indices = list(range(self.num_qubits))
            pauli_terms.append(Term(coeff, pauli_str, qubit_indices))
        # (For now constant_coeff is set to 0; adjust as needed.)
        observable = Observable(self.num_qubits, pauli_terms=pauli_terms, constant_coeff=0)
        return observable

    def get_cirq_observable(self):
        """
        Returns the Hamiltonian as a Cirq PauliSum.
        """
        # Create a list of Cirq qubits.
        qubits = [cirq.LineQubit(i) for i in range(self.num_qubits)]
        # Build a PauliSum from individual PauliStrings.
        pauli_sum = cirq.PauliSum()  # start with an empty sum
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
            # Construct the PauliString with the given coefficient.
            term_pauli_string = cirq.PauliString(ps_dict, coefficient=coeff)
            pauli_sum += term_pauli_string
        return pauli_sum

# ------------------------------
# The HamiltonianVariationalAnsatz class
# ------------------------------

class HamiltonianVariationalAnsatz:
    def __init__(self, ansatz, num_qubits, hamiltonian_terms, num_params):
        """
        Initializes the variational ansatz algorithm.
        
        Args:
            ansatz: A Qiskit circuit representing the variational ansatz.
            num_qubits (int): Number of qubits.
            hamiltonian_terms (list): List of Hamiltonian terms (tuples or dicts).
            num_params (int): Dimension of the variational parameter space.
        """
        self.ansatz = ansatz
        self.num_qubits = num_qubits
        self.num_params = num_params
        # Create the HamiltonianBuilder instance.
        self.ham_builder = HamiltonianBuilder(num_qubits, hamiltonian_terms)

    def optimize_with_qlm(self, optimization_methods=None):
        """
        Performs variational optimization using QLM.
        The Qiskit ansatz is converted to a QLM circuit and combined with
        the Hamiltonian observable from the HamiltonianBuilder.
        
        Args:
            optimization_methods (list or None): List of optimizer names to try.
                If None, defaults to ["COBYLA", "Nelder-Mead", "BFGS"].
        
        Returns:
            dict: Mapping from optimization method names to their result objects.
        """
        print("Optimizing with QLM...")
        # Convert the Qiskit ansatz to a QLM circuit.
        qlm_circuit = qiskit_to_qlm(self.ansatz)
        qlm_circuit().display()
        print("QLM circuit variables:", qlm_circuit.get_variables())
        
        # Get the QLM observable from the HamiltonianBuilder.
        observable = self.ham_builder.get_qlm_observable()
        print("QLM Observable:")
        print(observable)
        
        # Create a job for observable estimation.
        job = qlm_circuit.to_job(job_type="OBS", observable=observable, nbshots=0)
        
        # Generate a random initial parameter vector.
        theta_0 = np.random.random(size=self.num_params)
        
        # Get the default QLM QPU.
        linalg_qpu = get_default_qpu()
        
        if optimization_methods is None:
            optimization_methods = ["COBYLA", "Nelder-Mead", "BFGS"]
        
        results = {}
        traces = {}
        for method in optimization_methods:
            print(f"\nUsing optimizer: {method}")
            optimizer_scipy = ScipyMinimizePlugin(
                method=method,
                tol=1e-6,
                options={"maxiter": 200},
                x0=theta_0
            )
            qpu = optimizer_scipy | linalg_qpu
            res = qpu.submit(job)
            results[method] = res
            print(f"Minimum VQE energy ({method}) = {res.value}")
            if 'optimization_trace' in res.meta_data:
                traces[method] = eval(res.meta_data['optimization_trace'])
        
        if traces:
            plt.figure()
            for method, trace in traces.items():
                plt.plot(trace, label=method)
            plt.xlabel("Steps")
            plt.ylabel("Energy")
            plt.title("QLM Optimization Trace")
            plt.legend(loc="best")
            plt.grid(True)
            plt.show()
        
        return results

    def simulate_with_cirq(self, num_samples=1000):
        """
        Simulates the variational ansatz using Cirq by sampling random parameters.
        The QLM circuit is first converted to a Cirq circuit. Then, using a Cirq simulator,
        the expectation value of the corresponding observable is computed.
        
        Args:
            num_samples (int): Number of random parameter sets to sample.
        
        Returns:
            tuple: (min_energy, energies, random_params)
                - min_energy: the minimum energy found.
                - energies: list of energies computed.
                - random_params: list of parameter vectors used.
        """
        print("Simulating with Cirq...")
        # Convert the Qiskit ansatz to a QLM circuit and then to a Cirq circuit.
        qlm_circuit = qiskit_to_qlm(self.ansatz)
        cirq_circuit = qlm_to_cirq(qlm_circuit)
        print("Cirq Circuit:")
        print(cirq_circuit)
        
        # Get the Cirq observable from the HamiltonianBuilder.
        observable = self.ham_builder.get_cirq_observable()
        
        simulator = cirq.Simulator()
        energies = []
        random_params = []

        for _ in range(num_samples):
            # Generate a random parameter vector (values in [-1,1]).
            params = [2 * random.random() - 1 for _ in range(self.num_params)]
            # Here we assume the variational parameters in the QLM/Cirq circuit
            # are named as 't[0]', 't[1]', etc.
            resolver_dict = {f't[{i}]': params[i] for i in range(self.num_params)}
            resolver = cirq.ParamResolver(resolver_dict)
            resolved_circuit = cirq.resolve_parameters(cirq_circuit, resolver)
            # Ensure we have a Cirq Circuit.
            if not isinstance(resolved_circuit, cirq.Circuit):
                circuit_to_simulate = cirq.Circuit(resolved_circuit)
            else:
                circuit_to_simulate = resolved_circuit
            # Compute the expectation value.
            ev_list = simulator.simulate_expectation_values(
                circuit_to_simulate,
                observables=[observable],
                permit_terminal_measurements=True
            )
            energies.append(ev_list[0].real)
            random_params.append(params)
        
        min_energy = min(energies)
        print("The minimum energy found with Cirq is:", min_energy)
        return min_energy, energies, random_params

# ------------------------------
# Example Usage
# ------------------------------

if __name__ == '__main__':
    # For demonstration we construct a simple Qiskit variational ansatz.
    # (In practice, replace this with your own ansatz.)
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector

    num_qubits = 4
    num_params = 4  # Adjust as appropriate for your ansatz
    params = ParameterVector('t', length=num_params)
    qc = QuantumCircuit(num_qubits)
    # A simple ansatz: apply a parameterized RX gate on each qubit.
    for i in range(num_qubits):
        qc.rx(params[i], i)
    # (You may add entangling gates or more layers as needed.)
    
    # Define the list of Hamiltonian terms.
    # (Each term is a tuple: (coefficient, pauli_string).)
    terms = [
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
    
    # Instantiate the variational ansatz algorithm.
    hv_ansatz = HamiltonianVariationalAnsatz(
        ansatz=qc,
        num_qubits=num_qubits,
        hamiltonian_terms=terms,
        num_params=num_params
    )
    
    # Run QLM optimization.
    qlm_results = hv_ansatz.optimize_with_qlm()
    
    # Run Cirq simulation.
    min_energy, energies, random_params = hv_ansatz.simulate_with_cirq(num_samples=1000)
