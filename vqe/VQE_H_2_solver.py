import numpy as np
import pylab
import matplotlib.pyplot as plt
import random

# --- Qiskit Imports ---
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit_algorithms.optimizers import SLSQP
from qiskit.circuit.library import TwoLocal
from qiskit.primitives import Estimator
from qiskit_algorithms import VQE

# --- QLM and Cirq Imports ---
from qat.interop.qiskit import qiskit_to_qlm
from qat.qpus import get_default_qpu
from qat.plugins import ScipyMinimizePlugin
from qat.interop.cirq import qlm_to_cirq
import cirq

# --- Hamiltonian Imports ---
from hamiltonians.h_2 import get_h2_hamiltonian_terms
from hamiltonians.hamiltonian_builder import HamiltonianBuilder


class VQEH2Solver:
    def __init__(self, num_qubits=4, maxiter=100):
        """
        Initializes immutable parameters for the solver.
        
        Note: The Qiskit Hamiltonian, ansatz, estimator, and optimizer are now
        created locally within the methods.
        
        Args:
            num_qubits (int): Number of qubits.
            maxiter (int): Maximum iterations for Qiskit's SLSQP optimizer.
        """
        self.num_qubits = num_qubits
        self.maxiter = maxiter

    def solve_with_qiskit(self):
        """
        Solves the VQE problem using Qiskit's VQE.
        All required objects are created locally within this method.
        
        Returns:
            result: The result object from Qiskit's VQE.
        """
        # Create local Qiskit objects:
        # Build the Hamiltonian and get its Qiskit version
        h2_hamiltonian = HamiltonianBuilder(get_h2_hamiltonian_terms)
        qiskit_h2_hamiltonian = h2_hamiltonian.get_qiskit_hamiltonian()
        
        # Create a parameterized ansatz
        ansatz = TwoLocal(self.num_qubits, "ry", "cz")
        
        # Initialize Qiskit Estimator and Optimizer
        estimator = Estimator()
        optimizer = SLSQP(maxiter=self.maxiter)
        
        # Local callback state to track convergence
        counts = []
        values = []
        def callback(eval_count, parameters, mean, std):
            counts.append(eval_count)
            values.append(mean)
        
        print("Solving with Qiskit VQE...")
        # Create and run the VQE solver
        vqe_solver = VQE(estimator=estimator,
                         ansatz=ansatz,
                         optimizer=optimizer,
                         callback=callback)
        
        result = vqe_solver.compute_minimum_eigenvalue(qiskit_h2_hamiltonian)
        print("Qiskit VQE result:")
        print(result)
        
        # Plot the convergence
        pylab.rcParams["figure.figsize"] = (12, 4)
        plt.figure()
        plt.plot(counts, values, marker='o')
        plt.xlabel("Evaluation Count")
        plt.ylabel("Energy")
        plt.title("Qiskit VQE Convergence")
        plt.grid(True)
        plt.show()
        
        return values[-1]

    def solve_with_qlm(self, optimization_methods=None):
        """
        Solves the VQE problem using QLM with Scipy optimizers.
        All required objects are created locally within this method.
        
        Args:
            optimization_methods (list or None): List of optimization methods to try.
                Defaults to ["COBYLA", "Nelder-Mead", "BFGS"] if None.
        
        Returns:
            dict: Mapping from optimization method names to their result objects.
        """
        print("Solving with QLM...")
        # Create local objects:
        h2_hamiltonian = HamiltonianBuilder(get_h2_hamiltonian_terms)
        ansatz = TwoLocal(self.num_qubits, "ry", "cz")
        decomposed_ansatz = ansatz.decompose()
        qlm_circuit = qiskit_to_qlm(decomposed_ansatz)
        
        # Optionally display the QLM circuit and its variables
        qlm_circuit().display()
        print("QLM circuit variables:", qlm_circuit.get_variables())
        
        # Get the QLM observable
        qlm_observable = h2_hamiltonian.get_qlm_observable()
        print("QLM Observable:", qlm_observable)
        
        # Create a job for the QLM circuit
        job = qlm_circuit.to_job(job_type="OBS",
                                 observable=qlm_observable,
                                 nbshots=0)
        
        # Generate a random initial parameter vector
        theta_0 = np.random.random(size=self.num_qubits * 4)
        
        # Get the default QPU for QLM
        linalg_qpu = get_default_qpu()
        
        if optimization_methods is None:
            optimization_methods = ["COBYLA", "Nelder-Mead", "BFGS"]
        
        results = {}
        traces = {}

        minimum_energy = None
        for method in optimization_methods:
            print(f"\nOptimizing with method: {method}")
            optimizer_scipy = ScipyMinimizePlugin(method=method,
                                                  tol=1e-6,
                                                  options={"maxiter": 200},
                                                  x0=theta_0)
            qpu = optimizer_scipy | linalg_qpu
            res = qpu.submit(job)
            results[method] = res
            print(f"Minimum VQE energy ({method}) = {res.value}")
            if 'optimization_trace' in res.meta_data:
                traces[method] = eval(res.meta_data['optimization_trace'])

            if minimum_energy is None:
                minimum_energy = res.value
            else:
                minimum_energy = min(minimum_energy, res.value)
        
        # Plot the optimization traces if available
        if traces:
            plt.figure()
            for method, trace in traces.items():
                plt.plot(trace, label=method)
            plt.grid(True)
            plt.legend(loc="best")
            plt.xlabel("Steps")
            plt.ylabel("Energy")
            plt.title("QLM Optimization Trace")
            plt.show()
        
        return minimum_energy

    def solve_with_cirq(self, num_samples=1000):
        """
        Solves the VQE problem using Cirq by sampling random parameters.
        All required objects are created locally within this method.
        
        Args:
            num_samples (int): Number of random parameter sets to sample.
        
        Returns:
            float: The minimum energy found.
        """
        print("Solving with Cirq...")
        # Create local objects:
        h2_hamiltonian = HamiltonianBuilder(get_h2_hamiltonian_terms)
        ansatz = TwoLocal(self.num_qubits, "ry", "cz")
        decomposed_ansatz = ansatz.decompose()
        qlm_circuit = qiskit_to_qlm(decomposed_ansatz)
        cirq_circuit = qlm_to_cirq(qlm_circuit)
        print("Cirq Circuit:")
        print(cirq_circuit)
        
        # Get the Cirq observable
        cirq_observable = h2_hamiltonian.get_cirq_observable()
        simulator = cirq.Simulator()
        energies = []
        
        # Determine the number of parameters to optimize (fallback if necessary)
        num_params = len(qlm_circuit.get_variables())
        if num_params == 0:
            num_params = self.num_qubits * 4
        
        # TODO mhh: Check if there is room for parallelization here.
        for _ in range(num_samples):
            # Generate random parameter values in [-1, 1]
            x = [2 * random.random() - 1 for _ in range(num_params)]
            resolver_dict = {f'θ[{i}]': x[i] for i in range(num_params)}
            resolver = cirq.ParamResolver(resolver_dict)
            resolved_circuit = cirq.resolve_parameters(cirq_circuit, resolver)
            circuit_to_simulate = (cirq.Circuit(resolved_circuit)
                                   if not isinstance(resolved_circuit, cirq.Circuit)
                                   else resolved_circuit)
            
            ev_list = simulator.simulate_expectation_values(
                circuit_to_simulate,
                observables=[cirq_observable],
                permit_terminal_measurements=True
            )
            energies.append(ev_list[0].real)
        
        min_energy = min(energies)
        print("The minimum energy found with Cirq is:", min_energy)
        return min_energy


# --- Example Usage ---
if __name__ == "__main__":
    solver = VQEH2Solver(num_qubits=4, maxiter=100)
    
    # Solve using Qiskit's VQE
    qiskit_result = solver.solve_with_qiskit()
    
    # Solve using QLM with several Scipy optimizers
    qlm_results = solver.solve_with_qlm()
    
    # Solve using Cirq by sampling random parameter sets
    cirq_min_energy = solver.solve_with_cirq(num_samples=1000)
