import numpy as np
from mathematical_tools.expectations import ExpectationCalculator
from circuit_type.circuit import CircuitType
from qiskit import Aer, execute
import cirq
from qat.qpus import get_default_qpu

class QNSPSA_PSR_Calculator:
    def __init__(self, circuit_type: CircuitType):
        """
        Initialize the gradient calculator with a given circuit type.
        This creates an internal ExpectationCalculator to compute expectation values.
        :param circuit_type: A CircuitType specifying the backend.
        """
        self.expectation_calculator = ExpectationCalculator(circuit_type)
        self.circuit_type = circuit_type

    
    def assign_parameters(self, circuit, parameters):
        """
        Assign or bind parameters to a quantum circuit for Qiskit, Cirq, or QLM.

        Args:
            circuit: The quantum circuit object
            parameters: A list, numpy array, or dictionary of parameter values
            backend: 'qiskit', 'cirq', or 'qlm'

        Returns:
            Circuit with assigned (bound) parameters
        """
        # ============================================================
        #  QISKIT BACKEND
        # ============================================================
        if self.circuit_type == CircuitType.QISKIT:
            return circuit.assign_parameters(parameters)
        # ============================================================
        #  CIRQ BACKEND
        # ============================================================
        elif self.circuit_type == CircuitType.CIRQ:
            param_dict = {sym: val for sym, val in zip(circuit.all_parameters, parameters)}
            return cirq.resolve_parameters(circuit, param_dict)
        # ============================================================
        #  QLM BACKEND
        # ============================================================
        elif self.circuit_type == CircuitType.QLM:
            vs = [v for v in circuit.get_variables()]
            variables = {vs[i]: parameters[i] for i in range(len(vs))}
            return circuit.bind_variables(variables)


    def swap_test(self,circuit1, circuit2):
        """
        Compute the fidelity between two quantum states using statevector simulation.

        Args:
            circuit1: First quantum circuit (Qiskit, Cirq, or QLM object)
            circuit2: Second quantum circuit

        Returns:
            fidelity (float): |<psi|phi>|²
        """

        # ============================================================
        #  QISKIT BACKEND
        # ============================================================
        if self.circuit_type == CircuitType.QISKIT:

            backend_sim = Aer.get_backend("statevector_simulator")

            def get_state(circ):
                result = execute(circ, backend_sim).result()
                return np.array(result.get_statevector())

            state1 = get_state(circuit1)
            state2 = get_state(circuit2)

        # ============================================================
        #  CIRQ BACKEND
        # ============================================================
        elif self.circuit_type == CircuitType.CIRQ:

            simulator = cirq.Simulator()

            def get_state(circ):
                result = simulator.simulate(circ)
                return np.array(result.final_state_vector)

            state1 = get_state(circuit1)
            state2 = get_state(circuit2)

        # ============================================================
        #  QLM BACKEND
        # ============================================================
        elif self.circuit_type == CircuitType.QLM:

            qpu = get_default_qpu()

            def get_state(circ):
                job = circ.to_job()
                result = qpu.submit(job)
                return np.array(result.state.vector)

            state1 = get_state(circuit1)
            state2 = get_state(circuit2)

        else:
            raise ValueError("Unsupported backend. Please use 'qiskit', 'cirq', or 'qlm' instead.")

        # ============================================================
        #  Compute Fidelity
        # ============================================================
        overlap = np.vdot(state1, state2)
        fidelity = np.abs(overlap) ** 2

        return fidelity

    def calculate_gradient(self, qcs, H, w, num_qubits, parameters):
        """
        Compute the gradient and cost for a set of quantum circuits.
        This method performs a stochastic calculation of Fubini study matrix and exact PSR gradient gradient in each circuit (except for the last two qubits)
        and uses the appropriate expectation calculator.

        :param qcs: List of quantum circuits.
        :param H: The observable (Hamiltonian) for which the expectation value is computed.
        :param w: List or array of weights corresponding to each circuit.
        :param num_qubits: Total number of qubits in the circuits.
        :param parameters: List (or array) of parameters for the circuits.
        :return: A tuple (gradients, cost) where gradients is an array of partial derivatives.
        """

        gradients = np.zeros((len(parameters),))
        cost = 0
        # Ensure parameters is mutable
        parameters = list(parameters)
        delta = 0.1 # perturbation size (can be adjusted depending on the k-th iteration, such as delta_k = delta/(k+1)^µ)


        # Iterate over circuits; note: using num_qubits-2 as in original implementation
        for i in range(num_qubits - 2):
            spsa_vector_1 = np.random.choice([-1, 1], size=len(parameters)) # perturbation vector 1
            spsa_vector_2 = np.random.choice([-1, 1], size=len(parameters)) # perturbation vector 2
            internal_initial_point = parameters[:]
            # Parameter shift points
            initial_plus1_plus2 = np.add(internal_initial_point, np.add(delta*spsa_vector_1, delta*spsa_vector_2))
            initial_plus1 = np.add(internal_initial_point, delta*spsa_vector_1)
            initial_minus1_plus2 = np.subtract(internal_initial_point, np.subtract(delta*spsa_vector_1, delta*spsa_vector_2))
            initial_minus1 = np.subtract(internal_initial_point,delta*spsa_vector_1)
            # Circuit evaluations at shifted points
            ansatz_initial = self.assign_parameters(qcs[i], internal_initial_point)
            ansatz_plus1_plus2 = self.assign_parameters(qcs[i], initial_plus1_plus2)  
            ansatz_plus1 = self.assign_parameters(qcs[i], initial_plus1)  
            ansatz_minus1_plus2 = self.assign_parameters(qcs[i], initial_minus1_plus2)   
            ansatz_minus1 =  self.assign_parameters(qcs[i], initial_minus1)  
            # QNSPSA calculations
            deltaF = self.swapTest(ansatz_initial, ansatz_plus1_plus2) - self.swapTest(ansatz_initial, ansatz_plus1) - self.swapTest(ansatz_initial, ansatz_minus1_plus2) + self.swapTest(ansatz_initial, ansatz_minus1)
            spsa_fubini_term = -1/4*(deltaF/(2*delta**2))*(np.outer(spsa_vector_1, spsa_vector_2) + np.outer(spsa_vector_2, spsa_vector_1))

            # Parameter shift rule for each parameter
            grad_vec = np.zeros((len(parameters),))
            for j in range(len(parameters)):
                # Create copies for parameter shift
                parameters1 = parameters[:]
                parameters1[j] += np.pi / 2
                parameters2 = parameters[:]
                parameters2[j] += -np.pi / 2
                term = w[i] * 0.5 * (self.expectation_calculator.calculate(qcs[i], parameters1, H) -
                                     self.expectation_calculator.calculate(qcs[i], parameters2, H))
                grad_vec[j] = term
                
            # Update gradients using pseudo-inverse of Fubini matrix approximation
            term_update = np.linalg.pinv(spsa_fubini_term).dot(grad_vec)
            gradients += term_update
            cost += w[i] * self.expectation_calculator.calculate(qcs[i], parameters, H)
        return gradients, cost
