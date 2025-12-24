import numpy as np
from qiskit import Aer, execute
import cirq
from circuit_type.circuit import CircuitType

class ExpectationCalculator:
    def __init__(self, circuit_type: CircuitType):
        """
        Initialize the calculator with a specific circuit type.
        :param circuit_type: A string specifying the circuit type ('qiskit', 'qlm', or 'cirq')
        """
        self.circuit_type = circuit_type

    def calculate(self, circuit, params, H):
        """
        Compute the expectation value using the appropriate backend.
        :param circuit: The quantum circuit object.
        :param params: A list (or array) of parameters.
        :param H: The observable (Hamiltonian) for which the expectation value is computed.
        :return: The real expectation value.
        """
        if self.circuit_type == CircuitType.QISKIT:
            return self._expectation_qiskit(circuit, params, H)
        elif self.circuit_type == CircuitType.QLM:
            return self._expectation_qlm(circuit, params, H)
        elif self.circuit_type == CircuitType.CIRQ:
            return self._expectation_cirq(circuit, params, H)
        else:
            raise ValueError(f"Unsupported circuit type: {self.circuit_type}")

    def _expectation_qiskit(self, circuit, params, H):
        """
        Expectation value using Qiskit.
        Assumes `circuit` is a Qiskit QuantumCircuit and H is a Qiskit operator.
        """
        # Assign parameters to the circuit
        circ = circuit.assign_parameters(params)
        # Use the statevector simulator
        simulator = Aer.get_backend('statevector_simulator')
        # Execute the circuit
        result = execute(circ, simulator).result().get_statevector()
        # Calculate expectation value
        expectation_value = result.expectation_value(H).real
        return expectation_value

    def _expectation_qlm(self, crq, params, H):
        """
        Expectation value using a QLM-style circuit.
        Assumes `crq` has methods get_variables(), bind_variables(), and eval() that returns an object
        with a statevector attribute.
        """
        # Get circuit variables and create a parameter mapping
        vs = [v for v in crq.get_variables()]
        variables = {vs[i]: params[i] for i in range(len(vs))}
        # Bind the parameters to the circuit
        crq = crq.bind_variables(variables)
        # Evaluate to obtain the statevector
        state = crq.eval().statevector
        # Compute <ψ|H|ψ>
        result1 = np.matmul(state.transpose().conjugate(), H.to_matrix(sparse=False))
        expectation_value = np.matmul(result1, state).real
        return float(expectation_value)

    def _expectation_cirq(self, circuit, params, H):
        """
        Expectation value using Cirq.
        Assumes `circuit` is a cirq.Circuit with parameters and H is a Cirq observable.
        """
        # Create a resolver from the circuit's parameters.
        # Here, we sort the circuit's parameters (you may adjust this ordering as needed).
        parameters = sorted(circuit.all_parameters(), key=lambda x: str(x))
        if len(parameters) != len(params):
            raise ValueError("Number of provided parameters does not match the circuit's parameters.")
        resolver = cirq.ParamResolver({param: params[i] for i, param in enumerate(parameters)})
        
        # Resolve the circuit parameters and drop any terminal measurements
        resolved_circuit = cirq.resolve_parameters(circuit, resolver)
        resolved_circuit = cirq.drop_terminal_measurements(resolved_circuit)
        
        # Use Cirq's Simulator to compute expectation values
        simulator = cirq.Simulator()
        result = simulator.simulate_expectation_values(resolved_circuit,
                                                       observables=[H],
                                                       permit_terminal_measurements=False)
        return result[0].real
