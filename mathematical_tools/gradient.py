import numpy as np
from mathematical_tools.expectations import ExpectationCalculator
from circuit_type.circuit import CircuitType


# TODO: mhh seperate functionalities between gradient and cost calculations
class CircuitGradientCalculator:
    def __init__(self, circuit_type: CircuitType):
        """
        Initialize the gradient calculator with a given circuit type.
        This creates an internal ExpectationCalculator to compute expectation values.
        :param circuit_type: A CircuitType specifying the backend.
        """
        self.expectation_calculator = ExpectationCalculator(circuit_type)

    def calculate_gradient(self, qcs, H, w, num_qubits, parameters):
        """
        Compute the gradient and cost for a set of quantum circuits.
        This method performs a parameter shift for each parameter in each circuit (except for the last two qubits)
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
        # Iterate over circuits; note: using num_qubits-2 as in original implementation
        for i in range(num_qubits - 2):
            for j in range(len(parameters)):
                # Create copies for parameter shift
                parameters1 = parameters[:]
                parameters1[j] += np.pi / 2
                parameters2 = parameters[:]
                parameters2[j] += -np.pi / 2
                term = w[i] * 0.5 * (self.expectation_calculator.calculate(qcs[i], parameters1, H) -
                                     self.expectation_calculator.calculate(qcs[i], parameters2, H))
                gradients[j] += term
            cost += w[i] * self.expectation_calculator.calculate(qcs[i], parameters, H)
        return gradients, cost
