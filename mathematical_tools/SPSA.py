import numpy as np
from mathematical_tools.expectations import ExpectationCalculator
from circuit_type.circuit import CircuitType


# TODO: mhh seperate functionalities between gradient and cost calculations
class SPSAGradientCalculator:
    def __init__(self, circuit_type: CircuitType):
        """
        Initialize the gradient calculator with a given circuit type.
        This creates an internal ExpectationCalculator to compute expectation values.
        :param circuit_type: A CircuitType specifying the backend.
        """
        self.expectation_calculator = ExpectationCalculator(circuit_type)

    def calculate_gradient(self, qcs, H, w, num_qubits, parameters,order=1):
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
        delta = 0.1 # perturbation size (can be adjusted depending on the k-th iteration, such as delta_k = delta/(k+1)^µ)
        spsa_vector_1 = np.random.choice([-1, 1], size=len(parameters)) # perturbation vector 1
        spsa_vector_2 = np.random.choice([-1, 1], size=len(parameters)) # perturbation vector 2
        spsa_vector = delta*spsa_vector_1 # perturbation vector, quite simplified due to -1/+1 even-choice distribution. Can be modified for more complex distributions.
        # Create copies for parameter shift
        parameters1 = parameters[:]
        parameters2 = parameters[:]
        # Parameter vector all shift at once
        parameters1 += spsa_vector
        parameters2 += -spsa_vector
        if order==1:
        # Iterate over circuits; note: using num_qubits-2 as in original implementation
            for i in range(num_qubits - 2):
                term = w[i] * 1/delta * (self.expectation_calculator.calculate(qcs[i], parameters1, H) -
                                        self.expectation_calculator.calculate(qcs[i], parameters2, H)) * spsa_vector_1
                gradients += term
                cost += w[i] * self.expectation_calculator.calculate(qcs[i], parameters, H)
        
        
        else:
            for i in range(num_qubits - 2):
            
                # 2nd-SPSA amplitude perturbations    
                second_term = 1/(8*delta**2) * (self.expectation_calculator.calculate(qcs[i], parameters + delta*(spsa_vector_1+spsa_vector_2), H) - self.expectation_calculator.calculate(qcs[i], parameters + delta*(spsa_vector_1), H)- self.expectation_calculator.calculate(qcs[i], parameters - delta*(spsa_vector_1-spsa_vector_2), H) + self.expectation_calculator.calculate(qcs[i], parameters - delta*spsa_vector_1, H))
                spsa_fubini_term = second_term*(np.outer(spsa_vector_1, spsa_vector_2) + np.outer(spsa_vector_2, spsa_vector_1))
                # 1st-SPSA gradient estimation
                term = w[i] * 1/delta * (self.expectation_calculator.calculate(qcs[i], parameters1, H) -
                                        self.expectation_calculator.calculate(qcs[i], parameters2, H)) * spsa_vector_1
                
                # Update gradients using pseudo-inverse of Fubini matrix 
                grad = np.linalg.pinv(spsa_fubini_term).dot(term)
                gradients += term
                cost += w[i] * self.expectation_calculator.calculate(qcs[i], parameters, H)
        return gradients, cost
