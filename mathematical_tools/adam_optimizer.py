import math
import numpy as np
from circuit_type.circuit import CircuitType
from mathematical_tools.expectations import ExpectationCalculator
from mathematical_tools.gradient import CircuitGradientCalculator

class AdamOptimizer:
    def __init__(self, n_iter, alpha, beta1, beta2, circuit_type: CircuitType, eps=1e-8):
        """
        Adam optimizer integrated with circuit gradient calculation.
        
        Args:
            circuit_type (CircuitType): Specifies the circuit backend.
        """
        self.circuit_type = circuit_type
        self._calculate_gradient = CircuitGradientCalculator(self.circuit_type).calculate_gradient

    def optimize(self, qcs, n_iter, alpha, beta1, beta2, H, w, num_qubits, eps=1e-8):
        """
        Optimize parameters using Adam with integrated circuit gradient computation.
        
        Args:
            qcs: List of quantum circuits.
            H: Observable (Hamiltonian).
            w: Weights for the circuits.
            num_qubits: Total number of qubits.
            n_iter (int): Number of iterations.
            alpha (float): Learning rate.
            beta1 (float): Exponential decay rate for first moment.
            beta2 (float): Exponential decay rate for second moment.
            circuit_type (CircuitType): Specifies the circuit backend.
            eps (float): Small constant to avoid division by zero.

            
        Returns:
            A tuple (best_params, best_cost, trace) after the optimization iterations.
        """
        mean = 0
        std_dev = 2*np.pi
        num_samples = len(list(qcs[0].parameters))
        x = np.array([float(random.gauss(mean, std_dev)) for _ in range(num_samples)], dtype=np.float32)
        m = np.zeros_like(x)
        v = np.zeros_like(x)
        local_trace = []
        
        for t in range(n_iter):
            # Compute gradient and cost with the integrated gradient calculation.
            g, cost = self._calculate_gradient(qcs, H, w, num_qubits, x)
            
            # Update each parameter with the Adam rule.
            for i in range(len(x)):
                m[i] = beta1 * m[i] + (1.0 - beta1) * g[i]
                v[i] = beta2 * v[i] + (1.0 - beta2) * (g[i] ** 2)
                mhat = m[i] / (1.0 - beta1 ** (t + 1))
                vhat = v[i] / (1.0 - beta2 ** (t + 1))
                x[i] = x[i] - alpha * mhat / (math.sqrt(vhat) + eps)
            
            local_trace.append((t, x.copy(), cost))
            print(f"Iteration {t}: cost = {cost:.5f}")
        
        return x, cost, local_trace
