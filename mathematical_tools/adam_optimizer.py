"""Adam optimizer implementation for OpenVQA."""

import numpy as np


class AdamOptimizer:
    """
    Adam optimizer for variational quantum algorithms.
    
    Adaptive Moment Estimation (Adam) is a method for efficient stochastic 
    optimization that only requires first-order gradients with little memory requirement.
    """
    
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Initialize Adam optimizer.
        
        Args:
            learning_rate (float): Learning rate (step size).
            beta1 (float): Exponential decay rate for first moment estimates.
            beta2 (float): Exponential decay rate for second moment estimates.
            epsilon (float): Small constant for numerical stability.
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # First moment vector
        self.v = None  # Second moment vector
        self.t = 0     # Timestep
    
    def step(self, params, gradients):
        """
        Perform a single optimization step.
        
        Args:
            params (np.ndarray): Current parameter values.
            gradients (np.ndarray): Gradient of the objective function.
            
        Returns:
            np.ndarray: Updated parameters.
        """
        params = np.asarray(params)
        gradients = np.asarray(gradients)
        
        # Initialize moment vectors on first step
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        
        # Increment timestep
        self.t += 1
        
        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
        
        # Update biased second raw moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradients ** 2)
        
        # Compute bias-corrected first moment estimate
        m_hat = self.m / (1 - self.beta1 ** self.t)
        
        # Compute bias-corrected second raw moment estimate
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        # Update parameters
        params = params - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        return params
    
    def reset(self):
        """Reset optimizer state."""
        self.m = None
        self.v = None
        self.t = 0
