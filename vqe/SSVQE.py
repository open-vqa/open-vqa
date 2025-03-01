import numpy as np
import pylab
import matplotlib.pyplot as plt
import random

# --- QLM and Cirq Imports ---
from qat.interop.qiskit import qiskit_to_qlm
from qat.qpus import get_default_qpu
from qat.plugins import ScipyMinimizePlugin
from qat.interop.cirq import qlm_to_cirq
import cirq

# --- Hamiltonian Imports ---
from hamiltonians.h_2 import get_h2_hamiltonian_terms
from hamiltonians.hamiltonian_builder import HamiltonianBuilder


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
from mathematical_tools.adam_optimizer import AdamOptimizer
from circuit_type.circuit import CircuitType
from mathematical_tools.expectations import ExpectationCalculator

# TODO: mhh currenctly building the circuit contains a lot of repetitive code
# the goal is to build a generalized ciruit builder or a vqa ciruitbuilder
class SSVQEH2Solver:
    def __init__(self):
        pass

    def _build_qiskit_circuits():
        qc = QuantumCircuit(4)

        D1 = 2
        D2 = 8
        num_p = 4*D1 + 8*D2 + 8

        prs = [Parameter('p'+str(i)) for i in range(num_p)]

        k = 0
        for i in range(D1):
            qc.rx(prs[k], 2)
            k = k+1
            qc.rx(prs[k], 3)
            k = k+1

            qc.rz(prs[k], 2)
            k = k+1
            qc.rz(prs[k], 3)
            k = k+1

            qc.cz(2, 3)

        for i in range(D2):
            qc.rx(prs[k], 0)
            k = k+1
            qc.rx(prs[k], 1)
            k = k+1
            qc.rx(prs[k], 2)
            k = k+1
            qc.rx(prs[k], 3)
            k = k+1

            qc.rz(prs[k], 0)
            k = k+1
            qc.rz(prs[k], 1)
            k = k+1
            qc.rz(prs[k], 2)
            k = k+1
            qc.rz(prs[k], 3)
            k = k+1

            

            qc.cz(0, 1)
            qc.cz(1, 2)
            qc.cz(2, 3)

        qc.rx(prs[k], 0)
        k = k+1
        qc.rx(prs[k], 1)
        k = k+1
        qc.rx(prs[k], 2)
        k = k+1
        qc.rx(prs[k], 3)
        k = k+1

        qc.rz(prs[k], 0)
        k = k+1
        qc.rz(prs[k], 1)
        k = k+1
        qc.rz(prs[k], 2)
        k = k+1
        qc.rz(prs[k], 3)
        k = k+1
        qc2 = QuantumCircuit(4)
        qc2.x(0)
        qc2 = qc2.compose(qc)
        return [qc, qc2]

    # TODO mhh: those 3 solvers can be merged in one if we give them parameters 
    def solve_with_qiskit(self):
        h2_hamiltonian_builder = HamiltonianBuilder(get_h2_hamiltonian_terms())
        qiskit_h2_hamiltonian = h2_hamiltonian_builder.get_qiskit_hamiltonian()

        qcs = self._build_qiskit_circuits()

        w = np.arange(4, 2, -1)

        local_trace = []
        # define the total iterations
        max_n_iter = 35
        # steps size
        alpha = 0.1
        # factor for average gradient
        beta1 = 0.9
        # factor for average squared gradient
        beta2 = 0.999
        # perform the gradient descent search with adam

        adam_optimizer = AdamOptimizer(circuit_type=CircuitType.QISKIT)
        best, score, local_trace = \
            adam_optimizer.optimize(qcs = qcs, n_iter=max_n_iter,
                                    alpha=alpha, beta1=beta1, beta2=beta2,
                                    H=qiskit_h2_hamiltonian, w=w, num_qubits=4,
                                    eps=1e-8)
        print('Done!')
        print('f(%s) = %f' % (best, score))
        # here we are not returning energies
        energies = [local_trace[i][2] for i in range(len(local_trace))]
        best=local_trace[-1][1]

        expectation_calculator = ExpectationCalculator(circuit_type=CircuitType.QISKIT)
        ground_state, excited_state \
            = expectation_calculator.calculate(qcs[0], best, qiskit_h2_hamiltonian), \
              expectation_calculator.calculate(qcs[1], best, qiskit_h2_hamiltonian)
        return ground_state, excited_state
    
    def solve_with_qlm(self):
        h2_hamiltonian_builder = HamiltonianBuilder(get_h2_hamiltonian_terms())
        qlm_hamiltonian = h2_hamiltonian_builder.get_qlm_observable()

        qc1, qc2 = self._build_qiskit_circuits()
        qcs = [qiskit_to_qlm(qc1), qiskit_to_qlm(qc2)]
        combined = []

        w = np.arange(4, 2, -1)
        # define the total iterations
        max_n_iter = 35
        # steps size
        alpha = 0.1
        # factor for average gradient
        beta1 = 0.9
        # factor for average squared gradient
        beta2 = 0.999
        
        adam_optimizer = AdamOptimizer(circuit_type=CircuitType.QISKIT)
        best, score, local_trace = \
            adam_optimizer.optimize(qcs = qcs, n_iter=max_n_iter,
                                    alpha=alpha, beta1=beta1, beta2=beta2,
                                    H=qlm_hamiltonian, w=w, num_qubits=4,
                                    eps=1e-8)
        print('Done!')
        print('f(%s) = %f' % (best, score))
        # here we are not returning energies
        energies = [local_trace[i][2] for i in range(len(local_trace))]
        best=local_trace[-1][1]

        expectation_calculator = ExpectationCalculator(circuit_type=CircuitType.QISKIT)
        ground_state, excited_state \
            = expectation_calculator.calculate(qcs[0], best, qlm_hamiltonian), \
              expectation_calculator.calculate(qcs[1], best, qlm_hamiltonian)
        return ground_state, excited_state
    
    def solve_with_cirq(self):
        h2_hamiltonian_builder = HamiltonianBuilder(get_h2_hamiltonian_terms())
        hamiltonian = h2_hamiltonian_builder.get_cirq_observable()

        qc1, qc2 = self._build_qiskit_circuits()
        qcs = [qlm_to_cirq(qiskit_to_qlm(qc1)), qlm_to_cirq(qiskit_to_qlm(qc2))]

        w = np.arange(4, 2, -1)
        # define the total iterations
        max_n_iter = 35
        # steps size
        alpha = 0.1
        # factor for average gradient
        beta1 = 0.9
        # factor for average squared gradient
        beta2 = 0.999
        
        adam_optimizer = AdamOptimizer(circuit_type=CircuitType.QISKIT)
        best, score, local_trace = \
            adam_optimizer.optimize(qcs = qcs, n_iter=max_n_iter,
                                    alpha=alpha, beta1=beta1, beta2=beta2,
                                    H=hamiltonian, w=w, num_qubits=4,
                                    eps=1e-8)
        print('Done!')
        print('f(%s) = %f' % (best, score))
        # here we are not returning energies
        energies = [local_trace[i][2] for i in range(len(local_trace))]
        best=local_trace[-1][1]

        expectation_calculator = ExpectationCalculator(circuit_type=CircuitType.QISKIT)
        ground_state, excited_state \
            = expectation_calculator.calculate(qcs[0], best, hamiltonian), \
              expectation_calculator.calculate(qcs[1], best, hamiltonian)
        return ground_state, excited_state
