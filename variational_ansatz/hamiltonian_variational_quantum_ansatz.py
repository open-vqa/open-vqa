# 
# TODO mhh: sometimes imports are slow, we should do lazy imports when we can and it make sense
# aka when imports takes more than 1 second and it will be used in limited scenarios.
import numpy as np
import random
import matplotlib.pyplot as plt
import cirq
from qat.lang import *
from qat import *
from qat.interop.qiskit import qiskit_to_qlm
from qiskit.compiler import transpile
from qiskit.primitives import Estimator
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SPSA
from qat.interop.qiskit import qiskit_to_qlm
from qat.interop.cirq import qlm_to_cirq
from qat.qpus import get_default_qpu
from qiskit.circuit.library import EvolvedOperatorAnsatz
from qat.plugins import ScipyMinimizePlugin
from hamiltonians.tfi_hamiltonian_builder import TFIHamiltonianBuilder
from hamiltonians.h_2 import get_h2_tfi_hamiltonian_terms
from mathematical_tools.scipy_minimize_plugin_method import ScipyMinimizePluginMethod

class HamiltonianVariationalAnsatz:
    def __init__(self):
        """
        Initializes the variational ansatz algorithm.
        
        """
        self.counts = []
        self.values = []

    def _store_intermediate_result(self, eval_count, parameters, mean, std):
        self.counts.append(eval_count)
        self.values.append(mean)

    def _build_qiskit_circuit(self):
        hamiltonian = TFIHamiltonianBuilder(6, get_h2_tfi_hamiltonian_terms()).get_qiskit_hamiltonian()
        qc = EvolvedOperatorAnsatz(operators=hamiltonian.group_commuting(), reps=2)
        basis_gates = ['rx', 'u2', 'cx', 'ry', 'h', 'rz', 'x', 'y', 'z']
        qc_trans = transpile(qc, basis_gates=basis_gates)
        return qc_trans, hamiltonian

    def compute_with_qiskit(self):
        qc_trans, hamiltonian = self._build_qiskit_circuit()
        estimator = Estimator()
        optimizer = SPSA(maxiter=100)
        vqe = VQE(estimator, qc_trans, optimizer, callback=self._store_intermediate_result)
        result = vqe.compute_minimum_eigenvalue(hamiltonian)
        return result

    def build_with_qlm(self, method:ScipyMinimizePluginMethod):
        """
        Performs variational optimization using QLM.
        The Qiskit ansatz is converted to a QLM circuit and combined with
        the Hamiltonian observable from the HamiltonianBuilder.
        """
        qlm_circuit, _ = qiskit_to_qlm(self._build_qiskit_circuit())
        hamiltonian = TFIHamiltonianBuilder(6, get_h2_tfi_hamiltonian_terms()).get_qlm_observable()
    
        job = qlm_circuit.to_job(job_type="OBS",
                        observable=hamiltonian,
                        nbshots=0)

        theta_0 = np.random.random(size=4)

        linalg_qpu = get_default_qpu()
        optimizer_scipy = ScipyMinimizePlugin(method=method,
                                            tol=1e-6,
                                            options={"maxiter": 200},
                                            x0=theta_0)
        qpu = optimizer_scipy | linalg_qpu
        result = qpu.submit(job)
        return result.value
        

    def simulate_with_cirq(self, num_samples=1000):
        """
        Simulates the variational ansatz using Cirq by sampling random parameters.
        The QLM circuit is first converted to a Cirq circuit. Then, using a Cirq simulator,
        the expectation value of the corresponding observable is computed.
        """
        cirq_circuit, _ = qlm_to_cirq(qiskit_to_qlm(self._build_qiskit_circuit()))
        observable = TFIHamiltonianBuilder(6, get_h2_tfi_hamiltonian_terms()).get_cirq_observable()
        energies = []
        random_params = []
        simulator = cirq.Simulator()
        for _ in range(1000):
            # Generate a list of 4 random values between -1 and 1
            x = [2 * random.random() - 1 for _ in range(4)]
            resolver = cirq.ParamResolver({'t[0]': x[0], 't[1]': x[1], 't[2]': x[2], 't[3]': x[3]})
            resolved_circuit = cirq.resolve_parameters(cirq_circuit, resolver)

            ev_list = simulator.simulate_expectation_values(
                cirq.Circuit(resolved_circuit), observables=[observable], permit_terminal_measurements=True
            )
            energies.append(ev_list[0].real)
            random_params.append(x)
        return min(energies)
