from qiskit import *
import numpy as np
#from qiskit.tools.jupyter import *
from qiskit.visualization import *
from qiskit.opflow import X,Y,Z,I,CX
from qiskit.circuit.library import QFT
# ------------------------------------------------
# The QPEAlgorithm Class
# ------------------------------------------------
class QPEAlgorithm:
    def __init__(self, evolution_gate, w_qubits = 3, s_qubits = 2,
                 trotter_number=1, t=1, initial_state=np.array([1, 0, 0, 0])):
        """
        Initializes the QPE algorithm.

        Args:
            w_qubits (int): Number of witness (ancilla) qubits.
            s_qubits (int): Number of system qubits.
            evolution_operator (np.ndarray): The unitary evolution operator (matrix)
                for the system (e.g. U = exp(iHt)).
            trotter_number (int): Number of trotterization steps.
            t (float): Time parameter (delta t) used in the evolution.
            initial_state (np.ndarray or None): Initial state vector for the system
                qubits. Its length should be 2^(s_qubits). If None, no initialization
                is performed.
        """
        self.w_qubits = w_qubits
        self.s_qubits = s_qubits
        self.trotter_number = trotter_number
        self.t = t
        self.initial_state = initial_state
        # Build the controlled evolution gate.
        # First, construct a minimal circuit that implements the operator.

    @staticmethod
    def qiskit_qc_from_operator(operator):
        """
        Constructs a QuantumCircuit that applies a given unitary operator.
        
        Args:
            operator (np.ndarray): A unitary matrix.
        
        Returns:
            QuantumCircuit: A circuit implementing the operator on the minimal number of qubits.
        """
        qubit_list = list(range(int(np.log(len(operator))
                                /np.log(2)))) #extract the no. of qubits req. in circuit 
        qc = QuantumCircuit(len(qubit_list))
        qc.unitary(operator,qubit_list)
        return qc

    def build_qpe_circuit(self, evolution_gate):
        """
        Constructs the full QPE circuit using trotterization and an inverse QFT.
        
        Returns:
            QuantumCircuit: The constructed QPE circuit.
        """
        total_qubits = self.w_qubits + self.s_qubits
        # Create a circuit with total_qubits and classical bits for witness measurement.
        qpe_circ = QuantumCircuit(total_qubits, self.w_qubits)
        
        # Optionally initialize the system register.
        if self.initial_state is not None:
            qpe_circ.initialize(self.initial_state, list(range(self.w_qubits, total_qubits)))
        
        # Apply Hadamard gates to all witness qubits.
        for i in range(self.w_qubits):
            qpe_circ.h(i)
        
        # --- Trotterization ---
        # For each trotter step, apply controlled-U^(2^k) operations.
        for _ in range(self.trotter_number):
            # TODO mhh: ask hussein why repetitions is set to 1 all the times in notebook
            repetitions = 1
            for counting_qubit in range(self.w_qubits):
                for _i in range(repetitions):
                    # The controlled gate acts on the counting (witness) qubit
                    # and all system qubits.
                    qubit_list = [counting_qubit] + list(range(self.w_qubits, total_qubits))
                    qpe_circ.append(evolution_gate, qubit_list)
                repetitions *= 2  # Increase the power for the next witness qubit.
        
        # --- Inverse QFT ---
        # Create an inverse QFT on the witness qubits.
        inv_qft = QFT(self.w_qubits, do_swaps=True, inverse=True)
        qpe_circ = qpe_circ.compose(inv_qft, list(range(self.w_qubits)))
        
        return qpe_circ

    def add_measurements(self, circuit: QuantumCircuit):
        """
        Adds measurement to the witness (ancilla) qubits.
        
        Args:
            circuit (QuantumCircuit): The circuit to modify.
        
        Returns:
            QuantumCircuit: The circuit with measurements added.
        """
        circuit.measure(list(range(self.w_qubits)), list(range(self.w_qubits)))
        return circuit

    def plot_to_eigenval(self, count,t,n=1):
        w_qubits = len(list(count.keys())[0])
        list_ = []
        #if time step is zero then phase cannot be determined hence return a default 0
        if t == 0: 
            return 0
        #will choose n maximum count values
        lists = sorted(count, key=count.get, reverse=True)[:n] 
        for j in range(len(lists)):
            #convert those binary keys into decimal
            lists[j] =  int(str(lists[j]), 2) 
        for j in range(len(lists)):
            #for positive eigenvalues choose this equation
            list_.append((2*pi*(2**w_qubits - lists[j]))/((2**w_qubits)*t))
            # if it is negative then choose this value
            lists[j] = -2*pi*(lists[j])/((2**w_qubits)*t) 
        return lists,list_

    def run_qiskit(self, shots=8192):
        """
        Builds the QPE circuit (with measurements) and runs it on Qiskit's qasm simulator.
        
        Args:
            shots (int): Number of shots for the simulation.
        
        Returns:
            dict: The counts dictionary from the simulation.
        """

        pi = np.pi
        sin = np.sin
        cos = np.cos
        exp = np.exp

        # taking no. of trotter step = 1, as all the operators in H commutes in this case.
        trotter_number = 2
        t = 1 # delta t, time
        H = (0.33*t/trotter_number*Z^I)+(3.24*t/trotter_number*I^Z)+(1.17*t/trotter_number*Z^Z)
        U = H.exp_i() #obtaining the evolution operator for H with t = 1
        U = U.to_matrix()
        U_gate = self.qiskit_qc_from_operator(U).to_gate(label = 'U').control(1)

        circuit = self.build_qpe_circuit(U_gate)
        instructions = self.add_measurements(circuit)

        simulator = Aer.get_backend('qasm_simulator')
        result = execute(circuit, backend=simulator, shots=shots).result()
        counts = result.get_counts(circuit)
        # TODO mhh: figure out hussein intensions here
        return counts

    def transpile_circuit(self, basis_gates=['rx', 'u2', 'cx', 'ry', 'h', 'rz', 'P']):
        """
        Transpiles the QPE circuit to a given basis.
        
        Args:
            basis_gates (list): List of basis gates.
        
        Returns:
            QuantumCircuit: The transpiled circuit.
        """
        circuit = self.build_qpe_circuit()
        return transpile(circuit, basis_gates=basis_gates)

    def to_qlm(self):
        """
        Converts the transpiled QPE circuit to a QLM circuit.
        
        Returns:
            A QLM circuit.
        """
        circuit_trans = self.transpile_circuit()
        return qiskit_to_qlm(circuit_trans)

    def run_qlm(self, shots=100):
        """
        Submits the QLM version of the circuit to the default QLM QPU.
        
        Args:
            shots (int): Number of shots.
        
        Returns:
            The QLM job result.
        """
        qlm_circuit = self.to_qlm()
        job = qlm_circuit.to_job(nbshots=shots)
        from qat.qpus import get_default_qpu
        result = get_default_qpu().submit(job)
        return result

    def to_cirq(self):
        """
        Converts the QLM circuit to a Cirq circuit.
        
        Returns:
            A Cirq circuit.
        """
        qlm_circuit = self.to_qlm()
        return qlm_to_cirq(qlm_circuit)

    def run_cirq(self, repetitions=1):
        """
        Runs the QPE circuit using Cirq's simulator.
        
        Args:
            repetitions (int): Number of repetitions.
        
        Returns:
            The Cirq simulation result.
        """
        cirq_circuit = self.to_cirq()
        simulator = cirq.Simulator()
        # Append measurement on witness qubits (assumed to be the first w_qubits).
        witness_qubits = sorted(list(cirq_circuit.all_qubits()), key=lambda q: q.x if hasattr(q, 'x') else 0)
        cirq_circuit.append(cirq.measure(*witness_qubits, key='result'))
        result = simulator.run(cirq_circuit, repetitions=repetitions)
        return result
 