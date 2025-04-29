import cirq
from qiskit.quantum_info.operators import Operator, Pauli
from qat.core import Observable, Term
from circuit_type.circuit import CircuitType
from hamiltonians.h_2 import get_h2_tfi_hamiltonian_terms
from qiskit.quantum_info import SparsePauliOp, PauliList

# =============================================================================
# The HamiltonianBuilder class
# =============================================================================

class TFIHamiltonianBuilder:
    def __init__(self, num_qubits, terms):
        """
        Parameters:
            num_qubits (int): Number of qubits (and the length of each Pauli string).
            terms (list): A list of terms. Each term can be either:
                          - a tuple: (coefficient, pauli_string) or
                          - a dict with keys 'coefficient' and 'pauli'
                          For example: (-0.2427, 'IIZI')
        """
        self.num_qubits = num_qubits
        self.terms = []
        for term in terms:
            if isinstance(term, dict):
                coeff = term.get('coefficient')
                pauli = term.get('pauli')
            elif isinstance(term, (tuple, list)):
                pauli, coeff = term
            else:
                raise ValueError("Each term must be a tuple or dict.")
            if len(pauli) != num_qubits:
                raise ValueError(f"Pauli string '{pauli}' length does not equal num_qubits={num_qubits}.")
            self.terms.append({'coefficient': coeff, 'pauli': pauli})


    def get_qiskit_hamiltonian(self):
        """
        Returns the Hamiltonian as a sum of Operator(Pauli) objects.
        """
        built_terms = []
        for term in self.terms:
            coeff = term['coefficient']
            pauli_str = term['pauli']
            built_terms.append((pauli_str, coeff))
        hamiltonian = SparsePauliOp.from_list(built_terms)
        return hamiltonian

    def get_qlm_observable(self)-> Observable:
        """
        Returns the Hamiltonian as an Observable object made of Term entries.
        """
        pauli_terms = []
        for term in self.terms:
            coeff = term['coefficient']
            pauli_str = term['pauli']
            # Here the qubit indices are assumed to be 0, 1, ..., num_qubits-1.
            qubit_indices = list(range(self.num_qubits))
            pauli_terms.append(Term(coeff, pauli_str, qubit_indices))
        # TODO mhh: implement different constant coeff
        observable = Observable(self.num_qubits, pauli_terms=pauli_terms, constant_coeff=0)
        return observable

    def get_cirq_observable(self):
        """
        Returns the Hamiltonian as a Cirq PauliSum.
        """
        # Create a list of Cirq qubits.
        qubits = [cirq.LineQubit(i) for i in range(self.num_qubits)]
        # Build a PauliSum from individual PauliStrings.
        pauli_sum = 0 # start with an empty sum
        for term in self.terms:
            coeff = term['coefficient']
            pauli_str = term['pauli']
            term_pauli_string = coeff
            for i, letter in enumerate(pauli_str):
                if letter == 'I':
                    continue
                elif letter == 'X':
                    term_pauli_string *= cirq.X(qubits[i])
                elif letter == 'Y':
                    term_pauli_string *= cirq.X(qubits[i])
                elif letter == 'Z':
                    term_pauli_string *= cirq.X(qubits[i])
                else:
                    raise ValueError(f"Unsupported Pauli letter: '{letter}' in term '{pauli_str}'")
            # Construct the PauliString with the given coefficient.

            if term_pauli_string != coeff:
                pauli_sum += term_pauli_string
        return pauli_sum

    def get_hamiltonian(self, circuit_type: CircuitType):
        if circuit_type == CircuitType.CIRQ:
            return self.get_cirq_observable()
        elif circuit_type == CircuitType.QLM:
            return self.get_qlm_observable()
        elif circuit_type == CircuitType.QISKIT:
            return self.get_qiskit_hamiltonian()

# =============================================================================
# Example usage
# =============================================================================

if __name__ == '__main__':
    # Define the list of Hamiltonian terms.
    # (Each term: (coefficient, pauli_string))
    h = 0.25
    hamiltonian_terms = [
        ("ZZIIII", -1),
        ("IZZIII", -1),
        ("IIZZII", -1),
        ("IIIZZI", -1),
        ("IIIIZZ", -1),
        ("XIIIII", -h),
        ("IXIIII", -h),
        ("IIXIII", -h),
        ("IIIXII", -h),
        ("IIIIXI", -h),
        ("IIIIIX", -h)]

