# =============================================================================
# The HamiltonianBuilder class
# =============================================================================

class HamiltonianBuilder:
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
                coeff, pauli = term
            else:
                raise ValueError("Each term must be a tuple or dict.")
            if len(pauli) != num_qubits:
                raise ValueError(f"Pauli string '{pauli}' length does not equal num_qubits={num_qubits}.")
            self.terms.append({'coefficient': coeff, 'pauli': pauli})

    def get_qiskit_hamiltonian(self):
        """
        Returns the Hamiltonian as a sum of Operator(Pauli) objects.
        """
        hamiltonian = None
        for term in self.terms:
            coeff = term['coefficient']
            pauli_str = term['pauli']
            # Create an operator from the Pauli string.
            term_op = coeff * Operator(Pauli(pauli_str))
            if hamiltonian is None:
                hamiltonian = term_op
            else:
                hamiltonian = hamiltonian + term_op
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
        pauli_sum = cirq.PauliSum()  # start with an empty sum
        for term in self.terms:
            coeff = term['coefficient']
            pauli_str = term['pauli']
            ps_dict = {}
            for i, letter in enumerate(pauli_str):
                if letter == 'I':
                    continue
                elif letter == 'X':
                    ps_dict[qubits[i]] = cirq.X
                elif letter == 'Y':
                    ps_dict[qubits[i]] = cirq.Y
                elif letter == 'Z':
                    ps_dict[qubits[i]] = cirq.Z
                else:
                    raise ValueError(f"Unsupported Pauli letter: '{letter}' in term '{pauli_str}'")
            # Construct the PauliString with the given coefficient.
            term_pauli_string = cirq.PauliString(ps_dict, coefficient=coeff)
            pauli_sum += term_pauli_string
        return pauli_sum

# =============================================================================
# Example usage
# =============================================================================

if __name__ == '__main__':
    # Define the list of Hamiltonian terms.
    # (Each term: (coefficient, pauli_string))
    terms = [
        (-0.24274280046588792, 'IIZI'),
        (-0.24274280046588792, 'IIIZ'),
        (-0.04207898539364302, 'IIII'),
        (0.17771287502681438, 'ZIII'),
        (0.1777128750268144,  'IZII'),
        (0.12293305045316086, 'ZIZI'),
        (0.12293305045316086, 'IZIZ'),
        (0.16768319431887935, 'ZIIZ'),
        (0.16768319431887935, 'IZZI'),
        (0.1705973836507714,  'ZZII'),
        (0.1762764072240811,  'IIZZ'),
        (-0.044750143865718496, 'YYXX'),
        (-0.044750143865718496, 'XXYY'),
        (0.044750143865718496, 'YXXY'),
        (0.044750143865718496, 'XYYX')
    ]
    
    num_qubits = 4
    
    # Instantiate the builder.
    ham_builder = HamiltonianBuilder(num_qubits, terms)
    
    # Get the first representation (Operator/Pauli objects).
    operator_hamiltonian = ham_builder.get_qiskit_hamiltonian()
    print("Operator representation:")
    print(operator_hamiltonian)
    print()
    
    # Get the second representation (Observable object).
    observable_hamiltonian = ham_builder.get_qlm_observable()
    print("Observable representation:")
    print(observable_hamiltonian)
    print()
    
    # Get the third representation (Cirq PauliSum).
    cirq_hamiltonian = ham_builder.get_cirq_hamiltonian()
    print("Cirq representation:")
    print(cirq_hamiltonian)
