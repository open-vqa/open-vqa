# OpenVQA Getting Started Guide

## Overview

OpenVQA is a high-level Python framework for running **Variational Quantum Algorithms (VQA)** with an elegant, intuitive API. It abstracts away the complexity of quantum circuit construction, optimization, and result handling.

The framework supports:
- **VQE** (Variational Quantum Eigensolver) - Find ground state energies of molecules
- **QAOA** (Quantum Approximate Optimization Algorithm) - Solve combinatorial problems
- **Multiple backends** - QLM (myQLM), Qiskit, and more
- **Pre-built components** - Hamiltonians, ansatze, and optimizers

## Installation

OpenVQA is integrated into the `open_vqa` repository. Install dependencies:

```bash
# Navigate to workspace
cd d:\open_vqa

# Install required packages
pip install qiskit qiskit-algorithms qiskit-aer
pip install myqlm myqlm-interop
pip install numpy matplotlib networkx
```

## Basic Concepts

### 1. Engine (VQA_myQLM)
The **engine** manages quantum circuit execution and optimization.

```python
from openvqa import VQA_myQLM

engine = VQA_myQLM(
    n_qubits=4,              # Number of qubits
    backend="qat.simulators.qubit",  # Backend
    shots=1000,              # Measurement shots
    seed=42,                 # Reproducibility
    verbose=True             # Print progress
)
```

### 2. Algorithm (VQE or QAOA)
The **algorithm** defines what problem to solve.

**For VQE:**
```python
from openvqa.algorithms import VQE

algorithm = VQE(
    hamiltonian="H2",        # Target Hamiltonian
    ansatz="ryrz",           # Circuit template
    optimizer="cobyla",      # Optimizer
    max_iterations=100,      # Max optimization steps
    n_layers=2               # Circuit depth
)
```

**For QAOA:**
```python
from openvqa.algorithms import QAOA

algorithm = QAOA(
    problem="maxcut",        # Problem type
    graph_size=4,            # Problem size
    optimizer="cobyla",      # Optimizer
    max_iterations=50,       # Max steps
    p=2                      # QAOA depth
)
```

### 3. Execution and Results
Run the algorithm and get formatted results.

```python
# Execute
results = engine.run(algorithm)

# Print formatted table
print(results)

# Access individual values
print(f"Energy: {results.energy:.10f}")
print(f"Time: {results.wall_time:.2f}s")
print(f"Converged: {results.convergence}")
```

## Common Usage Patterns

### Pattern 1: Simple VQE for H2

```python
from openvqa import VQA_myQLM
from openvqa.algorithms import VQE

# Setup
engine = VQA_myQLM(n_qubits=2, seed=42)
vqe = VQE(hamiltonian="H2", ansatz="ryrz", optimizer="cobyla")

# Run
results = engine.run(vqe)

# Results
print(f"Ground state energy: {results.energy:.6f} Ha")
print(f"Converged: {results.convergence}")
```

### Pattern 2: QAOA for MaxCut

```python
from openvqa import VQA_myQLM
from openvqa.algorithms import QAOA

# Setup
engine = VQA_myQLM(n_qubits=5, seed=42)
qaoa = QAOA(problem="maxcut", graph_size=5, optimizer="cobyla", p=2)

# Run
results = engine.run(qaoa)

# Results
print(f"MaxCut value: {results.energy:.4f}")
print(f"Optimal params: {results.optimal_parameters}")
```

### Pattern 3: Compare Optimizers

```python
from openvqa import VQA_myQLM
from openvqa.algorithms import VQE

optimizers = ["cobyla", "nelder-mead", "bfgs"]
energies = {}

for opt_name in optimizers:
    engine = VQA_myQLM(n_qubits=2, seed=42)
    vqe = VQE(hamiltonian="H2", optimizer=opt_name, max_iterations=100)
    results = engine.run(vqe)
    energies[opt_name] = results.energy

# Print comparison
print("Optimizer Comparison:")
for opt, energy in sorted(energies.items(), key=lambda x: x[1]):
    print(f"  {opt:12s}: {energy:.8f} Ha")
```

### Pattern 4: Vary Ansatz Depth

```python
from openvqa import VQA_myQLM
from openvqa.algorithms import VQE

depths = [1, 2, 3, 4]
results_by_depth = {}

for depth in depths:
    engine = VQA_myQLM(n_qubits=2, seed=42)
    vqe = VQE(hamiltonian="H2", n_layers=depth)
    results = engine.run(vqe)
    results_by_depth[depth] = results.energy

print("Energy vs Circuit Depth:")
for depth, energy in results_by_depth.items():
    print(f"  Depth {depth}: {energy:.8f} Ha")
```

## Understanding Results

The `VQAResults` object contains:

```
VQA OPTIMIZATION RESULTS
===============================================================
Status                       : ✓ Success
Ground-State Energy (Ha)     : -1.1361285395 Ha
Optimal Parameters           : {'θ_RY_0_0': 4.213, ...}
Number of Qubits             : 2
Circuit Depth               : 16
Optimizer                    : cobyla
Function Evaluations        : 45
Converged                    : True
Barren Plateau Detected     : False
Execution Time (s)          : 2.34
Backend Used                : qat.simulators.qubit
===============================================================
```

**Key Metrics:**
- **Energy**: Ground state energy (VQE) or objective value (QAOA)
- **Convergence**: Did optimizer find stationary point?
- **Barren Plateau**: Does parameter landscape have flat regions?
- **Evaluations**: Number of times objective was computed
- **Time**: Wall-clock execution time

## VQE Deep Dive

### Supported Hamiltonians

**H2 (Hydrogen Molecule)**
- Qubits: 2+
- Ground State: -1.174 Ha
- Use: Testing, benchmarking

**LiH (Lithium Hydride)**
- Qubits: 4+
- Ground State: -7.88 Ha
- Use: Chemistry problems

### Supported Ansatze

**RY-RZ**
- Structure: Hadamard → (RY-RZ layers with CNOT chain)
- Parameters: 4 × layers × qubits
- Best for: General purpose

**Linear**
- Structure: Hadamard → (RX with CNOT chain)
- Parameters: 1 × layers × qubits
- Best for: Shallow circuits, fast execution

**UCCSD**
- Structure: Hartree-Fock reference + UCC gates
- Parameters: ~qubits
- Best for: Chemical accuracy

### Optimizer Selection

| Optimizer | Pros | Cons | Best for |
|-----------|------|------|----------|
| COBYLA | Robust, gradient-free | Slower | General, first try |
| Nelder-Mead | Simplex-based | Slow convergence | Small problems |
| BFGS | Fast convergence | Needs gradient | Medium problems |
| SLSQP | Constrained | Complex | Advanced users |

## QAOA Deep Dive

### MaxCut Problem

Given a graph, partition vertices into two sets to maximize edge cuts.

**Example:**
```python
# 4-node complete graph
engine = VQA_myQLM(n_qubits=4)
qaoa = QAOA(problem="maxcut", graph_size=4, p=1)
results = engine.run(qaoa)

# results.energy ≈ optimal/approximate cut value
```

### QAOA Depth (p)

- **p=1**: Fast, lower quality approximation
- **p=2**: Balanced, good approximation
- **p≥3**: Better solutions, slower execution

## Troubleshooting Guide

### Q: "ModuleNotFoundError: No module named 'openvqa'"

**Solution:**
```bash
# Make sure you're in the d:\open_vqa directory
cd d:\open_vqa

# Add to path in Python
import sys
sys.path.insert(0, 'd:\\open_vqa')
```

### Q: "ImportError: qat not found"

**Solution:**
```bash
pip install myqlm myqlm-interop
```

### Q: Optimization not converging

**Try:**
1. Increase `max_iterations` (e.g., 500)
2. Change optimizer (COBYLA is most robust)
3. Reduce circuit depth (`n_layers=1`)
4. Use different `seed` value

### Q: Results look wrong or are NaN

**Check:**
1. Correct Hamiltonian? H2 needs 2+ qubits
2. Backend installed? Run: `python -c "from qat.qpus import get_default_qpu; print(get_default_qpu())"`
3. Parameters passed? Use `verbose=True` to see details

## Advanced Topics

### Custom Hamiltonian

```python
from qat.core import Observable, Term
from openvqa.algorithms import VQE

class CustomVQE(VQE):
    def build_hamiltonian(self, n_qubits):
        # Build custom observable
        terms = [
            Term(0.5, [], []),       # Identity
            Term(0.25, ["Z"], [0]),  # Single Z
            Term(0.1, ["X", "X"], [0, 1]),  # XX coupling
        ]
        return Observable(n_qubits, pauli_terms=terms)

# Use it
engine = VQA_myQLM(n_qubits=2)
vqe = CustomVQE(ansatz="ryrz")
results = engine.run(vqe)
```

### Custom Ansatz

```python
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from openvqa.algorithms import VQE

class CustomVQE(VQE):
    def build_ansatz(self, n_qubits):
        qc = QuantumCircuit(n_qubits)
        params = []
        
        # Your custom circuit
        for q in range(n_qubits):
            p = Parameter(f'θ_{q}')
            qc.ry(p, q)
            params.append(p)
        
        for q in range(n_qubits-1):
            qc.cx(q, q+1)
        
        return qc, params

# Use it
engine = VQA_myQLM(n_qubits=2)
vqe = CustomVQE(hamiltonian="H2")
results = engine.run(vqe)
```

## Performance Tips

1. **Use seed for reproducibility** - Same seed = same results
2. **Start with p=1 for QAOA** - Then increase if needed
3. **Use COBYLA first** - Most robust optimizer
4. **Reduce iterations if testing** - Use `max_iterations=20` for quick tests
5. **Check verbose output** - Set `verbose=True` to debug

## Example Notebooks

Run the included Jupyter notebooks:

```bash
# VQE example
jupyter notebook openvqa/examples/vqe_demo.ipynb

# QAOA example
jupyter notebook openvqa/examples/qaoa_demo.ipynb
```

## Next Steps

1. ✓ Run VQE example
2. ✓ Run QAOA example
3. ✓ Try with different optimizers
4. ✓ Try different ansatze
5. ✓ Create custom Hamiltonian
6. ✓ Extend with new algorithms

## Support

For issues or questions:
1. Check this guide
2. Check `openvqa/README.md`
3. See docstrings: `help(VQA_myQLM)`, `help(VQE)`
4. Review examples in notebooks

## References

- **VQE**: [Nature Computational Science 1, 305-309 (2021)](https://www.nature.com/articles/s43588-021-00066-3)
- **QAOA**: [arXiv:1411.4028](https://arxiv.org/abs/1411.4028)
- **Ansatz Design**: [arXiv:2106.08850](https://arxiv.org/abs/2106.08850)
- **Barren Plateaus**: [Nature Communications 12, 1771 (2021)](https://www.nature.com/articles/s41467-021-21728-w)

---

**Happy Quantum Computing! 🚀**

*Questions? Check the examples or consult the docstrings.*
