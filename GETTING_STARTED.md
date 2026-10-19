# OpenVQA Getting Started Guide

## Overview

OpenVQA is a high-level Python framework for running **Variational Quantum Algorithms (VQA)** with an elegant, intuitive API. It abstracts away the complexity of quantum circuit construction, optimization, and result handling.

The framework supports:
- **VQE** (Variational Quantum Eigensolver) - Find the lowest energy of a Hamiltonian (built-in illustrative H2 and LiH models)
- **QAOA** (Quantum Approximate Optimization Algorithm) - Solve combinatorial problems
- **Two backends** - the myQLM simulator (circuits built with Qiskit) and PennyLane `default.qubit`
- **Pre-built components** - Hamiltonians, ansatze, and optimizers

## Installation

**Python version:** use Python 3.10, 3.11 or 3.12 (tested on 3.10 and 3.12).
Python 3.13 and newer do not work yet, because myQLM publishes no builds for them.

```bash
git clone https://github.com/open-vqa/open-vqa.git
cd open-vqa

# Tested versions of every library, plus OpenVQA itself (editable)
python -m pip install -r requirements.txt

# Check: prints the version and the folder it was imported from
python -c "import openvqa; print(openvqa.__version__, openvqa.__file__)"
```

Use the same Python for these commands as for your notebooks (select that
interpreter as the Jupyter kernel). `requirements.txt` pins the exact tested
versions; `python -m pip install -e ".[notebooks]"` installs OpenVQA with the
supported version ranges instead.

For a step-by-step guide (conda or venv) with checks that the installation works,
see [tutorial/setup_environment.ipynb](tutorial/setup_environment.ipynb).

## Basic Concepts

### 1. Engine (VQA_myQLM)
The **engine** manages quantum circuit execution and optimization.

```python
from openvqa import VQA_myQLM

engine = VQA_myQLM(
    n_qubits=4,              # Number of qubits
    backend="qat.simulators.qubit",  # myQLM simulator, or "pennylane"
    shots=0,                 # 0 = exact expectation values; >0 = sampled
    seed=42,                 # Reproducible runs (with shots=0)
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
engine = VQA_myQLM(n_qubits=2, shots=0, seed=42)
vqe = VQE(hamiltonian="H2", ansatz="ryrz", optimizer="cobyla", max_iterations=300)

# Run
results = engine.run(vqe)

# Results
print(f"Energy: {results.energy:.6f} (exact minimum -0.5)")
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
print(f"Expected cut value: {-results.energy:.4f}")  # energy = -cut
print(f"Optimal params: {results.optimal_parameters}")
```

### Pattern 3: Compare Optimizers

```python
from openvqa import VQA_myQLM
from openvqa.algorithms import VQE

optimizers = ["cobyla", "nelder-mead", "bfgs"]
energies = {}

for opt_name in optimizers:
    engine = VQA_myQLM(n_qubits=2, shots=0, seed=42)
    vqe = VQE(hamiltonian="H2", optimizer=opt_name, max_iterations=100)
    results = engine.run(vqe)
    energies[opt_name] = results.energy

# Print comparison
print("Optimizer Comparison:")
for opt, energy in sorted(energies.items(), key=lambda x: x[1]):
    print(f"  {opt:12s}: {energy:.8f}")
```

### Pattern 4: Vary Ansatz Depth

```python
from openvqa import VQA_myQLM
from openvqa.algorithms import VQE

depths = [1, 2, 3, 4]
results_by_depth = {}

for depth in depths:
    engine = VQA_myQLM(n_qubits=2, shots=0, seed=42)
    vqe = VQE(hamiltonian="H2", n_layers=depth)
    results = engine.run(vqe)
    results_by_depth[depth] = results.energy

print("Energy vs Circuit Depth:")
for depth, energy in results_by_depth.items():
    print(f"  Depth {depth}: {energy:.8f}")
```

## Understanding Results

The `VQAResults` object contains:

```
======================================================================
VQA OPTIMIZATION RESULTS
======================================================================
Status                       : ✓ Completed
Objective Value (energy)     : -0.5000000000
Optimal Parameters           : {'θ_RY_0_0': 1.5707953973406406, ...}
Number of Qubits             : 2
Circuit Depth                : 9
Optimizer                    : cobyla
Function Evaluations         : 197
Optimizer Converged          : True
Barren Plateau Check         : Not assessed
Shots                        : 0 (exact expectation)
Execution Time (s)           : 2.28
Backend Used                 : myQLM qat.clinalg.qpu.CLinalg
======================================================================
```

**Key Metrics:**
- **Objective Value**: VQE energy in model units; for QAOA, minus the expected cut
- **Optimizer Converged**: the optimizer's own success flag; `False` usually means `max_iterations` ran out. "Not reported" for Adam, which runs a fixed number of steps
- **Function Evaluations**: objective evaluations performed; every value is in `results.intermediate_energies`
- **Circuit Depth**: depth of the circuit actually executed ("Not measured" on the PennyLane path)
- **Barren Plateau Check**: not implemented yet, always "Not assessed"
- **Shots**: 0 means exact expectation values; otherwise energies are sampled estimates
- **Backend Used**: the simulator that executed the job
- **Time**: Wall-clock execution time

## VQE Deep Dive

### Supported Hamiltonians

Both built-in models are small illustrative Hamiltonians named after molecules.
They have no geometry, basis set or qubit mapping behind them, so energies are
in model units, not Hartree.

**H2-labelled model**
- Qubits: exactly 2
- H = 0.5 I + 0.5 Z0 + 0.5 Z1 + 0.25 X0X1 + 0.25 Y0Y1
- Exact eigenvalues: -0.5, 0, 1, 1.5 (minimum -0.5)
- Use: first experiments, checking an optimizer

**LiH-labelled model**
- Qubits: exactly 4
- 10 Pauli terms plus a constant (-0.198)
- Exact minimum: -1.983153
- Use: a harder landscape; optimizers can stop in a local minimum
  (e.g. -1.498 with `ryrz` + COBYLA, seed 42)

### Supported Ansatze

**RY-RZ** (`"ryrz"`)
- Structure: Hadamard → (RY and RZ layers with a CNOT ring)
- Parameters: 2 × layers × qubits
- Best for: General purpose

**Linear** (`"linear"`)
- Structure: RX layers with a CNOT chain, starting from |0…0⟩
- Parameters: 1 × layers × qubits
- Best for: Shallow circuits, fast execution

**UCCSD** (`"uccsd"`)
- Structure: X on the first half of the qubits, then RY + CNOT per qubit
- Parameters: number of qubits
- Note: illustrative only; not a fermionic UCCSD, and it does not reach the
  H2-model minimum

Qiskit circuits (`"RealAmplitudes"`, `"EfficientSU2"`, `"TwoLocal"`) and PennyLane
template names (`"StronglyEntanglingLayers"`, `"BasicEntanglerLayers"`,
`"RandomLayers"`) are also accepted; see `openvqa.options.AVAILABLE_OPTIONS`.

### Optimizer Selection

| Optimizer | Pros | Cons | Best for |
|-----------|------|------|----------|
| COBYLA | Robust, gradient-free, copes with `shots > 0` | Slower | General, first try |
| Nelder-Mead | Simplex-based | Needs many iterations | Small problems |
| BFGS | Fast convergence with `shots=0` | Numerical gradients stall on sampling noise | Exact simulations |
| Adam | Adaptive steps | Runs on PennyLane only, with `ryrz` or a PennyLane template; fixed number of steps | Gradient-based training experiments |

## QAOA Deep Dive

### MaxCut Problem

Given a graph, partition vertices into two sets to maximize edge cuts.

**Example:**
```python
# 4-node complete graph
engine = VQA_myQLM(n_qubits=4)
qaoa = QAOA(problem="maxcut", graph_size=4, p=1)
results = engine.run(qaoa)

# -results.energy = expected cut value (energy is minimised, H = -C)
```

### QAOA Depth (p)

- **p=1**: Fast, lower quality approximation
- **p=2**: Balanced, good approximation
- **p≥3**: Better solutions, slower execution

## Troubleshooting Guide

### Q: "ModuleNotFoundError: No module named 'openvqa'"

**Solution:** Install the package from the repository root, using the same
Python interpreter as your notebook kernel:
```bash
python -m pip install -e .
python -c "import openvqa; print(openvqa.__file__)"
```

### Q: "ImportError: qat not found"

**Solution:**
```bash
python -m pip install -r requirements.txt
```

### Q: Optimization not converging

**Try:**
1. Increase `max_iterations` (e.g., 500)
2. Change optimizer (COBYLA is most robust)
3. Reduce circuit depth (`n_layers=1`)
4. Use different `seed` value

### Q: Results look wrong or are NaN

**Check:**
1. Correct Hamiltonian? H2 needs exactly 2 qubits, LiH exactly 4
2. Backend installed? Run: `python -c "from qat.qpus import get_default_qpu; print(get_default_qpu())"`
3. Parameters passed? Use `verbose=True` to see details

## Advanced Topics

### Custom Hamiltonian

```python
from qat.core import Observable, Term
from openvqa import VQA_myQLM
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
from openvqa import VQA_myQLM
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

1. **Use seed for reproducibility** - Same seed = same results with `shots=0` (sampling with `shots > 0` is not seeded)
2. **Start with p=1 for QAOA** - Then increase if needed
3. **Use COBYLA first** - Most robust optimizer
4. **Reduce iterations if testing** - Use `max_iterations=20` for quick tests
5. **Check verbose output** - Set `verbose=True` to debug

## Example Notebooks

Run the included Jupyter notebooks:

```bash
# VQE example
jupyter notebook examples/vqe_demo.ipynb

# QAOA example
jupyter notebook examples/qaoa_demo.ipynb
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
2. Check `README.md`
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
