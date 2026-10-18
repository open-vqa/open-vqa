# OpenVQA: High-Level Quantum Variational Algorithm Framework

A unified, easy-to-use interface for running Variational Quantum Algorithms (VQA) including **VQE** and **QAOA** on multiple quantum backends.

## Features

✅ **Unified API** - Simple, intuitive interface for VQA algorithms  
✅ **Two Backends** - the myQLM simulator (circuits built with Qiskit, including Qiskit versions of PennyLane templates) and PennyLane's `default.qubit`  
✅ **Pre-built Components** - Illustrative Hamiltonians, ansatze, and optimizers  
✅ **Automatic Optimization** - Seamless classical optimization  
✅ **Scientific Output** - Result tables that report only what was measured  
✅ **Clear Errors** - Invalid settings raise immediately with a message  

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

## Quick Start

### Example 1: VQE on the H2-labelled model

```python
from openvqa import VQA_myQLM
from openvqa.algorithms import VQE

# Initialize engine (shots=0: exact expectation values; e.g. shots=1000 adds sampling noise)
engine = VQA_myQLM(n_qubits=2, shots=0, seed=42)

# Create VQE algorithm
algorithm = VQE(
    hamiltonian="H2",
    ansatz="ryrz",
    optimizer="cobyla",
    max_iterations=300,
    n_layers=2
)

# Run optimization
results = engine.run(algorithm)

# Display results
print(results)
print(f"Energy: {results.energy:.10f} (exact minimum: -0.5)")
```

**Output** (from a real run; parameters shortened, `Execution Time` varies):
```
======================================================================
VQA OPTIMIZATION RESULTS
======================================================================
Status                       : ✓ Completed
Objective Value (energy)     : -0.5000000000
Optimal Parameters           : {'θ_RY_0_0': 1.5707953973406406, 'θ_RY_0_1': 2.2056951336360924, ...}
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
Energy: -0.5000000000 (exact minimum: -0.5)
```

"H2" here is a small illustrative model, not the hydrogen molecule; see
[Supported Hamiltonians](#supported-hamiltonians). With `shots > 0` the energy is
a sampled estimate: it changes between runs and can even fall slightly below the
exact minimum.

### Example 2: QAOA for MaxCut

```python
from openvqa import VQA_myQLM
from openvqa.algorithms import QAOA

# Initialize engine
engine = VQA_myQLM(n_qubits=4, shots=1000, seed=42)

# Create QAOA algorithm
algorithm = QAOA(
    problem="maxcut",
    graph_size=4,
    optimizer="cobyla",
    max_iterations=50,
    p=2
)

# Run optimization
results = engine.run(algorithm)

# Display results
print(results)
print(f"Expected cut value: {-results.energy:.6f}")  # energy = -cut
```

## Architecture

### Core Components

```
openvqa/
├── __init__.py           # Package exports
├── engine.py             # VQA_myQLM engine and VQAResults
├── options.py            # AVAILABLE_OPTIONS (supported names)
├── algorithms/
│   ├── __init__.py
│   ├── vqe.py            # VQE algorithm
│   └── qaoa.py           # QAOA algorithm
├── interop/
│   └── pennylane_to_qlm.py  # Qiskit versions of PennyLane templates
├── mathematical_tools/   # Adam optimizer, optimizer names
├── examples/             # Interactive demo notebooks
└── tutorial/             # Tutorial notebooks
```

### Class Hierarchy

#### VQA_myQLM (Engine)
Main execution engine for managing quantum simulations and optimization.

**Parameters:**
- `n_qubits` (int): Number of qubits (2 for H2, 4 for LiH)
- `backend` (str): `"qat.simulators.qubit"` (default, myQLM simulator) or `"pennylane"` (PennyLane `default.qubit`); any other name raises
- `shots` (int): Measurement shots; `0` gives exact expectation values (default: 1000, applied on the myQLM path)
- `seed` (int): Seeds NumPy, which makes `shots=0` runs reproducible (sampling with `shots > 0` is not seeded)
- `verbose` (bool): Enable verbose output
- `raise_on_error` (bool): Re-raise errors from `run()` (default: True); if False, a failed `VQAResults` with `error` set is returned

**Methods:**
- `run(algorithm)` → VQAResults
- `create_optimizer(name, max_iterations)` → Optimizer plugin

#### VQE (Algorithm)
Variational Quantum Eigensolver for ground state energy computation.

**Parameters:**
- `hamiltonian` (str): "H2" (2 qubits) or "LiH" (4 qubits), both illustrative models (see below)
- `ansatz` (str): "ryrz", "linear", "uccsd", a Qiskit circuit ("RealAmplitudes", "EfficientSU2", "TwoLocal") or a PennyLane template name ("StronglyEntanglingLayers", "BasicEntanglerLayers", "RandomLayers")
- `optimizer` (str): "cobyla", "nelder-mead", "bfgs", or "adam"
- `max_iterations` (int): Maximum optimizer iterations
- `n_layers` (int): Number of ansatz layers
- `feature_map`, `embedding` (str): Optional data-encoding circuits; VQE treats their inputs as extra trainable parameters
- `optimizer_options` (dict): e.g. `{"learning_rate": 0.05}` for Adam

**Methods:**
- `build_ansatz(n_qubits)` → (QuantumCircuit, params)
- `build_hamiltonian(n_qubits)` → Observable
- `execute(engine, qpu)` → VQAResults

#### QAOA (Algorithm)
Quantum Approximate Optimization Algorithm for combinatorial problems.

**Parameters:**
- `problem` (str): "maxcut" (currently the only problem)
- `graph_size` (int): Number of nodes (= qubits)
- `optimizer` (str): "cobyla", "nelder-mead", "bfgs", or "adam"
- `max_iterations` (int): Maximum optimizer iterations
- `p` (int): QAOA depth (layers)

Assign `algorithm.graph` (a networkx graph with nodes `0..graph_size-1`) before
running to use a fixed graph; otherwise a random weighted complete graph is
generated. `results.energy` is minus the expected cut value.

**Methods:**
- `build_circuit()` → (QuantumCircuit, params)
- `build_observable()` → Observable
- `execute(engine, qpu)` → VQAResults

#### VQAResults (Container)
Stores and formats optimization results.

**Attributes:**
- `energy` (float): Final objective value (VQE energy; for QAOA, minus the expected cut)
- `optimal_parameters` (dict/array): Optimized parameters
- `n_qubits` (int): Number of qubits used
- `circuit_depth` (int or None): Depth of the circuit actually executed (None where not measured)
- `n_evaluations` (int or None): Objective evaluations performed
- `intermediate_energies` (list): Objective value of every evaluation
- `convergence` (bool or None): The optimizer's own success flag (None for Adam, which runs a fixed number of steps)
- `barren_plateau_detected` (None): Not assessed yet
- `shots` (int): Shots used (0 = exact expectation values)
- `wall_time` (float): Execution time in seconds
- `backend` (str): Simulator that executed the job
- `optimizer_name` (str): Optimizer name
- `success` (bool): Whether the run completed
- `error` (str or None): Error message when `raise_on_error=False` and the run failed

## Supported Hamiltonians

| Hamiltonian | Qubits | Definition | Exact minimum |
|-------------|--------|------------|---------------|
| **H2** | 2 | 0.5 I + 0.5 Z0 + 0.5 Z1 + 0.25 X0X1 + 0.25 Y0Y1 | -0.5 |
| **LiH** | 4 | 10 Pauli terms plus a constant (-0.198) | -1.983153 |

Both are small illustrative models named after molecules, not molecular
Hamiltonians: there is no geometry, basis set or qubit mapping behind them, so
their energies are in model units, not Hartree. Compare VQE results with the
exact minimum above.

*Extensible*: Add custom Hamiltonians by extending the VQE class.

## Supported Ansatze

| Ansatz | Description | Parameters | Notes |
|--------|-------------|-----------|-------|
| **ryrz** | Hadamards, then per layer RY and RZ on every qubit and a CNOT ring | 2LN | Default |
| **linear** | Per layer RX on every qubit and a CNOT chain | LN | Shallow circuits |
| **uccsd** | X on the first N/2 qubits, then RY + CNOT per qubit | N | Illustrative only: not a fermionic UCCSD, and it does not reach the H2-model minimum |
| **RealAmplitudes**, **EfficientSU2**, **TwoLocal** | Qiskit circuit library (`TwoLocal` uses RY + CX) | see Qiskit | myQLM path only |
| **StronglyEntanglingLayers**, **BasicEntanglerLayers**, **RandomLayers** | PennyLane template names | 3LN, LN, 3LN | On myQLM these are hand-written Qiskit circuits in the same spirit, not the exact PennyLane circuits; with `backend="pennylane"` or `optimizer="adam"` the real PennyLane templates run |

*Parameters:* L = layers, N = qubits. With `optimizer="adam"` or `backend="pennylane"`,
only `ryrz` and the PennyLane templates are available.

## Supported Optimizers

| Optimizer | Runs on | Features |
|-----------|---------|----------|
| **COBYLA** | SciPy, via myQLM's `ScipyMinimizePlugin` | Gradient-free and robust; copes with `shots > 0` |
| **Nelder-Mead** | SciPy, via myQLM | Simplex-based; needs many iterations |
| **BFGS** | SciPy, via myQLM | Quasi-Newton with numerical gradients; fast with `shots=0`, stalls on sampling noise |
| **Adam** | OpenVQA `AdamOptimizer` on PennyLane `default.qubit` | Finite-difference gradients, fixed number of steps; `ryrz` and PennyLane templates only |

## Advanced Usage

### Custom Hamiltonian

```python
from qat.core import Observable, Term

class CustomVQE(VQE):
    def build_hamiltonian(self, n_qubits):
        # Define custom Pauli terms
        terms = [
            Term(0.5, [], []),  # Identity term
            Term(0.25, ["Z"], [0]),
            Term(0.25, ["X", "X"], [0, 1]),
        ]
        return Observable(n_qubits, pauli_terms=terms)

# Use it
engine = VQA_myQLM(n_qubits=2)
algorithm = CustomVQE(ansatz="ryrz")
results = engine.run(algorithm)
```

### Multiple Optimizers Comparison

```python
optimizers = ["cobyla", "nelder-mead", "bfgs"]
results_dict = {}

for opt in optimizers:
    engine = VQA_myQLM(n_qubits=2, shots=0, seed=42)
    algorithm = VQE(hamiltonian="H2", optimizer=opt, max_iterations=100)
    results = engine.run(algorithm)
    results_dict[opt] = results.energy

# Compare energies
for opt, energy in results_dict.items():
    print(f"{opt:15s}: {energy:.10f}")
```

## Performance

No benchmark suite is published yet. The examples above take a few seconds on
a laptop; exact simulation cost grows exponentially with the number of qubits.
Every result reports its own `wall_time`, `n_evaluations` and `circuit_depth`.

## Troubleshooting

### Installation fails on Python 3.13 or newer

**Error:** `No matching distribution found for qat-core` or
`requires a different Python`

**Solution:** myQLM has no builds for Python 3.13+ yet. Install Python 3.12
(or 3.10/3.11), create a fresh environment with it, and install again.

### Import Errors

**Error:** `ModuleNotFoundError: No module named 'openvqa'`

**Solution:** Install the package from the repository root with the same Python
your notebook kernel uses: `python -m pip install -e .`

**Error:** `ImportError: No module named 'qat'`

**Solution:** Install myQLM
```bash
python -m pip install -r requirements.txt
```

### Backend Not Found

**Error:** `Backend qat.simulators.qubit not found`

**Solution:** Ensure QLM is properly installed
```bash
python -c "from qat.qpus import get_default_qpu; print(get_default_qpu())"
```

### Optimization Not Converging

**Solution:** Try these steps:
1. Check `results.convergence`: it is the optimizer's own success flag, and
   `False` usually means `max_iterations` ran out (see `results.n_evaluations`)
2. Increase `max_iterations`
3. Use `shots=0` while developing: sampling noise can stall BFGS and Nelder-Mead
4. Change `optimizer` (try COBYLA first)
5. Reduce `n_layers`
6. Try different `seed` values

## API Reference

See docstrings in source files:
- `openvqa.engine.VQA_myQLM`
- `openvqa.algorithms.vqe.VQE`
- `openvqa.algorithms.qaoa.QAOA`
- `openvqa.engine.VQAResults`

## Examples

Run the demo notebooks:

```bash
jupyter notebook examples/vqe_demo.ipynb
jupyter notebook examples/qaoa_demo.ipynb
```

## Contributing

To extend OpenVQA:

1. **Add new Hamiltonian**: Extend `VQE.build_hamiltonian()`
2. **Add new Ansatz**: Extend `VQE.build_ansatz()`
3. **Add new Algorithm**: Create new class inheriting from algorithm template
4. **Add new Backend**: Extend `VQA_myQLM` with backend support

## Citation

```bibtex
@software{openvqa2025,
  title={OpenVQA: High-Level Quantum Variational Algorithm Framework},
  author={OpenVQA Contributors},
  year={2025},
  url={https://github.com/open-vqa/open-vqa}
}
```

## References

- [Variational Quantum Eigensolver (VQE)](https://arxiv.org/abs/1509.04279)
- [Quantum Approximate Optimization (QAOA)](https://arxiv.org/abs/1411.4028)
- [QLM Documentation](https://myqlm.github.io/)
- [Qiskit Documentation](https://qiskit.org/)

## License

MIT License - See LICENSE.md

---

**Version:** 0.1.0  
**Last Updated:** October 2026  
**Status:** Alpha: APIs and results may still change
