# Optimizer Integration in OpenVQA

## Overview

The OpenVQA package now supports multiple classical optimizers for VQE and QAOA algorithms, mirroring the functionality from the `open-vqa-mhh-add_various_notebooks` repository.

## Supported Optimizers

### 1. **COBYLA** (Constrained Optimization BY Linear Approximation)
- **Type**: Derivative-free
- **Best for**: Constrained optimization, noisy objectives
- **Default**: Yes
- **Usage**: `optimizer="cobyla"`

### 2. **Nelder-Mead**
- **Type**: Derivative-free, Simplex-based
- **Best for**: Low-dimensional problems, smooth objectives
- **Usage**: `optimizer="nelder-mead"`

### 3. **BFGS** (Broyden-Fletcher-Goldfarb-Shanno)
- **Type**: Quasi-Newton (gradient-based)
- **Best for**: Smooth objectives with accessible gradients
- **Usage**: `optimizer="bfgs"`

### 4. **Adam** (Adaptive Moment Estimation)
- **Type**: Gradient-based with adaptive learning rates
- **Best for**: Large-scale optimization, noisy gradients
- **Usage**: `optimizer="adam"`
- **Options**: Can specify `learning_rate` in `optimizer_options`

## New Folder Structure

```
openvqa/
├── mathematical_tools/
│   ├── __init__.py
│   ├── scipy_minimize_plugin_method.py  # Scipy optimizer enum
│   └── adam_optimizer.py                # Adam optimizer implementation
├── algorithms/
│   ├── vqe.py                           # Updated with optimizer support
│   └── qaoa.py                          # Updated with optimizer support
└── examples/
    ├── vqe_demo.ipynb                   # Updated with optimizer selection
    └── qaoa_demo.ipynb                  # Updated with optimizer selection
```

## Usage Examples

### VQE with Different Optimizers

```python
from openvqa import VQA_myQLM
from openvqa.algorithms import VQE

# Initialize engine
engine = VQA_myQLM(n_qubits=2, backend='qat.simulators.qubit')

# Example 1: COBYLA (default)
vqe_cobyla = VQE(
    hamiltonian="H2",
    ansatz="RealAmplitudes",
    optimizer="cobyla",
    max_iterations=50
)

# Example 2: Nelder-Mead
vqe_nelder = VQE(
    hamiltonian="H2",
    ansatz="RealAmplitudes",
    optimizer="nelder-mead",
    max_iterations=100
)

# Example 3: BFGS
vqe_bfgs = VQE(
    hamiltonian="H2",
    ansatz="RealAmplitudes",
    optimizer="bfgs",
    max_iterations=75
)

# Example 4: Adam with custom learning rate
vqe_adam = VQE(
    hamiltonian="H2",
    ansatz="RealAmplitudes",
    optimizer="adam",
    max_iterations=200,
    optimizer_options={'learning_rate': 0.01}
)

# Run optimization
results = engine.run(vqe_cobyla)
print(f"Energy: {results.energy:.10f}")
```

### QAOA with Different Optimizers

```python
from openvqa import VQA_myQLM
from openvqa.algorithms import QAOA

# Initialize engine
engine = VQA_myQLM(n_qubits=4, backend='qat.simulators.qubit')

# Example 1: COBYLA
qaoa_cobyla = QAOA(
    problem="maxcut",
    graph_size=4,
    optimizer="cobyla",
    p=2,
    max_iterations=50
)

# Example 2: BFGS
qaoa_bfgs = QAOA(
    problem="maxcut",
    graph_size=4,
    optimizer="bfgs",
    p=2,
    max_iterations=100
)

# Example 3: Adam
qaoa_adam = QAOA(
    problem="maxcut",
    graph_size=4,
    optimizer="adam",
    p=2,
    max_iterations=150,
    optimizer_options={'learning_rate': 0.02}
)

# Run optimization
results = engine.run(qaoa_cobyla)
print(f"MaxCut Value: {results.energy:.6f}")
```

## Interactive Notebook Features

Both `vqe_demo.ipynb` and `qaoa_demo.ipynb` now include:

1. **Interactive Optimizer Selection**: Choose from 4 optimizers at runtime
2. **Optimizer-Specific Options**: Configure learning rate for Adam
3. **Clear Documentation**: Each optimizer includes a description
4. **Validation**: Automatic validation of optimizer choices

### Notebook Workflow

1. **Backend Selection**: Choose QLM, Qiskit, or PennyLane
2. **Template Selection**: Choose feature maps and ansatz
3. **Optimizer Selection**: Choose from COBYLA, Nelder-Mead, BFGS, or Adam
4. **Configuration**: Set optimizer-specific parameters (e.g., learning rate)
5. **Execution**: Run VQE/QAOA with selected configuration
6. **Visualization**: View convergence plots and results

## API Changes

### VQE Class

```python
VQE(
    hamiltonian: str = "H2",
    ansatz: str = "ryrz",
    embedding: str = None,
    feature_map: str = None,
    optimizer: str = "cobyla",           # NEW: Now supports 4 optimizers
    max_iterations: int = 100,
    n_layers: int = 2,
    optimizer_options: dict = None,      # NEW: Optimizer-specific options
)
```

### QAOA Class

```python
QAOA(
    problem: str = "maxcut",
    graph_size: int = 4,
    optimizer: str = "cobyla",           # NEW: Now supports 4 optimizers
    max_iterations: int = 50,
    p: int = 1,
    ansatz: str = None,
    embedding: str = None,
    feature_map: str = None,
    optimizer_options: dict = None,      # NEW: Optimizer-specific options
)
```

## Optimizer Selection Guidelines

| Problem Type | Recommended Optimizer | Reason |
|-------------|----------------------|---------|
| Small VQE (< 10 params) | COBYLA or Nelder-Mead | Fast, derivative-free |
| Large VQE (> 10 params) | BFGS or Adam | Gradient-based efficiency |
| QAOA (low depth) | COBYLA | Robust to noise |
| QAOA (high depth) | Adam | Handles many parameters |
| Noisy simulations | COBYLA or Nelder-Mead | Robust to noise |
| High precision needed | BFGS | Fast convergence |

## Validation

All optimizer names are validated at initialization:

```python
# Valid optimizers
valid_optimizers = ['cobyla', 'nelder-mead', 'bfgs', 'adam']

# Automatic validation
vqe = VQE(optimizer="unknown")  # Raises ValueError
```

## Migration from open-vqa-mhh-add_various_notebooks

The new `openvqa` package enhances the optimizer support from the original repository:

### Original Repository
- Located in `mathematical_tools/`
- Tightly coupled with circuit backends
- Required manual integration

### New openvqa Package
- Simplified standalone implementations
- High-level API integration
- Interactive notebook support
- Validation and error handling
- Optimizer-specific options

## Next Steps

Future enhancements could include:
1. Custom optimizer callbacks for monitoring
2. Additional optimizers (L-BFGS-B, Powell, etc.)
3. Automatic optimizer selection based on problem size
4. Parallel optimizer runs for comparison
5. Gradient computation optimizations for Adam/BFGS

## Files Modified/Created

### New Files
- `openvqa/mathematical_tools/__init__.py`
- `openvqa/mathematical_tools/scipy_minimize_plugin_method.py`
- `openvqa/mathematical_tools/adam_optimizer.py`

### Modified Files
- `openvqa/algorithms/vqe.py` - Added optimizer support
- `openvqa/algorithms/qaoa.py` - Added optimizer support
- `openvqa/examples/vqe_demo.ipynb` - Added optimizer selection cell
- `openvqa/examples/qaoa_demo.ipynb` - Added optimizer selection cell

---

**Note**: This integration provides the same optimizer choices as the `open-vqa-mhh-add_various_notebooks` repository while maintaining the clean, high-level API design of the new `openvqa` package.
