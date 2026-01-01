# OpenVQA Optimizer Quick Reference

## Available Optimizers

| Optimizer | Type | Command | Best For |
|-----------|------|---------|----------|
| **COBYLA** | Derivative-free | `optimizer="cobyla"` | Default, constrained problems |
| **Nelder-Mead** | Simplex | `optimizer="nelder-mead"` | Low-dimensional, smooth |
| **BFGS** | Quasi-Newton | `optimizer="bfgs"` | Gradient-based, fast |
| **Adam** | Adaptive | `optimizer="adam"` | Large-scale, noisy |

## Quick Start

### VQE Example
```python
from openvqa import VQA_myQLM
from openvqa.algorithms import VQE

engine = VQA_myQLM(n_qubits=2)
vqe = VQE(hamiltonian="H2", optimizer="cobyla")
results = engine.run(vqe)
```

### QAOA Example
```python
from openvqa import VQA_myQLM
from openvqa.algorithms import QAOA

engine = VQA_myQLM(n_qubits=4)
qaoa = QAOA(problem="maxcut", optimizer="bfgs", p=2)
results = engine.run(qaoa)
```

### Adam with Custom Options
```python
vqe = VQE(
    hamiltonian="H2",
    optimizer="adam",
    optimizer_options={'learning_rate': 0.01}
)
```

## Optimizer Comparison

### Performance Characteristics

| Feature | COBYLA | Nelder-Mead | BFGS | Adam |
|---------|--------|-------------|------|------|
| Requires Gradients | ❌ | ❌ | ✅ | ✅ |
| Handles Constraints | ✅ | ❌ | ❌ | ❌ |
| Noise Robust | ✅ | ✅ | ❌ | ✅ |
| Fast Convergence | ⚡ | ⚡ | ⚡⚡⚡ | ⚡⚡ |
| Memory Efficient | ✅ | ✅ | ❌ | ✅ |
| Large-scale | ❌ | ❌ | ✅ | ✅ |

### Typical Iteration Counts

| Problem | COBYLA | Nelder-Mead | BFGS | Adam |
|---------|--------|-------------|------|------|
| VQE (2 qubits) | 50-100 | 50-100 | 30-50 | 100-200 |
| VQE (4 qubits) | 100-200 | 100-200 | 50-100 | 200-400 |
| QAOA (p=1) | 30-50 | 30-50 | 20-30 | 50-100 |
| QAOA (p=2) | 50-100 | 50-100 | 30-60 | 100-200 |

## Interactive Notebook Selection

In both `vqe_demo.ipynb` and `qaoa_demo.ipynb`:

```
Select optimizer:
  1) COBYLA (default)
  2) Nelder-Mead
  3) BFGS
  4) Adam
Enter 1/2/3/4 (default 1):
```

## Troubleshooting

### Optimizer Not Converging?
- **COBYLA**: Increase `max_iterations`
- **Nelder-Mead**: Try COBYLA instead
- **BFGS**: Check if gradients are available
- **Adam**: Reduce `learning_rate`

### Slow Optimization?
- **COBYLA/Nelder-Mead**: Try BFGS if smooth
- **BFGS**: Ensure gradients are efficient
- **Adam**: Increase `learning_rate` (carefully)

### Getting NaN or Inf?
- **All**: Reduce parameter ranges
- **Adam**: Reduce `learning_rate`
- **BFGS**: Add bounds or switch to COBYLA

## Advanced Usage

### Custom Learning Rate for Adam
```python
vqe = VQE(
    optimizer="adam",
    optimizer_options={
        'learning_rate': 0.005  # Lower for stability
    }
)
```

### Combining with Different Ansatz
```python
# With Qiskit ansatz
vqe = VQE(ansatz="RealAmplitudes", optimizer="bfgs")

# With PennyLane template
vqe = VQE(ansatz="StronglyEntanglingLayers", optimizer="adam")
```

## Optimizer Theory (Brief)

### COBYLA
- **Idea**: Build linear approximations of constraints
- **Pros**: No derivatives, handles constraints
- **Cons**: Can be slow for high dimensions

### Nelder-Mead
- **Idea**: Move simplex toward minimum
- **Pros**: Simple, no derivatives
- **Cons**: Can get stuck in local minima

### BFGS
- **Idea**: Approximate Hessian using gradients
- **Pros**: Fast convergence, efficient
- **Cons**: Requires gradients, memory intensive

### Adam
- **Idea**: Adaptive learning rates per parameter
- **Pros**: Works well with noisy gradients
- **Cons**: Requires tuning, many iterations

## References

- **COBYLA**: Powell, M.J.D. (1994)
- **Nelder-Mead**: Nelder & Mead (1965)
- **BFGS**: Broyden, Fletcher, Goldfarb, Shanno (1970)
- **Adam**: Kingma & Ba (2014)
