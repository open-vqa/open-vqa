# OpenVQA Optimizer Quick Reference

## Available Optimizers

| Optimizer | Type | Command | Best For |
|-----------|------|---------|----------|
| **COBYLA** | Derivative-free | `optimizer="cobyla"` | Default, constrained problems |
| **Nelder-Mead** | Simplex | `optimizer="nelder-mead"` | Low-dimensional, smooth |
| **BFGS** | Quasi-Newton | `optimizer="bfgs"` | Gradient-based, fast |
| **Adam** | Adaptive | `optimizer="adam"` | Large-scale, noisy |

Adam runs on PennyLane's `default.qubit` (also with the myQLM engine) and supports
`ryrz` and the PennyLane templates only.

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

### Iteration Counts

Counts depend on the problem, `shots` and the starting point, so check each run:
`results.n_evaluations` gives the evaluations used and `results.convergence` the
optimizer's own success flag. For example, VQE on the 2-qubit H2 model with
`ryrz` and `shots=0` needed 197 COBYLA evaluations, while Nelder-Mead was still
at +0.06 after 100 iterations and reached -0.5 within 400.

## Choosing the Optimizer in the Demo Notebooks

In `examples/vqe_demo.ipynb` and `examples/qaoa_demo.ipynb`, set it in the
**Settings** cell:

```python
OPTIMIZER = "cobyla"   # or "nelder-mead", "bfgs", "adam"
```

## Troubleshooting

### Optimizer Not Converging?
- **COBYLA**: Increase `max_iterations`
- **Nelder-Mead**: Try COBYLA instead
- **BFGS**: Use `shots=0`; its numerical gradients stall on sampling noise
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
