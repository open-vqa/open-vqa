# OpenVQA Tutorials

Welcome to the OpenVQA tutorial collection! These notebooks provide hands-on, beginner-friendly examples to help you get started with quantum variational algorithms.

## 📚 Available Tutorials

### 1. [VQE Tutorial](vqe_tutorial.ipynb) - Variational Quantum Eigensolver
**Learn to find ground state energies of molecular systems**

- **What you'll learn**: Basic VQE workflow, optimizer selection, ansatz configuration
- **Difficulty**: Beginner
- **Time**: ~15 minutes
- **Topics covered**:
  - H2 molecule ground state calculation
  - Comparing COBYLA, BFGS, and Adam optimizers
  - Comparing RealAmplitudes and EfficientSU2 ansätze
  - Result visualization and analysis
  - Comparing with exact solutions

**Key Examples**:
```python
from openvqa import VQA_myQLM
from openvqa.algorithms import VQE

engine = VQA_myQLM(n_qubits=2)
vqe = VQE(hamiltonian="H2", ansatz="RealAmplitudes", optimizer="cobyla")
results = engine.run(vqe)
```

### 1b. [VQE Tutorial (Fixed Configuration)](vqe_fixed_tutorial.ipynb)
**Non-interactive notebook with a single backend + feature map + ansatz**

- **What you'll learn**: How to run VQE end-to-end without `input()` prompts
- **Best for**: scripting, CI runs, and copy/paste into `.py` scripts

### 2. [QAOA Tutorial](qaoa_tutorial.ipynb) - Quantum Approximate Optimization Algorithm
**Learn to solve combinatorial optimization problems**

- **What you'll learn**: QAOA basics, MaxCut problem, parameter tuning
- **Difficulty**: Beginner
- **Time**: ~15 minutes
- **Topics covered**:
  - MaxCut problem on 4-node graphs
  - QAOA depth (p) parameter effects
  - Optimizer comparison (COBYLA, BFGS, Nelder-Mead)
  - Performance analysis and visualization
  - Circuit depth vs. solution quality trade-offs

**Key Examples**:
```python
from openvqa import VQA_myQLM
from openvqa.algorithms import QAOA

engine = VQA_myQLM(n_qubits=4)
qaoa = QAOA(problem="maxcut", graph_size=4, optimizer="bfgs", p=2)
results = engine.run(qaoa)
```

### 2b. [QAOA Tutorial (Fixed Configuration)](qaoa_fixed_tutorial.ipynb)
**Non-interactive notebook with a single backend and fixed settings**

- **What you'll learn**: Standard MaxCut QAOA run without prompts
- **Best for**: scripting, CI runs, and quick smoke tests

### 3. [QLM Backend (Quick Start)](qlm_backend_simplified.ipynb)
**See how to use Qiskit circuits, convert to QLM, and run VQE in one short flow**

- **What you'll learn**: Why QLM is the primary execution path here and how to convert/run a Qiskit ansatz via QLM
- **Difficulty**: Beginner
- **Time**: ~5 minutes

## 🚀 Getting Started

### Prerequisites

Make sure you have OpenVQA installed:
```bash
pip install -r ../../requirements.txt
```

### Running the Tutorials

1. **Open Jupyter Notebook/Lab**:
   ```bash
   jupyter notebook
   ```
   
2. **Navigate to the tutorial folder**:
   - Click on `tutorial/`
   - Open either `vqe_tutorial.ipynb` or `qaoa_tutorial.ipynb`

3. **Run cells sequentially**:
   - Use `Shift + Enter` to run each cell
   - Follow along with the explanations
   - Experiment with parameters!

### Quick Test

Run this in a Python environment to verify your setup:
```python
import sys
sys.path.insert(0, '../..')
from openvqa import VQA_myQLM
from openvqa.algorithms import VQE, QAOA
print("✓ OpenVQA is ready!")
```

## 📖 Tutorial Structure

Each tutorial follows this pattern:

1. **Introduction** - What the algorithm does and why it's useful
2. **Setup** - Import libraries and initialize the engine
3. **Examples** - Multiple practical examples with different configurations
4. **Comparison** - Visual comparison of results
5. **Summary** - Key takeaways and next steps

## 🎯 Learning Path

**Recommended order for beginners**:

1. **Start with VQE Tutorial** → Understand the basics of variational quantum algorithms
2. **Then QAOA Tutorial** → Apply concepts to optimization problems
3. **Explore Interactive Demos** → Try [vqe_demo.ipynb](../examples/vqe_demo.ipynb) and [qaoa_demo.ipynb](../examples/qaoa_demo.ipynb) for advanced features

## 💡 Key Concepts

### What is a Variational Quantum Algorithm?

Variational quantum algorithms are hybrid quantum-classical algorithms that:
- Use parametrized quantum circuits (ansätze)
- Optimize parameters using classical optimizers
- Find approximate solutions to complex problems

### Supported Optimizers

| Optimizer | Type | Best For |
|-----------|------|----------|
| **COBYLA** | Gradient-free | General purpose, robust |
| **Nelder-Mead** | Simplex | Smooth landscapes |
| **BFGS** | Quasi-Newton | Fast convergence with gradients |
| **Adam** | Adaptive | Large-scale, noisy problems |

See [OPTIMIZER_QUICKREF.md](../OPTIMIZER_QUICKREF.md) for details.

## 🔬 Experiment Ideas

After completing the tutorials, try:

### For VQE:
- Change the number of layers (`n_layers=3, 4, 5`)
- Try LiH molecule instead of H2
- Experiment with different learning rates for Adam
- Compare convergence speed across optimizers

### For QAOA:
- Increase QAOA depth (`p=3, 4, 5`)
- Try larger graphs (`graph_size=6, 8`)
- Test with different random seeds
- Analyze how p affects solution quality

## 📊 Expected Results

### VQE (H2 Molecule):
- Ground state energy: ≈ -1.137 Ha
- Typical execution time: 1-10 seconds
- Convergence: 20-100 iterations

### QAOA (4-node MaxCut):
- MaxCut value: varies by graph
- Typical execution time: 1-10 seconds
- Better results with higher p

## 🐛 Troubleshooting

**Issue**: Import errors
```
Solution: Ensure you're in the right directory and have installed dependencies
cd openvqa/tutorial
pip install -r ../../requirements.txt
```

**Issue**: Slow execution
```
Solution: Reduce max_iterations or use a simpler ansatz
vqe = VQE(max_iterations=20, n_layers=1)
```

**Issue**: Poor convergence
```
Solution: Try a different optimizer or increase iterations
vqe = VQE(optimizer="bfgs", max_iterations=200)
```

## 📚 Additional Resources

### Documentation
- [OpenVQA Main README](../README.md)
- [Getting Started Guide](../GETTING_STARTED.md)
- [Optimizer Quick Reference](../OPTIMIZER_QUICKREF.md)

### Interactive Demos
- [VQE Interactive Demo](../examples/vqe_demo.ipynb) - Full-featured with all options
- [QAOA Interactive Demo](../examples/qaoa_demo.ipynb) - Comprehensive QAOA examples

### External Learning
- [Qiskit Textbook - VQE](https://qiskit.org/textbook/ch-applications/vqe-molecules.html)
- [Pennylane Tutorials](https://pennylane.ai/qml/demos_quantum-computing.html)
- [QAOA Paper](https://arxiv.org/abs/1411.4028) - Original Farhi et al. paper

## 🤝 Contributing

Have ideas for new tutorials? Found an issue? Contributions are welcome!

1. Create a new tutorial notebook following the existing format
2. Add it to this README
3. Submit a pull request

## 📝 Tutorial Checklist

When creating new tutorials:

- [ ] Clear title and introduction
- [ ] Prerequisites listed
- [ ] Step-by-step explanations
- [ ] Multiple working examples
- [ ] Visualization of results
- [ ] Summary with key takeaways
- [ ] Links to related resources
- [ ] Tested and runs without errors

## 🎓 Next Steps

After mastering these tutorials:

1. **Explore Advanced Topics**:
   - Custom Hamiltonians
   - Hardware noise simulation
   - Error mitigation techniques

2. **Real Applications**:
   - Molecular chemistry calculations
   - Portfolio optimization
   - Network design problems

3. **Contribute**:
   - Share your own examples
   - Help improve documentation
   - Report issues or suggestions

---

## 📞 Support

Need help? Check out:
- [GitHub Issues](../../issues)
- [Documentation](../README.md)
- [Examples folder](../examples/)

Happy learning! 🚀🔬
