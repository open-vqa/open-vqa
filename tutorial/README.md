# OpenVQA notebooks

Two tutorials to learn with, and two demos to experiment with. Every notebook
runs from top to bottom in about a minute or less, and compares its results
with an answer you can check yourself.

## Setup (once)

From the repository root, with Python 3.10, 3.11 or 3.12:

```bash
python -m pip install -r requirements.txt
jupyter lab
```

In Jupyter, use the kernel of the same Python you installed with.

Step-by-step guide, with conda and venv options and checks that everything works:
[setup_environment.ipynb](setup_environment.ipynb).

## Tutorials: start here

| Notebook | What you do | Answer to compare with |
|----------|-------------|------------------------|
| [vqe_tutorial.ipynb](vqe_tutorial.ipynb) | Find the lowest energy of a 2-qubit model, watch the optimizer, compare optimizers, add measurement noise | −0.5, computed exactly in the notebook |
| [qaoa_tutorial.ipynb](qaoa_tutorial.ipynb) | Solve MaxCut on a triangle, see which splits QAOA picks, compare 1 and 2 layers on a square | Maximum cut 2 (triangle) and 4 (square), found by checking every split |

Recommended order: VQE first, then QAOA.

## Demos: experiment

| Notebook | What you can change |
|----------|---------------------|
| [vqe_demo.ipynb](../examples/vqe_demo.ipynb) | Backend, Hamiltonian (`"H2"` or `"LiH"`), ansatz, optimizer, shots, all in one settings cell |
| [qaoa_demo.ipynb](../examples/qaoa_demo.ipynb) | Graph size and random seed, QAOA layers `p`, optimizer, backend, shots; compared with the brute-force best cut |

## Expected results (with `shots=0`)

- **VQE tutorial:** energy −0.5. COBYLA and BFGS report convergence; Nelder-Mead
  reaches −0.499995 within its budget.
- **QAOA tutorial:** expected cut 2.0 on the triangle, with probability 1/6 on each
  of the six best splits. On the square: 3.0 with `p=1`, 4.0 with `p=2`.
- With `shots > 0` the numbers change from run to run; the tutorials explain why.

## Troubleshooting

- **`ModuleNotFoundError: No module named 'openvqa'`**: the notebook kernel is a
  different Python from the one you installed with. Select the right kernel, or
  run the install command with the kernel's Python.
- **Installation fails on Python 3.13 or newer**: myQLM has no builds for it yet.
  Use Python 3.12 (or 3.10/3.11).

## More

- [Main README](../README.md): installation, API, supported Hamiltonians and ansätze
- [Getting Started Guide](../GETTING_STARTED.md)
- [Optimizer Quick Reference](../OPTIMIZER_QUICKREF.md)
