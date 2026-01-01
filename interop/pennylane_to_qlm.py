"""Simple Pennylane -> Qiskit circuit builders for common templates.

This module provides small, explicit mappings for a handful of PennyLane
templates to equivalent Qiskit `QuantumCircuit` constructions. The goal is
to allow executing PennyLane-style templates via the existing QLM path by
producing a Qiskit circuit which can then be converted with
`qat.interop.qiskit.qiskit_to_qlm`.

This is not a full reimplementation of all PennyLane templates; it provides
useful common templates (StronglyEntanglingLayers, BasicEntanglerLayers,
AngleEmbedding, AmplitudeEmbedding, RandomLayers) with parameter counts so
the notebook can display expected parameter sizes.
"""
from typing import Tuple, List
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
import numpy as np


def available_templates() -> List[str]:
    return [
        "StronglyEntanglingLayers",
        "BasicEntanglerLayers",
        "RandomLayers",
        "AngleEmbedding",
        "AmplitudeEmbedding",
    ]


def strongly_entangling_layers_qiskit(n_qubits: int, n_layers: int) -> Tuple[QuantumCircuit, int]:
    qc = QuantumCircuit(n_qubits)
    n_params = 3 * n_qubits * n_layers
    param_idx = 0
    for layer in range(n_layers):
        for q in range(n_qubits):
            # Use three rotation angles per wire
            qc.ry(Parameter(f'p_{param_idx}'), q)
            param_idx += 1
            qc.rz(Parameter(f'p_{param_idx}'), q)
            param_idx += 1
            qc.rx(Parameter(f'p_{param_idx}'), q)
            param_idx += 1
        # Entangling as a chain with wrap-around
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
        if n_qubits > 1:
            qc.cx(n_qubits - 1, 0)
    return qc, n_params


def basic_entangler_layers_qiskit(n_qubits: int, n_layers: int) -> Tuple[QuantumCircuit, int]:
    qc = QuantumCircuit(n_qubits)
    n_params = n_qubits * n_layers
    param_idx = 0
    for layer in range(n_layers):
        for q in range(n_qubits):
            qc.ry(Parameter(f'p_{param_idx}'), q)
            param_idx += 1
        # simple entangling ring
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
        if n_qubits > 1:
            qc.cx(n_qubits - 1, 0)
    return qc, n_params


def random_layers_qiskit(n_qubits: int, n_layers: int) -> Tuple[QuantumCircuit, int]:
    qc = QuantumCircuit(n_qubits)
    n_params = 3 * n_qubits * n_layers
    param_idx = 0
    for layer in range(n_layers):
        for q in range(n_qubits):
            qc.rz(Parameter(f'p_{param_idx}'), q)
            param_idx += 1
            qc.ry(Parameter(f'p_{param_idx}'), q)
            param_idx += 1
            qc.rx(Parameter(f'p_{param_idx}'), q)
            param_idx += 1
        for q in range(n_qubits - 1):
            qc.cz(q, q + 1)
    return qc, n_params


def angle_embedding_qiskit(n_qubits: int) -> Tuple[QuantumCircuit, int]:
    qc = QuantumCircuit(n_qubits)
    # AngleEmbedding typically uses one parameter per wire
    for q in range(n_qubits):
        qc.ry(0.0, q)
    return qc, n_qubits


def basis_embedding_qiskit(n_qubits: int) -> Tuple[QuantumCircuit, int]:
    qc = QuantumCircuit(n_qubits)
    # BasisEmbedding encodes binary strings, no parameters
    return qc, 0


def displacement_embedding_qiskit(n_qubits: int) -> Tuple[QuantumCircuit, int]:
    qc = QuantumCircuit(n_qubits)
    # DisplacementEmbedding for continuous-variable, but for qubits, approximate with RY
    for q in range(n_qubits):
        qc.ry(0.0, q)
        qc.rz(0.0, q)
    return qc, 2 * n_qubits


def amplitude_embedding_qiskit(n_qubits: int) -> Tuple[QuantumCircuit, int]:
    qc = QuantumCircuit(n_qubits)
    # AmplitudeEmbedding maps a statevector into the circuit: use Initialize
    # in Qiskit if desired. Parameter count is 0 for trainable parameters.
    return qc, 0


def template_to_qiskit(name: str, n_qubits: int, n_layers: int = 1) -> Tuple[QuantumCircuit, int]:
    name = name or ""
    key = name.lower()
    if key == "stronglyentanglinglayers" or key == "stronglyentanglinglayers":
        return strongly_entangling_layers_qiskit(n_qubits, n_layers)
    if key == "basicentanglerlayers" or key == "basicentanglerlayers":
        return basic_entangler_layers_qiskit(n_qubits, n_layers)
    if key == "randomlayers":
        return random_layers_qiskit(n_qubits, n_layers)
    if key == "angleembedding":
        return angle_embedding_qiskit(n_qubits)
    if key == "amplitudeembedding":
        return amplitude_embedding_qiskit(n_qubits)
    if key == "basisembedding":
        return basis_embedding_qiskit(n_qubits)
    if key == "displacementembedding":
        return displacement_embedding_qiskit(n_qubits)

    # Default fallback: simple ry-rz layers similar to RYRZ
    qc = QuantumCircuit(n_qubits)
    n_params = 2 * n_qubits * n_layers
    for layer in range(n_layers):
        for q in range(n_qubits):
            qc.ry(0.0, q)
        for q in range(n_qubits):
            qc.rz(0.0, q)
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
    return qc, n_params


def build_vqe_circuit_qiskit(embedding_name: str, ansatz_name: str, n_qubits: int, n_layers: int = 1) -> Tuple[QuantumCircuit, int]:
    """Build a full VQE circuit: embedding + ansatz."""
    qc = QuantumCircuit(n_qubits)
    total_params = 0
    
    # Add embedding
    if embedding_name:
        embed_qc, embed_params = template_to_qiskit(embedding_name, n_qubits, 1)  # embeddings don't have layers
        qc.compose(embed_qc, inplace=True)
        total_params += embed_params
    
    # Add ansatz
    ansatz_qc, ansatz_params = template_to_qiskit(ansatz_name, n_qubits, n_layers)
    qc.compose(ansatz_qc, inplace=True)
    total_params += ansatz_params
    
    return qc, total_params
