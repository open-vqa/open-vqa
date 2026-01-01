"""Public, discoverable list of supported algorithm options.

This module exists mainly for notebooks / quickstarts: users can inspect
available feature maps, ansatz and optimizers without reading source or
reconstructing string lists.
"""

from __future__ import annotations

from typing import Any, Dict

from openvqa.algorithms.vqe import VQE
from openvqa.algorithms.qaoa import QAOA
from openvqa.interop.pennylane_to_qlm import available_templates


AVAILABLE_OPTIONS: Dict[str, Any] = {
    "vqe": {
        "optimizers": list(VQE.SUPPORTED_OPTIMIZERS),
        "feature_maps": list(VQE.SUPPORTED_FEATURE_MAPS),
        "ansatz": {
            "internal": list(VQE.SUPPORTED_ANSATZ_INTERNAL),
            "qiskit_templates": list(VQE.SUPPORTED_ANSATZ_QISKIT),
            "pennylane_templates_via_qlm": list(available_templates()),
        },
    },
    "qaoa": {
        "optimizers": list(QAOA.SUPPORTED_OPTIMIZERS),
        "feature_maps": list(QAOA.SUPPORTED_FEATURE_MAPS),
    },
}
