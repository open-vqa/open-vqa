"""Scipy optimization methods supported in OpenVQA."""

from enum import Enum


class ScipyMinimizePluginMethod(Enum):
    """Enumeration of supported scipy.optimize.minimize methods."""
    
    COBYLA = "COBYLA"
    NELDER_MEAD = "Nelder-Mead"
    BFGS = "BFGS"
    
    @classmethod
    def get_all_methods(cls):
        """Return list of all available optimizer names."""
        return [method.value for method in cls]
    
    @classmethod
    def is_valid(cls, method_name):
        """Check if a method name is valid."""
        return method_name.upper() in [m.name for m in cls] or method_name in [m.value for m in cls]
