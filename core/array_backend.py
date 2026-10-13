"""Small, lazy NumPy/CuPy boundary; host copies are always explicit."""
import numpy as np


def array_module(*values):
    """Select CuPy when a CUDA array is present, without importing it for CPU use."""
    if any(hasattr(value, '__cuda_array_interface__') for value in values):
        import cupy
        return cupy
    return np


def to_numpy(value):
    """Explicitly transfer device data to NumPy at an output/diagnostic boundary."""
    xp = array_module(value)
    return np.asarray(value) if xp is np else xp.asnumpy(value)
