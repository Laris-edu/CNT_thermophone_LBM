"""Complex amplitude convention shared by Phase 2 and Phase 3."""

from __future__ import annotations

import math

import numpy as np


def complex_amplitude(t, signal, frequency_hz: float, convention: str = "exp+iOmega_t") -> complex:
    """Return ``x_hat`` such that ``x(t)=Re[x_hat exp(i Omega t)]``."""

    if convention != "exp+iOmega_t":
        raise ValueError("only x(t)=Re[x_hat exp(i Omega t)] is supported")
    t = np.asarray(t, dtype=float)
    signal = np.asarray(signal, dtype=float)
    omega = 2.0 * math.pi * float(frequency_hz)
    matrix = np.column_stack((np.cos(omega * t), np.sin(omega * t)))
    coeffs, *_ = np.linalg.lstsq(matrix, signal, rcond=None)
    a, b = coeffs
    return complex(a - 1j * b)


def spl_db_from_p_hat(p_hat: complex, p_ref: float = 20.0e-6) -> float:
    p_rms = abs(p_hat) / math.sqrt(2.0)
    return float(20.0 * math.log10(p_rms / p_ref))

