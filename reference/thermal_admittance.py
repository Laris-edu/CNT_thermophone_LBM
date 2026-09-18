"""Closed-form half-space thermal admittance helpers."""

from __future__ import annotations

import numpy as np

from .constants import PhysicalParams, default_params, omega_from_frequency


def thermal_wavenumber(
    f_hz: float,
    params: PhysicalParams | None = None,
    *,
    alpha0: float | None = None,
) -> complex:
    """Return m_T = sqrt(i Omega / alpha0), choosing Re(m_T) > 0."""

    params = params or default_params()
    alpha = params.alpha0 if alpha0 is None else alpha0
    m_T = np.sqrt(1j * omega_from_frequency(f_hz) / alpha)
    if np.real(m_T) < 0.0:
        m_T = -m_T
    return complex(m_T)


def thermal_admittance_halfspace(
    f_hz: float,
    params: PhysicalParams | None = None,
    *,
    kg: float | None = None,
    alpha0: float | None = None,
) -> complex:
    """Return q_hat / T_s_hat for the one-sided gas half-space."""

    params = params or default_params()
    conductivity = params.kg if kg is None else kg
    return conductivity * thermal_wavenumber(f_hz, params, alpha0=alpha0)


def heat_flux_from_wall_temperature(
    f_hz: float,
    T_s_hat: complex,
    params: PhysicalParams | None = None,
) -> complex:
    return thermal_admittance_halfspace(f_hz, params) * T_s_hat


def wall_temperature_from_heat_flux(
    f_hz: float,
    q_hat: complex,
    params: PhysicalParams | None = None,
) -> complex:
    return q_hat / thermal_admittance_halfspace(f_hz, params)


def temperature_profile(
    y: np.ndarray,
    f_hz: float,
    T_s_hat: complex,
    params: PhysicalParams | None = None,
) -> np.ndarray:
    m_T = thermal_wavenumber(f_hz, params)
    return np.asarray(T_s_hat * np.exp(-m_T * np.asarray(y, dtype=float)), dtype=np.complex128)

