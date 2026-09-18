"""Analytical and semi-analytical Phase 1 comparison models."""

from __future__ import annotations

import numpy as np

from .constants import PhysicalParams, default_params, omega_from_frequency
from .film_ode import level_c_closed_form
from .thermal_admittance import (
    heat_flux_from_wall_temperature,
    temperature_profile,
    thermal_admittance_halfspace,
    thermal_wavenumber,
    wall_temperature_from_heat_flux,
)


def pressure_profile_from_exponential_temperature(
    f_hz: float,
    T_s_hat: complex,
    y: np.ndarray,
    params: PhysicalParams | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Driven-wave near-field pressure for T_hat = T_s_hat exp(-m_T y).

    The model solves p'' + k^2 p = gamma*k^2 (p0/T0) T_s exp(-m_T y)
    with dp/dy = 0 at the film and an outgoing exp(-i k y) acoustic branch.
    T_s denotes the isobaric diffusive thermal mode, not total gas temperature:
    T_total = T_thermal + (gamma-1)/gamma * T0/p0 * p to leading order.
    Eliminating the acoustic temperature supplies the gamma factor. This remains
    a thin-thermal-layer, inviscid-acoustic approximation, not the full NSF BVP.
    """

    params = params or default_params()
    y = np.asarray(y, dtype=float)
    Omega = omega_from_frequency(f_hz)
    k = Omega / params.c0
    if k == 0.0:
        zeros = np.zeros_like(y, dtype=np.complex128)
        return zeros, zeros
    m_T = thermal_wavenumber(f_hz, params)
    source = params.gamma * k * k * (params.p0 / params.T0) * T_s_hat
    particular = source / (m_T * m_T + k * k)
    outgoing = 1j * (m_T / k) * particular
    thermal_branch = particular * np.exp(-m_T * y)
    acoustic_branch = outgoing * np.exp(-1j * k * y)
    p_hat = thermal_branch + acoustic_branch
    # With exp(+i Omega t), i Omega rho0 u = -dp/dy.  The acoustic impedance
    # relation applies only to the outgoing acoustic branch; applying it to the
    # diffusive thermal branch violates the no-penetration condition at y=0.
    u_thermal = -1j * m_T * thermal_branch / (Omega * params.rho0)
    u_acoustic = acoustic_branch / (params.rho0 * params.c0)
    u_hat = u_thermal + u_acoustic
    return np.asarray(p_hat, dtype=np.complex128), np.asarray(u_hat, dtype=np.complex128)


def mcdonald_wetsel_like(
    f_hz: float,
    *,
    T_s_hat: complex | None = None,
    q_hat: complex | None = None,
    P_hat: complex | None = None,
    C_A: float | None = None,
    beta0: float = 0.0,
    y: np.ndarray | None = None,
    params: PhysicalParams | None = None,
) -> dict[str, np.ndarray | complex]:
    """Return a compact McDonald-Wetsel-like thermal-acoustic reference."""

    params = params or default_params()
    if T_s_hat is None:
        if q_hat is not None:
            T_s_hat = wall_temperature_from_heat_flux(f_hz, q_hat, params)
        elif P_hat is not None:
            T_s_hat = level_c_closed_form(
                f_hz, P_hat, C_A=C_A, params=params, beta0=beta0
            ).T_s_hat
        else:
            raise ValueError("Provide one of T_s_hat, q_hat, or P_hat.")
    if y is None:
        y = np.array([0.0], dtype=float)
    y = np.asarray(y, dtype=float)
    q_g_hat = heat_flux_from_wall_temperature(f_hz, T_s_hat, params)
    T_hat = temperature_profile(y, f_hz, T_s_hat, params)
    p_hat, u_hat = pressure_profile_from_exponential_temperature(f_hz, T_s_hat, y, params)
    return {
        "T_s_hat": complex(T_s_hat),
        "q_g_hat": complex(q_g_hat),
        "T_hat": T_hat,
        "p_hat": p_hat,
        "u_hat": u_hat,
    }


def lim2013_nearfield(**kwargs: object) -> dict[str, np.ndarray | complex]:
    """Main-line Lim-type freestanding CNT near-field reference.

    The default path keeps beta0=0. Non-zero beta0 may be supplied explicitly
    for sensitivity studies, but it is not part of the main M1 gate.
    """

    return mcdonald_wetsel_like(**kwargs)


def arnold_crandall_limit(
    f_hz: float,
    P_hat: complex,
    y: np.ndarray,
    params: PhysicalParams | None = None,
) -> np.ndarray:
    """Small-thermal-layer trend limit used only for sanity checks."""

    params = params or default_params()
    closed = level_c_closed_form(f_hz, P_hat, params=params)
    p_hat, _ = pressure_profile_from_exponential_temperature(
        f_hz, closed.T_s_hat, y, params
    )
    return p_hat


__all__ = [
    "arnold_crandall_limit",
    "lim2013_nearfield",
    "mcdonald_wetsel_like",
    "pressure_profile_from_exponential_temperature",
    "thermal_admittance_halfspace",
]
