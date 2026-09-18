"""P4-D3 compact-source map: certified near-wall temperature -> thermoacoustic pumping velocity.

Project: docs/Phase_4/M4/P4_D3_Multidomain_Acoustic_Project.md (D3-4, section 12).

The D3-4 handoff verdict (section 12): the fine/M3 domain must NOT hand its emission to the
coarse acoustic domain by radiating it (the volume-injection floor swamps the physical wave
31-57x); instead the M3-certified near-wall state determines the source strength analytically
and the one-way soft source (section 11) injects it. This module is that map -- the single
source of truth for the formula (the extraction probe imports it, so the on-stack fit in
scripts/phase4_d3_source_extraction_probe.py genuinely tests THIS code).

Physics (half-space oscillating wall temperature, x(t)=Re[x_hat exp(+i Omega t)] convention,
frozen since P3-0): the wall imposes T'(0,t)=Re[T_s_hat e^{i Omega t}]; the gas temperature is
the decaying diffusion wave T_hat(y) = T_s_hat exp(-(1+i) y/delta_T), delta_T = sqrt(2 alpha/
Omega). The layer expands isobarically, d u/d y = (1/T0) dT'/dt, so the pumping velocity above
the layer is the layer integral

    u_src = (i Omega/T0) * Integral T_hat dy = (1+i)/2 * Omega delta_T * (T_s_hat/T0),

i.e. |u_src| = Omega delta_T/sqrt(2) * |T_s_hat|/T0 leading T_s_hat by +45 deg. The one-way
soft source then injects the outgoing plane wave dp_hat = Z0 * u_src (per side; a freestanding
film pumps both sides symmetrically -- the upper-side handoff carries NO factor 2; the film
energy bookkeeping's 2 q_g'' double-sided factor lives in the film ODE, not here).

Validity/error budget (documented, not hidden): u_src inherits the Level C T_s_hat +/-5.4%
amplitude / ~2 deg phase band linearly; the map itself is leading-order half-space theory --
isobaric-layer and compactness corrections are O(k delta_T) ~ 5e-3 at 10 kHz (negligible
against the +/-5.4% band); the M4 amplitude reference remains contract section 10.3 R0/R1/R2,
never this closed form used as its own truth."""

from __future__ import annotations

import math

import numpy as np


def thermal_boundary_layer_thickness_m(alpha_m2_s: float, omega_rad_s: float) -> float:
    """delta_T = sqrt(2 alpha / Omega)."""

    if alpha_m2_s <= 0.0 or omega_rad_s <= 0.0:
        raise ValueError("alpha and omega must be positive")
    return math.sqrt(2.0 * alpha_m2_s / omega_rad_s)


def thermal_pumping_velocity_m_s(
    T_s_hat_K: complex,
    *,
    T0_K: float,
    omega_rad_s: float,
    alpha_m2_s: float,
) -> complex:
    """Compact-source pumping u_src = (1+i)/2 * Omega delta_T * T_s_hat/T0 (m/s, complex).

    ``T_s_hat_K`` is the complex wall-temperature amplitude in Kelvin under the frozen
    x(t)=Re[x_hat exp(+i Omega t)] convention (e.g. Level C ``T_s_hat``)."""

    delta = thermal_boundary_layer_thickness_m(alpha_m2_s, omega_rad_s)
    return complex(0.5, 0.5) * omega_rad_s * delta * (complex(T_s_hat_K) / float(T0_K))


def soft_source_pressure_amplitude_pa(
    u_src_m_s: complex,
    *,
    rho0_kg_m3: float,
    c0_m_s: float,
) -> complex:
    """One-way handoff amplitude dp_hat = Z0 * u_src for the section-11 soft source (per side)."""

    return float(rho0_kg_m3) * float(c0_m_s) * complex(u_src_m_s)


def pumping_velocity_profile_m_s(
    y_m: np.ndarray,
    T_s_hat_K: complex,
    *,
    T0_K: float,
    omega_rad_s: float,
    alpha_m2_s: float,
) -> np.ndarray:
    """Half-space in-layer velocity profile u(y) = u_src (1 - exp(-(1+i) y/delta_T)).

    The on-stack extraction fit (source-extraction probe RIG 1) uses exactly this shape plus a
    constant closed-box backflow term."""

    y = np.asarray(y_m, dtype=float)
    if np.any(y < 0.0):
        raise ValueError("y must be non-negative")
    delta = thermal_boundary_layer_thickness_m(alpha_m2_s, omega_rad_s)
    u_src = thermal_pumping_velocity_m_s(
        T_s_hat_K, T0_K=T0_K, omega_rad_s=omega_rad_s, alpha_m2_s=alpha_m2_s)
    return u_src * (1.0 - np.exp(-(1.0 + 1.0j) * y / delta))


def temperature_profile_K(
    y_m: np.ndarray,
    T_s_hat_K: complex,
    *,
    omega_rad_s: float,
    alpha_m2_s: float,
) -> np.ndarray:
    """Decaying thermal diffusion wave T_hat(y) = T_s_hat exp(-(1+i) y/delta_T)."""

    y = np.asarray(y_m, dtype=float)
    delta = thermal_boundary_layer_thickness_m(alpha_m2_s, omega_rad_s)
    return complex(T_s_hat_K) * np.exp(-(1.0 + 1.0j) * y / delta)
