"""P4-D3 compact-source map fixtures (D3-4, project section 12).

The map farfield/compact_source.py is the D3-4 handoff formula (certified T_s_hat -> pumping
u_src -> one-way soft-source dp = Z0 u_src). These fixtures pin it against independent
derivations -- NOT against itself:

  * the closed form u_src = (1+i)/2 Omega delta_T T_s_hat/T0 vs a fine trapezoidal quadrature
    of the defining layer integral (i Omega/T0) Integral T_hat(y) dy (non-tautological: the
    quadrature never calls the closed form);
  * magnitude/phase identities (|u_src| = Omega delta_T/sqrt(2) |T_hat|/T0, +45 deg lead);
  * profile limits (u(0)=0, u(infinity)=u_src) and linearity in the complex T_s_hat;
  * the handoff amplitude dp = Z0 u_src at the 10 kHz air working point, with the documented
    per-side convention (no factor 2).

The ON-STACK realization of this map (does the frozen M3 stack's velocity field actually
follow it?) is measured by scripts/phase4_d3_source_extraction_probe.py RIG 1, not here."""

from __future__ import annotations

import cmath
import math

import numpy as np

from farfield.compact_source import (
    pumping_velocity_profile_m_s,
    soft_source_pressure_amplitude_pa,
    temperature_profile_K,
    thermal_boundary_layer_thickness_m,
    thermal_pumping_velocity_m_s,
)

# 10 kHz air working point (frozen Level C physical scales)
ALPHA = 2.2233775895e-5
T0 = 300.0
F_HZ = 1.0e4
OMEGA = 2.0 * math.pi * F_HZ
RHO0 = 1.177
C0 = 347.0


def test_closed_form_matches_layer_integral_quadrature():
    """u_src closed form == (i Omega/T0) trapz(T_hat(y)) on a fine grid (independent path)."""

    T_s = complex(7.5, -2.0)
    delta = thermal_boundary_layer_thickness_m(ALPHA, OMEGA)
    y = np.linspace(0.0, 40.0 * delta, 400_001)   # deep cutoff: e^-40 tail is negligible
    integral = np.trapezoid(temperature_profile_K(y, T_s, omega_rad_s=OMEGA, alpha_m2_s=ALPHA), y)
    u_quad = 1j * OMEGA / T0 * integral
    u_closed = thermal_pumping_velocity_m_s(T_s, T0_K=T0, omega_rad_s=OMEGA, alpha_m2_s=ALPHA)
    assert abs(u_quad - u_closed) / abs(u_closed) < 1.0e-6


def test_magnitude_and_phase_identities():
    """|u_src| = Omega delta_T/sqrt(2) |T_hat|/T0 and u_src leads a real T_s_hat by +45 deg."""

    T_s = 10.0
    delta = thermal_boundary_layer_thickness_m(ALPHA, OMEGA)
    u = thermal_pumping_velocity_m_s(T_s, T0_K=T0, omega_rad_s=OMEGA, alpha_m2_s=ALPHA)
    assert abs(abs(u) - OMEGA * delta / math.sqrt(2.0) * T_s / T0) / abs(u) < 1.0e-12
    assert abs(math.degrees(cmath.phase(u)) - 45.0) < 1.0e-9


def test_profile_limits_and_linearity():
    """u(0)=0, u(y>>delta)->u_src; the map is linear in the complex T_s_hat."""

    T_s = complex(3.0, 4.0)
    delta = thermal_boundary_layer_thickness_m(ALPHA, OMEGA)
    u_src = thermal_pumping_velocity_m_s(T_s, T0_K=T0, omega_rad_s=OMEGA, alpha_m2_s=ALPHA)
    prof = pumping_velocity_profile_m_s(
        np.array([0.0, 30.0 * delta]), T_s, T0_K=T0, omega_rad_s=OMEGA, alpha_m2_s=ALPHA)
    assert abs(prof[0]) < 1.0e-15
    assert abs(prof[1] - u_src) / abs(u_src) < 1.0e-12
    scale = complex(-0.7, 1.3)
    u_scaled = thermal_pumping_velocity_m_s(
        scale * T_s, T0_K=T0, omega_rad_s=OMEGA, alpha_m2_s=ALPHA)
    assert abs(u_scaled - scale * u_src) / abs(u_src) < 1.0e-12


def test_handoff_amplitude_at_working_point():
    """dp = Z0 u_src (per side, no factor 2): sane 10 kHz magnitude for T_s_hat = 10 K."""

    u = thermal_pumping_velocity_m_s(10.0, T0_K=T0, omega_rad_s=OMEGA, alpha_m2_s=ALPHA)
    dp = soft_source_pressure_amplitude_pa(u, rho0_kg_m3=RHO0, c0_m_s=C0)
    assert abs(dp - RHO0 * C0 * u) == 0.0
    assert 10.0 < abs(dp) < 25.0    # ~16 Pa: Omega delta/sqrt(2) * (10/300) * Z0
