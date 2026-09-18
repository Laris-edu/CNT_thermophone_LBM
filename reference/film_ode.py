"""Thin-film ODE coupling and energy residuals for Phase 1."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .constants import PhysicalParams, default_params, omega_from_frequency
from .thermal_admittance import thermal_admittance_halfspace


@dataclass(frozen=True)
class LevelCClosedForm:
    T_s_hat: complex
    q_g_hat: complex
    energy_residual_hat: complex
    energy_residual_rel: float


def energy_residual_frequency(
    *,
    P_hat: complex,
    q_g_hat: complex,
    T_s_hat: complex,
    Omega: float,
    C_A: float,
    beta0: float = 0.0,
) -> complex:
    return P_hat - 2.0 * q_g_hat - 1j * Omega * C_A * T_s_hat - 2.0 * beta0 * T_s_hat


def relative_energy_residual(
    *,
    residual: complex,
    P_hat: complex,
    q_g_hat: complex,
    T_s_hat: complex,
    Omega: float,
    C_A: float,
    beta0: float = 0.0,
) -> float:
    scale = max(
        abs(P_hat),
        abs(2.0 * q_g_hat),
        abs(1j * Omega * C_A * T_s_hat),
        abs(2.0 * beta0 * T_s_hat),
        1e-300,
    )
    return float(abs(residual) / scale)


def level_c_closed_form(
    f_hz: float,
    P_hat: complex,
    C_A: float | None = None,
    params: PhysicalParams | None = None,
    *,
    beta0: float | None = None,
) -> LevelCClosedForm:
    """Closed-form Level C solution using one-sided half-space admittance."""

    params = params or default_params()
    C_A = params.C_A if C_A is None else C_A
    beta0 = params.beta0 if beta0 is None else beta0
    Omega = omega_from_frequency(f_hz)
    Y_T = thermal_admittance_halfspace(f_hz, params)
    T_s_hat = P_hat / (1j * Omega * C_A + 2.0 * Y_T + 2.0 * beta0)
    q_g_hat = Y_T * T_s_hat
    residual = energy_residual_frequency(
        P_hat=P_hat,
        q_g_hat=q_g_hat,
        T_s_hat=T_s_hat,
        Omega=Omega,
        C_A=C_A,
        beta0=beta0,
    )
    return LevelCClosedForm(
        T_s_hat=complex(T_s_hat),
        q_g_hat=complex(q_g_hat),
        energy_residual_hat=complex(residual),
        energy_residual_rel=relative_energy_residual(
            residual=residual,
            P_hat=P_hat,
            q_g_hat=q_g_hat,
            T_s_hat=T_s_hat,
            Omega=Omega,
            C_A=C_A,
            beta0=beta0,
        ),
    )


def energy_residual_time(
    P_in: np.ndarray,
    q_g: np.ndarray,
    T_s: np.ndarray,
    dT_s_dt: np.ndarray,
    *,
    C_A: float,
    beta0: float = 0.0,
) -> np.ndarray:
    return P_in - 2.0 * q_g - C_A * dT_s_dt - 2.0 * beta0 * T_s

