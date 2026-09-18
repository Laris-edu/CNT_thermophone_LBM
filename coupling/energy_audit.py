"""Energy-audit helpers for Phase_3 film and Level C coupling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .film_ode import FilmOdeParams, energy_residual_cumulative


@dataclass(frozen=True)
class EnergyAuditResult:
    """Integrated film energy audit over a sampled trajectory."""

    residual_si: np.ndarray
    scale_si: float
    max_abs_residual_si: float
    max_relative_residual: float
    final_residual_si: float
    tolerance: float

    @property
    def passed(self) -> bool:
        return bool(np.isfinite(self.max_relative_residual) and self.max_relative_residual <= self.tolerance)


def cumulative_trapezoid_abs(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Cumulative trapezoid integral of ``abs(y)``."""

    y_abs = np.abs(np.asarray(y, dtype=float))
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(y_abs, dtype=float)
    if y_abs.size >= 2:
        out[1:] = np.cumsum(0.5 * (y_abs[1:] + y_abs[:-1]) * np.diff(x))
    return out


def film_energy_scale(
    *,
    t_si: np.ndarray,
    P_in_si: np.ndarray,
    q_g_one_sided_si: np.ndarray,
    T_s_K: np.ndarray,
    params: FilmOdeParams,
) -> float:
    """Return a nonzero work scale for relative integrated energy residuals."""

    t = np.asarray(t_si, dtype=float)
    P = np.asarray(P_in_si, dtype=float)
    q = np.asarray(q_g_one_sided_si, dtype=float)
    T = np.asarray(T_s_K, dtype=float)
    leak = params.linear_leak_conductance_si * (T - params.T_ref_K)
    input_work = cumulative_trapezoid_abs(P, t)[-1] if t.size else 0.0
    heat_loss_work = cumulative_trapezoid_abs(params.gas_flux_factor * q + leak, t)[-1] if t.size else 0.0
    storage = float(abs(params.C_A_si * (T[-1] - T[0]))) if T.size else 0.0
    return float(max(input_work, heat_loss_work, storage, 1.0e-300))


def audit_film_energy(
    *,
    t_si: np.ndarray,
    P_in_si: np.ndarray,
    q_g_one_sided_si: np.ndarray,
    T_s_K: np.ndarray,
    params: FilmOdeParams,
    tolerance: float,
) -> EnergyAuditResult:
    """Audit the contract integrated film energy balance."""

    residual = energy_residual_cumulative(
        t_si=t_si,
        P_in_si=P_in_si,
        q_g_one_sided_si=q_g_one_sided_si,
        T_s_K=T_s_K,
        params=params,
    )
    scale = film_energy_scale(
        t_si=t_si,
        P_in_si=P_in_si,
        q_g_one_sided_si=q_g_one_sided_si,
        T_s_K=T_s_K,
        params=params,
    )
    max_abs = float(np.max(np.abs(residual))) if residual.size else 0.0
    final = float(residual[-1]) if residual.size else 0.0
    return EnergyAuditResult(
        residual_si=residual,
        scale_si=scale,
        max_abs_residual_si=max_abs,
        max_relative_residual=float(max_abs / scale),
        final_residual_si=final,
        tolerance=float(tolerance),
    )


__all__ = [
    "EnergyAuditResult",
    "audit_film_energy",
    "cumulative_trapezoid_abs",
    "film_energy_scale",
]
