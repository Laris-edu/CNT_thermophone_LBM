"""Standalone Phase_3 thin-film ODE utilities."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Literal

import numpy as np

from .drive import DriveSignal, evaluate_drive

ScalarOrSignal = float | Callable[[float], float]
StepMethod = Literal["euler", "heun", "rk4"]


@dataclass(frozen=True)
class FilmOdeParams:
    """Thin-film areal heat-capacity and standalone leak parameters.

    The Phase_3 gas heat flux ``q_g''`` is one-sided by contract; the default
    ``gas_flux_factor=2`` applies the freestanding two-sided air-film factor.
    ``linear_leak_conductance_si`` is a standalone fixture term with units
    W/(m^2 K) and is already the total effective leak.
    """

    C_A_si: float = 7.0e-4
    T_ref_K: float = 300.0
    linear_leak_conductance_si: float = 0.0
    gas_flux_factor: float = 2.0

    def __post_init__(self) -> None:
        if self.C_A_si <= 0.0:
            raise ValueError("C_A_si must be positive")
        if self.linear_leak_conductance_si < 0.0:
            raise ValueError("linear_leak_conductance_si must be non-negative")
        if self.gas_flux_factor < 0.0:
            raise ValueError("gas_flux_factor must be non-negative")


@dataclass(frozen=True)
class FilmTrajectory:
    """Time history from a standalone film ODE integration."""

    t_si: np.ndarray
    T_s_K: np.ndarray
    P_in_si: np.ndarray
    q_g_one_sided_si: np.ndarray
    dT_s_dt_K_s: np.ndarray
    ode_pointwise_residual_si: np.ndarray
    energy_residual_si: np.ndarray


def _eval_signal(signal: ScalarOrSignal, t_si: float) -> float:
    if callable(signal):
        return float(signal(float(t_si)))
    return float(signal)


def film_rhs(
    T_s_K: float,
    t_si: float,
    *,
    params: FilmOdeParams,
    drive: DriveSignal | float,
    q_g_one_sided_si: ScalarOrSignal = 0.0,
) -> float:
    """Return ``dT_s/dt`` for the standalone film energy equation."""

    P_in = evaluate_drive(drive, t_si)
    q_g = _eval_signal(q_g_one_sided_si, t_si)
    leak = params.linear_leak_conductance_si * (float(T_s_K) - params.T_ref_K)
    return float((P_in - params.gas_flux_factor * q_g - leak) / params.C_A_si)


def ode_pointwise_residual(
    *,
    P_in_si: np.ndarray,
    q_g_one_sided_si: np.ndarray,
    T_s_K: np.ndarray,
    dT_s_dt_K_s: np.ndarray,
    params: FilmOdeParams,
) -> np.ndarray:
    """Pointwise rhs-consistency residual ``P_in - factor*q_g - leak - C_A dT_s/dt`` (W/m^2).

    This only checks that ``dT_s_dt_K_s`` matches the rhs at each sample. When the
    derivative is the analytic :func:`film_rhs` (as produced by :func:`integrate_film_ode`)
    it is ~0 by construction and does **not** measure integration accuracy or energy
    conservation. Use :func:`integrated_energy_residual` for the contract §7.3 audit.
    """

    P_in = np.asarray(P_in_si, dtype=float)
    q_g = np.asarray(q_g_one_sided_si, dtype=float)
    T_s = np.asarray(T_s_K, dtype=float)
    dTdt = np.asarray(dT_s_dt_K_s, dtype=float)
    leak = params.linear_leak_conductance_si * (T_s - params.T_ref_K)
    return P_in - params.gas_flux_factor * q_g - leak - params.C_A_si * dTdt


def _cumulative_trapezoid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(y, dtype=float)
    if y.size >= 2:
        out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))
    return out


def energy_residual_cumulative(
    *,
    t_si: np.ndarray,
    P_in_si: np.ndarray,
    q_g_one_sided_si: np.ndarray,
    T_s_K: np.ndarray,
    params: FilmOdeParams,
) -> np.ndarray:
    """Contract §7.3 running energy balance ``R_E(t0, t_i)`` in W/m^2 * s.

    ``R_E(t0,t_i) = integral_{t0}^{t_i}(P_in - factor*q_g - leak) dt - C_A*(T_s_i - T_s_0)``.

    Unlike :func:`ode_pointwise_residual`, this integrates the actual *stepped* trajectory,
    so it grows when the integrator drifts; ``max|R_E| / input_scale < 1%`` is the energy
    audit. This is the non-vacuous series stored as the ``/film/energy_residual_si`` field.
    """

    t = np.asarray(t_si, dtype=float)
    P_in = np.asarray(P_in_si, dtype=float)
    q_g = np.asarray(q_g_one_sided_si, dtype=float)
    T_s = np.asarray(T_s_K, dtype=float)
    leak = params.linear_leak_conductance_si * (T_s - params.T_ref_K)
    net = P_in - params.gas_flux_factor * q_g - leak
    return _cumulative_trapezoid(net, t) - params.C_A_si * (T_s - T_s[0])


def integrated_energy_residual(
    *,
    t_si: np.ndarray,
    P_in_si: np.ndarray,
    q_g_one_sided_si: np.ndarray,
    T_s_K: np.ndarray,
    params: FilmOdeParams,
) -> float:
    """Scalar net energy imbalance over the whole window (= cumulative residual at ``t[-1]``)."""

    series = energy_residual_cumulative(
        t_si=t_si,
        P_in_si=P_in_si,
        q_g_one_sided_si=q_g_one_sided_si,
        T_s_K=T_s_K,
        params=params,
    )
    return float(series[-1]) if series.size else 0.0


def euler_step(
    T_s_K: float,
    t_si: float,
    dt_si: float,
    *,
    params: FilmOdeParams,
    drive: DriveSignal | float,
    q_g_one_sided_si: ScalarOrSignal = 0.0,
) -> float:
    return float(
        T_s_K
        + dt_si
        * film_rhs(
            T_s_K,
            t_si,
            params=params,
            drive=drive,
            q_g_one_sided_si=q_g_one_sided_si,
        )
    )


def heun_step(
    T_s_K: float,
    t_si: float,
    dt_si: float,
    *,
    params: FilmOdeParams,
    drive: DriveSignal | float,
    q_g_one_sided_si: ScalarOrSignal = 0.0,
) -> float:
    k1 = film_rhs(
        T_s_K,
        t_si,
        params=params,
        drive=drive,
        q_g_one_sided_si=q_g_one_sided_si,
    )
    T_predict = T_s_K + dt_si * k1
    k2 = film_rhs(
        T_predict,
        t_si + dt_si,
        params=params,
        drive=drive,
        q_g_one_sided_si=q_g_one_sided_si,
    )
    return float(T_s_K + 0.5 * dt_si * (k1 + k2))


def rk4_step(
    T_s_K: float,
    t_si: float,
    dt_si: float,
    *,
    params: FilmOdeParams,
    drive: DriveSignal | float,
    q_g_one_sided_si: ScalarOrSignal = 0.0,
) -> float:
    def rhs(T_value: float, t_value: float) -> float:
        return film_rhs(
            T_value,
            t_value,
            params=params,
            drive=drive,
            q_g_one_sided_si=q_g_one_sided_si,
        )

    k1 = rhs(T_s_K, t_si)
    k2 = rhs(T_s_K + 0.5 * dt_si * k1, t_si + 0.5 * dt_si)
    k3 = rhs(T_s_K + 0.5 * dt_si * k2, t_si + 0.5 * dt_si)
    k4 = rhs(T_s_K + dt_si * k3, t_si + dt_si)
    return float(T_s_K + (dt_si / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))


def integrate_film_ode(
    *,
    t_si: np.ndarray,
    T_initial_K: float,
    params: FilmOdeParams,
    drive: DriveSignal | float,
    q_g_one_sided_si: ScalarOrSignal = 0.0,
    method: StepMethod = "rk4",
) -> FilmTrajectory:
    """Integrate the standalone film ODE on a supplied monotone time grid."""

    t = np.asarray(t_si, dtype=float)
    if t.ndim != 1 or t.size < 1:
        raise ValueError("t_si must be a one-dimensional array with at least one item")
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("t_si must be strictly increasing")

    stepper = {
        "euler": euler_step,
        "heun": heun_step,
        "rk4": rk4_step,
    }.get(method)
    if stepper is None:
        raise ValueError(f"unsupported film ODE step method: {method}")

    T = np.empty_like(t, dtype=float)
    T[0] = float(T_initial_K)
    for i in range(t.size - 1):
        dt = float(t[i + 1] - t[i])
        T[i + 1] = stepper(
            float(T[i]),
            float(t[i]),
            dt,
            params=params,
            drive=drive,
            q_g_one_sided_si=q_g_one_sided_si,
        )

    P_in = np.asarray([evaluate_drive(drive, item) for item in t], dtype=float)
    q_g = np.asarray([_eval_signal(q_g_one_sided_si, item) for item in t], dtype=float)
    dTdt = np.asarray(
        [
            film_rhs(
                float(T_value),
                float(t_value),
                params=params,
                drive=drive,
                q_g_one_sided_si=q_g_one_sided_si,
            )
            for T_value, t_value in zip(T, t, strict=True)
        ],
        dtype=float,
    )
    pointwise_residual = ode_pointwise_residual(
        P_in_si=P_in,
        q_g_one_sided_si=q_g,
        T_s_K=T,
        dT_s_dt_K_s=dTdt,
        params=params,
    )
    cumulative_residual = energy_residual_cumulative(
        t_si=t,
        P_in_si=P_in,
        q_g_one_sided_si=q_g,
        T_s_K=T,
        params=params,
    )
    return FilmTrajectory(
        t_si=t,
        T_s_K=T,
        P_in_si=P_in,
        q_g_one_sided_si=q_g,
        dT_s_dt_K_s=dTdt,
        ode_pointwise_residual_si=pointwise_residual,
        energy_residual_si=cumulative_residual,
    )


def adiabatic_ramp_solution(
    t_si: np.ndarray,
    *,
    T_initial_K: float,
    P_constant_si: float,
    C_A_si: float,
    t_start_si: float = 0.0,
) -> np.ndarray:
    """Closed-form adiabatic ramp: ``T_s = T0 + P t / C_A``."""

    t = np.asarray(t_si, dtype=float)
    return float(T_initial_K) + (float(P_constant_si) / float(C_A_si)) * (t - t_start_si)


def linear_leak_step_solution(
    t_si: np.ndarray,
    *,
    T_initial_K: float,
    P_constant_si: float,
    params: FilmOdeParams,
    t_start_si: float = 0.0,
) -> np.ndarray:
    """Closed-form step response with a total effective linear leak."""

    G = params.linear_leak_conductance_si
    if G == 0.0:
        return adiabatic_ramp_solution(
            t_si,
            T_initial_K=T_initial_K,
            P_constant_si=P_constant_si,
            C_A_si=params.C_A_si,
            t_start_si=t_start_si,
        )
    t = np.asarray(t_si, dtype=float)
    tau = params.C_A_si / G
    T_equilibrium = params.T_ref_K + float(P_constant_si) / G
    return T_equilibrium + (float(T_initial_K) - T_equilibrium) * np.exp(-(t - t_start_si) / tau)


def sinusoidal_temperature_hat(
    *,
    params: FilmOdeParams,
    frequency_hz: float,
    P_hat_si: complex,
    q_g_hat_one_sided_si: complex = 0.0j,
) -> complex:
    """Closed-form periodic ``T_s_hat`` for harmonic forcing."""

    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")
    omega = 2.0 * math.pi * float(frequency_hz)
    denominator = params.linear_leak_conductance_si + 1j * omega * params.C_A_si
    if denominator == 0.0:
        raise ValueError("periodic response is undefined for zero frequency and zero leak")
    numerator = complex(P_hat_si) - params.gas_flux_factor * complex(q_g_hat_one_sided_si)
    return complex(numerator / denominator)


def sinusoidal_mean_temperature(
    *,
    params: FilmOdeParams,
    mean_power_density_si: float = 0.0,
    q_g_mean_one_sided_si: float = 0.0,
) -> float:
    """Return the bounded periodic mean temperature for the linear fixture."""

    mean_net = float(mean_power_density_si) - params.gas_flux_factor * float(q_g_mean_one_sided_si)
    if params.linear_leak_conductance_si == 0.0:
        if abs(mean_net) > 0.0:
            raise ValueError("non-zero mean forcing without leak creates a ramp, not a periodic mean")
        return float(params.T_ref_K)
    return float(params.T_ref_K + mean_net / params.linear_leak_conductance_si)


def sinusoidal_steady_temperature(
    t_si: np.ndarray,
    *,
    params: FilmOdeParams,
    frequency_hz: float,
    P_hat_si: complex,
    mean_power_density_si: float = 0.0,
    q_g_hat_one_sided_si: complex = 0.0j,
    q_g_mean_one_sided_si: float = 0.0,
) -> np.ndarray:
    """Sample the closed-form steady periodic film temperature."""

    t = np.asarray(t_si, dtype=float)
    omega = 2.0 * math.pi * float(frequency_hz)
    T_mean = sinusoidal_mean_temperature(
        params=params,
        mean_power_density_si=mean_power_density_si,
        q_g_mean_one_sided_si=q_g_mean_one_sided_si,
    )
    T_hat = sinusoidal_temperature_hat(
        params=params,
        frequency_hz=frequency_hz,
        P_hat_si=P_hat_si,
        q_g_hat_one_sided_si=q_g_hat_one_sided_si,
    )
    phase = np.exp(1j * omega * t)
    return np.asarray(T_mean + np.real(T_hat * phase), dtype=float)


__all__ = [
    "FilmOdeParams",
    "FilmTrajectory",
    "adiabatic_ramp_solution",
    "energy_residual_cumulative",
    "euler_step",
    "film_rhs",
    "heun_step",
    "integrate_film_ode",
    "integrated_energy_residual",
    "linear_leak_step_solution",
    "ode_pointwise_residual",
    "rk4_step",
    "sinusoidal_mean_temperature",
    "sinusoidal_steady_temperature",
    "sinusoidal_temperature_hat",
]
