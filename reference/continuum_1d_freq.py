"""Frequency-domain Phase 1 1D thermal-acoustic reference solver."""

from __future__ import annotations

import math
import numpy as np

from .analytical_models import pressure_profile_from_exponential_temperature
from .constants import (
    DEFAULT_PROBES_OVER_DELTA_T,
    PhysicalParams,
    default_params,
    omega_from_frequency,
    thermal_scales,
)
from .film_ode import level_c_closed_form
from .result_schema import ReferenceResult
from .thermal_admittance import (
    heat_flux_from_wall_temperature,
    temperature_profile,
    thermal_wavenumber,
    wall_temperature_from_heat_flux,
)


def _grid_metadata(delta_T: float, Ly_over_deltaT: float, dy_over_deltaT: float) -> tuple[int, float, float]:
    L_y = Ly_over_deltaT * delta_T
    dy = dy_over_deltaT * delta_T
    N_y = int(math.ceil(L_y / dy)) + 1
    return N_y, L_y, dy


def _make_result(
    *,
    case_name: str,
    level: str,
    f_hz: float,
    params: PhysicalParams,
    C_A: float,
    beta0: float,
    P_hat: complex | None,
    T_s_hat_input: complex | None,
    q_hat_input: complex | None,
    T_s_hat: complex,
    q_g_hat: complex,
    energy_residual_hat: complex,
    energy_residual_rel: float,
    probes_y_over_delta_T: tuple[float, ...] = DEFAULT_PROBES_OVER_DELTA_T,
    Ly_over_deltaT: float = 15.0,
    dy_over_deltaT: float = 1.0 / 30.0,
    solver_name: str = "semi_analytic_thermal_acoustic",
    metadata: dict[str, object] | None = None,
) -> ReferenceResult:
    scales = thermal_scales(f_hz, params, P_hat=abs(P_hat or 1000.0), C_A=C_A)
    delta_T = scales["delta_T"]
    probes_over = np.asarray(probes_y_over_delta_T, dtype=float)
    probes_y = probes_over * delta_T
    T_hat = temperature_profile(probes_y, f_hz, T_s_hat, params)
    p_hat, u_hat = pressure_profile_from_exponential_temperature(f_hz, T_s_hat, probes_y, params)
    N_y, L_y, dy = _grid_metadata(delta_T, Ly_over_deltaT, dy_over_deltaT)
    return ReferenceResult(
        case_name=case_name,
        level=level,
        f_Hz=float(f_hz),
        Omega=omega_from_frequency(f_hz),
        T0=params.T0,
        p0=params.p0,
        rho0=params.rho0,
        gamma=params.gamma,
        cp=params.cp,
        kg=params.kg,
        alpha0=params.alpha0,
        nu0=params.nu0,
        Pr=params.Pr,
        C_A=C_A,
        beta0=beta0,
        P_hat=P_hat,
        T_s_hat_input=T_s_hat_input,
        q_hat_input=q_hat_input,
        m_T=thermal_wavenumber(f_hz, params),
        delta_T=delta_T,
        delta_v=scales["delta_v"],
        Pi_C=scales["Pi_C"],
        epsilon_P=scales["epsilon_P"],
        k_delta_T=scales["k_delta_T"],
        k_a=scales["k_a"],
        T_s_hat=complex(T_s_hat),
        q_g_hat=complex(q_g_hat),
        probes_y=probes_y,
        probes_y_over_delta_T=probes_over,
        p_hat_probes=p_hat,
        T_hat_probes=T_hat,
        u_hat_probes=u_hat,
        energy_residual_hat=complex(energy_residual_hat),
        energy_residual_rel=float(energy_residual_rel),
        solver_name=solver_name,
        N_y=N_y,
        L_y=L_y,
        dy_min=dy,
        dy_max=dy,
        bc_far_pressure="outgoing_exp(-ik y)",
        bc_far_temperature="T_hat(L)=0; Ly=15 delta_T compact thermal domain",
        metadata=metadata or {},
    )


def solve_level_A_frequency(
    f_hz: float,
    T_s_hat: complex,
    params: PhysicalParams | None = None,
    **kwargs: object,
) -> ReferenceResult:
    params = params or default_params()
    q_hat = heat_flux_from_wall_temperature(f_hz, T_s_hat, params)
    return _make_result(
        case_name=f"phase1_levelA_{float(f_hz):g}Hz",
        level="A",
        f_hz=f_hz,
        params=params,
        C_A=params.C_A,
        beta0=params.beta0,
        P_hat=None,
        T_s_hat_input=T_s_hat,
        q_hat_input=None,
        T_s_hat=T_s_hat,
        q_g_hat=q_hat,
        energy_residual_hat=0.0 + 0.0j,
        energy_residual_rel=0.0,
        metadata={"thermal_bc": "prescribed_wall_temperature"},
        **kwargs,
    )


def solve_level_B_frequency(
    f_hz: float,
    q_hat: complex,
    params: PhysicalParams | None = None,
    **kwargs: object,
) -> ReferenceResult:
    params = params or default_params()
    T_s_hat = wall_temperature_from_heat_flux(f_hz, q_hat, params)
    return _make_result(
        case_name=f"phase1_levelB_{float(f_hz):g}Hz",
        level="B",
        f_hz=f_hz,
        params=params,
        C_A=params.C_A,
        beta0=params.beta0,
        P_hat=None,
        T_s_hat_input=None,
        q_hat_input=q_hat,
        T_s_hat=T_s_hat,
        q_g_hat=q_hat,
        energy_residual_hat=0.0 + 0.0j,
        energy_residual_rel=0.0,
        metadata={"thermal_bc": "prescribed_one_sided_heat_flux"},
        **kwargs,
    )


def solve_level_C_frequency(
    f_hz: float,
    P_hat: complex,
    C_A: float | None = None,
    params: PhysicalParams | None = None,
    *,
    beta0: float | None = None,
    **kwargs: object,
) -> ReferenceResult:
    params = params or default_params()
    C_A = params.C_A if C_A is None else C_A
    beta0 = params.beta0 if beta0 is None else beta0
    closed = level_c_closed_form(f_hz, P_hat, C_A=C_A, params=params, beta0=beta0)
    return _make_result(
        case_name=f"phase1_levelC_{float(f_hz):g}Hz",
        level="C",
        f_hz=f_hz,
        params=params,
        C_A=C_A,
        beta0=beta0,
        P_hat=P_hat,
        T_s_hat_input=None,
        q_hat_input=None,
        T_s_hat=closed.T_s_hat,
        q_g_hat=closed.q_g_hat,
        energy_residual_hat=closed.energy_residual_hat,
        energy_residual_rel=closed.energy_residual_rel,
        metadata={"thermal_bc": "film_ode_conjugate", "double_sided": True},
        **kwargs,
    )

