"""Midpoint film storage coupled to actual diffuse-wall energy exchange."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

from scipy.optimize import brentq

from core.planar_open_solver import OpenStepAudit, PlanarOpenSolver
from coupling.drive import evaluate_drive
from coupling.film_ode import FilmOdeParams

if TYPE_CHECKING:
    from core.two_level_planar_solver import FineWallAudit, TwoLevelPlanarOpenSolver


@dataclass(frozen=True)
class OpenFilmStep:
    T_s_K: float
    wall_temperature_K: float
    heat_time_s: float
    power_W_m2: float
    input_energy_J_m2: float
    leak_energy_J_m2: float
    audit: OpenStepAudit | FineWallAudit


@dataclass(frozen=True)
class TwoLevelOpenFilmStep:
    T_s_K: float
    steps: tuple[OpenFilmStep, OpenFilmStep]
    audit: OpenStepAudit


def _validate_film_state(params, T_s_K):
    values = (T_s_K, params.C_A_si, params.T_ref_K,
              params.linear_leak_conductance_si, params.gas_flux_factor)
    if (not all(math.isfinite(v) for v in values) or T_s_K <= 0
            or params.T_ref_K <= 0 or params.C_A_si <= 0
            or params.linear_leak_conductance_si < 0 or params.gas_flux_factor != 2):
        raise ValueError('finite positive film state and symmetric gas_flux_factor=2 required')


def solve_film_midpoint(*, params: FilmOdeParams, T_s_K: float, dt_s: float,
                        power_W_m2: float, exchange_J_m2) -> float:
    """Solve film storage using the actual one-sided interval gas exchange.

    This pure solve neither advances nor edits a gas state. ``exchange_J_m2``
    maps a trial midpoint wall temperature to energy per area over this step.
    G is the total two-sided additional heat leak.
    """
    _validate_film_state(params, T_s_K)
    if not math.isfinite(dt_s) or dt_s <= 0:
        raise ValueError('film time step must be finite and positive')
    if not math.isfinite(power_W_m2):
        raise ValueError('input power must be finite')
    dt = dt_s
    work = power_W_m2 * dt
    capacity, leak = params.C_A_si, params.linear_leak_conductance_si

    def residual(temperature):
        return (2*capacity*(temperature-T_s_K) + 2*exchange_J_m2(temperature)
                + leak*dt*(temperature-params.T_ref_K) - work)

    center = residual(T_s_K)
    if center == 0:
        return T_s_K
    # A positive endpoint requires T_mid > T_old/2. This is a root
    # domain restriction, never a clipped temperature or population.
    lower_limit = math.nextafter(T_s_K/2, math.inf)
    width = max(.01, abs(work)/(2*capacity))
    for _ in range(48):
        lo = max(lower_limit, T_s_K-width)
        hi = T_s_K+width
        if center > 0:
            a, b = lo, T_s_K
            fa, fb = residual(a), center
        else:
            a, b = T_s_K, hi
            fa, fb = center, residual(b)
        if not (math.isfinite(fa) and math.isfinite(fb)):
            raise FloatingPointError('nonfinite film energy residual')
        if fa <= 0 <= fb:
            root = brentq(residual, a, b, xtol=1e-12, rtol=1e-15, maxiter=100)
            if not math.isfinite(2*root-T_s_K) or 2*root-T_s_K <= 0:
                raise FloatingPointError('film solve produced a nonpositive endpoint')
            return root
        if center > 0 and lo == lower_limit:
            break
        width *= 2
    raise FloatingPointError('could not bracket a positive-temperature film solution')


def step_open_film(*, solver: PlanarOpenSolver, params: FilmOdeParams,
                   T_s_K: float, drive) -> OpenFilmStep:
    """Advance one symmetric two-sided film step and one upper gas-domain step.

    The caller replaces its endpoint film temperature only after this returns.
    No analytical heat admittance or row heat-flux moment enters this solve.
    """
    _validate_film_state(params, T_s_K)
    dt = solver.mapping.lattice.dt_s
    midpoint = (solver.t_lu + .5) * dt
    power = evaluate_drive(drive, midpoint)
    if not math.isfinite(power):
        raise ValueError('input power must be finite')
    root = None

    def solve(exchange):
        nonlocal root
        root = solve_film_midpoint(params=params, T_s_K=T_s_K, dt_s=dt,
                                  power_W_m2=power, exchange_J_m2=exchange)
        return root

    audit = solver.step_wall_energy(solve)
    return OpenFilmStep(2*root-T_s_K, root, midpoint, power, power*dt,
        params.linear_leak_conductance_si*(root-params.T_ref_K)*dt, audit)


def step_two_level_open_film(*, solver: TwoLevelPlanarOpenSolver,
                             params: FilmOdeParams, T_s_K: float,
                             drive) -> TwoLevelOpenFilmStep:
    """Atomically advance two fine film intervals and one gas synchronization.

    Fine records contain actual wall exchange only; full gas storage and outlet
    exchange belong to the synchronized ``audit``. Temporary film endpoints are
    local until the solver has accepted the entire macrostep.
    """
    _validate_film_state(params, T_s_K)
    dt = solver.mapping.lattice.dt_s
    trial_temperature = T_s_K
    records = []

    def solve(exchange, substep_index, midpoint_time_s):
        nonlocal trial_temperature
        power = evaluate_drive(drive, midpoint_time_s)
        root = solve_film_midpoint(params=params, T_s_K=trial_temperature,
            dt_s=dt, power_W_m2=power, exchange_J_m2=exchange)
        trial_temperature = 2*root-trial_temperature
        records.append((trial_temperature, root, midpoint_time_s, power,
            power*dt, params.linear_leak_conductance_si*(root-params.T_ref_K)*dt))
        return root

    audit = solver.step_wall_energy(solve)
    steps = tuple(OpenFilmStep(*record, wall_audit)
        for record, wall_audit in zip(records, solver.last_substep_audits, strict=True))
    return TwoLevelOpenFilmStep(trial_temperature, steps, audit)
