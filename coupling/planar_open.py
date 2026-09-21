"""Midpoint film storage coupled to actual diffuse-wall energy exchange."""
from dataclasses import dataclass
import math

from scipy.optimize import brentq

from core.planar_open_solver import OpenStepAudit, PlanarOpenSolver
from coupling.drive import evaluate_drive
from coupling.film_ode import FilmOdeParams


@dataclass(frozen=True)
class OpenFilmStep:
    T_s_K: float
    wall_temperature_K: float
    heat_time_s: float
    power_W_m2: float
    input_energy_J_m2: float
    leak_energy_J_m2: float
    audit: OpenStepAudit


def step_open_film(*, solver: PlanarOpenSolver, params: FilmOdeParams,
                   T_s_K: float, drive) -> OpenFilmStep:
    """Advance one symmetric two-sided film step and one upper gas-domain step.

    Input power is per projected area; G is the total additional heat leak.
    The caller replaces its endpoint film temperature only after this returns.
    No analytical heat admittance or row heat-flux moment enters this solve.
    """
    values = (T_s_K, params.C_A_si, params.T_ref_K,
              params.linear_leak_conductance_si, params.gas_flux_factor)
    if (not all(math.isfinite(v) for v in values) or T_s_K <= 0
            or params.T_ref_K <= 0 or params.C_A_si <= 0
            or params.linear_leak_conductance_si < 0 or params.gas_flux_factor != 2):
        raise ValueError('finite positive film state and symmetric gas_flux_factor=2 required')
    dt = solver.mapping.lattice.dt_s
    midpoint = (solver.t_lu + .5) * dt
    power = evaluate_drive(drive, midpoint)
    if not math.isfinite(power):
        raise ValueError('input power must be finite')
    work = power * dt
    capacity, leak = params.C_A_si, params.linear_leak_conductance_si
    root = None

    def solve(exchange):
        nonlocal root

        def residual(temperature):
            return (2*capacity*(temperature-T_s_K) + 2*exchange(temperature)
                    + leak*dt*(temperature-params.T_ref_K) - work)

        center = residual(T_s_K)
        if center == 0:
            root = T_s_K
            return root
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

    audit = solver.step_wall_energy(solve)
    return OpenFilmStep(2*root-T_s_K, root, midpoint, power, work,
                        leak*(root-params.T_ref_K)*dt, audit)
