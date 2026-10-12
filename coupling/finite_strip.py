"""Spatial midpoint film coupling on a uniform strip with adiabatic ends.

Temperatures live at film-bin centres. Power is W/m^2, sheet conductance
is k*h in W/K, and the callback supplies actual one-sided interval wall
exchange in J/m^2 for each bin. No gas heat-flux moment or reference
admittance is substituted for that exchange.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np
from scipy.linalg import solve, solve_banded

from coupling.film_ode import FilmOdeParams


@dataclass(frozen=True)
class StripFilmEnergyAudit:
    """Independent interval totals per unit out-of-plane length, in J/m.

    Positive gas and leak energies leave the film. End conduction is
    positive into the film and is zero for this adiabatic-end interface.
    The residual is storage + gas + leak - input - end conduction.
    """

    storage_change_J_m: float
    input_energy_J_m: float
    gas_energy_J_m: float
    leak_energy_J_m: float
    end_conduction_energy_J_m: float
    residual_J_m: float


def _temperature_array(temperature, *, name):
    values = np.asarray(temperature, dtype=float)
    if (values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values))
            or np.any(values <= 0)):
        raise ValueError(f'{name} must be a nonempty finite positive 1D temperature array')
    return values.copy()


def _validate_params(params, dx_m, dt_s):
    values = (params.C_A_si, params.T_ref_K, params.linear_leak_conductance_si,
              params.gas_flux_factor, dx_m, dt_s)
    if (not all(math.isfinite(value) for value in values)
            or params.C_A_si <= 0 or params.T_ref_K <= 0
            or params.linear_leak_conductance_si < 0
            or params.gas_flux_factor != 2 or dx_m <= 0 or dt_s <= 0):
        raise ValueError('finite positive film capacity, reference, dx, dt; '
                         'nonnegative leak and symmetric gas_flux_factor=2 required')


def _bin_values(value, shape, *, name, allow_scalar=False):
    values = np.asarray(value, dtype=float)
    if allow_scalar and values.ndim == 0:
        values = np.full(shape, float(values))
    if values.shape != shape:
        raise ValueError(f'{name} must have the film temperature shape {shape}')
    if not np.all(np.isfinite(values)):
        raise ValueError(f'{name} must be finite')
    return values.copy()


def _conduction_W_m2(temperature, conductance, dx):
    """Conservative internal face flux differences; no external end flux."""
    heating = np.zeros_like(temperature)
    face_flux = conductance*np.diff(temperature)/dx
    heating[:-1] += face_flux/dx
    heating[1:] -= face_flux/dx
    return heating


def solve_strip_film_midpoint(*, params: FilmOdeParams, T_s_K: np.ndarray,
        dx_m: float, sheet_conductance_W_K: float, dt_s: float,
        power_W_m2: float | np.ndarray,
        exchange_J_m2: Callable[[np.ndarray], np.ndarray],
        exchange_is_local: bool = False) -> np.ndarray:
    """Return midpoint temperatures without advancing or editing a gas state.

    The endpoint is ``2*T_mid-T_s_K``. The exchange callback must be pure:
    trials refer to the same prepared gas interval and must not commit it.
    Negative exchange (gas heating the film) is allowed. Power is the value
    at the time midpoint, evaluated by the caller using ``evaluate_drive``
    for a time-dependent drive. G in FilmOdeParams is an explicitly chosen
    total two-sided leak; its existing default is zero.

    Setting ``exchange_is_local=True`` asserts each returned bin depends
    only on its own trial temperature. This enables simultaneous finite
    differences and a tridiagonal Newton solve, O(N) per iteration. The
    default supports a general nonlocal callback with a dense Jacobian.
    A false locality assertion is not automatically detected.

    Newton backtracking restricts trials to positive midpoint and endpoint
    temperatures; it never clips temperatures or the returned exchange.
    """
    _validate_params(params, dx_m, dt_s)
    old = _temperature_array(T_s_K, name='T_s_K')
    power = _bin_values(power_W_m2, old.shape, name='power_W_m2', allow_scalar=True)
    if (not math.isfinite(sheet_conductance_W_K) or sheet_conductance_W_K < 0):
        raise ValueError('sheet conductance must be finite and nonnegative')
    if not isinstance(exchange_is_local, (bool, np.bool_)):
        raise ValueError('exchange_is_local must be boolean')
    work = power*dt_s
    if not np.all(np.isfinite(work)):
        raise ValueError('input interval energy must be finite')
    capacity, leak = params.C_A_si, params.linear_leak_conductance_si
    n = old.size
    ratio = dt_s*sheet_conductance_W_K/dx_m**2
    if not math.isfinite(ratio):
        raise ValueError('film conduction coefficient must be finite')

    def exchange(temperature):
        return _bin_values(exchange_J_m2(temperature.copy()), old.shape,
                           name='one-sided interval wall exchange')

    def residual(temperature, wall_energy):
        storage = 2*capacity*(temperature-old)
        leakage = leak*dt_s*(temperature-params.T_ref_K)
        conduction = dt_s*_conduction_W_m2(temperature, sheet_conductance_W_K, dx_m)
        value = storage+2*wall_energy+leakage-work-conduction
        if not np.all(np.isfinite(value)):
            raise FloatingPointError('nonfinite strip film energy residual')
        # Tolerance applies to interval energy, never to a normalized film RHS.
        scale = max(float(np.max(np.abs(item)))
                    for item in (storage, 2*wall_energy, leakage, work, conduction))
        return value, 1e-13+1e-10*scale

    diagonal = np.full(n, 2*capacity+leak*dt_s)
    if n > 1:
        diagonal[:-1] += ratio
        diagonal[1:] += ratio
    midpoint = old.copy()
    wall = exchange(midpoint)
    value, tolerance = residual(midpoint, wall)
    for _ in range(40):
        norm = float(np.max(np.abs(value)))
        if norm <= tolerance:
            return midpoint
        perturbation = np.sqrt(np.finfo(float).eps)*np.maximum(1., np.abs(midpoint))
        if exchange_is_local:
            perturbed = midpoint+perturbation
            derivative = (exchange(perturbed)-wall)/(perturbed-midpoint)
            banded = np.zeros((3, n))
            banded[1] = diagonal+2*derivative
            banded[0, 1:] = -ratio
            banded[2, :-1] = -ratio
            try:
                direction = solve_banded((1, 1), banded, -value)
            except np.linalg.LinAlgError as error:
                raise FloatingPointError('singular strip film Jacobian') from error
        else:
            jacobian = np.diag(diagonal)
            if n > 1:
                indexes = np.arange(n-1)
                jacobian[indexes, indexes+1] = -ratio
                jacobian[indexes+1, indexes] = -ratio
            for column in range(n):
                perturbed = midpoint.copy()
                perturbed[column] += perturbation[column]
                derivative = (exchange(perturbed)-wall)/(perturbed[column]-midpoint[column])
                jacobian[:, column] += 2*derivative
            try:
                direction = solve(jacobian, -value)
            except np.linalg.LinAlgError as error:
                raise FloatingPointError('singular strip film Jacobian') from error
        if not np.all(np.isfinite(direction)):
            raise FloatingPointError('nonfinite strip film Newton direction')
        fraction = 1.
        for _ in range(48):
            trial = midpoint+fraction*direction
            if (np.all(np.isfinite(trial)) and np.all(2*trial-old > 0)
                    and np.all(trial > 0)):
                try:
                    trial_wall = exchange(trial)
                    trial_value, trial_tolerance = residual(trial, trial_wall)
                except FloatingPointError:
                    fraction *= .5
                    continue
                trial_norm = float(np.max(np.abs(trial_value)))
                if (trial_norm <= trial_tolerance
                        or trial_norm < (1-1e-4*fraction)*norm):
                    midpoint, wall = trial, trial_wall
                    value, tolerance = trial_value, trial_tolerance
                    break
            fraction *= .5
        else:
            raise FloatingPointError('could not converge to a positive-temperature strip film solution')
    raise FloatingPointError('strip film midpoint did not converge')


def audit_strip_film_step(*, params: FilmOdeParams, T_old_K: np.ndarray,
        T_new_K: np.ndarray, dx_m: float, dt_s: float,
        power_W_m2: float | np.ndarray,
        wall_energy_J_m2: np.ndarray) -> StripFilmEnergyAudit:
    """Compare endpoint storage with supplied actual interval exchanges.

    Pass wall energies measured by the accepted gas boundary step, not an
    energy reconstructed from the film equation. Internal conductive faces
    cancel in the global balance; the two end faces are adiabatic. This
    checks film storage only, not independent storage in the gas domain.
    """
    _validate_params(params, dx_m, dt_s)
    old = _temperature_array(T_old_K, name='T_old_K')
    new = _temperature_array(T_new_K, name='T_new_K')
    if new.shape != old.shape:
        raise ValueError('film endpoint temperatures must have the same shape')
    power = _bin_values(power_W_m2, old.shape, name='power_W_m2', allow_scalar=True)
    wall = _bin_values(wall_energy_J_m2, old.shape, name='wall_energy_J_m2')
    storage = params.C_A_si*dx_m*math.fsum(new-old)
    work = dt_s*dx_m*math.fsum(power)
    gas = 2*dx_m*math.fsum(wall)
    leak = params.linear_leak_conductance_si*dt_s*dx_m*math.fsum(
        .5*old+.5*new-params.T_ref_K)
    end_conduction = 0.
    residual = math.fsum((storage, gas, leak, -work, -end_conduction))
    if not all(math.isfinite(item) for item in (storage, work, gas, leak, residual)):
        raise FloatingPointError('nonfinite strip film energy audit')
    return StripFilmEnergyAudit(storage, work, gas, leak, end_conduction, residual)
