"""Film midpoint storage and atomic two-substep gas coupling."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from config_io import load_config
from coupling.drive import SinusoidalDrive
from coupling.film_ode import FilmOdeParams


def test_shared_midpoint_solve_closes_independent_film_storage():
    from coupling.planar_open import solve_film_midpoint

    params = FilmOdeParams(C_A_si=7e-4, linear_leak_conductance_si=57.8)
    temperature = 300.02
    dt = 2e-9
    power = 1255.56
    conductance = 1700.

    def exchange(midpoint):
        return conductance * (midpoint-300.) * dt

    midpoint = solve_film_midpoint(params=params, T_s_K=temperature,
        dt_s=dt, power_W_m2=power, exchange_J_m2=exchange)
    endpoint = 2*midpoint-temperature
    expected = ((2*params.C_A_si*temperature
                 + (2*conductance+57.8)*dt*300. + power*dt)
                / (2*params.C_A_si+(2*conductance+57.8)*dt))
    assert midpoint == pytest.approx(expected, abs=1e-12)
    independent_storage = params.C_A_si*(endpoint-temperature)
    assert abs(independent_storage-power*dt
               + 57.8*(midpoint-300.)*dt+2*exchange(midpoint)) < 1e-14


@pytest.mark.parametrize('dt,power', [(0., 1.), (-1., 1.), (np.nan, 1.),
                                   (1e-9, np.nan), (1e-9, -1e15)])
def test_shared_midpoint_rejects_invalid_or_unbracketed_step(dt, power):
    from coupling.planar_open import solve_film_midpoint

    with pytest.raises((ValueError, FloatingPointError)):
        solve_film_midpoint(params=FilmOdeParams(), T_s_K=300., dt_s=dt,
                            power_W_m2=power, exchange_J_m2=lambda T: 0.)


def make_two_level_solver(backend):
    from core.two_level_planar_solver import TwoLevelPlanarOpenSolver

    config = load_config(Path('configs/gas_air_10k_d2q37_qmatched_local_transport.yaml'))
    config['numerics'].update(nx=1, ny=48, collision_backend=backend,
                              boundary='planar_open_y')
    config['grid'] = dict(type='two_level', ratio=2,
                          interface_m=16*config['lattice']['dx_m'])
    return TwoLevelPlanarOpenSolver(config)


def gas_energy(solver):
    state = solver.get_macro()
    return float(np.sum(state.E_tot[:, 0]*solver.cell_width_m)
                 * solver.mapping.pressure_scale)


@pytest.mark.parametrize('backend', ['numpy', 'numba'])
def test_two_level_film_uses_each_actual_fine_exchange_and_independent_storage(backend):
    from coupling.planar_open import step_two_level_open_film

    solver = make_two_level_solver(backend)
    params = FilmOdeParams(linear_leak_conductance_si=57.8)
    drive = SinusoidalDrive(1255.56, -1255.56, 20000.)
    dt = solver.mapping.lattice.dt_s
    temperature = initial_temperature = 300.02
    initial_energy = gas_energy(solver)
    work = leak = wall = outlet = 0.
    for n in range(80):
        before_temperature = temperature
        before_energy = gas_energy(solver)
        result = step_two_level_open_film(solver=solver, params=params,
            T_s_K=temperature, drive=drive)
        assert len(result.steps) == 2
        for j, step in enumerate(result.steps):
            expected_midpoint = (2*n+j+.5)*dt
            expected_power = 1255.56*(1-np.cos(2*np.pi*20000.*expected_midpoint))
            assert step.heat_time_s == pytest.approx(expected_midpoint, abs=1e-21)
            assert step.power_W_m2 == pytest.approx(expected_power, abs=1e-12)
            assert step.wall_temperature_K == pytest.approx(
                (temperature+step.T_s_K)/2, abs=1e-12)
            expected_work = expected_power*dt
            expected_leak = 57.8*((temperature+step.T_s_K)/2-300.)*dt
            film_storage = params.C_A_si*(step.T_s_K-temperature)
            assert abs(film_storage-expected_work+expected_leak
                       + 2*step.audit.wall_energy_J_m2) < 1e-10
            assert abs(step.audit.wall_mass_lu) < 1e-12
            assert not hasattr(step.audit, 'energy_residual_J_m2')
            temperature = step.T_s_K
            work += expected_work
            leak += expected_leak
            wall += step.audit.wall_energy_J_m2
        assert result.T_s_K == temperature
        assert result.audit.wall_energy_J_m2 == pytest.approx(
            sum(item.audit.wall_energy_J_m2 for item in result.steps), abs=1e-20)
        outlet += result.audit.top_energy_J_m2
        gas_storage = gas_energy(solver)-before_energy
        assert abs(gas_storage-result.audit.wall_energy_J_m2
                   - result.audit.top_energy_J_m2) < 1e-10
        total_storage = params.C_A_si*(temperature-before_temperature)+2*gas_storage
        total_work = sum(s.input_energy_J_m2-s.leak_energy_J_m2 for s in result.steps)
        assert abs(total_storage-total_work-2*result.audit.top_energy_J_m2) < 1e-10
    film_change = params.C_A_si*(temperature-initial_temperature)
    gas_change = gas_energy(solver)-initial_energy
    assert abs(film_change-work+leak+2*wall) < 1e-10
    assert abs(gas_change-wall-outlet) < 1e-10
    assert abs(film_change+2*gas_change-work+leak-2*outlet) < 1e-10
    assert solver.t_lu == 160


@pytest.mark.parametrize('backend', ['numpy', 'numba'])
@pytest.mark.parametrize('failed_power', [np.nan, -1e15])
def test_second_fine_film_failure_rolls_back_complete_macrostep(backend, failed_power):
    from coupling.planar_open import step_two_level_open_film

    solver = make_two_level_solver(backend)
    params = FilmOdeParams(linear_leak_conductance_si=57.8)
    first = step_two_level_open_film(solver=solver, params=params,
        T_s_K=300.02, drive=400.)
    temperature = first.T_s_K
    untouched = deepcopy(solver)
    old_audit = solver.last_audit
    old_substeps = solver.last_substep_audits
    samples = []

    def fail_second(t_s):
        samples.append(t_s)
        return 400. if len(samples) == 1 else failed_power

    with pytest.raises((ValueError, FloatingPointError)):
        result = step_two_level_open_film(solver=solver, params=params,
            T_s_K=temperature, drive=fail_second)
        temperature = result.T_s_K
    assert len(samples) == 2
    assert temperature == first.T_s_K
    np.testing.assert_array_equal(solver.f, untouched.f)
    np.testing.assert_array_equal(solver.g, untouched.g)
    assert solver.t_lu == untouched.t_lu
    assert solver.last_audit is old_audit
    assert solver.last_substep_audits is old_substeps
    resumed = step_two_level_open_film(solver=solver, params=params,
        T_s_K=temperature, drive=400.)
    expected = step_two_level_open_film(solver=untouched, params=params,
        T_s_K=temperature, drive=400.)
    assert resumed == expected
    np.testing.assert_array_equal(solver.f, untouched.f)
    np.testing.assert_array_equal(solver.g, untouched.g)


@pytest.mark.parametrize('backend', ['numpy', 'numba'])
def test_two_level_unforced_film_stays_at_equilibrium(backend):
    from coupling.planar_open import step_two_level_open_film

    solver = make_two_level_solver(backend)
    initial_pressure = solver.get_macro().p.copy()
    temperature = 300.
    for _ in range(24):
        result = step_two_level_open_film(solver=solver, params=FilmOdeParams(),
            T_s_K=temperature, drive=0.)
        temperature = result.T_s_K
    assert abs(temperature-300.) < 1e-10
    np.testing.assert_allclose(solver.get_macro().p, initial_pressure, atol=1e-12, rtol=0.)
