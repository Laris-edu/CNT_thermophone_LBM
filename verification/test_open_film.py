"""Power coupling must exchange actual wall energy without partial steps."""
import numpy as np
import pytest

from coupling.drive import SinusoidalDrive
from coupling.film_ode import FilmOdeParams
from simulation.planar import make_solver


def test_energy_callback_preserves_prescribed_temperature_results():
    left = make_solver(length=40e-6)
    right = make_solver(length=40e-6)
    for n in range(12):
        temperature = 300 + .03 * np.sin(n)
        expected = left.step_temperature(temperature)
        measured = []

        def choose(exchange):
            measured.append(exchange(temperature))
            return temperature

        actual = right.step_wall_energy(choose)
        np.testing.assert_array_equal(right.f, left.f)
        np.testing.assert_array_equal(right.g, left.g)
        assert actual == expected
        assert measured[0] == actual.wall_energy_J_m2


def test_failed_wall_solve_leaves_state_time_and_audit_unchanged():
    solver = make_solver(length=40e-6)
    solver.step_temperature(300.01)
    f, g, time, audit = solver.f.copy(), solver.g.copy(), solver.t_lu, solver.last_audit

    def fail(exchange):
        exchange(300.1)
        exchange(299.9)
        raise RuntimeError('root failed')

    with pytest.raises(RuntimeError, match='root failed'):
        solver.step_wall_energy(fail)
    np.testing.assert_array_equal(solver.f, f)
    np.testing.assert_array_equal(solver.g, g)
    assert solver.t_lu == time
    assert solver.last_audit is audit


def test_nonpositive_trial_gas_state_is_rejected_before_commit(monkeypatch):
    import core.planar_open_solver as module
    solver = make_solver(length=40e-6)
    solver.step_temperature(300.)
    f, g, time, audit = solver.f.copy(), solver.g.copy(), solver.t_lu, solver.last_audit
    original = module.characteristic_top

    def invalid(*args):
        original(*args)
        args[2][-1, 0, 0] = -1e-6

    monkeypatch.setattr(module, 'characteristic_top', invalid)
    with pytest.raises(FloatingPointError, match='nonpositive'):
        solver.step_wall_energy(lambda exchange: 300.)
    np.testing.assert_array_equal(solver.f, f)
    np.testing.assert_array_equal(solver.g, g)
    assert solver.t_lu == time
    assert solver.last_audit is audit


def test_open_film_independent_storage_closes_with_both_outlet_and_leak():
    from coupling.planar_open import step_open_film

    solver = make_solver(length=40e-6)
    mapping = solver.mapping
    params = FilmOdeParams(linear_leak_conductance_si=57.8)
    drive = SinusoidalDrive(1255.56, -1255.56, 20000)
    temperature = 300.02
    initial_temperature = temperature
    initial_energy = np.sum(solver.get_macro().E_tot) * mapping.pressure_scale * mapping.lattice.dx_m
    dt = mapping.lattice.dt_s
    work = leak = wall = outlet = 0.
    for n in range(160):
        old_temperature = temperature
        step = step_open_film(solver=solver, params=params, T_s_K=temperature, drive=drive)
        temperature = step.T_s_K
        # Derive input independently from the specified waveform and midpoint clock.
        expected_power = 1255.56 * (1 - np.cos(2*np.pi*20000*(n+.5)*dt))
        assert step.power_W_m2 == pytest.approx(expected_power, abs=1e-12)
        assert step.wall_temperature_K == pytest.approx((temperature+old_temperature)/2, abs=1e-12)
        work += expected_power * dt
        leak += 57.8 * ((temperature+old_temperature)/2-300) * dt
        wall += step.audit.wall_energy_J_m2
        outlet += step.audit.top_energy_J_m2
        assert abs(step.audit.wall_mass_lu) < 1e-12
    gas_change = np.sum(solver.get_macro().E_tot)*mapping.pressure_scale*mapping.lattice.dx_m-initial_energy
    film_change = 7e-4 * (temperature-initial_temperature)
    assert abs(film_change-work+leak+2*wall) < 1e-10
    assert abs(gas_change-wall-outlet) < 1e-10
    assert abs(film_change+2*gas_change-work+leak-2*outlet) < 1e-10
    assert solver.t_lu == 160


def test_unforced_open_film_remains_at_equilibrium():
    from coupling.planar_open import step_open_film

    solver = make_solver(length=40e-6)
    initial = solver.get_macro().p.copy()
    temperature = 300.
    for _ in range(32):
        step = step_open_film(solver=solver, params=FilmOdeParams(), T_s_K=temperature, drive=0.)
        temperature = step.T_s_K
    assert abs(temperature-300) < 1e-10
    np.testing.assert_allclose(solver.get_macro().p, initial, atol=1e-12, rtol=0)


@pytest.mark.parametrize('power', [float('nan'), -1e15])
def test_invalid_or_unsolvable_power_does_not_commit(power):
    from coupling.planar_open import step_open_film

    solver = make_solver(length=40e-6)
    solver.get_macro()  # Initialize the real solver before taking a rollback snapshot.
    f, g = solver.f.copy(), solver.g.copy()
    with pytest.raises((ValueError, FloatingPointError)):
        step_open_film(solver=solver, params=FilmOdeParams(), T_s_K=300., drive=power)
    np.testing.assert_array_equal(solver.f, f)
    np.testing.assert_array_equal(solver.g, g)
    assert solver.t_lu == 0


def test_small_signal_power_temperature_and_direct_pressure_match_independent_reference():
    from coupling.planar_open import step_open_film
    from postproc.diagnostics import compare, fit_window
    from reference.lim2013 import matched_gas, film_temperature_hat, exact_pressure_hat

    solver=make_solver(refine=2,length=200e-6)
    params=FilmOdeParams(linear_leak_conductance_si=57.8)
    frequency=20000.; amplitude=-50.; dt=solver.mapping.lattice.dt_s
    drive=SinusoidalDrive(0.,amplitude,frequency)
    row=int(np.argmin(abs(solver.z_m-120e-6)))
    temperature=300.; history=[]
    steps=int(np.ceil(6/frequency/dt))
    for n in range(steps):
        step=step_open_film(solver=solver,params=params,T_s_K=temperature,drive=drive)
        temperature=step.T_s_K
        if n % 20 == 0:
            pressure=solver.get_macro().p[row,0]*solver.mapping.pressure_scale
            history.append(((n+1)*dt,temperature,pressure))
    history=np.asarray(history)
    gas=matched_gas(solver.mapping)
    reference_temperature=film_temperature_hat(frequency,amplitude,params.C_A_si,
        beta0_per_side_W_m2K=28.9,gas=gas)
    reference_pressure=exact_pressure_hat(frequency,reference_temperature,solver.z_m[row],gas=gas)
    for column,reference in ((1,reference_temperature),(2,reference_pressure)):
        measured,_=fit_window(history[:,0],history[:,column],frequency,4,6)
        error=compare(measured,reference)
        assert abs(error['amplitude_error_percent']) < 5, error
        assert abs(error['phase_error_deg']) < 5, error
