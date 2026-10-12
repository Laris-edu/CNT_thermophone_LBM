"""Finite strip storage uses local interval exchanges and conservative conduction."""
import importlib.util

import numpy as np
import pytest

from coupling.film_ode import FilmOdeParams


def _interface():
    assert importlib.util.find_spec('coupling.finite_strip') is not None, (
        'finite strip film coupling is not implemented')
    from coupling.finite_strip import audit_strip_film_step, solve_strip_film_midpoint
    return solve_strip_film_midpoint, audit_strip_film_step


def test_local_actual_exchange_has_two_sides_without_spatial_averaging():
    solve, audit_step = _interface()
    params = FilmOdeParams(C_A_si=.012888)
    old = np.full(4, 300.)
    original = old.copy()
    exchange = np.array([.001, -.002, .003, .004])
    power = np.array([100., 120., 80., 110.])
    dt, dx = 1e-4, 2.5e-4
    mid = solve(params=params, T_s_K=old, dx_m=dx,
                sheet_conductance_W_K=0., dt_s=dt,
                power_W_m2=power, exchange_J_m2=lambda _: exchange)
    new = 2*mid-old
    expected = old+(power*dt-2*exchange)/params.C_A_si
    np.testing.assert_allclose(new, expected, rtol=0, atol=1e-11)
    np.testing.assert_array_equal(old, original)
    audit = audit_step(params=params, T_old_K=old, T_new_K=new, dx_m=dx,
        dt_s=dt, power_W_m2=power, wall_energy_J_m2=exchange)
    assert audit.input_energy_J_m == pytest.approx(1.025e-5)
    assert audit.gas_energy_J_m == pytest.approx(3e-6)
    assert audit.storage_change_J_m == pytest.approx(7.25e-6, abs=1e-15)
    assert abs(audit.residual_J_m) < 1e-15


def test_adiabatic_strip_cosine_mode_decays_with_midpoint_factor():
    solve, audit_step = _interface()
    n, dx, dt, conductance = 8, 1.25e-4, 1e-4, 5.7996e-7
    params = FilmOdeParams(C_A_si=.012888)
    mode = np.cos(np.pi*(np.arange(n)+.5)/n)
    old = 300+2*mode
    lam = 4*conductance*np.sin(np.pi/(2*n))**2/(params.C_A_si*dx**2)
    expected = 300+2*(1-.5*lam*dt)/(1+.5*lam*dt)*mode
    mid = solve(params=params, T_s_K=old, dx_m=dx,
        sheet_conductance_W_K=conductance, dt_s=dt, power_W_m2=0.,
        exchange_J_m2=lambda temperature: np.zeros_like(temperature))
    new = 2*mid-old
    np.testing.assert_allclose(new, expected, atol=2e-11, rtol=0)
    assert np.sum(new) == pytest.approx(np.sum(old), abs=1e-10)
    audit = audit_step(params=params, T_old_K=old, T_new_K=new, dx_m=dx,
        dt_s=dt, power_W_m2=0., wall_energy_J_m2=np.zeros(n))
    assert audit.end_conduction_energy_J_m == 0.
    assert abs(audit.storage_change_J_m) < 2e-15


def test_uniform_equilibrium_is_unaffected_by_internal_conduction():
    solve, _ = _interface()
    old = np.full(5, 300.)
    mid = solve(params=FilmOdeParams(C_A_si=.012888), T_s_K=old,
        dx_m=1e-6, sheet_conductance_W_K=5.7996e-7, dt_s=1e-4,
        power_W_m2=0., exchange_J_m2=lambda temperature: np.zeros_like(temperature))
    np.testing.assert_array_equal(mid, old)


def test_nonlocal_exchange_is_solved_as_a_vector_root():
    solve, _ = _interface()
    params = FilmOdeParams(C_A_si=.02)
    old = np.array([301., 299., 300.])
    # A neighbouring gas bin can respond to another wall bin. Averaging this
    # exchange or treating its Jacobian as diagonal gives a different solution.
    response = np.array([[2., .5, .1], [.2, 1., .4], [.1, .3, 3.]])
    dt = .003
    power = np.array([3., 5., 7.])
    target_mid_excess = np.linalg.solve(
        2*params.C_A_si*np.eye(3)+2*dt*response,
        2*params.C_A_si*(old-300)+dt*power)
    mid = solve(params=params, T_s_K=old, dx_m=1e-4,
        sheet_conductance_W_K=0., dt_s=dt, power_W_m2=power,
        exchange_J_m2=lambda temperature: dt*response@(temperature-300))
    np.testing.assert_allclose(mid-300, target_mid_excess, atol=1e-10, rtol=0)


def test_local_exchange_cost_does_not_grow_with_bin_count():
    solve, _ = _interface()
    params = FilmOdeParams(C_A_si=.012888)
    counts = []
    for n in (8, 128):
        dx, dt, conductance = 1e-3/n, 2e-5, 5.7996e-7
        mode = np.cos(np.pi*(np.arange(n)+.5)/n)
        old = 300+.02*mode
        local_h = np.linspace(20., 60., n)
        calls = 0

        def exchange(temperature):
            nonlocal calls
            calls += 1
            return dt*local_h*(temperature-300)

        mid_excess = .03*mode
        decay = 4*conductance*np.sin(np.pi/(2*n))**2/dx**2
        power = 2*params.C_A_si*(mid_excess-(old-300))/dt
        power += 2*local_h*mid_excess+decay*mid_excess
        mid = solve(params=params, T_s_K=old, dx_m=dx,
            sheet_conductance_W_K=conductance, dt_s=dt, power_W_m2=power,
            exchange_J_m2=exchange, exchange_is_local=True)
        np.testing.assert_allclose(mid, 300+mid_excess, atol=1e-10, rtol=0)
        counts.append(calls)
    assert max(counts) <= 12
    assert abs(counts[1]-counts[0]) <= 2


def test_nonlinear_exchange_finds_positive_root_without_negative_trials():
    solve, _ = _interface()
    params = FilmOdeParams(C_A_si=.012888)
    old = np.array([300., 300.])
    target = np.array([290., 304.])
    dt = .01

    def exchange(temperature):
        assert np.all(2*temperature-old > 0), 'nonpositive endpoint evaluated'
        return dt*1e-8*(temperature**4-300.**4)

    power = (2*params.C_A_si*(target-old)+2*exchange(target))/dt
    mid = solve(params=params, T_s_K=old, dx_m=1e-4,
        sheet_conductance_W_K=0., dt_s=dt, power_W_m2=power,
        exchange_J_m2=exchange)
    np.testing.assert_allclose(mid, target, atol=1e-10, rtol=0)


def test_energy_audit_detects_wrong_endpoint_from_independent_storage():
    _, audit_step = _interface()
    params = FilmOdeParams(C_A_si=.012888)
    audit = audit_step(params=params, T_old_K=np.array([300., 301.]),
        T_new_K=np.array([300.1, 301.]), dx_m=5e-4, dt_s=1e-4,
        power_W_m2=0., wall_energy_J_m2=np.zeros(2))
    assert audit.residual_J_m == pytest.approx(6.444e-7, abs=1e-17)


@pytest.mark.parametrize('bad_exchange', [lambda t: float(0),
    lambda t: np.zeros(3), lambda t: np.array([np.nan, 0.])])
def test_invalid_wall_exchange_is_rejected(bad_exchange):
    solve, _ = _interface()
    with pytest.raises((ValueError, FloatingPointError)):
        solve(params=FilmOdeParams(C_A_si=.012888), T_s_K=np.full(2, 300.),
            dx_m=1e-4, sheet_conductance_W_K=5.7996e-7, dt_s=1e-4,
            power_W_m2=10., exchange_J_m2=bad_exchange)


def test_cooling_beyond_positive_endpoint_is_rejected_without_mutation():
    solve, _ = _interface()
    old = np.array([300., 301.])
    original = old.copy()
    with pytest.raises(FloatingPointError, match='positive|converge'):
        solve(params=FilmOdeParams(C_A_si=.012888), T_s_K=old,
            dx_m=1e-4, sheet_conductance_W_K=0., dt_s=.1,
            power_W_m2=-1e6, exchange_J_m2=lambda t: np.zeros_like(t))
    np.testing.assert_array_equal(old, original)


@pytest.mark.parametrize('kwargs', [dict(T_s_K=np.array([300., 0.])),
    dict(T_s_K=np.empty(0)), dict(dx_m=0.), dict(dt_s=-1.),
    dict(sheet_conductance_W_K=-1.), dict(power_W_m2=np.inf),
    dict(params=FilmOdeParams(C_A_si=np.nan)),
    dict(params=FilmOdeParams(gas_flux_factor=1.))])
def test_invalid_strip_state_or_units_are_rejected(kwargs):
    solve, _ = _interface()
    arguments = dict(params=FilmOdeParams(C_A_si=.012888), T_s_K=np.full(2, 300.),
        dx_m=1e-4, sheet_conductance_W_K=5.7996e-7, dt_s=1e-4,
        power_W_m2=0., exchange_J_m2=lambda t: np.zeros_like(t))
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        solve(**arguments)
