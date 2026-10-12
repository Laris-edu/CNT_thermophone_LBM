"""Finite-strip link geometry, per-bin exchange, and independent gas storage."""
import importlib
import importlib.util

import numpy as np
import pytest

from core.lattice import make_lattice
from simulation.planar import make_solver


def api():
    assert importlib.util.find_spec('boundary.finite_strip') is not None, 'finite-strip boundary is missing'
    assert importlib.util.find_spec('core.finite_strip_solver') is not None, 'finite-strip solver is missing'
    return importlib.import_module('boundary.finite_strip'), importlib.import_module('core.finite_strip_solver')


def solver(*, film_cells=8, sponge_cells=0, sponge_rate=0., backend='numpy'):
    _, module = api()
    config = make_solver(length=80e-6).config
    config['numerics'].update(nx=24, ny=16, collision_backend=backend)
    dx = config['lattice']['dx_m']
    return module.FiniteStripSolver(config, film_width_m=film_cells*dx,
                                   sponge_cells=sponge_cells, sponge_rate=sponge_rate)


def test_open_xy_stream_has_no_periodic_wrap_in_either_direction():
    boundary, _ = api()
    lat = make_lattice()
    f = np.arange(8*9*lat.q, dtype=float).reshape(8, 9, lat.q)
    out = boundary.pull_stream_open_xy(f, lattice=lat)
    for y in range(8):
        for x in range(9):
            for a, (cx, cy) in enumerate(lat.c.astype(int)):
                if 0 <= y-cy < 8 and 0 <= x-cx < 9:
                    assert out[y, x, a] == f[y-cy, x-cx, a]
                else:
                    assert np.isnan(out[y, x, a])


def test_specular_multispeed_link_keeps_horizontal_travel():
    boundary, _ = api()
    gas = solver(film_cells=0)
    lat = gas.lattice
    f = np.arange(gas.ny*gas.nx*lat.q, dtype=float).reshape(gas.ny, gas.nx, lat.q)
    g = f + 17
    fn, gn = (boundary.pull_stream_open_xy(v, lattice=lat) for v in (f, g))
    gas.boundary.write_specular(f, g, fn, gn)
    # (cx,cy)=(1,2): a backward ray from (x,y)=(5,0) reflects at y=0
    # and ends at the post-collision source (4,1), direction (1,-2).
    a = np.flatnonzero(np.all(lat.c == (1, 2), axis=1))[0]
    b = np.flatnonzero(np.all(lat.c == (1, -2), axis=1))[0]
    assert fn[0, 5, a] == f[1, 4, b]
    assert gn[0, 5, a] == g[1, 4, b]


def test_multispeed_wall_uses_intersection_not_destination_column():
    boundary, _ = api()
    gas = solver(film_cells=8)
    lat = gas.lattice
    f, g = gas._require_state()
    fn, gn = (boundary.pull_stream_open_xy(v, lattice=lat) for v in (f, g))
    gas.boundary.write_specular(f, g, fn, gn)
    wall = boundary.PreparedStripWall(f, g, gas.mapping, gas.lattice, gas.boundary)
    wall.write(np.full(8, 300.1), fn, gn)
    # Film covers columns [8,16). At row 0, (3,1) shifts the wall hit
    # by 1.5 cells: destination column 8 hits x=7 (symmetry); column16
    # hits x=15 (film), even though its cell centre is outside the film.
    a = np.flatnonzero(np.all(lat.c == (3, 1), axis=1))[0]
    b = np.flatnonzero(np.all(lat.c == (3, -1), axis=1))[0]
    assert fn[0, 8, a] == f[0, 5, b]
    assert fn[0, 16, a] != f[0, 13, b]
    assert np.isfinite(fn[0, 16, a])


@pytest.mark.parametrize('backend', ['numpy', 'numba'])
def test_finite_strip_equilibrium_generates_no_heat_or_velocity(backend):
    gas = solver(backend=backend, sponge_cells=3, sponge_rate=.04)
    initial = gas.get_macro().p.copy()
    for _ in range(12):
        audit = gas.step_temperature(300.)
        np.testing.assert_allclose(audit.wall_mass_lu, 0, atol=2e-14)
        np.testing.assert_allclose(audit.wall_energy_J_m2, 0, atol=2e-13)
    np.testing.assert_allclose(gas.get_macro().p, initial, rtol=0, atol=8e-14)
    np.testing.assert_allclose(gas.get_macro().u, 0, rtol=0, atol=8e-14)


def test_per_bin_heat_and_sponge_close_against_independent_storage():
    gas = solver(sponge_cells=3, sponge_rate=.07)
    m = gas.mapping
    dx = m.lattice.dx_m
    wall_energy = outer_energy = sponge_energy = 0.
    initial = gas.get_macro()
    mass0 = np.sum(initial.rho)
    energy0 = np.sum(initial.E_tot)*m.pressure_scale*dx**2
    outer_mass = sponge_mass = 0.
    for n in range(25):
        trials = []
        temperature = 300 + np.linspace(.01, .09, gas.film_x_m.size)

        def solve(exchange):
            trials.append(exchange(temperature))
            exchange(temperature+.1)  # Last trial is deliberately not the root.
            return temperature

        audit = gas.step_wall_energy(solve)
        np.testing.assert_allclose(audit.wall_mass_lu, 0, atol=2e-14)
        np.testing.assert_allclose(audit.wall_energy_J_m2, trials[0], atol=2e-15)
        wall_energy += np.sum(audit.wall_energy_J_m2)*gas.film_dx_m
        outer_energy += audit.outer_energy_J_m
        sponge_energy += audit.sponge_energy_J_m
        outer_mass += audit.outer_mass_lu
        sponge_mass += audit.sponge_mass_lu
        assert abs(audit.energy_residual_J_m) < 2e-16
    final = gas.get_macro()
    change = np.sum(final.E_tot)*m.pressure_scale*dx**2-energy0
    assert abs(change-wall_energy-outer_energy-sponge_energy) < 2e-15
    assert abs(np.sum(final.rho)-mass0-outer_mass-sponge_mass) < 1e-10
    assert wall_energy > 0
    assert abs(sponge_energy) > 1e-15


def test_failed_midpoint_callback_does_not_commit_state_or_clock():
    gas = solver()
    gas.step_temperature(300.01)
    f, g, t, audit = gas.f.copy(), gas.g.copy(), gas.t_lu, gas.last_audit

    def fail(exchange):
        exchange(np.full(gas.film_x_m.size, 300.2))
        raise RuntimeError('no root')

    with pytest.raises(RuntimeError, match='no root'):
        gas.step_wall_energy(fail)
    np.testing.assert_array_equal(gas.f, f)
    np.testing.assert_array_equal(gas.g, g)
    assert gas.t_lu == t
    assert gas.last_audit is audit


def test_no_film_supports_symmetric_pulse_and_rejects_periodic_step():
    gas = solver(film_cells=0)
    m = gas.mapping
    perturb = 1e-5*np.exp(-((gas.x_m[None, :]/(3*m.lattice.dx_m))**2
                          +((gas.y_m[:, None]-6*m.lattice.dx_m)/(3*m.lattice.dx_m))**2))
    gas.initialize_from_macro(m.lattice.rho_ref_lu*(1+perturb), np.zeros(2),
                              m.theta_ref_lu*(1+perturb)**.4)
    for _ in range(5):
        audit = gas.step_temperature(300.)
        assert audit.wall_energy_J_m2.size == 0
        assert abs(audit.energy_residual_J_m) < 2e-16
    with pytest.raises(TypeError, match='step_temperature'):
        gas.step()


def test_uniform_heating_preserves_left_right_symmetry_at_exact_edge_hits():
    gas = solver()
    for _ in range(10):
        gas.step_temperature(300.1)
    state = gas.get_macro()
    np.testing.assert_allclose(state.p, state.p[:, ::-1], rtol=0, atol=1e-13)
    np.testing.assert_allclose(state.u[..., 0], -state.u[:, ::-1, 0], rtol=0, atol=1e-13)
    np.testing.assert_allclose(state.u[..., 1], state.u[:, ::-1, 1], rtol=0, atol=1e-13)


def test_exchange_temperature_dependence_is_strictly_local_in_each_bin():
    boundary, _ = api()
    gas = solver()
    f, g = gas._require_state()
    wall = boundary.PreparedStripWall(f, g, gas.mapping, gas.lattice, gas.boundary)
    base_temperature = np.full(gas.film_x_m.size, 300.01)
    base = wall.energy_exchange_lu(base_temperature)
    shifted = base_temperature.copy()
    shifted[3] += .1
    difference = wall.energy_exchange_lu(shifted)-base
    assert difference[3] > 0
    np.testing.assert_array_equal(np.delete(difference, 3), 0)


@pytest.mark.parametrize('temperature', [np.nan, -1., np.inf])
def test_invalid_temperature_never_commits(temperature):
    gas = solver()
    gas.step_temperature(300.)
    f, g, audit = gas.f.copy(), gas.g.copy(), gas.last_audit
    with pytest.raises(ValueError, match='temperatures'):
        gas.step_temperature(temperature)
    np.testing.assert_array_equal(gas.f, f)
    np.testing.assert_array_equal(gas.g, g)
    assert gas.t_lu == 1
    assert gas.last_audit is audit


def test_bottom_corner_outer_ray_reflects_its_velocity_direction():
    boundary, _ = api()
    gas = solver(film_cells=0)
    m, lat = gas.mapping, gas.lattice
    gas.initialize_from_macro(m.lattice.rho_ref_lu, np.array([0., .002]), m.theta_ref_lu)
    f, g = gas._require_state()
    fn, gn = (boundary.pull_stream_open_xy(v, lattice=lat) for v in (f, g))
    boundary.characteristic_outer(f, g, fn, gn, m, lat, gas.boundary)
    a = np.flatnonzero(np.all(lat.c == (1, 2), axis=1))[0]
    b = np.flatnonzero(np.all(lat.c == (1, -2), axis=1))[0]
    # Backward from (0.5,0.5), this ray hits bottom at t=.25, then the
    # left boundary at t=.5 and y=.5. It entered with cy=-2, not cy=+2.
    assert fn[0, 0, a] == pytest.approx(f[0, 0, b], abs=2e-16)
    assert gn[0, 0, a] == pytest.approx(g[0, 0, b], abs=2e-16)
