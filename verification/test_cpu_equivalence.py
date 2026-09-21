"""Independent moving-state moments and prepared-wall exchange regressions."""
import numpy as np
import pytest

import boundary.planar_open as boundary
import core.collision_smrt as collision
from core.equilibrium import equilibrium_fg
from core.macroscopic import recover_macro, central_energy_flux_lu
from core.streaming import pull_stream_open_y
from simulation.planar import make_solver


@pytest.mark.parametrize('nx', [1, 4])
def test_moving_nonequilibrium_moments_match_direct_tensor_contractions(nx):
    solver = make_solver(length=40e-6)
    lattice = solver.lattice
    rng = np.random.default_rng(831)
    shape = (11, nx)
    rho = rng.uniform(.8, 1.2, shape)
    theta = solver.mapping.theta_ref_lu * rng.uniform(.95, 1.15, shape)
    u = rng.uniform(-.15, .15, (*shape, 2))
    f, g = equilibrium_fg(rho, u, theta, 3, lattice)
    f *= rng.uniform(.98, 1.02, f.shape)
    g *= rng.uniform(.98, 1.02, g.shape)
    state = recover_macro(f, g, lattice=lattice)
    xi = lattice.c - state.u[..., None, :]
    xi2 = np.sum(xi * xi, axis=-1)
    kinetic = .5 * np.sum(f * xi2, axis=-1)
    qf = .5 * np.einsum('...a,...a,...ai->...i', f, xi2, xi)
    qg = np.einsum('...a,...ai->...i', g, xi)
    fe, _ = equilibrium_fg(state.rho, state.u, state.theta, 3, lattice)
    stress = np.einsum('...a,...ai,...aj->...ij', f-fe, xi, xi)
    np.testing.assert_allclose(state.K_tr, kinetic, rtol=2e-15, atol=2e-15)
    np.testing.assert_allclose(central_energy_flux_lu(f, g, state.u, lattice), qf+qg, rtol=2e-13, atol=2e-15)
    np.testing.assert_allclose(collision._central_translational_energy_flux_lu(f, state.u, lattice), qf, rtol=2e-13, atol=2e-15)
    np.testing.assert_allclose(collision._central_internal_energy_flux_lu(g, state.u, lattice), qg, rtol=2e-13, atol=2e-15)
    np.testing.assert_allclose(collision._nonequilibrium_central_stress(f, fe, state.u, lattice), stress, rtol=2e-13, atol=2e-15)


def _direct_wall(f, g, fn, gn, mapping, lattice, temperature):
    """Explicit multispeed-link definition, independent of prepared-wall code."""
    nx = f.shape[1]
    fw, gw = equilibrium_fg(np.ones(nx), np.zeros((nx, 2)),
        np.full(nx, temperature / mapping.temperature_scale), 3, lattice)
    outgoing = np.zeros(nx)
    denominator = np.zeros(nx)
    for a, (_, cy) in enumerate(lattice.c.astype(int)):
        if cy < 0:
            outgoing += np.sum(f[:-cy, :, a], axis=0)
        if cy > 0:
            denominator += cy * fw[:, a]
    density = outgoing / denominator
    for a, (_, cy) in enumerate(lattice.c.astype(int)):
        if cy > 0:
            fn[:cy, :, a] = density * fw[:, a]
            gn[:cy, :, a] = density * gw[:, a]


def _direct_exchange(f, g, fn, gn, lattice):
    energy = 0.
    for a, velocity in enumerate(lattice.c.astype(int)):
        cy = velocity[1]
        if cy:
            before = .5 * np.dot(velocity, velocity) * f[:abs(cy), :, a] + g[:abs(cy), :, a]
            after = .5 * np.dot(velocity, velocity) * fn[:abs(cy), :, a] + gn[:abs(cy), :, a]
            energy += np.sum(after) if cy > 0 else -np.sum(before)
    return energy / f.shape[1]


@pytest.mark.parametrize('top,incoming', [(False, False), (False, True), (True, False), (True, True)])
def test_crossing_totals_include_all_multispeed_rows_and_columns(top, incoming):
    lattice = make_solver(length=40e-6).lattice
    rng = np.random.default_rng(812)
    f, g = rng.uniform(.001, .1, (2, 9, 4, lattice.q))
    mass = energy = 0.
    for a, (cx, cy) in enumerate(lattice.c.astype(int)):
        sign = -1 if top == incoming else 1
        if cy * sign > 0:
            for row in (range(9-abs(cy), 9) if top else range(abs(cy))):
                for x in range(4):
                    mass += f[row, x, a] / 4
                    energy += (.5*(cx*cx+cy*cy)*f[row, x, a]+g[row, x, a]) / 4
    np.testing.assert_allclose(boundary.crossing_totals(f, g, lattice, top=top, incoming=incoming),
        [mass, energy], rtol=2e-15, atol=2e-15)


@pytest.mark.parametrize('nx', [1, 4])
def test_prepared_wall_trials_match_actual_links_and_leave_input_unchanged(nx):
    assert hasattr(boundary, 'PreparedDiffuseWall'), 'prepare fixed outgoing links once per time step'
    solver = make_solver(length=40e-6)
    rng = np.random.default_rng(941)
    shape = (12, nx)
    f, g = equilibrium_fg(rng.uniform(.9, 1.1, shape),
        rng.uniform(-.03, .03, (*shape, 2)),
        solver.mapping.theta_ref_lu * rng.uniform(.98, 1.06, shape), 3, solver.lattice)
    f *= rng.uniform(.99, 1.01, f.shape)
    old_f, old_g = f.copy(), g.copy()
    wall = boundary.PreparedDiffuseWall(f, g, solver.mapping, solver.lattice)
    for temperature in (299., 315., 303., 299.):
        fn = pull_stream_open_y(f, lattice=solver.lattice)
        gn = pull_stream_open_y(g, lattice=solver.lattice)
        expected_f, expected_g = fn.copy(), gn.copy()
        _direct_wall(f, g, expected_f, expected_g, solver.mapping, solver.lattice, temperature)
        actual_energy = wall.energy_exchange_lu(temperature)
        wall.write(temperature, fn, gn)
        np.testing.assert_allclose(fn, expected_f, rtol=3e-15, atol=2e-16)
        np.testing.assert_allclose(gn, expected_g, rtol=3e-15, atol=2e-16)
        assert actual_energy == pytest.approx(_direct_exchange(f, g, expected_f, expected_g, solver.lattice), abs=2e-15)
    np.testing.assert_array_equal(f, old_f)
    np.testing.assert_array_equal(g, old_g)


def test_wall_root_trials_write_only_the_selected_boundary_once(monkeypatch):
    assert hasattr(boundary, 'PreparedDiffuseWall'), 'wall trials should not mutate the streamed state'
    solver = make_solver(length=40e-6)
    calls = []
    original = boundary.PreparedDiffuseWall.write

    def record(self, temperature, f, g):
        calls.append(temperature)
        return original(self, temperature, f, g)

    monkeypatch.setattr(boundary.PreparedDiffuseWall, 'write', record)

    def choose(exchange):
        for temperature in (300.1, 299.9, 300.2):
            exchange(temperature)
        return 300.1

    solver.step_wall_energy(choose)
    assert calls == [300.1]
