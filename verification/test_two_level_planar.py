"""Conservative physical links, subcycling and atomicity of the two-level grid."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from config_io import load_config
from core.two_level_planar_solver import TwoLevelPlanarOpenSolver
from core.unit_mapping import local_transport_coefficients


def two_level_config(*, model='sutherland', backend='numpy'):
    config = deepcopy(load_config(Path('configs/gas_air_10k_d2q37_qmatched_local_transport.yaml')))
    config['numerics'].update(nx=1, ny=48, collision_backend=backend)
    config['transport']['model'] = model
    config['grid'] = dict(type='two_level', ratio=2, interface_m=16*config['lattice']['dx_m'])
    return config


@pytest.mark.parametrize('direction', range(37))
def test_all_population_links_have_correct_cell_centered_transfer(monkeypatch, direction):
    solver = TwoLevelPlanarOpenSolver(two_level_config())
    solver.initialize_equilibrium()
    lattice = solver.lattice
    cy = int(lattice.c[direction, 1])
    fine_f, fine_g = solver.fine_solver.f.copy(), solver.fine_solver.g.copy()
    coarse_f, coarse_g = solver.coarse_solver.f.copy(), solver.coarse_solver.g.copy()
    # Distinct donor values exercise every subcell, including |cy|=2 and 3.
    for row in range(6):
        fine_f[-6+row, 0, direction] *= 1 + (row+1)*1e-3
        fine_g[-6+row, 0, direction] *= 1 + (row+1)*2e-3
        coarse_f[row, 0, direction] *= 1 + (row+1)*3e-3
        coarse_g[row, 0, direction] *= 1 + (row+1)*4e-3
    solver.fine_solver.f, solver.fine_solver.g = fine_f, fine_g
    solver.coarse_solver.f, solver.coarse_solver.g = coarse_f, coarse_g
    monkeypatch.setattr('core.two_level_planar_solver.collide_fg',
        lambda f, g, *args, **kwargs: (f.copy(), g.copy()))
    old = solver.conserved_totals_si()
    audit = solver.step_temperature(300.)
    if cy > 0:
        for previous, actual in ((fine_f, solver.coarse_solver.f), (fine_g, solver.coarse_solver.g)):
            expected = previous[-2*cy:, 0, direction].reshape(cy, 2).mean(axis=1)
            np.testing.assert_array_equal(actual[:cy, 0, direction], expected)
        # Only the missing coarse populations are replaced by merged shadows.
        for previous, actual in ((coarse_f, solver.coarse_solver.f), (coarse_g, solver.coarse_solver.g)):
            np.testing.assert_array_equal(actual[cy:, 0, direction], previous[:-cy, 0, direction])
    elif cy < 0:
        for previous, actual in ((coarse_f, solver.fine_solver.f), (coarse_g, solver.fine_solver.g)):
            np.testing.assert_array_equal(actual[2*cy:, 0, direction],
                                          np.repeat(previous[:-cy, 0, direction], 2))
    else:
        np.testing.assert_array_equal(solver.fine_solver.f[:, :, direction], fine_f[:, :, direction])
        np.testing.assert_array_equal(solver.coarse_solver.g[:, :, direction], coarse_g[:, :, direction])
    exchange = solver.last_interface_audit.as_array()
    # SI channels have different units: assess cancellation in common LU, at
    # the rounding scale of the order-one populations that were independently summed.
    m = solver.mapping
    h = m.lattice.dx_m
    si = np.array([m.rho_scale*h, m.rho_scale*m.velocity_scale*h,
                   m.rho_scale*m.velocity_scale*h, m.pressure_scale*h])
    np.testing.assert_allclose((exchange[0] + exchange[1])/si, 0.,
                               atol=4*np.finfo(float).eps, rtol=0.)
    np.testing.assert_array_equal(exchange[2], exchange[0] + exchange[1])
    new = solver.conserved_totals_si()
    scale = solver.mapping.rho_scale*solver.mapping.lattice.dx_m
    assert abs(new[0]-old[0]-(audit.wall_mass_lu+audit.top_mass_lu)*scale) < 1e-19
    assert abs(new[3]-old[3]-audit.wall_energy_J_m2-audit.top_energy_J_m2) < 1e-14
    # All perturbations are farther than one macro-step from a physical end.
    # The stationary wall and top exchanges cancel, so both momentum components
    # also stay fixed without relying on the solver's reported residual.
    np.testing.assert_allclose(new[1:3], old[1:3], rtol=0., atol=1e-18)
    np.testing.assert_allclose(solver.last_momentum_residual_kg_m_s, 0., atol=1e-18)


def test_only_physical_cells_collide_and_time_ratios_are_exact(monkeypatch):
    solver = TwoLevelPlanarOpenSolver(two_level_config())
    calls, midpoint = [], []

    def collision(f, g, mapping, **kwargs):
        calls.append((len(f), mapping.lattice.dt_s))
        return f.copy(), g.copy()

    monkeypatch.setattr('core.two_level_planar_solver.collide_fg', collision)
    solver.step_wall_energy(lambda exchange, index, time: midpoint.append((index, time)) or 300.)
    dt = solver.mapping.lattice.dt_s
    assert calls == [(16, 2*dt), (16, dt), (16, dt)]
    assert midpoint == [(0, .5*dt), (1, 1.5*dt)]
    assert solver.t_lu == 2
    assert solver.fine_solver.t_lu == 2
    assert solver.coarse_solver.t_lu == 1
    assert solver.level_steps == {'fine': 2, 'coarse': 1}
    assert solver.cell_updates == 48
    assert solver.f.shape == (32, 1, 37)
    np.testing.assert_allclose(solver.cell_width_m[:16], dt*solver.mapping.velocity_scale)
    np.testing.assert_allclose(solver.cell_width_m[16:], 2*dt*solver.mapping.velocity_scale)


def test_second_substep_failure_restores_entire_synchronized_state():
    solver = TwoLevelPlanarOpenSolver(two_level_config())
    solver.step_temperature(300.1)
    state = (solver.f.copy(), solver.g.copy(), solver.t_lu, solver.last_audit,
             solver.last_substep_audits, solver.last_interface_audit,
             solver.level_steps.copy(), solver.cell_updates)

    def failing(exchange, index, time):
        if index == 1:
            raise ValueError('second fine root failed')
        return 300.2

    with pytest.raises(ValueError, match='second fine root'):
        solver.step_wall_energy(failing)
    np.testing.assert_array_equal(solver.f, state[0])
    np.testing.assert_array_equal(solver.g, state[1])
    assert solver.t_lu == state[2]
    assert solver.last_audit is state[3]
    assert solver.last_substep_audits is state[4]
    assert solver.last_interface_audit is state[5]
    assert solver.level_steps == state[6]
    assert solver.cell_updates == state[7]


def test_second_substep_nonpositive_population_does_not_commit(monkeypatch):
    solver = TwoLevelPlanarOpenSolver(two_level_config())
    solver.step_temperature(300.1)
    f, g, audit = solver.f.copy(), solver.g.copy(), solver.last_audit
    calls = 0

    def collision(f, g, mapping, **kwargs):
        nonlocal calls
        calls += 1
        f, g = f.copy(), g.copy()
        if calls == 3:
            zero = np.flatnonzero(np.all(solver.lattice.c == 0, axis=1))[0]
            f[8, 0, zero] = -1.
        return f, g

    monkeypatch.setattr('core.two_level_planar_solver.collide_fg', collision)
    with pytest.raises(FloatingPointError, match='nonpositive'):
        solver.step_temperature(300.2)
    np.testing.assert_array_equal(solver.f, f)
    np.testing.assert_array_equal(solver.g, g)
    assert solver.t_lu == 2
    assert solver.level_steps == {'fine': 2, 'coarse': 1}
    assert solver.last_audit is audit


@pytest.mark.parametrize('backend', ['numpy', 'numba'])
def test_hot_wall_storage_uses_actual_widths_and_interface_cancels(backend):
    solver = TwoLevelPlanarOpenSolver(two_level_config(backend=backend))
    h = solver.mapping.lattice.dx_m
    widths = np.r_[np.full(16, h), np.full(16, 2*h)]
    m = solver.mapping
    wall_energy, top_energy = 0., 0.
    initial = solver.get_macro()
    initial_energy = np.dot(widths, initial.E_tot[:, 0])*m.pressure_scale
    initial_mass = np.dot(widths, initial.rho[:, 0])*m.rho_scale
    boundary_mass = 0.
    for _ in range(100):
        a = solver.step_temperature(300.1)
        wall_energy += a.wall_energy_J_m2
        top_energy += a.top_energy_J_m2
        boundary_mass += (a.wall_mass_lu+a.top_mass_lu)*m.rho_scale*h
        macro = solver.get_macro()
        # Independent macro recovery and physical cell integration, not _totals.
        energy = np.dot(widths, macro.E_tot[:, 0])*m.pressure_scale
        mass = np.dot(widths, macro.rho[:, 0])*m.rho_scale
        assert abs(energy-initial_energy-wall_energy-top_energy) < 1e-10
        assert abs(mass-initial_mass-boundary_mass)/(m.rho_scale*h) < 1e-10
        assert abs(a.wall_mass_lu) < 1e-12
        assert abs(a.energy_residual_J_m2) < 1e-10
        assert len(solver.last_substep_audits) == 2
        assert a.wall_energy_J_m2 == sum(s.wall_energy_J_m2 for s in solver.last_substep_audits)
    assert wall_energy > 0.


@pytest.mark.parametrize('model', ['constant', 'sutherland'])
def test_identical_local_state_has_identical_physical_transport_on_both_levels(model):
    solver = TwoLevelPlanarOpenSolver(two_level_config(model=model))
    rho = np.array([.8, 1., 1.2])
    theta = solver.mapping.theta_ref_lu*np.array([.95, 1., 1.15])
    fine = local_transport_coefficients(solver.mapping, rho, theta)
    coarse = local_transport_coefficients(solver.coarse_mapping, rho, theta)
    for name in ('nu_lu', 'alpha_lu'):
        a = getattr(fine, name)*solver.mapping.lattice.dx_m**2/solver.mapping.lattice.dt_s
        b = getattr(coarse, name)*solver.coarse_mapping.lattice.dx_m**2/solver.coarse_mapping.lattice.dt_s
        np.testing.assert_allclose(a, b, rtol=3e-15)
    np.testing.assert_array_equal(fine.tau22, .5*np.ones_like(rho))
    np.testing.assert_array_equal(coarse.tau22, fine.tau22)
    for name in ('tau21', 'tau32'):
        np.testing.assert_allclose(getattr(coarse, name), .5+(getattr(fine, name)-.5)/2,
                                   rtol=0, atol=np.spacing(.5))


@pytest.mark.parametrize('backend', ['numpy', 'numba'])
def test_equilibrium_has_no_resolvable_spurious_sound(backend):
    solver = TwoLevelPlanarOpenSolver(two_level_config(backend=backend))
    initial = solver.get_macro()
    for _ in range(60):
        audit = solver.step_temperature(300.)
        assert abs(audit.wall_mass_lu) < 1e-14
        assert abs(audit.wall_energy_J_m2) < 1e-13
    np.testing.assert_allclose(solver.get_macro().p, initial.p, rtol=0, atol=5e-14)
    np.testing.assert_allclose(solver.get_macro().u, 0., rtol=0, atol=5e-14)


@pytest.mark.parametrize('change,match', [
    ({'ratio': 3}, 'ratio'), ({'interface_m': 15*2.61175e-6}, 'coarse.*face'),
    ({'interface_m': 16.1*2.61175e-6}, 'face'),
])
def test_invalid_grid_is_rejected(change, match):
    config = two_level_config()
    config['grid'].update(change)
    with pytest.raises(ValueError, match=match):
        TwoLevelPlanarOpenSolver(config)


def test_nonzero_bulk_route_and_periodic_step_are_rejected():
    config = two_level_config()
    config['collision']['bulk_viscosity_policy'] = 'specified'
    config['collision']['nu_b_lu'] = .05
    with pytest.raises(ValueError, match='diagnostic_zero'):
        TwoLevelPlanarOpenSolver(config)
    with pytest.raises(TypeError, match='step_temperature'):
        TwoLevelPlanarOpenSolver(two_level_config()).step()
