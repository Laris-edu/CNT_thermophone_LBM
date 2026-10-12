"""Compiled arithmetic must retain the public collision's physical invariants."""
from copy import deepcopy
import json

import numpy as np
import pytest

from core.collision_smrt import collide_fg
from core.equilibrium import equilibrium_fg
from core.planar_open_solver import PlanarOpenSolver
from core.solver import GasSolver2D
from simulation.planar import make_solver
from simulation import prepare_case, run_case


@pytest.mark.parametrize('model', ['constant', 'sutherland'])
@pytest.mark.parametrize('bulk', [0., .01])
def test_compiled_collision_matches_numpy_on_moving_nonequilibrium_states(model, bulk):
    config = deepcopy(make_solver(length=40e-6).config)
    config['transport']['model'] = model
    config['collision'].update(bulk_viscosity_policy='specified', nu_b_lu=bulk)
    solver = GasSolver2D(config)
    rng = np.random.default_rng(2301)
    shape = (8, 6)
    f, g = equilibrium_fg(rng.uniform(.8, 1.2, shape),
        rng.uniform(-.12, .12, (*shape, 2)),
        solver.mapping.theta_ref_lu*rng.uniform(.95, 1.2, shape), 3, solver.lattice)
    f *= rng.uniform(.98, 1.02, f.shape)
    g *= rng.uniform(.98, 1.02, g.shape)
    f, g = f[:, ::2], g[:, ::2]  # Non-contiguous inputs remain supported.
    original_f, original_g = f.copy(), g.copy()
    expected = collide_fg(f, g, solver.mapping, lattice=solver.lattice)
    fp, gp, diagnostic = collide_fg(f, g, solver.mapping, lattice=solver.lattice,
        backend='numba', return_diagnostics=True)
    for actual, reference in zip((fp, gp), expected):
        np.testing.assert_allclose(actual, reference, rtol=3e-13, atol=3e-15)
    np.testing.assert_array_equal(f, original_f)
    np.testing.assert_array_equal(g, original_g)
    # Independent raw total energy, not the kernel's energy-correction residual.
    c2 = np.sum(solver.lattice.c**2, axis=1)
    np.testing.assert_allclose(np.sum(.5*fp*c2+gp, axis=-1),
        np.sum(.5*f*c2+g, axis=-1), rtol=0, atol=1e-13)
    assert np.max(abs(diagnostic.mass_residual)) < 1e-13
    assert np.max(abs(diagnostic.momentum_residual)) < 1e-13
    assert np.max(abs(diagnostic.energy_residual)) < 1e-13
    assert min(fp.min(), gp.min()) > 0
    assert not diagnostic.clipping_used


@pytest.mark.parametrize('model', ['constant', 'sutherland'])
def test_parallel_collision_preserves_nonequilibrium_cell_invariants(model, monkeypatch):
    import numba
    import core.collision_numba as compiled

    config = deepcopy(make_solver(length=40e-6).config)
    config['transport']['model'] = model
    solver = GasSolver2D(config)
    rng = np.random.default_rng(52319)
    shape = (128, 256)
    f, g = equilibrium_fg(rng.uniform(.85, 1.15, shape),
        rng.uniform(-.10, .10, (*shape, 2)),
        solver.mapping.theta_ref_lu*rng.uniform(.97, 1.12, shape), 3, solver.lattice)
    f *= rng.uniform(.985, 1.015, f.shape)
    g *= rng.uniform(.985, 1.015, g.shape)
    before_f, before_g = f.copy(), g.copy()
    expected = collide_fg(f, g, solver.mapping, lattice=solver.lattice, backend='numpy')
    assert hasattr(compiled, '_measure_cells_parallel'), 'Parallel measured-moment kernel is missing'
    assert hasattr(compiled, '_collide_cells_parallel'), 'Parallel reconstruction kernel is missing'
    # Exercise parallel arithmetic with bounded test memory, independent of the
    # production crossover chosen by performance measurements.
    monkeypatch.setattr(compiled, 'PARALLEL_MIN_CELLS', 1)
    old_threads = numba.get_num_threads()
    if numba.config.NUMBA_NUM_THREADS < 2:
        pytest.skip('parallel regression requires two available Numba threads')
    try:
        numba.set_num_threads(1)
        serial = collide_fg(f, g, solver.mapping, lattice=solver.lattice, backend='numba')
        numba.set_num_threads(2)
        actual = collide_fg(f, g, solver.mapping, lattice=solver.lattice, backend='numba')
    finally:
        numba.set_num_threads(old_threads)
    assert compiled._measure_cells_parallel.signatures
    assert compiled._collide_cells_parallel.signatures
    for kernel in (compiled._measure_cells_parallel, compiled._collide_cells_parallel):
        assert kernel.targetoptions['parallel'] is True
        assert kernel.targetoptions['fastmath'] is False
    for result, reference, sequential in zip(actual, expected, serial):
        np.testing.assert_allclose(result, reference, rtol=3e-13, atol=3e-15)
        np.testing.assert_array_equal(result, sequential)
    fp, gp = actual
    lat = solver.lattice
    np.testing.assert_allclose(fp.sum(axis=-1), f.sum(axis=-1), rtol=0, atol=1e-13)
    np.testing.assert_allclose(np.einsum('...a,ai->...i', fp, lat.c),
        np.einsum('...a,ai->...i', f, lat.c), rtol=0, atol=1e-13)
    kinetic = .5*np.sum(lat.c**2, axis=1)
    np.testing.assert_allclose(np.sum(fp*kinetic+gp, axis=-1),
        np.sum(f*kinetic+g, axis=-1), rtol=0, atol=1e-13)
    np.testing.assert_array_equal(f, before_f)
    np.testing.assert_array_equal(g, before_g)
    assert min(fp.min(), gp.min()) > 0


def test_compiled_collision_avoids_full_field_numpy_preparation(monkeypatch):
    import core.collision_smrt as collision

    gas = make_solver(length=40e-6)
    rng = np.random.default_rng(8173)
    shape = (13, 11)
    f, g = equilibrium_fg(rng.uniform(.9, 1.1, shape),
        rng.uniform(-.08, .08, (*shape, 2)),
        gas.mapping.theta_ref_lu*rng.uniform(.97, 1.08, shape), 3, gas.lattice)
    f *= rng.uniform(.99, 1.01, f.shape)
    g *= rng.uniform(.99, 1.01, g.shape)
    expected = collision.collide_fg(f, g, gas.mapping, lattice=gas.lattice, backend='numpy')

    def reject_broadcast_preparation(*args, **kwargs):
        raise AssertionError('compiled hot path still builds full-field NumPy preparation temporaries')

    monkeypatch.setattr(collision, 'recover_macro', reject_broadcast_preparation)
    monkeypatch.setattr(collision, 'equilibrium_fg', reject_broadcast_preparation)
    actual = collision.collide_fg(f, g, gas.mapping, lattice=gas.lattice, backend='numba')
    for value, reference in zip(actual, expected):
        np.testing.assert_allclose(value, reference, rtol=3e-13, atol=3e-15)


def test_compiled_preparation_matches_public_macro_and_equilibrium():
    import core.collision_numba as compiled
    from core.macroscopic import recover_macro

    gas = make_solver(length=40e-6)
    rng = np.random.default_rng(35129)
    shape = (9, 10)
    f, g = equilibrium_fg(rng.uniform(.75, 1.25, shape),
        rng.uniform(-.15, .15, (*shape, 2)),
        gas.mapping.theta_ref_lu*rng.uniform(.94, 1.18, shape), 3, gas.lattice)
    f *= rng.uniform(.97, 1.03, f.shape)
    g *= rng.uniform(.97, 1.03, g.shape)
    f, g = f[:, ::2], g[:, ::2]
    expected = recover_macro(f, g, D=2, S=3, lattice=gas.lattice)
    assert hasattr(compiled, 'recover_macro_compiled'), 'bounded-memory macro preparation is missing'
    actual = compiled.recover_macro_compiled(f, g, D=2, S=3, lattice=gas.lattice)
    for name in ('rho', 'u', 'theta', 'p', 'e', 'mach', 'K_tr', 'G_int', 'E_tot'):
        np.testing.assert_allclose(getattr(actual, name), getattr(expected, name), rtol=2e-14, atol=2e-14)
    equilibrium = compiled.equilibrium_compiled(actual.rho, actual.u, actual.theta, 3, gas.lattice)
    reference = equilibrium_fg(actual.rho, actual.u, actual.theta, 3, gas.lattice)
    for value, target in zip(equilibrium, reference):
        np.testing.assert_allclose(value, target, rtol=3e-14, atol=3e-15)


@pytest.mark.parametrize('backend', ['numpy', 'numba'])
@pytest.mark.parametrize('bad', ['nonfinite', 'negative_density', 'negative_equilibrium'])
def test_preparation_rejects_invalid_state_without_mutating_populations(backend, bad):
    from core.equilibrium import EquilibriumDomainError

    gas = make_solver(length=40e-6)
    f, g = (value.copy() for value in gas._require_state())
    if bad == 'nonfinite':
        f[0, 0, 0] = np.nan
    elif bad == 'negative_density':
        f *= -1
    else:
        g *= 40
    original = f.copy(), g.copy()
    error = EquilibriumDomainError if bad == 'negative_equilibrium' else ValueError
    with pytest.raises(error):
        collide_fg(f, g, gas.mapping, lattice=gas.lattice, backend=backend)
    np.testing.assert_array_equal(f, original[0])
    np.testing.assert_array_equal(g, original[1])


def test_backend_is_explicit_and_unknown_selection_is_rejected():
    config = deepcopy(make_solver(length=40e-6).config)
    assert GasSolver2D(config).collision_backend == 'numpy'
    config['numerics']['collision_backend'] = 'typo'
    with pytest.raises(ValueError, match='collision_backend'):
        GasSolver2D(config)
    from simulation.study import validate_effective
    config = prepare_case(dict(name='bad', mode='pulse', refine=1))
    config['numerics']['collision_backend'] = 'typo'
    with pytest.raises(ValueError, match='collision_backend'):
        validate_effective(config)


def test_compiled_backend_is_used_by_periodic_and_open_solvers(monkeypatch):
    import core.collision_numba as compiled
    original = compiled.collide_prepared
    calls = []

    def measured(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(compiled, 'collide_prepared', measured)
    config = deepcopy(make_solver(length=40e-6).config)
    config['numerics']['collision_backend'] = 'numba'
    GasSolver2D(config).step(1)
    PlanarOpenSolver(config).step_temperature(300.)
    assert len(calls) == 2


def test_compiled_open_failure_keeps_populations_time_and_audit():
    config = deepcopy(make_solver(length=40e-6).config)
    config['numerics']['collision_backend'] = 'numba'
    solver = PlanarOpenSolver(config)
    solver.step_temperature(300.01)
    f, g, t, audit = solver.f.copy(), solver.g.copy(), solver.t_lu, solver.last_audit

    def fail(exchange):
        exchange(300.1)
        raise RuntimeError('trial failed')

    with pytest.raises(RuntimeError, match='trial failed'):
        solver.step_wall_energy(fail)
    np.testing.assert_array_equal(solver.f, f)
    np.testing.assert_array_equal(solver.g, g)
    assert solver.t_lu == t and solver.last_audit is audit


def test_compiled_run_records_backend_and_compiler_versions(tmp_path):
    config = prepare_case(dict(name='compiled', mode='power', refine=1,
        length_m=30e-6, frequency_Hz=2e6, cycles=.1,
        probes_m=[8e-6, 18e-6], stopping={'stop_when_stable':False}))
    config['numerics']['collision_backend'] = 'numba'
    run_case(config, tmp_path/'compiled')
    manifest = json.loads((tmp_path/'compiled/manifest.json').read_text())
    assert manifest['collision_backend'] == 'numba'
    assert manifest['numba'] and manifest['llvmlite']
