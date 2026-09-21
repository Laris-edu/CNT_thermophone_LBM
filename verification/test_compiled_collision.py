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
