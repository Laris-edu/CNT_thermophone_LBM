"""PML equilibrium increments retain kinetic departures and actual accounting."""
import numpy as np
import pytest

from core.equilibrium import EquilibriumDomainError, equilibrium_fg
from simulation.planar import make_solver


@pytest.mark.parametrize('parallel', [False, True])
def test_masked_increment_preserves_nonequilibrium_and_measures_actual_delta(parallel, monkeypatch):
    import numba
    import core.collision_numba as compiled

    gas = make_solver(length=40e-6)
    rng = np.random.default_rng(74091)
    shape = (19, 14)
    rho = rng.uniform(.91, 1.08, shape)
    u = rng.uniform(-.08, .08, (*shape, 2))
    theta = gas.mapping.theta_ref_lu*rng.uniform(.98, 1.10, shape)
    old_fe, old_ge = equilibrium_fg(rho, u, theta, 3, gas.lattice)
    f = old_fe*rng.uniform(.985, 1.015, old_fe.shape)
    g = old_ge*rng.uniform(.985, 1.015, old_ge.shape)
    new_rho = rho*rng.uniform(.999, 1.001, shape)
    new_u = u+rng.uniform(-.001, .001, (*shape, 2))
    new_theta = theta*rng.uniform(.999, 1.001, shape)
    mask = rng.random(shape) < .43
    # Also exercise noncontiguous fields. Inactive new states are deliberately
    # undefined: masked equilibrium evaluation must never inspect them.
    new_rho[~mask], new_u[~mask], new_theta[~mask] = np.nan, np.nan, np.nan
    f, g = f[:, ::2], g[:, ::2]
    rho, u, theta = rho[:, ::2], u[:, ::2], theta[:, ::2]
    new_rho, new_u, new_theta = new_rho[:, ::2], new_u[:, ::2], new_theta[:, ::2]
    mask = mask[:, ::2]
    inputs = (f, g, rho, u, theta, new_rho, new_u, new_theta, mask)
    saved = tuple(value.copy() for value in inputs)
    before = equilibrium_fg(rho[mask], u[mask], theta[mask], 3, gas.lattice)
    after = equilibrium_fg(new_rho[mask], new_u[mask], new_theta[mask], 3, gas.lattice)
    expected_f, expected_g = f.copy(), g.copy()
    expected_f[mask] += after[0]-before[0]
    expected_g[mask] += after[1]-before[1]
    assert hasattr(compiled, 'apply_equilibrium_increment'), 'compiled masked increment is missing'
    monkeypatch.setattr(compiled, 'PARALLEL_MIN_CELLS', 1)
    old_threads = numba.get_num_threads()
    if parallel and numba.config.NUMBA_NUM_THREADS < 2:
        pytest.skip('parallel regression needs two Numba threads')
    try:
        numba.set_num_threads(2 if parallel else 1)
        fn, gn, change = compiled.apply_equilibrium_increment(*inputs, gas.lattice, 3)
    finally:
        numba.set_num_threads(old_threads)
    for value, reference in zip((fn, gn), (expected_f, expected_g)):
        np.testing.assert_allclose(value, reference, rtol=3e-14, atol=3e-15)
    for value, original in zip(inputs, saved):
        np.testing.assert_array_equal(value, original)
    np.testing.assert_array_equal(fn[~mask].view(np.uint64), f[~mask].copy().view(np.uint64))
    np.testing.assert_array_equal(gn[~mask].view(np.uint64), g[~mask].copy().view(np.uint64))
    np.testing.assert_allclose(fn[mask]-after[0], f[mask]-before[0], rtol=0, atol=3e-15)
    np.testing.assert_allclose(gn[mask]-after[1], g[mask]-before[1], rtol=0, atol=3e-15)
    # Account from the returned population writes, independently of macro targets.
    df, dg = fn[mask]-f[mask], gn[mask]-g[mask]
    kinetic = .5*np.sum(gas.lattice.c**2, axis=1)
    expected_change = np.array([df.sum(), (df*kinetic+dg).sum()])
    np.testing.assert_allclose(change, expected_change, rtol=2e-14, atol=1e-17)
    if parallel:
        kernel = compiled._equilibrium_increment_cells_parallel
        assert kernel.signatures and kernel.targetoptions['parallel'] is True
        assert kernel.targetoptions['fastmath'] is False


@pytest.mark.parametrize('bad', ['nonfinite', 'negative_density', 'negative_equilibrium'])
def test_increment_rejects_invalid_active_equilibrium_without_mutation(bad):
    import core.collision_numba as compiled

    gas = make_solver(length=40e-6)
    f, g = (value.copy() for value in gas._require_state())
    shape = f.shape[:-1]
    rho, u = np.ones(shape), np.zeros((*shape, 2))
    theta = np.full(shape, gas.mapping.theta_ref_lu)
    new_rho, new_theta = rho.copy(), theta.copy()
    if bad == 'nonfinite':
        new_theta[0, 0] = np.nan
    elif bad == 'negative_density':
        new_rho[0, 0] = -1
    else:
        new_theta[0, 0] *= 40
    saved = f.copy(), g.copy()
    assert hasattr(compiled, 'apply_equilibrium_increment'), 'compiled masked increment is missing'
    with pytest.raises(EquilibriumDomainError):
        compiled.apply_equilibrium_increment(f, g, rho, u, theta, new_rho, u,
            new_theta, np.ones(shape, dtype=bool), gas.lattice, 3)
    np.testing.assert_array_equal(f, saved[0])
    np.testing.assert_array_equal(g, saved[1])
