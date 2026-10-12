"""Compiled acoustic memory RHS matches the existing nonperiodic stencil."""
import importlib
import importlib.util

import numba
import numpy as np
import pytest

from boundary.acoustic_pml import AcousticPML
from verification.test_finite_strip import solver


@pytest.mark.parametrize('parallel', [False, True])
@pytest.mark.parametrize('kind', ['smooth', 'random_strided'])
def test_compiled_rhs_matches_numpy_with_edges_and_corners(parallel, kind, monkeypatch):
    import core.collision_numba as dispatch

    gas = solver(film_cells=0)
    ny, nx = 13, 17
    pml = AcousticPML(gas.mapping, gas.lattice, ny, nx,
                      cells=5, target_return=.001, alpha_Hz=1000.)
    y, x = np.mgrid[:ny, :nx]
    if kind == 'smooth':
        v = np.stack([.02*np.sin((j+1)*x/11.)+.01*np.cos((j+2)*y/13.)
                      + .0003*j*x*y for j in range(7)])
        rho = .95+.02*np.sin(x/8.)+.01*np.cos(y/7.)
        c2 = pml.c2*(1.05+.01*np.sin((x+y)/9.))
    else:
        rng = np.random.default_rng(8714)
        v = rng.normal(0., .2, (7, 2*ny, 2*nx))[:, ::2, ::2]
        rho = rng.uniform(.9, 1.1, (2*ny, 2*nx))[::2, ::2]
        c2 = rng.uniform(.95, 1.15, (2*ny, 2*nx))[::2, ::2]
        assert not v.flags.c_contiguous
    values = (v, rho, c2, pml.sigma_x, pml.sigma_y, pml.decay_x, pml.decay_y)
    saved = tuple(value.copy() for value in values)
    expected = pml._rhs(v, rho, c2)
    assert importlib.util.find_spec('boundary.acoustic_pml_numba') is not None, \
        'compiled acoustic RHS is missing'
    compiled = importlib.import_module('boundary.acoustic_pml_numba')
    old_threads = numba.get_num_threads()
    try:
        numba.set_num_threads(min(2, numba.config.NUMBA_NUM_THREADS))
        monkeypatch.setattr(dispatch, 'PARALLEL_MIN_CELLS', 1 if parallel else 10**9)
        assert compiled._parallel_field(ny*nx) == parallel
        actual = compiled.acoustic_rhs(*values)
    finally:
        numba.set_num_threads(old_threads)
    assert actual.shape == v.shape and actual.dtype == np.float64
    np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=3e-15)
    for initial, unchanged in zip(saved, values):
        np.testing.assert_array_equal(unchanged, initial)


def test_compiled_rhs_accepts_uniform_background_scalars():
    gas = solver(film_cells=0)
    pml = AcousticPML(gas.mapping, gas.lattice, gas.ny, gas.nx, cells=5)
    rng = np.random.default_rng(388)
    v = rng.normal(0., .01, (7, gas.ny, gas.nx))
    assert importlib.util.find_spec('boundary.acoustic_pml_numba') is not None, \
        'compiled acoustic RHS is missing'
    compiled = importlib.import_module('boundary.acoustic_pml_numba')
    actual = compiled.acoustic_rhs(v, .95, 1.05*pml.c2,
        pml.sigma_x, pml.sigma_y, pml.decay_x, pml.decay_y)
    np.testing.assert_allclose(actual, pml._rhs(v, .95, 1.05*pml.c2),
                               rtol=3e-13, atol=3e-15)
